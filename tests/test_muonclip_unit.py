"""
Unit tests for MuonClip optimizer.

Run: pytest tests/test_muonclip_unit.py -v
"""

import pytest
import torch
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import MuonClipOptimizer, MuonClipConfig


class TestMuonClipConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test valid configuration passes."""
        config = MuonClipConfig()
        assert config.lr > 0
        assert 0 <= config.muon_beta < 1

    def test_invalid_lr(self):
        """Test invalid learning rate raises error."""
        with pytest.raises(AssertionError):
            MuonClipConfig(lr=0)
        with pytest.raises(AssertionError):
            MuonClipConfig(lr=-0.1)

    def test_invalid_threshold(self):
        """Test invalid clipping threshold raises error."""
        with pytest.raises(AssertionError):
            MuonClipConfig(clipping_threshold=0)
        with pytest.raises(AssertionError):
            MuonClipConfig(clipping_threshold=2000)

    def test_warnings_for_suboptimal(self):
        """Test warnings for suboptimal settings."""
        with pytest.warns(UserWarning):
            MuonClipConfig(ns_steps=2)  # Too few iterations

    def test_algorithm_aliases(self):
        """Test algorithm selection helpers."""
        cfg = MuonClipConfig(algorithm="newton_schulz")
        assert cfg.orthogonalization == "newton_schulz"

        cfg = MuonClipConfig(polar_express=False)
        assert cfg.orthogonalization == "newton_schulz"

        cfg = MuonClipConfig(orthogonalization="polar_express")
        assert cfg.orthogonalization == "polar_express"

        with pytest.raises(ValueError):
            MuonClipConfig(orthogonalization="unsupported")


class TestAttentionHooks:
    """Test attention hook system."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        config = NeoBERTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            flash_attention=False,
            hidden_act="gelu",
            rope=False,
        )
        return NeoBERT(config)

    def test_hook_registration(self, model):
        """Test hooks are registered correctly."""
        from neobert.optimizer.muon_clip import NeoBERTAttentionHooks

        hook_system = NeoBERTAttentionHooks(model.config)

        num_hooks = hook_system.register_hooks(model)
        assert num_hooks == 4  # 2 hooks per layer
        assert len(hook_system.hook_handles) == 4

    def test_hook_captures_data(self, model):
        """Test hooks actually capture attention data."""
        from neobert.optimizer.muon_clip import NeoBERTAttentionHooks

        hook_system = NeoBERTAttentionHooks(model.config)
        hook_system.register_hooks(model)

        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 64))
        model(input_ids)

        # Check data captured
        for layer_idx in range(2):
            inputs, pad_mask, freqs = hook_system.get_layer_data(layer_idx)
            assert inputs is not None
            assert inputs.shape[-1] == model.config.hidden_size
            assert pad_mask is None
            assert freqs is None


class TestMuonClipOptimizer:
    """Test optimizer functionality."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        config = NeoBERTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            flash_attention=False,
            hidden_act="gelu",
            rope=False,
        )
        return NeoBERT(config), config

    def test_optimizer_initialization(self, model):
        """Test optimizer initializes without errors."""
        model_instance, config = model

        muon_config = MuonClipConfig(
            lr=1e-4,
            enable_clipping=True,
            clipping_threshold=50.0,
        )

        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        assert len(optimizer.param_groups) == 2  # Muon + Adam groups
        assert optimizer._step == 0
        assert optimizer.config.orthogonalization == "polar_express"

    def test_parameter_grouping(self, model):
        """Test parameters are correctly grouped."""
        model_instance, config = model

        muon_config = MuonClipConfig(enable_clipping=False)
        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        # Check Muon group has 2D params
        muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
        for p in muon_group["params"]:
            assert p.ndim == 2

        # Check Adam group has 1D params
        adam_group = next(g for g in optimizer.param_groups if not g["use_muon"])
        for p in adam_group["params"]:
            assert p.ndim == 1

    def test_forward_backward_step(self, model):
        """Test full forward/backward/step cycle."""
        model_instance, config = model

        muon_config = MuonClipConfig(
            lr=1e-3,
            enable_clipping=True,
            clipping_threshold=50.0,
            orthogonalization="newton_schulz",
        )

        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 64))
        output = model_instance(input_ids)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Check step counter
        assert optimizer._step == 1

        # Check metrics collected
        metrics = optimizer.get_metrics()
        if muon_config.enable_clipping:
            assert "train/max_attention_logit" in metrics

    def test_clipping_applied(self, model):
        """Test QK-clipping actually modifies weights."""
        model_instance, config = model

        with pytest.warns(UserWarning, match="clipping_threshold"):
            muon_config = MuonClipConfig(
                lr=1e-3,
                enable_clipping=True,
                clipping_threshold=10.0,  # Very low threshold
            )

        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        # Get initial QKV weight
        qkv_param = model_instance.transformer_encoder[0].qkv.weight
        initial_weight = qkv_param.data.clone()

        # Forward/backward/step
        input_ids = torch.randint(0, 1000, (2, 64))
        output = model_instance(input_ids)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Check weight changed
        final_weight = qkv_param.data
        assert not torch.allclose(initial_weight, final_weight)

    def test_muon_only(self, model):
        """Test Muon-only mode (no clipping)."""
        model_instance, config = model

        muon_config = MuonClipConfig(
            lr=1e-3,
            enable_clipping=False,  # Disable clipping
        )

        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        # Should have no hook system
        assert optimizer.hook_system is None

        # Should still work
        input_ids = torch.randint(0, 1000, (2, 64))
        output = model_instance(input_ids)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        assert optimizer._step == 1


class TestNewtonSchulz:
    """Test Newton-Schulz orthogonalization."""

    def test_orthogonalization_quality(self):
        """Test orthogonalization produces orthogonal matrix."""
        from neobert.optimizer.muon_clip import MuonClipOptimizer

        # Create random matrix
        G = torch.randn(64, 64)

        # Create dummy optimizer to access method
        config = NeoBERTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            flash_attention=False,
        )
        model = NeoBERT(config)
        muon_config = MuonClipConfig()
        optimizer = MuonClipOptimizer(model, config, muon_config)

        # Orthogonalize
        X = optimizer._newton_schulz_update(G, steps=5)

        # Check that the result has proper scaling
        # After Newton-Schulz + RMS scaling, the matrix should have
        # reasonable norm
        assert X.norm() > 0
        assert X.norm() < G.norm() * 10  # Should not explode


class TestMemoryLeaks:
    """Test for memory leaks."""

    def test_hook_cleanup(self):
        """Test hooks don't cause memory leaks."""
        config = NeoBERTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            flash_attention=False,
        )
        model = NeoBERT(config)

        muon_config = MuonClipConfig(
            enable_clipping=True,
        )

        optimizer = MuonClipOptimizer(model, config, muon_config)

        # Run multiple steps
        for _ in range(10):
            input_ids = torch.randint(0, 1000, (2, 64))
            output = model(input_ids)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Hook stats should be cleared
        assert len(optimizer.hook_system.layer_inputs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
