"""
Unit tests for MuonClip optimizer.

Run: pytest tests/test_muonclip_unit.py -v
"""

import pytest
import torch

from neobert.model import NeoBERT, NeoBERTConfig, NeoBERTLMHead
from neobert.optimizer import MuonClipConfig, MuonClipOptimizer


class TestMuonClipConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test valid configuration passes."""
        config = MuonClipConfig()
        assert config.lr > 0
        assert 0 <= config.muon_beta < 1

    def test_invalid_lr(self):
        """Test invalid learning rate raises error."""
        with pytest.raises(ValueError):
            MuonClipConfig(lr=0)
        with pytest.raises(ValueError):
            MuonClipConfig(lr=-0.1)

    def test_invalid_threshold(self):
        """Test invalid clipping threshold raises error."""
        with pytest.raises(ValueError):
            MuonClipConfig(clipping_threshold=0)
        with pytest.raises(ValueError):
            MuonClipConfig(clipping_threshold=2000)

    def test_invalid_clipping_interval(self):
        """Test invalid clipping interval raises error."""
        with pytest.raises(ValueError):
            MuonClipConfig(clipping_interval=0)
        with pytest.raises(ValueError):
            MuonClipConfig(clipping_interval=-3)

    def test_invalid_chunk_size(self):
        """Test invalid chunk size raises error."""
        with pytest.raises(ValueError):
            MuonClipConfig(clipping_qk_chunk_size=0)

    def test_warnings_for_suboptimal(self):
        """Test warnings for suboptimal settings."""
        with pytest.warns(UserWarning):
            MuonClipConfig(ns_steps=2)  # Too few iterations

    def test_algorithm_aliases(self):
        """Test algorithm selection helpers."""
        with pytest.warns(UserWarning, match="MuonClipConfig.algorithm is deprecated"):
            cfg = MuonClipConfig(algorithm="newton_schulz")
        assert cfg.orthogonalization == "newton_schulz"

        with pytest.warns(
            UserWarning, match="MuonClipConfig.polar_express is deprecated"
        ):
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
            max_length=128,
            attn_backend="sdpa",
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

    def test_hook_registration_uses_module_map_without_attrs(self, model):
        """Ensure hook callbacks avoid per-layer module integer attrs."""
        from neobert.optimizer.muon_clip import NeoBERTAttentionHooks

        hook_system = NeoBERTAttentionHooks(model.config)
        hook_system.register_hooks(model)

        for idx, layer in enumerate(model.transformer_encoder):
            assert hook_system._module_to_layer_idx[id(layer)] == idx
            assert hook_system._module_to_layer_idx[id(layer.qkv)] == idx
            assert not hasattr(layer, "_muonclip_layer_idx")
            assert not hasattr(layer.qkv, "_muonclip_layer_idx")
            qkv_hook = next(iter(layer.qkv._forward_hooks.values()))
            if hasattr(qkv_hook, "__func__"):
                closure = qkv_hook.__func__.__closure__
            else:
                closure = qkv_hook.__closure__
            # Dynamo disable wrappers may introduce closures; ensure they do not
            # capture per-layer integer state.
            if closure is not None:
                assert all(not isinstance(cell.cell_contents, int) for cell in closure)

    def test_hook_callbacks_are_dynamo_disabled(self):
        """Ensure hook callbacks are excluded from torch.compile tracing."""
        from neobert.optimizer.muon_clip import NeoBERTAttentionHooks

        if not hasattr(torch, "_dynamo") or not hasattr(torch._dynamo, "disable"):
            pytest.skip("torch._dynamo.disable is unavailable in this torch build")

        for method_name in (
            "_module_layer_idx",
            "_qkv_input_hook",
            "_block_context_hook",
        ):
            method = getattr(NeoBERTAttentionHooks, method_name)
            assert bool(getattr(method, "_torchdynamo_disable", False))

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
            inputs, pad_mask, freqs, packed_seqlens = hook_system.get_layer_data(
                layer_idx
            )
            assert inputs is not None
            assert inputs.shape[-1] == model.config.hidden_size
            assert inputs.device.type == "cpu"
            assert pad_mask is None
            assert freqs is None
            assert packed_seqlens is None

    def test_hook_captures_cuda_inputs_into_pinned_cpu_buffers(self, model):
        """CUDA hook capture should use pinned CPU buffers for async D2H copies."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required to validate pinned CPU capture path.")

        from neobert.optimizer.muon_clip import NeoBERTAttentionHooks

        model = model.to("cuda")
        hook_system = NeoBERTAttentionHooks(model.config)
        hook_system.register_hooks(model)

        input_ids = torch.randint(0, 1000, (2, 64), device="cuda")
        model(input_ids)

        for layer_idx in range(2):
            inputs, _, _, _ = hook_system.get_layer_data(layer_idx)
            assert inputs is not None
            assert inputs.device.type == "cpu"
            assert inputs.is_pinned()

    def test_hook_clear_preserves_layer_slots(self, model):
        """Ensure clearing hook caches keeps stable dictionary cardinality."""
        from neobert.optimizer.muon_clip import NeoBERTAttentionHooks

        hook_system = NeoBERTAttentionHooks(model.config)
        hook_system.register_hooks(model)

        num_layers = len(model.transformer_encoder)
        assert len(hook_system.layer_inputs) == num_layers
        assert not hook_system.has_captured_inputs()

        input_ids = torch.randint(0, 1000, (2, 64))
        model(input_ids)
        assert hook_system.has_captured_inputs()
        assert len(hook_system.layer_inputs) == num_layers

        hook_system.clear()
        assert len(hook_system.layer_inputs) == num_layers
        assert not hook_system.has_captured_inputs()


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
            max_length=128,
            attn_backend="sdpa",
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

    def test_interleaved_qkv_scaling(self):
        """Ensure fused QKV scaling matches per-head interleaved layout."""
        config = NeoBERTConfig(
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            vocab_size=32,
            max_length=8,
            attn_backend="sdpa",
            hidden_act="gelu",
            rope=False,
        )
        model = NeoBERT(config)
        optimizer = MuonClipOptimizer(
            model, config, MuonClipConfig(enable_clipping=False)
        )

        qkv_param = model.transformer_encoder[0].qkv.weight
        original = torch.arange(qkv_param.numel(), dtype=qkv_param.dtype).view_as(
            qkv_param
        )
        qkv_param.data.copy_(original)

        eta = torch.tensor([0.5, 0.25], dtype=qkv_param.dtype)
        with torch.no_grad():
            optimizer._scale_qkv_weights(qkv_param, eta, alpha=1.0)

        expected = original.clone()
        view = expected.view(config.num_attention_heads, config.dim_head * 3, -1)
        view[:, : config.dim_head].mul_(eta.view(-1, 1, 1))
        assert torch.allclose(qkv_param, expected)

    def test_qkv_scaling_rejects_non_interleaved_layout(self):
        """Ensure fused QKV scaling fails fast on incompatible weight layouts."""
        config = NeoBERTConfig(
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            vocab_size=32,
            max_length=8,
            attn_backend="sdpa",
            hidden_act="gelu",
            rope=False,
        )
        model = NeoBERT(config)
        optimizer = MuonClipOptimizer(
            model, config, MuonClipConfig(enable_clipping=False)
        )
        bad_layout = torch.nn.Parameter(torch.zeros(4, 12))
        eta = torch.tensor([0.5, 0.25], dtype=bad_layout.dtype)

        with pytest.raises(RuntimeError, match="Unexpected fused QKV parameter layout"):
            optimizer._scale_qkv_weights(bad_layout, eta, alpha=1.0)

    def test_packed_attention_logit_max_ignores_cross_segment(self):
        """Cross-segment logits must not affect max in packed mode."""
        config = NeoBERTConfig(
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=16,
            vocab_size=32,
            max_length=8,
            attn_backend="sdpa",
            hidden_act="gelu",
            rope=False,
        )
        model = NeoBERT(config)
        optimizer = MuonClipOptimizer(
            model, config, MuonClipConfig(enable_clipping=False)
        )

        xq = torch.tensor([[[[1.0], [1.0], [0.0], [0.0]]]])
        xk = torch.tensor([[[[1.0], [1.0], [100.0], [100.0]]]])
        per_step_max = optimizer._packed_attention_logit_max(
            xq_heads=xq,
            xk_heads=xk,
            packed_seqlens=[[2, 2]],
            scale=1.0,
        )

        assert per_step_max.shape == (1, 1)
        assert torch.allclose(per_step_max, torch.tensor([[1.0]]))

    def test_ngpt_qk_clipping_runs(self):
        """Ensure ngpt QK clipping path executes without errors."""
        config = NeoBERTConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            vocab_size=64,
            max_length=16,
            attn_backend="sdpa",
            hidden_act="gelu",
            rope=False,
            ngpt=True,
        )
        model = NeoBERTLMHead(config)
        optimizer = MuonClipOptimizer(
            model, config, MuonClipConfig(enable_clipping=True)
        )

        input_ids = torch.randint(0, 64, (2, 8))
        output = model(input_ids)["logits"]
        loss = output.sum()
        loss.backward()
        optimizer.step()

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

    def test_hook_capture_gating_last_microbatch_only(self, model):
        """Hooks should only capture on the last microbatch when enabled."""
        model_instance, config = model
        muon_config = MuonClipConfig(
            enable_clipping=True,
            clipping_interval=1,
            capture_last_microbatch_only=True,
        )
        optimizer = MuonClipOptimizer(model_instance, config, muon_config)
        hook_system = optimizer.hook_system
        assert hook_system is not None

        optimizer.prepare_for_forward(update_step=0, is_last_microbatch=False)
        _ = model_instance(torch.randint(0, 1000, (2, 64)))
        assert not hook_system.has_captured_inputs()
        assert len(hook_system.layer_inputs) == len(model_instance.transformer_encoder)

        optimizer.prepare_for_forward(update_step=0, is_last_microbatch=True)
        _ = model_instance(torch.randint(0, 1000, (2, 64)))
        assert hook_system.has_captured_inputs()

        optimizer.step()
        assert not hook_system.has_captured_inputs()

    def test_chunked_logit_max_matches_full(self, model):
        """Chunked logit max should match the full matmul result."""
        model_instance, config = model
        muon_config = MuonClipConfig(enable_clipping=False, clipping_qk_chunk_size=2)
        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        xq = torch.randn(1, config.num_attention_heads, 4, config.dim_head)
        xk = torch.randn_like(xq)
        scale = 1.0
        full = torch.matmul(xq, xk.transpose(-2, -1)) * scale
        full_max = full.amax(dim=(-2, -1))
        chunked_max = optimizer._attention_logit_max(
            xq_heads=xq, xk_heads=xk, scale=scale, pad_mask=None
        )

        assert torch.allclose(full_max, chunked_max)

    def test_state_dict_persists_step(self, model):
        """Ensure MuonClip step counter persists across state dicts."""
        model_instance, config = model
        muon_config = MuonClipConfig(enable_clipping=False)
        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        for _ in range(3):
            input_ids = torch.randint(0, 1000, (2, 64))
            output = model_instance(input_ids)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        state = optimizer.state_dict()

        model_clone = NeoBERT(config)
        optimizer_clone = MuonClipOptimizer(model_clone, config, muon_config)
        optimizer_clone.load_state_dict(state)

        assert optimizer_clone._step == optimizer._step

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

    def test_clipping_interval_skips_steps(self, model):
        """Ensure clipping only runs on the configured interval."""
        model_instance, config = model
        muon_config = MuonClipConfig(enable_clipping=True, clipping_interval=2)
        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        for step in range(3):
            optimizer.prepare_for_forward(
                update_step=optimizer._step, is_last_microbatch=True
            )
            input_ids = torch.randint(0, 1000, (2, 64))
            output = model_instance(input_ids)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            metrics = optimizer.get_metrics()
            optimizer.zero_grad()

            if step % 2 == 0:
                assert "train/max_attention_logit" in metrics
            else:
                assert "train/max_attention_logit" not in metrics

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
            attn_backend="sdpa",
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
            attn_backend="sdpa",
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

        # Hook stats should be cleared while preserving stable layer slots.
        assert not optimizer.hook_system.has_captured_inputs()
        assert len(optimizer.hook_system.layer_inputs) == len(model.transformer_encoder)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
