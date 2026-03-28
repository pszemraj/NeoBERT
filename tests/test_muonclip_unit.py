"""
Unit tests for MuonClip optimizer.

Run: pytest tests/test_muonclip_unit.py -v
"""

import copy
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from neobert.model import NeoBERT, NeoBERTConfig, NeoBERTLMHead
from neobert.optimizer import MuonClipConfig, MuonClipOptimizer


def _legacy_polar_express_reference(
    grad: torch.Tensor, *, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Reproduce the pre-refactor Polar Express update with built-in scaling."""
    if grad.ndim != 2:
        return grad

    steps = max(1, int(steps))
    is_transpose = grad.size(0) > grad.size(1)
    working = grad.T if is_transpose else grad

    original_dtype = working.dtype
    spectral_norm = torch.linalg.norm(working)
    if spectral_norm == 0 or not torch.isfinite(spectral_norm):
        return torch.zeros_like(grad)

    working = working / (spectral_norm * 1.01 + eps)

    coeffs_base = [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.948690853482295, -2.908902115962949, 0.5518191394370137),
        (3.318419657370602, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.668903984574749, 0.4188073119525673),
        (1.891301407787398, -1.267995827194587, 0.3768040894852483),
        (1.875001480853448, -1.250001645399949, 0.3750001645474248),
        (1.875000000000000, -1.250000000000000, 0.375000000000000),
    ]
    dampening_factor = 1.01
    coeffs = [
        (
            a / dampening_factor,
            b / (dampening_factor**3),
            c / (dampening_factor**5),
        )
        for (a, b, c) in coeffs_base[:-1]
    ]
    coeffs.append(coeffs_base[-1])

    if steps <= len(coeffs):
        coeffs = coeffs[:steps]
    else:
        coeffs.extend([coeffs[-1]] * (steps - len(coeffs)))

    for a, b, c in coeffs:
        A = working @ working.T
        B = b * A + c * (A @ A)
        working = a * working + B @ working

    scale_factor = 0.4 * max(working.size(0), working.size(1)) ** 0.5
    working = scale_factor * working
    if working.dtype != original_dtype:
        working = working.to(original_dtype)
    return working.T if is_transpose else working


class TestMuonClipConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test valid configuration passes."""
        config = MuonClipConfig()
        assert config.lr > 0
        assert 0 <= config.muon_beta < 1
        assert config.norm_factor == "legacy_compat"
        assert config.param_policy == "transformer_only"

    def test_invalid_numeric_fields_raise(self):
        """Test invalid numeric config values fail fast."""
        cases = [
            {"lr": 0},
            {"lr": -0.1},
            {"clipping_threshold": 0},
            {"clipping_threshold": 2000},
            {"clipping_interval": 0},
            {"clipping_interval": -3},
            {"clipping_qk_chunk_size": 0},
            {"norm_factor": "unsupported"},
            {"param_policy": "unsupported"},
        ]
        for kwargs in cases:
            with pytest.raises(ValueError):
                MuonClipConfig(**kwargs)

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
            dropout=0.0,
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

        assert len(optimizer.param_groups) >= 2  # Muon + Adam fallback groups
        assert optimizer._step == 0
        assert optimizer.config.orthogonalization == "polar_express"

    def test_parameter_grouping(self, model):
        """Default grouping should keep embeddings/output weights on Adam."""
        model_instance, config = model

        muon_config = MuonClipConfig(enable_clipping=False)
        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
        muon_params = set(muon_group["params"])
        for p in muon_group["params"]:
            assert p.ndim == 2

        assert muon_group["param_policy"] == "transformer_only"
        assert model_instance.encoder.weight not in muon_params
        if hasattr(model_instance, "positional_embedding"):
            assert model_instance.positional_embedding.weight not in muon_params
        assert model_instance.transformer_encoder[0].qkv.weight in muon_params

        adam_groups = [g for g in optimizer.param_groups if not g["use_muon"]]
        adam_params = {param for group in adam_groups for param in group["params"]}
        assert model_instance.encoder.weight in adam_params
        if hasattr(model_instance, "positional_embedding"):
            assert model_instance.positional_embedding.weight in adam_params

    def test_grouping_all_2d_routes_embeddings_and_lm_head_to_muon(self):
        """all_2d should restore v0.1.3-style Muon scope."""
        config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=257,
            max_length=64,
            attn_backend="sdpa",
            hidden_act="gelu",
            rope=False,
            tie_word_embeddings=False,
        )
        model = NeoBERTLMHead(config)
        optimizer = MuonClipOptimizer(
            model,
            config,
            MuonClipConfig(enable_clipping=False, param_policy="all_2d"),
        )

        muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
        muon_params = set(muon_group["params"])
        adam_groups = [g for g in optimizer.param_groups if not g["use_muon"]]
        adam_params = {param for group in adam_groups for param in group["params"]}

        assert model.model.encoder.weight in muon_params
        assert model.decoder.weight in muon_params
        assert model.model.encoder.weight not in adam_params
        assert model.decoder.weight not in adam_params
        assert model.model.transformer_encoder[0].qkv.weight in muon_params

    def test_grouping_transformer_only_excludes_embeddings_and_lm_head(self):
        """transformer_only should keep embeddings/output projection in Adam."""
        config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=257,
            max_length=64,
            attn_backend="sdpa",
            hidden_act="gelu",
            rope=False,
            tie_word_embeddings=False,
        )
        model = NeoBERTLMHead(config)
        optimizer = MuonClipOptimizer(
            model,
            config,
            MuonClipConfig(enable_clipping=False, param_policy="transformer_only"),
        )

        muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
        muon_params = set(muon_group["params"])
        adam_groups = [g for g in optimizer.param_groups if not g["use_muon"]]
        adam_params = {param for group in adam_groups for param in group["params"]}

        assert muon_group["param_policy"] == "transformer_only"
        assert model.model.encoder.weight not in muon_params
        assert model.model.encoder.weight in adam_params
        assert model.decoder.weight not in muon_params
        assert model.decoder.weight in adam_params
        assert model.model.transformer_encoder[0].qkv.weight in muon_params

    def test_adam_fallback_splits_decay_and_no_decay(self):
        """MuonClip Adam fallback should mirror the repo's AdamW decay policy."""
        config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=257,
            max_length=64,
            attn_backend="sdpa",
            hidden_act="gelu",
            rope=False,
            tie_word_embeddings=False,
        )
        model = NeoBERTLMHead(config)
        optimizer = MuonClipOptimizer(
            model,
            config,
            MuonClipConfig(
                enable_clipping=False,
                adam_decay=0.1,
                param_policy="transformer_only",
            ),
        )

        adam_groups = [
            group for group in optimizer.param_groups if not group["use_muon"]
        ]
        decay_group = next(
            group
            for group in adam_groups
            if group["weight_decay"] == pytest.approx(0.1)
        )
        no_decay_group = next(
            group
            for group in adam_groups
            if group["weight_decay"] == pytest.approx(0.0)
        )
        decay_params = set(decay_group["params"])
        no_decay_params = set(no_decay_group["params"])

        assert model.decoder.weight in decay_params
        assert model.model.encoder.weight in no_decay_params

        bias_params = {
            param
            for name, param in model.named_parameters()
            if name.lower().endswith(".bias")
        }
        norm_params = {
            param for name, param in model.named_parameters() if "norm" in name.lower()
        }
        assert bias_params.issubset(no_decay_params)
        assert norm_params.issubset(no_decay_params)

    def test_norm_factor_modes(self):
        """Normalization modes should apply the expected scalar factors."""
        config = NeoBERTConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            vocab_size=64,
            max_length=16,
            attn_backend="sdpa",
            hidden_act="gelu",
            rope=False,
        )
        model = NeoBERT(config)
        update = torch.ones(6, 3)
        param_shape = torch.Size([6, 3])
        expected_scales = {
            "none": 1.0,
            "spectral": (6 / 3) ** 0.5,
            "match_rms_adamw": 0.2 * (6**0.5),
            "legacy_compat": 0.4 * (6**0.5),
        }

        for mode, expected_scale in expected_scales.items():
            optimizer = MuonClipOptimizer(
                model,
                config,
                MuonClipConfig(enable_clipping=False, norm_factor=mode),
            )
            normalized = optimizer._normalize_muon_update(update, param_shape)
            assert torch.allclose(normalized, update * expected_scale)

    def test_default_norm_factor_matches_pre_refactor_polar_step(self):
        """Default normalization should match pre-refactor local Polar Express math."""
        torch.manual_seed(0)
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
        )
        model = NeoBERT(config)
        muon_config = MuonClipConfig(
            lr=1e-3,
            muon_beta=0.95,
            muon_decay=0.0,
            adam_decay=0.0,
            ns_steps=5,
            enable_clipping=False,
            orthogonalization="polar_express",
        )
        assert muon_config.norm_factor == "legacy_compat"
        optimizer = MuonClipOptimizer(model, config, muon_config)

        target = model.transformer_encoder[0].qkv.weight
        grad = torch.randn_like(target)
        initial = target.detach().clone()

        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.zeros_like(param)
        target.grad = grad.clone()

        momentum = (1 - muon_config.muon_beta) * grad
        expected_update = _legacy_polar_express_reference(
            momentum, steps=muon_config.ns_steps
        )
        expected = initial - muon_config.lr * expected_update

        optimizer.step()
        assert torch.allclose(target, expected, atol=1e-6, rtol=1e-6)

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

    def test_distributed_checkpoint_api_restores_step_and_next_update(self, model):
        """Distributed checkpoint APIs should preserve Muon step semantics."""
        model_instance, config = model
        muon_config = MuonClipConfig(
            enable_clipping=False,
            param_policy="transformer_only",
        )
        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        for _ in range(3):
            input_ids = torch.randint(0, 1000, (2, 64))
            output = model_instance(input_ids)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model_clone = NeoBERT(config)
        model_clone.load_state_dict(model_instance.state_dict())
        optimizer_clone = MuonClipOptimizer(model_clone, config, muon_config)

        optim_state = get_optimizer_state_dict(model_instance, optimizer)
        assert "muonclip_step" not in optim_state

        set_optimizer_state_dict(model_clone, optimizer_clone, optim_state)
        assert optimizer_clone._step == optimizer._step

        input_ids = torch.randint(0, 1000, (2, 64))
        output = model_instance(input_ids)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        clone_output = model_clone(input_ids)
        clone_loss = clone_output.sum()
        clone_loss.backward()
        optimizer_clone.step()
        optimizer_clone.zero_grad()

        max_diff = 0.0
        max_name = ""
        for (name, param), (clone_name, clone_param) in zip(
            model_instance.named_parameters(),
            model_clone.named_parameters(),
            strict=True,
        ):
            assert clone_name == name
            diff = float((param - clone_param).abs().max().item())
            if diff > max_diff:
                max_diff = diff
                max_name = name

        assert max_diff <= 5e-5, (
            "Distributed checkpoint optimizer restore drifted on the next update: "
            f"param={max_name} max_diff={max_diff:.6e}"
        )

    def test_distributed_checkpoint_filesystem_roundtrip_restores_muon_state(
        self,
        model,
        tmp_path: Path,
    ):
        """Filesystem DCP round-trips should use the FQN optimizer schema."""
        model_instance, config = model
        muon_config = MuonClipConfig(
            enable_clipping=False,
            param_policy="transformer_only",
        )
        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        for _ in range(3):
            input_ids = torch.randint(0, 1000, (2, 64))
            output = model_instance(input_ids)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model_clone = NeoBERT(config)
        model_clone.load_state_dict(model_instance.state_dict())
        optimizer_clone = MuonClipOptimizer(model_clone, config, muon_config)

        checkpoint_dir = tmp_path / "optimizer_dcp"
        loaded_optim_state = {
            "optimizer": get_optimizer_state_dict(model_clone, optimizer_clone)
        }
        with pytest.warns(
            UserWarning,
            match="assuming the intent is to save in a single process",
        ):
            dist_cp.save(
                state_dict={
                    "optimizer": get_optimizer_state_dict(model_instance, optimizer)
                },
                storage_writer=dist_cp.FileSystemWriter(str(checkpoint_dir)),
            )
        with pytest.warns(
            UserWarning,
            match="assuming the intent is to load in a single process",
        ):
            dist_cp.load(
                loaded_optim_state,
                checkpoint_id=str(checkpoint_dir),
                storage_reader=dist_cp.FileSystemReader(str(checkpoint_dir)),
            )
        set_optimizer_state_dict(
            model_clone,
            optimizer_clone,
            loaded_optim_state["optimizer"],
        )
        assert optimizer_clone._step == optimizer._step

        input_ids = torch.randint(0, 1000, (2, 64))
        output = model_instance(input_ids)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        clone_output = model_clone(input_ids)
        clone_loss = clone_output.sum()
        clone_loss.backward()
        optimizer_clone.step()
        optimizer_clone.zero_grad()

        max_diff = 0.0
        max_name = ""
        for (name, param), (clone_name, clone_param) in zip(
            model_instance.named_parameters(),
            model_clone.named_parameters(),
            strict=True,
        ):
            assert clone_name == name
            diff = float((param - clone_param).abs().max().item())
            if diff > max_diff:
                max_diff = diff
                max_name = name

        assert max_diff < 1e-6, (
            f"Next Muon step diverged after DCP reload at {max_name}"
        )

    def test_load_state_dict_restores_missing_group_metadata(self, model):
        """Resumes should restore Muon-specific group metadata before loading."""
        model_instance, config = model
        muon_config = MuonClipConfig(enable_clipping=False)
        optimizer = MuonClipOptimizer(model_instance, config, muon_config)

        for _ in range(2):
            input_ids = torch.randint(0, 1000, (2, 64))
            output = model_instance(input_ids)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        stripped_state = copy.deepcopy(optimizer.state_dict())
        for group in stripped_state["param_groups"]:
            group.pop("use_muon", None)
            group.pop("param_info", None)
            group.pop("beta", None)

        model_ref = NeoBERT(config)
        model_ref.load_state_dict(model_instance.state_dict())
        optimizer_ref = MuonClipOptimizer(model_ref, config, muon_config)
        optimizer_ref.load_state_dict(optimizer.state_dict())

        model_clone = NeoBERT(config)
        model_clone.load_state_dict(model_instance.state_dict())
        optimizer_clone = MuonClipOptimizer(model_clone, config, muon_config)
        optimizer_clone.load_state_dict(stripped_state)

        assert all("use_muon" in group for group in optimizer_clone.param_groups)
        assert all("param_info" in group for group in optimizer_clone.param_groups)
        muon_group = next(
            group for group in optimizer_clone.param_groups if group["use_muon"]
        )
        assert "beta" in muon_group
        assert len(muon_group["param_info"]) == len(muon_group["params"])

        input_ids = torch.randint(0, 1000, (2, 64))
        ref_output = model_ref(input_ids)
        ref_loss = ref_output.sum()
        ref_loss.backward()
        optimizer_ref.step()
        optimizer_ref.zero_grad()

        clone_output = model_clone(input_ids)
        clone_loss = clone_output.sum()
        clone_loss.backward()
        optimizer_clone.step()
        optimizer_clone.zero_grad()

        max_diff = 0.0
        max_name = ""
        for (name, param), (clone_name, clone_param) in zip(
            model_ref.named_parameters(),
            model_clone.named_parameters(),
            strict=True,
        ):
            assert clone_name == name
            diff = float((param - clone_param).abs().max().item())
            if diff > max_diff:
                max_diff = diff
                max_name = name

        assert max_diff <= 5e-5, (
            "MuonClip stripped-group resume drifted on the next update: "
            f"param={max_name} max_diff={max_diff:.6e}"
        )

    def test_loaded_state_rejects_tensor_momentum_for_dtensor_param(
        self, model, monkeypatch: pytest.MonkeyPatch
    ):
        """Sharded Muon params must not accept local Tensor momentum buffers."""
        model_instance, config = model
        optimizer = MuonClipOptimizer(
            model_instance,
            config,
            MuonClipConfig(enable_clipping=False),
        )

        class _FakeMesh:
            ndim = 1
            mesh = torch.tensor([0, 1])

        class _FakeShard:
            def __init__(self, dim: int):
                self.dim = dim

        class _FakeDTensor:
            def __init__(self):
                self.device_mesh = _FakeMesh()
                self.placements = (_FakeShard(0),)

        fake_param = _FakeDTensor()
        optimizer.param_groups = [
            {
                "use_muon": True,
                "params": [fake_param],
                "param_info": [
                    {
                        "name": "transformer_encoder.0.qkv.weight",
                        "layer_idx": 0,
                        "is_qkv": True,
                        "proj_type": "qkv",
                    }
                ],
            }
        ]
        optimizer.state = {fake_param: {"momentum_buffer": torch.zeros(2, 2)}}
        monkeypatch.setattr(
            optimizer,
            "_is_dtensor",
            lambda tensor: isinstance(tensor, _FakeDTensor),
        )

        with pytest.raises(RuntimeError, match="local Tensor momentum buffer"):
            optimizer._validate_loaded_muon_state_topology()

    def test_loaded_state_rejects_mismatched_dtensor_topology(
        self, model, monkeypatch: pytest.MonkeyPatch
    ):
        """Sharded Muon state must match the target DTensor mesh and placement."""
        model_instance, config = model
        optimizer = MuonClipOptimizer(
            model_instance,
            config,
            MuonClipConfig(enable_clipping=False),
        )

        class _FakeMesh:
            def __init__(self, ranks: list[int]):
                self.ndim = 1
                self.mesh = torch.tensor(ranks)

        class _FakeShard:
            def __init__(self, dim: int):
                self.dim = dim

        class _FakeDTensor:
            def __init__(self, ranks: list[int], dim: int):
                self.device_mesh = _FakeMesh(ranks)
                self.placements = (_FakeShard(dim),)

        fake_param = _FakeDTensor([0, 1], 0)
        fake_buffer = _FakeDTensor([0, 1], 1)
        optimizer.param_groups = [
            {
                "use_muon": True,
                "params": [fake_param],
                "param_info": [
                    {
                        "name": "transformer_encoder.0.qkv.weight",
                        "layer_idx": 0,
                        "is_qkv": True,
                        "proj_type": "qkv",
                    }
                ],
            }
        ]
        optimizer.state = {fake_param: {"momentum_buffer": fake_buffer}}
        monkeypatch.setattr(
            optimizer,
            "_is_dtensor",
            lambda tensor: isinstance(tensor, _FakeDTensor),
        )

        with pytest.raises(RuntimeError, match="mesh/placement metadata"):
            optimizer._validate_loaded_muon_state_topology()

    def test_loaded_state_rejects_mismatched_momentum_shape(
        self, model, monkeypatch: pytest.MonkeyPatch
    ):
        """Muon state should fail fast when momentum-buffer shape mismatches param."""
        model_instance, config = model
        optimizer = MuonClipOptimizer(
            model_instance,
            config,
            MuonClipConfig(enable_clipping=False),
        )

        class _FakeMesh:
            def __init__(self, ranks: list[int]):
                self.ndim = 1
                self.mesh = torch.tensor(ranks)

        class _FakeShard:
            def __init__(self, dim: int):
                self.dim = dim

        class _FakeDTensor:
            def __init__(self, ranks: list[int], dim: int, shape: tuple[int, int]):
                self.device_mesh = _FakeMesh(ranks)
                self.placements = (_FakeShard(dim),)
                self.shape = torch.Size(shape)

        fake_param = _FakeDTensor([0, 1], 0, (8, 4))
        fake_buffer = _FakeDTensor([0, 1], 0, (7, 4))
        optimizer.param_groups = [
            {
                "use_muon": True,
                "params": [fake_param],
                "param_info": [
                    {
                        "name": "transformer_encoder.0.qkv.weight",
                        "layer_idx": 0,
                        "is_qkv": True,
                        "proj_type": "qkv",
                    }
                ],
            }
        ]
        optimizer.state = {fake_param: {"momentum_buffer": fake_buffer}}
        monkeypatch.setattr(
            optimizer,
            "_is_dtensor",
            lambda tensor: isinstance(tensor, _FakeDTensor),
        )

        with pytest.raises(RuntimeError, match="momentum state with shape"):
            optimizer._validate_loaded_muon_state_topology()

    def test_state_dict_param_info_avoids_live_parameter_objects(self, model):
        """Optimizer checkpoint metadata must not embed live Parameter objects."""
        model_instance, config = model
        optimizer = MuonClipOptimizer(
            model_instance,
            config,
            MuonClipConfig(enable_clipping=False),
        )

        state = optimizer.state_dict()
        for group in state["param_groups"]:
            assert "param_info" in group
            for info in group["param_info"]:
                assert "param" not in info
                assert set(info).issuperset(
                    {"name", "layer_idx", "is_qkv", "proj_type"}
                )

    def test_sharded_runtime_clipping_disable_preserves_config_intent(self, model):
        """Runtime disable should not rewrite user config intent."""
        model_instance, config = model
        optimizer = MuonClipOptimizer(
            model_instance,
            config,
            MuonClipConfig(enable_clipping=True, clipping_interval=1),
        )
        assert optimizer.config.enable_clipping
        assert optimizer.should_clip_update(0)

        optimizer._disable_clipping_for_sharded_runtime()

        assert optimizer.config.enable_clipping
        assert not optimizer.should_clip_update(0)
        assert not optimizer._runtime_clipping_enabled
        if optimizer.hook_system is not None:
            assert not optimizer.hook_system.enabled

    def test_dtensor_update_rejects_multidim_device_mesh(self, model, monkeypatch):
        """DTensor Muon path should fail fast on unsupported multi-axis meshes."""
        import neobert.optimizer.muon_clip as muon_clip_module

        if muon_clip_module.Shard is None:
            pytest.skip("DTensor Shard placement API unavailable in this torch build.")

        model_instance, config = model
        optimizer = MuonClipOptimizer(
            model_instance,
            config,
            MuonClipConfig(enable_clipping=False),
        )

        class _FakeMesh:
            ndim = 2

            def get_group(self):
                raise AssertionError(
                    "multi-axis mesh should be rejected before collectives"
                )

        class _FakeDTensor:
            placements = (muon_clip_module.Shard(0),)
            device_mesh = _FakeMesh()

        monkeypatch.setattr(muon_clip_module.dist, "is_available", lambda: True)
        monkeypatch.setattr(muon_clip_module.dist, "is_initialized", lambda: True)

        target = model_instance.transformer_encoder[0].qkv.weight
        with pytest.raises(RuntimeError, match="device_mesh.ndim=2"):
            optimizer._orthogonalize_dtensor_update(
                momentum_buffer=_FakeDTensor(),
                param=target,
                group_params=[target],
            )

    def test_dtensor_newton_schulz_owner_compute_uses_global_shape_without_gather(
        self, model, monkeypatch: pytest.MonkeyPatch
    ):
        """NS DTensor path should use global shape and avoid row-count collectives."""
        import neobert.optimizer.muon_clip as muon_clip_module

        model_instance, config = model
        optimizer = MuonClipOptimizer(
            model_instance,
            config,
            MuonClipConfig(
                enable_clipping=False,
                orthogonalization="newton_schulz",
                norm_factor="spectral",
            ),
        )
        full_matrix = torch.tensor(
            [[1.0, 2.0], [3.0, -4.0], [5.0, 6.0]],
            dtype=torch.float32,
        )
        local_shard = full_matrix[:2].clone()
        remote_shard = full_matrix[2:].clone()
        process_group = object()

        class _FakeShard:
            def __init__(self, dim: int):
                self.dim = dim

        class _FakeMesh:
            ndim = 1

            def get_group(self):
                return process_group

        class _FakeDTensor:
            def __init__(
                self,
                local: torch.Tensor,
                *,
                device_mesh: _FakeMesh,
                placements: tuple[_FakeShard, ...],
                shape: torch.Size,
                stride: tuple[int, ...],
            ):
                self._local = local
                self.device_mesh = device_mesh
                self.placements = placements
                self.shape = torch.Size(shape)
                self._stride = tuple(stride)

            def to_local(self) -> torch.Tensor:
                return self._local

            def stride(self) -> tuple[int, ...]:
                return self._stride

            @staticmethod
            def from_local(
                local: torch.Tensor,
                *,
                device_mesh: _FakeMesh,
                placements: tuple[_FakeShard, ...],
                run_check: bool = False,
                shape: torch.Size,
                stride: tuple[int, ...],
            ) -> "_FakeDTensor":
                assert not run_check
                assert tuple(shape) == tuple(full_matrix.shape)
                assert tuple(stride) == tuple(full_matrix.stride())
                return _FakeDTensor(
                    local,
                    device_mesh=device_mesh,
                    placements=placements,
                    shape=shape,
                    stride=stride,
                )

        class _FakeParam:
            shape = torch.Size([2, 2])

            @staticmethod
            def numel() -> int:
                return int(full_matrix.numel())

        fake_mesh = _FakeMesh()
        fake_placements = (_FakeShard(0),)
        momentum_buffer = _FakeDTensor(
            local_shard,
            device_mesh=fake_mesh,
            placements=fake_placements,
            shape=full_matrix.shape,
            stride=full_matrix.stride(),
        )
        fake_param = _FakeParam()
        remote_padded = torch.cat(
            (remote_shard, torch.zeros_like(local_shard[:1])),
            dim=0,
        )

        monkeypatch.setattr(muon_clip_module, "DTensor", _FakeDTensor)
        monkeypatch.setattr(muon_clip_module, "Shard", _FakeShard)
        monkeypatch.setattr(muon_clip_module.dist, "is_available", lambda: True)
        monkeypatch.setattr(muon_clip_module.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(
            muon_clip_module.dist,
            "get_process_group_ranks",
            lambda _group: [0, 1],
        )
        monkeypatch.setattr(
            muon_clip_module.dist,
            "get_world_size",
            lambda _group=None: 2,
        )
        monkeypatch.setattr(
            muon_clip_module.dist,
            "get_rank",
            lambda _group=None: 0,
        )

        def _unexpected_all_gather(*args, **kwargs):
            del args, kwargs
            raise AssertionError("row-count all_gather should not run in normal mode")

        def _fake_gather(
            tensor: torch.Tensor,
            gather_list: list[torch.Tensor] | None,
            *,
            group: object,
            group_dst: int,
        ) -> None:
            assert group is process_group
            assert group_dst == 0
            assert gather_list is not None
            gather_list[0].copy_(tensor)
            gather_list[1].copy_(remote_padded)

        def _fake_scatter(
            tensor: torch.Tensor,
            scatter_list: list[torch.Tensor] | None,
            *,
            group: object,
            group_src: int,
        ) -> None:
            assert group is process_group
            assert group_src == 0
            assert scatter_list is not None
            tensor.copy_(scatter_list[0])

        monkeypatch.setattr(muon_clip_module.dist, "all_gather", _unexpected_all_gather)
        monkeypatch.setattr(muon_clip_module.dist, "gather", _fake_gather)
        monkeypatch.setattr(muon_clip_module.dist, "scatter", _fake_scatter)

        update = optimizer._orthogonalize_dtensor_update(
            momentum_buffer=momentum_buffer,
            param=fake_param,
            group_params=[fake_param],
        )

        expected_full = optimizer._orthogonalize_update(full_matrix)
        expected_full = optimizer._normalize_muon_update(
            expected_full, full_matrix.shape
        )
        torch.testing.assert_close(update.to_local(), expected_full[:2])
        assert tuple(update.shape) == tuple(full_matrix.shape)
        assert tuple(update.stride()) == tuple(full_matrix.stride())

    def test_owner_rank_tie_breaks_on_param_name(
        self, model, monkeypatch: pytest.MonkeyPatch
    ):
        """Owner assignment should not depend on incidental parameter order."""
        model_instance, config = model
        optimizer = MuonClipOptimizer(
            model_instance,
            config,
            MuonClipConfig(enable_clipping=False),
        )

        class _FakeParam:
            def __init__(self, size: int):
                self._size = size

            def numel(self) -> int:
                return self._size

        z_param = _FakeParam(128)
        a_param = _FakeParam(128)
        monkeypatch.setattr(
            optimizer,
            "_process_group_cache_key",
            lambda _process_group: (0, 1),
        )

        owner_a = optimizer._resolve_owner_rank(
            param=a_param,
            group_params=[z_param, a_param],
            group_param_info=[{"name": "z.weight"}, {"name": "a.weight"}],
            world_size=2,
            process_group=object(),
        )
        owner_z = optimizer._resolve_owner_rank(
            param=z_param,
            group_params=[z_param, a_param],
            group_param_info=[{"name": "z.weight"}, {"name": "a.weight"}],
            world_size=2,
            process_group=object(),
        )

        assert owner_a == 0
        assert owner_z == 1

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

    def test_orthogonalization_rejects_fp16(self):
        """Ensure MuonClip orthogonalization rejects unsupported fp16 grads."""
        from neobert.optimizer.muon_clip import MuonClipOptimizer

        G = torch.randn(32, 32, dtype=torch.float16)
        config = NeoBERTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            attn_backend="sdpa",
        )
        model = NeoBERT(config)
        muon_config = MuonClipConfig()
        optimizer = MuonClipOptimizer(model, config, muon_config)

        with pytest.raises(RuntimeError, match="fp16/float16"):
            optimizer._newton_schulz_update(G, steps=5)


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
