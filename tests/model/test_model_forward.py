#!/usr/bin/env python3
"""Test NeoBERT model forward passes and functionality."""

import unittest
from unittest.mock import PropertyMock, patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neobert.model import (
    NeoBERT,
    NeoBERTConfig,
    NeoBERTForSequenceClassification,
    NeoBERTHFForSequenceClassification,
    NeoBERTLMHead,
    NormNeoBERT,
)
from neobert.kernels.attention import (
    prepare_packed_flash_metadata as _prepare_packed_flash_metadata_real,
)


class TestModelForward(unittest.TestCase):
    """Test NeoBERT model forward passes."""

    def _make_tokenizer(self) -> PreTrainedTokenizerFast:
        """Build a minimal tokenizer for tests."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
        )

    def setUp(self):
        """Set up test fixtures."""
        # Create tiny config for testing
        self.tiny_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            dropout=0.1,
            vocab_size=1000,
            max_length=128,
            attn_backend="sdpa",  # Use SDPA attention for CPU testing
            ngpt=False,
            hidden_act="gelu",  # Use GELU instead of SwiGLU for CPU testing
        )

        # Sample input data
        self.batch_size = 2
        self.seq_length = 10
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        self.attention_mask = torch.ones(self.batch_size, self.seq_length)

        # Create additive mask for NeoBERT (expects float, not bool)
        self.pad_mask = torch.where(self.attention_mask == 0, float("-inf"), float(0.0))

    def test_neobert_forward(self):
        """Test basic NeoBERT forward pass."""
        model = NeoBERT(self.tiny_config)
        model.eval()

        with torch.no_grad():
            outputs = model(self.input_ids, self.pad_mask)

        # Check output shape
        expected_shape = (
            self.batch_size,
            self.seq_length,
            self.tiny_config.hidden_size,
        )
        self.assertEqual(outputs.shape, expected_shape)
        # Check that outputs are not NaN or inf
        self.assertFalse(torch.isnan(outputs).any())
        self.assertFalse(torch.isinf(outputs).any())

    def test_config_canonicalizes_attn_backend_alias(self):
        """Ensure attention backend aliases are canonicalized."""
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=128,
            max_length=16,
            attn_backend="flash",
            hidden_act="gelu",
        )
        self.assertEqual(config.attn_backend, "flash_attn_varlen")

    def test_config_rejects_invalid_backends(self):
        """Ensure invalid backend values fail fast for attention and kernel backends."""
        with self.assertRaisesRegex(ValueError, "Unknown attn_backend"):
            NeoBERTConfig(attn_backend="bad_backend")
        with self.assertRaisesRegex(ValueError, "Unknown kernel_backend"):
            NeoBERTConfig(kernel_backend="bad_backend")

    def test_neobert_accepts_tensor_packed_seqlens(self):
        """Ensure tensor packed_seqlens metadata works in model forward."""
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            dropout=0.0,
            vocab_size=256,
            max_length=32,
            attn_backend="sdpa",
            ngpt=False,
            hidden_act="gelu",
        )
        model = NeoBERT(config)
        model.eval()
        x = torch.randint(0, 256, (2, 8))
        packed = torch.tensor([[8, 0], [6, 0]], dtype=torch.int32)
        with torch.no_grad():
            out = model(x, pad_mask=None, packed_seqlens=packed)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_flash_metadata_is_reused_across_layers(self):
        """Ensure flash packed metadata is prepared once and reused in all layers."""
        import neobert.model.model as model_module

        calls: dict[str, list[int] | int] = {"prepare": 0, "meta_ids": []}

        def _count_prepare(*args, **kwargs):
            calls["prepare"] = int(calls["prepare"]) + 1
            return _prepare_packed_flash_metadata_real(*args, **kwargs)

        def _fake_attention_forward(
            xq: torch.Tensor,
            xk: torch.Tensor,
            xv: torch.Tensor,
            pad_mask: torch.Tensor | None,
            packed_seqlens: torch.Tensor | list[list[int]] | None,
            dropout_p: float,
            scale: float | None,
            attn_backend: str,
            packed_flash_metadata=None,
        ) -> torch.Tensor:
            assert packed_flash_metadata is not None
            meta_ids = calls["meta_ids"]
            assert isinstance(meta_ids, list)
            meta_ids.append(id(packed_flash_metadata))
            return torch.zeros_like(xq)

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            dropout=0.0,
            vocab_size=256,
            max_length=32,
            attn_backend="flash_attn_varlen",
            ngpt=False,
            hidden_act="gelu",
        )
        model = NeoBERT(config)
        model.eval()
        x = torch.randint(0, 256, (2, 8))
        packed = torch.tensor([[8, 0], [6, 0]], dtype=torch.int32)

        with (
            patch.object(
                model_module,
                "prepare_packed_flash_metadata",
                side_effect=_count_prepare,
            ),
            patch.object(
                model_module, "attention_forward", side_effect=_fake_attention_forward
            ),
        ):
            with torch.no_grad():
                out = model(x, pad_mask=None, packed_seqlens=packed)

        self.assertEqual(out.shape, (2, 8, 32))
        self.assertEqual(calls["prepare"], 1)
        meta_ids = calls["meta_ids"]
        assert isinstance(meta_ids, list)
        self.assertEqual(len(meta_ids), config.num_hidden_layers)
        self.assertEqual(len(set(meta_ids)), 1)

    def test_gradient_checkpointing_matches_baseline(self):
        """Ensure checkpointed gradients match non-checkpointed forward."""
        torch.manual_seed(0)
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            dropout=0.0,
            vocab_size=256,
            max_length=32,
            attn_backend="sdpa",
            ngpt=False,
            hidden_act="gelu",
        )

        model_ref = NeoBERT(config)
        model_ckpt = NeoBERT(config)
        model_ckpt.load_state_dict(model_ref.state_dict())

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        attention_mask = torch.ones_like(input_ids)
        pad_mask = torch.where(attention_mask == 0, float("-inf"), float(0.0))

        model_ref.train()
        model_ckpt.train()
        model_ckpt.gradient_checkpointing_enable()

        loss_ref = model_ref(input_ids, pad_mask).sum()
        loss_ref.backward()
        loss_ckpt = model_ckpt(input_ids, pad_mask).sum()
        loss_ckpt.backward()

        for name, param_ref in model_ref.named_parameters():
            param_ckpt = dict(model_ckpt.named_parameters())[name]
            if param_ref.grad is None:
                continue
            self.assertTrue(
                torch.allclose(param_ref.grad, param_ckpt.grad, atol=1e-6, rtol=1e-5)
            )

    def test_training_vs_hf_encoder_parity(self):
        """Ensure training and HF encoder paths match for shared weights."""
        from neobert.huggingface.modeling_neobert import NeoBERT as HFNeoBERT

        torch.manual_seed(0)
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            dropout=0.0,
            vocab_size=100,
            max_length=16,
            attn_backend="sdpa",
            hidden_act="swiglu",
        )

        train_model = NeoBERT(config)
        hf_model = HFNeoBERT(config)
        hf_model.load_state_dict(train_model.state_dict())

        train_model.eval()
        hf_model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]])
        with torch.no_grad():
            # Mask normalization is covered by separate tests; compare raw math here.
            train_out = train_model(input_ids, None)
            hf_out = hf_model(
                input_ids=input_ids, attention_mask=None
            ).last_hidden_state

        self.assertTrue(torch.allclose(train_out, hf_out, atol=1e-6))

    def test_norm_neobert_forward(self):
        """Test NormNeoBERT (nGPT-style) forward pass."""
        ngpt_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            dropout=0.1,
            vocab_size=1000,
            max_length=128,
            attn_backend="sdpa",
            ngpt=True,  # Enable nGPT mode
        )

        model = NormNeoBERT(ngpt_config)
        model.eval()

        with torch.no_grad():
            outputs = model(self.input_ids, self.pad_mask)

        expected_shape = (self.batch_size, self.seq_length, ngpt_config.hidden_size)
        self.assertEqual(outputs.shape, expected_shape)
        self.assertFalse(torch.isnan(outputs).any())
        self.assertFalse(torch.isinf(outputs).any())

    def test_neobert_lm_head(self):
        """Test NeoBERT with language modeling head."""
        model = NeoBERTLMHead(self.tiny_config)
        model.eval()

        with torch.no_grad():
            outputs = model(self.input_ids, self.pad_mask)
            hidden_only_outputs = model(
                self.input_ids, self.pad_mask, return_logits=False
            )

        # Check that we get both hidden states and logits
        self.assertIn("hidden_representation", outputs)
        self.assertIn("logits", outputs)
        self.assertIn("hidden_representation", hidden_only_outputs)
        self.assertNotIn("logits", hidden_only_outputs)

        hidden_shape = (self.batch_size, self.seq_length, self.tiny_config.hidden_size)
        logits_shape = (self.batch_size, self.seq_length, self.tiny_config.vocab_size)

        self.assertEqual(outputs["hidden_representation"].shape, hidden_shape)
        self.assertEqual(outputs["logits"].shape, logits_shape)
        self.assertEqual(
            hidden_only_outputs["hidden_representation"].shape, hidden_shape
        )
        self.assertIsNone(model.decoder.bias)

    def test_neobert_sequence_classification(self):
        """Test NeoBERT for sequence classification."""
        num_labels = 3
        model = NeoBERTForSequenceClassification(
            self.tiny_config, num_labels=num_labels
        )
        model.eval()

        with torch.no_grad():
            outputs = model(self.input_ids, self.pad_mask)

        self.assertIn("hidden_representation", outputs)
        self.assertIn("logits", outputs)

        # Classification logits should be [batch_size, num_labels]
        expected_logits_shape = (self.batch_size, num_labels)
        self.assertEqual(outputs["logits"].shape, expected_logits_shape)

    def test_sequence_classification_accepts_additive_mask(self):
        """Ensure additive attention masks are preserved for classification."""
        import types

        num_labels = 2
        model = NeoBERTForSequenceClassification(
            self.tiny_config, num_labels=num_labels
        )
        model.eval()

        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        additive_mask = torch.where(attention_mask == 1, float(0.0), float("-inf"))

        captured = {}

        def _fake_forward(self, src, pad_mask=None, packed_seqlens=None):
            captured["mask"] = pad_mask
            return torch.zeros(
                (src.shape[0], src.shape[1], self.config.hidden_size),
                device=src.device,
            )

        model.model.forward = types.MethodType(_fake_forward, model.model)

        with torch.no_grad():
            outputs = model(self.input_ids, additive_mask)

        self.assertIn("logits", outputs)
        self.assertTrue(torch.equal(captured["mask"], additive_mask.to(torch.float32)))

    def test_neobert_hf_sequence_classification(self):
        """Test HuggingFace-compatible sequence classification."""
        hf_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            dropout=0.1,
            vocab_size=1000,
            max_length=128,
            attn_backend="sdpa",
            ngpt=False,
            num_labels=2,
            hidden_act="gelu",
        )

        model = NeoBERTHFForSequenceClassification(hf_config)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                return_dict=True,
            )

        # Should return SequenceClassifierOutput
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertTrue(hasattr(outputs, "hidden_states"))

        expected_logits_shape = (self.batch_size, 2)  # num_labels=2
        self.assertEqual(outputs.logits.shape, expected_logits_shape)

    def test_hf_attention_mask_semantics_and_mode_equivalence(self):
        """Ensure HF mask normalization and SDPA/eager paths stay equivalent."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=16,
            flash_attention=False,
        )
        model = NeoBERT(config)
        model.eval()

        padded_input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        bool_mask = attention_mask == 1
        additive_mask = torch.where(attention_mask == 1, float(0.0), float("-inf"))

        with torch.no_grad():
            outputs_int = model(
                input_ids=padded_input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            outputs_bool = model(
                input_ids=padded_input_ids,
                attention_mask=bool_mask,
                output_attentions=True,
            )
            outputs_add = model(
                input_ids=padded_input_ids,
                attention_mask=additive_mask,
                output_attentions=True,
            )
            outputs_sdpa = model(
                input_ids=padded_input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
            )
        attn_weights = outputs_int.attentions[0]
        pad_positions = (
            (attention_mask == 0).unsqueeze(1).unsqueeze(2).expand_as(attn_weights)
        )
        self.assertLess(attn_weights.masked_select(pad_positions).max().item(), 1e-6)

        for other in (outputs_bool, outputs_add):
            self.assertTrue(
                torch.allclose(
                    outputs_int.last_hidden_state,
                    other.last_hidden_state,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
        self.assertTrue(
            torch.allclose(
                outputs_int.last_hidden_state,
                outputs_sdpa.last_hidden_state,
                atol=1e-6,
                rtol=1e-5,
            )
        )

        no_pad_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        all_ones = torch.ones_like(no_pad_ids)
        zero_additive = torch.zeros_like(no_pad_ids, dtype=torch.float32)
        for output_attentions in (False, True):
            with torch.no_grad():
                outputs_none = model(
                    input_ids=no_pad_ids,
                    attention_mask=None,
                    output_attentions=output_attentions,
                )
                outputs_ones = model(
                    input_ids=no_pad_ids,
                    attention_mask=all_ones,
                    output_attentions=output_attentions,
                )
                outputs_zero = model(
                    input_ids=no_pad_ids,
                    attention_mask=zero_additive,
                    output_attentions=output_attentions,
                )
            self.assertTrue(
                torch.allclose(
                    outputs_none.last_hidden_state,
                    outputs_ones.last_hidden_state,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
            self.assertTrue(
                torch.allclose(
                    outputs_none.last_hidden_state,
                    outputs_zero.last_hidden_state,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )

        all_false = torch.zeros_like(no_pad_ids, dtype=torch.bool)
        normalized = model._normalize_attention_mask(all_false)
        self.assertFalse(normalized.any().item())
        keep_all = torch.ones_like(no_pad_ids, dtype=torch.bool)
        normalized_keep_all = model._normalize_attention_mask(keep_all)
        self.assertTrue(normalized_keep_all.all().item())
        mixed_mask = torch.tensor(
            [[True, False, True, False], [False, True, False, True]]
        )
        normalized_mixed = model._normalize_attention_mask(mixed_mask)
        self.assertTrue(torch.equal(normalized_mixed, mixed_mask))

    def test_hf_eager_all_masked_rows_are_finite_and_match_sdpa(self):
        """Ensure fully-masked rows do not produce NaNs in eager attention."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=16,
            flash_attention=False,
            dropout=0.0,
        )
        model = NeoBERT(config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        all_masked = torch.zeros_like(input_ids)

        with torch.no_grad():
            outputs_sdpa = model(
                input_ids=input_ids,
                attention_mask=all_masked,
                output_attentions=False,
            )
            outputs_eager = model(
                input_ids=input_ids,
                attention_mask=all_masked,
                output_attentions=True,
            )

        self.assertFalse(torch.isnan(outputs_eager.last_hidden_state).any())
        self.assertFalse(torch.isnan(outputs_eager.attentions[0]).any())
        self.assertTrue(
            torch.equal(
                outputs_eager.attentions[0],
                torch.zeros_like(outputs_eager.attentions[0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs_sdpa.last_hidden_state,
                outputs_eager.last_hidden_state,
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_hf_position_ids_can_exceed_sequence_length(self):
        """Ensure RoPE cache grows when callers pass large explicit position IDs."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=8,
            flash_attention=False,
        )
        model = NeoBERT(config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4]])
        position_ids = torch.tensor([[8, 9, 10, 11]])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, position_ids=position_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (1, 4, 32))
        self.assertGreaterEqual(
            model.freqs_cis.shape[0],
            int(position_ids.max().item()) + 1,
        )

    def test_hf_hidden_states_follow_hf_convention(self):
        """Ensure hidden_states includes embeddings and ends at final output."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=16,
            flash_attention=False,
            dropout=0.0,
        )
        model = NeoBERT(config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

        self.assertIsInstance(outputs.hidden_states, tuple)
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
        self.assertTrue(
            torch.allclose(
                outputs.hidden_states[-1],
                outputs.last_hidden_state,
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_hf_sdpa_compat_omits_scale_kw_when_unavailable(self):
        """Ensure SDPA compat path never passes unsupported ``scale`` kwarg."""
        import neobert.huggingface.modeling_neobert as hf_mod

        captured_queries: list[torch.Tensor] = []

        def _fake_sdpa(
            *,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
        ) -> torch.Tensor:
            del key, value, attn_mask, dropout_p, is_causal
            captured_queries.append(query.detach().clone())
            return torch.zeros_like(query)

        query = torch.randn(1, 2, 3, 8)
        key = torch.randn(1, 2, 3, 8)
        value = torch.randn(1, 2, 3, 8)

        with (
            patch.object(hf_mod, "_SDPA_SUPPORTS_SCALE", False),
            patch.object(hf_mod, "scaled_dot_product_attention", new=_fake_sdpa),
        ):
            out_default = hf_mod.scaled_dot_product_attention_compat(
                query=query, key=key, value=value, scale=None
            )
            out_scaled = hf_mod.scaled_dot_product_attention_compat(
                query=query, key=key, value=value, scale=0.5
            )

        self.assertEqual(out_default.shape, query.shape)
        self.assertEqual(out_scaled.shape, query.shape)
        self.assertEqual(len(captured_queries), 2)
        self.assertFalse(torch.allclose(captured_queries[0], captured_queries[1]))

    def test_hf_sequence_classification_respects_return_dict_default(self):
        """Ensure return_dict=None follows config.use_return_dict in HF classifier."""
        from neobert.huggingface.modeling_neobert import (
            NeoBERTConfig,
            NeoBERTForSequenceClassification,
        )

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=16,
            flash_attention=False,
            num_labels=2,
        )
        self.assertTrue(config.use_return_dict)

        model = NeoBERTForSequenceClassification(config)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs_default = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs_tuple = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )

        self.assertFalse(isinstance(outputs_default, tuple))
        self.assertTrue(hasattr(outputs_default, "logits"))
        self.assertIsInstance(outputs_tuple, tuple)

    def test_hf_base_model_respects_return_dict_flag(self):
        """Ensure NeoBERT forward supports return_dict=False tuple outputs."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=16,
            flash_attention=False,
        )
        model = NeoBERT(config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=False,
            )

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, (1, 4, 32))
        self.assertIsInstance(outputs[1], tuple)
        self.assertIsInstance(outputs[2], tuple)

    def test_hf_lm_head_respects_return_dict_and_labels(self):
        """Ensure MLM head supports loss labels and tuple outputs."""
        from neobert.huggingface.modeling_neobert import NeoBERTConfig, NeoBERTLMHead

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=50,
            max_length=16,
            flash_attention=False,
        )
        model = NeoBERTLMHead(config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[1, 2, -100, 4]])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
            outputs_tuple = model(
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                return_dict=False,
            )

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (1, 4, 50))
        self.assertIsInstance(outputs_tuple, tuple)
        self.assertEqual(len(outputs_tuple), 3)  # loss, logits, hidden_states

    def test_hf_input_validation_errors(self):
        """Ensure HF model fails fast for invalid tensor and position inputs."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        base_config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=16,
            flash_attention=False,
        )
        model = NeoBERT(base_config)
        model.eval()

        with self.assertRaisesRegex(TypeError, "input_ids must be an integer tensor"):
            with torch.no_grad():
                model(input_ids=torch.randn(1, 4))

        input_ids = torch.tensor([[1, 2, 3, 4]])
        bad_mask = torch.tensor([[1.0, float("nan"), 1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "must not contain NaN"):
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=bad_mask)

        non_rope_config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=4,
            rope=False,
            flash_attention=False,
        )
        non_rope_model = NeoBERT(non_rope_config)
        non_rope_model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        position_ids = torch.tensor([[1, 2, 3, 5]])

        with self.assertRaisesRegex(
            ValueError, "position_ids exceed configured max_length"
        ):
            with torch.no_grad():
                non_rope_model(input_ids=input_ids, position_ids=position_ids)

    def test_hf_flash_attention_silently_ignored(self):
        """Ensure flash_attention=True is silently accepted for HF config compat."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        # flash_attention=True should be accepted without warning (HF config compat)
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=16,
            flash_attention=True,
        )
        self.assertTrue(config.flash_attention)

        # Model should still work (uses SDPA regardless)
        model = NeoBERT(config)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        self.assertEqual(outputs.last_hidden_state.shape, (1, 4, 32))

    def test_lm_head_ties_word_embeddings(self):
        """Ensure LM heads tie input/output embeddings by default."""
        from neobert.huggingface.modeling_neobert import (
            NeoBERTConfig as HFNeoBERTConfig,
            NeoBERTLMHead as HFNeoBERTLMHead,
        )
        from neobert.model import NeoBERTConfig, NeoBERTLMHead

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=16,
            max_length=8,
            attn_backend="sdpa",
            ngpt=False,
            hidden_act="gelu",
            tie_word_embeddings=True,
        )
        model = NeoBERTLMHead(config)
        self.assertEqual(
            model.decoder.weight.data_ptr(), model.model.encoder.weight.data_ptr()
        )

        hf_config = HFNeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=16,
            max_length=8,
            flash_attention=False,
            hidden_act="gelu",
            tie_word_embeddings=True,
        )
        hf_model = HFNeoBERTLMHead(hf_config)
        self.assertEqual(
            hf_model.decoder.weight.data_ptr(), hf_model.model.encoder.weight.data_ptr()
        )

    def test_lm_head_ngpt_does_not_tie_embeddings(self):
        """Ensure ngpt LM head does not tie token embeddings to decoder weights."""
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=16,
            max_length=8,
            attn_backend="sdpa",
            ngpt=True,
            hidden_act="gelu",
            tie_word_embeddings=True,
        )
        model = NeoBERTLMHead(config)
        self.assertNotEqual(
            model.decoder.weight.data_ptr(), model.model.encoder.weight.data_ptr()
        )
        self.assertFalse(model.config.tie_word_embeddings)

    def test_packed_seqlens_cuda_is_supported(self):
        """Ensure CUDA packed_seqlens metadata is accepted."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required to validate packed_seqlens on-device path.")
        from neobert.model.model import _normalize_packed_seqlens

        packed = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        normalized = _normalize_packed_seqlens(packed, seq_len=3)
        assert normalized is not None
        self.assertEqual(normalized.device.type, "cuda")
        self.assertEqual(normalized.dtype, torch.int32)

    def test_hf_positional_embedding_mode_runs_for_pad_token_variants(self):
        """Ensure HF positional-embedding mode supports default and nonzero pad tokens."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        cases = [
            (
                NeoBERTConfig(
                    hidden_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                    vocab_size=100,
                    max_length=16,
                    rope=False,
                    hidden_act="gelu",
                    flash_attention=False,
                ),
                torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]),
                torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
            ),
            (
                NeoBERTConfig(
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=32,
                    vocab_size=50,
                    max_length=8,
                    pad_token_id=7,
                    rope=False,
                    hidden_act="gelu",
                    flash_attention=False,
                ),
                torch.tensor([[1, 2, 3, 4, 5, 6, 8, 9]]),
                torch.ones((1, 8), dtype=torch.long),
            ),
        ]
        for config, input_ids, attention_mask in cases:
            model = NeoBERT(config)
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            self.assertEqual(
                outputs.last_hidden_state.shape,
                (input_ids.shape[0], input_ids.shape[1], config.hidden_size),
            )

    def test_rope_vs_positional_embeddings(self):
        """Test both RoPE and positional embedding modes."""
        # Test with RoPE (default)
        rope_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            rope=True,
            attn_backend="sdpa",
            vocab_size=1000,
            hidden_act="gelu",
        )
        rope_model = NeoBERT(rope_config)

        # Test with positional embeddings
        pos_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            rope=False,
            attn_backend="sdpa",
            vocab_size=1000,
            hidden_act="gelu",
        )
        pos_model = NeoBERT(pos_config)

        with torch.no_grad():
            rope_outputs = rope_model(self.input_ids, self.pad_mask)
            pos_outputs = pos_model(self.input_ids, self.pad_mask)

        # Both should produce valid outputs
        expected_shape = (self.batch_size, self.seq_length, 64)
        self.assertEqual(rope_outputs.shape, expected_shape)
        self.assertEqual(pos_outputs.shape, expected_shape)

        # Outputs should be different (different position encoding methods)
        self.assertFalse(torch.allclose(rope_outputs, pos_outputs, atol=1e-4))
        pos_nonzero_pad_config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=50,
            max_length=8,
            pad_token_id=7,
            rope=False,
            attn_backend="sdpa",
            hidden_act="gelu",
        )
        model = NeoBERT(pos_nonzero_pad_config)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 8, 9]])
        pad_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(input_ids, pad_mask)
        self.assertEqual(
            outputs.shape,
            (
                input_ids.shape[0],
                input_ids.shape[1],
                pos_nonzero_pad_config.hidden_size,
            ),
        )

    def test_rope_freqs_cis_is_buffer(self):
        """Ensure RoPE freqs_cis stays registered as a buffer."""
        rope_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            rope=True,
            attn_backend="sdpa",
            vocab_size=1000,
            hidden_act="gelu",
        )
        model = NeoBERT(rope_config)
        buffers = dict(model.named_buffers())
        self.assertIn("freqs_cis", buffers)
        self.assertNotIn("freqs_cis", model.state_dict())
        self.assertEqual(model.freqs_cis.shape[0], rope_config.max_length)
        ptr_before = model.freqs_cis.data_ptr()

        with torch.no_grad():
            _ = model(self.input_ids, self.pad_mask)
        buffers = dict(model.named_buffers())
        self.assertIn("freqs_cis", buffers)
        self.assertNotIn("freqs_cis", model.state_dict())
        self.assertGreater(model.freqs_cis.numel(), 0)
        self.assertEqual(model.freqs_cis.data_ptr(), ptr_before)

    def test_normalize_pad_mask_type_handling(self):
        """Ensure additive mask normalization enforces expected dtype semantics."""
        from neobert.model.model import _normalize_pad_mask

        pad_mask = torch.zeros((2, 4), dtype=torch.bfloat16)
        normalized = _normalize_pad_mask(pad_mask)
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertEqual(normalized.shape, (2, 1, 1, 4))

        int_mask = torch.ones((2, 4), dtype=torch.int64)
        with self.assertRaises(TypeError):
            _normalize_pad_mask(int_mask)

    def test_infer_single_segment_fallback_conditions(self):
        """Ensure packed-length inference falls back for unsupported masks."""
        from neobert.model.model import (
            _infer_single_segment_packed_seqlens_from_pad_mask,
        )

        fully_masked = torch.full((2, 4), float("-inf"), dtype=torch.float32)
        fully_masked[1, :2] = 0.0
        inferred = _infer_single_segment_packed_seqlens_from_pad_mask(
            fully_masked, seq_len=4
        )
        self.assertIsNone(inferred)

        pad_mask = torch.zeros((2, 4), dtype=torch.float32)
        with patch.object(
            torch.Tensor,
            "is_cuda",
            new_callable=PropertyMock,
            return_value=True,
        ):
            inferred = _infer_single_segment_packed_seqlens_from_pad_mask(
                pad_mask, seq_len=4
            )
        self.assertIsNone(inferred)

    def test_rotary_accepts_batched_freqs(self):
        """Ensure rotary helper supports batched frequency tensors."""
        from neobert.model.rotary import apply_rotary_emb, precompute_freqs_cis

        batch_size = 2
        seq_len = 4
        num_heads = 1
        head_dim = 4

        xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, num_heads, head_dim)
        freqs = precompute_freqs_cis(head_dim, seq_len)
        positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        batched_freqs = freqs[positions]

        xq_out, xk_out = apply_rotary_emb(xq, xk, batched_freqs)
        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)

    def test_rotary_training_matches_hf_export(self):
        """Ensure training and HF rotary helpers stay numerically aligned."""
        from neobert.huggingface.rotary import (
            apply_rotary_emb as hf_apply_rotary_emb,
            precompute_freqs_cis as hf_precompute_freqs_cis,
        )
        from neobert.model.rotary import (
            apply_rotary_emb as train_apply_rotary_emb,
            precompute_freqs_cis as train_precompute_freqs_cis,
        )

        batch_size, seq_len, num_heads, head_dim = 2, 6, 3, 8
        xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, num_heads, head_dim)

        train_freqs = train_precompute_freqs_cis(head_dim, seq_len)
        hf_freqs = hf_precompute_freqs_cis(head_dim, seq_len)

        train_q, train_k = train_apply_rotary_emb(xq, xk, train_freqs)
        hf_q, hf_k = hf_apply_rotary_emb(xq, xk, hf_freqs)

        torch.testing.assert_close(train_q, hf_q, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(train_k, hf_k, atol=1e-5, rtol=1e-5)

    def test_activation_functions(self):
        """Test different activation functions."""
        # Test SwiGLU
        swiglu_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_act="swiglu",
            attn_backend="sdpa",
            vocab_size=1000,
        )
        swiglu_model = NeoBERT(swiglu_config)

        # Test GELU
        gelu_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_act="GELU",
            attn_backend="sdpa",
            vocab_size=1000,
        )
        gelu_model = NeoBERT(gelu_config)

        with torch.no_grad():
            swiglu_outputs = swiglu_model(self.input_ids, self.pad_mask)
            gelu_outputs = gelu_model(self.input_ids, self.pad_mask)

        # Both should produce valid outputs
        expected_shape = (self.batch_size, self.seq_length, 64)
        self.assertEqual(swiglu_outputs.shape, expected_shape)
        self.assertEqual(gelu_outputs.shape, expected_shape)

    def test_hf_swiglu_uses_unpacked_weights(self):
        """Ensure HF SwiGLU uses unpacked w1/w2/w3 weights."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        hf_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=1000,
            max_length=32,
            hidden_act="swiglu",
            flash_attention=False,
        )
        model = NeoBERT(hf_config)
        state = model.state_dict()
        self.assertTrue(any(".ffn.w1.weight" in key for key in state))
        self.assertTrue(any(".ffn.w2.weight" in key for key in state))
        self.assertTrue(any(".ffn.w3.weight" in key for key in state))
        self.assertFalse(any(".ffn.w12.weight" in key for key in state))

    def test_hf_lm_head_is_biasless(self):
        """Ensure HF LM head decoder uses a biasless projection."""
        from neobert.huggingface.modeling_neobert import NeoBERTConfig, NeoBERTLMHead

        hf_config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=128,
            max_length=16,
            hidden_act="gelu",
            flash_attention=False,
        )
        model = NeoBERTLMHead(hf_config)
        self.assertIsNone(model.decoder.bias)

    def test_invalid_activation_raises(self):
        """Ensure unsupported activations fail fast."""
        with self.assertRaises(ValueError):
            NeoBERTConfig(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                hidden_act="relu",
                attn_backend="sdpa",
                vocab_size=1000,
            )

    def test_attention_mask_handling(self):
        """Test proper attention mask handling."""
        model = NeoBERT(self.tiny_config)
        model.eval()

        # Create input with padding
        padded_input = torch.tensor(
            [
                [1, 2, 3, 0, 0],  # Padded sequence
                [4, 5, 6, 7, 8],  # Full sequence
            ]
        )

        # Create attention mask (1 = attend, 0 = don't attend)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]).float()

        # Convert to additive mask for NeoBERT
        pad_mask = torch.where(attention_mask == 0, float("-inf"), float(0.0))

        with torch.no_grad():
            outputs = model(padded_input, pad_mask)

        # Check that outputs are valid
        self.assertFalse(torch.isnan(outputs).any())
        self.assertFalse(torch.isinf(outputs).any())

        # Padded positions should still have valid outputs (model handles internally)
        self.assertEqual(outputs.shape, (2, 5, self.tiny_config.hidden_size))

    def test_block_attention_mask(self):
        """Test 3D block attention masks are accepted."""
        model = NeoBERT(self.tiny_config)
        model.eval()

        block_mask = torch.full(
            (self.batch_size, self.seq_length, self.seq_length),
            float("-inf"),
        )
        eye = torch.eye(self.seq_length).unsqueeze(0).expand(self.batch_size, -1, -1)
        block_mask = torch.where(eye == 1, float(0.0), block_mask)

        with torch.no_grad():
            outputs = model(self.input_ids, block_mask)

        expected_shape = (
            self.batch_size,
            self.seq_length,
            self.tiny_config.hidden_size,
        )
        self.assertEqual(outputs.shape, expected_shape)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = NeoBERT(self.tiny_config)
        model.train()

        outputs = model(self.input_ids, self.pad_mask)

        # Create dummy loss
        loss = outputs.mean()
        loss.backward()

        # Check that gradients exist for key parameters
        self.assertIsNotNone(model.encoder.weight.grad)

        # Check that gradients are not zero
        self.assertNotEqual(model.encoder.weight.grad.sum().item(), 0.0)

    def test_model_determinism(self):
        """Test that model outputs are deterministic given same inputs."""
        torch.manual_seed(42)
        model1 = NeoBERT(self.tiny_config)

        torch.manual_seed(42)
        model2 = NeoBERT(self.tiny_config)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            outputs1 = model1(self.input_ids, self.pad_mask)
            outputs2 = model2(self.input_ids, self.pad_mask)

        # Outputs should be identical for same random seed
        self.assertTrue(torch.allclose(outputs1, outputs2, atol=1e-6))

    def test_seq_class_init_preserves_backbone_weights(self):
        """Ensure _init_weights does not overwrite backbone linear weights."""
        # Use a very large classifier_init_range so we can distinguish it from backbone init.
        large_range = 0.5
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=8,
            attn_backend="sdpa",
            hidden_act="gelu",
            decoder_init_range=0.02,
        )
        model = NeoBERTForSequenceClassification(
            config, num_labels=2, classifier_init_range=large_range
        )
        # Backbone QKV weights should be initialized with decoder_init_range (0.02),
        # NOT with classifier_init_range (0.5).
        for layer in model.model.transformer_encoder:
            qkv_weight = layer.qkv.weight
            max_val = qkv_weight.abs().max().item()
            self.assertLessEqual(
                max_val,
                config.decoder_init_range + 0.01,
                f"Backbone QKV weight max {max_val} exceeds decoder_init_range "
                f"({config.decoder_init_range}); _init_weights may be overwriting backbone.",
            )

    def test_sequence_classification_backbone_selection(self):
        """Ensure sequence-classification wrappers force SDPA and select backbone by ngpt."""
        cases = [
            (
                "train_flash",
                NeoBERTForSequenceClassification,
                NeoBERTConfig(
                    hidden_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                    vocab_size=100,
                    max_length=8,
                    attn_backend="flash_attn_varlen",
                    hidden_act="gelu",
                ),
                {"num_labels": 2},
                NeoBERT,
            ),
            (
                "train_ngpt",
                NeoBERTForSequenceClassification,
                NeoBERTConfig(
                    hidden_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                    vocab_size=100,
                    max_length=8,
                    attn_backend="sdpa",
                    ngpt=True,
                    hidden_act="gelu",
                ),
                {"num_labels": 2},
                NormNeoBERT,
            ),
            (
                "hf_flash",
                NeoBERTHFForSequenceClassification,
                NeoBERTConfig(
                    hidden_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                    vocab_size=100,
                    max_length=8,
                    attn_backend="flash_attn_varlen",
                    hidden_act="gelu",
                    num_labels=3,
                ),
                {},
                NeoBERT,
            ),
            (
                "hf_ngpt",
                NeoBERTHFForSequenceClassification,
                NeoBERTConfig(
                    hidden_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                    vocab_size=100,
                    max_length=8,
                    attn_backend="sdpa",
                    ngpt=True,
                    hidden_act="gelu",
                    num_labels=3,
                ),
                {},
                NormNeoBERT,
            ),
        ]
        for _, model_cls, config, kwargs, expected_backbone in cases:
            model = model_cls(config, **kwargs)
            self.assertEqual(model.model.config.attn_backend, "sdpa")
            self.assertIsInstance(model.model, expected_backbone)

    def test_training_config_default_vocab_matches_repo_defaults(self):
        """Ensure training config default vocab matches YAML/HF defaults."""
        config = NeoBERTConfig()
        self.assertEqual(config.vocab_size, 30522)

    def test_mteb_encode_with_sdpa_backend(self):
        """Ensure SDPA MTEB encode honors model device even when CUDA is available."""
        from neobert.model import NeoBERTConfig, NeoBERTForMTEB

        tokenizer = self._make_tokenizer()
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=10,
            max_length=8,
            attn_backend="sdpa",
            ngpt=False,
            hidden_act="gelu",
        )
        model = NeoBERTForMTEB(
            config=config,
            tokenizer=tokenizer,
            max_length=8,
            batch_size=2,
            pooling="avg",
        )
        model.to("cpu")
        model.eval()
        with patch("torch.cuda.is_available", return_value=True):
            embeddings = model.encode(["hello world", "hello"])
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], config.hidden_size)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA required for flash_attn_varlen MTEB test"
    )
    def test_mteb_encode_flash_attn_varlen_no_crash(self):
        """Ensure MTEB encode does not crash with attn_backend='flash_attn_varlen' on CUDA."""
        from neobert.model import NeoBERTConfig, NeoBERTForMTEB
        from neobert.kernels.attention import FLASH_ATTN_AVAILABLE

        if not FLASH_ATTN_AVAILABLE:
            self.skipTest(
                "flash-attn not installed; flash_attn_varlen MTEB test skipped."
            )

        tokenizer = self._make_tokenizer()
        device = torch.device("cuda")
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=10,
            max_length=8,
            attn_backend="flash_attn_varlen",
            ngpt=False,
            hidden_act="gelu",
        )
        model = NeoBERTForMTEB(
            config=config,
            tokenizer=tokenizer,
            max_length=8,
            batch_size=2,
            pooling="avg",
        )
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        embeddings = model.encode(["hello world", "hello"])
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], config.hidden_size)


if __name__ == "__main__":
    unittest.main()
