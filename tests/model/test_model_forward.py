#!/usr/bin/env python3
"""Test NeoBERT model forward passes and functionality."""

import unittest
from unittest.mock import patch

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
            flash_attention=False,  # Use regular attention for CPU testing
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
            flash_attention=False,
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
            flash_attention=False,
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
            flash_attention=False,
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

        # Check that we get both hidden states and logits
        self.assertIn("hidden_representation", outputs)
        self.assertIn("logits", outputs)

        hidden_shape = (self.batch_size, self.seq_length, self.tiny_config.hidden_size)
        logits_shape = (self.batch_size, self.seq_length, self.tiny_config.vocab_size)

        self.assertEqual(outputs["hidden_representation"].shape, hidden_shape)
        self.assertEqual(outputs["logits"].shape, logits_shape)

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
            flash_attention=False,
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

    def test_hf_attention_mask_blocks_padding(self):
        """Ensure HF attention masks properly zero out padding attention."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=16,
        )
        model = NeoBERT(config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        attn_weights = outputs.attentions[0]
        pad_mask = attention_mask == 0
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2).expand_as(attn_weights)
        self.assertLess(attn_weights.masked_select(pad_mask).max().item(), 1e-6)

    def test_hf_additive_attention_mask_supported(self):
        """Ensure additive 0/-inf masks are accepted in HF model."""
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

        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        additive_mask = torch.where(attention_mask == 1, float(0.0), float("-inf"))

        with torch.no_grad():
            outputs_bool = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs_add = model(input_ids=input_ids, attention_mask=additive_mask)

        self.assertTrue(
            torch.allclose(
                outputs_bool.last_hidden_state,
                outputs_add.last_hidden_state,
                atol=1e-6,
            )
        )

    def test_hf_sdpa_bool_mask_without_padding(self):
        """Ensure HF-style bool masks work when there is no padding."""
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

        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        # All-False bool mask (HF: keep nothing) -> SDPA: mask all
        all_false = torch.zeros_like(input_ids, dtype=torch.bool)
        normalized = model._normalize_attention_mask(all_false)
        self.assertTrue(normalized.all().item())  # All masked

        # All-True bool mask (HF: keep all) -> SDPA: mask none
        keep_all = torch.ones_like(input_ids, dtype=torch.bool)
        normalized_keep_all = model._normalize_attention_mask(keep_all)
        self.assertFalse(normalized_keep_all.any().item())  # None masked

        # Mixed mask: True=keep, False=mask -> inverted for SDPA
        mixed_mask = torch.tensor(
            [[True, False, True, False], [False, True, False, True]]
        )
        normalized_mixed = model._normalize_attention_mask(mixed_mask)
        expected = ~mixed_mask
        self.assertTrue(torch.equal(normalized_mixed, expected))

    def test_hf_bool_attention_mask_supported(self):
        """Ensure HF-style bool masks (True=keep) are accepted."""
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

        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        bool_mask = attention_mask == 1

        with torch.no_grad():
            outputs_int = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs_bool = model(input_ids=input_ids, attention_mask=bool_mask)

        self.assertTrue(
            torch.allclose(
                outputs_int.last_hidden_state,
                outputs_bool.last_hidden_state,
                atol=1e-6,
            )
        )

    def test_hf_flash_attention_silently_ignored(self):
        """Ensure flash_attention=True is silently accepted for config compat."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        # flash_attention=True should be accepted without warning
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

    def test_mteb_encode_respects_model_device(self):
        """Ensure MTEB encoder uses the model device over CUDA availability."""
        from neobert.model import NeoBERTConfig, NeoBERTForMTEB

        tokenizer = self._make_tokenizer()
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=10,
            max_length=8,
            flash_attention=False,
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

        with patch("torch.cuda.is_available", return_value=True):
            embeddings = model.encode(["hello world"])

        self.assertEqual(embeddings.shape[0], 1)

    def test_hf_rope_disabled_uses_positional_embeddings(self):
        """Ensure HF model runs when RoPE is disabled."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_length=16,
            rope=False,
            hidden_act="gelu",
            flash_attention=False,
        )
        model = NeoBERT(config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(
            outputs.last_hidden_state.shape,
            (input_ids.shape[0], input_ids.shape[1], config.hidden_size),
        )

    def test_hf_positional_embeddings_nonzero_pad_token(self):
        """Ensure HF positional embeddings handle non-zero pad_token_id."""
        from neobert.huggingface.modeling_neobert import NeoBERT, NeoBERTConfig

        config = NeoBERTConfig(
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
        )
        model = NeoBERT(config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 8, 9]])
        attention_mask = torch.ones_like(input_ids)

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
            flash_attention=False,
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
            flash_attention=False,
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

    def test_positional_embeddings_nonzero_pad_token(self):
        """Ensure positional embeddings handle non-zero pad_token_id."""
        pos_config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=50,
            max_length=8,
            pad_token_id=7,
            rope=False,
            flash_attention=False,
            hidden_act="gelu",
        )
        model = NeoBERT(pos_config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 8, 9]])
        pad_mask = torch.zeros_like(input_ids, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(input_ids, pad_mask)

        self.assertEqual(
            outputs.shape,
            (input_ids.shape[0], input_ids.shape[1], pos_config.hidden_size),
        )

    def test_rope_freqs_cis_is_buffer(self):
        """Ensure RoPE freqs_cis stays registered as a buffer."""
        rope_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            rope=True,
            flash_attention=False,
            vocab_size=1000,
            hidden_act="gelu",
        )
        model = NeoBERT(rope_config)
        buffers = dict(model.named_buffers())
        self.assertIn("freqs_cis", buffers)
        self.assertNotIn("freqs_cis", model.state_dict())

        with torch.no_grad():
            _ = model(self.input_ids, self.pad_mask)
        buffers = dict(model.named_buffers())
        self.assertIn("freqs_cis", buffers)
        self.assertNotIn("freqs_cis", model.state_dict())
        self.assertGreater(model.freqs_cis.numel(), 0)

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

    def test_activation_functions(self):
        """Test different activation functions."""
        # Test SwiGLU
        swiglu_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_act="swiglu",
            flash_attention=False,
            vocab_size=1000,
        )
        swiglu_model = NeoBERT(swiglu_config)

        # Test GELU
        gelu_config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_act="GELU",
            flash_attention=False,
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

    def test_hf_init_zeros_linear_biases(self):
        """Ensure HF init matches training bias initialization."""
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
        self.assertTrue(torch.all(model.decoder.bias == 0))

    def test_invalid_activation_raises(self):
        """Ensure unsupported activations fail fast."""
        with self.assertRaises(ValueError):
            NeoBERTConfig(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                hidden_act="relu",
                flash_attention=False,
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


if __name__ == "__main__":
    unittest.main()
