#!/usr/bin/env python3
"""Test NeoBERT model forward passes and functionality."""

import unittest

import torch

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

    def test_activation_functions(self):
        """Test different activation functions."""
        # Skip SwiGLU test due to xformers version mismatch
        # Test SwiGLU (default)
        # swiglu_config = NeoBERTConfig(
        #     hidden_size=64,
        #     num_hidden_layers=1,
        #     num_attention_heads=2,
        #     hidden_act="SwiGLU",
        #     flash_attention=False,
        #     vocab_size=1000,
        # )
        # swiglu_model = NeoBERT(swiglu_config)

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
            # swiglu_outputs = swiglu_model(self.input_ids, self.pad_mask)
            gelu_outputs = gelu_model(self.input_ids, self.pad_mask)

        # Both should produce valid outputs
        expected_shape = (self.batch_size, self.seq_length, 64)
        # self.assertEqual(swiglu_outputs.shape, expected_shape)
        self.assertEqual(gelu_outputs.shape, expected_shape)

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
