"""Tests for attention backend dispatch."""

import pytest
import torch

from neobert.kernels.attention import (
    FLASH_ATTN_AVAILABLE,
    attention_forward,
)


class TestAttentionForwardSDPA:
    """Tests for SDPA attention dispatch (unpacked)."""

    def test_unpacked_basic(self):
        B, S, H, D = 2, 8, 4, 16
        xq = torch.randn(B, S, H, D)
        xk = torch.randn(B, S, H, D)
        xv = torch.randn(B, S, H, D)
        out = attention_forward(xq, xk, xv, None, None, 0.0, None, "sdpa")
        assert out.shape == (B, S, H, D)

    def test_unpacked_with_mask(self):
        B, S, H, D = 2, 8, 4, 16
        xq = torch.randn(B, S, H, D)
        xk = torch.randn(B, S, H, D)
        xv = torch.randn(B, S, H, D)
        # 4D additive mask (B, 1, 1, S)
        mask = torch.zeros(B, 1, 1, S)
        mask[:, :, :, -2:] = float("-inf")
        out = attention_forward(xq, xk, xv, mask, None, 0.0, None, "sdpa")
        assert out.shape == (B, S, H, D)


class TestAttentionForwardSDPAPacked:
    """Tests for SDPA per-segment fallback (packed)."""

    def test_packed_sdpa_fallback(self):
        B, S, H, D = 2, 16, 2, 8
        xq = torch.randn(B, S, H, D)
        xk = torch.randn(B, S, H, D)
        xv = torch.randn(B, S, H, D)
        packed = [[8, 8], [16]]
        out = attention_forward(xq, xk, xv, None, packed, 0.0, None, "sdpa")
        assert out.shape == (B, S, H, D)

    def test_packed_sdpa_tensor_metadata(self):
        B, S, H, D = 2, 16, 2, 8
        xq = torch.randn(B, S, H, D)
        xk = torch.randn(B, S, H, D)
        xv = torch.randn(B, S, H, D)
        packed = torch.tensor([[8, 8, 0], [16, 0, 0]], dtype=torch.int32)
        out = attention_forward(xq, xk, xv, None, packed, 0.0, None, "sdpa")
        assert out.shape == (B, S, H, D)

    def test_packed_sdpa_zeros_in_padding(self):
        """Padding positions should be zero in output."""
        B, S, H, D = 1, 8, 2, 4
        xq = torch.randn(B, S, H, D)
        xk = torch.randn(B, S, H, D)
        xv = torch.randn(B, S, H, D)
        packed = [[4]]  # Only 4 of 8 tokens are valid
        out = attention_forward(xq, xk, xv, None, packed, 0.0, None, "sdpa")
        # Tokens 4-7 should be zero (padding region)
        assert torch.allclose(out[0, 4:], torch.zeros(4, H, D))

    def test_flash_backend_on_cpu_raises(self, monkeypatch: pytest.MonkeyPatch):
        B, S, H, D = 1, 8, 2, 4
        xq = torch.randn(B, S, H, D)
        packed = [[8]]
        monkeypatch.setattr(
            "neobert.kernels.attention.FLASH_ATTN_AVAILABLE", True, raising=False
        )
        monkeypatch.setattr(
            "neobert.kernels.attention._flash_attn_varlen_func", object(), raising=False
        )
        with pytest.raises(RuntimeError, match="requires CUDA tensors"):
            attention_forward(xq, xq, xq, None, packed, 0.0, None, "flash_attn_varlen")


@pytest.mark.skipif(
    not (
        FLASH_ATTN_AVAILABLE
        and torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    ),
    reason="flash-attn not installed, no CUDA, or no CUDA bf16 support",
)
class TestAttentionForwardFlash:
    """Tests for flash_attn_varlen dispatch (GPU only)."""

    def test_flash_packed(self):
        B, S, H, D = 2, 16, 2, 32
        xq = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        xk = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        xv = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        packed = [[8, 8], [16]]
        out = attention_forward(
            xq, xk, xv, None, packed, 0.0, None, "flash_attn_varlen"
        )
        assert out.shape == (B, S, H, D)
        assert out.device.type == "cuda"
