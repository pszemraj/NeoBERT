"""Tests for attention backend dispatch."""

import pytest
import torch

from neobert.kernels.attention import (
    FLASH_ATTN_AVAILABLE,
    attention_forward,
    packed_seqlens_to_cu_seqlens,
    resolve_attn_backend,
)


class TestResolveAttnBackend:
    """Tests for resolve_attn_backend()."""

    def test_sdpa(self):
        assert resolve_attn_backend("sdpa") == "sdpa"

    def test_flash_attn_varlen(self):
        if FLASH_ATTN_AVAILABLE:
            assert resolve_attn_backend("flash_attn_varlen") == "flash_attn_varlen"
        else:
            with pytest.raises(ImportError, match="flash-attn"):
                resolve_attn_backend("flash_attn_varlen")

    def test_flash_alias(self):
        if FLASH_ATTN_AVAILABLE:
            assert resolve_attn_backend("flash") == "flash_attn_varlen"

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown attn_backend"):
            resolve_attn_backend("invalid")


class TestPackedSeqlensToCuSeqlens:
    """Tests for packed_seqlens_to_cu_seqlens()."""

    def test_basic(self):
        packed = [[100, 50], [200], [80, 70]]
        cu, max_sl = packed_seqlens_to_cu_seqlens(packed, torch.device("cpu"))
        assert cu.dtype == torch.int32
        assert cu.tolist() == [0, 100, 150, 350, 430, 500]
        assert max_sl == 200

    def test_empty(self):
        cu, max_sl = packed_seqlens_to_cu_seqlens([], torch.device("cpu"))
        assert cu.tolist() == [0]
        assert max_sl == 0

    def test_single_segment(self):
        packed = [[128]]
        cu, max_sl = packed_seqlens_to_cu_seqlens(packed, torch.device("cpu"))
        assert cu.tolist() == [0, 128]
        assert max_sl == 128


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


@pytest.mark.skipif(
    not (FLASH_ATTN_AVAILABLE and torch.cuda.is_available()),
    reason="flash-attn not installed or no CUDA",
)
class TestAttentionForwardFlash:
    """Tests for flash_attn_varlen dispatch (GPU only)."""

    def test_flash_packed(self):
        B, S, H, D = 2, 16, 2, 32
        xq = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        xk = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        xv = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        packed = [[8, 8], [16]]
        out = attention_forward(
            xq, xk, xv, None, packed, 0.0, None, "flash_attn_varlen"
        )
        assert out.shape == (B, S, H, D)
        assert out.device.type == "cuda"
