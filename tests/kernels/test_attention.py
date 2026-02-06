"""Tests for attention backend dispatch."""

import pytest
import torch

from neobert.kernels.attention import (
    FLASH_ATTN_AVAILABLE,
    _flash_varlen_attention,
    attention_forward,
    canonicalize_attn_backend,
    packed_seqlens_to_cu_seqlens,
    resolve_attn_backend,
    resolve_runtime_attn_backend,
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


class TestResolveRuntimeAttnBackend:
    """Tests for runtime backend resolution with optional fallback."""

    def test_runtime_flash_fallbacks_without_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "neobert.kernels.attention.FLASH_ATTN_AVAILABLE", True, raising=False
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert resolve_runtime_attn_backend("flash", fallback_to_sdpa=True) == "sdpa"
        with pytest.raises(RuntimeError, match="requires CUDA"):
            resolve_runtime_attn_backend("flash", fallback_to_sdpa=False)

    def test_runtime_flash_fallbacks_when_flash_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "neobert.kernels.attention.FLASH_ATTN_AVAILABLE", False, raising=False
        )
        monkeypatch.setattr(
            "neobert.kernels.attention.FLASH_ATTN_ERROR",
            "flash-attn not installed",
            raising=False,
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert resolve_runtime_attn_backend("flash", fallback_to_sdpa=True) == "sdpa"
        with pytest.raises(ImportError, match="flash-attn"):
            resolve_runtime_attn_backend("flash", fallback_to_sdpa=False)


class TestCanonicalizeAttnBackend:
    """Tests for canonicalize_attn_backend()."""

    def test_canonicalize_aliases(self):
        assert canonicalize_attn_backend("flash") == "flash_attn_varlen"
        assert canonicalize_attn_backend("flash_attn") == "flash_attn_varlen"
        assert canonicalize_attn_backend("sdpa") == "sdpa"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown attn_backend"):
            canonicalize_attn_backend("bad_backend")


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


class TestFlashVarlenAttentionInternals:
    """Tests for flash varlen metadata flattening internals."""

    def test_flash_varlen_compacts_sparse_segment_columns(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        B, S, H, D = 2, 6, 2, 8
        xq = torch.randn(B, S, H, D)
        xk = torch.randn(B, S, H, D)
        xv = torch.randn(B, S, H, D)
        packed = torch.tensor([[3, 0, 2], [1, 0, 0]], dtype=torch.int32)

        captured: dict[str, object] = {}

        def _fake_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs):
            captured["tokens"] = q.shape[0]
            captured["cu"] = kwargs["cu_seqlens_q"].detach().cpu().tolist()
            return torch.zeros_like(q)

        monkeypatch.setattr(
            "neobert.kernels.attention._flash_attn_varlen_func",
            _fake_flash,
            raising=False,
        )

        out = _flash_varlen_attention(xq, xk, xv, packed, dropout_p=0.0, scale=None)

        assert out.shape == (B, S, H, D)
        assert captured["tokens"] == 6
        assert captured["cu"] == [0, 3, 5, 6]


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
