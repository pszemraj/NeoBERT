"""Masked-only MLM objective utilities for pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:  # pragma: no cover - guarded by optional dependency
    LigerFusedLinearCrossEntropyLoss = None  # type: ignore[assignment]


@dataclass
class MaskedObjectiveOut:
    """Container returned by ``MaskedPositionsOnlyMLMObjective``.

    :param torch.Tensor loss_sum_local: Local sum-reduction loss (float32 scalar).
    :param torch.Tensor num_masked_local: Local masked-token count (int64 scalar).
    :param str used_path: Identifier for the active compute path.
    :param torch.Tensor | None num_correct_local: Optional local correct count.
    """

    loss_sum_local: torch.Tensor
    num_masked_local: torch.Tensor
    used_path: str
    num_correct_local: Optional[torch.Tensor] = None


def gather_masked_index_select(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Gather masked positions with ``index_select``.

    :param torch.Tensor hidden_states: Hidden states of shape ``(B,S,H)`` or ``(N,H)``.
    :param torch.Tensor labels: Labels of shape ``(B,S)`` or ``(N,)``.
    :param int ignore_index: Ignore label value, defaults to ``-100``.
    :return tuple[torch.Tensor, torch.Tensor, int]:
        ``(masked_hidden, masked_targets, num_masked)``.
    """
    if hidden_states.ndim == 3:
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    elif hidden_states.ndim == 2:
        flat_hidden = hidden_states
    else:
        raise ValueError(
            "hidden_states must be rank-2 or rank-3, got "
            f"shape={tuple(hidden_states.shape)}"
        )

    flat_labels = labels.reshape(-1).to(device=flat_hidden.device, dtype=torch.long)
    masked = flat_labels.ne(ignore_index)
    index = torch.nonzero(masked, as_tuple=False).squeeze(1)
    num_masked = int(index.numel())

    if num_masked == 0:
        return flat_hidden[:0], flat_labels[:0], 0

    masked_hidden = torch.index_select(flat_hidden, dim=0, index=index)
    masked_targets = torch.index_select(flat_labels, dim=0, index=index)
    return masked_hidden, masked_targets, num_masked


def _estimate_masked_logits_bytes(
    num_tokens: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> int:
    """Estimate bytes for a masked ``(N_masked, vocab)`` logits tensor.

    :param int num_tokens: Number of masked tokens.
    :param int vocab_size: Vocabulary size.
    :param torch.dtype dtype: Logits dtype.
    :return int: Estimated bytes.
    """
    bytes_per_elem = 2 if dtype == torch.bfloat16 else 4
    return int(num_tokens) * int(vocab_size) * int(bytes_per_elem)


@torch.no_grad()
def streaming_argmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_chunk: int = 2048,
    vocab_chunk: int = 8192,
    accum_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute argmax over vocab without allocating ``(N,V)`` logits.

    :param torch.Tensor hidden: Hidden states of shape ``(N,H)``.
    :param torch.Tensor weight: Decoder weight of shape ``(V,H)``.
    :param int token_chunk: Token chunk size.
    :param int vocab_chunk: Vocab chunk size.
    :param torch.dtype accum_dtype: Accumulation dtype.
    :return torch.Tensor: Predicted token IDs of shape ``(N,)``.
    """
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("Expected hidden=(N,H) and weight=(V,H)")

    n_tokens, hidden_size = hidden.shape
    vocab_size, weight_hidden = weight.shape
    if hidden_size != weight_hidden:
        raise ValueError(
            "Hidden-size mismatch between hidden and weight: "
            f"{hidden_size} != {weight_hidden}"
        )

    preds = torch.empty((n_tokens,), device=hidden.device, dtype=torch.long)
    minus_inf = torch.tensor(float("-inf"), device=hidden.device, dtype=accum_dtype)

    for tok_start in range(0, n_tokens, token_chunk):
        tok_end = min(tok_start + token_chunk, n_tokens)
        hidden_chunk = hidden[tok_start:tok_end].to(accum_dtype)
        local_count = hidden_chunk.shape[0]

        best_values = minus_inf.expand(local_count).clone()
        best_indices = torch.zeros(
            (local_count,), device=hidden.device, dtype=torch.long
        )

        for vocab_start in range(0, vocab_size, vocab_chunk):
            vocab_end = min(vocab_start + vocab_chunk, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end].to(accum_dtype)
            logits_chunk = hidden_chunk @ weight_chunk.t()
            chunk_values, chunk_argmax = logits_chunk.max(dim=1)
            better = chunk_values > best_values
            if better.any():
                best_values[better] = chunk_values[better]
                best_indices[better] = chunk_argmax[better] + vocab_start

        preds[tok_start:tok_end] = best_indices

    return preds


@torch.no_grad()
def streaming_ce_sum(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    token_chunk: int = 2048,
    vocab_chunk: int = 8192,
    accum_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute CE sum without materializing full masked logits.

    :param torch.Tensor hidden: Hidden states of shape ``(N,H)``.
    :param torch.Tensor weight: Decoder weight of shape ``(V,H)``.
    :param torch.Tensor target: Targets of shape ``(N,)``.
    :param int token_chunk: Token chunk size.
    :param int vocab_chunk: Vocab chunk size.
    :param torch.dtype accum_dtype: Accumulation dtype.
    :return torch.Tensor: Float scalar CE sum.
    """
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("Expected hidden=(N,H) and weight=(V,H)")

    n_tokens, hidden_size = hidden.shape
    vocab_size, weight_hidden = weight.shape
    if hidden_size != weight_hidden:
        raise ValueError(
            "Hidden-size mismatch between hidden and weight: "
            f"{hidden_size} != {weight_hidden}"
        )

    target = target.to(device=hidden.device, dtype=torch.long)
    total = torch.zeros((), device=hidden.device, dtype=accum_dtype)

    for tok_start in range(0, n_tokens, token_chunk):
        tok_end = min(tok_start + token_chunk, n_tokens)
        hidden_chunk = hidden[tok_start:tok_end].to(accum_dtype)
        target_chunk = target[tok_start:tok_end]
        local_count = hidden_chunk.shape[0]

        running_max = torch.full(
            (local_count,), float("-inf"), device=hidden.device, dtype=accum_dtype
        )
        running_sum = torch.zeros(
            (local_count,), device=hidden.device, dtype=accum_dtype
        )
        target_logit = torch.full(
            (local_count,), float("-inf"), device=hidden.device, dtype=accum_dtype
        )

        for vocab_start in range(0, vocab_size, vocab_chunk):
            vocab_end = min(vocab_start + vocab_chunk, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end].to(accum_dtype)
            logits_chunk = hidden_chunk @ weight_chunk.t()

            in_chunk = (target_chunk >= vocab_start) & (target_chunk < vocab_end)
            if in_chunk.any():
                rows = torch.nonzero(in_chunk, as_tuple=False).squeeze(1)
                cols = (target_chunk[in_chunk] - vocab_start).to(torch.long)
                target_logit[in_chunk] = logits_chunk[rows, cols]

            chunk_max = logits_chunk.max(dim=1).values
            new_max = torch.maximum(running_max, chunk_max)
            running_sum = running_sum * torch.exp(running_max - new_max)
            running_sum = running_sum + torch.exp(
                logits_chunk - new_max.unsqueeze(1)
            ).sum(dim=1)
            running_max = new_max

        lse = running_max + torch.log(running_sum)
        total = total + (lse - target_logit).sum()

    return total


class MaskedPositionsOnlyMLMObjective(nn.Module):
    """Masked-token-only MLM objective with fused-first dispatch.

    :param int ignore_index: Ignore label value, defaults to ``-100``.
    :param int token_chunk_train: Token chunk size for checkpoint fallback.
    :param str eval_loss_mode: ``auto``, ``masked_logits``, or ``streaming``.
    :param int max_masked_logits_bytes_eval: Max bytes to allow masked logits in eval.
    :param int token_chunk_eval: Token chunk size for eval streaming kernels.
    :param int vocab_chunk_eval: Vocab chunk size for eval streaming kernels.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        token_chunk_train: int = 2048,
        eval_loss_mode: Literal["auto", "masked_logits", "streaming"] = "auto",
        max_masked_logits_bytes_eval: int = 512 * 1024 * 1024,
        token_chunk_eval: int = 2048,
        vocab_chunk_eval: int = 8192,
    ) -> None:
        """Initialize masked-only objective settings and fallback policy.

        :param int ignore_index: Ignore label value, defaults to ``-100``.
        :param int token_chunk_train: Token chunk size for checkpointed fallback.
        :param str eval_loss_mode: ``auto``, ``masked_logits``, or ``streaming``.
        :param int max_masked_logits_bytes_eval: Max masked-logits bytes in eval.
        :param int token_chunk_eval: Eval token chunk size for streaming kernels.
        :param int vocab_chunk_eval: Eval vocab chunk size for streaming kernels.
        """
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.token_chunk_train = int(token_chunk_train)
        self.eval_loss_mode = eval_loss_mode
        self.max_masked_logits_bytes_eval = int(max_masked_logits_bytes_eval)
        self.token_chunk_eval = int(token_chunk_eval)
        self.vocab_chunk_eval = int(vocab_chunk_eval)

        if LigerFusedLinearCrossEntropyLoss is None:
            self._flce = None
        else:
            self._flce = LigerFusedLinearCrossEntropyLoss(
                ignore_index=self.ignore_index,
                reduction="sum",
            )

    def _masked_num_correct(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        lm_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked-token accuracy count without ``(N,V)`` allocation.

        :param torch.Tensor hidden_states: Masked hidden states ``(N,H)``.
        :param torch.Tensor targets: Masked targets ``(N,)``.
        :param torch.Tensor lm_weight: Decoder weight ``(V,H)``.
        :return torch.Tensor: Scalar correct count.
        """
        preds = streaming_argmax(
            hidden=hidden_states,
            weight=lm_weight,
            token_chunk=self.token_chunk_eval,
            vocab_chunk=self.vocab_chunk_eval,
            accum_dtype=torch.float32,
        )
        return (preds == targets).sum(dtype=torch.long)

    def _checkpointed_masked_ce_sum(
        self,
        masked_hidden: torch.Tensor,
        masked_targets: torch.Tensor,
        lm_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked CE sum with activation checkpointing.

        :param torch.Tensor masked_hidden: Masked hidden states ``(N,H)``.
        :param torch.Tensor masked_targets: Masked targets ``(N,)``.
        :param torch.Tensor lm_weight: Decoder weight ``(V,H)``.
        :return torch.Tensor: Float32 CE sum.
        """

        def _chunk_loss(
            hidden_chunk: torch.Tensor,
            weight: torch.Tensor,
            target_chunk: torch.Tensor,
        ) -> torch.Tensor:
            """Compute CE sum for one masked token chunk.

            :param torch.Tensor hidden_chunk: Chunk hidden states ``(N,H)``.
            :param torch.Tensor weight: Decoder weight ``(V,H)``.
            :param torch.Tensor target_chunk: Chunk targets ``(N,)``.
            :return torch.Tensor: Float scalar chunk loss sum.
            """
            logits = F.linear(hidden_chunk, weight)
            return F.cross_entropy(logits.float(), target_chunk, reduction="sum")

        total = torch.zeros((), device=masked_hidden.device, dtype=torch.float32)
        num_masked = int(masked_targets.numel())
        for tok_start in range(0, num_masked, self.token_chunk_train):
            tok_end = min(tok_start + self.token_chunk_train, num_masked)
            total = total + checkpoint(
                _chunk_loss,
                masked_hidden[tok_start:tok_end],
                lm_weight,
                masked_targets[tok_start:tok_end],
                use_reentrant=False,
            )
        return total

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        lm_weight: torch.Tensor,
        *,
        compute_accuracy: bool = False,
    ) -> MaskedObjectiveOut:
        """Compute local loss sum and token count for masked MLM.

        :param torch.Tensor hidden_states: Hidden states ``(B,S,H)`` or ``(N,H)``.
        :param torch.Tensor labels: Labels ``(B,S)`` or ``(N,)``.
        :param torch.Tensor lm_weight: Decoder weight ``(V,H)``.
        :param bool compute_accuracy: Whether to compute masked accuracy.
        :return MaskedObjectiveOut: Local objective outputs.
        """
        if hidden_states.dtype == torch.float16 or lm_weight.dtype == torch.float16:
            raise RuntimeError(
                "fp16 is not supported in masked-only MLM objective. "
                "Use bf16 autocast or fp32."
            )

        grad_enabled = torch.is_grad_enabled()
        device = hidden_states.device

        masked_hidden, masked_targets, num_masked = gather_masked_index_select(
            hidden_states,
            labels,
            ignore_index=self.ignore_index,
        )
        num_masked_local = torch.tensor(num_masked, device=device, dtype=torch.long)

        if num_masked == 0:
            # Touch both hidden_states and lm_weight to keep DDP/FSDP parameter usage stable.
            loss_sum_local = (hidden_states.sum() * 0.0) + (lm_weight.sum() * 0.0)
            num_correct_local = (
                torch.zeros((), device=device, dtype=torch.long)
                if compute_accuracy
                else None
            )
            return MaskedObjectiveOut(
                loss_sum_local=loss_sum_local.float(),
                num_masked_local=num_masked_local,
                used_path="zero_masked",
                num_correct_local=num_correct_local,
            )

        can_try_flce = (
            self._flce is not None
            and masked_hidden.is_cuda
            and lm_weight.is_cuda
            and masked_hidden.dtype in (torch.bfloat16, torch.float32)
            and lm_weight.dtype in (torch.bfloat16, torch.float32)
        )

        if can_try_flce:
            try:
                flce_weight = lm_weight
                if flce_weight.dtype != masked_hidden.dtype:
                    # FLCE requires matching dtypes and is notably faster when
                    # bf16 activations run against bf16 weights under autocast.
                    # Keep the cast local to this call so optimizer state can
                    # remain fp32.
                    flce_weight = flce_weight.to(dtype=masked_hidden.dtype)
                flce_hidden = (
                    masked_hidden
                    if masked_hidden.is_contiguous()
                    else masked_hidden.contiguous()
                )
                flce_weight = (
                    flce_weight
                    if flce_weight.is_contiguous()
                    else flce_weight.contiguous()
                )
                flce_loss = self._flce(flce_weight, flce_hidden, masked_targets)
                loss_sum_local = flce_loss.float()
                num_correct_local = None
                if compute_accuracy:
                    num_correct_local = self._masked_num_correct(
                        masked_hidden,
                        masked_targets,
                        lm_weight,
                    )
                return MaskedObjectiveOut(
                    loss_sum_local=loss_sum_local,
                    num_masked_local=num_masked_local,
                    used_path="liger_flce",
                    num_correct_local=num_correct_local,
                )
            except (RuntimeError, ValueError):
                # Fall through to masked-only checkpoint fallback.
                pass

        if grad_enabled:
            loss_sum_local = self._checkpointed_masked_ce_sum(
                masked_hidden,
                masked_targets,
                lm_weight,
            )
            num_correct_local = None
            if compute_accuracy:
                num_correct_local = self._masked_num_correct(
                    masked_hidden,
                    masked_targets,
                    lm_weight,
                )
            return MaskedObjectiveOut(
                loss_sum_local=loss_sum_local,
                num_masked_local=num_masked_local,
                used_path="train_checkpointed_masked_ce",
                num_correct_local=num_correct_local,
            )

        vocab_size = int(lm_weight.size(0))
        estimated_bytes = _estimate_masked_logits_bytes(
            num_tokens=num_masked,
            vocab_size=vocab_size,
            dtype=masked_hidden.dtype,
        )

        if self.eval_loss_mode == "masked_logits":
            use_masked_logits = True
        elif self.eval_loss_mode == "streaming":
            use_masked_logits = False
        else:
            use_masked_logits = estimated_bytes <= self.max_masked_logits_bytes_eval

        if use_masked_logits:
            masked_logits = F.linear(masked_hidden, lm_weight)
            loss_sum_local = F.cross_entropy(
                masked_logits.float(),
                masked_targets,
                reduction="sum",
            )
            num_correct_local = None
            if compute_accuracy:
                num_correct_local = (
                    masked_logits.argmax(dim=-1)
                    .eq(masked_targets)
                    .sum(dtype=torch.long)
                )
            return MaskedObjectiveOut(
                loss_sum_local=loss_sum_local,
                num_masked_local=num_masked_local,
                used_path="eval_masked_logits_ce",
                num_correct_local=num_correct_local,
            )

        loss_sum_local = streaming_ce_sum(
            hidden=masked_hidden,
            weight=lm_weight,
            target=masked_targets,
            token_chunk=self.token_chunk_eval,
            vocab_chunk=self.vocab_chunk_eval,
            accum_dtype=torch.float32,
        ).float()
        num_correct_local = None
        if compute_accuracy:
            num_correct_local = self._masked_num_correct(
                masked_hidden,
                masked_targets,
                lm_weight,
            )
        return MaskedObjectiveOut(
            loss_sum_local=loss_sum_local,
            num_masked_local=num_masked_local,
            used_path="eval_streaming_ce",
            num_correct_local=num_correct_local,
        )
