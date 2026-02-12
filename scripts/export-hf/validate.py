#!/usr/bin/env python3
"""
Validate an exported Hugging Face model directory.

Checks:
  - required files exist
  - AutoModel loads and runs a forward pass
  - AutoTokenizer loads and tokenizes
  - AutoModelForMaskedLM loads and runs (to catch missing head / bad init)
  - simple end-to-end encode on a few texts
  - cosine-similarity sanity on CLS embeddings

Notes:
  - Does NOT capture or mute transformers logs; warnings/errors will print normally.
  - Deterministically fails on loader issues via `output_loading_info` (missing/unexpected/mismatched keys).
  - `--strict` turns Python warnings into errors (does not touch transformers logging).
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


def _load_config(model_path: Path) -> Dict[str, Any]:
    """Load config.json if present.

    :param Path model_path: Model directory path.
    :return dict[str, Any]: Parsed config mapping (empty if missing).
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    import json

    with open(config_path, "r") as f:
        return json.load(f)


def _check_required_config_fields(config: Dict[str, Any]) -> Optional[str]:
    """Verify config contains fields required by HF NeoBERT.

    :param dict[str, Any] config: Loaded config mapping.
    :return str | None: Issue summary if missing fields.
    """
    required = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "norm_eps",
        "pad_token_id",
        "rms_norm",
        "rope",
        "hidden_act",
        "dropout",
        "flash_attention",
    ]
    missing = [field for field in required if field not in config]
    if missing:
        return f"missing config fields: {missing}"
    return None


def _check_swiglu_layout(model: object, config: Dict[str, Any]) -> Optional[str]:
    """Check that SwiGLU weights are unpacked (w1/w2/w3).

    :param object model: Loaded model instance.
    :param dict[str, Any] config: Loaded config mapping.
    :return str | None: Issue description if layout is wrong, else None.
    """
    if not config:
        return None
    if str(config.get("hidden_act", "")).lower() != "swiglu":
        return None
    state = model.state_dict()
    has_w12 = any(".ffn.w12." in key for key in state.keys())
    has_w1 = any(".ffn.w1." in key for key in state.keys())
    has_w2 = any(".ffn.w2." in key for key in state.keys())
    has_w3 = any(".ffn.w3." in key for key in state.keys())
    if has_w12:
        return "packed SwiGLU weights (ffn.w12) found; expected unpacked w1/w2/w3"
    if not (has_w1 and has_w2 and has_w3):
        return "missing unpacked SwiGLU weights (w1/w2/w3)"
    return None


def validate_model_files(model_path: Path) -> List[str]:
    """Check for required files in an exported model directory.

    :param Path model_path: Model directory to inspect.
    :return list[str]: Missing file names.
    """
    # Project-specific expectations
    required_files = [
        "config.json",
        "model.py",
        "rotary.py",
    ]
    weight_files = ["model.safetensors", "pytorch_model.bin"]

    missing = [f for f in required_files if not (model_path / f).exists()]
    if not any((model_path / f).exists() for f in weight_files):
        missing.append("model weights (model.safetensors or pytorch_model.bin)")
    config = _load_config(model_path)
    config_issues = _check_required_config_fields(config)
    if config_issues:
        missing.append(config_issues)
    return missing


def _from_pretrained_with_info(
    model_cls: Any, model_path: Path, verbose: bool = False
) -> Tuple[object, Optional[dict]]:
    """Load a model with output_loading_info when supported.

    :param Any model_cls: AutoModel class to load.
    :param Path model_path: Model directory path.
    :param bool verbose: Whether to print model info.
    :return tuple[object, dict | None]: Loaded model and loading info.
    """
    # Prefer output_loading_info for deterministic checks; fall back if remote code doesn't support it.
    try:
        model, loading_info = model_cls.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            output_loading_info=True,
        )
        if verbose:
            print("Model info:")
            print(model)

    except TypeError:
        model = model_cls.from_pretrained(str(model_path), trust_remote_code=True)
        loading_info = None
    return model, loading_info


def _format_loading_issues(loading_info: Optional[dict]) -> Optional[str]:
    """Format loading issues from transformers output.

    :param dict | None loading_info: Loading info from transformers.
    :return str | None: Formatted issue summary.
    """
    if not loading_info:
        return None
    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    if not (missing or unexpected or mismatched):
        return None
    parts = []
    if missing:
        parts.append(f"missing_keys={len(missing)} e.g. {missing[:5]}")
    if unexpected:
        parts.append(f"unexpected_keys={len(unexpected)} e.g. {unexpected[:5]}")
    if mismatched:
        parts.append(
            f"mismatched_keys={len(mismatched)} e.g. {[str(m) for m in mismatched[:5]]}"
        )
    return "; ".join(parts)


def test_model_loading(model_path: Path) -> Tuple[bool, str]:
    """
    test_model_loading - tries to load the model (AutoModel) and run a forward pass

    :param Path model_path: path to model directory
    :return Tuple[bool, str]: (success, message)
    """
    try:
        model, loading_info = _from_pretrained_with_info(AutoModel, model_path)

        # Check if model has MLM head weights that AutoModel doesn't expect
        # This is normal for models that support AutoModelForMaskedLM
        issues = _format_loading_issues(loading_info)
        if issues:
            # Load config to check if this model supports MaskedLM
            config_path = model_path / "config.json"
            if config_path.exists():
                import json

                with open(config_path, "r") as f:
                    config = json.load(f)
                architectures = config.get("architectures", [])
                auto_map = config.get("auto_map", {})

                # If model supports MaskedLM, decoder weights are expected
                if "AutoModelForMaskedLM" in auto_map or any(
                    "MaskedLM" in arch for arch in architectures
                ):
                    # Filter out decoder-related unexpected keys
                    if loading_info and loading_info.get("unexpected_keys"):
                        unexpected = loading_info["unexpected_keys"]
                        # Keep only unexpected keys that are NOT decoder-related
                        non_decoder_unexpected = [
                            k for k in unexpected if not k.startswith("decoder.")
                        ]
                        if not non_decoder_unexpected:
                            # Only decoder weights were unexpected, which is fine
                            issues = None

        if issues:
            return False, "load issues: " + issues

        config = _load_config(model_path)
        config_issues = _check_required_config_fields(config)
        if config_issues:
            return False, "config issues: " + config_issues
        swiglu_issues = _check_swiglu_layout(model, config)
        if swiglu_issues:
            return False, "swiglu layout mismatch: " + swiglu_issues

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        inputs = tokenizer("forward pass check.", return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)

        if getattr(out, "last_hidden_state", None) is None:
            return False, "forward produced no last_hidden_state"

        print(f"model_load: OK; last_hidden_state {tuple(out.last_hidden_state.shape)}")
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_tokenizer_loading(model_path: Path) -> Tuple[bool, str]:
    """Test that the tokenizer loads and returns input_ids.

    :param Path model_path: Model directory path.
    :return tuple[bool, str]: Success flag and message.
    """
    try:
        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        toks = tok("tokenizer check.", return_tensors="pt")
        if getattr(toks, "input_ids", None) is None:
            return False, "tokenizer returned None input_ids"
        print(f"tokenizer: OK; n_tokens {len(toks.input_ids[0])}")
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_masked_lm_loading(model_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    """Test AutoModelForMaskedLM loading and forward pass.

    :param Path model_path: Model directory path.
    :param bool verbose: Whether to print model info.
    :return tuple[bool, str]: Success flag and message.
    """
    try:
        mlm, loading_info = _from_pretrained_with_info(
            AutoModelForMaskedLM, model_path, verbose=verbose
        )
        issues = _format_loading_issues(loading_info)
        if issues:
            return False, "mlm load issues: " + issues

        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        inp = tok("sanity [MASK] token.", return_tensors="pt")
        with torch.no_grad():
            out = mlm(**inp)

        if getattr(out, "logits", None) is None:
            return False, "mlm forward produced no logits"

        print(f"mlm_load: OK; logits {tuple(out.logits.shape)}")
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_attention_mask_parity(
    model_path: Path,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Tuple[bool, str]:
    """Verify attention-mask parity across equivalent mask representations.

    :param Path model_path: Model directory path.
    :param float atol: Absolute tolerance for tensor comparisons.
    :param float rtol: Relative tolerance for tensor comparisons.
    :return tuple[bool, str]: Success flag and message.
    """
    try:
        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
        model.eval()

        inputs = tok(
            [
                "mask parity check sentence.",
                "short mask test.",
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        int_mask = inputs.get("attention_mask", None)
        if int_mask is None:
            return False, "tokenizer did not return attention_mask"

        ones_mask = torch.ones_like(int_mask)
        bool_mask = int_mask.bool()
        additive_mask = torch.where(
            int_mask == 1,
            torch.tensor(0.0),
            torch.tensor(float("-inf")),
        )
        additive_mask = additive_mask.to(torch.float32)

        with torch.no_grad():
            out_none = model(input_ids=input_ids).last_hidden_state
            out_ones = model(
                input_ids=input_ids,
                attention_mask=ones_mask,
            ).last_hidden_state
            out_int_sdpa = model(
                input_ids=input_ids,
                attention_mask=int_mask,
                output_attentions=False,
            ).last_hidden_state
            out_int_eager = model(
                input_ids=input_ids,
                attention_mask=int_mask,
                output_attentions=True,
            ).last_hidden_state
            out_bool = model(
                input_ids=input_ids,
                attention_mask=bool_mask,
            ).last_hidden_state
            out_add = model(
                input_ids=input_ids,
                attention_mask=additive_mask,
            ).last_hidden_state

        checks = [
            ("none_vs_ones", out_none, out_ones, atol, rtol),
            ("int_vs_bool", out_int_sdpa, out_bool, atol, rtol),
            ("int_vs_additive", out_int_sdpa, out_add, atol, rtol),
            # Eager softmax and SDPA can differ slightly on full-sized models.
            ("sdpa_vs_eager", out_int_sdpa, out_int_eager, 2e-5, rtol),
        ]
        for label, left, right, local_atol, local_rtol in checks:
            if not torch.allclose(left, right, atol=local_atol, rtol=local_rtol):
                max_diff = (left - right).abs().max().item()
                return False, f"{label} mismatch (max_abs_diff={max_diff:.6f})"

        print("mask_parity: OK")
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_end_to_end_pipeline(model_path: Path) -> Tuple[bool, str]:
    """Test end-to-end tokenization and encoding.

    :param Path model_path: Model directory path.
    :return tuple[bool, str]: Success flag and message.
    """
    try:
        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)

        texts = [
            "NeoBERT efficiency check.",
            "Models improve rapidly.",
            "The quick brown fox jumps.",
        ]
        shapes = []
        with torch.no_grad():
            for t in texts:
                inp = tok(t, return_tensors="pt", padding=True, truncation=True)
                out = model(**inp)
                if getattr(out, "last_hidden_state", None) is None:
                    return False, "pipeline: missing last_hidden_state"
                shapes.append(tuple(out.last_hidden_state.shape))
        print(f"pipeline: OK; shapes {shapes}")
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_cosine_similarity_sanity(model_path: Path) -> Tuple[bool, str]:
    """Sanity-check cosine similarities of mean-pooled sentence embeddings.

    :param Path model_path: Model directory path.
    :return tuple[bool, str]: Success flag and message.
    """
    try:
        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)

        sents = [
            "The cat sat on the mat.",
            "The feline rested on the rug.",
            "I love programming in Python.",
            "I enjoy coding with Python.",
            "The cat sat on the mat.",
            "Quantum physics explores subatomic particles.",
        ]
        embs = []
        with torch.no_grad():
            for s in sents:
                inp = tok(s, return_tensors="pt")
                out = model(**inp)
                mask = (
                    inp["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
                )
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp(min=1)
                embs.append(pooled.squeeze())

        def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Compute cosine similarity between two vectors.

            :param torch.Tensor a: First vector.
            :param torch.Tensor b: Second vector.
            :return torch.Tensor: Cosine similarity value.
            """
            return (a @ b) / (a.norm() * b.norm())

        sim1 = cos(embs[0], embs[1]).item()
        sim2 = cos(embs[2], embs[3]).item()
        diff = cos(embs[4], embs[5]).item()
        print(f"cosine: sim1 {sim1:.3f} sim2 {sim2:.3f} diff {diff:.3f}")

        # lenient thresholds for plain-MLM encoders
        if sim1 < 0.5:
            return False, f"low sim on similar pair1 ({sim1:.3f})"
        if sim2 < 0.5:
            return False, f"low sim on similar pair2 ({sim2:.3f})"
        if diff > 0.95:
            return False, f"high sim on different pair ({diff:.3f})"
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    """Run the export validation CLI."""
    p = argparse.ArgumentParser(
        description="Validate an exported HF model (concise output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("model_path", type=str, help="Path to exported model dir")
    p.add_argument(
        "--strict", action="store_true", help="treat Python warnings as errors"
    )
    p.add_argument(
        "-print", "--print_model_info", action="store_true", help="print model info"
    )
    args = p.parse_args()

    # Ensure warnings are visible; do not touch transformers logging except to make sure warnings show.
    if args.strict:
        warnings.simplefilter("error")
    else:
        warnings.simplefilter("default")

    # Make sure transformers warnings are not silenced by a too-high level elsewhere.
    tlog = logging.getLogger("transformers")
    if tlog.level > logging.WARNING:
        tlog.setLevel(logging.WARNING)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: model dir not found: {model_path}")
        sys.exit(1)

    print(f"validate: {model_path}")

    all_ok = True

    # 1) Files
    missing = validate_model_files(model_path)
    if missing:
        print("files: FAIL - missing " + ", ".join(missing))
        all_ok = False
    else:
        print("files: OK")

    # 2) Model
    ok, msg = test_model_loading(model_path)
    print(("AutoModel: OK" if ok else "AutoModel: FAIL") + ("" if ok else f" - {msg}"))
    all_ok &= ok

    # 3) Tokenizer
    ok, msg = test_tokenizer_loading(model_path)
    print(("tokenizer: OK" if ok else "tokenizer: FAIL") + ("" if ok else f" - {msg}"))
    all_ok &= ok

    # 4) Masked LM
    ok, msg = test_masked_lm_loading(model_path, verbose=args.print_model_info)
    print(
        ("AutoModelForMaskedLM: OK" if ok else "AutoModelForMaskedLM: FAIL")
        + ("" if ok else f" - {msg}")
    )
    all_ok &= ok

    # 5) Pipeline
    ok, msg = test_attention_mask_parity(model_path)
    print(
        ("mask parity: OK" if ok else "mask parity: FAIL") + ("" if ok else f" - {msg}")
    )
    all_ok &= ok

    # 6) Pipeline
    ok, msg = test_end_to_end_pipeline(model_path)
    print(("pipeline: OK" if ok else "pipeline: FAIL") + ("" if ok else f" - {msg}"))
    all_ok &= ok

    # 7) Cosine sanity
    ok, msg = test_cosine_similarity_sanity(model_path)
    print(("cosine: OK" if ok else "cosine: FAIL") + ("" if ok else f" - {msg}"))
    all_ok &= ok

    # Exit code only; no emojis, no fluff.
    if all_ok:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
