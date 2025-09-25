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
from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


def validate_model_files(model_path: Path) -> List[str]:
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
    return missing


def _from_pretrained_with_info(
    model_cls, model_path: Path, verbose: bool = False
) -> Tuple[object, Optional[dict]]:
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
        issues = _format_loading_issues(loading_info)
        if issues:
            return False, "load issues: " + issues

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


def test_end_to_end_pipeline(model_path: Path) -> Tuple[bool, str]:
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
                embs.append(out.last_hidden_state[:, 0, :].squeeze())

        def cos(a, b):
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


def main():
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
    ok, msg = test_end_to_end_pipeline(model_path)
    print(("pipeline: OK" if ok else "pipeline: FAIL") + ("" if ok else f" - {msg}"))
    all_ok &= ok

    # 6) Cosine sanity
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
