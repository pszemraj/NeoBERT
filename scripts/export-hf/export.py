#!/usr/bin/env python3
"""Export NeoBERT pretraining checkpoint to HuggingFace format.

This script converts a NeoBERT checkpoint from the training format
(state_dict.pt + config.yaml) to HuggingFace format with all necessary
files for loading with transformers library.

Usage:
    python scripts/export-hf/export.py outputs/neobert_100m_100k/model_checkpoints/100000

The script will create an hf/ directory in the parent folder with the exported model.
"""

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
import yaml
from safetensors.torch import save_file
from transformers import AutoTokenizer


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse config.yaml.

    :param Path config_path: Path to the config file.
    :return dict[str, Any]: Parsed configuration mapping.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_torch_dtype_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """Infer the torch dtype from the state dict weights.

    :param dict[str, torch.Tensor] state_dict: Model state dict.
    :return str: Torch dtype string (e.g. "float32").
    """
    if not state_dict:
        raise ValueError("State dict is empty, cannot infer dtype")

    # Get the dtype of the first weight tensor
    first_weight = next(iter(state_dict.values()))
    if not isinstance(first_weight, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(first_weight)}")

    dtype = first_weight.dtype
    if dtype == torch.float32:
        return "float32"
    elif dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float64:
        return "float64"
    else:
        # For any other dtype, fall back to the string representation
        dtype_str = str(dtype).replace("torch.", "")
        print(f"Warning: Found unexpected dtype {dtype}, using '{dtype_str}'")
        return dtype_str


def load_tokenizer_info(tokenizer_info_path: Path) -> Optional[Dict[str, Any]]:
    """Load tokenizer_info.json if present.

    :param Path tokenizer_info_path: Path to tokenizer_info.json.
    :return dict[str, Any] | None: Parsed tokenizer info or None if missing.
    """
    if not tokenizer_info_path.exists():
        return None
    with open(tokenizer_info_path, "r") as f:
        return json.load(f)


def maybe_alias_decoder_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Ensure decoder weights are available under the expected key prefix.

    :param dict[str, torch.Tensor] state_dict: Loaded state dict.
    :return dict[str, torch.Tensor]: State dict with model.decoder.* aliases if needed.
    """
    if "model.decoder.weight" not in state_dict and "decoder.weight" in state_dict:
        state_dict["model.decoder.weight"] = state_dict["decoder.weight"]
    if "model.decoder.bias" not in state_dict and "decoder.bias" in state_dict:
        state_dict["model.decoder.bias"] = state_dict["decoder.bias"]
    return state_dict


def _swiglu_intermediate_size(intermediate_size: int, multiple_of: int = 8) -> int:
    """Compute the reduced SwiGLU hidden size used in training.

    :param int intermediate_size: Config intermediate size.
    :param int multiple_of: Alignment multiple.
    :return int: Effective SwiGLU hidden size.
    """
    hidden = int(2 * intermediate_size / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def _check_shape(
    state_dict: Dict[str, torch.Tensor], key: str, expected: Tuple[int, ...]
) -> None:
    """Verify a tensor exists and matches the expected shape."""
    if key not in state_dict:
        raise ValueError(f"Missing required weight: {key}")
    actual = tuple(state_dict[key].shape)
    if actual != expected:
        raise ValueError(f"Shape mismatch for {key}: expected {expected}, got {actual}")


def validate_state_dict_layout(
    state_dict: Dict[str, torch.Tensor], model_config: Dict[str, Any]
) -> None:
    """Validate checkpoint tensors against the training config."""
    state_dict = maybe_alias_decoder_weights(state_dict)
    hidden_size = model_config["hidden_size"]
    num_layers = model_config["num_hidden_layers"]
    vocab_size = model_config["vocab_size"]
    hidden_act = str(model_config.get("hidden_act", "swiglu")).lower()

    if hidden_act not in {"swiglu", "gelu"}:
        raise ValueError(
            f"Unsupported hidden_act '{hidden_act}'. Supported: swiglu, gelu."
        )

    _check_shape(state_dict, "model.encoder.weight", (vocab_size, hidden_size))
    _check_shape(state_dict, "model.decoder.weight", (vocab_size, hidden_size))
    if "model.decoder.bias" in state_dict:
        _check_shape(state_dict, "model.decoder.bias", (vocab_size,))

    if any(".ffn.w12." in key for key in state_dict.keys()):
        raise ValueError(
            "Packed SwiGLU weights (ffn.w12) found. Export expects unpacked "
            "w1/w2/w3 weights from training."
        )

    for layer_idx in range(num_layers):
        prefix = f"model.transformer_encoder.{layer_idx}"
        _check_shape(state_dict, f"{prefix}.qkv.weight", (hidden_size * 3, hidden_size))
        _check_shape(state_dict, f"{prefix}.wo.weight", (hidden_size, hidden_size))

        if hidden_act == "swiglu":
            mlp_hidden = _swiglu_intermediate_size(model_config["intermediate_size"])
            _check_shape(
                state_dict, f"{prefix}.ffn.w1.weight", (mlp_hidden, hidden_size)
            )
            _check_shape(
                state_dict, f"{prefix}.ffn.w2.weight", (mlp_hidden, hidden_size)
            )
            _check_shape(
                state_dict, f"{prefix}.ffn.w3.weight", (hidden_size, mlp_hidden)
            )
        else:
            _check_shape(
                state_dict,
                f"{prefix}.ffn.0.weight",
                (model_config["intermediate_size"], hidden_size),
            )
            _check_shape(
                state_dict,
                f"{prefix}.ffn.2.weight",
                (hidden_size, model_config["intermediate_size"]),
            )


def run_forward_sanity_check(
    hf_config: Dict[str, Any], mapped_state_dict: Dict[str, torch.Tensor]
) -> None:
    """Instantiate the HF model and run a lightweight forward pass."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from neobert.huggingface.modeling_neobert import NeoBERTConfig, NeoBERTLMHead

    model_config = NeoBERTConfig(**hf_config)
    model = NeoBERTLMHead(model_config)
    model.load_state_dict(mapped_state_dict, strict=True)
    model.eval()

    seq_len = max(1, min(8, int(hf_config["max_position_embeddings"])))
    batch_size = 2
    vocab_size = int(hf_config["vocab_size"])
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    if outputs.logits.shape != (batch_size, seq_len, vocab_size):
        raise ValueError(
            f"Sanity check failed: expected logits shape "
            f"{(batch_size, seq_len, vocab_size)}, got {outputs.logits.shape}"
        )
    if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
        raise ValueError("Sanity check failed: logits contain NaNs or Infs.")


def validate_required_config_fields(model_config: Dict[str, Any]) -> None:
    """Validate that all required config fields are present."""
    required_fields = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "norm_eps",
        "pad_token_id",
        # Architecture-affecting fields that must be explicit for correct export.
        "rope",
        "rms_norm",
        "hidden_act",
    ]

    missing_fields = []
    for field in required_fields:
        if field not in model_config:
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(
            f"Missing required config fields in model config: {missing_fields}. "
            "The training config must contain all required fields for HF export."
        )


def create_hf_config(
    neobert_config: Dict[str, Any], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    """Convert NeoBERT config.yaml to Hugging Face config.json format.

    :param dict[str, Any] neobert_config: Loaded NeoBERT config mapping.
    :param dict[str, torch.Tensor] state_dict: Model state dict.
    :return dict[str, Any]: Hugging Face config mapping.
    """
    model_config = neobert_config.get("model", {})

    # Validate that we have all required fields
    validate_required_config_fields(model_config)

    # Infer dtype from actual weights
    torch_dtype = get_torch_dtype_from_state_dict(state_dict)

    hidden_act = str(model_config.get("hidden_act", "swiglu")).lower()
    if hidden_act not in {"swiglu", "gelu"}:
        raise ValueError(
            f"Unsupported hidden_act '{hidden_act}' for HF export. Supported: swiglu, gelu."
        )

    if model_config.get("ngpt", False):
        raise ValueError("ngpt/NormNeoBERT is not supported by the HF export path.")

    # Map our config to HF format - using the original HF model structure
    hf_config = {
        "architectures": ["NeoBERTLMHead"],
        "model_type": "neobert",
        "auto_map": {
            "AutoConfig": "model.NeoBERTConfig",
            "AutoModel": "model.NeoBERT",
            "AutoModelForMaskedLM": "model.NeoBERTLMHead",
            "AutoModelForSequenceClassification": "model.NeoBERTForSequenceClassification",
        },
        "hidden_size": model_config["hidden_size"],
        "num_hidden_layers": model_config["num_hidden_layers"],
        "num_attention_heads": model_config["num_attention_heads"],
        "intermediate_size": model_config["intermediate_size"],
        "vocab_size": model_config["vocab_size"],
        "max_length": model_config["max_position_embeddings"],
        "max_position_embeddings": model_config["max_position_embeddings"],
        "embedding_init_range": model_config.get("embedding_init_range", 0.02),
        "decoder_init_range": model_config.get("decoder_init_range", 0.02),
        "norm_eps": model_config["norm_eps"],
        "rms_norm": model_config.get("rms_norm", True),
        "rope": model_config.get("rope", True),
        "hidden_act": hidden_act,
        "dropout": model_config.get("dropout", model_config.get("dropout_prob", 0.0)),
        "flash_attention": model_config.get("flash_attention", False),
        "pad_token_id": model_config["pad_token_id"],
        "torch_dtype": torch_dtype,
        "transformers_version": transformers.__version__,
    }

    return hf_config


def map_weights(
    state_dict: Dict[str, torch.Tensor], model_config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Map weights from our training format to HF model format.

    Our training format has "model." prefix and uses SwiGLU with unpacked w1/w2/w3.
    HF format mirrors the same unpacked layout.

    Export structure for HF compatibility:
    - Base model weights with "model." prefix for NeoBERTLMHead
    - Decoder weights at top level for LM head
    :param dict[str, torch.Tensor] state_dict: Training state dict.
    :param dict[str, Any] model_config: Model config mapping.
    :return dict[str, torch.Tensor]: Remapped state dict.
    """
    mapped = {}

    for key, value in state_dict.items():
        if key.startswith("model.decoder."):
            # Decoder weights go to top level
            new_key = key.replace("model.", "")
            mapped[new_key] = value
        elif key.startswith("model."):
            # Keep model. prefix for NeoBERTLMHead compatibility
            mapped[key] = value
        else:
            # Any other weights map directly (shouldn't be any)
            mapped[key] = value

    return mapped


def validate_tokenizer_special_tokens(tokenizer_dir: Path) -> None:
    """Validate that the tokenizer has all required special tokens."""
    special_tokens_path = tokenizer_dir / "special_tokens_map.json"
    if not special_tokens_path.exists():
        raise FileNotFoundError(
            f"Tokenizer special_tokens_map.json not found at {special_tokens_path}"
        )

    with open(special_tokens_path, "r") as f:
        special_tokens = json.load(f)

    required_tokens = ["cls_token", "sep_token", "pad_token", "mask_token", "unk_token"]
    missing_tokens = []

    for token in required_tokens:
        if token not in special_tokens:
            missing_tokens.append(token)

    if missing_tokens:
        raise ValueError(
            f"Tokenizer missing required special tokens: {missing_tokens}. "
            "All special tokens must be defined for proper HF compatibility."
        )


def _rewrite_export_model_imports(model_path: Path) -> None:
    """Rewrite local imports in exported modeling code for HF dynamic modules.

    :param Path model_path: Path to the exported model.py file.
    """
    contents = model_path.read_text()
    updated = contents.replace(
        "from ..modeling_utils import swiglu_intermediate_size",
        "from .modeling_utils import swiglu_intermediate_size",
    )
    if updated != contents:
        model_path.write_text(updated)


def copy_hf_modeling_files(target_dir: Path) -> None:
    """Copy the Hugging Face modeling files from src/neobert/huggingface/.

    :param Path target_dir: Destination directory for modeling files.
    """
    print("Copying HuggingFace modeling files...")

    # Go up to repo root then down to src/neobert/huggingface
    src_dir = Path(__file__).parent.parent.parent / "src" / "neobert" / "huggingface"

    if not src_dir.exists():
        raise ValueError(f"HuggingFace model files not found at {src_dir}")

    # Copy the required files
    files_to_copy = ["modeling_neobert.py", "rotary.py"]
    for filename in files_to_copy:
        src_file = src_dir / filename
        dst_name = "model.py" if filename == "modeling_neobert.py" else filename
        if not src_file.exists():
            raise ValueError(f"Required file {src_file} not found")
        shutil.copy(src_file, target_dir / dst_name)
        print(f"  Copied {filename} -> {dst_name}")

    modeling_utils_src = src_dir.parent / "modeling_utils.py"
    if modeling_utils_src.exists():
        shutil.copy(modeling_utils_src, target_dir / "modeling_utils.py")
        print("  Copied modeling_utils.py -> modeling_utils.py")

    model_path = target_dir / "model.py"
    if model_path.exists():
        _rewrite_export_model_imports(model_path)


def export_checkpoint(checkpoint_path: Path, output_dir: Path | None = None) -> Path:
    """Export a NeoBERT checkpoint to HuggingFace format.

    :param Path checkpoint_path: Checkpoint directory with state_dict.pt and config.yaml.
    :param Path | None output_dir: Optional output directory.
    :return Path: Output directory containing exported files.
    """
    checkpoint_path = Path(checkpoint_path).resolve()

    # Validate input
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    state_dict_path = checkpoint_path / "state_dict.pt"
    config_path = checkpoint_path / "config.yaml"

    if not state_dict_path.exists():
        raise FileNotFoundError(f"state_dict.pt not found in {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {checkpoint_path}")

    # Determine output directory
    if output_dir is None:
        checkpoint_name = checkpoint_path.name  # e.g., "100000"
        parent_name = checkpoint_path.parent.parent.name  # e.g., "neobert_100m_100k"
        output_dir = (
            checkpoint_path.parent.parent / "hf" / f"{parent_name}_{checkpoint_name}"
        )
    else:
        output_dir = Path(output_dir)

    # 1. Load state dict first to get dtype info
    print("Loading model weights...")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    if not state_dict:
        raise ValueError(f"Loaded state dict is empty from {state_dict_path}")
    print(
        f"  Loaded {len(state_dict)} weight tensors with dtype: {next(iter(state_dict.values())).dtype}"
    )

    # 2. Load and validate config
    print("Validating config and weights...")
    neobert_config = load_config(config_path)
    model_config = neobert_config.get("model", {})
    if not model_config:
        raise ValueError("Model config section is empty or missing from config.yaml")

    validate_required_config_fields(model_config)

    # Validate vocab_size and pad_token_id using checkpoint metadata.
    tokenizer_info = load_tokenizer_info(checkpoint_path / "tokenizer_info.json")
    if tokenizer_info is None:
        print(
            "  Warning: tokenizer_info.json not found; validating vocab_size against weights only."
        )
    else:
        info_vocab = tokenizer_info.get("vocab_size")
        if info_vocab is not None and info_vocab != model_config["vocab_size"]:
            print(
                "  Warning: tokenizer_info vocab_size "
                f"({info_vocab}) does not match config vocab_size "
                f"({model_config['vocab_size']}); continuing."
            )
        info_pad = tokenizer_info.get("pad_token_id")
        if info_pad is not None and info_pad != model_config["pad_token_id"]:
            raise ValueError(
                f"tokenizer_info pad_token_id ({info_pad}) does not match config "
                f"pad_token_id ({model_config['pad_token_id']})."
            )

    embedding_vocab = state_dict["model.encoder.weight"].shape[0]
    if embedding_vocab != model_config["vocab_size"]:
        raise ValueError(
            f"Config vocab_size ({model_config['vocab_size']}) does not match "
            f"embedding weight shape ({embedding_vocab})."
        )

    validate_state_dict_layout(state_dict, model_config)

    hf_config = create_hf_config(neobert_config, state_dict)

    # 3. Map weights
    print("Converting and mapping model weights...")
    mapped_state_dict = map_weights(state_dict, model_config)

    # 3a. Sanity-check that the HF model loads and runs.
    print("Running HF forward pass sanity check...")
    run_forward_sanity_check(hf_config, mapped_state_dict)
    print("  Sanity check passed")

    # Save config.json
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting checkpoint to: {output_dir}")
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    print("  Saved config.json")

    # Save as safetensors (preferred) and pytorch formats
    save_file(
        mapped_state_dict, output_dir / "model.safetensors", metadata={"format": "pt"}
    )
    print("  Saved model.safetensors")

    torch.save(mapped_state_dict, output_dir / "pytorch_model.bin")
    print("  Saved pytorch_model.bin")

    # 4. Load tokenizer, validate special tokens, fix model_max_length, and save
    print("Loading and validating tokenizer...")
    tokenizer_dir = checkpoint_path / "tokenizer"
    if tokenizer_dir.exists():
        # Validate tokenizer has all required special tokens
        validate_tokenizer_special_tokens(tokenizer_dir)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        # Get max_position_embeddings from model config
        max_pos = model_config["max_position_embeddings"]

        # Set the correct model_max_length
        tokenizer.model_max_length = max_pos

        # Save the tokenizer with corrected config
        tokenizer.save_pretrained(str(output_dir))
        print(f"  Saved tokenizer with model_max_length={max_pos}")
    else:
        raise FileNotFoundError(
            f"Tokenizer directory not found at {tokenizer_dir}. "
            "Tokenizer is required for HF export."
        )

    # 5. Copy HF modeling files
    copy_hf_modeling_files(output_dir)

    # 6. Create README with HuggingFace YAML header
    # Get dataset info from config
    dataset_name = neobert_config.get("dataset", {}).get("name", "")
    if not dataset_name and neobert_config.get("dataset", {}).get("path"):
        # If using local path, try to extract dataset name from path
        dataset_path = neobert_config.get("dataset", {}).get("path", "")
        dataset_name = dataset_path.split("/")[-1] if dataset_path else ""

    # Build HF YAML header
    yaml_header = "---\n"
    yaml_header += "library_name: transformers\n"
    yaml_header += "license: mit\n"
    if dataset_name:
        yaml_header += f"datasets:\n- {dataset_name}\n"
    yaml_header += "language:\n- en\n"
    yaml_header += "---\n\n"

    # Get repo_id from output directory name
    repo_id = output_dir.name

    # Check if tokenizer needs special handling for mask tokens
    # Load the saved tokenizer to get actual special tokens
    from transformers import AutoTokenizer as HFAutoTokenizer

    saved_tokenizer = HFAutoTokenizer.from_pretrained(str(output_dir))
    actual_mask_token = saved_tokenizer.mask_token
    mask_display = (
        "[MASK]" if actual_mask_token in ["[MASK]", "[mask]"] else actual_mask_token
    )

    # Check if tokenizer uses Metaspace/SentencePiece (has ▁ token)
    try:
        space_token_id = saved_tokenizer.convert_tokens_to_ids("▁")
        has_metaspace = (
            space_token_id is not None
            and space_token_id != saved_tokenizer.unk_token_id
        )
    except Exception:
        has_metaspace = False

    # Convert full config to YAML string for details section
    config_yaml = yaml.dump(neobert_config, default_flow_style=False, sort_keys=False)

    # Generate MLM example based on actual tokenizer
    if has_metaspace:
        # Include space token handling for Metaspace tokenizers
        mlm_example = f'''```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

repo_id = "{repo_id}"  # Update this to your HF repo ID
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(repo_id, trust_remote_code=True)

# Example: Fill in masked tokens
text = "NeoBERT is the most {mask_display} model of its kind!"

# Replace display mask with actual mask token if different
text = text.replace("{mask_display}", tokenizer.mask_token)

inputs = tokenizer(text, return_tensors="pt")

# Handle Metaspace tokenizer quirk: remove extra space tokens before mask
# Get the space token ID dynamically
try:
    space_token_id = tokenizer.convert_tokens_to_ids("▁")
except:
    space_token_id = None

if space_token_id is not None:
    input_ids = inputs["input_ids"][0].tolist()
    cleaned_ids = []
    for i, token_id in enumerate(input_ids):
        # Skip space token if it's immediately before mask token
        if token_id == space_token_id and i < len(input_ids) - 1 and input_ids[i + 1] == tokenizer.mask_token_id:
            continue
        cleaned_ids.append(token_id)

    if len(cleaned_ids) != len(input_ids):
        inputs["input_ids"] = torch.tensor([cleaned_ids])
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if len(mask_positions[0]) == 0:
        raise ValueError("No mask token found in input")
    mask_pos = mask_positions[1][0]
    predictions = outputs.logits[0, mask_pos].topk(5)

# Display top predictions
for idx, score in zip(predictions.indices, predictions.values):
    token = tokenizer.decode([idx])
    print(f"{{token}}: {{score:.2f}}")
```'''
    else:
        # Simpler version for non-Metaspace tokenizers
        mlm_example = f'''```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

repo_id = "{repo_id}"  # Update this to your HF repo ID
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(repo_id, trust_remote_code=True)

# Example: Fill in masked tokens
text = "NeoBERT is the most {mask_display} model of its kind!"

# Replace display mask with actual mask token if different
text = text.replace("{mask_display}", tokenizer.mask_token)

inputs = tokenizer(text, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if len(mask_positions[0]) == 0:
        raise ValueError("No mask token found in input")
    mask_pos = mask_positions[1][0]
    predictions = outputs.logits[0, mask_pos].topk(5)

# Display top predictions
for idx, score in zip(predictions.indices, predictions.values):
    token = tokenizer.decode([idx])
    print(f"{{token}}: {{score:.2f}}")
```'''

    readme_content = f"""{yaml_header}# NeoBERT Model

This is a NeoBERT model trained with [pszemraj/NeoBERT](https://github.com/pszemraj/NeoBERT) and exported to `transformers` format.

## Model Details
- **Architecture**: NeoBERT
- **Hidden Size**: {hf_config["hidden_size"]}
- **Layers**: {hf_config["num_hidden_layers"]}
- **Attention Heads**: {hf_config["num_attention_heads"]}
- **Vocab Size**: {hf_config["vocab_size"]}
- **Max Length**: {hf_config["max_length"]}
- **Dtype**: {hf_config["torch_dtype"]}

## Usage

> [!IMPORTANT]
> Ensure you update `repo_id` to your actual HuggingFace repo ID or local path.

### For Masked Language Modeling (Fill-Mask)

{mlm_example}

### For Embeddings / Feature Extraction

```python
from transformers import AutoModel, AutoTokenizer

repo_id = "{repo_id}"  # Update this to your HF repo ID
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Example: Generate embeddings
text = "NeoBERT is an efficient transformer model!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get CLS token embedding
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(f"Embedding shape: {{cls_embedding.shape}}")
```

## Training Configuration

<details>
  <summary><strong>Full Config</strong> (click to expand)</summary>

Full training config:

```yaml
{config_yaml}
```

</details>
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    print("  Created README.md")

    print(f"\n✅ exported checkpoint to:\n\t{output_dir}")

    return output_dir


def main() -> None:
    """Run the Hugging Face export CLI."""
    parser = argparse.ArgumentParser(
        description="Export NeoBERT checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
            # Export checkpoint 100000 from neobert_100m_100k
            python %(prog)s outputs/neobert_100m_100k/model_checkpoints/100000

            # Export to specific directory
            python %(prog)s outputs/neobert_100m_100k/model_checkpoints/100000 --output my_model
        """
        ),
    )

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint directory containing state_dict.pt and config.yaml",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for HF model (default: creates hf/{model_name} in parent dir)",
    )

    args = parser.parse_args()

    try:
        export_checkpoint(
            checkpoint_path=Path(args.checkpoint_path),
            output_dir=Path(args.output) if args.output else None,
        )
    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
