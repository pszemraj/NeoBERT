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
import textwrap
from pathlib import Path
from typing import Any, Dict

import torch
import transformers
import yaml
from safetensors.torch import save_file
from transformers import AutoTokenizer


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse config.yaml."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_torch_dtype_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """Infer the torch dtype from the state dict weights."""
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
    """Convert NeoBERT config.yaml to HuggingFace config.json format."""
    model_config = neobert_config.get("model", {})

    # Validate that we have all required fields
    validate_required_config_fields(model_config)

    # Infer dtype from actual weights
    torch_dtype = get_torch_dtype_from_state_dict(state_dict)

    # Infer actual vocab_size from embedding weights if available
    if "model.encoder.weight" in state_dict:
        actual_vocab_size = state_dict["model.encoder.weight"].shape[0]
        if actual_vocab_size != model_config.get("vocab_size"):
            print(
                f"  Note: Using actual vocab_size from weights ({actual_vocab_size}) instead of config ({model_config.get('vocab_size')})"
            )
            model_config["vocab_size"] = actual_vocab_size

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
        "embedding_init_range": model_config.get("embedding_init_range", 0.02),
        "decoder_init_range": model_config.get("decoder_init_range", 0.02),
        "norm_eps": model_config["norm_eps"],
        "pad_token_id": model_config["pad_token_id"],
        "torch_dtype": torch_dtype,
        "transformers_version": transformers.__version__,
    }

    return hf_config


def map_weights(
    state_dict: Dict[str, torch.Tensor], model_config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Map weights from our training format to HF model format.

    Our training format has "model." prefix and uses SwiGLU with w12 (concatenated w1+w2).
    HF format expects w12 (xformers.SwiGLU expects the concatenated format).

    Export structure for HF compatibility:
    - Base model weights with "model." prefix for NeoBERTLMHead
    - Decoder weights at top level for LM head
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


def copy_hf_modeling_files(target_dir: Path):
    """Copy the HuggingFace modeling files from src/neobert/huggingface/."""
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


def export_checkpoint(checkpoint_path: Path, output_dir: Path = None):
    """Export a NeoBERT checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to checkpoint directory containing state_dict.pt and config.yaml
        output_dir: Optional output directory. If None, creates hf/{checkpoint_name} in parent dir
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

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting checkpoint to: {output_dir}")

    # 1. Load state dict first to get dtype info
    print("Loading model weights...")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    if not state_dict:
        raise ValueError(f"Loaded state dict is empty from {state_dict_path}")
    print(
        f"  Loaded {len(state_dict)} weight tensors with dtype: {next(iter(state_dict.values())).dtype}"
    )

    # 2. Load and convert config
    print("Converting config...")
    neobert_config = load_config(config_path)
    model_config = neobert_config.get("model", {})
    if not model_config:
        raise ValueError("Model config section is empty or missing from config.yaml")

    hf_config = create_hf_config(neobert_config, state_dict)

    # Save config.json
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    print("  Saved config.json")

    # 3. Map weights
    print("Converting and mapping model weights...")
    mapped_state_dict = map_weights(state_dict, model_config)

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
    mask_display = "[MASK]" if actual_mask_token in ["[MASK]", "[mask]"] else actual_mask_token

    # Check if tokenizer uses Metaspace/SentencePiece (has ▁ token)
    try:
        space_token_id = saved_tokenizer.convert_tokens_to_ids("▁")
        has_metaspace = space_token_id is not None and space_token_id != saved_tokenizer.unk_token_id
    except:
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


def main():
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
