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


def create_hf_config(neobert_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NeoBERT config.yaml to HuggingFace config.json format."""
    model_config = neobert_config.get("model", {})

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
        "hidden_size": model_config.get("hidden_size", 768),
        "num_hidden_layers": model_config.get("num_hidden_layers", 12),
        "num_attention_heads": model_config.get("num_attention_heads", 12),
        "intermediate_size": model_config.get("intermediate_size", 3072),
        "vocab_size": model_config.get("vocab_size", 30522),
        "max_length": model_config.get("max_position_embeddings", 512),
        "embedding_init_range": 0.02,
        "decoder_init_range": 0.02,
        "norm_eps": model_config.get("layer_norm_eps", 1e-5),
        "pad_token_id": 0,
        "torch_dtype": "float32",
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

    # 1. Load and convert config
    print("Converting config...")
    neobert_config = load_config(config_path)
    hf_config = create_hf_config(neobert_config)

    # Save config.json
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    print("  Saved config.json")

    # 2. Load and map weights
    print("Converting and mapping model weights...")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model_config = neobert_config.get("model", {})
    mapped_state_dict = map_weights(state_dict, model_config)

    # Save as safetensors (preferred) and pytorch formats
    save_file(
        mapped_state_dict, output_dir / "model.safetensors", metadata={"format": "pt"}
    )
    print("  Saved model.safetensors")

    torch.save(mapped_state_dict, output_dir / "pytorch_model.bin")
    print("  Saved pytorch_model.bin")

    # 3. Load tokenizer, fix model_max_length, and save
    print("Loading and fixing tokenizer...")
    tokenizer_dir = checkpoint_path / "tokenizer"
    if tokenizer_dir.exists():
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        # Get max_position_embeddings from model config
        max_pos = model_config.get("max_position_embeddings", 4096)

        # Set the correct model_max_length
        tokenizer.model_max_length = max_pos

        # Save the tokenizer with corrected config
        tokenizer.save_pretrained(str(output_dir))
        print(f"  Saved tokenizer with model_max_length={max_pos}")

    # 4. Copy HF modeling files
    copy_hf_modeling_files(output_dir)

    # 5. Create README with HuggingFace YAML header
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

    # Convert full config to YAML string for details section
    config_yaml = yaml.dump(neobert_config, default_flow_style=False, sort_keys=False)

    readme_content = f"""{yaml_header}# NeoBERT Model

This is a NeoBERT model trained with [pszemraj/NeoBERT](https://github.com/pszemraj/NeoBERT) and exported to `transformers` format.

## Model Details
- **Architecture**: NeoBERT
- **Hidden Size**: {hf_config["hidden_size"]}
- **Layers**: {hf_config["num_hidden_layers"]}
- **Attention Heads**: {hf_config["num_attention_heads"]}
- **Vocab Size**: {hf_config["vocab_size"]}
- **Max Length**: {hf_config["max_length"]}

## Usage

> [!IMPORTANT]
> Ensure you update `repo_id` to your actual HuggingFace repo ID or local path.

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
