#!/usr/bin/env python3
"""Export NeoBERT pretraining checkpoint to HuggingFace format.

This script converts a NeoBERT checkpoint from the training format
(state_dict.pt + config.yaml) to HuggingFace format with all necessary
files for loading with transformers library.

Usage:
    python scripts/export_to_huggingface.py outputs/neobert_100m_100k/model_checkpoints/100000
    
The script will create an hf/ directory in the parent folder with the exported model.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from safetensors.torch import save_file


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
            "AutoModelForSequenceClassification": "model.NeoBERTForSequenceClassification"
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
        "transformers_version": "4.48.2",
    }
    
    # Add training info if available
    if "trainer" in neobert_config:
        trainer = neobert_config["trainer"]
        hf_config["training_info"] = {
            "num_train_epochs": trainer.get("num_train_epochs"),
            "max_steps": trainer.get("max_steps"),
            "batch_size": trainer.get("per_device_train_batch_size"),
            "learning_rate": neobert_config.get("optimizer", {}).get("lr"),
        }
    
    return hf_config


def map_weights(state_dict: Dict[str, torch.Tensor], model_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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
    """Copy the HuggingFace modeling files."""
    print("Copying HuggingFace modeling files...")
    
    # First check if we have the HF modeling files in src/neobert/huggingface/
    src_dir = Path(__file__).parent.parent / "src" / "neobert" / "huggingface"
    
    if src_dir.exists():
        # Use our HF-compatible modeling files
        files_to_copy = ["modeling_neobert.py", "rotary.py"]
        for filename in files_to_copy:
            src_file = src_dir / filename
            dst_name = "model.py" if filename == "modeling_neobert.py" else filename
            if src_file.exists():
                shutil.copy(src_file, target_dir / dst_name)
                print(f"  Copied {filename} -> {dst_name}")
    else:
        # Fallback: copy from original HF repo
        orig_dir = Path(__file__).parent.parent / "outputs" / "original-NeoBERT-hf"
        if orig_dir.exists():
            for filename in ["model.py", "rotary.py"]:
                src_file = orig_dir / filename
                if src_file.exists():
                    shutil.copy(src_file, target_dir / filename)
                    print(f"  Copied {filename} from original HF repo")


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
        output_dir = checkpoint_path.parent.parent / "hf" / f"{parent_name}_{checkpoint_name}"
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
    save_file(mapped_state_dict, output_dir / "model.safetensors", metadata={"format": "pt"})
    print("  Saved model.safetensors")
    
    torch.save(mapped_state_dict, output_dir / "pytorch_model.bin")
    print("  Saved pytorch_model.bin")
    
    # 3. Copy tokenizer files
    print("Copying tokenizer files...")
    tokenizer_dir = checkpoint_path / "tokenizer"
    if tokenizer_dir.exists():
        for file in tokenizer_dir.glob("*.json"):
            shutil.copy(file, output_dir / file.name)
            print(f"  Copied {file.name}")
    
    # Also copy vocab.txt if it exists in original HF repo
    orig_vocab = Path(__file__).parent.parent / "outputs" / "original-NeoBERT-hf" / "vocab.txt"
    if orig_vocab.exists():
        shutil.copy(orig_vocab, output_dir / "vocab.txt")
        print("  Copied vocab.txt")
    
    # 4. Copy HF modeling files
    copy_hf_modeling_files(output_dir)
    
    # 5. Create README
    readme_content = f"""# NeoBERT Model

This is a NeoBERT model exported from training checkpoint.

## Model Details
- **Architecture**: NeoBERT
- **Hidden Size**: {hf_config['hidden_size']}
- **Layers**: {hf_config['num_hidden_layers']}
- **Attention Heads**: {hf_config['num_attention_heads']}
- **Vocab Size**: {hf_config['vocab_size']}
- **Max Length**: {hf_config['max_length']}

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{output_dir.name}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{output_dir.name}")

# For masked language modeling
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("{output_dir.name}", trust_remote_code=True)

# Example usage
text = "NeoBERT is an efficient model!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
```

## Training Information
{json.dumps(hf_config.get('training_info', {}), indent=2) if 'training_info' in hf_config else 'Not available'}
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    print("  Created README.md")
    
    print(f"\n✅ Successfully exported checkpoint to: {output_dir}")
    print(f"\nTo test the exported model:")
    print(f"  python -c \"from transformers import AutoModel; model = AutoModel.from_pretrained('{output_dir}', trust_remote_code=True); print(model)\"")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export NeoBERT checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export checkpoint 100000 from neobert_100m_100k
  python scripts/export_to_huggingface.py outputs/neobert_100m_100k/model_checkpoints/100000
  
  # Export to specific directory
  python scripts/export_to_huggingface.py outputs/neobert_100m_100k/model_checkpoints/100000 --output my_model
        """
    )
    
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint directory containing state_dict.pt and config.yaml"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for HF model (default: creates hf/{model_name} in parent dir)"
    )
    
    args = parser.parse_args()
    
    try:
        output_dir = export_checkpoint(
            checkpoint_path=Path(args.checkpoint_path),
            output_dir=Path(args.output) if args.output else None
        )
    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise


if __name__ == "__main__":
    main()