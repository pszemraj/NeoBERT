#!/usr/bin/env python3
"""Test script for validating exported HuggingFace models.

This script performs comprehensive validation of models exported via 
scripts/export_to_huggingface.py to ensure they load correctly and
don't have any initialization warnings or missing weights.

Usage:
    python tests/test_huggingface_export.py path/to/exported/model
    
Example:
    python tests/test_huggingface_export.py outputs/neobert_100m_100k/hf/neobert_100m_100k_100000
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from transformers import logging as transformers_logging


class ExportValidationError(Exception):
    """Raised when exported model validation fails."""
    pass


class InitializationWarningCapture:
    """Context manager to capture initialization warnings."""
    
    def __init__(self):
        self.warnings = []
        self.original_showwarning = None
        
    def __enter__(self):
        self.original_showwarning = warnings.showwarning
        
        def capture_warning(message, category, filename, lineno, file=None, line=None):
            warning_str = str(message).lower()
            # Check for critical initialization warnings
            if any(phrase in warning_str for phrase in [
                "not initialized",
                "random initialization", 
                "randomly initialized",
                "newly initialized",
                "weights of",
                "were not initialized from"
            ]):
                self.warnings.append(str(message))
            # Still show the warning normally
            if self.original_showwarning:
                self.original_showwarning(message, category, filename, lineno, file, line)
        
        warnings.showwarning = capture_warning
        return self
    
    def __exit__(self, *args):
        warnings.showwarning = self.original_showwarning


def validate_model_files(model_path: Path) -> List[str]:
    """Validate that all required files exist."""
    required_files = [
        "config.json",
        "model.py",
        "rotary.py",
    ]
    
    # At least one of these weight formats should exist
    weight_files = [
        "model.safetensors",
        "pytorch_model.bin"
    ]
    
    missing_files = []
    
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    # Check that at least one weight file exists
    if not any((model_path / f).exists() for f in weight_files):
        missing_files.append("model weights (model.safetensors or pytorch_model.bin)")
    
    return missing_files


def test_model_loading(model_path: Path) -> Tuple[bool, str]:
    """Test if model loads without initialization warnings."""
    
    try:
        # Suppress non-critical warnings but capture initialization warnings
        transformers_logging.set_verbosity_error()
        
        with InitializationWarningCapture() as warning_capture:
            # Test AutoModel loading
            print("  Testing AutoModel.from_pretrained...")
            model = AutoModel.from_pretrained(
                str(model_path), 
                trust_remote_code=True
            )
            
            # Check for captured warnings
            if warning_capture.warnings:
                error_msg = "Model has initialization warnings:\n"
                for warning in warning_capture.warnings:
                    error_msg += f"    - {warning}\n"
                return False, error_msg
        
        # Test that model produces output with tokenizer
        print("  Testing model forward pass...")
        # Load tokenizer for proper testing
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        test_text = "Testing model forward pass."
        inputs = tokenizer(test_text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        
        if output.last_hidden_state is None:
            return False, "Model forward pass returned None"
        
        print(f"    ✓ Output shape: {output.last_hidden_state.shape}")
        
        return True, "Model loaded successfully"
        
    except Exception as e:
        return False, f"Failed to load model: {e}"


def test_tokenizer_loading(model_path: Path) -> Tuple[bool, str]:
    """Test if tokenizer loads correctly."""
    
    try:
        print("  Testing AutoTokenizer.from_pretrained...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        # Test tokenization
        test_text = "NeoBERT is an efficient transformer model."
        tokens = tokenizer(test_text, return_tensors="pt")
        
        if tokens.input_ids is None:
            return False, "Tokenizer produced None output"
        
        print(f"    ✓ Tokenized {len(tokens.input_ids[0])} tokens")
        
        return True, "Tokenizer loaded successfully"
        
    except Exception as e:
        return False, f"Failed to load tokenizer: {e}"


def test_masked_lm_loading(model_path: Path) -> Tuple[bool, str]:
    """Test if model loads as masked LM model."""
    
    try:
        print("  Testing AutoModelForMaskedLM.from_pretrained...")
        
        with InitializationWarningCapture() as warning_capture:
            model = AutoModelForMaskedLM.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            if warning_capture.warnings:
                error_msg = "MaskedLM model has initialization warnings:\n"
                for warning in warning_capture.warnings:
                    error_msg += f"    - {warning}\n"
                return False, error_msg
        
        # Test forward pass with tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        test_text = "Testing [MASK] model."
        inputs = tokenizer(test_text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        
        if output.logits is None:
            return False, "MaskedLM forward pass returned None"
        
        print(f"    ✓ Logits shape: {output.logits.shape}")
        
        return True, "MaskedLM model loaded successfully"
        
    except Exception as e:
        return False, f"Failed to load MaskedLM model: {e}"


def test_end_to_end_pipeline(model_path: Path) -> Tuple[bool, str]:
    """Test complete pipeline with tokenizer and model."""
    
    try:
        print("  Testing end-to-end pipeline...")
        
        # Load both tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        with InitializationWarningCapture() as warning_capture:
            model = AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            if warning_capture.warnings:
                return False, f"Pipeline has initialization warnings: {warning_capture.warnings}"
        
        # Test with real text
        test_texts = [
            "NeoBERT is the most efficient model of its kind!",
            "Machine learning models are improving rapidly.",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        for i, text in enumerate(test_texts, 1):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            if outputs.last_hidden_state is None:
                return False, f"Pipeline failed on text {i}"
            
            # Get CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
            print(f"    ✓ Text {i} -> embedding shape: {embedding.shape}")
        
        return True, "End-to-end pipeline works correctly"
        
    except Exception as e:
        return False, f"Pipeline failed: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Validate exported HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test exported model
  python tests/test_huggingface_export.py outputs/neobert_100m_100k/hf/neobert_100m_100k_100000
  
  # Test with verbose output
  python tests/test_huggingface_export.py outputs/neobert_100m_100k/hf/neobert_100m_100k_100000 --verbose
        """
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to exported HuggingFace model directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any warning, not just initialization warnings"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"❌ Error: Model directory not found: {model_path}")
        sys.exit(1)
    
    print(f"\n🔍 Validating exported model: {model_path}")
    print("=" * 60)
    
    all_passed = True
    results = []
    
    # 1. Check required files
    print("\n📁 Checking required files...")
    missing_files = validate_model_files(model_path)
    if missing_files:
        print(f"  ❌ Missing files: {', '.join(missing_files)}")
        results.append(("File validation", False, f"Missing: {', '.join(missing_files)}"))
        all_passed = False
    else:
        print("  ✅ All required files present")
        results.append(("File validation", True, "All files present"))
    
    # 2. Test model loading
    print("\n🤖 Testing model loading...")
    success, message = test_model_loading(model_path)
    if not success:
        print(f"  ❌ {message}")
        all_passed = False
    else:
        print(f"  ✅ {message}")
    results.append(("Model loading", success, message))
    
    # 3. Test tokenizer loading
    print("\n📝 Testing tokenizer loading...")
    success, message = test_tokenizer_loading(model_path)
    if not success:
        print(f"  ❌ {message}")
        all_passed = False
    else:
        print(f"  ✅ {message}")
    results.append(("Tokenizer loading", success, message))
    
    # 4. Test masked LM loading
    print("\n🎭 Testing MaskedLM model loading...")
    success, message = test_masked_lm_loading(model_path)
    if not success:
        print(f"  ❌ {message}")
        all_passed = False
    else:
        print(f"  ✅ {message}")
    results.append(("MaskedLM loading", success, message))
    
    # 5. Test end-to-end pipeline
    print("\n🔄 Testing end-to-end pipeline...")
    success, message = test_end_to_end_pipeline(model_path)
    if not success:
        print(f"  ❌ {message}")
        all_passed = False
    else:
        print(f"  ✅ {message}")
    results.append(("Pipeline test", success, message))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {test_name:20}")
        if not success and args.verbose:
            print(f"           {message}")
    
    print("=" * 60)
    
    if all_passed:
        print("✅ All tests passed! Model is ready for use.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()