# Using Custom Tokenizers

NeoBERT supports any HuggingFace-compatible tokenizer. This guide shows how to use custom tokenizers for your specific domain or language.

## Quick Start

```bash
# Train with a custom tokenizer
python scripts/pretraining/pretrain.py \
    --config configs/train_small_custom_tokenizer.yaml \
    --tokenizer.name "your-org/your-tokenizer"
```

## Supported Tokenizers

NeoBERT automatically handles special tokens for these tokenizers:
- `bert-base-uncased`
- `google-bert/bert-base-uncased`
- `BEE-spoke-data/wordpiece-tokenizer-32k-en_code-msp`

For other tokenizers, special tokens are automatically added if missing.

## Using a Custom Tokenizer

### 1. Basic Usage

```python
from transformers import AutoTokenizer
from neobert.tokenizer import get_tokenizer

# Load any HuggingFace tokenizer
tokenizer = get_tokenizer("microsoft/codebert-base")

# Or use a local tokenizer
tokenizer = get_tokenizer("./path/to/my/tokenizer")
```

### 2. Configuration

In your YAML config:
```yaml
tokenizer:
  name: "your-org/your-tokenizer"
  vocab_size: 32000  # Must match your tokenizer
  max_length: 512
  
model:
  vocab_size: 32000  # Must match tokenizer vocab_size
```

### 3. Tokenizing Datasets

For raw text datasets, tokenize before training:

```bash
python scripts/pretraining/tokenize_dataset.py \
    --tokenizer "your-org/your-tokenizer" \
    --dataset "your-dataset" \
    --max_length 512 \
    --output "./tokenized_data/custom"
```

## Creating a Custom Tokenizer

### 1. Train a New Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Create a WordPiece tokenizer
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train on your corpus
trainer = trainers.WordPieceTrainer(
    vocab_size=32000,
    special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
)

files = ["corpus1.txt", "corpus2.txt"]
tokenizer.train(files, trainer)

# Save it
tokenizer.save("my-tokenizer.json")
```

### 2. Convert to HuggingFace Format

```python
from transformers import PreTrainedTokenizerFast

# Load the trained tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my-tokenizer.json",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    unk_token="[UNK]"
)

# Save in HuggingFace format
tokenizer.save_pretrained("./my-hf-tokenizer")
```

### 3. Upload to HuggingFace Hub

```bash
# Login to HuggingFace
huggingface-cli login

# Upload tokenizer
huggingface-cli upload your-org/your-tokenizer ./my-hf-tokenizer
```

## Domain-Specific Examples

### Code Tokenizer

For programming languages:
```python
from tokenizers import pre_tokenizers

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation()
])

# Add programming-specific tokens
special_tokens = [
    "[INDENT]", "[DEDENT]", "[NEWLINE]",
    "[COMMENT]", "[STRING]", "[NUMBER]"
]
```

### Scientific Tokenizer

For scientific text:
```python
# Include special tokens for formulas
special_tokens = [
    "[FORMULA]", "[CITE]", "[REF]",
    "[SUB]", "[SUP]", "[FRAC]"
]

# Custom pre-tokenization for LaTeX
tokenizer.pre_tokenizer = pre_tokenizers.Regex(
    r'\$[^$]+\$|[^\s$]+',
    behavior="isolated"
)
```

### Multilingual Tokenizer

```python
# Use SentencePiece for better multilingual support
from tokenizers import models

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

# Train on multilingual corpus
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
)
```

## Tokenizer Compatibility

### Vocabulary Size

Ensure consistency:
```python
# Check tokenizer vocab size
tokenizer = AutoTokenizer.from_pretrained("your-tokenizer")
print(f"Vocab size: {tokenizer.vocab_size}")

# Update config
config.model.vocab_size = tokenizer.vocab_size
config.tokenizer.vocab_size = tokenizer.vocab_size
```

### Special Tokens

Required special tokens:
- `pad_token`: For padding sequences
- `cls_token`: Start of sequence
- `sep_token`: Separator token
- `mask_token`: For MLM pretraining
- `unk_token`: Unknown token

Check if your tokenizer has them:
```python
print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
print(f"Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
```

## Performance Considerations

### Fast vs Slow Tokenizers

Always use fast tokenizers when available:
```python
# Slow (Python-based)
from transformers import BertTokenizer

# Fast (Rust-based) - Recommended
from transformers import BertTokenizerFast
```

### Tokenization Strategies

1. **Pre-tokenize datasets** for faster training:
```bash
python scripts/pretraining/tokenize_dataset.py \
    --num_proc 16  # Use multiple processes
```

2. **Optimize sequence length**:
```python
# Analyze token lengths
lengths = [len(tokenizer.encode(text)) for text in texts]
print(f"Average length: {np.mean(lengths)}")
print(f"95th percentile: {np.percentile(lengths, 95)}")
```

3. **Dynamic padding** for efficiency:
```yaml
datacollator:
  pad_to_multiple_of: 8  # For tensor cores
```

## Troubleshooting

### Vocabulary Mismatch

Error: `size mismatch for embeddings.weight`

Solution:
```bash
# Check both configs
--model.vocab_size 32000 \
--tokenizer.vocab_size 32000
```

### Missing Special Tokens

Error: `AttributeError: 'NoneType' object has no attribute 'ids'`

Solution:
```python
# Add missing tokens
tokenizer.add_special_tokens({
    'pad_token': '[PAD]',
    'mask_token': '[MASK]'
})
```

### Tokenizer Not Found

Error: `OSError: Can't load tokenizer`

Solution:
```bash
# Use full path or HuggingFace ID
--tokenizer.name "bert-base-uncased"  # HF Hub
--tokenizer.name "./my-tokenizer"     # Local path
```

## Examples

### Example 1: Domain-Specific Tokenizer

Training on biomedical text:
```bash
# Use BioBERT tokenizer
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --tokenizer.name "dmis-lab/biobert-v1.1" \
    --model.vocab_size 28996
```

### Example 2: Code-Aware Tokenizer

For source code:
```bash
# Use CodeBERT tokenizer
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --tokenizer.name "microsoft/codebert-base" \
    --model.vocab_size 50265
```

### Example 3: Efficient Small Tokenizer

For resource-constrained settings:
```bash
# Use a smaller vocabulary
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --tokenizer.name "google/bert_uncased_L-2_H-128_A-2" \
    --model.vocab_size 30522 \
    --model.hidden_size 128
```

## Next Steps

- Learn about [Training](training.md) with custom tokenizers
- Explore [Model Architecture](architecture.md) adaptations
- See [Evaluation](evaluation.md) with domain-specific metrics