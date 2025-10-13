# NeoBERT Architecture

This document describes the technical architecture of NeoBERT, a modern BERT-style encoder with several improvements over the original BERT architecture.

## Overview

NeoBERT is a transformer encoder architecture that incorporates several modern improvements:

- **RoPE (Rotary Position Embeddings)**: More flexible positional encoding
- **RMSNorm**: More efficient normalization
- **SwiGLU Activation**: Better activation function (requires xformers)
- **Flash Attention**: Faster and more memory-efficient attention
- **Optional nGPT-style normalization**: Experimental normalized transformer variant

## Core Components

### 1. Embedding Layer

```python
class NeoBERT(nn.Module):
    def __init__(self, config):
        # Token embeddings only (no position embeddings with RoPE)
        self.encoder = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
```

Key differences from BERT:

- No position embedding layer (uses RoPE instead)
- No token type embeddings
- Simpler, more efficient design

### 2. Rotary Position Embeddings (RoPE)

```python
def compute_rope_embeddings(seq_len, dim, base=10000):
    """Compute rotary position embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs, freqs], dim=-1)
```

Benefits:

- Relative position encoding
- Extrapolates to longer sequences
- No learned parameters

### 3. RMSNorm Normalization

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (norm + self.eps) * self.scale
```

Advantages over LayerNorm:

- ~2x faster
- No mean centering
- More stable training

### 4. Multi-Head Attention

```python
class MultiheadAttention(nn.Module):
    def __init__(self, config):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
```

Features:

- Supports Flash Attention for efficiency
- RoPE integration for positions
- Grouped Query Attention compatible

### 5. SwiGLU Activation

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

Benefits:

- Better than GELU/ReLU
- Smoother gradients
- Improved performance

### 6. Encoder Block

```python
class EncoderBlock(nn.Module):
    def __init__(self, config):
        # Normalization layers
        self.attention_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

        # Core layers
        self.attention = MultiheadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, pad_mask=None, freqs_cis=None):
        # Pre-norm architecture
        h = x + self.attention(self.attention_norm(x), pad_mask, freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

## Model Variants

### 1. Base NeoBERT

Standard encoder for general use:

```python
model = NeoBERT(config)
embeddings = model(input_ids, attention_mask)
```

### 2. NeoBERTLMHead

For masked language modeling:

```python
model = NeoBERTLMHead(config)
outputs = model(input_ids, labels=labels)
loss = outputs["loss"]
logits = outputs["logits"]
```

### 3. NeoBERTForSequenceClassification

For classification tasks:

```python
model = NeoBERTForSequenceClassification(config, num_labels=2)
outputs = model(input_ids, attention_mask, labels=labels)
```

### 4. NormNeoBERT

Experimental normalized transformer (nGPT-style):

```python
# All weights constrained to unit norm
model = NormNeoBERT(config)
```

## Configuration Options

### Model Sizes

**NeoBERT-Small** (110M parameters):

```yaml
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
```

**NeoBERT-Base** (220M parameters):

```yaml
model:
  hidden_size: 1024
  num_hidden_layers: 12
  num_attention_heads: 16
  intermediate_size: 4096
```

**NeoBERT-Large** (440M parameters):

```yaml
model:
  hidden_size: 1024
  num_hidden_layers: 24
  num_attention_heads: 16
  intermediate_size: 4096
```

### Activation Functions

```yaml
model:
  hidden_act: "swiglu"  # Default - best performance
  # hidden_act: "gelu"  # Alternative - no dependencies
  # hidden_act: "relu"  # Fastest but lower quality
```

**SwiGLU Requirements:**

- Install xformers for optimal performance: `pip install xformers==0.0.28.post3`
- Falls back to native PyTorch implementation if xformers unavailable
- ~5-10% performance improvement over GELU

### Attention Variants

```yaml
model:
  flash_attention: true  # Faster, less memory
  rope: true            # Rotary embeddings
  rope_theta: 10000     # RoPE base frequency
```

## Memory and Compute

### Memory Usage

| Model Size | Parameters | Memory (FP32) | Memory (BF16) |
| ---------- | ---------- | ------------- | ------------- |
| Small      | 110M       | 440MB         | 220MB         |
| Base       | 220M       | 880MB         | 440MB         |
| Large      | 440M       | 1.76GB        | 880MB         |

**Note**: BF16 (bfloat16) is recommended for all modern GPUs (NVIDIA Ampere/RTX 30xx and newer). BF16 provides better numerical stability than FP16 while maintaining the same memory efficiency.

### Compute Requirements

Training throughput (examples/second):

| Model | V100 (32GB) | A100 (40GB) | RTX 4090 |
| ----- | ----------- | ----------- | -------- |
| Small | 450         | 720         | 380      |
| Base  | 230         | 380         | 180      |
| Large | 110         | 190         | 85       |

## Optimizations

### 1. Flash Attention

Enable for 2-4x speedup:

```python
config.flash_attention = True
```

Requirements:

- GPU with compute capability >= 7.0
- flash-attn package installed (tested with v2.7.3)

### 2. Gradient Checkpointing

Trade compute for memory:

```python
model.gradient_checkpointing_enable()
```

### 3. Mixed Precision

```python
# Automatic mixed precision
with torch.cuda.amp.autocast():
    outputs = model(input_ids)
```

### 4. Torch Compile (PyTorch 2.0+)

```python
model = torch.compile(model)
```

## Implementation Details

### Attention Mask Format

NeoBERT uses additive attention masks:

```python
# Convert HuggingFace format
def convert_attention_mask(attention_mask):
    # HF format: 1 = keep, 0 = mask
    # NeoBERT format: 0 = keep, -inf = mask
    return torch.where(
        attention_mask == 0,
        float("-inf"),
        float(0.0)
    )
```

### Position Encoding

RoPE is applied in the attention layer:

```python
def apply_rope(q, k, freqs_cis):
    q_complex = view_as_complex(q)
    k_complex = view_as_complex(k)

    q_rotated = q_complex * freqs_cis
    k_rotated = k_complex * freqs_cis

    return view_as_real(q_rotated), view_as_real(k_rotated)
```

### Initialization

Weight initialization follows BERT:

```python
def _init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
```

## Differences from BERT

| Feature           | BERT               | NeoBERT         |
| ----------------- | ------------------ | --------------- |
| Position Encoding | Learned embeddings | RoPE            |
| Normalization     | LayerNorm          | RMSNorm         |
| Activation        | GELU               | SwiGLU          |
| Attention         | Standard           | Flash Attention |
| Token Types       | Yes                | No              |
| Architecture      | Post-norm          | Pre-norm        |

## Extension Points

### Custom Activations

```python
class CustomActivation(nn.Module):
    def forward(self, x):
        # Your activation here
        return x

# Register it
ACTIVATION_FUNCTIONS["custom"] = CustomActivation
```

### Custom Attention

```python
class CustomAttention(MultiheadAttention):
    def forward(self, x, mask=None):
        # Your attention mechanism
        return x
```

### Model Plugins

```python
class NeoBERTWithAdapter(NeoBERT):
    def __init__(self, config):
        super().__init__(config)
        self.adapters = nn.ModuleList([
            Adapter(config) for _ in range(config.num_hidden_layers)
        ])
```

## Next Steps

- Learn about [Training](/docs/training.md) NeoBERT models
- Explore [Configuration](/docs/configuration.md) options
- Review [Evaluation](/docs/evaluation.md) guide
