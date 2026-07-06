# Architecture

## Overview

DeepZero uses a standard decoder-only transformer architecture (GPT-style) with:

- **RMSNorm** (instead of LayerNorm)
- **SwiGLU** feed-forward (instead of ReLU)
- **Scaled Dot-Product Attention** (PyTorch native, FlashAttention-compatible)
- **Weight tying** between token embedding and LM head

## Components

### RMSNorm (`deepzero/models/layers.py`)

Root Mean Square Layer Normalization. More efficient than standard LayerNorm:

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma $$

### CausalSelfAttention (`deepzero/models/layers.py`)

Multi-head causal self-attention. Uses `F.scaled_dot_product_attention` for fused QKV computation and memory-efficient attention.

- `n_heads`: number of attention heads
- `head_dim = d_model / n_heads`

### MLP (SwiGLU) (`deepzero/models/layers.py`)

Gated feed-forward with SiLU activation:

$$ \text{SwiGLU}(x) = (\text{SiLU}(x W_g) \odot (x W_u)) W_d $$

### GPT (`deepzero/models/transformer.py`)

The top-level transformer model:

1. Token embedding + learned positional embedding
2. Stack of `n_layers` TransformerBlocks
3. Final RMSNorm + linear projection to vocabulary

## Configuration

All model parameters are in `ModelConfig` dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| vocab_size | 5000 | Vocabulary size |
| d_model | 384 | Embedding dimension |
| n_layers | 8 | Number of transformer blocks |
| n_heads | 8 | Number of attention heads |
| d_ff | 1536 | Feed-forward hidden dimension |
| max_seq_len | 512 | Maximum sequence length |
| dropout | 0.1 | Dropout rate |
| device | "cpu" | Computation device |

## Parameter Count

For the default configuration:
- `d_model=384, n_layers=8, n_heads=8, d_ff=1536, vocab_size=1300`
- **~19.6M parameters**

Formula:
$$ N = V \cdot d + n_l \cdot (4 \cdot d^2 + 3 \cdot d \cdot d_{ff}) + d \cdot V $$
