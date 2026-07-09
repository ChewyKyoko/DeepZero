# Roadmap

```mermaid
timeline
    title DeepZero Development Roadmap

    R0.1 : Base model
         : GPT-style decoder-only transformer
         : RMSNorm + SwiGLU + causal attention
         : Weight tying (embed ↔ lm_head)
         : Character tokenizer
         : tiny-textbooks dataset
         : PackedDataset chunking
         : Training loop + gradient accumulation
         : LR warmup + cosine decay

    R0.2 : Optimizer benchmarks
         : AdamW / Muon / Sophia / Lion via registry
         : Opt-in weight decay param groups
         : Experiment tracking
         : MetricsTracker + time-to-target
         : ExperimentManager + run dirs
         : Rich terminal dashboard
         : 6 matplotlib plot types
         : Auto-generated reports
         : 5-mode checkpoint system + RNG resume
         : Early stopping + config inheritance

    R0.3 : Evaluation + Search
         : Perplexity + 5-category benchmarks
         : Hyperparameter grid search
         : Dataset versioning
         : Lock Sophia as default optimizer
         : tiny.yaml for rapid research (<1h per experiment)

    R1 Tiny : Datasets, Tokenizers, Architectures
            : Explore 6+ dataset sources + mixtures
            : Compare BPE / Byte BPE / Unigram / Character
            : Architecture variants (MOE, linear attention, etc.)
            : All experiments on tiny.yaml with Sophia

    R1 Full : Scale & Ship
             : Scale to 100M+ parameters
             : Multi-GPU training (DDP / FSDP)
             : Verify Sophia holds at larger scale
             : ONNX export
             : GGUF export
             : Inference server (REST API)
```

## Release Plan

| Release | Focus | Model | Optimizer | Iteration Speed |
|---------|-------|-------|-----------|-----------------|
| **R0.x Tiny** | Lock infrastructure | 1.25M (tiny.yaml) | **Sophia** | ~1h per experiment |
| **R1 Tiny** | Dataset/tokenizer/architecture research | 1.25M (tiny.yaml) | Sophia | ~1h per experiment |
| **R1 Full** | Scale, deploy, verify | 100M+ | Sophia (verify at scale) | hours to days |

## Optimizer Decision

**Sophia is locked as the default** for all R0.x and R1 Tiny research. At R1 Full, we re-verify whether Sophia still wins at 100M+ parameters — optimizer performance can change with model scale.
