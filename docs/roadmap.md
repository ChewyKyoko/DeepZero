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
         : Training loop with gradient accumulation
         : LR warmup + cosine decay

    R0.2 : Optimizer benchmarks
         : AdamW / Muon / Sophia / Lion via registry
         : Opt-in weight decay param groups
         : Experiment tracking
         : MetricsTracker (per-step + time-to-target)
         : ExperimentManager (run dir, metadata, finalize)
         : Rich terminal dashboard
         : 6 matplotlib plot types
         : Auto-generated Markdown reports
         : 5-mode checkpoint system
         : RNG save/restore for resume
         : Early stopping with patience
         : Config inheritance (base: key)

    R0.3 : Evaluation framework
         : Perplexity + 5-category benchmarks
         : Code completion, syntax, math, reasoning, trivia
         : Hyperparameter search
         : Grid sweep over search: config section
         : Trial leaderboard CSV
         : Dataset versioning (dataset.json)

    R0.4 : Model scaling
         : Scale to 100M+ parameters
         : Multi-GPU training (DDP / FSDP)
         : Deployment
         : ONNX export
         : GGUF export
         : Inference server (REST API)
```

## Release Plan

| Release | Focus | Key Deliverables |
|---------|-------|------------------|
| **R0.1** | Foundation | Transformer core, tokenizer, data pipeline, basic training |
| **R0.2** | Experimentation | Optimizer comparison, metrics tracking, visualization, reports, checkpoints |
| **R0.3** | Evaluation & Search | Model eval suite, hyperparameter grid search, dataset versioning |
| **R0.4** | Scale & Ship | Large model training, multi-GPU, ONNX/GGUF export, inference serving |
