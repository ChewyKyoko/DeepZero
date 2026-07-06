import torch


class Config:
    # --- Tokenizer ---
    vocab_size: int = 5000

    # --- Model ---
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = d_model * 4
    max_seq_len: int = 512
    dropout: float = 0.1

    # --- Training ---
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    max_steps: int = 10000
    grad_clip: float = 1.0
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    eval_steps: int = 10
    device: str = "auto"

    # --- Generation ---
    temperature: float = 0.8
    top_k: int = 40
    max_gen_len: int = 256

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
