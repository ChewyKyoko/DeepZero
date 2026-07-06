import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from deepzero.models.layers import TransformerBlock, RMSNorm


@dataclass
class ModelConfig:
    vocab_size: int = 5000
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1536
    max_seq_len: int = 512
    dropout: float = 0.1
    device: str = "cpu"


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = x.shape
        assert T <= self.config.max_seq_len
        pos = torch.arange(0, T, device=x.device, dtype=torch.long)
        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def generate(self, tokenizer, prompt: str, max_len: int = 256,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        self.eval()
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.config.device)
        for _ in range(max_len):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.EOS:
                break
        return tokenizer.decode(input_ids[0].tolist())

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, seq_len: int) -> int:
        cfg = self.config
        return 2 * cfg.n_layers * seq_len * (4 * cfg.d_model * cfg.d_ff + 4 * cfg.d_model ** 2)
