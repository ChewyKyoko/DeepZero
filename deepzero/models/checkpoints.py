import os
import torch
from deepzero.models.transformer import GPT


def save_checkpoint(model: GPT, path: str, step: int = 0, loss: float = 0.0, extra: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"step": step, "model": model.state_dict(), "loss": loss, "config": model.config}
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[GPT, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    config = state.get("config", state.get("cfg"))
    if config is None:
        raise ValueError("No config found in checkpoint")
    model = GPT(config)
    model.load_state_dict(state["model"])
    return model, state
