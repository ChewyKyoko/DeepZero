import os
import random

import torch
from deepzero.models.transformer import GPT


def _strip_compile_prefix(state_dict: dict) -> dict:
    return ({k[len("_orig_mod."):]: v for k, v in state_dict.items()}
            if any(k.startswith("_orig_mod.") for k in state_dict) else state_dict)


def _get_rng_state() -> dict:
    import numpy as np
    return {
        "torch": torch.random.get_rng_state(),
        "random": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _set_rng_state(state: dict):
    import numpy as np
    if "torch" in state:
        torch.random.set_rng_state(state["torch"].cpu())
    if "random" in state:
        random.setstate(state["random"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])


CHECKPOINT_NAMES = {
    "best_loss": "best_loss.pt",
    "best_val": "best_val.pt",
    "latest": "latest.pt",
    "fastest": "fastest.pt",
    "final": "final.pt",
}


def save_checkpoint(model: GPT, path: str, step: int = 0, loss: float = 0.0,
                    tag: str = "latest", extra: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    state = {
        "step": step, "model": sd, "loss": loss, "config": model.config,
        "tag": tag, "rng": _get_rng_state(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu", model: GPT = None,
                    restore_rng: bool = False) -> tuple[GPT, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    config = state.get("config", state.get("cfg"))
    if config is None:
        raise ValueError("No config found in checkpoint")
    if model is None:
        model = GPT(config)
    model.load_state_dict(_strip_compile_prefix(state["model"]))
    if restore_rng and "rng" in state:
        _set_rng_state(state["rng"])
    return model, state


def try_resume(checkpoint_dir: str, model: GPT, device: str = "cpu",
               optimizer=None, scheduler=None,
               restore_rng: bool = False) -> int:
    for name in ("latest.pt", "best_val.pt", "best.pt", "final.pt"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(_strip_compile_prefix(state["model"]))
            step = state.get("step", 0)
            if optimizer is not None and "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None:
                from torch.optim.lr_scheduler import LambdaLR
                if isinstance(scheduler, LambdaLR):
                    scheduler._last_lr = [scheduler.lr_lambdas[0](max(0, step - 1))]
                else:
                    scheduler.last_epoch = max(0, step - 1)
            if restore_rng and "rng" in state:
                _set_rng_state(state["rng"])
            print(f"Resumed from {path} (step {step}, loss {state.get('loss', '?'):.4f})")
            return step
    return 0
