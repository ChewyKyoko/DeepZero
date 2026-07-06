import torch
from typing import Optional
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_metrics(model: torch.nn.Module, loader: Optional[DataLoader], device: str = "cpu") -> dict:
    if loader is None:
        return {}
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    avg_loss = total_loss / max(1, n)
    return {"val_loss": avg_loss, "perplexity": torch.exp(torch.tensor(avg_loss)).item()}
