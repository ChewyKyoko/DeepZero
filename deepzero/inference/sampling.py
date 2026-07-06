import torch
import torch.nn.functional as F


def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    logits[logits < v[:, -1:]] = float("-inf")
    return F.softmax(logits, dim=-1)


def sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float("-inf")
    return F.softmax(logits, dim=-1)
