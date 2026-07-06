import torch
import torch.nn.functional as F
from deepzero.models.transformer import GPT


@torch.no_grad()
def generate(model: GPT, tokenizer, prompt: str, max_len: int = 256,
             temperature: float = 0.8, top_k: int = 40, top_p: float = 1.0,
             stop_tokens: list[str] = None) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.config.device)
    stop_ids = set(tokenizer.encode(t) for t in (stop_tokens or []))
    for _ in range(max_len):
        if input_ids.size(1) > model.config.max_seq_len:
            input_ids = input_ids[:, -model.config.max_seq_len:]
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == tokenizer.EOS:
            break
        if stop_ids and next_id.item() in stop_ids:
            break
    return tokenizer.decode(input_ids[0].tolist())
