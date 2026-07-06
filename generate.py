import torch
import torch.nn.functional as F


def generate(
    model,
    tokenizer,
    prompt: str,
    max_len: int = 256,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.0,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.cfg.device)

    with torch.no_grad():
        for _ in range(max_len):
            if input_ids.size(1) > model.cfg.max_seq_len:
                input_ids = input_ids[:, -model.cfg.max_seq_len :]
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id.item() == tokenizer.EOS:
                break

    return tokenizer.decode(input_ids[0].tolist())
