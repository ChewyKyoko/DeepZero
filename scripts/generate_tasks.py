import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, yaml
import torch
from deepzero.models.checkpoints import load_checkpoint
from deepzero.tokenizers.base import create_tokenizer
from deepzero.datasets.base import create_dataset
from deepzero.evaluation.benchmark import ALL_TASKS


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/rl_buffer.jsonl"

    tokenizer_path = "data/bpe_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print("Training tokenizer on tiny_codes...")
        ds = create_dataset("tiny_codes")
        ds.preprocess()
        texts = ds.load_texts()
        tokenizer = create_tokenizer("bpe", vocab_size=5000)
        tokenizer.train(texts)
        tokenizer.save(tokenizer_path)
    else:
        from deepzero.tokenizers.bpe import BPETokenizer
        tokenizer = BPETokenizer.from_pretrained(tokenizer_path)

    model, state = load_checkpoint(checkpoint_path)
    model.eval()
    print(f"Loaded checkpoint step {state.get('step', '?')}, loss {state.get('loss', '?'):.4f}")

    existing = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing.add(entry.get("task_id", ""))

    generated = 0
    for task in ALL_TASKS:
        task_id = f"gen_{task.id}"
        if task_id in existing:
            continue

        prompt = task.prompt
        if task.test_code:
            prompt += "\n\nUse this test:\n" + task.test_code

        try:
            output = model.generate(tokenizer, prompt, max_len=200, temperature=0.8, top_k=40)
            entry = {
                "text": f"{task.prompt}\n```python\n{output}\n```",
                "priority": 1.0,
                "source": "generate",
                "task_id": task_id,
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            generated += 1
            print(f"  {task.id}: generated ({len(output)} chars)")
        except Exception as e:
            print(f"  {task.id}: failed ({e})")

    print(f"\nGenerated {generated} tasks to {output_path}")


if __name__ == "__main__":
    main()
