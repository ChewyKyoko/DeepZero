"""Export a model checkpoint to a format suitable for GGUF conversion."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deepzero.models.checkpoints import load_checkpoint, save_checkpoint


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/exported.pt"

    model, state = load_checkpoint(checkpoint_path)
    state["model"] = model.state_dict()
    state["cfg"] = model.config
    torch.save(state, output_path)
    print(f"Exported {model.n_params:,} param model to {output_path}")


if __name__ == "__main__":
    main()
