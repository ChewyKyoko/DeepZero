import random
import torch

try:
    import numpy as np
except ImportError:
    np = None


def set_seed(seed: int = 42):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
