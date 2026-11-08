"""One place to fix every source of randomness, so reruns give the same numbers."""
import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy and (if installed) PyTorch. Returns the seed used.

    Call this once at the start of every script or notebook, and log the seed
    in results/experiment_log.csv. Different seeds are used on purpose when we
    report the average over several runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses started after this call
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Slower but repeatable GPU behaviour; models may switch this off if speed matters.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed (for example during EDA); that is fine
    return seed
