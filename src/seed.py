"""One place to fix every source of randomness, so reruns give the same numbers.

Purpose:
    ``set_seed`` seeds Python's ``random``, NumPy's global generator and, if it is installed, PyTorch
    (CPU and GPU).

Why this file exists:
    Training a model, sampling tweets and shuffling data all use random numbers. Without a fixed seed two runs
    of the same script give different scores, and a result could not be checked by someone else. Calling
    one function at the start of every script makes that setup identical everywhere, and the seed that
    was used can be logged in results/experiment_log.csv.

Place in the pipeline:
    Called once at the start of a script or notebook, before any data is split, shuffled or any model is built.

Inputs:
    An integer seed (default ``DEFAULT_SEED`` = 42).

Outputs:
    No data. Side effects: it changes global random-number state (``random``, ``numpy.random``,
    ``torch``), the environment variable ``PYTHONHASHSEED`` and two PyTorch cuDNN flags. It returns the seed.

Used by:
    src/baselines_tfidf.py, src/cmi_evaluation.py (for ``DEFAULT_SEED``), src/eda_part2.py,
    src/tfidf_error_analysis.py, src/train_bilstm.py, src/train_transformer.py, src/indian_transformer.py,
    src/zero_shot_llm.py, src/error_analysis_models.py and tests/test_data_seed.py.

Depends on:
    NumPy. PyTorch is optional and imported inside the function, so this module also works in environments
    without it (for example while only doing data analysis).

Design decisions:
    * Seeds must be set per run. When we report an average over several runs, different seeds are used on
      purpose (42, 43, 44 in this project).
    * ``cudnn.deterministic = True`` and ``benchmark = False`` trade some GPU speed for repeatable results.

What this does NOT cover (open points for anyone who needs stricter reproducibility):
    * ``numpy.random.default_rng(...)`` generators are separate objects and must be seeded where they are
      created (src/cmi_evaluation.py does this per file).
    * ``PYTHONHASHSEED`` only affects Python processes started after the call; the hash seed of the current
      interpreter is fixed when it starts.
    * It does not call ``torch.use_deterministic_algorithms`` and does not seed data-loader worker processes.
"""
import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy and (if installed) PyTorch, and return the seed used.

    Purpose:
        Make a run repeatable. Call this once at the start of every script or notebook, and log the seed in
        results/experiment_log.csv. Different seeds are used on purpose when we report the average over
        several runs.

    How it works:
        ``random.seed`` and ``np.random.seed`` seed the two global generators. The seed is also put into the
        environment variable ``PYTHONHASHSEED`` for child processes. PyTorch is imported inside a ``try``
        block: if it is present, its CPU and all-GPU generators are seeded and cuDNN is switched to
        deterministic mode.

    Args:
        seed: Integer seed. Default 42.

    Returns:
        The same seed, so a caller can write ``seed = set_seed(43)`` and log it.

    Side effects:
        Changes global random state, ``os.environ["PYTHONHASHSEED"]`` and, when PyTorch is installed,
        ``torch.backends.cudnn.deterministic`` and ``torch.backends.cudnn.benchmark``. These flags stay set
        for the rest of the process, also when no GPU is present.

    Edge cases:
        If PyTorch is not installed the ImportError is ignored. ``torch.cuda.manual_seed_all`` is safe
        without a GPU.

    Example:
        set_seed(43)  ->  43   (and all later random draws repeat on every run)
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
