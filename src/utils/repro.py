"""Reproducibility utilities for setting random seeds across libraries.

This module provides functions to ensure reproducible results by setting
random seeds for Python's random module, NumPy, and scikit-learn.
"""

import random
from typing import Optional

import numpy as np

try:
    from ..constants import RANDOM_SEED
except ImportError:  # pragma: no cover - fallback for script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from constants import RANDOM_SEED


def set_global_seed(seed: Optional[int] = None) -> None:
    """Set random seeds for reproducibility across all libraries.

    This function configures random number generators for:
    - Python's built-in random module
    - NumPy
    - Any environment variables that affect determinism

    Parameters
    ----------
    seed : int, optional
        The random seed to use. If None, uses the default RANDOM_SEED constant.
    """
    if seed is None:
        seed = RANDOM_SEED

    # Set Python's random module seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Note: scikit-learn models accept random_state parameter directly,
    # so we pass the seed to each model rather than setting a global state
