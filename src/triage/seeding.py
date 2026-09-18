"""One place to seed everything, so a clean clone reproduces every artefact."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and the hash seed. Model seeds are set in their configs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def rng(seed: int) -> np.random.Generator:
    """Return a fresh NumPy generator for a seeded draw."""
    return np.random.default_rng(seed)
