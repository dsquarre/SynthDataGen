"""Reproducible per-call seeding for stochastic generators.

The adaptive generation loop calls a generator once per batch. Re-seeding every
call with the same constant would make every batch of a given size identical, so
each call instead draws a fresh seed from one master stream. The whole run is
reproducible from BASE_SEED as long as the call order is unchanged.
"""
import random

import numpy as np

BASE_SEED = 2026
_seed_stream = np.random.default_rng(BASE_SEED)


def next_seed() -> int:
    """Next seed from the master stream."""
    return int(_seed_stream.integers(0, 2**31 - 1))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch global RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
