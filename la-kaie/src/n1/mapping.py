"""Exchange pairing and mapping (spec §8).

All three mappings reduce to x' = x_d with the full mask and to x_r with
an empty mask (T-MAP), which makes the representations comparable.
"""
from __future__ import annotations

import numpy as np


def ranks(F: np.ndarray) -> np.ndarray:
    """Agent indices ordered by fitness; ties -> lower index (stable sort)."""
    return np.argsort(F, kind="stable")


def pick_receiver(rng, F: np.ndarray) -> int:
    """Binary tournament of two distinct agents; lower f wins, ties -> lower index."""
    a, b = rng.choice(F.shape[0], size=2, replace=False)
    a, b = int(a), int(b)
    if F[a] < F[b] or (F[a] == F[b] and a < b):
        return a
    return b


def draw_mask(rng, n_units: int, p_x: float) -> np.ndarray:
    """Each unit with probability p_x; if none is drawn, one uniformly (spec §8.2)."""
    m = rng.random(n_units) < p_x
    if not m.any():
        m[int(rng.integers(n_units))] = True
    return m


def n_units(kind: str, D: int, partition=None) -> int:
    return len(partition) if kind == "HB" else D


def apply_mask(kind: str, x_r, x_d, s, m, partition=None, U=None) -> np.ndarray:
    """Deterministic part of the mapping for a given unit mask m."""
    d = x_d - x_r
    if kind == "HI":
        return x_r + m * d
    if kind == "HB":
        x = x_r.copy()
        for b, sel in zip(partition, m):
            if sel:
                x[b] = x_r[b] + d[b]
        return x
    if kind == "HR":
        return x_r + s * (U @ (m * (U.T @ (d / s))))
    raise KeyError(kind)


def map_candidate(kind: str, x_r, x_d, s, rng, p_x: float, partition=None, U=None):
    """Returns (x', mask, n_units). No clipping here (the caller clips)."""
    k = n_units(kind, x_r.shape[0], partition)
    m = draw_mask(rng, k, p_x)
    return apply_mask(kind, x_r, x_d, s, m.astype(float) if kind != "HB" else m, partition, U), m, k
