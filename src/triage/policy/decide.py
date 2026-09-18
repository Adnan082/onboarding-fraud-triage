"""Conformal set -> decision band. Rule 5: there is never an automatic decline."""

from __future__ import annotations

from typing import Literal

import numpy as np

Decision = Literal["approve", "review", "verify"]
BANDS: tuple[Decision, ...] = ("approve", "review", "verify")


def decide(has_legit: bool, has_fraud: bool) -> Decision:
    """Map one conformal set to a band.

    ``{legit}`` approves, ``{legit, fraud}`` and ``{}`` go to review, and
    ``{fraud}`` goes to verify (step-up checks or a human).
    """
    if has_legit and not has_fraud:
        return "approve"
    if has_fraud and not has_legit:
        return "verify"
    return "review"


def decide_many(sets: np.ndarray) -> np.ndarray:
    """Vectorised :func:`decide` over a boolean ``(n, 2)`` array of ``[legit, fraud]``.

    Returns an array of band names, in the same row order.
    """
    matrix = np.asarray(sets, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError(f"expected a boolean (n, 2) set matrix, got shape {matrix.shape}")

    has_legit, has_fraud = matrix[:, 0], matrix[:, 1]
    decisions = np.full(matrix.shape[0], "review", dtype="<U7")
    decisions[has_legit & ~has_fraud] = "approve"
    decisions[has_fraud & ~has_legit] = "verify"
    return decisions


def band_shares(decisions: np.ndarray) -> dict[str, float]:
    """Share of applications in each band. The three shares sum to 1."""
    values = np.asarray(decisions)
    if values.size == 0:
        return dict.fromkeys(BANDS, 0.0)
    return {band: float((values == band).mean()) for band in BANDS}
