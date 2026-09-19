"""Injected upstream data bugs, each starting at window 5 of month 6.

(a) swap the two most frequent ``employment_status`` codes;
(b) mirror ``income`` through a lookup over its observed values (lowest <-> highest),
    never ``1 - x``, so the values stay exactly within the contract;
(c) set ``phone_mobile_valid`` to 1 for every row;
(d) multiply ``income`` by 10 -- out of contract, so batch validation AND the API
    must reject it.

Bugs (a) to (c) are the interesting ones: every value stays legal, so no schema
check can catch them. They are what the label-free monitor is *for*. Bug (d) is
the control -- it should never reach a detector at all, because the contract
rejects it at the door.

For each bug we report the detection delay (applications from the first bugged
row to ALERT), false alarms on clean windows, and model degradation (TPR at the
deployment threshold, before against after).
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("triage")


def swap_top_categories(frame: pd.DataFrame, column: str, n: int = 2) -> pd.DataFrame:
    """Bug (a): swap the ``n`` most frequent codes in ``column``.

    A plausible upstream mistake: a mapping table edited in the wrong order. The
    category set is unchanged, and so are the overall counts -- only the meaning
    of each code moves, which no schema check can see.
    """
    if n != 2:
        raise ValueError("only a pairwise swap is defined; got n=" + str(n))

    top = frame[column].value_counts().index[:n].tolist()
    if len(top) < n:
        raise ValueError(f"{column} has fewer than {n} categories to swap")

    first, second = top[0], top[1]
    out = frame.copy()
    out[column] = (
        frame[column]
        .map({first: second, second: first})
        .fillna(frame[column].astype(object))
        .astype(frame[column].dtype)
    )
    return out


def mirror_lookup(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Bug (b): map the sorted observed values onto their reverse.

    Deliberately a lookup and not ``1 - x``: every output is a value the column
    already takes, so the result stays exactly inside the contract. A decile-coded
    income flipped end for end is a realistic upstream inversion and completely
    invisible to a range check.
    """
    values = sorted(frame[column].dropna().unique().tolist())
    lookup = dict(zip(values, reversed(values), strict=True))

    out = frame.copy()
    out[column] = frame[column].map(lookup).astype(frame[column].dtype)
    return out


def set_constant(frame: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    """Bug (c): overwrite a column with one value.

    The classic upstream failure: a verification service goes down and something
    defaults its answer to "valid" for everyone.
    """
    out = frame.copy()
    out[column] = value
    return out


def multiply(frame: pd.DataFrame, column: str, factor: float) -> pd.DataFrame:
    """Bug (d): scale a column out of its contracted range.

    A unit change upstream. Unlike the others this one is *not* subtle, which is
    the point: the contract must stop it before any model sees it.
    """
    out = frame.copy()
    out[column] = frame[column] * factor
    return out


BUGS = {
    "swap_top_categories": swap_top_categories,
    "mirror_lookup": mirror_lookup,
    "constant": set_constant,
    "multiply": multiply,
}


def apply_bug(frame: pd.DataFrame, spec) -> pd.DataFrame:
    """Apply one configured bug spec to a frame."""
    kind = str(spec.kind)
    column = str(spec.column)

    if kind == "swap_top_categories":
        return swap_top_categories(frame, column, int(spec.get("n", 2)))
    if kind == "mirror_lookup":
        return mirror_lookup(frame, column)
    if kind == "constant":
        return set_constant(frame, column, spec.value)
    if kind == "multiply":
        return multiply(frame, column, float(spec.factor))
    raise ValueError(f"unknown bug kind {kind!r}, expected one of {sorted(BUGS)}")
