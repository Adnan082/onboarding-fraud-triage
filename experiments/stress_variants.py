"""Stress test: score the Base-trained champion on months 6-7 of Variant IV and Variant V.

Alarms on the untouched months 6-7 of Base are natural drift, reported separately
and never called false alarms.
"""

from __future__ import annotations

from omegaconf import DictConfig


def run(cfg: DictConfig) -> dict:
    """Score each configured variant and return detector values per window."""
    raise NotImplementedError("TODO(week 2): score variant_iv and variant_v, months 6-7")
