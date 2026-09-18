"""Shared plumbing for the Hydra stage apps.

The implementation lives in :mod:`triage.runtime`, so that ``triage.data.load``
and ``triage.data.contract`` -- which are Hydra apps too, but not stages -- share
exactly the same seeding and provenance logging.
"""

from __future__ import annotations

from triage.runtime import CONFIG_PATH, log, start

__all__ = ["CONFIG_PATH", "log", "start"]
