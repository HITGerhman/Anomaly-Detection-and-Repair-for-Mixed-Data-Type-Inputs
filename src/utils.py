"""Backward-compatible utility exports.

Historically, app code imported training helpers from this module.
The actual algorithm implementation now lives in `src.training_core`.
"""

from __future__ import annotations

from .training_core import load_system_state, process_and_train, save_system_state

__all__ = ["process_and_train", "save_system_state", "load_system_state"]

