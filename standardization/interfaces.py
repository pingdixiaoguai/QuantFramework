"""Standardization layer interface — see docs/DESIGN.md §2.3.

This module re-exports the public API so that callers can do:
    from standardization.interfaces import standardize
"""

from standardization.methods import standardize

__all__ = ["standardize"]
