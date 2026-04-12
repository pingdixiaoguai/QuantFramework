"""Factor layer interface — see docs/DESIGN.md §2.2.

This module re-exports the public API so that callers can do:
    from factors.interfaces import load_registered_factors, validate
"""

from factors.registry import load_registered_factors
from factors.validator import validate

__all__ = ["load_registered_factors", "validate"]
