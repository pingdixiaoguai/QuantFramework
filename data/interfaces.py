"""Data layer interface — see docs/DESIGN.md §2.1.

This module re-exports the public API so that callers can do:
    from data.interfaces import query
"""

from data.store import query

__all__ = ["query"]
