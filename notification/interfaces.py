"""Notification layer interface — see docs/DESIGN.md §2.6."""

from abc import ABC, abstractmethod


class Notifier(ABC):
    """Base class for notification adapters.

    Each channel (DingTalk, WeChat, log file) implements this interface.
    See docs/DESIGN.md §2.6 for the full interface contract.
    """

    @abstractmethod
    def send(self, message: str) -> None:
        """Send a formatted message to the external channel."""
        ...
