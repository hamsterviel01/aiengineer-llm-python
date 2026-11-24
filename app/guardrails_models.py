"""Data models shared by the safety guards."""

from typing import NamedTuple


class GuardVerdict(NamedTuple):
    """Safety guard result; unpacks like a tuple or reads by attribute."""

    is_safe: bool
    reason: str
    categories: list[str]
