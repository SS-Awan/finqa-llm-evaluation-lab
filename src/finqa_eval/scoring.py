"""Numeric answer parsing and comparison for FinQA evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


_NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class NumericAnswer:
    """A normalized numerical answer with its percentage unit preserved."""

    value: Decimal
    is_percent: bool


def parse_numeric_answer(raw_value: str) -> NumericAnswer | None:
    """Parse a simple financial number such as '$ 1,234.50' or '64.9%'."""
    text = raw_value.strip()

    if not text:
        return None

    is_parenthetical_negative = text.startswith("(") and text.endswith(")")
    if is_parenthetical_negative:
        text = text[1:-1].strip()

    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()

    text = text.replace("$", "").replace(",", "").strip()

    if not _NUMBER_PATTERN.fullmatch(text):
        return None

    try:
        value = Decimal(text)
    except InvalidOperation:
        return None

    if is_parenthetical_negative:
        value = -abs(value)

    return NumericAnswer(value=value, is_percent=is_percent)


def is_numeric_answer(raw_value: str) -> bool:
    """Return whether an answer belongs in the numerical evaluation task."""
    return parse_numeric_answer(raw_value) is not None


def answers_match(
    expected: NumericAnswer,
    predicted: NumericAnswer,
    *,
    absolute_tolerance: Decimal = Decimal("0.01"),
    relative_tolerance: Decimal = Decimal("0.001"),
) -> bool:
    """Compare normalized answers while requiring matching percentage units."""
    if expected.is_percent != predicted.is_percent:
        return False

    allowed_difference = max(
        absolute_tolerance,
        abs(expected.value) * relative_tolerance,
    )
    return abs(expected.value - predicted.value) <= allowed_difference