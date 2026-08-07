"""Lossless scalar Fredkin primitives used as the semantic oracle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FredkinResult:
    """All three outputs of a conventional positive-control Fredkin gate."""

    control: bool
    first: bool
    second: bool

    def as_tuple(self) -> tuple[bool, bool, bool]:
        return (self.control, self.first, self.second)


def fredkin_gate(control: bool, first: bool, second: bool) -> FredkinResult:
    """Swap the data lines when ``control`` is true, retaining every output."""

    control = bool(control)
    first = bool(first)
    second = bool(second)
    if control:
        return FredkinResult(control, second, first)
    return FredkinResult(control, first, second)


def fredkin_literal_condition(
    action_include: bool, literal_truth: bool
) -> FredkinResult:
    """Gate a literal for conjunction without discarding the garbage line.

    The gate receives ``(action_include, True, literal_truth)``. An included
    literal is routed to ``first``; an excluded literal produces the neutral
    conjunction value ``True``. ``second`` must be retained for reversibility.
    """

    return fredkin_gate(action_include, True, literal_truth)

