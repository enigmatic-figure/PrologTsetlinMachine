"""Executable reference objects admitted by the PTA exact-lowering gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..representation import LiteralDescriptor


@dataclass(frozen=True, slots=True)
class ExecutableBinaryClause:
    """An exact conjunction over materialized literal identities.

    The object executes on the Booleanized PTM substrate. Literal evaluation
    from raw records remains the catalog's responsibility; this object owns the
    exact clause activation semantics after those literal truth values exist.
    """

    literal_descriptors: tuple[LiteralDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.literal_descriptors:
            raise ValueError("executable binary clause must contain a literal")
        if any(
            not isinstance(descriptor, LiteralDescriptor)
            for descriptor in self.literal_descriptors
        ):
            raise TypeError("binary clause entries must be LiteralDescriptor")
        literal_ids = tuple(
            descriptor.literal_id for descriptor in self.literal_descriptors
        )
        if literal_ids != tuple(sorted(set(literal_ids))):
            raise ValueError(
                "executable binary clause literal IDs must be sorted and unique"
            )

    @property
    def literal_ids(self) -> tuple[int, ...]:
        return tuple(
            descriptor.literal_id for descriptor in self.literal_descriptors
        )

    def evaluate(self, literal_truth: Mapping[int, bool]) -> bool:
        """Return conjunction activation for one Booleanized example."""
        if not isinstance(literal_truth, Mapping):
            raise TypeError("literal_truth must be a mapping")
        values: list[bool] = []
        for literal_id in self.literal_ids:
            try:
                value = literal_truth[literal_id]
            except KeyError as exc:
                raise KeyError(f"missing truth value for literal {literal_id}") from exc
            if type(value) is not bool:
                raise TypeError(
                    f"truth value for literal {literal_id} must be bool"
                )
            values.append(value)
        return all(values)
