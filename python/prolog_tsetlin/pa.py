"""Portable fixed-shape Prolog Automaton buffer and kernel oracle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class PortSemantic(str, Enum):
    LITERAL_TRUTH = "literal_truth"
    TA_ACTION = "ta_action"
    LITERAL_CONDITION = "literal_condition"
    CLAUSE_OUTPUT = "clause_output"


class FixedBitBlock:
    """A mutable, word-packed bit block with an explicit semantic type."""

    __slots__ = ("bit_count", "semantic", "words")

    def __init__(self, bit_count: int, semantic: PortSemantic) -> None:
        if bit_count not in (1024, 4096):
            raise ValueError("PA bit blocks must contain 1024 or 4096 bits")
        self.bit_count = bit_count
        self.semantic = semantic
        self.words = [0] * (bit_count // 64)

    @classmethod
    def from_bools(
        cls,
        values: Iterable[bool],
        *,
        bit_count: int,
        semantic: PortSemantic,
    ) -> "FixedBitBlock":
        result = cls(bit_count, semantic)
        for index, value in enumerate(values):
            if index >= bit_count:
                raise ValueError("too many values for fixed bit block")
            result.set(index, value)
        return result

    def _check_index(self, index: int) -> None:
        if not 0 <= index < self.bit_count:
            raise IndexError(index)

    def get(self, index: int) -> bool:
        self._check_index(index)
        return bool((self.words[index // 64] >> (index % 64)) & 1)

    def set(self, index: int, value: bool) -> None:
        self._check_index(index)
        mask = 1 << (index % 64)
        word = index // 64
        if value:
            self.words[word] |= mask
        else:
            self.words[word] &= ~mask

    def population(self) -> int:
        return sum(word.bit_count() for word in self.words)

    def copy(self) -> "FixedBitBlock":
        result = FixedBitBlock(self.bit_count, self.semantic)
        result.words[:] = self.words
        return result


@dataclass(frozen=True, slots=True)
class PAResult:
    value: bool
    matched_count: int
    selected_count: int
    matched_words: tuple[int, ...]
    missing_words: tuple[int, ...]


class MaskedThresholdKernel:
    """Reference kernel for conjunction, disjunction, and k-of-n rules."""

    __slots__ = ("selection", "minimum_true")

    def __init__(self, selection: FixedBitBlock, minimum_true: int) -> None:
        if minimum_true < 0:
            raise ValueError("minimum_true cannot be negative")
        selected = selection.population()
        if minimum_true > selected:
            raise ValueError("minimum_true cannot exceed selected slot count")
        self.selection = selection.copy()
        self.minimum_true = minimum_true

    def evaluate(self, inputs: FixedBitBlock) -> PAResult:
        if inputs.bit_count != self.selection.bit_count:
            raise ValueError("input and selection shapes differ")
        matched_words = tuple(
            input_word & select_word
            for input_word, select_word in zip(inputs.words, self.selection.words)
        )
        missing_words = tuple(
            (~input_word) & select_word & ((1 << 64) - 1)
            for input_word, select_word in zip(inputs.words, self.selection.words)
        )
        matched = sum(word.bit_count() for word in matched_words)
        selected = self.selection.population()
        return PAResult(
            value=matched >= self.minimum_true,
            matched_count=matched,
            selected_count=selected,
            matched_words=matched_words,
            missing_words=missing_words,
        )

