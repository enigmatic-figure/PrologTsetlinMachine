"""Validation for the deliberately small configurable-regex language."""

from __future__ import annotations

import re


UNICODE_WORD_TOKEN_PATTERN = r"[^\W_]+['’][^\W_]+|[^\W_]+"
_MAX_FINITE_REPEAT = 64
_MAX_BACKTRACK_STATES = 256


def _parser_module() -> object:
    """Return the regex parser module for this interpreter.

    ``re._parser`` exists on Python 3.11+; on 3.10 the same implementation
    lives as the top-level ``sre_parse`` module.
    """

    parser = getattr(re, "_parser", None)
    if parser is not None:
        return parser
    import sre_parse  # type: ignore[import-not-found]

    return sre_parse


_PARSER = _parser_module()


def compile_safe(pattern: str, flags: int = 0) -> re.Pattern[str]:
    """Compile *pattern* from PTM's deliberately restricted regex subset.

    Python does not expose its parser publicly.  ``re._parser`` is nevertheless
    preferable to attempting to parse escapes and character classes ourselves;
    this module is the single compatibility boundary for that use.

    General branches may contain only one terminal unbounded repetition.
    Finite repetitions are capped and have a bounded aggregate choice budget.
    PTM's built-in Unicode word tokenizer is separately admitted as an audited
    delimiter-separated language whose fallback branch consumes each word run.
    """

    try:
        parsed = _PARSER.parse(pattern, flags)  # type: ignore[union-attr]
    except re.error:
        raise

    repeat_ops = {  # type: ignore[attr-defined]
        _PARSER.MAX_REPEAT,  # type: ignore[union-attr]
        _PARSER.MIN_REPEAT,  # type: ignore[union-attr]
    }
    unsupported = {
        _PARSER.ASSERT,  # type: ignore[union-attr]
        _PARSER.ASSERT_NOT,  # type: ignore[union-attr]
        _PARSER.GROUPREF,  # type: ignore[union-attr]
        _PARSER.GROUPREF_EXISTS,  # type: ignore[union-attr]
    }
    simple = {
        _PARSER.LITERAL,  # type: ignore[union-attr]
        _PARSER.NOT_LITERAL,  # type: ignore[union-attr]
        _PARSER.ANY,  # type: ignore[union-attr]
        _PARSER.IN,  # type: ignore[union-attr]
        _PARSER.AT,  # type: ignore[union-attr]
        _PARSER.CATEGORY,  # type: ignore[union-attr]
    }

    audited_tokenizer = pattern == UNICODE_WORD_TOKEN_PATTERN

    def flatten(nodes: object) -> list[tuple[object, object]]:
        flattened: list[tuple[object, object]] = []
        for operation, argument in nodes:  # type: ignore[union-attr]
            if operation is _PARSER.SUBPATTERN:  # type: ignore[union-attr]
                flattened.extend(flatten(argument[-1]))
            else:
                flattened.append((operation, argument))
        return flattened

    def visit(nodes: object, *, repeated: bool = False) -> None:
        flattened = flatten(nodes)
        branches = [
            (index, argument)
            for index, (operation, argument) in enumerate(flattened)
            if operation is _PARSER.BRANCH  # type: ignore[union-attr]
        ]
        if branches:
            if repeated:
                raise ValueError("quantified alternation is unsupported")
            if len(flattened) != 1 or len(branches) != 1:
                raise ValueError("alternation is supported only at the top level")
            for branch in branches[0][1][1]:
                visit(branch, repeated=False)
            return

        repeat_positions = [
            index
            for index, (operation, _argument) in enumerate(flattened)
            if operation in repeat_ops
        ]
        if repeated and repeat_positions:
            raise ValueError("nested quantifiers are unsupported")
        if any(
            right == left + 1
            for left, right in zip(repeat_positions, repeat_positions[1:])
        ):
            raise ValueError("adjacent quantifiers are unsupported")

        finite_choice_budget = 1
        for index, (operation, argument) in enumerate(flattened):
            if operation in unsupported:
                raise ValueError(
                    "lookaround, backreferences, and conditionals are unsupported"
                )
            if operation in repeat_ops:
                minimum, maximum, child = argument
                visit(child, repeated=True)
                if child.getwidth() != (1, 1):
                    raise ValueError(
                        "quantifiers may repeat only one-character atoms"
                    )
                if maximum == _PARSER.MAXREPEAT:  # type: ignore[union-attr]
                    if not audited_tokenizer and index != len(flattened) - 1:
                        raise ValueError(
                            "an unbounded quantifier must terminate its branch"
                        )
                else:
                    if maximum > _MAX_FINITE_REPEAT:
                        raise ValueError(
                            f"finite quantifiers may repeat at most {_MAX_FINITE_REPEAT} times"
                        )
                    finite_choice_budget *= maximum - minimum + 1
                    if finite_choice_budget > _MAX_BACKTRACK_STATES:
                        raise ValueError(
                            "finite quantifiers exceed the branch choice budget"
                        )
            elif operation not in simple:
                # This deny-by-default rule also makes new CPython parser
                # operations unsupported until their complexity is reviewed.
                raise ValueError("regex construct is unsupported")

    visit(parsed)
    return re.compile(pattern, flags)
