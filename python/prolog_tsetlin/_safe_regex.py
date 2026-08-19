"""Validation for the deliberately small configurable-regex language."""

from __future__ import annotations

import re


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
    """Compile *pattern* after excluding backtracking-amplifying constructs.

    Python does not expose its parser publicly.  ``re._parser`` is nevertheless
    preferable to attempting to parse escapes and character classes ourselves;
    this module is the single compatibility boundary for that use.
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

    def visit(nodes: object, *, repeated: bool = False) -> None:
        for operation, argument in nodes:  # type: ignore[union-attr]
            if operation in unsupported:
                raise ValueError(
                    "lookaround, backreferences, and conditionals are unsupported"
                )
            if operation in repeat_ops:
                if repeated:
                    raise ValueError("nested quantifiers are unsupported")
                _minimum, _maximum, child = argument
                visit(child, repeated=True)
            elif operation is _PARSER.SUBPATTERN:  # type: ignore[union-attr]
                visit(argument[-1], repeated=repeated)
            elif operation is _PARSER.BRANCH:  # type: ignore[union-attr]
                if repeated:
                    raise ValueError("quantified alternation is unsupported")
                for branch in argument[1]:
                    visit(branch, repeated=False)
            elif operation not in simple:
                # This deny-by-default rule also makes new CPython parser
                # operations unsupported until their complexity is reviewed.
                raise ValueError("regex construct is unsupported")

    visit(parsed)
    return re.compile(pattern, flags)
