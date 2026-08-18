"""Validation for the deliberately small configurable-regex language."""

from __future__ import annotations

import re


def compile_safe(pattern: str, flags: int = 0) -> re.Pattern[str]:
    """Compile *pattern* after excluding backtracking-amplifying constructs.

    Python does not expose its parser publicly.  ``re._parser`` is nevertheless
    preferable to attempting to parse escapes and character classes ourselves;
    this module is the single compatibility boundary for that use.
    """

    try:
        parsed = re._parser.parse(pattern, flags)  # type: ignore[attr-defined]
    except re.error:
        raise

    repeat_ops = {  # type: ignore[attr-defined]
        re._parser.MAX_REPEAT,
        re._parser.MIN_REPEAT,
    }
    unsupported = {
        re._parser.ASSERT,  # type: ignore[attr-defined]
        re._parser.ASSERT_NOT,  # type: ignore[attr-defined]
        re._parser.GROUPREF,  # type: ignore[attr-defined]
        re._parser.GROUPREF_EXISTS,  # type: ignore[attr-defined]
    }
    simple = {
        re._parser.LITERAL,  # type: ignore[attr-defined]
        re._parser.NOT_LITERAL,  # type: ignore[attr-defined]
        re._parser.ANY,  # type: ignore[attr-defined]
        re._parser.IN,  # type: ignore[attr-defined]
        re._parser.AT,  # type: ignore[attr-defined]
        re._parser.CATEGORY,  # type: ignore[attr-defined]
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
            elif operation is re._parser.SUBPATTERN:  # type: ignore[attr-defined]
                visit(argument[-1], repeated=repeated)
            elif operation is re._parser.BRANCH:  # type: ignore[attr-defined]
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
