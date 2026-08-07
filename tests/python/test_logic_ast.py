from __future__ import annotations

import itertools
import unittest

from prolog_tsetlin import (
    LogicASTKind,
    LogicChildRole,
    PrimitiveLogicOp,
    parse_logic_tokens,
)


class LogicASTTests(unittest.TestCase):
    def test_conditional_ast_and_primitive_lowering_are_equivalent(self) -> None:
        program = parse_logic_tokens(
            (
                "(",
                "A",
                "&",
                "-",
                "B",
                ")",
                "if",
                "(",
                "C",
                ")",
                "$",
                "(",
                "D",
                "x",
                "E",
                ")",
            )
        )
        self.assertEqual(program.nodes[program.root_id].kind, LogicASTKind.CONDITIONAL)
        self.assertEqual(
            program.nodes[program.root_id].child_roles,
            (
                LogicChildRole.CONDITION,
                LogicChildRole.TRUE_BRANCH,
                LogicChildRole.FALSE_BRANCH,
            ),
        )
        lowered = program.lower()
        for bindings in itertools.product((False, True), repeat=5):
            a, b, c, d, e = bindings
            expected = (a and not b) if c else (d or e)
            self.assertEqual(program.evaluate(bindings), expected)
            self.assertEqual(lowered.evaluate(bindings), expected)

    def test_chained_not_equal_uses_python_comparison_semantics(self) -> None:
        program = parse_logic_tokens(("D", "!=", "D", "!=", "A"))
        root = program.nodes[program.root_id]
        self.assertEqual(root.kind, LogicASTKind.NOT_EQUAL_CHAIN)
        self.assertEqual(len(root.children), 3)
        for bindings in itertools.product((False, True), repeat=5):
            self.assertFalse(program.evaluate(bindings))
            self.assertFalse(program.lower().evaluate(bindings))

    def test_double_negation_is_eliminated_by_primitive_graph(self) -> None:
        program = parse_logic_tokens(("-", "(", "-", "A", ")"))
        lowered = program.lower()
        self.assertEqual(lowered.nodes[lowered.root_id].operation, PrimitiveLogicOp.INPUT)
        self.assertEqual(lowered.nodes[lowered.root_id].variable, "A")

    def test_facts_retain_edges_depth_variables_and_bindings(self) -> None:
        program = parse_logic_tokens(("A", "&", "-", "B"))
        facts = program.facts((True, False, False, True, False))
        predicates = {fact.predicate for fact in facts}
        expected = {
            "root",
            "operator",
            "depth",
            "references",
            "child",
            "bound_value",
        }
        self.assertTrue(expected <= predicates)
        self.assertIn(
            ("bound_value", ("A", True)),
            {(fact.predicate, fact.arguments) for fact in facts},
        )

    def test_parser_rejects_unapproved_python_syntax(self) -> None:
        invalid_sources = (
            ("unknown",),
            ("True",),
            ("A", "+", "B"),
            ("f", "(", "A", ")"),
        )
        for tokens in invalid_sources:
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    parse_logic_tokens(tokens)


if __name__ == "__main__":
    unittest.main()
