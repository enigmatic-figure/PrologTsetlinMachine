from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prolog_tsetlin import (
    LogicDataset,
    LogicEncoder,
    LogicEncoding,
    LogicProblem,
    collision_report,
    encode_logic_split,
    evaluation_signature_report,
    load_logic_dataset,
    stratified_logic_split,
)


def problem(row_id: int, tokens: tuple[str, ...], target: int) -> LogicProblem:
    return LogicProblem(
        row_id=row_id,
        natural_problem=f"problem-{row_id}",
        symbolic_source="fixture",
        expression_tokens=tokens,
        bindings=(False, True, False, True, False),
        target=target,
    )


class LogicDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problems = (
            problem(0, ("A", "&", "-", "B"), 1),
            problem(1, ("-", "A", "&", "B"), 0),
            problem(2, ("A", "&", "B"), 1),
            problem(3, ("A", "x", "B"), 0),
        )
        self.dataset = LogicDataset(self.problems, "sha256:" + "0" * 64)

    def test_paired_csv_loader_skips_symbolic_blank_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            natural = root / "natural.csv"
            symbolic = root / "symbolic.csv"
            with natural.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Problem", "Solution"))
                writer.writerow(("first", "False"))
                writer.writerow(("second", "True"))
            with symbolic.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(())
                writer.writerow(
                    ("'A&-B'/{'A':0,'B':1,'C':0,'D':1,'E':0}", "0")
                )
                writer.writerow(
                    ("'-A&B'/{'A':0,'B':1,'C':0,'D':1,'E':0}", "1")
                )
            loaded = load_logic_dataset(natural, symbolic)

        self.assertEqual(len(loaded.problems), 2)
        self.assertEqual(loaded.problems[0].expression_tokens, ("A", "&", "-", "B"))
        self.assertEqual(loaded.problems[0].bindings, (False, True, False, True, False))
        self.assertTrue(loaded.source_digest.startswith("sha256:"))

    def test_encodings_expose_collision_ceiling(self) -> None:
        reports = {}
        for encoding in LogicEncoding:
            encoder = LogicEncoder.fit(encoding, self.problems)
            encoded = encoder.encode(self.dataset, tuple(range(4)))
            reports[encoding] = collision_report(encoded.rows, encoded.targets)

        self.assertEqual(
            reports[LogicEncoding.TOKEN_PRESENCE].optimistic_ceiling, 0.75
        )
        self.assertEqual(
            reports[LogicEncoding.TOKEN_COUNT_THRESHOLD].optimistic_ceiling,
            0.75,
        )
        self.assertEqual(
            reports[LogicEncoding.POSITION_ONE_HOT].optimistic_ceiling, 1.0
        )
        self.assertEqual(
            reports[LogicEncoding.AST_RELATIONAL].optimistic_ceiling, 1.0
        )
        position = LogicEncoder.fit(LogicEncoding.POSITION_ONE_HOT, self.problems)
        self.assertEqual(len(position.literals), 4 * 13 + 5)
        self.assertEqual(
            position.literals[0].literal_id,
            LogicEncoder.fit(
                LogicEncoding.POSITION_ONE_HOT, reversed(self.problems)
            ).literals[0].literal_id,
        )
        self.assertEqual(
            len({literal.literal_id for literal in position.literals}),
            len(position.literals),
        )

        relational = LogicEncoder.fit(LogicEncoding.AST_RELATIONAL, self.problems)
        relational_rows = relational.encode(self.dataset, tuple(range(4)))
        self.assertGreater(len(relational.literals), 5)
        self.assertEqual(relational_rows.truncated_rows, 0)

    def test_split_is_stratified_and_reproducible(self) -> None:
        first = stratified_logic_split(
            self.dataset, evaluation_fraction=0.5, seed=91
        )
        second = stratified_logic_split(
            self.dataset, evaluation_fraction=0.5, seed=91
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.train_indices), 2)
        self.assertEqual(len(first.evaluation_indices), 2)
        self.assertEqual(
            {self.problems[index].target for index in first.train_indices}, {0, 1}
        )
        encoded = encode_logic_split(
            self.dataset, first, LogicEncoding.TOKEN_PRESENCE
        )
        self.assertEqual(encoded.feature_count, 18)

    def test_evaluation_signature_report_separates_unseen_rows(self) -> None:
        report = evaluation_signature_report(
            ((0, 0), (0, 0), (1, 0)),
            (0, 1, 1),
            ((0, 0), (1, 1)),
            (1, 0),
        )
        self.assertEqual(report.seen_rows, 1)
        self.assertEqual(report.unseen_rows, 1)
        self.assertEqual(report.conflicting_seen_rows, 1)
        self.assertEqual(report.majority_disagreement_rows, 1)


if __name__ == "__main__":
    unittest.main()
