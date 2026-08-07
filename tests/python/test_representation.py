from __future__ import annotations

import unittest

from prolog_tsetlin import FeatureSchema, FieldKind, LiteralCatalog, NullPolicy


class RepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = FeatureSchema.from_fields(
            age=FieldKind.NUMBER,
            color=FieldKind.CATEGORY,
            note=FieldKind.TEXT,
        )

    def test_dual_view_and_provenance(self) -> None:
        catalog = LiteralCatalog(self.schema)
        ge_18 = catalog.numeric_ge("age", 18)
        ge_25 = catalog.numeric_ge("age", 25)
        interval = catalog.numeric_between("age", 25, 40)
        blue = catalog.category_eq("color", "blue")
        token = catalog.token_contains("note", "urgent")
        missing = catalog.is_missing("age")

        batch = catalog.encode(
            [{"age": 37, "color": "blue", "note": "Very URGENT case"}],
            row_ids=["person-7"],
        )

        self.assertEqual(
            batch.ta.row_values(0),
            (True, True, True, True, True, False),
        )
        self.assertEqual(batch.raw_records[0]["age"], 37)
        self.assertEqual(batch.symbolic[0][0].predicate, "age")
        self.assertEqual(batch.symbolic[0][0].value, 37)
        self.assertEqual(batch.traces[0][1].literal_id, ge_25.literal_id)
        self.assertEqual(batch.traces[0][1].raw_value, 37)
        self.assertTrue(batch.traces[0][1].result)
        self.assertEqual(ge_18.source_field_id, interval.source_field_id)
        self.assertNotEqual(ge_18.literal_id, ge_25.literal_id)
        self.assertEqual(blue.source_field, "color")
        self.assertEqual(token.source_field, "note")
        self.assertEqual(missing.source_field, "age")

    def test_ids_are_deterministic_and_duplicates_are_interned(self) -> None:
        first = LiteralCatalog(self.schema)
        second = LiteralCatalog(self.schema)
        a = first.numeric_ge("age", 25)
        b = second.numeric_ge("age", 25)
        duplicate = first.numeric_ge("age", 25)
        self.assertEqual(a.literal_id, b.literal_id)
        self.assertIs(a, duplicate)
        self.assertEqual(len(first.literals), 1)

    def test_null_policies_are_explicit(self) -> None:
        catalog = LiteralCatalog(self.schema)
        catalog.numeric_ge("age", 18, null_policy=NullPolicy.TRUE)
        catalog.numeric_ge("age", 21, null_policy=NullPolicy.FALSE)
        catalog.is_missing("age")
        self.assertEqual(
            catalog.encode([{"age": None}]).ta.row_values(0),
            (True, False, True),
        )

        strict = LiteralCatalog(self.schema)
        strict.numeric_ge("age", 18, null_policy=NullPolicy.ERROR)
        with self.assertRaises(ValueError):
            strict.encode([{"age": None}])

    def test_more_than_one_machine_word_is_packed_correctly(self) -> None:
        catalog = LiteralCatalog(self.schema)
        for threshold in range(70):
            catalog.numeric_ge("age", threshold)
        batch = catalog.encode([{"age": 65}])
        self.assertEqual(len(batch.ta.words[0]), 2)
        self.assertTrue(batch.ta.bit(0, 65))
        self.assertFalse(batch.ta.bit(0, 66))

    def test_native_feature_major_pages_preserve_literal_bits(self) -> None:
        catalog = LiteralCatalog(self.schema)
        for threshold in range(70):
            catalog.numeric_ge("age", threshold)
        batch = catalog.encode([{"age": age} for age in range(65)])
        first, first_valid = batch.ta.feature_major_words64()
        self.assertEqual(first_valid, (1 << 64) - 1)
        self.assertEqual(first[0], (1 << 64) - 1)
        self.assertEqual(first[10], ((1 << 64) - 1) ^ ((1 << 10) - 1))
        self.assertEqual(first[69], 0)
        second, second_valid = batch.ta.feature_major_words64(64)
        self.assertEqual(second_valid, 1)
        self.assertEqual(second[64], 1)
        self.assertEqual(second[65], 0)
        with self.assertRaises(IndexError):
            batch.ta.feature_major_words64(65)


if __name__ == "__main__":
    unittest.main()
