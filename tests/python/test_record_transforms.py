from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given, settings, strategies as st
import pytest

from prolog_tsetlin import (
    AggregateOperation,
    AggregateTransform,
    RecordTransformPipeline,
    RegexOperation,
    RegexTransform,
    RelationalOperation,
    RelationalTransform,
    SequenceOperation,
    SequenceTransform,
    TemporalOperation,
    TemporalTransform,
    TransformError,
)


def test_pipeline_runs_every_transform_family_and_round_trips() -> None:
    pipeline = RecordTransformPipeline(
        (
            RegexTransform(
                "message", "error_count", r"error", RegexOperation.COUNT,
                ignore_case=True,
            ),
            AggregateTransform(
                ("samples",), "sample_mean", AggregateOperation.MEAN
            ),
            RelationalTransform(
                "lower", "upper", "ordered", RelationalOperation.LE
            ),
            SequenceTransform(
                "tokens", "starts_ready", SequenceOperation.STARTS_WITH,
                argument=("ready",),
            ),
            TemporalTransform(
                "event_time", "event_year", TemporalOperation.YEAR
            ),
        )
    )
    record = {
        "message": "ERROR then error",
        "samples": [1, 2, 6],
        "lower": 2,
        "upper": 3,
        "tokens": ("ready", "set"),
        "event_time": "2026-08-18T01:30:00-05:00",
    }

    result = pipeline.transform(record)

    assert result["error_count"] == 2
    assert result["sample_mean"] == 3
    assert result["ordered"] is True
    assert result["starts_ready"] is True
    assert result["event_year"] == 2026
    restored = RecordTransformPipeline.from_dict(pipeline.to_dict())
    assert restored == pipeline
    assert restored.pipeline_id == pipeline.pipeline_id
    assert record == {
        "message": "ERROR then error",
        "samples": [1, 2, 6],
        "lower": 2,
        "upper": 3,
        "tokens": ("ready", "set"),
        "event_time": "2026-08-18T01:30:00-05:00",
    }


def test_regex_operations_are_bounded_and_typed() -> None:
    record = {"text": "abc 123 abc"}
    assert RegexTransform("text", "x", r"\d+", RegexOperation.SEARCH).evaluate(record)
    assert RegexTransform("text", "x", r"[a-z ]+", RegexOperation.FULLMATCH).evaluate(record) is False
    assert RegexTransform("text", "x", r"(\d+)", RegexOperation.EXTRACT, group=1).evaluate(record) == "123"
    assert RegexTransform("text", "x", r"abc", RegexOperation.COUNT).evaluate(record) == 2
    with pytest.raises(TransformError, match="input ceiling"):
        RegexTransform("text", "x", r"a", max_input_chars=2).evaluate(record)


def test_aggregate_null_boolean_and_numeric_rules() -> None:
    record = {"values": [1, None, 3], "flags": [True, False, True]}
    assert AggregateTransform(("values",), "x", AggregateOperation.SUM).evaluate(record) == 4
    assert AggregateTransform(("values",), "x", AggregateOperation.COUNT).evaluate(record) == 2
    assert AggregateTransform(("flags",), "x", AggregateOperation.ANY).evaluate(record) is True
    with pytest.raises(TransformError, match="null"):
        AggregateTransform(
            ("values",), "x", AggregateOperation.SUM, skip_nulls=False
        ).evaluate(record)
    with pytest.raises(TransformError, match="Boolean"):
        AggregateTransform(("values",), "x", AggregateOperation.ALL).evaluate(record)


def test_relational_comparisons_are_type_strict_and_have_missing_policy() -> None:
    equal = RelationalTransform("left", "right", "x")
    assert equal.evaluate({"left": 1, "right": 1}) is True
    assert equal.evaluate({"left": True, "right": 1}) is False
    assert RelationalTransform(
        "left", "right", "x", missing="true"
    ).evaluate({"left": 1}) is True
    with pytest.raises(TransformError, match="same type"):
        RelationalTransform(
            "left", "right", "x", RelationalOperation.LT
        ).evaluate({"left": 1, "right": 2.0})


@pytest.mark.parametrize(
    ("operation", "kwargs", "expected"),
    (
        (SequenceOperation.LENGTH, {}, 4),
        (SequenceOperation.UNIQUE_COUNT, {}, 3),
        (SequenceOperation.CONTAINS, {"argument": True}, True),
        (SequenceOperation.STARTS_WITH, {"argument": (1, True)}, True),
        (SequenceOperation.ENDS_WITH, {"argument": ("x", 1)}, True),
        (SequenceOperation.ITEM, {"index": -2}, "x"),
    ),
)
def test_sequence_operations_preserve_typed_values(
    operation: SequenceOperation, kwargs: dict[str, object], expected: object
) -> None:
    transform = SequenceTransform("items", "x", operation, **kwargs)
    assert transform.evaluate({"items": [1, True, "x", 1]}) == expected
    assert SequenceTransform(
        "items", "x", SequenceOperation.CONTAINS, argument=1
    ).evaluate({"items": [True]}) is False


def test_temporal_operations_normalize_to_utc_and_require_timezone() -> None:
    record = {
        "start": "2026-08-18T00:00:00Z",
        "event": "2026-08-17T20:00:30-04:00",
    }
    assert TemporalTransform(
        "event", "x", TemporalOperation.HOUR
    ).evaluate(record) == 0
    assert TemporalTransform(
        "event", "x", TemporalOperation.SECONDS_SINCE,
        reference_field="start",
    ).evaluate(record) == 30
    assert TemporalTransform(
        "event", "x", TemporalOperation.WITHIN_SECONDS,
        reference_field="start", window_seconds=30,
    ).evaluate(record) is True
    assert TemporalTransform(
        "event", "x", TemporalOperation.EPOCH_SECONDS
    ).evaluate({"event": datetime(1970, 1, 1, tzinfo=timezone.utc)}) == 0
    with pytest.raises(TransformError, match="timezone"):
        TemporalTransform("event", "x", TemporalOperation.YEAR).evaluate(
            {"event": "2026-08-18T00:00:00"}
        )


def test_pipeline_rejects_output_collisions_and_noncanonical_descriptors() -> None:
    pipeline = RecordTransformPipeline(
        (RegexTransform("message", "matched", "ok"),)
    )
    with pytest.raises(TransformError, match="already contains"):
        pipeline.transform({"message": "ok", "matched": False})
    document = pipeline.to_dict()
    document["extra"] = True
    with pytest.raises(ValueError, match="not canonical"):
        RecordTransformPipeline.from_dict(document)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(st.integers(), st.integers())
def test_relational_integer_order_matches_python(left: int, right: int) -> None:
    transform = RelationalTransform(
        "left", "right", "ordered", RelationalOperation.LE
    )
    assert transform.evaluate({"left": left, "right": right}) is (left <= right)
