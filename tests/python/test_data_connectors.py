from __future__ import annotations

from pathlib import Path

import pytest

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    ImageAdapter,
    LiteralCatalog,
    PreprocessingContract,
    RecordTransformPipeline,
    RegexTransform,
    SequenceOperation,
    SequenceTransform,
    TokenAdapter,
    iter_arrow_records,
    iter_parquet_records,
)


def test_token_adapter_is_versioned_normalized_and_bounded() -> None:
    adapter = TokenAdapter("message", output_field="words", max_tokens=4)
    result = adapter.adapt({"message": "ＣＡＦÉ café Don't"})
    assert result["words"] == ("café", "café", "don't")
    assert result["words__count"] == 3
    assert TokenAdapter.from_dict(adapter.to_dict()) == adapter
    assert TokenAdapter.from_dict(adapter.to_dict()).adapter_id == adapter.adapter_id

    with pytest.raises(ValueError, match="descriptor is invalid"):
        TokenAdapter.from_dict({**adapter.to_dict(), "unicode_version": "0.0"})
    with pytest.raises(ValueError, match="adapter ceiling"):
        TokenAdapter("message", max_tokens=1).adapt({"message": "one two"})
    assert TokenAdapter(
        "message", max_tokens=1, overflow="truncate"
    ).tokenize("one two") == ("one",)


@pytest.mark.parametrize(
    ("hostile", "message"),
    [
        (r"(a+)+$", "nested quantifiers"),
        (r"a*a*a*a*b", "adjacent quantifiers"),
        (r"a+b", "must terminate its branch"),
    ],
)
def test_token_adapter_rejects_backtracking_amplification(
    hostile: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TokenAdapter("message", pattern=hostile)
    descriptor = {**TokenAdapter("message").to_dict(), "pattern": hostile}
    with pytest.raises(ValueError, match="descriptor is invalid"):
        TokenAdapter.from_dict(descriptor)


def test_token_adapter_accepts_audited_and_bounded_quantified_patterns() -> None:
    assert TokenAdapter("message").tokenize("alpha don't beta") == (
        "alpha",
        "don't",
        "beta",
    )
    bounded = TokenAdapter("message", pattern=r"[0-9]{1,4}|[a-z]+")
    assert bounded.tokenize("abc 12345 67") == ("abc", "1234", "5", "67")


def test_image_adapter_materializes_stable_scalar_pixel_fields() -> None:
    adapter = ImageAdapter(2, 2, output_prefix="glyph")
    result = adapter.adapt_pixels([0, 64, 128, 255], record={"label": "x"})
    assert result == {
        "label": "x",
        "glyph__width": 2,
        "glyph__height": 2,
        "glyph__mode": "L",
        "glyph_pixel_0000": 0,
        "glyph_pixel_0001": 64,
        "glyph_pixel_0002": 128,
        "glyph_pixel_0003": 255,
    }
    assert ImageAdapter.from_dict(adapter.to_dict()) == adapter
    assert ImageAdapter.from_dict(adapter.to_dict()).adapter_id == adapter.adapter_id


def test_arrow_stream_composes_with_tokens_pipeline_and_preprocessing() -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table(
        {
            "message": ["alert ready", "quiet", "unknown"],
            "expected": [True, False, False],
        }
    )
    adapter = TokenAdapter("message", output_field="words")
    pipeline = RecordTransformPipeline(
        (
            RegexTransform("message", "has_alert", r"\balert\b"),
            SequenceTransform(
                "words", "has_ready", SequenceOperation.CONTAINS,
                argument="ready",
            ),
        )
    )
    records = iter_arrow_records(table, batch_size=1)
    adapted = adapter.iter_adapt(records)
    transformed = tuple(pipeline.iter_transform(adapted))
    assert [row["has_alert"] for row in transformed[:2]] == [True, False]
    assert transformed[0]["has_ready"] is True

    catalog = LiteralCatalog(
        FeatureSchema.from_fields(
            has_alert=FieldKind.BOOLEAN,
            has_ready=FieldKind.BOOLEAN,
        )
    )
    catalog.category_eq("has_alert", True)
    catalog.category_eq("has_ready", True)
    preprocessing = PreprocessingContract.from_catalog(catalog)
    assert preprocessing.materialize(transformed[0]) == (True, True)
    assert preprocessing.materialize(transformed[1]) == (False, False)


def test_arrow_reader_and_column_projection_are_streaming() -> None:
    pa = pytest.importorskip("pyarrow")
    schema = pa.schema((pa.field("keep", pa.int64()), pa.field("drop", pa.string())))
    batches = (
        pa.RecordBatch.from_arrays(
            [pa.array([1, 2]), pa.array(["a", "b"])], schema=schema
        ),
        pa.RecordBatch.from_arrays(
            [pa.array([3]), pa.array(["c"])], schema=schema
        ),
    )
    reader = pa.RecordBatchReader.from_batches(schema, batches)
    assert list(iter_arrow_records(reader, columns=("keep",), batch_size=1)) == [
        {"keep": 1},
        {"keep": 2},
        {"keep": 3},
    ]


def test_parquet_dataset_streams_files_and_projection(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.table({"value": [1, 2, 3, 4, 5], "name": list("abcde")}),
        path,
        row_group_size=2,
    )
    records = list(
        iter_parquet_records(path, columns=("value",), batch_size=2)
    )
    assert records == [
        {"value": 1},
        {"value": 2},
        {"value": 3},
        {"value": 4},
        {"value": 5},
    ]


def test_pillow_image_path_adapter_resizes_with_nearest(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "pixels.png"
    image = Image.new("L", (2, 2))
    if hasattr(image, "putdata"):
        image.putdata([7, 7, 7, 7])
    image.save(path)
    adapter = ImageAdapter(1, 1, output_prefix="sample")
    result = adapter.adapt_image(path)
    assert result["sample_pixel_0000"] == 7
    assert result["sample__width"] == 1
