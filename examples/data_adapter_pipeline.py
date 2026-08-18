"""Stream records through token and typed record transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prolog_tsetlin import (
    AggregateOperation,
    AggregateTransform,
    RecordTransformPipeline,
    RegexTransform,
    RelationalOperation,
    RelationalTransform,
    SequenceOperation,
    SequenceTransform,
    TemporalOperation,
    TemporalTransform,
    TokenAdapter,
    iter_parquet_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=Path,
        help="optional Parquet dataset with the same fields as the built-in rows",
    )
    arguments = parser.parse_args()

    records = (
        {
            "message": "Alert: service ready",
            "samples": [10, 12, 11],
            "lower": 10,
            "upper": 12,
            "event_time": "2026-08-18T09:30:00-05:00",
        },
        {
            "message": "Service quiet",
            "samples": [4, 5],
            "lower": 4,
            "upper": 5,
            "event_time": "2026-08-19T01:00:00Z",
        },
    )
    source = (
        iter_parquet_records(arguments.parquet, batch_size=64)
        if arguments.parquet
        else iter(records)
    )
    tokens = TokenAdapter("message", output_field="tokens")
    pipeline = RecordTransformPipeline(
        (
            RegexTransform("message", "has_alert", r"\balert\b", ignore_case=True),
            AggregateTransform(
                ("samples",), "sample_mean", AggregateOperation.MEAN
            ),
            RelationalTransform(
                "lower", "upper", "bounds_valid", RelationalOperation.LE
            ),
            SequenceTransform(
                "tokens", "has_ready", SequenceOperation.CONTAINS,
                argument="ready",
            ),
            TemporalTransform(
                "event_time", "event_year", TemporalOperation.YEAR
            ),
        )
    )
    print(json.dumps({"pipeline_id": pipeline.pipeline_id}, indent=2))
    for record in pipeline.iter_transform(tokens.iter_adapt(source)):
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
