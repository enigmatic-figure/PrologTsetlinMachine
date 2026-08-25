from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from prolog_tsetlin.benchmark_campaign import (
    BenchmarkCampaignError,
    CampaignDatasetManifest,
    CampaignRunRequest,
    import_dense_bit_dataset,
    load_dense_bit_split,
    prepare_parity_ladder,
    prepare_xor_noise,
    run_campaign_attempt,
)


def _load(path: Path) -> CampaignDatasetManifest:
    return CampaignDatasetManifest.load(path)


def _request(
    manifest_path: Path,
    output_directory: Path,
    *,
    run_id: str,
    implementation: str = "ptm.scalar-reference",
) -> CampaignRunRequest:
    manifest = _load(manifest_path)
    return CampaignRunRequest(
        campaign_id="local-smoke-v1",
        run_id=run_id,
        pass_name="smoke",
        track="shared",
        dataset_manifest=str(manifest_path.resolve()),
        dataset_manifest_digest=manifest.manifest_digest,
        train_split="train",
        score_splits=("evaluation", "validation"),
        model={
            "implementation": implementation,
            "backend": "python-scalar-reference",
            "commit": "test",
            "config": {
                "clauses": 4,
                "states_per_action": 10,
                "specificity": 3,
                "threshold": 5,
                "epochs": 1,
                "seed": 7,
                "inference_repeats": 2,
                "inference_warmup_repeats": 1,
            },
        },
        output_directory=str(output_directory.resolve()),
    )


def test_xor_noise_material_is_deterministic_and_changes_only_training_labels(
    tmp_path: Path,
) -> None:
    arguments = {
        "seed": 17,
        "feature_count": 20,
        "train_rows": 25,
        "validation_rows": 9,
        "evaluation_rows": 11,
        "noise_basis_points": (0, 2_000),
    }
    first_paths = prepare_xor_noise(tmp_path / "first", **arguments)
    second_paths = prepare_xor_noise(tmp_path / "second", **arguments)

    for first_path, second_path in zip(first_paths, second_paths):
        assert first_path.read_bytes() == second_path.read_bytes()
        first = _load(first_path)
        second = _load(second_path)
        assert first.manifest_digest == second.manifest_digest
        for split_name in first.split_map:
            assert (
                first_path.parent / first.split_map[split_name].path
            ).read_bytes() == (
                second_path.parent / second.split_map[split_name].path
            ).read_bytes()

    clean = _load(first_paths[0])
    noisy = _load(first_paths[1])
    assert clean.representation_digest == noisy.representation_digest
    assert clean.manifest_digest != noisy.manifest_digest
    assert (
        clean.split_map["evaluation"].label_digest
        == noisy.split_map["evaluation"].label_digest
    )
    assert (
        clean.split_map["validation"].label_digest
        == noisy.split_map["validation"].label_digest
    )
    assert clean.split_map["train"].label_digest != noisy.split_map["train"].label_digest

    clean_rows, clean_labels = load_dense_bit_split(first_paths[0], clean, "train")
    noisy_rows, noisy_labels = load_dense_bit_split(first_paths[1], noisy, "train")
    assert clean_rows == noisy_rows
    assert sum(left != right for left, right in zip(clean_labels, noisy_labels)) == 5
    assert all(label == (row[0] ^ row[1]) for row, label in zip(clean_rows, clean_labels))
    seen = set(clean_rows)
    for split_name in ("validation", "evaluation"):
        rows, _ = load_dense_bit_split(first_paths[0], clean, split_name)
        assert not seen.intersection(rows)
        seen.update(rows)
    assert len(seen) == sum(
        arguments[name]
        for name in ("train_rows", "validation_rows", "evaluation_rows")
    )


def test_parity_ladder_exhausts_each_domain_without_split_overlap(
    tmp_path: Path,
) -> None:
    first_paths = prepare_parity_ladder(tmp_path / "first", seed=23, widths=(3, 4))
    second_paths = prepare_parity_ladder(tmp_path / "second", seed=23, widths=(3, 4))

    for width, first_path, second_path in zip((3, 4), first_paths, second_paths):
        assert first_path.read_bytes() == second_path.read_bytes()
        manifest = _load(first_path)
        seen: set[tuple[int, ...]] = set()
        for split_name in ("train", "validation", "evaluation"):
            rows, labels = load_dense_bit_split(first_path, manifest, split_name)
            assert not seen.intersection(rows)
            seen.update(rows)
            assert all(label == (sum(row) & 1) for row, label in zip(rows, labels))
        assert len(seen) == 1 << width


def test_dataset_manifest_detects_split_tampering(tmp_path: Path) -> None:
    manifest_path = prepare_parity_ladder(
        tmp_path / "parity", seed=29, widths=(3,)
    )[0]
    manifest = _load(manifest_path)
    train_path = manifest_path.parent / manifest.split_map["train"].path
    train_path.write_bytes(train_path.read_bytes() + b"0 0 0 0\n")

    with pytest.raises(BenchmarkCampaignError, match="file digest mismatch"):
        CampaignDatasetManifest.load(manifest_path)


def test_dense_import_preserves_exact_archived_bytes_and_receipts_them(
    tmp_path: Path,
) -> None:
    train = tmp_path / "source-train.txt"
    evaluation = tmp_path / "source-evaluation.txt"
    train.write_bytes(b"0 1 1\r\n1 0 1\r\n")
    evaluation.write_bytes(b"0 0 0\r\n1 1 0\r\n")

    manifest_path = import_dense_bit_dataset(
        tmp_path / "imported",
        dataset_id="test.archived.v1",
        variant_id="crlf",
        representation_id="boolean-2",
        feature_count=2,
        split_paths={"evaluation": evaluation, "train": train},
        source={"kind": "test-archive"},
    )
    manifest = _load(manifest_path)

    assert (manifest_path.parent / "train.txt").read_bytes() == train.read_bytes()
    assert (
        manifest_path.parent / "evaluation.txt"
    ).read_bytes() == evaluation.read_bytes()
    rows, labels = load_dense_bit_split(manifest_path, manifest, "train")
    assert rows == ((0, 1), (1, 0))
    assert labels == (1, 1)
    assert manifest.source["files"]["train"] == manifest.split_map["train"].file_digest


def test_scalar_wrapper_attempt_is_independently_scored_and_appended(
    tmp_path: Path,
) -> None:
    manifest_path = prepare_parity_ladder(
        tmp_path / "materials", seed=31, widths=(3,)
    )[0]
    request = _request(
        manifest_path,
        tmp_path / "run-ok",
        run_id="ptm-scalar-ok",
    )
    raw_jsonl = tmp_path / "raw.jsonl"

    record = run_campaign_attempt(
        request,
        [sys.executable, "-m", "prolog_tsetlin.benchmark_campaign", "wrapper-ptm-scalar"],
        raw_jsonl=raw_jsonl,
        timeout_seconds=30,
    )

    assert record["status"] == "ok"
    assert set(record["metrics"]) == {"evaluation", "validation"}
    assert isinstance(record["timing"]["preprocessing_materialization_s"], float)
    assert isinstance(record["timing"]["adaptive_training_s"], float)
    assert len(record["timing"]["resident_inference_samples_s"]["evaluation"]) == 2
    assert record["timing"]["pta_lifecycle_episode_s"] == "n/a"
    persisted = [json.loads(line) for line in raw_jsonl.read_text(encoding="utf-8").splitlines()]
    assert persisted == [record]
    assert (tmp_path / "run-ok" / "predictions-evaluation.txt").is_file()
    assert record["artifacts"]["stdout_digest"].startswith("sha256:")
    assert record["artifacts"]["stderr_digest"].startswith("sha256:")


def test_failed_and_unsupported_attempts_are_both_retained(tmp_path: Path) -> None:
    manifest_path = prepare_parity_ladder(
        tmp_path / "materials", seed=37, widths=(3,)
    )[0]
    raw_jsonl = tmp_path / "raw.jsonl"

    failed = run_campaign_attempt(
        _request(manifest_path, tmp_path / "run-failed", run_id="failed"),
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('intentional failure'); raise SystemExit(7)",
        ],
        raw_jsonl=raw_jsonl,
        timeout_seconds=30,
    )
    unsupported = run_campaign_attempt(
        _request(
            manifest_path,
            tmp_path / "run-unsupported",
            run_id="unsupported",
            implementation="some.other.model",
        ),
        [sys.executable, "-m", "prolog_tsetlin.benchmark_campaign", "wrapper-ptm-scalar"],
        raw_jsonl=raw_jsonl,
        timeout_seconds=30,
    )

    assert failed["status"] == "failed"
    assert failed["failure"]["class"] == "nonzero_exit"
    assert failed["return_code"] == 7
    assert (tmp_path / "run-failed" / "stderr.bin").read_bytes() == b"intentional failure"
    assert unsupported["status"] == "unsupported"
    assert unsupported["failure"]["class"] == "unsupported"
    persisted = [json.loads(line) for line in raw_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [item["status"] for item in persisted] == ["failed", "unsupported"]
