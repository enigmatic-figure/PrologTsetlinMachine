#!/usr/bin/env python3
"""Run one plumbing-only shared-input attempt through PTM and both incumbents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from prolog_tsetlin.benchmark_campaign import (
    CampaignDatasetManifest,
    CampaignRunRequest,
    run_campaign_attempt,
)


PYTSETLIN_COMMIT = "d6c1cf0e4aaa4a8ae2f2818ba27878fb89d31dc5"
TMU_COMMIT = "5605ff070a18549328028c907a9acf68e063346e"


def _request(
    manifest_path: Path,
    output: Path,
    *,
    run_id: str,
    model: dict[str, object],
) -> CampaignRunRequest:
    manifest = CampaignDatasetManifest.load(manifest_path)
    return CampaignRunRequest(
        campaign_id="initial-capacity-plumbing-v1",
        run_id=run_id,
        pass_name="smoke",
        track="shared",
        dataset_manifest=str(manifest_path),
        dataset_manifest_digest=manifest.manifest_digest,
        train_split="train",
        score_splits=("evaluation", "validation"),
        model=model,
        output_directory=str(output / run_id),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--material-root", type=Path, required=True)
    parser.add_argument("--incumbent-root", type=Path, required=True)
    parser.add_argument("--ptm-native-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    project = arguments.project_root.resolve()
    materials = arguments.material_root.resolve()
    incumbents = arguments.incumbent_root.resolve()
    output = arguments.output.resolve()
    native_executable = arguments.ptm_native_executable.resolve()
    manifest_path = (
        materials / "parity-ladder" / "n-06" / "manifest.json"
    ).resolve()
    wrapper = (project / "benchmarks" / "initial_capacity" / "incumbent_wrapper.py").resolve()
    common = {
        "threshold": 15,
        "specificity": 3.9,
        "epochs": 1,
        "inference_repeats": 2,
        "inference_warmup_repeats": 1,
        "seed": 7,
        "boost_true_positive_feedback": 1,
        "weighted_clauses": False,
        "feature_negation": True,
        "max_included_literals": None,
    }
    attempts = (
        (
            "ptm-scalar-plumbing",
            {
                "implementation": "ptm.scalar-reference",
                "backend": "python-scalar-reference",
                "commit": "working-tree",
                "config": {
                    "clauses": 20,
                    "states_per_action": 100,
                    "specificity": 3.9,
                    "threshold": 15,
                    "epochs": 1,
                    "seed": 7,
                    "inference_repeats": 2,
                    "inference_warmup_repeats": 1,
                },
            },
            [sys.executable, "-m", "prolog_tsetlin.benchmark_campaign", "wrapper-ptm-scalar"],
        ),
        (
            "ptm-native-plumbing",
            {
                "implementation": "ptm.native-binary",
                "backend": "cpp-scalar-train+packed-cpu-inference",
                "commit": "working-tree",
                "config": {
                    "clauses": 20,
                    "states_per_action": 100,
                    "specificity": 3.9,
                    "threshold": 15,
                    "epochs": 1,
                    "seed": 7,
                    "inference_repeats": 2,
                    "inference_warmup_repeats": 1,
                },
            },
            [
                sys.executable,
                "-m",
                "prolog_tsetlin.benchmark_campaign",
                "wrapper-ptm-native",
                str(native_executable),
            ],
        ),
        (
            "pytsetlinmachine-plumbing",
            {
                "implementation": "pytsetlinmachine.multiclass",
                "backend": "cpu",
                "commit": PYTSETLIN_COMMIT,
                "config": common
                | {
                    "clauses": 10,
                    "number_of_state_bits": 8,
                    "indexed": True,
                },
            },
            [
                str(incumbents / "envs" / "pytsetlinmachine" / "bin" / "python"),
                str(wrapper),
                "pytsetlinmachine",
            ],
        ),
        (
            "tmu-plumbing",
            {
                "implementation": "tmu.vanilla-classifier",
                "backend": "cpu",
                "commit": TMU_COMMIT,
                "config": common
                | {
                    "clauses": 10,
                    "number_of_state_bits": 8,
                    "platform": "CPU",
                    "shuffle": True,
                },
            },
            [
                str(incumbents / "envs" / "tmu" / "bin" / "python"),
                str(wrapper),
                "tmu",
            ],
        ),
    )
    raw_jsonl = output / "raw.jsonl"
    summaries = []
    for run_id, model, command in attempts:
        record = run_campaign_attempt(
            _request(
                manifest_path,
                output,
                run_id=run_id,
                model=model,
            ),
            command,
            raw_jsonl=raw_jsonl,
            timeout_seconds=120,
        )
        summaries.append(
            {
                "run_id": run_id,
                "status": record["status"],
                "failure": record["failure"],
            }
        )
    print(json.dumps({"plumbing_only": True, "attempts": summaries}, indent=2))
    return int(any(summary["status"] != "ok" for summary in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
