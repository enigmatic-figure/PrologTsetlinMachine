#!/usr/bin/env python3
"""Execute a small, resumable local capacity-campaign matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import sys
from typing import Mapping, Sequence

from prolog_tsetlin.benchmark_campaign import (
    CampaignDatasetManifest,
    CampaignRunRequest,
    run_campaign_attempt,
)
from prolog_tsetlin.model_generation import canonical_json_bytes, content_digest
from prolog_tsetlin.services._atomic import publish_bytes


PYTSETLIN_COMMIT = "d6c1cf0e4aaa4a8ae2f2818ba27878fb89d31dc5"
TMU_COMMIT = "5605ff070a18549328028c907a9acf68e063346e"
DEFAULT_ROUTES = ("ptm-native", "pytsetlinmachine", "tmu")
FAMILY_DATASETS = {
    "xor": "synthetic.xor20-noise.v1",
    "parity": "synthetic.parity-ladder.v1",
    "logic": "ptm.logic-pairs.v1",
}
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _require_existing_file(path: Path, label: str) -> Path:
    result = path.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} is absent: {result}")
    return result


def _selected_manifests(
    material_root: Path,
    families: Sequence[str],
) -> tuple[tuple[Path, CampaignDatasetManifest, tuple[str, ...]], ...]:
    wanted = {FAMILY_DATASETS[family] for family in families}
    selected = []
    for path in sorted(material_root.resolve().rglob("manifest.json")):
        manifest = CampaignDatasetManifest.load(path)
        if manifest.dataset_id not in wanted:
            continue
        score_splits = (
            ("validation",)
            if "validation" in manifest.split_map
            else ("evaluation",)
        )
        selected.append((path, manifest, score_splits))
    if not selected:
        raise ValueError("the selected campaign families have no material manifests")
    represented = {manifest.dataset_id for _, manifest, _ in selected}
    missing = sorted(wanted - represented)
    if missing:
        raise ValueError("campaign material is missing dataset families: " + ", ".join(missing))
    return tuple(selected)


def _model(
    route: str,
    *,
    ptm_commit: str,
    total_clauses: int,
    state_bits: int,
    threshold: int,
    specificity: float,
    epochs: int,
    seed: int,
    inference_repeats: int,
    inference_warmup_repeats: int,
) -> dict[str, object]:
    common = {
        "threshold": threshold,
        "specificity": specificity,
        "epochs": epochs,
        "seed": seed,
        "inference_repeats": inference_repeats,
        "inference_warmup_repeats": inference_warmup_repeats,
    }
    if route in ("ptm-scalar", "ptm-native"):
        return {
            "implementation": (
                "ptm.scalar-reference" if route == "ptm-scalar" else "ptm.native-binary"
            ),
            "backend": (
                "python-scalar-reference"
                if route == "ptm-scalar"
                else "cpp-scalar-train+packed-cpu-inference"
            ),
            "commit": ptm_commit,
            "config": common
            | {
                "clauses": total_clauses,
                "states_per_action": 1 << (state_bits - 1),
            },
        }
    incumbent_clauses = total_clauses // 2
    incumbent = common | {
        "clauses": incumbent_clauses,
        "number_of_state_bits": state_bits,
        "boost_true_positive_feedback": 1,
        "weighted_clauses": False,
        "feature_negation": True,
        "max_included_literals": None,
    }
    if route == "pytsetlinmachine":
        return {
            "implementation": "pytsetlinmachine.multiclass",
            "backend": "cpu",
            "commit": PYTSETLIN_COMMIT,
            "config": incumbent | {"indexed": True},
        }
    if route == "tmu":
        return {
            "implementation": "tmu.vanilla-classifier",
            "backend": "cpu",
            "commit": TMU_COMMIT,
            "config": incumbent | {"platform": "CPU", "shuffle": True},
        }
    raise ValueError(f"unknown campaign route: {route}")


def _command(
    route: str,
    *,
    project_root: Path,
    incumbent_root: Path | None,
    ptm_native_executable: Path | None,
) -> list[str]:
    if route == "ptm-scalar":
        return [
            sys.executable,
            "-m",
            "prolog_tsetlin.benchmark_campaign",
            "wrapper-ptm-scalar",
        ]
    if route == "ptm-native":
        if ptm_native_executable is None:
            raise ValueError("the PTM native route requires --ptm-native-executable")
        return [
            sys.executable,
            "-m",
            "prolog_tsetlin.benchmark_campaign",
            "wrapper-ptm-native",
            str(_require_existing_file(ptm_native_executable, "PTM native executable")),
        ]
    if incumbent_root is None:
        raise ValueError("incumbent routes require --incumbent-root")
    wrapper = _require_existing_file(
        project_root / "benchmarks" / "initial_capacity" / "incumbent_wrapper.py",
        "incumbent wrapper",
    )
    environment = "pytsetlinmachine" if route == "pytsetlinmachine" else "tmu"
    python = _require_existing_file(
        incumbent_root.resolve() / "envs" / environment / "bin" / "python",
        f"{environment} Python",
    )
    return [str(python), str(wrapper), environment]


def _read_attempts(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        return ()
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"raw campaign JSONL is malformed at line {line_number}") from error
        if not isinstance(value, dict) or type(value.get("run_id")) is not str:
            raise ValueError(f"raw campaign record is malformed at line {line_number}")
        records.append(value)
    return tuple(records)


def _cpu_model() -> str:
    if sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "n/a"


def _publish_environment(output: Path) -> str:
    payload = {
        "schema": "ptm.local-campaign-environment.v1",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "model": _cpu_model(),
        },
        "thread_environment": {
            name: os.environ.get(name, "unset") for name in THREAD_ENVIRONMENT
        },
    }
    identity = content_digest(payload)
    complete = payload | {"environment_digest": identity}
    data = json.dumps(
        complete,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path = output / "environment.json"
    if path.is_file() and path.read_bytes() != data:
        raise ValueError("existing campaign environment disagrees with this host")
    publish_bytes(path, data, overwrite=True)
    return identity


def _publish_plan(output: Path, payload: Mapping[str, object]) -> str:
    identity = content_digest(dict(payload))
    complete = dict(payload) | {"plan_digest": identity}
    data = json.dumps(
        complete,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path = output / "plan.json"
    if path.is_file() and path.read_bytes() != data:
        raise ValueError("existing campaign plan disagrees with requested matrix")
    publish_bytes(path, data, overwrite=True)
    return identity


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--material-root", type=Path, required=True)
    parser.add_argument("--incumbent-root", type=Path)
    parser.add_argument("--ptm-native-executable", type=Path)
    parser.add_argument("--ptm-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass-name", default="scout")
    parser.add_argument("--family", action="append", choices=sorted(FAMILY_DATASETS))
    parser.add_argument(
        "--route",
        action="append",
        choices=("ptm-scalar", *DEFAULT_ROUTES),
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--total-clauses", type=int, default=200)
    parser.add_argument("--state-bits", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--specificity", type=float, default=3.9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--inference-warmup-repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1_800)
    parser.add_argument("--plan-only", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.total_clauses < 4 or arguments.total_clauses % 4:
        parser.error("--total-clauses must be a positive multiple of four")
    if not 2 <= arguments.state_bits <= 16:
        parser.error("--state-bits must be between 2 and 16")
    if arguments.threshold < 1 or arguments.epochs < 1:
        parser.error("--threshold and --epochs must be positive")
    if arguments.inference_repeats < 1 or arguments.inference_warmup_repeats < 0:
        parser.error("inference repeats are invalid")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")

    project = arguments.project_root.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    families = tuple(arguments.family or sorted(FAMILY_DATASETS))
    routes = tuple(arguments.route or DEFAULT_ROUTES)
    seeds = tuple(arguments.seed or (20260826,))
    if any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        parser.error("--seed values must be unique nonnegative integers")
    manifests = _selected_manifests(arguments.material_root, families)

    attempts = []
    for manifest_path, manifest, score_splits in manifests:
        for seed in seeds:
            for route in routes:
                model = _model(
                    route,
                    ptm_commit=arguments.ptm_commit,
                    total_clauses=arguments.total_clauses,
                    state_bits=arguments.state_bits,
                    threshold=arguments.threshold,
                    specificity=arguments.specificity,
                    epochs=arguments.epochs,
                    seed=seed,
                    inference_repeats=arguments.inference_repeats,
                    inference_warmup_repeats=arguments.inference_warmup_repeats,
                )
                run_id = (
                    f"{arguments.pass_name}-{manifest.variant_id}-{route}-s{seed}"
                )
                attempts.append(
                    {
                        "run_id": run_id,
                        "route": route,
                        "seed": seed,
                        "manifest": str(
                            manifest_path.resolve().relative_to(
                                arguments.material_root.resolve()
                            )
                        ).replace("\\", "/"),
                        "manifest_digest": manifest.manifest_digest,
                        "dataset_id": manifest.dataset_id,
                        "variant_id": manifest.variant_id,
                        "score_splits": list(score_splits),
                        "model": model,
                    }
                )
    plan = {
        "schema": "ptm.local-campaign-plan.v1",
        "campaign_id": "initial-capacity-local-v1",
        "pass": arguments.pass_name,
        "families": list(families),
        "routes": list(routes),
        "seeds": list(seeds),
        "attempts": attempts,
    }
    run_ids = {str(attempt["run_id"]) for attempt in attempts}
    if len(run_ids) != len(attempts):
        raise ValueError("campaign plan contains duplicate run IDs")
    plan_digest = _publish_plan(output, plan)
    environment_digest = _publish_environment(output)
    if arguments.plan_only:
        print(
            json.dumps(
                {
                    "plan_digest": plan_digest,
                    "environment_digest": environment_digest,
                    "attempts": len(attempts),
                },
                indent=2,
            )
        )
        return 0

    raw_jsonl = output / "raw.jsonl"
    prior = _read_attempts(raw_jsonl)
    completed = {str(record["run_id"]) for record in prior}
    if len(completed) != len(prior):
        raise ValueError("raw campaign JSONL contains duplicate run IDs")
    unexpected = sorted(completed - run_ids)
    if unexpected:
        raise ValueError(
            "raw campaign JSONL contains run IDs outside the immutable plan: "
            + ", ".join(unexpected)
        )
    commands = {
        route: _command(
            route,
            project_root=project,
            incumbent_root=arguments.incumbent_root,
            ptm_native_executable=arguments.ptm_native_executable,
        )
        for route in routes
    }

    for index, attempt in enumerate(attempts, 1):
        run_id = str(attempt["run_id"])
        if run_id in completed:
            print(f"[{index}/{len(attempts)}] skip recorded {run_id}", flush=True)
            continue
        manifest_path = (
            arguments.material_root.resolve() / str(attempt["manifest"])
        ).resolve()
        request = CampaignRunRequest(
            campaign_id="initial-capacity-local-v1",
            run_id=run_id,
            pass_name=arguments.pass_name,
            track="shared",
            dataset_manifest=str(manifest_path),
            dataset_manifest_digest=str(attempt["manifest_digest"]),
            train_split="train",
            score_splits=tuple(attempt["score_splits"]),
            model=attempt["model"],
            output_directory=str((output / "runs" / run_id).resolve()),
        )
        route = str(attempt["route"])
        print(f"[{index}/{len(attempts)}] run {run_id}", flush=True)
        record = run_campaign_attempt(
            request,
            commands[route],
            raw_jsonl=raw_jsonl,
            timeout_seconds=arguments.timeout,
        )
        completed.add(run_id)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": record["status"],
                    "metrics": record["metrics"],
                    "training_s": record["timing"]["adaptive_training_s"],
                    "failure": record["failure"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records = _read_attempts(raw_jsonl)
    statuses = Counter(str(record.get("status")) for record in records)
    summary = {
        "schema": "ptm.local-campaign-summary.v1",
        "plan_digest": plan_digest,
        "environment_digest": environment_digest,
        "attempts_planned": len(attempts),
        "attempts_recorded": len(records),
        "statuses": dict(sorted(statuses.items())),
    }
    publish_bytes(
        output / "summary.json",
        canonical_json_bytes(summary) + b"\n",
        overwrite=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(len(records) != len(attempts) or any(status != "ok" for status in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
