#!/usr/bin/env python3
"""Run and analyze one deterministic MNIST checkpoint-mixing campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Callable, Sequence

from mnist_checkpoint_search import analyze_capture


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _windows_to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    if not drive:
        raise RuntimeError("cannot translate a non-drive path to WSL")
    return f"/mnt/{drive}/{path.as_posix()[2:].lstrip('/')}"


def _runner_command(runner: Path) -> tuple[tuple[str, ...], Callable[[Path], str]]:
    if os.name == "nt" and runner.suffix.lower() != ".exe":
        return ("wsl.exe", _windows_to_wsl(runner)), _windows_to_wsl
    return (str(runner),), str


def _git_commit(project: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clause_counts(value: str) -> tuple[int, ...]:
    fields = value.split(",")
    if len(fields) == 1:
        fields *= 10
    if len(fields) != 10:
        raise argparse.ArgumentTypeError(
            "clauses must be one positive integer or 10 comma-separated integers"
        )
    try:
        counts = tuple(int(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError("clauses must be integers") from error
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("clauses must be positive")
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--material-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clauses", type=_clause_counts, default=(100,) * 10)
    parser.add_argument("--states", type=int, default=128)
    parser.add_argument("--specificity", type=float, default=8.0)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=(10, 20, 30))
    parser.add_argument("--feedback", choices=("standard", "boost"), default="boost")
    args = parser.parse_args(argv)

    project = args.project_root.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    material = args.material_directory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if not runner.is_file():
        parser.error(f"runner does not exist: {runner}")
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not checkpoints or checkpoints[-1] != args.epochs:
        parser.error("checkpoints must include the final training epoch")
    if any(epoch <= 0 or epoch > args.epochs for epoch in checkpoints):
        parser.error("checkpoint epochs must lie within the training run")
    manifest_path = material / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    train = material / splits["train"]["path"]
    validation = material / splits["validation"]["path"]
    audit = material / splits["test"]["path"]
    if not all(path.is_file() for path in (train, validation, audit)):
        parser.error("MNIST material is incomplete")

    output.mkdir(parents=True)
    capture = output / "scores"
    capture.mkdir()
    log_path = output / "training.jsonl"
    prefix, convert = _runner_command(runner)
    command = (
        *prefix,
        convert(train),
        convert(validation),
        ",".join(str(count) for count in args.clauses),
        str(args.states),
        str(args.specificity),
        str(args.threshold),
        str(args.epochs),
        str(args.seed),
        "paired",
        args.feedback,
        convert(capture),
        ",".join(str(epoch) for epoch in checkpoints),
        convert(audit),
    )
    campaign = {
        "schema": "ptm.mnist-checkpoint-campaign.v1",
        "project_commit": _git_commit(project),
        "runner": str(runner),
        "runner_digest": _digest(runner),
        "material_manifest": str(manifest_path),
        "material_manifest_digest": _digest(manifest_path),
        "config": {
            "clauses_per_class": list(args.clauses),
            "states_per_action": args.states,
            "specificity": args.specificity,
            "threshold": args.threshold,
            "epochs": args.epochs,
            "seed": args.seed,
            "training_policy": "paired",
            "feedback": args.feedback,
            "checkpoints": list(checkpoints),
            "pixel_threshold": manifest["threshold"],
        },
    }
    (output / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    process = subprocess.Popen(
        command,
        cwd=project,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    expected_epoch = 1
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        assert process.stdout is not None
        for line in process.stdout:
            record = json.loads(line)
            if record.get("schema") != "ptm.mnist-ovr-epoch.v1":
                process.kill()
                raise RuntimeError("native runner emitted the wrong schema")
            if record.get("epoch") != expected_epoch:
                process.kill()
                raise RuntimeError("native runner epoch sequence is discontinuous")
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            print(
                f"epoch {expected_epoch}/{args.epochs}  "
                f"validation={float(record['validation_accuracy']):.2%}  "
                f"train={float(record['training_seconds']):.2f}s",
                flush=True,
            )
            expected_epoch += 1
    return_code = process.wait()
    stderr = process.stderr.read() if process.stderr is not None else ""
    if return_code != 0:
        raise RuntimeError(
            "native checkpoint campaign failed: "
            + (stderr.strip() or f"exit status {return_code}")
        )
    if expected_epoch != args.epochs + 1:
        raise RuntimeError("native runner omitted one or more epochs")

    result = analyze_capture(capture, log_path, checkpoints)
    result["campaign"] = campaign
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selection = result["selection"]
    audit_result = result["audit"]
    print(
        json.dumps(
            {
                "result": str(result_path),
                "selected_schedule": selection["schedule"],
                "classifier_epoch_sum": selection["classifier_epoch_sum"],
                "audit_uniform_accuracy": audit_result["uniform_final"]["accuracy"],
                "audit_selected_accuracy": audit_result["selected"]["accuracy"],
                "audit_correct_gain": audit_result["correct_gain"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
