from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from prolog_tsetlin.benchmark_campaign import prepare_parity_ladder


def test_campaign_file_validation_preserves_venv_executable_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "base-python"
    target.write_bytes(b"executable")
    link = tmp_path / "venv-python"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"test host cannot create a file symlink: {error}")
    project = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(
        str(project / "benchmarks" / "initial_capacity" / "run_local_matrix.py")
    )

    validated = namespace["_require_existing_file"](link, "test executable")

    assert validated == Path(os.path.abspath(link))
    assert validated != target.resolve()


def test_local_matrix_is_predeclared_and_resumable(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    materials = tmp_path / "materials" / "parity-ladder"
    prepare_parity_ladder(materials, seed=47, widths=(3,))
    output = tmp_path / "campaign"
    command = [
        sys.executable,
        str(
            project
            / "benchmarks"
            / "initial_capacity"
            / "run_local_matrix.py"
        ),
        "--project-root",
        str(project),
        "--material-root",
        str(tmp_path / "materials"),
        "--ptm-commit",
        "test-commit",
        "--output",
        str(output),
        "--pass-name",
        "test-scout",
        "--family",
        "parity",
        "--route",
        "ptm-scalar",
        "--seed",
        "47",
        "--total-clauses",
        "4",
        "--total-clauses",
        "8",
        "--threshold-policy",
        "clamp-to-polarity",
        "--epochs",
        "1",
        "--inference-repeats",
        "1",
        "--inference-warmup-repeats",
        "0",
        "--timeout",
        "30",
    ]

    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)

    assert "[1/2] run test-scout-n-03-c0004-ptm-scalar-s47" in first.stdout
    assert "[2/2] run test-scout-n-03-c0008-ptm-scalar-s47" in first.stdout
    assert "[1/2] skip recorded test-scout-n-03-c0004-ptm-scalar-s47" in second.stdout
    assert "[2/2] skip recorded test-scout-n-03-c0008-ptm-scalar-s47" in second.stdout
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == "ptm.local-campaign-plan.v2"
    assert plan["attempts"][0]["score_splits"] == ["validation"]
    assert plan["attempts"][0]["model"]["config"]["clauses"] == 4
    assert plan["attempts"][0]["model"]["config"]["threshold"] == 2
    assert plan["attempts"][1]["model"]["config"]["clauses"] == 8
    assert plan["attempts"][1]["model"]["config"]["threshold"] == 4
    assert plan["total_clause_counts"] == [4, 8]
    assert plan["threshold_policy"] == {
        "name": "clamp-to-polarity",
        "requested_threshold": 15,
    }
    environment = json.loads(
        (output / "environment.json").read_text(encoding="utf-8")
    )
    assert environment["schema"] == "ptm.local-campaign-environment.v1"
    assert environment["cpu"]["logical_count"]
    assert environment["environment_digest"].startswith("sha256:")
    records = [
        json.loads(line)
        for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert all(record["status"] == "ok" for record in records)
    assert all(set(record["metrics"]) == {"validation"} for record in records)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["attempts_planned"] == 2
    assert summary["attempts_recorded"] == 2
    assert summary["statuses"] == {"ok": 2}
    assert summary["environment_digest"] == environment["environment_digest"]

    evaluation_output = tmp_path / "evaluation-plan"
    evaluation_command = command.copy()
    output_index = evaluation_command.index(str(output))
    evaluation_command[output_index] = str(evaluation_output)
    evaluation_command.extend(
        [
            "--variant",
            "n-03",
            "--score-split",
            "evaluation",
            "--plan-only",
        ]
    )
    subprocess.run(evaluation_command, check=True, capture_output=True, text=True)
    evaluation_plan = json.loads(
        (evaluation_output / "plan.json").read_text(encoding="utf-8")
    )
    assert len(evaluation_plan["attempts"]) == 2
    assert evaluation_plan["variants"] == ["n-03"]
    assert evaluation_plan["requested_score_splits"] == ["evaluation"]
    assert evaluation_plan["attempts"][0]["score_splits"] == ["evaluation"]
