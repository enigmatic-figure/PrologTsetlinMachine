from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from prolog_tsetlin.benchmark_campaign import prepare_parity_ladder


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

    assert "[1/1] run test-scout-n-03-ptm-scalar-s47" in first.stdout
    assert "[1/1] skip recorded test-scout-n-03-ptm-scalar-s47" in second.stdout
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == "ptm.local-campaign-plan.v1"
    assert plan["attempts"][0]["score_splits"] == ["validation"]
    assert plan["attempts"][0]["model"]["config"]["clauses"] == 4
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
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert set(records[0]["metrics"]) == {"validation"}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["attempts_planned"] == 1
    assert summary["attempts_recorded"] == 1
    assert summary["statuses"] == {"ok": 1}
    assert summary["environment_digest"] == environment["environment_digest"]
