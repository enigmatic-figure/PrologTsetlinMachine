from __future__ import annotations

from pathlib import Path
import runpy

import pytest

np = pytest.importorskip("numpy")


def _namespace() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    return runpy.run_path(
        str(
            project
            / "benchmarks"
            / "initial_capacity"
            / "run_mnist_jit_distillation.py"
        ),
        run_name="mnist_jit_distillation_test",
    )


def test_softmax_temperature_produces_normalized_probabilities() -> None:
    softmax = _namespace()["_softmax"]
    probabilities = softmax(
        np.asarray([[1000.0, 999.0], [-1000.0, -999.0]]),
        2.0,
    )

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]


def test_paired_counts_collective_fixes_and_regressions() -> None:
    paired = _namespace()["_paired"]
    result = paired(
        np.asarray([0, 1, 2, 3]),
        np.asarray([0, 0, 2, 2]),
        np.asarray([1, 1, 2, 2]),
    )

    assert result == {"fixes": 1, "regressions": 1, "disagreements": 2}


def test_classification_uses_collective_argmax() -> None:
    classification = _namespace()["_classification"]
    truth = np.asarray([0, 1, 2])
    scores = np.full((3, 10), -4, dtype=np.int64)
    scores[0, 0] = 3
    scores[1, 1] = 2
    scores[2, 4] = 5

    result = classification(truth, scores)

    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["predictions"].tolist() == [0, 1, 4]
