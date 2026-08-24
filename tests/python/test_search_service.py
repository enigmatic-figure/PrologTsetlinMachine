from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

import prolog_tsetlin.services.search as search_service
from prolog_tsetlin import (
    DecisionTreeSearchProblem,
    GNUPrologSearch,
    PrologBridgeError,
    PrologSearchCancelled,
    load_model_artifact,
)
from prolog_tsetlin.cli import main as cli_main
from prolog_tsetlin.services.search import (
    BoundedSearchRequest,
    SearchKind,
    demo_search_document,
    run_bounded_search,
    search_request_budget,
)


GPROLOG_VALUE = (
    os.environ.get("PTM_GPROLOG")
    or shutil.which("gprolog")
    or r"C:\GNU-Prolog\bin\gprolog.exe"
)
GPROLOG = Path(GPROLOG_VALUE)


def test_demo_search_requests_expose_finite_budgets_without_launching() -> None:
    expected = {
        SearchKind.THRESHOLD: 12,
        SearchKind.FEATURE_TEMPLATE: 2,
        SearchKind.TA_CLAUSE: 10,
        SearchKind.DECISION_TREE: 74,
        SearchKind.REPAIR: 74,
    }
    for kind, candidate_bound in expected.items():
        request = BoundedSearchRequest.from_dict(
            demo_search_document(kind), expected_kind=kind
        )
        budget = search_request_budget(request)
        assert budget["candidate_upper_bound"] == candidate_bound
        assert budget["timeout_seconds"] == 30


def test_search_request_rejects_kind_mismatch_and_unknown_fields() -> None:
    document = demo_search_document(SearchKind.DECISION_TREE)
    with pytest.raises(ValueError, match="does not match"):
        BoundedSearchRequest.from_dict(
            document, expected_kind=SearchKind.TA_CLAUSE
        )
    document["surprise"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        BoundedSearchRequest.from_dict(document)

    invalid_bound = demo_search_document(SearchKind.DECISION_TREE)
    invalid_bound["problem"]["slot_count"] = True
    with pytest.raises(ValueError, match="slot_count must be an integer"):
        search_request_budget(BoundedSearchRequest.from_dict(invalid_bound))


def test_request_deadline_includes_validation_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SearchMustNotLaunch:
        def search(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("expired request launched GNU Prolog")

    request_document = demo_search_document(SearchKind.THRESHOLD)
    request_document["timeout_seconds"] = 0.1
    request = BoundedSearchRequest.from_dict(request_document)
    clock = iter((10.0, 10.2))
    monkeypatch.setattr(search_service, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        search_service,
        "GNUPrologSearch",
        lambda executable: SearchMustNotLaunch(),
    )

    with pytest.raises(PrologBridgeError, match="request timed out"):
        run_bounded_search(request)


@pytest.mark.skipif(not GPROLOG.is_file(), reason="GNU Prolog is not installed")
def test_search_services_return_typed_results_for_every_demo() -> None:
    results = {
        kind: run_bounded_search(
            BoundedSearchRequest.from_dict(demo_search_document(kind)),
            executable=GPROLOG,
        )
        for kind in SearchKind
    }
    assert results[SearchKind.THRESHOLD].report["selected_slots"] == [0, 1]
    assert (
        results[SearchKind.FEATURE_TEMPLATE].report["candidate"]["template_id"]
        == "categorical_v1"
    )
    assert results[SearchKind.TA_CLAUSE].report["included_literals"] == [0, 3]
    assert results[SearchKind.DECISION_TREE].report["depth"] == 2
    assert results[SearchKind.DECISION_TREE].exportable
    assert results[SearchKind.REPAIR].report["mismatches_before"] == 2
    assert len(results[SearchKind.REPAIR].report["counterexamples"]) == 4


@pytest.mark.skipif(not GPROLOG.is_file(), reason="GNU Prolog is not installed")
def test_search_cli_exports_verified_tree_artifact() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "xor-tree.ptm"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = cli_main(
                [
                    "search",
                    "decision-tree",
                    "--demo",
                    "--gprolog",
                    str(GPROLOG),
                    "--output",
                    str(output),
                ]
            )
        report = json.loads(stdout.getvalue())
        artifact = load_model_artifact(output)

    assert status == 0
    assert report["status"] == "solved"
    assert report["artifact"]["artifact_id"] == artifact.artifact_id
    assert artifact.verify_conformance()


@pytest.mark.skipif(not GPROLOG.is_file(), reason="GNU Prolog is not installed")
def test_search_cli_reports_no_solution_with_exit_three() -> None:
    document = demo_search_document(SearchKind.THRESHOLD)
    document["problem"] = {
        "slot_count": 2,
        "max_selected": 2,
        "positive_examples": [[0], [1]],
        "negative_examples": [[], [0, 1]],
    }
    with tempfile.TemporaryDirectory() as temporary:
        request = Path(temporary) / "xor-threshold.json"
        request.write_text(json.dumps(document), encoding="utf-8")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = cli_main(
                [
                    "search",
                    "threshold",
                    str(request),
                    "--gprolog",
                    str(GPROLOG),
                ]
            )
    report = json.loads(stdout.getvalue())
    assert status == 3
    assert report["status"] == "no_solution"


@pytest.mark.skipif(not GPROLOG.is_file(), reason="GNU Prolog is not installed")
def test_bridge_cancellation_terminates_active_prolog_process() -> None:
    problem = DecisionTreeSearchProblem.create(
        slot_count=2,
        max_depth=2,
        examples=[set(), {0}, {1}, {0, 1}],
        labels=[0, 1, 1, 0],
    )
    with pytest.raises(PrologSearchCancelled, match="cancelled"):
        GNUPrologSearch(GPROLOG).search_decision_tree(
            problem,
            timeout_seconds=30,
            cancel=lambda: True,
        )
