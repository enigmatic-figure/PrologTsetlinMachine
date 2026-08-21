from __future__ import annotations

import argparse
import contextlib
import io
import re
from pathlib import Path

import pytest

from prolog_tsetlin.cli import _parser, main as cli_main
from prolog_tsetlin.help_topics import (
    COMMAND_TOPICS,
    HELP_TOPICS,
    PARSER_TOPICS,
    TOPIC_ORDER,
    TUI_BINDINGS,
    TUI_VIEWS,
    TUI_VIEW_TOPICS,
    display_key,
    example_argv,
    render_manual_reference,
    render_tui_help,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATED_REFERENCE = ROOT / "docs" / "manual" / "reference" / "help-topics.md"


def _leaf_commands(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> set[str]:
    paths: set[str] = set()
    subparsers = tuple(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    if not subparsers:
        return {" ".join(prefix)}
    assert len(subparsers) == 1
    for name, child in subparsers[0].choices.items():
        paths.update(_leaf_commands(child, (*prefix, name)))
    return paths


def _parser_topics(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> dict[str, str | None]:
    topics: dict[str, str | None] = {}
    subparsers = tuple(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    if not subparsers:
        return topics
    assert len(subparsers) == 1
    for name, child in subparsers[0].choices.items():
        path = (*prefix, name)
        topics[" ".join(path)] = child.get_default("_help_topic")
        topics.update(_parser_topics(child, path))
    return topics


def test_every_cli_leaf_has_exactly_one_topic() -> None:
    assert _leaf_commands(_parser()) == set(COMMAND_TOPICS)
    assert set(COMMAND_TOPICS.values()) <= set(HELP_TOPICS)


def test_every_parser_uses_its_registered_topic() -> None:
    assert _parser_topics(_parser()) == dict(PARSER_TOPICS)


def test_topics_are_well_formed_and_manual_targets_exist() -> None:
    assert TOPIC_ORDER == tuple(HELP_TOPICS)
    assert len(TOPIC_ORDER) == len(set(TOPIC_ORDER))
    for topic_id, value in HELP_TOPICS.items():
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", topic_id)
        assert value.topic_id == topic_id
        assert value.title and value.summary and value.explanation
        assert value.manual_links
        assert all(
            (ROOT / manual_link).is_file() for manual_link in value.manual_links
        )
        tui_topic_ids = {
            tui_topic_id
            for topic_ids in TUI_VIEW_TOPICS.values()
            for tui_topic_id in topic_ids
        }
        assert topic_id in COMMAND_TOPICS.values() or topic_id in tui_topic_ids


def test_every_registered_example_parses_without_execution() -> None:
    parser = _parser()
    for topic_id, example in example_argv():
        arguments = parser.parse_args(example.argv)
        assert callable(arguments.handler), (topic_id, example.command)


def test_tui_bindings_and_contexts_are_complete() -> None:
    keys = [binding.key for binding in TUI_BINDINGS]
    assert len(keys) == len(set(keys))
    assert set(TUI_VIEW_TOPICS) == set(TUI_VIEWS)
    assert {
        topic_id
        for topic_ids in TUI_VIEW_TOPICS.values()
        for topic_id in topic_ids
    } <= set(HELP_TOPICS)
    for binding in TUI_BINDINGS:
        assert binding.contexts
        assert set(binding.contexts) <= set(TUI_VIEWS)
        assert binding.display_key and binding.action and binding.description

    textual = pytest.importorskip("textual")
    assert textual is not None
    from prolog_tsetlin.tui.app import PTMApp

    assert all(
        hasattr(PTMApp, f"action_{binding.action}") for binding in TUI_BINDINGS
    )
    actual = [
        (binding.key, binding.action, binding.description, binding.show)
        for binding in PTMApp.BINDINGS
    ]
    expected = [
        (binding.key, binding.action, binding.label, binding.show)
        for binding in TUI_BINDINGS
    ]
    assert actual == expected


def test_contextual_tui_help_uses_registered_controls() -> None:
    for view in TUI_VIEWS:
        title, body = render_tui_help(view)
        assert all(
            HELP_TOPICS[topic_id].title.upper() in title
            for topic_id in TUI_VIEW_TOPICS[view]
        )
        keyboard = body.split("KEYBOARD\n", 1)[1].split("\n\nREQUIREMENTS", 1)[0]
        expected = "\n".join(
            f"{binding.display_key:<6}{binding.description}"
            for binding in TUI_BINDINGS
            if view in binding.contexts
        )
        assert keyboard == expected


def test_tui_display_keys_are_derived_from_activation_keys() -> None:
    expected_overrides = {
        "f5": "F5",
        "f6": "F6",
        "question_mark": "?",
        "ctrl+l": "Ctrl+L",
    }
    for binding in TUI_BINDINGS:
        expected = expected_overrides.get(binding.key, binding.key)
        assert display_key(binding.key) == expected
        assert binding.display_key == expected


def test_generated_manual_reference_is_current() -> None:
    assert GENERATED_REFERENCE.read_text(encoding="utf-8") == render_manual_reference()


def test_ptm_help_lists_and_renders_topics() -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert cli_main(["help"]) == 0
    listing = stdout.getvalue()
    assert all(topic_id in listing for topic_id in TOPIC_ORDER)

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert cli_main(["help", "bounded-search"]) == 0
    rendered = stdout.getvalue()
    assert "BOUNDED SYMBOLIC SEARCH" in rendered
    assert "ptm search repair" in rendered
    assert "docs/manual/how-to/run-bounded-search.md" in rendered
    assert "docs/manual/reference/search-contracts.md" in rendered


def test_subcommand_help_links_to_shared_topic() -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as stopped:
        cli_main(["artifact", "run-record", "--help"])
    assert stopped.value.code == 0
    assert "ptm help preprocessing" in stdout.getvalue()


@pytest.mark.parametrize(
    ("argv", "expected_default"),
    (
        (("tui", "--help"), "(default: xor)"),
        (("export-logic", "--help"), "(default: A,B,C,D,E)"),
        (("search", "threshold", "--help"), None),
    ),
)
def test_subcommand_help_hides_internal_sentinel_defaults(
    argv: tuple[str, ...], expected_default: str | None
) -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as stopped:
        cli_main(argv)
    assert stopped.value.code == 0
    rendered = stdout.getvalue()
    assert "(default: None)" not in rendered
    assert "(default: [])" not in rendered
    assert "(default: )" not in rendered
    if expected_default is not None:
        assert rendered.count(expected_default) == 1
