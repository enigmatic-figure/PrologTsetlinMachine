"""Shared conceptual help topics and terminal-workbench control metadata."""

from __future__ import annotations

import argparse
import posixpath
import shlex
import textwrap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping


class PTMHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve authored paragraphs while exposing parser-owned defaults."""


@dataclass(frozen=True)
class HelpExample:
    """One parser-valid CLI example, excluding the executable name."""

    argv: tuple[str, ...]
    description: str

    @property
    def command(self) -> str:
        return f"ptm {shlex.join(self.argv)}"


@dataclass(frozen=True)
class HelpTopic:
    """Reusable conceptual help rendered at different levels of detail."""

    topic_id: str
    title: str
    summary: str
    explanation: tuple[str, ...]
    examples: tuple[HelpExample, ...]
    requirements: tuple[str, ...]
    manual_links: tuple[str, ...]


@dataclass(frozen=True)
class TUIBindingSpec:
    """One Textual application binding and its user-facing meaning."""

    key: str
    display_key: str
    action: str
    label: str
    description: str
    contexts: tuple[str, ...]
    show: bool = True


_TOPICS = (
    HelpTopic(
        topic_id="workbench",
        title="Terminal workbench",
        summary="Explore PTM through the keyboard-first Textual workbench.",
        explanation=(
            "The workbench provides five views for environment status, deterministic "
            "XOR training, clause inspection, portable artifacts, and bounded symbolic "
            "search. The native runtime and GNU Prolog remain optional until a workflow "
            "needs them.",
            "Use the footer for active shortcuts and open contextual help in any view for "
            "controls drawn from this shared registry.",
        ),
        examples=(
            HelpExample(("tui", "--demo", "xor"), "Launch the built-in XOR session."),
            HelpExample(("help", "training"), "Read the training topic in a terminal."),
        ),
        requirements=("Install the optional TUI extra to launch the workbench.",),
        manual_links=("docs/tui.md",),
    ),
    HelpTopic(
        topic_id="training",
        title="Deterministic training and clauses",
        summary="Train the scalar XOR oracle and inspect its learned clause state.",
        explanation=(
            "Training runs outside the Textual event loop and reports structured progress. "
            "Configuration changes mark completed results stale so they cannot be exported "
            "as though they represented the new settings.",
            "Clauses are signed pattern voters. Specificity controls specialization, states "
            "per action controls automaton memory depth, and the threshold scales the vote.",
        ),
        examples=(
            HelpExample(("tui", "--demo", "xor"), "Open the training workbench."),
        ),
        requirements=(),
        manual_links=("docs/tui.md",),
    ),
    HelpTopic(
        topic_id="artifacts",
        title="Portable artifacts",
        summary="Export, inspect, verify, and consume deterministic .ptm artifacts.",
        explanation=(
            "A portable artifact packages a model with its contracts, metadata, integrity "
            "information, and conformance cases. Verification should precede inference.",
            "Packed-TM, fixed-Logic, and masked-threshold PA artifacts share the container "
            "boundary while retaining kind-specific runtime contracts.",
        ),
        examples=(
            HelpExample(
                ("artifact", "inspect", "model.ptm", "--pretty"),
                "Inspect a portable artifact.",
            ),
            HelpExample(
                ("artifact", "verify", "model.ptm"),
                "Verify integrity, contracts, and conformance cases.",
            ),
            HelpExample(
                ("export", "snapshot.json", "model.ptm"),
                "Freeze a scalar snapshot into a portable artifact.",
            ),
        ),
        requirements=(),
        manual_links=("docs/model-export-runtime.md",),
    ),
    HelpTopic(
        topic_id="preprocessing",
        title="Typed record preprocessing",
        summary="Materialize raw typed records through an artifact's preprocessing contract.",
        explanation=(
            "Raw-record inference validates required fields and applies the exact transforms "
            "embedded in the artifact. The result includes the materialized Boolean features "
            "and a per-literal provenance trace.",
            "Use repeated --record values for interactive JSON objects or --jsonl for a "
            "newline-delimited stream. These sources are mutually exclusive.",
        ),
        examples=(
            HelpExample(
                (
                    "artifact",
                    "run-record",
                    "model.ptm",
                    "--record",
                    '{"left":false,"right":true}',
                    "--pretty",
                ),
                "Run one typed JSON record.",
            ),
            HelpExample(
                ("artifact", "run-record", "model.ptm", "--jsonl", "records.jsonl"),
                "Run newline-delimited records.",
            ),
        ),
        requirements=("The artifact must declare a portable raw-record contract.",),
        manual_links=("docs/preprocessing-contract.md",),
    ),
    HelpTopic(
        topic_id="bounded-search",
        title="Bounded symbolic search",
        summary="Run finite, deadline-limited GNU Prolog searches and inspect their budgets.",
        explanation=(
            "Each request declares a bounded search space and timeout. PTM reports the "
            "candidate ceiling before launch and distinguishes no-solution results from "
            "invalid requests or runtime failures.",
            "Decision-tree and repair searches can export fixed-Logic artifacts after an "
            "exact result is found.",
        ),
        examples=(
            HelpExample(
                ("search", "threshold", "--demo", "--pretty"),
                "Run the built-in threshold example.",
            ),
            HelpExample(
                (
                    "search",
                    "repair",
                    "--demo",
                    "--output",
                    "repair.ptm",
                    "--pretty",
                ),
                "Repair a demo tree and export fixed Logic.",
            ),
        ),
        requirements=(
            "GNU Prolog must be installed and discoverable through --gprolog, PTM_GPROLOG, "
            "or PATH.",
        ),
        manual_links=("docs/bounded-search.md",),
    ),
)

HELP_TOPICS: Mapping[str, HelpTopic] = MappingProxyType(
    {topic.topic_id: topic for topic in _TOPICS}
)
TOPIC_ORDER = tuple(topic.topic_id for topic in _TOPICS)

# Executable parser paths are assigned to exactly one conceptual topic. Parent
# grouping parsers such as ``artifact`` and ``search`` are intentionally absent.
COMMAND_TOPICS: Mapping[str, str] = MappingProxyType(
    {
        "help": "workbench",
        "tui": "workbench",
        "artifact inspect": "artifacts",
        "artifact verify": "artifacts",
        "artifact run-record": "preprocessing",
        "search threshold": "bounded-search",
        "search feature-template": "bounded-search",
        "search ta-clause": "bounded-search",
        "search decision-tree": "bounded-search",
        "search repair": "bounded-search",
        "export": "artifacts",
        "export-logic": "artifacts",
        "export-pa": "artifacts",
    }
)

TUI_VIEWS = ("overview", "train", "clauses", "artifacts", "search")
TUI_VIEW_TOPICS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "overview": ("workbench",),
        "train": ("training",),
        "clauses": ("training",),
        "artifacts": ("artifacts", "preprocessing"),
        "search": ("bounded-search",),
    }
)
_ALL_VIEWS = TUI_VIEWS

TUI_BINDINGS = (
    TUIBindingSpec(
        "1", "1", "show_overview", "Overview", "Open Overview.", _ALL_VIEWS
    ),
    TUIBindingSpec("2", "2", "show_train", "Train", "Open Train.", _ALL_VIEWS),
    TUIBindingSpec(
        "3", "3", "show_clauses", "Clauses", "Open Clauses.", _ALL_VIEWS
    ),
    TUIBindingSpec(
        "4", "4", "show_artifacts", "Artifacts", "Open Artifacts.", _ALL_VIEWS
    ),
    TUIBindingSpec("5", "5", "show_search", "Search", "Open Search.", _ALL_VIEWS),
    TUIBindingSpec(
        "t",
        "t",
        "train",
        "Train XOR",
        "Start XOR training.",
        ("overview", "train", "clauses"),
    ),
    TUIBindingSpec(
        "x",
        "x",
        "cancel",
        "Cancel",
        "Cancel active training or search.",
        ("train", "clauses", "search"),
    ),
    TUIBindingSpec(
        "e",
        "e",
        "export",
        "Export",
        "Export the completed training run.",
        ("train", "clauses", "artifacts"),
        False,
    ),
    TUIBindingSpec(
        "l",
        "l",
        "load_artifact",
        "Load artifact",
        "Load and verify the artifact path.",
        ("artifacts",),
        False,
    ),
    TUIBindingSpec(
        "r",
        "r",
        "run_record",
        "Run record",
        "Run the typed record through the loaded artifact.",
        ("artifacts",),
        False,
    ),
    TUIBindingSpec(
        "f5",
        "F5",
        "search",
        "Run search",
        "Run the bounded search request.",
        ("search",),
        False,
    ),
    TUIBindingSpec(
        "f6",
        "F6",
        "cancel_search",
        "Cancel search",
        "Cancel the active bounded search.",
        ("search",),
        False,
    ),
    TUIBindingSpec(
        "o", "o", "show_overview", "Overview", "Open Overview.", _ALL_VIEWS, False
    ),
    TUIBindingSpec(
        "c", "c", "show_clauses", "Clauses", "Open Clauses.", _ALL_VIEWS, False
    ),
    TUIBindingSpec(
        "p",
        "p",
        "command_palette",
        "Palette",
        "Open the command palette.",
        _ALL_VIEWS,
        False,
    ),
    TUIBindingSpec(
        "question_mark", "?", "help", "Help", "Open contextual help.", _ALL_VIEWS
    ),
    TUIBindingSpec(
        "ctrl+l",
        "Ctrl+L",
        "events",
        "Events",
        "Collapse or expand the event dock.",
        _ALL_VIEWS,
    ),
    TUIBindingSpec("q", "q", "quit", "Quit", "Quit the workbench.", _ALL_VIEWS),
)


def topic(topic_id: str) -> HelpTopic:
    """Return one topic or raise a stable error for callers outside argparse."""

    try:
        return HELP_TOPICS[topic_id]
    except KeyError as error:
        choices = ", ".join(TOPIC_ORDER)
        raise ValueError(
            f"unknown help topic {topic_id!r}; choose from {choices}"
        ) from error


def related_commands(topic_id: str) -> tuple[str, ...]:
    """Return parser command paths owned by one topic."""

    topic(topic_id)
    return tuple(
        command for command, owner in COMMAND_TOPICS.items() if owner == topic_id
    )


def binding_for_action(action: str) -> TUIBindingSpec:
    """Return the primary (first declared) binding for an application action."""

    for binding in TUI_BINDINGS:
        if binding.action == action:
            return binding
    raise KeyError(action)


def bindings_for_view(view: str) -> tuple[TUIBindingSpec, ...]:
    """Return bindings relevant to a workbench view."""

    if view not in TUI_VIEW_TOPICS:
        raise KeyError(view)
    return tuple(binding for binding in TUI_BINDINGS if view in binding.contexts)


def parser_topic_kwargs(topic_id: str) -> dict[str, object]:
    """Return shared argparse description and cross-link fields."""

    value = topic(topic_id)
    return {
        "description": value.summary,
        "epilog": f"Concepts and examples: ptm help {topic_id}",
        "formatter_class": PTMHelpFormatter,
    }


def _wrapped_lines(
    text: str,
    *,
    width: int = 79,
    indent: str = "",
    subsequent_indent: str | None = None,
) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent if subsequent_indent is None else subsequent_indent,
    ) or [indent]


def render_topic_index() -> str:
    """Render terminal topic discovery."""

    width = max(len(topic_id) for topic_id in TOPIC_ORDER)
    lines = ["PTM HELP TOPICS", ""]
    for topic_id in TOPIC_ORDER:
        value = HELP_TOPICS[topic_id]
        lines.append(f"  {topic_id:<{width}}  {value.summary}")
    lines.extend(("", "Run `ptm help TOPIC` for concepts, examples, and links."))
    return "\n".join(lines)


def render_topic(topic_id: str) -> str:
    """Render one topic for the terminal."""

    value = topic(topic_id)
    lines = [value.title.upper(), ""]
    lines.extend(_wrapped_lines(value.summary))
    for paragraph in value.explanation:
        lines.append("")
        lines.extend(_wrapped_lines(paragraph))
    if value.requirements:
        lines.extend(("", "Requirements:"))
        for requirement in value.requirements:
            lines.extend(
                _wrapped_lines(
                    requirement,
                    indent="  - ",
                    subsequent_indent="    ",
                )
            )
    if value.examples:
        lines.extend(("", "Examples:"))
        for example in value.examples:
            lines.append(f"  {example.command}")
            lines.extend(_wrapped_lines(example.description, indent="    "))
    commands = related_commands(topic_id)
    if commands:
        lines.extend(("", "Related commands:"))
        lines.extend(f"  ptm {command}" for command in commands)
    views = tuple(
        view for view, owners in TUI_VIEW_TOPICS.items() if topic_id in owners
    )
    if views:
        lines.extend(("", "TUI views:", f"  {', '.join(views)}"))
    lines.extend(("", "Manual:"))
    lines.extend(f"  {manual_link}" for manual_link in value.manual_links)
    return "\n".join(lines)


def render_tui_help(view: str) -> tuple[str, str]:
    """Render a compact contextual title and body for the Textual modal."""

    values = tuple(topic(topic_id) for topic_id in TUI_VIEW_TOPICS[view])
    bindings = bindings_for_view(view)
    navigation = tuple(binding for binding in bindings if binding.key.isdigit())
    controls = tuple(binding for binding in bindings if not binding.key.isdigit())
    labels = ", ".join(binding.label for binding in navigation)
    lines = [
        *(value.summary for value in values),
        "",
        "KEYBOARD",
        f"1-5  Switch views: {labels}",
    ]
    lines.extend(
        f"{binding.display_key:<6}{binding.description}" for binding in controls
    )
    requirements = tuple(
        dict.fromkeys(
            requirement for value in values for requirement in value.requirements
        )
    )
    if requirements:
        lines.extend(("", "REQUIREMENTS", *requirements))
    titles = " / ".join(value.title.upper() for value in values)
    return f"PTM WORKBENCH / {titles}", "\n".join(lines)


def _markdown_controls(topic_id: str) -> tuple[TUIBindingSpec, ...]:
    views = tuple(
        view for view, owners in TUI_VIEW_TOPICS.items() if topic_id in owners
    )
    seen: set[str] = set()
    controls: list[TUIBindingSpec] = []
    for view in views:
        for binding in bindings_for_view(view):
            if binding.key not in seen:
                seen.add(binding.key)
                controls.append(binding)
    return tuple(controls)


def render_manual_reference() -> str:
    """Render the tracked MyST reference page from authoritative registry data."""

    lines = [
        "<!-- Generated by scripts/render_help_reference.py; do not edit directly. -->",
        "# Help topics and workbench controls",
        "",
        "CLI syntax and defaults come from the live `argparse` tree. The conceptual",
        "topics, examples, requirements, related-command assignments, and workbench",
        "controls below come from `prolog_tsetlin.help_topics`.",
    ]
    for topic_id in TOPIC_ORDER:
        value = HELP_TOPICS[topic_id]
        lines.extend(("", f"## {value.title}", "", value.summary))
        for paragraph in value.explanation:
            lines.extend(("", paragraph))
        if value.requirements:
            lines.extend(("", "### Requirements", ""))
            lines.extend(f"- {requirement}" for requirement in value.requirements)
        if value.examples:
            lines.extend(("", "### Examples", ""))
            for example in value.examples:
                lines.extend(
                    (example.description, "", "```console", example.command, "```", "")
                )
            if lines[-1] == "":
                lines.pop()
        commands = related_commands(topic_id)
        if commands:
            lines.extend(("", "### Related commands", ""))
            lines.extend(f"- `ptm {command}`" for command in commands)
        controls = _markdown_controls(topic_id)
        if controls:
            lines.extend(
                (
                    "",
                    "### Workbench controls",
                    "",
                    "| Key | Action |",
                    "| --- | --- |",
                )
            )
            lines.extend(
                f"| `{binding.display_key}` | {binding.description} |"
                for binding in controls
            )
        lines.extend(("", "Authored guides:"))
        for manual_link in value.manual_links:
            relative_manual = posixpath.relpath(
                manual_link, start="docs/manual/reference"
            )
            lines.append(f"- [{manual_link}]({relative_manual})")
    return "\n".join(lines) + "\n"


def example_argv() -> Iterable[tuple[str, HelpExample]]:
    """Yield examples with their owner IDs for coverage validation."""

    for topic_id in TOPIC_ORDER:
        for example in HELP_TOPICS[topic_id].examples:
            yield topic_id, example
