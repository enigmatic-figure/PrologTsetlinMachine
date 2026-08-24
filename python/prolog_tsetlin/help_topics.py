"""Shared conceptual help topics and terminal-workbench control metadata."""

from __future__ import annotations

import argparse
import posixpath
import shlex
import textwrap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping


class PTMHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Preserve authored paragraphs and show only meaningful static defaults."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        help_text = action.help
        if help_text is None:
            return None
        default = action.default
        should_show_default = (
            "%(default)" not in help_text
            and default is not None
            and default is not argparse.SUPPRESS
            and default not in ("", [], (), {})
            and not isinstance(default, bool)
            and (action.option_strings or action.nargs in (argparse.OPTIONAL, argparse.ZERO_OR_MORE))
        )
        if should_show_default:
            return f"{help_text} (default: %(default)s)"
        return help_text


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
    action: str
    label: str
    description: str
    contexts: tuple[str, ...]
    show: bool = True

    @property
    def display_key(self) -> str:
        """Derive presentation from the activation key."""

        return display_key(self.key)


_TOPICS = (
    HelpTopic(
        topic_id="workbench",
        title="Terminal workbench",
        summary="Explore PTM through the keyboard-first Textual workbench.",
        explanation=(
            "The canonical workbench keeps system state and research telemetry visible "
            "while task views cover deterministic XOR training, clauses, TA populations, "
            "literals, temporal samples, portable artifacts, and bounded symbolic search. "
            "The native runtime and GNU Prolog remain optional until a workflow needs them.",
            "Use the footer for active shortcuts and open help for controls drawn from this "
            "shared registry. The former five-view interface remains available with "
            "`ptm tui --style classic`.",
        ),
        examples=(
            HelpExample(("tui", "--demo", "xor"), "Launch the built-in XOR session."),
            HelpExample(("help", "training"), "Read the training topic in a terminal."),
        ),
        requirements=("Install the optional TUI extra to launch the workbench.",),
        manual_links=(
            "docs/manual/tutorials/first-tui-session.md",
            "docs/manual/how-to/tui.md",
        ),
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
        manual_links=("docs/manual/tutorials/first-tui-session.md",),
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
        manual_links=(
            "docs/manual/how-to/export-artifacts.md",
            "docs/manual/reference/artifact-contract.md",
        ),
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
        manual_links=("docs/manual/reference/preprocessing.md",),
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
        manual_links=(
            "docs/manual/how-to/run-bounded-search.md",
            "docs/manual/reference/search-contracts.md",
        ),
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

GROUP_TOPICS: Mapping[str, str] = MappingProxyType(
    {
        "artifact": "artifacts",
        "search": "bounded-search",
    }
)
PARSER_TOPICS: Mapping[str, str] = MappingProxyType(
    {**GROUP_TOPICS, **COMMAND_TOPICS}
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
        "1", "show_overview", "Overview", "Open Overview.", _ALL_VIEWS
    ),
    TUIBindingSpec("2", "show_train", "Train", "Open Train.", _ALL_VIEWS),
    TUIBindingSpec(
        "3", "show_clauses", "Clauses", "Open Clauses.", _ALL_VIEWS
    ),
    TUIBindingSpec(
        "4", "show_artifacts", "Artifacts", "Open Artifacts.", _ALL_VIEWS
    ),
    TUIBindingSpec("5", "show_search", "Search", "Open Search.", _ALL_VIEWS),
    TUIBindingSpec(
        "t",
        "train",
        "Train XOR",
        "Start XOR training.",
        ("overview", "train", "clauses"),
    ),
    TUIBindingSpec(
        "x",
        "cancel",
        "Cancel",
        "Cancel active training or search.",
        ("train", "clauses", "search"),
    ),
    TUIBindingSpec(
        "e",
        "export",
        "Export",
        "Export the completed training run.",
        ("train", "clauses", "artifacts"),
        False,
    ),
    TUIBindingSpec(
        "l",
        "load_artifact",
        "Load artifact",
        "Load and verify the artifact path.",
        ("artifacts",),
        False,
    ),
    TUIBindingSpec(
        "r",
        "run_record",
        "Run record",
        "Run the typed record through the loaded artifact.",
        ("artifacts",),
        False,
    ),
    TUIBindingSpec(
        "f5",
        "search",
        "Run search",
        "Run the bounded search request.",
        ("search",),
        False,
    ),
    TUIBindingSpec(
        "f6",
        "cancel_search",
        "Cancel search",
        "Cancel the active bounded search.",
        ("search",),
        False,
    ),
    TUIBindingSpec(
        "o", "show_overview", "Overview", "Open Overview.", _ALL_VIEWS, False
    ),
    TUIBindingSpec(
        "c", "show_clauses", "Clauses", "Open Clauses.", _ALL_VIEWS, False
    ),
    TUIBindingSpec(
        "p",
        "command_palette",
        "Palette",
        "Open the command palette.",
        _ALL_VIEWS,
        False,
    ),
    TUIBindingSpec(
        "question_mark", "help", "Help", "Open contextual help.", _ALL_VIEWS
    ),
    TUIBindingSpec(
        "ctrl+l",
        "events",
        "Events",
        "Collapse or expand the event dock.",
        _ALL_VIEWS,
    ),
    TUIBindingSpec("q", "quit", "Quit", "Quit the workbench.", _ALL_VIEWS),
)

# Navigation is shell presentation, not a PTM command.  Keep the complete
# registry above for the classic frontend and its contextual help, while
# allowing other shells to share only behavior with the same semantics.
TUI_NAVIGATION_ACTIONS = frozenset(
    ("show_overview", "show_train", "show_clauses", "show_artifacts", "show_search")
)
TUI_SEMANTIC_ACTIONS = frozenset(
    (
        "train",
        "cancel",
        "export",
        "load_artifact",
        "run_record",
        "search",
        "cancel_search",
        "help",
        "quit",
    )
)
TUI_NAVIGATION_BINDINGS = tuple(
    binding for binding in TUI_BINDINGS if binding.action in TUI_NAVIGATION_ACTIONS
)
TUI_SEMANTIC_BINDINGS = tuple(
    binding for binding in TUI_BINDINGS if binding.action in TUI_SEMANTIC_ACTIONS
)

# The canonical shell owns its presentation bindings. ``single_pane`` remains
# a compatibility name for the layout, not the user-facing workbench identity.
_CANONICAL_CONTEXT = ("workbench",)
CANONICAL_TUI_BINDINGS = TUI_SEMANTIC_BINDINGS + (
    TUIBindingSpec("1", "tab_1", "System", "Open System.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec(
        "2", "tab_2", "Dashboard", "Open Dashboard.", _CANONICAL_CONTEXT, False
    ),
    TUIBindingSpec("3", "tab_3", "Clauses", "Open Clauses.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec("4", "tab_4", "TA States", "Open TA States.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec("5", "tab_5", "Literals", "Open Literals.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec("6", "tab_6", "Graphs", "Open Graphs.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec("7", "tab_7", "Artifacts", "Open Artifacts.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec("v", "show_timeline", "Timeline", "Open Timeline.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec("s", "show_search", "Search", "Open Search.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec("c", "show_config", "Config", "Open Config.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec(
        "p",
        "show_predictions",
        "Predictions",
        "Open Predictions.",
        _CANONICAL_CONTEXT,
        False,
    ),
    TUIBindingSpec("ctrl+l", "show_events", "Events", "Open Events.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec("d", "show_detail", "Detail", "Open Detail.", _CANONICAL_CONTEXT, False),
    TUIBindingSpec("slash", "filter", "Filter", "Filter clauses.", _CANONICAL_CONTEXT),
    TUIBindingSpec(
        "k",
        "prune",
        "Mark hidden",
        "Hide the selected clause locally.",
        _CANONICAL_CONTEXT,
    ),
    TUIBindingSpec(
        "enter",
        "inspect",
        "Inspect",
        "Inspect the selected clause.",
        _CANONICAL_CONTEXT,
        False,
    ),
)


_DISPLAY_KEY_OVERRIDES: Mapping[str, str] = MappingProxyType(
    {
        "question_mark": "?",
        "ctrl+l": "Ctrl+L",
    }
)


def display_key(key: str) -> str:
    """Return the display spelling for one Textual activation key."""

    if key.startswith("f") and key[1:].isdigit():
        return key.upper()
    return _DISPLAY_KEY_OVERRIDES.get(key, key)


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
        lines.extend(("", "Classic compatibility views:", f"  {', '.join(views)}"))
    lines.extend(("", "Manual:"))
    lines.extend(f"  {manual_link}" for manual_link in value.manual_links)
    return "\n".join(lines)


def render_tui_help(view: str) -> tuple[str, str]:
    """Render a compact contextual title and body for the Textual modal."""

    values = tuple(topic(topic_id) for topic_id in TUI_VIEW_TOPICS[view])
    bindings = bindings_for_view(view)
    lines = [
        *(value.summary for value in values),
        "",
        "KEYBOARD",
    ]
    lines.extend(
        f"{binding.display_key:<6}{binding.description}" for binding in bindings
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
    if topic_id == "workbench":
        return CANONICAL_TUI_BINDINGS
    views = tuple(
        view for view, owners in TUI_VIEW_TOPICS.items() if topic_id in owners
    )
    return tuple(
        binding
        for binding in TUI_SEMANTIC_BINDINGS
        if any(view in binding.contexts for view in views)
    )


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
