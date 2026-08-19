"""PTAReasoningSession — shared knowledge base boundary for the collective.

Python owns validation and safe fact serialization; GNU Prolog consumes those
facts and lets Input, De-escalation and Escalation PTAs communicate through
the shared ontology. Output remains the narrow typed proposal protocol.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .proposal import PTAEscalationProposal, PTAInsight


def _prolog_atom(value: str) -> str:
    """Encode Python string as safe Prolog atom (single-quoted, escaped)."""
    if not isinstance(value, str):
        raise TypeError("atom must be str")
    # Prolog atom escaping: single quote doubled, backslash escaped, control chars \x..
    escaped = value.replace("\\", "\\\\").replace("'", "''").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    # Also escape other control chars
    out = []
    for ch in escaped:
        o = ord(ch)
        if o < 0x20 and ch not in ("\n", "\r", "\t"):
            out.append(f"\\x{o:02x}\\")
        else:
            out.append(ch)
    # Re-assemble (note: we already handled common ones)
    # The above double-handles; simpler: rebuild from original with proper escaping
    # Use canonical approach: escape ', \, and control
    s = value
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "''")
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    # For remaining control chars 0x00-0x1F not yet handled, use hex
    res = []
    for ch in s:
        if ord(ch) < 0x20 and ch not in ("\\", "n", "r", "t"):
            # This case shouldn't happen after replacements, but handle
            res.append(f"\\x{ord(ch):02x}\\")
        else:
            # s already contains escape sequences like \n as two chars; keep them
            res.append(ch)
    # Actually simpler: iterate original value
    parts = ["'"]
    for ch in value:
        if ch == "'":
            parts.append("''")
        elif ch == "\\":
            parts.append("\\\\")
        elif ch == "\n":
            parts.append("\\n")
        elif ch == "\r":
            parts.append("\\r")
        elif ch == "\t":
            parts.append("\\t")
        elif ord(ch) < 0x20:
            parts.append(f"\\x{ord(ch):02x}\\")
        else:
            parts.append(ch)
    parts.append("'")
    return "".join(parts)


def _prolog_term(value: Any) -> str:
    """Encode Python value as safe Prolog term in small typed grammar."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        # Prolog atoms true/false (lowercase), not True/False variables
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float not allowed in Prolog term")
        return repr(value)
    if isinstance(value, str):
        return _prolog_atom(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        inner = ",".join(_prolog_term(v) for v in value)
        return f"[{inner}]"
    if isinstance(value, Mapping):
        # Encode as dict-like list of pairs
        inner = ",".join(f"{_prolog_atom(str(k))}-{_prolog_term(v)}" for k, v in sorted(value.items(), key=lambda kv: str(kv[0])))
        return f"[{inner}]"
    raise TypeError(f"unsupported Prolog term type: {type(value).__name__}")


@dataclass
class PTAReasoningSession:
    dataset_id: str
    generation: int = 0
    max_observations: int = 1024
    max_insights: int = 256
    observations: list[tuple[str, int, str, Any]] = field(default_factory=list)
    example_labels: list[tuple[int, int]] = field(default_factory=list)
    example_domains: set[int] = field(default_factory=set)
    literal_truths: list[tuple[int, int, int]] = field(default_factory=list)
    clause_truths: list[tuple[int, int, int]] = field(default_factory=list)
    clause_literals: list[tuple[int, int]] = field(default_factory=list)
    counterexamples: list[tuple[str, int, int, int]] = field(default_factory=list)
    insights: list[PTAInsight] = field(default_factory=list)
    proposals: list[PTAEscalationProposal] = field(default_factory=list)

    def add_observation(self, pta: str, example: int, field: str, raw_value: Any) -> None:
        if not isinstance(pta, str) or not pta or any(ord(c) < 0x20 for c in pta):
            raise ValueError("pta must be nonempty printable string")
        if type(example) is not int or isinstance(example, bool):
            raise ValueError("example must be strict int")
        if not isinstance(field, str) or not field or any(ord(c) < 0x20 for c in field):
            raise ValueError("field must be nonempty printable string")
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            raise ValueError("raw_value float must be finite")
        if isinstance(raw_value, str) and len(raw_value) > 1024:
            raise ValueError("raw_value string too long")
        if len(self.observations) >= self.max_observations:
            raise ValueError("observation budget exceeded")
        self.observations.append((pta, example, field, raw_value))

    def add_example_label(self, example: int, label: int) -> None:
        if type(example) is not int or isinstance(example, bool):
            raise ValueError("example must be strict int")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be 0/1")
        self.example_labels.append((example, label))
        self.example_domains.add(example)

    def add_example_domain(self, example: int) -> None:
        if type(example) is not int or isinstance(example, bool):
            raise ValueError("example must be strict int")
        self.example_domains.add(example)

    def add_literal_truth(self, literal: int, example: int, truth: int) -> None:
        if type(literal) is not int or type(example) is not int or type(truth) is not int:
            raise ValueError("literal_truth must be ints")
        if truth not in (0, 1):
            raise ValueError("truth must be 0/1")
        self.literal_truths.append((literal, example, truth))

    def add_counterexample(self, model: str, example: int, expected: int, actual: int) -> None:
        if not isinstance(model, str) or not model or any(ord(c) < 0x20 for c in model):
            raise ValueError("model must be nonempty printable")
        if type(example) is not int or type(expected) is not int or type(actual) is not int:
            raise ValueError("counterexample ints must be strict ints")
        self.counterexamples.append((model, example, expected, actual))

    def add_insight(self, insight: PTAInsight) -> None:
        if len(self.insights) >= self.max_insights:
            raise ValueError("insight budget exceeded")
        # Validate insight strings are printable
        for s in (insight.source_pta, insight.kind, insight.subject):
            if not isinstance(s, str) or any(ord(c) < 0x20 for c in s):
                raise ValueError("insight strings must be printable")
        self.insights.append(insight)

    def add_proposal(self, proposal: PTAEscalationProposal) -> None:
        self.proposals.append(proposal)

    def to_prolog_facts(self) -> str:
        lines: list[str] = ["% PTAReasoningSession facts — auto-generated, bounded, safe-encoded"]
        for pta, ex, fld, val in self.observations:
            lines.append(f"observation({_prolog_atom(pta)},{ex},{_prolog_atom(fld)},{_prolog_term(val)}).")
        for ex in sorted(self.example_domains):
            lines.append(f"example_domain({ex}).")
        for ex, label in self.example_labels:
            lines.append(f"example_label({ex},{label}).")
        for lit, ex, truth in self.literal_truths:
            lines.append(f"literal_truth({lit},{ex},{truth}).")
        for cid, ex, truth in self.clause_truths:
            lines.append(f"clause_truth({cid},{ex},{truth}).")
        for cid, lit in self.clause_literals:
            lines.append(f"clause_literal({cid},{lit}).")
        for ins in self.insights:
            lines.append(f"insight({_prolog_atom(ins.source_pta)},{_prolog_atom(ins.kind)},{_prolog_atom(ins.subject)},{_prolog_term(list(ins.evidence))}).")
        for model, ex, exp, act in self.counterexamples:
            lines.append(f"counterexample({_prolog_atom(model)},{ex},{exp},{act}).")
        for prop in self.proposals:
            lines.append(f"proposal({_prolog_atom(prop.proposal_id)},{_prolog_atom(prop.native_target)},{_prolog_atom(prop.proposal_hash())}).")
        return "\n".join(lines) + "\n"

    def consult_via_gprolog(self, *, timeout: float = 2.0) -> str:
        """Write facts to temp file and consult via GNU Prolog, returning query output.

        This is the operating collective path — not just serialization.
        Requires gprolog in PATH.
        """
        import subprocess, tempfile, os

        prolog_src = self.to_prolog_facts()
        # Include ontology
        from pathlib import Path

        ont_path = Path(__file__).parents[3] / "prolog" / "pta_ontology.pl"
        try:
            ont = ont_path.read_text(encoding="utf-8")
        except Exception:
            ont = "% ontology unavailable"
        with tempfile.TemporaryDirectory() as tmpdir:
            facts_path = Path(tmpdir) / "pta_facts.pl"
            driver_path = Path(tmpdir) / "driver.pl"
            facts_path.write_text(prolog_src, encoding="utf-8")
            driver_path.write_text(
                f":- include('{ont_path.as_posix()}').\n:- include('{facts_path.as_posix()}').\n:- initialization((findall(P, proposal(P,_,_), Ps), write(Ps), nl, halt)).\n",
                encoding="utf-8",
            )
            try:
                out = subprocess.run(
                    ["gprolog", "--consult-file", str(driver_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return out.stdout + out.stderr
            except FileNotFoundError:
                return "gprolog not found — facts serialized but not consulted"
            except subprocess.TimeoutExpired:
                return "gprolog timeout"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "generation": self.generation,
            "observations": list(self.observations),
            "example_labels": list(self.example_labels),
            "insights": [{"source_pta": i.source_pta, "kind": i.kind, "subject": i.subject, "evidence": list(i.evidence)} for i in self.insights],
            "proposals": [p.to_dict() for p in self.proposals],
        }
