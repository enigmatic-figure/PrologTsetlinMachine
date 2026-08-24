"""PTA communication ontology — minimal shared predicates.

Prolog and Python share these terms; Python mirrors them as dataclasses for
deterministic reasoning. Free-form terms are intentionally not allowed — every
insight that survives must be an instance of this ontology so a graph PTA can
consume a numeric de-escalation PTA's insight without re-discovery.

Prolog source of truth is prolog/pta_ontology.pl (compiled); Python reads it via shared resolver.
"""

from ..prolog_resources import resolve_prolog_module

def _resolve_ontology() -> str:
    return resolve_prolog_module("pta_ontology.pl").read_text(encoding="utf-8")

PROLOG_ONTOLOGY = _resolve_ontology()

__all__ = ["PROLOG_ONTOLOGY"]
