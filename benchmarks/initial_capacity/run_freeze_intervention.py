#!/usr/bin/env python3
"""Run the XOR parent-extension freeze intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from prolog_tsetlin.benchmark_campaign import CampaignDatasetManifest, load_dense_bit_split
from prolog_tsetlin.reference import ScalarBinaryTsetlinMachine


def _metrics(machine, rows, labels, clean_labels, flipped):
    predictions = machine.predict(rows)
    clean = sum(p == y for p, y in zip(predictions, clean_labels)) / len(rows)
    flipped_fit = sum(p == y for p, y, f in zip(predictions, labels, flipped) if f) / max(1, sum(flipped))
    included = []
    behaviors = set()
    # Behavior identity is measured on a fixed bounded probe to keep this
    # diagnostic separate from the full scoring pass.
    probe = rows[: min(len(rows), 1000)]
    for clause in range(machine.number_of_clauses):
        included.append(sum(machine.action_include(clause, lit) for lit in range(2 * machine.number_of_features)))
        behaviors.add(tuple(machine.clause_output(clause, row, prediction=True) for row in probe))
    state = {}
    for row, score in zip(rows, (machine.score(row) for row in rows)):
        key = f"{row[0]}{row[1]}"
        truth = row[0] ^ row[1]
        state.setdefault(key, []).append(score if truth else -score)
    return {
        "clean_accuracy": clean,
        "wrong_label_fit": flipped_fit,
        "unique_clause_behaviors": len(behaviors),
        "unique_signed_clause_behaviors": len({(polarity, tuple(machine.clause_output(clause, row, prediction=True) for row in probe)) for clause, polarity in ((i, 1 if i % 2 == 0 else -1) for i in range(machine.number_of_clauses))}),
        "mean_included_literals_per_clause": sum(included) / len(included),
        "state_mean_clean_margin": {key: sum(values) / len(values) for key, values in sorted(state.items())},
    }


def _adapt_new_only(machine, rows, labels, frozen_states, start_clause, epochs):
    for _ in range(epochs):
        for row, target in zip(rows, labels):
            target_value = int(machine._require_binary(target))
            literals = machine._literals(row)
            class_sum = machine.score(row)
            for clause in range(start_clause, machine.number_of_clauses):
                polarity = 1 if clause % 2 == 0 else -1
                probability = (machine.threshold + (1 - 2 * target_value) * polarity * class_sum) / (2 * machine.threshold)
                if machine._rng.random() > probability:
                    continue
                output = machine.clause_output(clause, row, prediction=False)
                target_polarity = (target_value == 1 and polarity == 1) or (target_value == 0 and polarity == -1)
                if target_polarity:
                    machine._type_i_feedback(clause, literals, output)
                else:
                    machine._type_ii_feedback(clause, literals, output)
    if tuple(tuple(v) for v in machine._states[:start_clause]) != frozen_states:
        raise RuntimeError("frozen clause state changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-root", type=Path, required=True)
    parser.add_argument("--noise", default="noise-02000bp")
    parser.add_argument("--parent-clauses", type=int, default=40)
    parser.add_argument("--total-clauses", type=int, default=400)
    parser.add_argument("--parent-epochs", type=int, default=5)
    parser.add_argument("--adaptation-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.total_clauses <= args.parent_clauses:
        parser.error("total clauses must exceed parent clauses")
    manifest_path = args.material_root / args.noise / "manifest.json"
    manifest = CampaignDatasetManifest.load(manifest_path)
    rows, noisy = load_dense_bit_split(manifest_path, manifest, "train")
    evaluation, evaluation_labels = load_dense_bit_split(manifest_path, manifest, "evaluation")
    clean = tuple(row[0] ^ row[1] for row in rows)
    flipped = tuple(a != b for a, b in zip(noisy, clean))
    machine = ScalarBinaryTsetlinMachine(args.parent_clauses, manifest.feature_count, states_per_action=128, specificity=3.9, threshold=15, seed=args.seed)
    machine.fit(rows, clean, epochs=args.parent_epochs)
    parent_snapshot = machine.snapshot()
    frozen_states = tuple(tuple(v) for v in parent_snapshot.states)
    frozen_rng = parent_snapshot.rng_state
    extended = ScalarBinaryTsetlinMachine(args.total_clauses, manifest.feature_count, states_per_action=128, specificity=3.9, threshold=15, seed=args.seed)
    extended._states[:args.parent_clauses] = [list(v) for v in frozen_states]
    extended._rng.setstate(frozen_rng)
    for clause in range(args.parent_clauses, args.total_clauses):
        extended._states[clause] = [extended.states_per_action] * (2 * manifest.feature_count)
    _adapt_new_only(extended, rows, noisy, frozen_states, args.parent_clauses, args.adaptation_epochs)
    all_adaptive = ScalarBinaryTsetlinMachine(args.total_clauses, manifest.feature_count, states_per_action=128, specificity=3.9, threshold=15, seed=args.seed)
    all_adaptive.fit(rows, noisy, epochs=args.adaptation_epochs)
    eval_clean = tuple(row[0] ^ row[1] for row in evaluation)
    result = {
        "schema": "ptm.xor-freeze-intervention.v1",
        "noise": args.noise,
        "parent_clauses": args.parent_clauses,
        "total_clauses": args.total_clauses,
        "parent_epochs": args.parent_epochs,
        "adaptation_epochs": args.adaptation_epochs,
        "seed": args.seed,
        "parent": {"train": _metrics(machine, rows, noisy, clean, flipped), "evaluation": _metrics(machine, evaluation, evaluation_labels, eval_clean, tuple(False for _ in evaluation))},
        "all_adaptive": {"train": _metrics(all_adaptive, rows, noisy, clean, flipped), "evaluation": _metrics(all_adaptive, evaluation, evaluation_labels, eval_clean, tuple(False for _ in evaluation))},
        "new_clauses_only": {"train": _metrics(extended, rows, noisy, clean, flipped), "evaluation": _metrics(extended, evaluation, evaluation_labels, eval_clean, tuple(False for _ in evaluation))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
