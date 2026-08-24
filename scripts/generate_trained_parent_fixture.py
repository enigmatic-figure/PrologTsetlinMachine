"""Regenerate the trained-parent PTA child golden artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from prolog_tsetlin.model_generation import (
    CorpusExample,
    CorpusRole,
    LabeledCorpus,
    LifecycleCorpora,
    OrderedLiteralManifest,
    PromotionAuditPolicy,
)
from prolog_tsetlin.reference import ScalarBinaryTsetlinMachine
from prolog_tsetlin.representation import FeatureSchema, FieldKind, LiteralCatalog
from prolog_tsetlin.services.model_generation import (
    ModelGenerationStore,
    execute_trained_parent_lifecycle,
    invent_threshold_for_corpus,
)


ROOT = Path(__file__).resolve().parents[1]


def _corpus(
    role: CorpusRole,
    first_id: int,
    values: tuple[int, ...],
    labels: tuple[int, ...],
) -> LabeledCorpus:
    return LabeledCorpus(
        "thermostat-generation-v1",
        role,
        tuple(
            CorpusExample(
                first_id + index,
                {
                    "temperature": value,
                    "mode": "heat",
                    "previous_state": False,
                },
                label,
            )
            for index, (value, label) in enumerate(zip(values, labels))
        ),
    )


def build_fixture(ptmrt: Path) -> bytes:
    parent_training = _corpus(
        CorpusRole.PARENT_TRAINING,
        0,
        (50, 55, 60, 65, 85, 90, 95, 100),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    corpora = LifecycleCorpora(
        _corpus(CorpusRole.INVENTION, 100, (62, 72, 78, 88), (0, 0, 1, 1)),
        _corpus(
            CorpusRole.ADAPTATION,
            200,
            (58, 64, 68, 71, 79, 82, 87, 92),
            (0, 0, 0, 0, 1, 1, 1, 1),
        ),
        _corpus(
            CorpusRole.PROMOTION,
            300,
            (61, 66, 73, 74, 76, 81, 86, 89),
            (0, 0, 0, 0, 1, 1, 1, 1),
        ),
        _corpus(
            CorpusRole.LIVE,
            400,
            (59, 63, 69, 72, 78, 83, 88, 94),
            (1, 1, 1, 1, 0, 0, 0, 0),
        ),
    )
    schema = FeatureSchema.from_fields(
        temperature=FieldKind.NUMBER,
        mode=FieldKind.CATEGORY,
        previous_state=FieldKind.BOOLEAN,
    )
    catalog = LiteralCatalog(schema)
    catalog.category_eq("mode", "heat")
    catalog.category_eq("previous_state", True)
    manifest = OrderedLiteralManifest.from_catalog(catalog)
    parent = ScalarBinaryTsetlinMachine(
        20,
        2,
        states_per_action=20,
        specificity=3.0,
        threshold=10,
        seed=7,
    )
    parent.fit_literal_batch(
        catalog.encode(parent_training.records).ta,
        parent_training.labels,
        epochs=150,
    )
    session, _, reviewed = invent_threshold_for_corpus(
        corpora.invention,
        manifest,
        numeric_field="temperature",
    )
    with TemporaryDirectory(prefix="ptm-generation-fixture-") as temporary:
        result = execute_trained_parent_lifecycle(
            parent_snapshot=parent.snapshot(),
            parent_manifest=manifest,
            parent_training_corpus=parent_training,
            corpora=corpora,
            invention_session=session,
            reviewed=reviewed,
            adaptation_epochs=5,
            promotion_policy=PromotionAuditPolicy(8),
            store=ModelGenerationStore(temporary),
            ptmrt_executable=ptmrt,
        )
    return result.child_artifact.serialized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "data" / "trained_parent_child_v1.hex",
    )
    parser.add_argument(
        "--ptmrt",
        type=Path,
        default=ROOT / "out" / "build" / "Release" / "ptmrt.exe",
    )
    arguments = parser.parse_args()
    serialized = build_fixture(arguments.ptmrt)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(serialized.hex() + "\n", encoding="ascii")
    print(f"wrote {arguments.output} ({len(serialized)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
