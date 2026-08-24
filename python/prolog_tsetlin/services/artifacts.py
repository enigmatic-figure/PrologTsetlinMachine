"""Artifact workflows shared by interactive PTM frontends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..model_artifact import export_packed_tm
from ..preprocessing import PreprocessingContract
from ..representation import FeatureSchema, FieldKind, LiteralCatalog
from ._atomic import publish_bytes
from .training import TrainingRun


@dataclass(frozen=True, slots=True)
class ArtifactExportRequest:
    """Reviewed metadata and destination for one frozen training run."""

    path: Path
    name: str = "xor-explorer"
    description: str = "XOR model trained in the PTM terminal workbench"
    author: str = ""
    license: str = "research"
    intended_use: str = "exploration"
    limitations: str = "Research prototype; validate before operational use."
    overwrite: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("artifact name cannot be empty")
        if not str(self.path).strip():
            raise ValueError("artifact path cannot be empty")
        if self.path.suffix.lower() != ".ptm":
            raise ValueError("artifact path must end in .ptm")


@dataclass(frozen=True, slots=True)
class ArtifactSummary:
    path: Path
    artifact_id: str
    artifact_kind: str
    byte_count: int
    conformance_examples: int


def export_training_run(
    run: TrainingRun,
    request: ArtifactExportRequest,
) -> ArtifactSummary:
    """Freeze a completed XOR run without silently replacing an existing file."""

    request.validate()
    destination = request.path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = FeatureSchema.from_fields(x0=FieldKind.BOOLEAN, x1=FieldKind.BOOLEAN)
    catalog = LiteralCatalog(schema)
    catalog.category_eq("x0", True)
    catalog.category_eq("x1", True)
    preprocessing = PreprocessingContract.from_catalog(catalog)
    validation_records = tuple(
        {"x0": bool(row[0]), "x1": bool(row[1])} for row in run.rows
    )
    artifact = export_packed_tm(
        run.snapshot,
        name=request.name.strip(),
        description=request.description.strip(),
        authors=(request.author.strip(),) if request.author.strip() else (),
        license=request.license.strip() or "unspecified",
        intended_use=request.intended_use.strip() or "research",
        limitations=request.limitations.strip() or "research prototype",
        feature_names=("x0", "x1"),
        feature_catalog_version="builtin-xor-v1",
        validation_rows=run.rows,
        preprocessing=preprocessing,
        validation_records=validation_records,
        validation_signature={
            "dataset": "builtin-xor-v1",
            "accuracy": run.accuracy,
            "seed": run.request.seed,
        },
    )
    publish_bytes(destination, artifact.serialized, overwrite=request.overwrite)
    validation = artifact.manifest.get("validation", {})
    return ArtifactSummary(
        path=destination,
        artifact_id=artifact.artifact_id,
        artifact_kind=str(artifact.manifest["artifact_kind"]),
        byte_count=len(artifact.serialized),
        conformance_examples=int(validation.get("conformance_example_count", 0)),
    )
