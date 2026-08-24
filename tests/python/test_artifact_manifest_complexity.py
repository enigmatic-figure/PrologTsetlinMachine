from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from prolog_tsetlin import (
    MODEL_MANIFEST_MAX_BYTES,
    MODEL_MANIFEST_MAX_DEPTH,
    MODEL_MANIFEST_MAX_NODES,
)
from prolog_tsetlin.model_artifact import (
    ModelArtifactError,
    export_packed_tm,
    load_model_artifact_from_bytes,
)
from prolog_tsetlin.reference import ScalarBinaryTsetlinMachine


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"
PORTABLE_FIXTURES = (
    ("xor_packed_tm_v1.hex", 69),
    ("conditional_logic_program_v1.hex", 86),
    ("masked_threshold_v1.hex", 93),
)


def _read_fixture(filename: str) -> bytes:
    return bytes.fromhex((DATA / filename).read_text(encoding="ascii"))


def _manifest(serialized: bytes) -> dict[str, object]:
    manifest_size = int.from_bytes(serialized[24:32], "little")
    value = json.loads(serialized[64 : 64 + manifest_size])
    assert isinstance(value, dict)
    return value


def _count_values(value: object) -> tuple[int, int]:
    nodes = 0
    maximum_depth = 0
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        maximum_depth = max(maximum_depth, depth)
        if isinstance(current, dict):
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
    return nodes, maximum_depth


def _replace_manifest(serialized: bytes, manifest: dict[str, object]) -> bytes:
    old_manifest_size = int.from_bytes(serialized[24:32], "little")
    payload_size = int.from_bytes(serialized[32:40], "little")
    payload_first = 64 + old_manifest_size
    manifest_bytes = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    header = bytearray(serialized[:64])
    header[24:32] = len(manifest_bytes).to_bytes(8, "little")
    content = bytes(header) + manifest_bytes + serialized[
        payload_first : payload_first + payload_size
    ]
    return content + hashlib.sha256(content).digest()


def _nested_probe(container_count: int) -> object:
    value: object = None
    for _ in range(container_count):
        value = {"x": value}
    return value


def test_python_and_public_c_contract_constants_cannot_drift() -> None:
    header = (ROOT / "include" / "ptm" / "runtime.h").read_text(encoding="utf-8")
    expected = {
        "PTMRT_MODEL_MANIFEST_MAX_BYTES": MODEL_MANIFEST_MAX_BYTES,
        "PTMRT_MODEL_MANIFEST_MAX_DEPTH": MODEL_MANIFEST_MAX_DEPTH,
        "PTMRT_MODEL_MANIFEST_MAX_NODES": MODEL_MANIFEST_MAX_NODES,
    }
    for name, value in expected.items():
        match = re.search(rf"^#define {name} (.+)$", header, re.MULTILINE)
        assert match is not None
        factors = tuple(int(item) for item in re.findall(r"\d+", match.group(1)))
        assert math.prod(factors) == value


@pytest.mark.parametrize(("filename", "base_nodes"), PORTABLE_FIXTURES)
def test_python_accepts_each_portable_kind_at_exact_complexity_limits(
    filename: str, base_nodes: int
) -> None:
    serialized = _read_fixture(filename)
    manifest = _manifest(serialized)
    assert _count_values(manifest) == (base_nodes, 5)

    manifest["zz_portability_probe"] = _nested_probe(
        MODEL_MANIFEST_MAX_DEPTH - 1
    )
    assert _count_values(manifest)[1] == MODEL_MANIFEST_MAX_DEPTH
    load_model_artifact_from_bytes(_replace_manifest(serialized, manifest))

    manifest = _manifest(serialized)
    manifest["zz_portability_probe"] = [
        None
    ] * (MODEL_MANIFEST_MAX_NODES - base_nodes - 1)
    assert _count_values(manifest)[0] == MODEL_MANIFEST_MAX_NODES
    load_model_artifact_from_bytes(_replace_manifest(serialized, manifest))


@pytest.mark.parametrize(("filename", "base_nodes"), PORTABLE_FIXTURES)
def test_python_rejects_each_portable_kind_one_past_complexity_limits(
    filename: str, base_nodes: int
) -> None:
    serialized = _read_fixture(filename)
    manifest = _manifest(serialized)
    manifest["zz_portability_probe"] = _nested_probe(MODEL_MANIFEST_MAX_DEPTH)
    assert _count_values(manifest)[1] == MODEL_MANIFEST_MAX_DEPTH + 1
    with pytest.raises(ModelArtifactError, match="nested too deeply"):
        load_model_artifact_from_bytes(_replace_manifest(serialized, manifest))

    manifest = _manifest(serialized)
    manifest["zz_portability_probe"] = [
        None
    ] * (MODEL_MANIFEST_MAX_NODES - base_nodes)
    assert _count_values(manifest)[0] == MODEL_MANIFEST_MAX_NODES + 1
    with pytest.raises(ModelArtifactError, match="too complex"):
        load_model_artifact_from_bytes(_replace_manifest(serialized, manifest))


@pytest.mark.parametrize(
    "validation_signature",
    (
        {"probe": _nested_probe(MODEL_MANIFEST_MAX_DEPTH)},
        {"probe": [None] * MODEL_MANIFEST_MAX_NODES},
    ),
)
def test_export_rejects_nonportable_manifest_before_publication(
    validation_signature: dict[str, object],
) -> None:
    machine = ScalarBinaryTsetlinMachine(2, 1, threshold=2, seed=3)
    with pytest.raises(ModelArtifactError):
        export_packed_tm(
            machine.snapshot(),
            name="nonportable",
            validation_signature=validation_signature,
        )


def test_exported_canonical_unicode_escape_round_trips() -> None:
    machine = ScalarBinaryTsetlinMachine(2, 1, threshold=2, seed=3)
    artifact = export_packed_tm(
        machine.snapshot(),
        name="control\x01character",
    )
    assert b"control\\u0001character" in artifact.serialized
    loaded = load_model_artifact_from_bytes(artifact.serialized)
    assert loaded.manifest["title"] == "control\x01character"
