#!/usr/bin/env python3
"""Convert the classic Python-2 MNIST pickle into compact PTM bit material."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import struct

import numpy as np


MAGIC = b"PTMMNIST"
VERSION = 1


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_split(path: Path, features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    bits = np.ascontiguousarray(features, dtype=np.uint8)
    targets = np.ascontiguousarray(labels, dtype=np.uint8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sIII", MAGIC, VERSION, bits.shape[0], bits.shape[1]))
        stream.write(bits.tobytes(order="C"))
        stream.write(targets.tobytes(order="C"))
    return {
        "path": path.name,
        "rows": int(bits.shape[0]),
        "features": int(bits.shape[1]),
        "positive_cells": int(bits.sum()),
        "digest": _digest(path),
        "class_counts": {str(value): int((targets == value).sum()) for value in range(10)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between zero and one")
    source_bytes = args.source.read_bytes()
    # The supplied 2014 artifact is the classic Python-2 pickle.
    payload = pickle.loads(source_bytes, encoding="latin1")
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise ValueError("MNIST pickle must contain train, validation, and test splits")
    args.output.mkdir(parents=True, exist_ok=True)
    splits: dict[str, object] = {}
    for name, split in zip(("train", "validation", "test"), payload):
        if not isinstance(split, tuple) or len(split) != 2:
            raise ValueError(f"MNIST {name} split is malformed")
        features, labels = split
        features = np.asarray(features)
        labels = np.asarray(labels)
        if features.ndim != 2 or features.shape[1] != 784 or labels.shape != (features.shape[0],):
            raise ValueError(f"MNIST {name} split has the wrong shape")
        if labels.min() != 0 or labels.max() != 9:
            raise ValueError(f"MNIST {name} labels are outside 0..9")
        splits[name] = _write_split(
            args.output / f"{name}.ptmb", features > args.threshold, labels
        )
    manifest = {
        "schema": "ptm.mnist-bits.v1",
        "source_digest": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        "threshold": args.threshold,
        "threshold_rule": "pixel > threshold",
        "splits": splits,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
