"""Tests for M4 adapters: multiclass, convolutional/patch, regression, graph."""

import pytest

from prolog_tsetlin.adapters import MultiClassAdapter, PatchAdapter, RegressionAdapter
from prolog_tsetlin.graph.connectors import GraphConnector
from prolog_tsetlin.graph.types import GraphInput


def test_multiclass_basic() -> None:
    mc = MultiClassAdapter(source_field="label", classes=("a", "b", "c"))
    rec = {"label": "b"}
    out = mc.adapt(rec)
    assert out["label__mc_1"] == 1
    assert out["label__mc_0"] == 0
    assert mc.inverse({"label__mc_0": 0, "label__mc_1": 1, "label__mc_2": 0}) == "b"
    # roundtrip via dict
    assert MultiClassAdapter.from_dict(mc.to_dict()).classes == mc.classes
    # tie -> lowest index
    assert mc.inverse({"label__mc_0": 1, "label__mc_1": 1, "label__mc_2": 0}) == "a"


def test_multiclass_hostile() -> None:
    with pytest.raises(ValueError):
        MultiClassAdapter(source_field="", classes=("a", "b"))
    with pytest.raises(ValueError):
        MultiClassAdapter(source_field="x", classes=("a",))
    with pytest.raises(ValueError):
        MultiClassAdapter(source_field="x", classes=("a", "a"))
    with pytest.raises(ValueError):
        MultiClassAdapter.from_dict({"source_field": "x", "classes": ["a"], "schema": "bad"})  # type: ignore[arg-type]


def test_patch_basic() -> None:
    pa = PatchAdapter(field_prefix="pix", rows=2, cols=3, kernel_rows=1, kernel_cols=2, stride_rows=1, stride_cols=1)
    # 2x3 = 6 fields pix_0000 .. pix_0005
    base = {f"pix_{i:04d}": i for i in range(6)}
    patches = list(pa.iter_patches(base))
    assert len(patches) == 4  # (2-1+1)*(3-2+1)=2*2
    assert patches[0]["patch_0_0"] == 0
    assert patches[0]["patch_0_1"] == 1
    assert patches[1]["patch_0_0"] == 1
    # dict roundtrip
    assert PatchAdapter.from_dict(pa.to_dict()).patch_count() == pa.patch_count()


def test_patch_hostile_kernel_larger() -> None:
    with pytest.raises(ValueError):
        PatchAdapter(field_prefix="pix", rows=2, cols=2, kernel_rows=3, kernel_cols=3)


def test_patch_missing_field() -> None:
    pa = PatchAdapter(field_prefix="pix", rows=2, cols=2, kernel_rows=1, kernel_cols=1)
    with pytest.raises(KeyError):
        list(pa.iter_patches({"pix_0000": 0}))


def test_regression_basic() -> None:
    rg = RegressionAdapter(source_field="y", thresholds=(0.0, 10.0, 20.0))
    out = rg.adapt({"y": 15})
    assert out["y__band_0"] == 1
    assert out["y__band_1"] == 1
    assert out["y__band_2"] == 0
    # inverse: ones=2 -> midpoint 10-20 => 15
    assert rg.inverse({"y__band_0": 1, "y__band_1": 1, "y__band_2": 0}) == 15.0
    # below lowest
    assert rg.inverse({"y__band_0": 0}) == -1.0
    # above highest
    assert rg.inverse({"y__band_0": 1, "y__band_1": 1, "y__band_2": 1}) == 21.0
    assert RegressionAdapter.from_dict(rg.to_dict()).thresholds == rg.thresholds


def test_regression_hostile() -> None:
    with pytest.raises(ValueError):
        RegressionAdapter(source_field="y", thresholds=(10.0, 5.0))
    with pytest.raises(ValueError):
        RegressionAdapter(source_field="", thresholds=(1.0,))
    with pytest.raises(ValueError):
        RegressionAdapter.from_dict({"source_field": "y", "thresholds": []})  # type: ignore[arg-type]


def test_graph_adapter() -> None:
    gc = GraphConnector()
    rec = {"nodes": 2, "edges": [(0, 1, 0)], "props": {0: ["a"], 1: ["b"]}}
    g = gc.adapt(rec)
    assert isinstance(g, GraphInput)
    assert g.node_count == 2
    assert len(g.edges) == 1


def test_patch_and_multiclass_iter_adapt() -> None:
    mc = MultiClassAdapter(source_field="lab", classes=(0, 1))
    rows = list(mc.iter_adapt([{"lab": 0}, {"lab": 1}]))
    assert rows[0]["lab__mc_0"] == 1
    rg = RegressionAdapter(source_field="v", thresholds=(0.0, 5.0))
    rows2 = list(rg.iter_adapt([{"v": 2}, {"v": 7}]))
    assert rows2[0]["v__band_0"] == 1 and rows2[0]["v__band_1"] == 0
