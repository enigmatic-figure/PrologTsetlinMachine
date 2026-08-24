# Capability maturity

> Transitional baseline: this table is hand-maintained alongside the registry until
> the maturity renderer and evidence validation are implemented.

This page lists PTM capabilities and their evidence-backed maturity stages. The registry `docs/_meta/capabilities.json` is the single source of truth.

| Capability | Maturity | Evidence |
|---|---|---|
| packed-tm-inference | stable | `docs/architecture/packed-tm.md` |
| cuda-packed-tm | experimental | `docs/architecture/cuda-packed-tm.md` |
| ptmrt-runtime | stable | `docs/manual/reference/artifact-contract.md` |
| bounded-search | stable | `docs/manual/reference/search-contracts.md` |
| pta-control-plane | proposed | `docs/rfcs/pta-control-plane.md` |
| pta-collective | experimental | `docs/developer/prolog-api.md`, `tests/python/test_pta_collective.py` |
| pta-threshold-native-slice | experimental | `docs/architecture/pta-threshold-materialization.md`, cross-runtime tests |
| tui-workbench | experimental | `docs/manual/how-to/tui.md` |

See [Architecture and contracts](../../architecture/index.md) for contracts and [RFCs](../../rfcs/index.md) for proposed directions.
