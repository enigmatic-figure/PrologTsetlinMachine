# PTM manual

The manual is for people installing, exploring, running, and embedding PTM.
Choose a tutorial for a guided first result, a how-to guide for a specific
task, reference for exact facts, or explanation for design context.

## Tutorials

- [First Python model](tutorials/first-python-model.md)
- [First native consumer](tutorials/first-native-consumer.md)
- [First terminal-workbench session](tutorials/first-tui-session.md)

## How-to guides

- [Install PTM](how-to/install.md)
- [Use the terminal workbench](how-to/tui.md)
- [Run bounded symbolic search](how-to/run-bounded-search.md)
- [Export, inspect, and verify artifacts](how-to/export-artifacts.md)
- [Embed the native runtime](how-to/embed-ptmrt.md)
- [Create a typed feature catalog](how-to/create-feature-catalog.md)

## Reference

- [Generated help topics and workbench controls](reference/help-topics.md)
- [Portable artifact and runtime reference](reference/artifact-contract.md)
- [Deterministic preprocessing contract](reference/preprocessing.md)
- [Bounded-search contracts](reference/search-contracts.md)
- [Streaming data connectors and transforms](reference/data-connectors.md)
- [Typed feature templates](reference/feature-templates.md)
- [C ABI](reference/c-api.md)

For tested version combinations, see the release-owned
[compatibility matrix](../compatibility.md).

## Explanation

- [Portable deployment](explanation/portable-deployment.md)
- [Host pipelines and portable preprocessing](explanation/host-pipelines.md)
- [Why symbolic search is bounded](explanation/symbolic-search.md)
- [How the workbench protects state](explanation/workbench.md)

```{toctree}
:hidden:
:maxdepth: 2

tutorials/first-python-model
tutorials/first-native-consumer
tutorials/first-tui-session
how-to/install
how-to/tui
how-to/run-bounded-search
how-to/export-artifacts
how-to/embed-ptmrt
how-to/create-feature-catalog
reference/help-topics
reference/artifact-contract
reference/preprocessing
reference/search-contracts
reference/data-connectors
reference/feature-templates
reference/c-api
explanation/portable-deployment
explanation/host-pipelines
explanation/symbolic-search
explanation/workbench

../../README
../../INSTALL
../consumer-tutorial
../tui
../bounded-search
../model-export-runtime
../data-connectors
../c-api
../preprocessing-contract
../feature-templates
```
