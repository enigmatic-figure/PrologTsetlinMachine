"""GNU Prolog search -> Class II artifact -> native fixed-shape kernel."""

from prolog_tsetlin import (
    FixedBitBlock,
    GNUPrologThresholdSearch,
    InputShape,
    NativePAKernel,
    PortSemantic,
    RestorationHandle,
    SlotBinding,
    SourceKind,
    ThresholdSearchProblem,
)


problem = ThresholdSearchProblem.create(
    slot_count=3,
    max_selected=3,
    positive_examples=[{0}, {1}, {0, 1}, {0, 2}],
    negative_examples=[set(), {2}],
)
bindings = [
    SlotBinding(slot, SourceKind.TA, f"ta-{slot}") for slot in range(3)
]
artifact = GNUPrologThresholdSearch().search_artifact(
    problem,
    input_shape=InputShape.PA_32X32,
    port_semantic=PortSemantic.TA_ACTION,
    mapping_version="example-map-v1",
    slot_bindings=bindings,
    restoration_handle=RestorationHandle(1, "snapshot:example-before-compile"),
)

print(artifact.to_json())

inputs = FixedBitBlock(1024, PortSemantic.TA_ACTION)
inputs.set(1, True)
result = NativePAKernel().evaluate_artifact(artifact, inputs)
print("native result:", result.value, "matches:", result.matched_count)
