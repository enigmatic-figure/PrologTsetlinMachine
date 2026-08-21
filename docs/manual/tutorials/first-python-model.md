# First Python model

This tutorial starts from an installed PTM checkout, exports a deterministic
raw-record XOR model, and consumes it through the Python artifact API. At the
end, you will have reproduced predictions from a portable `.ptm` file.

## Before you begin

Complete the [Python core installation](../how-to/install.md#python-core-or-terminal-workbench).
Run the commands below from the repository root.

## Export the example artifact

```bash
python examples/export_raw_xor_artifact.py out/consumer/raw-xor.ptm
```

The example trains the scalar oracle, attaches a portable preprocessing
contract for the `left` and `right` Boolean fields, exports the artifact, and
reloads it before returning.

## Consume it from Python

Create `out/consumer/predict.py`:

```python
from prolog_tsetlin import load_model_artifact

model = load_model_artifact("out/consumer/raw-xor.ptm")
records = (
    {"left": False, "right": False},
    {"left": False, "right": True},
    {"left": True, "right": False},
    {"left": True, "right": True},
)
print(model.predict_records(records))
```

Run it:

```bash
python out/consumer/predict.py
```

Expected output:

```text
(0, 1, 1, 0)
```

You have now loaded a content-addressed artifact, applied its embedded typed
preprocessing contract, and reproduced XOR without retaining the training
object. Continue with the [first native consumer](first-native-consumer.md) to
run the same artifact outside Python.
