# Create a typed feature catalog

Use feature templates to generate a stable `LiteralCatalog` for typed records,
then inspect a trained model's clause configuration.

## Define the schema and templates

```python
from prolog_tsetlin import FeatureSchema, FieldKind, create_feature_template_catalog

schema = FeatureSchema.from_fields(
    temperature=FieldKind.NUMBER,
    room=FieldKind.CATEGORY,
)
specs = {
    "temperature": {
        "template_id": "numeric_threshold_v1",
        "thresholds": [20, 25, 30, 35],
    },
    "room": {
        "template_id": "categorical_v1",
        "categories": ["kitchen", "bedroom"],
    },
}
catalog, generated = create_feature_template_catalog(schema, specs)
```

The catalog assigns stable literal identities and preserves generated order.
Encode records through that catalog before training.

Encoding enforces the declared `FieldKind` before it emits either Boolean
literals or typed facts. Numbers reject Booleans and non-finite floats,
category comparisons are type-strict (`True`, `1`, and `"1"` differ), Boolean
fields accept only `bool`, and text fields accept strings. Absent and explicit
null values remain governed by each literal's null policy.

## Analyze clause configuration

```python
from prolog_tsetlin import analyze_clause_configuration

configuration = analyze_clause_configuration(machine, batch.ta, targets)
configuration.save("out/clause-configuration.json")
```

The result records included and excluded literals, polarity, activation, and
contribution metrics for each clause. Create the destination directory before
saving.

## Run the complete example

```bash
python examples/feature_templates_example.py
```

See the [feature-template reference](../reference/feature-templates.md) for
built-in template IDs, parameters, data types, and metric definitions.
