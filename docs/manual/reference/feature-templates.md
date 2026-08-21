# Typed feature-template reference

Feature templates are parameterized literal generators. They add descriptors
to the existing `LiteralCatalog`; they do not create a separate feature or
model representation.

## Data types

`DataType` contains `continuous`, `discrete`, `categorical`, `binary`, `text`,
`temporal`, `spatial`, and `sensor`.

`SensorType` contains temperature, humidity, pressure, accelerometer,
gyroscope, magnetometer, light, sound, proximity, GPS, and custom sensor
labels. A sensor label is template metadata; portable semantics still come
from the generated literal descriptors.

## Built-in templates

| Template ID | Intended input | Required parameters | Generated literals |
| --- | --- | --- | --- |
| `numeric_threshold_v1` | numeric | `thresholds`; optional `inclusive` | one numeric greater-than-or-equal literal per threshold |
| `numeric_range_v1` | numeric | `ranges`; optional inclusivity flags | numeric interval literals |
| `categorical_v1` | category or Boolean | `categories` | type-strict category-equality literals |
| `sensor_data_v1` | numeric sensor | `sensor_ranges`, `threshold_percentiles` | bounded sensor thresholds and normal-range descriptors |
| `text_token_v1` | text-derived token field | `tokens`; optional `case_sensitive` | token-presence literals |

Template specifications map each schema field to a `template_id` plus its
parameters. `create_feature_template_catalog` resolves IDs through the default
registry, generates descriptors in declared order, and returns the catalog and
per-field descriptors.

## Custom templates

A custom template subclasses `FeatureTemplate`, implements
`generate_literals(field_name, catalog, **kwargs)`, and registers a stable
template ID with `TemplateRegistry`. Parameters are described by immutable
`TemplateParameter` values. Custom generators must use catalog methods so
literal identity and preprocessing semantics remain centralized.

## Clause-configuration analysis

`analyze_clause_configuration` returns a `TAClauseConfiguration` describing
the trained scalar machine and one `ClauseConfiguration` per clause.

| Field | Meaning |
| --- | --- |
| `clause_index` | zero-based clause identifier |
| `included_literals` | literal indices in Include state |
| `excluded_literals` | literal indices outside Include state |
| `polarity` | positive or negative clause vote |
| `activation_count` | analyzed samples on which the clause activated |
| `avg_activation_rate` | activation count divided by sample count |
| `contribution_score` | signed contribution metric over the analyzed samples |

The complete configuration can be serialized with `save` or `to_dict`.
Individual Python signatures and exceptions belong to generated API reference
in documentation checkpoint 4 rather than this authored page.

See [Create a typed feature catalog](../how-to/create-feature-catalog.md) for a
worked procedure and `examples/feature_templates_example.py` for the full
training example.
