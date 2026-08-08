# Typed Feature Templates and TA-Clause Configuration

This document describes the typed feature-template system and TA-clause-configuration export functionality added to the Prolog Tsetlin Machine project.

## Overview

These enhancements expand the utility of cellular automata by providing:

1. **Typed Feature Templates**: Parameterized generators for creating literals optimized for different data types (sensor data, numeric thresholds, categorical values, text tokens)
2. **TA-Clause Configuration Exports**: Detailed analysis and serialization of trained clause configurations with contribution metrics

## Feature Template System

### Data Types

The system supports multiple data types inspired by IoT and sensor literature:

- `CONTINUOUS`: Numeric continuous values
- `DISCRETE`: Integer or discrete numeric values  
- `CATEGORICAL`: Named categories
- `BINARY`: Boolean values
- `TEXT`: Text/token data
- `TEMPORAL`: Time-series data
- `SPATIAL`: Geographic/spatial data
- `SENSOR`: Optimized for sensor readings

### Sensor Types

Pre-defined sensor types from embedded systems literature:

- Temperature, Humidity, Pressure
- Accelerometer, Gyroscope, Magnetometer
- Light, Sound, Proximity
- GPS, Custom

### Built-in Templates

#### 1. Numeric Threshold Template (`numeric_threshold_v1`)

Generates threshold-based literals for numeric fields.

```python
from prolog_tsetlin import FeatureSchema, FieldKind, create_feature_template_catalog

schema = FeatureSchema.from_fields(temperature=FieldKind.NUMBER)

template_specs = {
    'temperature': {
        'template_id': 'numeric_threshold_v1',
        'thresholds': [20, 25, 30, 35]  # Generate GE literals for each
    }
}

catalog, generated = create_feature_template_catalog(schema, template_specs)
```

#### 2. Numeric Range Template (`numeric_range_v1`)

Generates range-based BETWEEN literals.

```python
template_specs = {
    'pressure': {
        'template_id': 'numeric_range_v1',
        'ranges': [(980, 1000), (1000, 1020), (1020, 1040)]
    }
}
```

#### 3. Categorical Template (`categorical_v1`)

Generates equality literals for categorical fields.

```python
template_specs = {
    'room': {
        'template_id': 'categorical_v1',
        'categories': ['kitchen', 'bedroom', 'living_room', 'bathroom']
    }
}
```

#### 4. Sensor Data Template (`sensor_data_v1`)

Optimized for sensor data with percentile-based thresholds and normal range detection.

```python
template_specs = {
    'temperature': {
        'template_id': 'sensor_data_v1',
        'sensor_ranges': {
            'min': 0,
            'max': 50,
            'normal_min': 18,
            'normal_max': 26
        },
        'threshold_percentiles': [10, 25, 50, 75, 90]
    }
}
```

#### 5. Text Token Template (`text_token_v1`)

Generates token-presence literals for text fields.

```python
template_specs = {
    'description': {
        'template_id': 'text_token_v1',
        'tokens': ['error', 'warning', 'critical'],
        'case_sensitive': False
    }
}
```

### Creating Custom Templates

```python
from prolog_tsetlin import FeatureTemplate, TemplateParameter, DataType

class CustomTemplate(FeatureTemplate):
    def generate_literals(self, field_name, catalog, **kwargs):
        # Custom literal generation logic
        literals = []
        # ... generate literals using catalog methods
        return literals

template = CustomTemplate(
    template_id='my_custom_v1',
    name='My Custom Template',
    data_type=DataType.CONTINUOUS,
    parameters=(
        TemplateParameter('param1', DataType.DISCRETE, required=True),
        TemplateParameter('param2', DataType.BINARY, default=False),
    )
)

registry = TemplateRegistry()
registry.register(template)
```

## TA-Clause Configuration Export

After training a Tsetlin Machine, you can analyze and export the clause configuration:

```python
from prolog_tsetlin import (
    ScalarBinaryTsetlinMachine,
    analyze_clause_configuration,
)

# Train your model
machine = ScalarBinaryTsetlinMachine(...)
machine.fit_literal_batch(batch.ta, targets, epochs=100)

# Analyze clause configuration
config = analyze_clause_configuration(machine, batch.ta, targets)

# Access metrics
print(f"Clauses: {config.number_of_clauses}")
print(f"Features: {config.number_of_features}")

for clause_cfg in config.clause_configs:
    print(f"Clause {clause_cfg.clause_index}:")
    print(f"  Included literals: {len(clause_cfg.included_literals)}")
    print(f"  Activation rate: {clause_cfg.avg_activation_rate:.2f}")
    print(f"  Contribution score: {clause_cfg.contribution_score:+.3f}")

# Export to JSON
config.save('/path/to/clause_config.json')

# Or get as dict
config_dict = config.to_dict()
```

### Clause Configuration Metrics

Each clause configuration includes:

- `clause_index`: Clause identifier
- `included_literals`: List of literal indices included in the clause
- `excluded_literals`: List of literal indices excluded from the clause
- `polarity`: +1 for positive clauses, -1 for negative
- `activation_count`: Number of samples where clause activated
- `avg_activation_rate`: Fraction of samples where clause activated
- `contribution_score`: Impact on prediction accuracy (positive = helpful)

## Complete Example

See `/workspace/examples/feature_templates_example.py` for a complete working example demonstrating:

1. Schema definition for multi-type sensor data
2. Template specification for each field
3. Catalog creation using templates
4. Data encoding and TM training
5. Clause configuration analysis
6. JSON export of results

## API Reference

### Classes

- `DataType`: Enum of supported data types
- `SensorType`: Enum of sensor types
- `FeatureTemplate`: Base class for templates
- `NumericThresholdTemplate`: Threshold-based template
- `NumericRangeTemplate`: Range-based template
- `CategoricalTemplate`: Category equality template
- `SensorDataTemplate`: Sensor-optimized template
- `TextTokenTemplate`: Token presence template
- `TemplateRegistry`: Template registration and lookup
- `ClauseConfiguration`: Single clause analysis
- `TAClauseConfiguration`: Complete model analysis

### Functions

- `create_feature_template_catalog(schema, template_specs)`: Create catalog from templates
- `analyze_clause_configuration(machine, batch, targets)`: Analyze trained TM
- `export_template_schema(registry)`: Export available templates
- `load_template_from_dict(data)`: Load template from serialized form

## Integration with Existing Systems

The feature template system integrates seamlessly with:

- `LiteralCatalog`: Templates generate literals using existing catalog methods
- `ScalarBinaryTsetlinMachine`: Trained models can be analyzed with clause configuration
- `Dashboard`: Templates can be exposed through the Dear PyGUI interface
- `Model Artifacts`: Clause configurations complement exported .ptm files

## Future Enhancements

Potential extensions:

1. Temporal feature templates for time-series data
2. Spatial templates for geographic coordinates
3. Multi-modal sensor correlation templates
4. Automated template selection based on data analysis
5. Template hyperparameter optimization
6. Visualization of clause-literal relationships
