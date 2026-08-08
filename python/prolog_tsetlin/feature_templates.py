"""Typed feature templates and TA-clause configuration exports for Tsetlin Machines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterable, Mapping, Sequence
from pathlib import Path

from .representation import (
    FeatureSchema,
    FieldDefinition,
    FieldKind,
    LiteralCatalog,
    LiteralDescriptor,
    TransformKind,
    NullPolicy,
)


class DataType(str, Enum):
    """Supported data types for feature templates."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TEXT = "text"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SENSOR = "sensor"


class SensorType(str, Enum):
    """Common sensor types from IoT and embedded systems literature."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    LIGHT = "light"
    SOUND = "sound"
    PROXIMITY = "proximity"
    GPS = "gps"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class RangeConfig:
    """Configuration for numeric range parameters."""
    min_value: float
    max_value: float
    step: float | None = None
    
    def validate(self, value: float) -> bool:
        """Check if value is within configured range."""
        return self.min_value <= value <= self.max_value


@dataclass(frozen=True, slots=True)
class TemplateParameter:
    """A parameter definition for a feature template."""
    name: str
    data_type: DataType
    required: bool = True
    default: Any = None
    range_config: RangeConfig | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class FeatureTemplate:
    """A typed feature template with parameterized generators."""
    template_id: str
    name: str
    data_type: DataType
    sensor_type: SensorType | None = None
    parameters: tuple[TemplateParameter, ...] = ()
    description: str = ""
    version: int = 1
    
    def generate_literals(
        self,
        field_name: str,
        catalog: LiteralCatalog,
        **kwargs: Any,
    ) -> list[LiteralDescriptor]:
        """Generate literals based on template parameters."""
        raise NotImplementedError("Subclasses must implement generate_literals")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize template to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "data_type": self.data_type.value,
            "sensor_type": self.sensor_type.value if self.sensor_type else None,
            "parameters": [
                {
                    "name": p.name,
                    "data_type": p.data_type.value,
                    "required": p.required,
                    "default": p.default,
                    "range_config": {
                        "min_value": p.range_config.min_value,
                        "max_value": p.range_config.max_value,
                        "step": p.range_config.step,
                    } if p.range_config else None,
                    "description": p.description,
                }
                for p in self.parameters
            ],
            "description": self.description,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class NumericThresholdTemplate(FeatureTemplate):
    """Template for numeric threshold features."""
    
    def __post_init__(self) -> None:
        if not self.parameters:
            object.__setattr__(
                self, 
                "parameters",
                (
                    TemplateParameter(
                        name="thresholds",
                        data_type=DataType.DISCRETE,
                        required=True,
                        description="List of threshold values",
                    ),
                    TemplateParameter(
                        name="inclusive",
                        data_type=DataType.BINARY,
                        required=False,
                        default=True,
                        description="Whether thresholds are inclusive",
                    ),
                ),
            )
    
    def generate_literals(
        self,
        field_name: str,
        catalog: LiteralCatalog,
        thresholds: Sequence[float],
        inclusive: bool = True,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> list[LiteralDescriptor]:
        """Generate GE literals for each threshold."""
        literals = []
        for threshold in thresholds:
            literal = catalog.numeric_ge(
                field_name,
                threshold,
                null_policy=null_policy,
            )
            literals.append(literal)
        return literals


@dataclass(frozen=True, slots=True)
class NumericRangeTemplate(FeatureTemplate):
    """Template for numeric range features."""
    
    def __post_init__(self) -> None:
        if not self.parameters:
            object.__setattr__(
                self,
                "parameters",
                (
                    TemplateParameter(
                        name="ranges",
                        data_type=DataType.DISCRETE,
                        required=True,
                        description="List of (lower, upper) tuples",
                    ),
                    TemplateParameter(
                        name="inclusive_bounds",
                        data_type=DataType.BINARY,
                        required=False,
                        default=(True, True),
                        description="Tuple of (inclusive_lower, inclusive_upper)",
                    ),
                ),
            )
    
    def generate_literals(
        self,
        field_name: str,
        catalog: LiteralCatalog,
        ranges: Sequence[tuple[float, float]],
        inclusive_bounds: tuple[bool, bool] = (True, True),
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> list[LiteralDescriptor]:
        """Generate BETWEEN literals for each range."""
        literals = []
        for lower, upper in ranges:
            literal = catalog.numeric_between(
                field_name,
                lower,
                upper,
                inclusive_lower=inclusive_bounds[0],
                inclusive_upper=inclusive_bounds[1],
                null_policy=null_policy,
            )
            literals.append(literal)
        return literals


@dataclass(frozen=True, slots=True)
class CategoricalTemplate(FeatureTemplate):
    """Template for categorical features."""
    
    def __post_init__(self) -> None:
        if not self.parameters:
            object.__setattr__(
                self,
                "parameters",
                (
                    TemplateParameter(
                        name="categories",
                        data_type=DataType.CATEGORICAL,
                        required=True,
                        description="List of category values",
                    ),
                ),
            )
    
    def generate_literals(
        self,
        field_name: str,
        catalog: LiteralCatalog,
        categories: Sequence[str | int | bool],
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> list[LiteralDescriptor]:
        """Generate EQ literals for each category."""
        literals = []
        for category in categories:
            literal = catalog.category_eq(
                field_name,
                category,
                null_policy=null_policy,
            )
            literals.append(literal)
        return literals


@dataclass(frozen=True, slots=True)
class SensorDataTemplate(FeatureTemplate):
    """Template optimized for sensor data based on IoT literature patterns."""
    
    def __post_init__(self) -> None:
        if not self.parameters:
            object.__setattr__(
                self,
                "parameters",
                (
                    TemplateParameter(
                        name="sensor_ranges",
                        data_type=DataType.CONTINUOUS,
                        required=True,
                        description="Dict with 'min', 'max', 'normal_min', 'normal_max'",
                    ),
                    TemplateParameter(
                        name="threshold_percentiles",
                        data_type=DataType.DISCRETE,
                        required=False,
                        default=[10, 25, 50, 75, 90],
                        description="Percentile thresholds to generate",
                    ),
                    TemplateParameter(
                        name="temporal_features",
                        data_type=DataType.BINARY,
                        required=False,
                        default=False,
                        description="Whether to include temporal derivative features",
                    ),
                ),
            )
    
    def generate_literals(
        self,
        field_name: str,
        catalog: LiteralCatalog,
        sensor_ranges: Mapping[str, float],
        threshold_percentiles: Sequence[float] = (10, 25, 50, 75, 90),
        temporal_features: bool = False,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> list[LiteralDescriptor]:
        """Generate sensor-optimized literals."""
        literals = []
        
        # Generate threshold literals based on sensor ranges
        min_val = sensor_ranges.get("min", 0)
        max_val = sensor_ranges.get("max", 100)
        normal_min = sensor_ranges.get("normal_min", min_val)
        normal_max = sensor_ranges.get("normal_max", max_val)
        
        # Add percentile-based thresholds
        range_span = max_val - min_val
        for percentile in threshold_percentiles:
            threshold = min_val + (range_span * percentile / 100.0)
            literal = catalog.numeric_ge(field_name, threshold, null_policy=null_policy)
            literals.append(literal)
        
        # Add normal range indicator
        if normal_min != min_val or normal_max != max_val:
            literal = catalog.numeric_between(
                field_name,
                normal_min,
                normal_max,
                null_policy=null_policy,
            )
            literals.append(literal)
        
        return literals


@dataclass(frozen=True, slots=True)
class TextTokenTemplate(FeatureTemplate):
    """Template for text/token features."""
    
    def __post_init__(self) -> None:
        if not self.parameters:
            object.__setattr__(
                self,
                "parameters",
                (
                    TemplateParameter(
                        name="tokens",
                        data_type=DataType.TEXT,
                        required=True,
                        description="List of tokens to search for",
                    ),
                    TemplateParameter(
                        name="case_sensitive",
                        data_type=DataType.BINARY,
                        required=False,
                        default=False,
                        description="Whether token matching is case-sensitive",
                    ),
                ),
            )
    
    def generate_literals(
        self,
        field_name: str,
        catalog: LiteralCatalog,
        tokens: Sequence[str],
        case_sensitive: bool = False,
        null_policy: NullPolicy = NullPolicy.FALSE,
    ) -> list[LiteralDescriptor]:
        """Generate TOKEN_CONTAINS literals for each token."""
        literals = []
        for token in tokens:
            literal = catalog.token_contains(
                field_name,
                token,
                case_sensitive=case_sensitive,
                null_policy=null_policy,
            )
            literals.append(literal)
        return literals


class TemplateRegistry:
    """Registry for feature templates."""
    
    def __init__(self) -> None:
        self._templates: dict[str, FeatureTemplate] = {}
        self._register_builtins()
    
    def _register_builtins(self) -> None:
        """Register built-in templates."""
        builtins = [
            NumericThresholdTemplate(
                template_id="numeric_threshold_v1",
                name="Numeric Threshold",
                data_type=DataType.CONTINUOUS,
                description="Generate threshold-based literals for numeric fields",
            ),
            NumericRangeTemplate(
                template_id="numeric_range_v1",
                name="Numeric Range",
                data_type=DataType.CONTINUOUS,
                description="Generate range-based literals for numeric fields",
            ),
            CategoricalTemplate(
                template_id="categorical_v1",
                name="Categorical",
                data_type=DataType.CATEGORICAL,
                description="Generate equality literals for categorical fields",
            ),
            SensorDataTemplate(
                template_id="sensor_data_v1",
                name="Sensor Data",
                data_type=DataType.SENSOR,
                sensor_type=SensorType.CUSTOM,
                description="Optimized template for sensor data with percentile thresholds",
            ),
            TextTokenTemplate(
                template_id="text_token_v1",
                name="Text Token",
                data_type=DataType.TEXT,
                description="Generate token presence literals for text fields",
            ),
        ]
        for template in builtins:
            self.register(template)
    
    def register(self, template: FeatureTemplate) -> None:
        """Register a feature template."""
        if template.template_id in self._templates:
            raise ValueError(f"Template ID already registered: {template.template_id}")
        self._templates[template.template_id] = template
    
    def get(self, template_id: str) -> FeatureTemplate:
        """Get a template by ID."""
        if template_id not in self._templates:
            raise KeyError(f"Unknown template ID: {template_id}")
        return self._templates[template_id]
    
    def list_templates(self) -> tuple[str, ...]:
        """List all registered template IDs."""
        return tuple(self._templates.keys())
    
    def to_dict(self) -> dict[str, Any]:
        """Export registry as dictionary."""
        return {
            tid: template.to_dict()
            for tid, template in self._templates.items()
        }


@dataclass(frozen=True, slots=True)
class ClauseConfiguration:
    """Configuration and analysis metrics for a single clause."""
    clause_index: int
    included_literals: tuple[int, ...]
    excluded_literals: tuple[int, ...]
    polarity: int  # 1 for positive, -1 for negative
    activation_count: int = 0
    avg_activation_rate: float = 0.0
    contribution_score: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "clause_index": self.clause_index,
            "included_literals": list(self.included_literals),
            "excluded_literals": list(self.excluded_literals),
            "polarity": self.polarity,
            "activation_count": self.activation_count,
            "avg_activation_rate": self.avg_activation_rate,
            "contribution_score": self.contribution_score,
        }


@dataclass(frozen=True, slots=True)
class TAClauseConfiguration:
    """Complete TA-clause configuration export with analysis."""
    number_of_clauses: int
    number_of_features: int
    states_per_action: int
    specificity: float
    threshold: int
    clause_configs: tuple[ClauseConfiguration, ...]
    literal_descriptors: tuple[dict[str, Any], ...] = ()
    training_epochs: int = 0
    final_accuracy: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "number_of_clauses": self.number_of_clauses,
            "number_of_features": self.number_of_features,
            "states_per_action": self.states_per_action,
            "specificity": self.specificity,
            "threshold": self.threshold,
            "clause_configs": [config.to_dict() for config in self.clause_configs],
            "literal_descriptors": list(self.literal_descriptors),
            "training_epochs": self.training_epochs,
            "final_accuracy": self.final_accuracy,
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str | Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def analyze_clause_configuration(
    machine: Any,
    batch: Any,
    targets: Sequence[int | bool],
    clause_indices: Sequence[int] | None = None,
) -> TAClauseConfiguration:
    """Analyze a trained TM and export TA-clause configuration."""
    from .reference import ScalarBinaryTsetlinMachine
    
    if not isinstance(machine, ScalarBinaryTsetlinMachine):
        raise TypeError("Expected ScalarBinaryTsetlinMachine instance")
    
    if clause_indices is None:
        clause_indices = list(range(machine.number_of_clauses))
    
    # Analyze each clause
    clause_configs = []
    for clause_idx in clause_indices:
        included = []
        excluded = []
        
        for literal_idx in range(2 * machine.number_of_features):
            if machine.action_include(clause_idx, literal_idx):
                included.append(literal_idx)
            else:
                excluded.append(literal_idx)
        
        polarity = 1 if clause_idx % 2 == 0 else -1
        
        # Calculate activation statistics
        activation_count = 0
        for row_idx in range(batch.row_count):
            row_values = batch.row_values(row_idx)
            if machine.clause_output(clause_idx, row_values, prediction=False):
                activation_count += 1
        
        avg_activation_rate = activation_count / batch.row_count if batch.row_count > 0 else 0.0
        
        # Simple contribution score based on correct predictions
        correct_with_clause = 0
        correct_without_clause = 0
        
        for row_idx, target in enumerate(targets):
            row_values = batch.row_values(row_idx)
            pred_with = machine.predict_one(row_values)
            
            # Temporarily exclude clause
            original_states = machine._states[clause_idx][:]
            for lit_idx in range(2 * machine.number_of_features):
                machine._states[clause_idx][lit_idx] = machine.states_per_action
            
            pred_without = machine.predict_one(row_values)
            
            # Restore states
            machine._states[clause_idx] = original_states
            
            if pred_with == target:
                correct_with_clause += 1
            if pred_without == target:
                correct_without_clause += 1
        
        contribution_score = (correct_with_clause - correct_without_clause) / len(targets) if targets else 0.0
        
        config = ClauseConfiguration(
            clause_index=clause_idx,
            included_literals=tuple(included),
            excluded_literals=tuple(excluded),
            polarity=polarity,
            activation_count=activation_count,
            avg_activation_rate=avg_activation_rate,
            contribution_score=contribution_score,
        )
        clause_configs.append(config)
    
    # Get literal descriptors if available
    literal_descriptors = ()
    if hasattr(batch, "literal_ids") and hasattr(machine, "_literal_catalog"):
        literal_descriptors = tuple(
            {"literal_id": lid} for lid in batch.literal_ids
        )
    
    return TAClauseConfiguration(
        number_of_clauses=machine.number_of_clauses,
        number_of_features=machine.number_of_features,
        states_per_action=machine.states_per_action,
        specificity=machine.specificity,
        threshold=machine.threshold,
        clause_configs=tuple(clause_configs),
        literal_descriptors=literal_descriptors,
        metadata={
            "analysis_type": "ta_clause_configuration",
            "version": 1,
        },
    )


def create_feature_template_catalog(
    schema: FeatureSchema,
    template_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[LiteralCatalog, dict[str, list[LiteralDescriptor]]]:
    """Create a literal catalog using feature templates.
    
    Args:
        schema: The feature schema defining field types
        template_specs: Dict mapping field names to template specifications
                       Each spec should have 'template_id' and template parameters
    
    Returns:
        Tuple of (LiteralCatalog, dict mapping field names to generated literals)
    """
    registry = TemplateRegistry()
    catalog = LiteralCatalog(schema)
    generated_literals: dict[str, list[LiteralDescriptor]] = {}
    
    for field_name, spec in template_specs.items():
        template_id = spec.get("template_id")
        if not template_id:
            raise ValueError(f"No template_id specified for field {field_name}")
        
        template = registry.get(template_id)
        params = {k: v for k, v in spec.items() if k != "template_id"}
        
        literals = template.generate_literals(field_name, catalog, **params)
        generated_literals[field_name] = literals
    
    return catalog, generated_literals


def export_template_schema(registry: TemplateRegistry | None = None) -> dict[str, Any]:
    """Export the complete template schema for documentation/serialization."""
    if registry is None:
        registry = TemplateRegistry()
    
    return {
        "schema_version": 1,
        "templates": registry.to_dict(),
        "data_types": [dt.value for dt in DataType],
        "sensor_types": [st.value for st in SensorType],
    }


def load_template_from_dict(data: dict[str, Any]) -> FeatureTemplate:
    """Load a feature template from dictionary representation."""
    template_type = data.get("data_type", "continuous")
    
    if template_type == "continuous" or template_type == "sensor":
        if "sensor_ranges" in data.get("parameters", []):
            return SensorDataTemplate(
                template_id=data["template_id"],
                name=data["name"],
                data_type=DataType(data["data_type"]),
                description=data.get("description", ""),
            )
        else:
            return NumericThresholdTemplate(
                template_id=data["template_id"],
                name=data["name"],
                data_type=DataType(data["data_type"]),
                description=data.get("description", ""),
            )
    elif template_type == "categorical":
        return CategoricalTemplate(
            template_id=data["template_id"],
            name=data["name"],
            data_type=DataType(data["data_type"]),
            description=data.get("description", ""),
        )
    elif template_type == "text":
        return TextTokenTemplate(
            template_id=data["template_id"],
            name=data["name"],
            data_type=DataType(data["data_type"]),
            description=data.get("description", ""),
        )
    else:
        return NumericRangeTemplate(
            template_id=data["template_id"],
            name=data["name"],
            data_type=DataType(data["data_type"]),
            description=data.get("description", ""),
        )
