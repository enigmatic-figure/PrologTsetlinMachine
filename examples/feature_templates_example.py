"""Example demonstrating typed feature templates and TA-clause configuration exports."""

from pathlib import Path

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    ScalarBinaryTsetlinMachine,
    create_feature_template_catalog,
    analyze_clause_configuration,
    export_template_schema,
)


def main():
    # Define schema for sensor data
    schema = FeatureSchema.from_fields(
        temperature=FieldKind.NUMBER,
        humidity=FieldKind.NUMBER,
        pressure=FieldKind.NUMBER,
        motion=FieldKind.BOOLEAN,
        room=FieldKind.CATEGORY,
    )

    # Configure feature templates for each field
    template_specs = {
        "temperature": {
            "template_id": "sensor_data_v1",
            "sensor_ranges": {
                "min": 0,
                "max": 50,
                "normal_min": 18,
                "normal_max": 26,
            },
            "threshold_percentiles": [20, 40, 60, 80],
        },
        "humidity": {
            "template_id": "numeric_threshold_v1",
            "thresholds": [30, 50, 70],
        },
        "pressure": {
            "template_id": "numeric_range_v1",
            "ranges": [(980, 1000), (1000, 1020), (1020, 1040)],
        },
        "motion": {
            "template_id": "categorical_v1",
            "categories": [True, False],
        },
        "room": {
            "template_id": "categorical_v1",
            "categories": ["kitchen", "bedroom", "living_room", "bathroom"],
        },
    }

    # Create catalog using templates
    catalog, generated = create_feature_template_catalog(schema, template_specs)

    print("=" * 60)
    print("FEATURE TEMPLATE CATALOG")
    print("=" * 60)
    print("\nGenerated literals per field:")
    for field, lits in generated.items():
        print(f"  {field:12s}: {len(lits)} literals")
    print(f"\nTotal literals: {sum(len(l) for l in generated.values())}")

    # Generate sample IoT sensor data
    records = [
        {"temperature": 22.5, "humidity": 45, "pressure": 1013, "motion": True, "room": "living_room"},
        {"temperature": 35.0, "humidity": 80, "pressure": 995, "motion": False, "room": "kitchen"},
        {"temperature": 15.0, "humidity": 30, "pressure": 1025, "motion": True, "room": "bedroom"},
        {"temperature": 28.0, "humidity": 60, "pressure": 1010, "motion": False, "room": "bathroom"},
        {"temperature": 20.0, "humidity": 55, "pressure": 1015, "motion": True, "room": "kitchen"},
        {"temperature": 32.0, "humidity": 75, "pressure": 990, "motion": False, "room": "living_room"},
        {"temperature": 18.0, "humidity": 40, "pressure": 1020, "motion": True, "room": "bedroom"},
        {"temperature": 25.0, "humidity": 50, "pressure": 1008, "motion": False, "room": "bathroom"},
    ]

    # Target: comfort level (1=comfortable, 0=uncomfortable)
    targets = [1, 0, 1, 0, 1, 0, 1, 1]

    # Encode data
    batch = catalog.encode(records)
    print(f"\nEncoded dataset: {batch.ta.row_count} samples, {batch.ta.literal_count} features")

    # Train Tsetlin Machine
    print("\nTraining Tsetlin Machine...")
    machine = ScalarBinaryTsetlinMachine(
        number_of_clauses=20,
        number_of_features=batch.ta.literal_count,
        states_per_action=100,
        specificity=3.5,
        threshold=15,
        seed=42,
    )
    machine.fit_literal_batch(batch.ta, targets, epochs=100)

    # Evaluate
    rows = [batch.ta.row_values(i) for i in range(batch.ta.row_count)]
    predictions = machine.predict(rows)
    accuracy = sum(p == t for p, t in zip(predictions, targets)) / len(targets)

    print(f"\nModel Performance:")
    print(f"  Training Accuracy: {accuracy * 100:.1f}%")
    print(f"  Predictions:       {predictions}")
    print(f"  Targets:           {targets}")

    # Analyze clause configuration
    print("\n" + "=" * 60)
    print("TA-CLAUSE CONFIGURATION ANALYSIS")
    print("=" * 60)

    config = analyze_clause_configuration(machine, batch.ta, targets)

    print(f"\nModel Configuration:")
    print(f"  Clauses:           {config.number_of_clauses}")
    print(f"  Features:          {config.number_of_features}")
    print(f"  States/Action:     {config.states_per_action}")
    print(f"  Specificity:       {config.specificity}")
    print(f"  Threshold:         {config.threshold}")

    print(f"\nTop Contributing Clauses:")
    sorted_configs = sorted(config.clause_configs, key=lambda c: abs(c.contribution_score), reverse=True)
    for i, clause_cfg in enumerate(sorted_configs[:5]):
        polarity_str = "+" if clause_cfg.polarity > 0 else "-"
        print(f"  Clause {clause_cfg.clause_index:2d} ({polarity_str}): "
              f"contribution={clause_cfg.contribution_score:+.3f}, "
              f"activation_rate={clause_cfg.avg_activation_rate:.2f}, "
              f"included_literals={len(clause_cfg.included_literals)}")

    # Export configuration to JSON
    output_path = Path("out/feature-templates/clause-configuration.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(output_path)
    print(f"\nClause configuration exported to: {output_path}")

    # Export template schema
    template_schema = export_template_schema()
    print(f"\nAvailable feature templates: {len(template_schema['templates'])}")
    for tid, tdata in template_schema["templates"].items():
        print(f"  - {tid}: {tdata['name']} ({tdata['data_type']})")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
