"""Class I raw records flowing into the scalar TM reference."""

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    LiteralCatalog,
    ScalarBinaryTsetlinMachine,
)


schema = FeatureSchema.from_fields(x0=FieldKind.BOOLEAN, x1=FieldKind.BOOLEAN)
catalog = LiteralCatalog(schema)
catalog.category_eq("x0", True)
catalog.category_eq("x1", True)

records = [
    {"x0": False, "x1": False},
    {"x0": False, "x1": True},
    {"x0": True, "x1": False},
    {"x0": True, "x1": True},
]
targets = [0, 1, 1, 0]
batch = catalog.encode(records, row_ids=["00", "01", "10", "11"])

machine = ScalarBinaryTsetlinMachine(
    number_of_clauses=20,
    number_of_features=batch.ta.literal_count,
    states_per_action=100,
    specificity=3.0,
    threshold=10,
    seed=7,
)
machine.fit_literal_batch(batch.ta, targets, epochs=150)

rows = [batch.ta.row_values(index) for index in range(batch.ta.row_count)]
predictions = machine.predict(rows)
print("literal IDs:", batch.ta.literal_ids)
print("packed words:", batch.ta.words)
print("predictions:", predictions)
print("targets:", targets)
