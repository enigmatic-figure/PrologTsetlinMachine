"""Budgeted persistence + multi-class / convolutional / regression / graph adapters demo (Milestone 4)."""

from prolog_tsetlin.representation import FeatureSchema, FieldKind
from prolog_tsetlin.budgeted_features import BudgetedFeatureStore
from prolog_tsetlin.adapters import MultiClassAdapter, PatchAdapter, RegressionAdapter
from prolog_tsetlin.graph.connectors import sequence_to_graph
from prolog_tsetlin.graph.graph_tm import GraphTsetlinMachine
from prolog_tsetlin.graph.types import GraphInput
import tempfile

# -- Budgeted store --
schema = FeatureSchema.from_fields(age=FieldKind.NUMBER, city=FieldKind.CATEGORY, text=FieldKind.TEXT)
store = BudgetedFeatureStore(schema, budget=3, policy="least_used")
# fill beyond budget; least_used will retire least-used literal
d1 = store.catalog.numeric_ge("age", 18)
store.record_use(d1.literal_id, 10)
d2 = store.catalog.numeric_ge("age", 65)
store.record_use(d2.literal_id, 1)
d3 = store.catalog.category_eq("city", "OSLO")
store.record_use(d3.literal_id, 5)
d4 = store.catalog.token_contains("text", "alert")
# d4 inserted with 0 use; least_used will retire d2 (use 1) not d4 protected this round, actually d2 is oldest among low-use
print(f"Budgeted size {store.size} literals {[d.parameter('threshold') if d.transform.value=='numeric_ge' else d.parameter('value') if d.transform.value=='category_eq' else d.parameter('token') for d in store.literals]}")
tmp = tempfile.mktemp(suffix=".json")
store.persist(tmp)
restored = BudgetedFeatureStore.load(tmp)
print(f"Persisted id {restored.to_dict()['store_id'][:16]} restored size {restored.size}")

# -- Multi-class (one-vs-rest) --
mc = MultiClassAdapter(source_field="label", classes=("cat","dog","fish"))
rec = {"label": "dog", "x": 1}
adapted = mc.adapt(rec)
print(f"Multi-class {adapted['label__mc_1']} votes -> inverse {mc.inverse({'label__mc_0':0,'label__mc_1':1,'label__mc_2':0})}")

# -- Patch / convolutional (sliding window) over ImageAdapter-like fields --
# simulate 2x3 image flattened via ImageAdapter naming pix_0000 ...
patch = PatchAdapter(field_prefix="pix", rows=2, cols=3, kernel_rows=1, kernel_cols=2)
base = {f"pix_{i:04d}": i for i in range(6)}
patches = list(patch.iter_patches(base))
print(f"Convolutional {len(patches)} patches, first patch {patches[0]['patch_0_0']},{patches[0]['patch_0_1']}")

# -- Regression thermometer --
reg = RegressionAdapter(source_field="y", thresholds=(0.0, 10.0, 20.0))
print(f"Regression bands for 15 -> {[reg.adapt({'y':15})[f'y__band_{i}'] for i in range(3)]} inverse {reg.inverse({'y__band_0':1,'y__band_1':1,'y__band_2':0})}")

# -- Graph adapter (sequence and grid reuse existing) --
seq_g = sequence_to_graph(["a","b","c"])
print(f"Graph seq nodes {seq_g.node_count} edges {len(seq_g.edges)}")
# tiny GraphTM on synthetic graphs
g1 = GraphInput.create(node_count=2, edges=[(0,1,0)], node_properties={0:["a"]})
g2 = GraphInput.create(node_count=2, edges=[], node_properties={0:["a"]})
gtm = GraphTsetlinMachine(depth=1, clauses=4, hv_dim=256, seed=1)
gtm.fit([g1,g2],[1,0], epochs=1)
print(f"Graph adapter GraphTM pred {gtm.predict(g1)} vs {gtm.predict(g2)}")

print("Milestone 4 demo done")
