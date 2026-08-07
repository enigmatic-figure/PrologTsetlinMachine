# Typed Logic AST and relational Class I interface

The Logic dataset front end parses each compact symbolic expression into a
typed, bounded abstract syntax tree. Parsing uses Python's expression parser for
precedence and chained-comparison behavior, then immediately converts through a
strict whitelist. The expression is never evaluated by Python and no calls,
attributes, constants, containers, or arbitrary names are accepted.

## Grammar and node types

The accepted surface tokens are variables `A` through `E`, parentheses, and:

| Symbolic token | Meaning | Typed AST node |
| --- | --- | --- |
| `-` | Boolean NOT | `not` |
| `&` | Boolean AND | `and` |
| `x` | Boolean OR | `or` |
| `!=` | Boolean inequality | `not_equal_chain` |
| `if` / `$` | true expression if condition else false expression | `conditional` |

Every node has a dense example-local ID, depth, optional variable name, parent,
children, and typed child roles. Roles are `operand`, `negated`,
`comparison_operand`, `condition`, `true_branch`, and `false_branch`.

`A != B != C` retains Python chained-comparison semantics:

```text
(A XOR B) AND (B XOR C)
```

It is not treated as left-associative three-input parity.

## Symbolic facts

The higher-resolution Class I view emits facts including:

```prolog
root(Node).
operator(Node, Kind).
depth(Node, Depth).
references(Node, Variable).
child(Parent, Role, Child).
bound_value(Variable, Value).
```

These facts retain the source tree even when the TA interface receives a flat
bounded encoding. They provide the direct input vocabulary for later Class III
Prolog search.

## Primitive logic lowering

Every AST also lowers into a structurally interned primitive Boolean graph with
`INPUT`, `NOT`, `AND`, `OR`, and `XOR`. The lowering eliminates conditionals:

```text
if C then T else F  ->  (C AND T) OR ((NOT C) AND F)
```

AND/OR nodes are flattened, commutative operands are ordered and deduplicated,
constants are folded, contradictions/tautologies are detected, duplicate XOR
operands cancel, and double negation is removed. This graph is the Python
front-end counterpart of the native canonical logic IR.

The paired dataset loader requires the typed AST evaluator and the independently
lowered primitive graph to agree with the supplied label. All 5,000 rows pass
both checks.

## Bounded AST-relational encoding

The first TA-facing structural catalog is fitted from training structure without
using labels and contains:

- cumulative counts for each AST node kind;
- node-kind-at-depth predicates;
- typed parent-role-child edges;
- typed two-hop paths;
- the five raw variable binding values.

Each feature has a stable SHA-256-derived 64-bit literal ID and a reversible
descriptor. On the fixed dataset split this produces 197 literals, 4,999 unique
vectors, no evaluation truncation, and a 100% optimistic collision ceiling.

With the deterministic scalar TM, the structural frontier is 63.8% evaluation
accuracy at 20 clauses and 20 epochs. This is a meaningful improvement over the
previous 60.4% best flat baseline, while the remaining gap shows that bounded
local paths are still not a recursive evaluator. Those unresolved trees are the
next natural input for bounded Class III rule search.
