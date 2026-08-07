#include "ptm/logic_ir.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

ptm::ScalarBinaryTM configured_xor() {
    ptm::ScalarBinaryTM machine(4, 2, 4, 3.0, 5, 9);
    for (std::size_t clause = 0; clause < 4; ++clause) {
        for (std::size_t literal = 0; literal < 4; ++literal) {
            machine.set_state(clause, literal, 4);
        }
    }
    machine.set_state(0, 0, 5);  // x AND NOT y, positive
    machine.set_state(0, 3, 5);
    machine.set_state(1, 0, 5);  // x AND y, negative
    machine.set_state(1, 2, 5);
    machine.set_state(2, 1, 5);  // NOT x AND y, positive
    machine.set_state(2, 2, 5);
    machine.set_state(3, 1, 5);  // NOT x AND NOT y, negative
    machine.set_state(3, 3, 5);
    return machine;
}

void test_canonical_simplification() {
    ptm::LogicGraph graph;
    const auto p = graph.input(0);
    const auto q = graph.input(1);
    const auto r = graph.input(2);
    const auto truth = graph.constant(true);
    const auto falsity = graph.constant(false);

    require(graph.negate(graph.negate(p)) == p,
            "double negation was not eliminated");
    require(graph.all({p, truth, p}) == p,
            "AND identity/idempotence was not eliminated");
    require(graph.any({p, falsity, p}) == p,
            "OR identity/idempotence was not eliminated");
    require(graph.node(graph.all({p, graph.negate(p)})).constant_value == false,
            "contradiction was not folded");
    require(graph.node(graph.any({p, graph.negate(p)})).constant_value,
            "tautology was not folded");
    require(graph.parity({p, p, q}) == q,
            "duplicate XOR operands were not cancelled");
    require(graph.parity({p, graph.negate(p)}) == truth,
            "XOR complement was not folded");

    const auto implication = graph.implies(p, q);
    const auto contrapositive =
        graph.implies(graph.negate(q), graph.negate(p));
    require(implication == contrapositive,
            "implication and contrapositive were not canonicalized");
    require(graph.all({implication, contrapositive}) == implication,
            "common implication was not deduplicated");

    const auto factored =
        graph.any({graph.all({p, q}), graph.all({p, r})});
    const auto expected = graph.all({p, graph.any({q, r})});
    require(factored == expected,
            "shared conjunction was not factored from disjunction");
    require(graph.any({p, graph.all({p, q})}) == p,
            "Boolean absorption was not applied");
}

void test_scalar_and_packed_equivalence() {
    ptm::LogicGraph graph;
    const auto p = graph.input(0);
    const auto q = graph.input(1);
    const auto r = graph.input(2);
    const std::array threshold_inputs{p, q, r};
    const auto majority = graph.threshold(threshold_inputs, 2);
    const auto implication = graph.implies(p, q);
    const std::array weighted_inputs{majority, implication, r};
    const std::array<std::int32_t, 3> weights{2, -1, 1};
    const auto root = graph.weighted_threshold(weighted_inputs, weights, 2);
    const auto program = ptm::LogicProgram::compile(graph, root);

    for (std::uint8_t encoded = 0; encoded < 8; ++encoded) {
        const std::array<std::uint8_t, 3> row{
            static_cast<std::uint8_t>((encoded >> 2U) & 1U),
            static_cast<std::uint8_t>((encoded >> 1U) & 1U),
            static_cast<std::uint8_t>(encoded & 1U),
        };
        const int majority_score = row[0] + row[1] + row[2];
        const bool expected =
            2 * (majority_score >= 2) - ((!row[0]) || row[1]) + row[2] >= 2;
        require(program.evaluate(row) == expected,
                "scalar compiled expression produced the wrong truth value");
    }

    constexpr std::size_t row_count = 137;
    constexpr std::size_t feature_count = 3;
    std::vector<std::uint8_t> rows(row_count * feature_count);
    std::mt19937 generator(1776);
    for (auto& value : rows) {
        value = static_cast<std::uint8_t>(generator() & 1U);
    }
    const auto scalar =
        program.evaluate_scalar_rows(rows, row_count, feature_count);
    const auto packed_inputs =
        ptm::PackedInputBatch::from_rows(rows, row_count, feature_count);
    const auto packed = program.evaluate_packed_rows(packed_inputs);
    require(scalar == packed,
            "64-example packed execution diverged from scalar execution");
    require(program.required_input_count() == feature_count,
            "compiled program input extent is incorrect");
}

void test_tm_lowering() {
    const auto machine = configured_xor();
    const auto lowered = ptm::lower_scalar_tm(machine);
    const auto program = ptm::LogicProgram::compile(lowered.graph, lowered.root);
    require(lowered.clause_roots.size() == machine.number_of_clauses(),
            "TM lowering lost a clause root");

    std::vector<std::uint8_t> rows;
    for (std::uint8_t x = 0; x < 2; ++x) {
        for (std::uint8_t y = 0; y < 2; ++y) {
            const std::array<std::uint8_t, 2> row{x, y};
            rows.insert(rows.end(), row.begin(), row.end());
            require(program.evaluate(row) == (machine.predict(row) != 0),
                    "compiled TM vote differs from the source TM");
        }
    }
    const auto packed = ptm::PackedInputBatch::from_rows(rows, 4, 2);
    const auto predictions = program.evaluate_packed_rows(packed);
    require(predictions == std::vector<std::uint8_t>({0, 1, 1, 0}),
            "compiled packed TM did not preserve XOR");
}

void test_planner() {
    ptm::LogicGraph graph;
    const auto p = graph.input(0);
    const auto q = graph.input(1);
    const auto root = graph.all({p, q});

    const auto small = ptm::LogicPlanner::choose(
        graph,
        root,
        ptm::LogicWorkload{32, 4096, 1,
                           ptm::LogicInputLayout::row_major_bytes});
    require(small.backend == ptm::LogicBackend::cpu_scalar,
            "planner should keep a tiny sparse batch scalar");

    const auto packed = ptm::LogicPlanner::choose(
        graph,
        root,
        ptm::LogicWorkload{64, 4096, 1,
                           ptm::LogicInputLayout::feature_major_packed});
    require(packed.backend == ptm::LogicBackend::cpu_packed_64,
            "planner ignored an already packed full word");

    const auto dense = ptm::LogicPlanner::choose(
        graph,
        root,
        ptm::LogicWorkload{512, 2, 1,
                           ptm::LogicInputLayout::row_major_bytes});
    require(dense.backend == ptm::LogicBackend::cpu_packed_64,
            "planner did not pack a dense large batch");
}

}  // namespace

int main() {
    try {
        test_canonical_simplification();
        test_scalar_and_packed_equivalence();
        test_tm_lowering();
        test_planner();
        std::cout << "PTM logic IR tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PTM logic IR test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
