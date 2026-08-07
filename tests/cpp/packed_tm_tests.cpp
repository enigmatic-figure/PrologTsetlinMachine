#include "ptm/packed_tm.hpp"
#include "ptm/scalar_tm.hpp"

#include <array>
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

ptm::ScalarBinaryTM xor_machine() {
    ptm::ScalarBinaryTM machine(4, 2, 3, 3.0, 8, 41);
    for (std::size_t clause = 0; clause < 4; ++clause) {
        for (std::size_t literal = 0; literal < 4; ++literal) {
            machine.set_state(clause, literal, 3);
        }
    }
    machine.set_state(0, 0, 4);
    machine.set_state(0, 3, 4);
    machine.set_state(1, 0, 4);
    machine.set_state(1, 2, 4);
    machine.set_state(2, 1, 4);
    machine.set_state(2, 2, 4);
    machine.set_state(3, 1, 4);
    machine.set_state(3, 3, 4);
    return machine;
}

std::vector<std::uint64_t> pack_rows(
    const std::vector<std::uint8_t>& rows,
    std::size_t row_count,
    std::size_t feature_count) {
    std::vector<std::uint64_t> words(feature_count, 0);
    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t feature = 0; feature < feature_count; ++feature) {
            if (rows[row * feature_count + feature] != 0) {
                words[feature] |= std::uint64_t{1} << row;
            }
        }
    }
    return words;
}

void compare_with_scalar(const ptm::ScalarBinaryTM& scalar,
                         const ptm::PackedTMModel64& packed,
                         const std::vector<std::uint8_t>& rows,
                         std::size_t row_count) {
    const auto words = pack_rows(rows, row_count, scalar.number_of_features());
    const auto valid = row_count == 64
                           ? ~std::uint64_t{0}
                           : (std::uint64_t{1} << row_count) - 1U;
    const auto result = packed.evaluate(words, valid);
    require(result.valid_example_mask == valid,
            "packed TM returned the wrong valid-lane mask");
    for (std::size_t row = 0; row < row_count; ++row) {
        const auto features = std::span<const std::uint8_t>(
            rows.data() + row * scalar.number_of_features(),
            scalar.number_of_features());
        require(result.scores[row] == scalar.score(features),
                "packed signed vote differs from scalar score");
        require(((result.prediction_mask >> row) & 1U) ==
                    static_cast<std::uint64_t>(scalar.predict(features)),
                "packed prediction differs from scalar prediction");
        for (std::size_t clause = 0; clause < scalar.number_of_clauses();
             ++clause) {
            require(((result.clause_outputs[clause] >> row) & 1U) ==
                        static_cast<std::uint64_t>(
                            scalar.clause_output(clause, features, true)),
                    "packed clause output differs from scalar clause");
            require(((result.feedback_clause_outputs[clause] >> row) & 1U) ==
                        static_cast<std::uint64_t>(
                            scalar.clause_output(clause, features, false)),
                    "packed feedback clause differs from scalar clause");
        }
    }
    for (std::size_t row = row_count; row < 64; ++row) {
        require(result.scores[row] == 0 &&
                    ((result.prediction_mask >> row) & 1U) == 0,
                "invalid packed lane was not suppressed");
    }
}

void test_xor_full_and_partial_batches() {
    const auto scalar = xor_machine();
    const auto packed = ptm::PackedTMModel64::from_scalar(scalar);
    std::vector<std::uint8_t> rows(64 * 2);
    for (std::size_t row = 0; row < 64; ++row) {
        rows[row * 2] = static_cast<std::uint8_t>((row >> 1U) & 1U);
        rows[row * 2 + 1U] = static_cast<std::uint8_t>(row & 1U);
    }
    compare_with_scalar(scalar, packed, rows, 64);
    rows.resize(37 * 2);
    compare_with_scalar(scalar, packed, rows, 37);
}

void test_state_planes_round_trip_exact_values() {
    constexpr std::size_t clauses = 7;
    constexpr std::size_t features = 65;
    constexpr std::uint16_t states_per_action = 100;
    std::vector<std::uint16_t> states(clauses * features * 2);
    for (std::size_t index = 0; index < states.size(); ++index) {
        states[index] = static_cast<std::uint16_t>(
            1U + (index * 37U) % (states_per_action * 2U));
    }
    const ptm::PackedTMModel64 packed(
        clauses, features, states_per_action, 15, states);
    require(packed.state_bit_count() == 8,
            "200-state automata require eight state planes");
    require(packed.state_word_count() == (states.size() + 63U) / 64U,
            "bit-sliced state word count is wrong");
    for (std::size_t clause = 0; clause < clauses; ++clause) {
        for (std::size_t literal = 0; literal < features * 2U; ++literal) {
            const auto expected = states[clause * features * 2U + literal];
            require(packed.state(clause, literal) == expected,
                    "bit-sliced TA state did not round-trip");
            require(packed.action_include(clause, literal) ==
                        (expected > states_per_action),
                    "derived Include action differs from exact TA state");
        }
    }
}

void test_empty_and_contradictory_clauses() {
    ptm::ScalarBinaryTM scalar(2, 1, 3, 3.0, 4, 7);
    for (std::size_t clause = 0; clause < 2; ++clause) {
        scalar.set_state(clause, 0, 3);
        scalar.set_state(clause, 1, 3);
    }
    scalar.set_state(1, 0, 4);
    scalar.set_state(1, 1, 4);
    const auto packed = ptm::PackedTMModel64::from_scalar(scalar);
    const std::array<std::uint64_t, 1> features{0xaaaaaaaaaaaaaaaaULL};
    const auto result = packed.evaluate(features);
    require(result.clause_outputs[0] == 0,
            "empty clause fired during packed prediction");
    require(result.feedback_clause_outputs[0] ==
                ~std::uint64_t{0},
            "empty clause did not fire under feedback semantics");
    require(result.clause_outputs[1] == 0,
            "contradictory clause fired during packed prediction");
    require(result.prediction_mask == 0,
            "empty/contradictory machine predicted positive");
}

void test_randomized_scalar_equivalence() {
    std::mt19937_64 random(20260806);
    const std::array<std::size_t, 3> clause_counts{1, 8, 21};
    const std::array<std::size_t, 3> feature_counts{1, 7, 65};
    const std::array<std::size_t, 4> row_counts{1, 17, 63, 64};
    for (const auto clauses : clause_counts) {
        for (const auto features : feature_counts) {
            ptm::ScalarBinaryTM scalar(
                clauses, features, 31, 3.9, 11, random());
            for (std::size_t clause = 0; clause < clauses; ++clause) {
                for (std::size_t literal = 0; literal < features * 2U;
                     ++literal) {
                    scalar.set_state(
                        clause,
                        literal,
                        static_cast<std::uint16_t>(1U + random() % 62U));
                }
            }
            const auto packed = ptm::PackedTMModel64::from_scalar(scalar);
            for (const auto rows_in_batch : row_counts) {
                std::vector<std::uint8_t> rows(rows_in_batch * features);
                for (auto& value : rows) {
                    value = static_cast<std::uint8_t>(random() & 1U);
                }
                compare_with_scalar(scalar, packed, rows, rows_in_batch);
            }
        }
    }
}

void require_same_result(const ptm::PackedTMResult64& expected,
                         const ptm::PackedTMResult64& actual,
                         std::string_view backend) {
    require(actual.valid_example_mask == expected.valid_example_mask,
            std::string(backend) + " changed the valid-example mask");
    require(actual.prediction_mask == expected.prediction_mask,
            std::string(backend) + " changed predictions");
    require(actual.scores == expected.scores,
            std::string(backend) + " changed signed votes");
    require(actual.clause_outputs == expected.clause_outputs,
            std::string(backend) + " changed clause outputs");
    require(actual.feedback_clause_outputs ==
                expected.feedback_clause_outputs,
            std::string(backend) + " changed feedback clause outputs");
}

void test_runtime_dispatch_equivalence() {
    constexpr std::size_t clauses = 21;
    constexpr std::size_t features = 67;
    constexpr std::uint16_t states_per_action = 31;
    std::mt19937_64 random(0x51a7d15aU);
    std::vector<std::uint16_t> states(clauses * features * 2U);
    for (auto& state : states) {
        state = static_cast<std::uint16_t>(1U + random() % 62U);
    }
    const ptm::PackedTMModel64 packed(
        clauses, features, states_per_action, 11, states);
    std::vector<std::uint64_t> feature_words(features);
    for (auto& word : feature_words) {
        word = random();
    }
    constexpr std::uint64_t valid = 0x7fff'ffff'ffff'13a5ULL;
    const auto scalar = packed.evaluate(
        feature_words, valid, ptm::PackedTMBackend::scalar);
    for (const auto backend : {ptm::PackedTMBackend::avx2,
                               ptm::PackedTMBackend::avx512}) {
        if (ptm::packed_tm_backend_available(backend)) {
            require_same_result(
                scalar, packed.evaluate(feature_words, valid, backend),
                ptm::packed_tm_backend_name(backend));
        }
    }
    require_same_result(
        scalar,
        packed.evaluate(feature_words, valid, ptm::PackedTMBackend::automatic),
        "automatic");

    const auto selected = packed.selected_backend();
    require(selected != ptm::PackedTMBackend::automatic &&
                ptm::packed_tm_backend_available(selected),
            "automatic dispatch selected an unavailable backend");
    const auto& capabilities = ptm::cpu_capabilities();
    require(!capabilities.brand.empty(), "CPU capability brand is empty");
    require(ptm::packed_tm_backend_available(ptm::PackedTMBackend::scalar),
            "portable scalar backend is unavailable");
}

void test_forced_unavailable_backend_is_rejected() {
    const auto packed = ptm::PackedTMModel64::from_scalar(xor_machine());
    const std::array<std::uint64_t, 2> features{};
    for (const auto backend : {ptm::PackedTMBackend::avx2,
                               ptm::PackedTMBackend::avx512}) {
        if (!ptm::packed_tm_backend_available(backend)) {
            try {
                static_cast<void>(packed.evaluate(
                    features, ~std::uint64_t{0}, backend));
            } catch (const std::invalid_argument&) {
                return;
            }
            throw std::runtime_error(
                "packed TM accepted an unavailable forced backend");
        }
    }
}

void test_density_aware_backend_selection() {
    constexpr std::size_t clauses = 20;
    constexpr std::size_t features = 64;
    constexpr std::uint16_t states_per_action = 100;
    std::vector<std::uint16_t> states(
        clauses * features * 2U, states_per_action);
    for (std::size_t clause = 0; clause < clauses; ++clause) {
        states[clause * features * 2U] = states_per_action + 1U;
    }
    const ptm::PackedTMModel64 sparse(
        clauses, features, states_per_action, 15, states);
    const auto expected =
        ptm::packed_tm_backend_available(ptm::PackedTMBackend::avx512)
            ? ptm::PackedTMBackend::avx512
            : ptm::packed_tm_backend_available(ptm::PackedTMBackend::avx2)
                  ? ptm::PackedTMBackend::avx2
                  : ptm::PackedTMBackend::scalar;
    require(sparse.selected_backend() == expected,
            "sparse plan selected the wrong available backend");

    for (std::size_t clause = 0; clause < clauses; ++clause) {
        states[clause * features * 2U + 1U] = states_per_action + 1U;
    }
    const ptm::PackedTMModel64 denser(
        clauses, features, states_per_action, 15, states);
    require(denser.selected_backend() == ptm::PackedTMBackend::scalar,
            "dense plan did not stay on the scalar backend");
}

void test_shape_validation() {
    const auto packed = ptm::PackedTMModel64::from_scalar(xor_machine());
    const std::array<std::uint64_t, 1> too_short{};
    try {
        static_cast<void>(packed.evaluate(too_short));
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("packed TM accepted the wrong feature width");
}

}  // namespace

int main() {
    try {
        test_xor_full_and_partial_batches();
        test_state_planes_round_trip_exact_values();
        test_empty_and_contradictory_clauses();
        test_randomized_scalar_equivalence();
        test_runtime_dispatch_equivalence();
        test_forced_unavailable_backend_is_rejected();
        test_density_aware_backend_selection();
        test_shape_validation();
        std::cout << "PTM packed TM tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PTM packed TM test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
