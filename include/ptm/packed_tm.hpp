#pragma once

#include "ptm/scalar_tm.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ptm {

class CudaPackedTMExecutor64;

inline constexpr std::size_t packed_tm_batch_width = 64;

enum class PackedTMBackend : std::uint8_t {
    automatic,
    scalar,
    avx2,
    avx512,
};

struct CpuCapabilities {
    bool x86{};
    bool os_xsave{};
    bool avx{};
    bool avx2{};
    bool avx512f{};
    bool compiled_avx2{};
    bool compiled_avx512{};
    std::string brand;
};

[[nodiscard]] const CpuCapabilities& cpu_capabilities() noexcept;
[[nodiscard]] const char* packed_tm_backend_name(PackedTMBackend backend) noexcept;
[[nodiscard]] bool packed_tm_backend_available(PackedTMBackend backend) noexcept;

struct PackedTMResult64 {
    std::uint64_t valid_example_mask{};
    std::uint64_t prediction_mask{};
    std::array<std::int32_t, packed_tm_batch_width> scores{};
    // One word per clause. Bit i is that clause's prediction output for lane i.
    std::vector<std::uint64_t> clause_outputs;
    // Feedback semantics differ only for empty clauses, which emit valid lanes.
    std::vector<std::uint64_t> feedback_clause_outputs;
};

// Immutable inference image of a binary scalar TM. Exact multi-state TA values
// are retained as bit planes across the clause-major automaton population;
// derived Include masks serve the packed clause evaluator.
class PackedTMModel64 {
public:
    PackedTMModel64(std::size_t number_of_clauses,
                    std::size_t number_of_features,
                    std::uint16_t states_per_action,
                    int threshold,
                    std::span<const std::uint16_t> states);

    [[nodiscard]] static PackedTMModel64 from_scalar(
        const ScalarBinaryTM& machine);

    [[nodiscard]] std::size_t number_of_clauses() const noexcept {
        return number_of_clauses_;
    }
    [[nodiscard]] std::size_t number_of_features() const noexcept {
        return number_of_features_;
    }
    [[nodiscard]] std::size_t literal_count() const noexcept {
        return literal_count_;
    }
    [[nodiscard]] std::uint16_t states_per_action() const noexcept {
        return states_per_action_;
    }
    [[nodiscard]] int threshold() const noexcept { return threshold_; }
    [[nodiscard]] std::size_t state_bit_count() const noexcept {
        return state_bit_count_;
    }
    [[nodiscard]] std::size_t state_word_count() const noexcept {
        return state_word_count_;
    }

    [[nodiscard]] std::uint16_t state(std::size_t clause,
                                      std::size_t literal) const;
    [[nodiscard]] bool action_include(std::size_t clause,
                                      std::size_t literal) const;
    [[nodiscard]] std::uint64_t state_plane_word(
        std::size_t bit,
        std::size_t word) const;

    // feature_words is feature-major: word f contains one truth bit for feature
    // f in each of 64 example lanes. Bits outside valid_example_mask are ignored.
    void evaluate_into(
        std::span<const std::uint64_t> feature_words,
        std::uint64_t valid_example_mask,
        std::span<std::uint64_t> clause_outputs,
        std::span<std::uint64_t> feedback_clause_outputs,
        std::span<std::int32_t, packed_tm_batch_width> scores,
        std::uint64_t& prediction_mask,
        PackedTMBackend backend = PackedTMBackend::automatic) const;

    [[nodiscard]] PackedTMResult64 evaluate(
        std::span<const std::uint64_t> feature_words,
        std::uint64_t valid_example_mask = ~std::uint64_t{0},
        PackedTMBackend backend = PackedTMBackend::automatic) const;

    [[nodiscard]] PackedTMBackend selected_backend() const noexcept;

private:
    friend class CudaPackedTMExecutor64;

    [[nodiscard]] std::size_t automaton_index(std::size_t clause,
                                              std::size_t literal) const;

    std::size_t number_of_clauses_{};
    std::size_t number_of_features_{};
    std::size_t literal_count_{};
    std::uint16_t states_per_action_{};
    int threshold_{};
    std::size_t state_bit_count_{};
    std::size_t state_word_count_{};
    std::size_t literal_word_count_{};
    std::vector<std::uint64_t> state_planes_{};
    std::vector<std::uint64_t> include_masks_{};
    std::vector<std::size_t> clause_literal_offsets_{};
    std::vector<std::size_t> literal_features_{};
    std::vector<std::uint8_t> literal_negated_{};
};

}  // namespace ptm
