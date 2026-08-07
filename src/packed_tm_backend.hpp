#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace ptm::detail {

struct PackedTMPlanView {
    std::span<const std::size_t> clause_literal_offsets;
    std::span<const std::size_t> literal_features;
    std::span<const std::uint8_t> literal_negated;

    [[nodiscard]] std::size_t clause_count() const noexcept {
        return clause_literal_offsets.size() - 1U;
    }
};

void evaluate_scalar_backend(
    PackedTMPlanView plan,
    std::span<const std::uint64_t> feature_words,
    std::uint64_t valid_example_mask,
    std::span<std::uint64_t> clause_outputs,
    std::span<std::uint64_t> feedback_clause_outputs,
    std::span<std::int32_t, 64> scores) noexcept;

#if defined(PTM_COMPILED_AVX2)
void evaluate_avx2_backend(
    PackedTMPlanView plan,
    std::span<const std::uint64_t> feature_words,
    std::uint64_t valid_example_mask,
    std::span<std::uint64_t> clause_outputs,
    std::span<std::uint64_t> feedback_clause_outputs,
    std::span<std::int32_t, 64> scores) noexcept;
#endif

#if defined(PTM_COMPILED_AVX512)
void evaluate_avx512_backend(
    PackedTMPlanView plan,
    std::span<const std::uint64_t> feature_words,
    std::uint64_t valid_example_mask,
    std::span<std::uint64_t> clause_outputs,
    std::span<std::uint64_t> feedback_clause_outputs,
    std::span<std::int32_t, 64> scores) noexcept;
#endif

}  // namespace ptm::detail
