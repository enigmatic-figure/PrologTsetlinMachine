#include "packed_tm_backend.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace ptm::detail {

void evaluate_avx512_backend(
    PackedTMPlanView plan,
    std::span<const std::uint64_t> feature_words,
    std::uint64_t valid_example_mask,
    std::span<std::uint64_t> clause_outputs,
    std::span<std::uint64_t> feedback_clause_outputs,
    std::span<std::int32_t, 64> scores) noexcept {
    std::fill(scores.begin(), scores.end(), 0);
    const auto all_ones = _mm512_set1_epi64(-1);
    const auto valid = _mm512_set1_epi64(
        static_cast<long long>(valid_example_mask));

    for (std::size_t base = 0; base < plan.clause_count(); base += 8U) {
        const auto lane_count = std::min<std::size_t>(8, plan.clause_count() - base);
        std::array<std::size_t, 8> first{};
        std::array<std::size_t, 8> count{};
        std::size_t maximum = 0;
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            first[lane] = plan.clause_literal_offsets[base + lane];
            count[lane] = plan.clause_literal_offsets[base + lane + 1U] -
                          first[lane];
            maximum = std::max(maximum, count[lane]);
        }
        auto output = valid;
        for (std::size_t step = 0; step < maximum; ++step) {
            alignas(64) std::array<std::int64_t, 8> indices{};
            alignas(64) std::array<std::uint64_t, 8> negated{};
            __mmask8 active = 0;
            for (std::size_t lane = 0; lane < lane_count; ++lane) {
                if (step < count[lane]) {
                    const auto literal = first[lane] + step;
                    indices[lane] = plan.literal_features[literal];
                    negated[lane] = plan.literal_negated[literal] != 0
                                        ? ~std::uint64_t{0}
                                        : 0;
                    active |= static_cast<__mmask8>(1U << lane);
                }
            }
            const auto index_vector = _mm512_load_si512(indices.data());
            auto truth = _mm512_i64gather_epi64(
                index_vector, feature_words.data(), 8);
            truth = _mm512_xor_si512(truth, _mm512_load_si512(negated.data()));
            truth = _mm512_mask_mov_epi64(all_ones, active, truth);
            output = _mm512_and_si512(output, truth);
        }
        alignas(64) std::array<std::uint64_t, 8> values{};
        _mm512_store_si512(values.data(), output);
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            const auto clause = base + lane;
            const auto value = values[lane] & valid_example_mask;
            feedback_clause_outputs[clause] = value;
            clause_outputs[clause] = count[lane] == 0 ? 0 : value;
        }
    }

    const auto one = _mm512_set1_epi32(1);
    for (std::size_t clause = 0; clause < plan.clause_count(); ++clause) {
        const auto word = clause_outputs[clause];
        if (word == 0) {
            continue;
        }
        const bool positive = (clause % 2U) == 0;
        for (std::size_t block = 0; block < 4; ++block) {
            const auto mask = static_cast<__mmask16>(word >> (block * 16U));
            auto current = _mm512_loadu_si512(scores.data() + block * 16U);
            current = positive
                          ? _mm512_mask_add_epi32(current, mask, current, one)
                          : _mm512_mask_sub_epi32(current, mask, current, one);
            _mm512_storeu_si512(scores.data() + block * 16U, current);
        }
    }
}

}  // namespace ptm::detail
