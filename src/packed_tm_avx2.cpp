#include "packed_tm_backend.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace ptm::detail {

namespace {

struct alignas(32) VoteRow8 {
    std::array<std::int32_t, 8> lanes{};
};

[[nodiscard]] const std::array<VoteRow8, 256>& vote_table() noexcept {
    static const auto table = [] {
        std::array<VoteRow8, 256> result{};
        for (std::size_t value = 0; value < result.size(); ++value) {
            for (std::size_t lane = 0; lane < 8; ++lane) {
                result[value].lanes[lane] =
                    static_cast<std::int32_t>((value >> lane) & 1U);
            }
        }
        return result;
    }();
    return table;
}

}  // namespace

void evaluate_avx2_backend(
    PackedTMPlanView plan,
    std::span<const std::uint64_t> feature_words,
    std::uint64_t valid_example_mask,
    std::span<std::uint64_t> clause_outputs,
    std::span<std::uint64_t> feedback_clause_outputs,
    std::span<std::int32_t, 64> scores) noexcept {
    std::fill(scores.begin(), scores.end(), 0);
    const auto all_ones = _mm256_set1_epi64x(-1);
    const auto valid = _mm256_set1_epi64x(
        static_cast<long long>(valid_example_mask));

    for (std::size_t base = 0; base < plan.clause_count(); base += 4U) {
        const auto lane_count = std::min<std::size_t>(4, plan.clause_count() - base);
        std::array<std::size_t, 4> first{};
        std::array<std::size_t, 4> count{};
        std::size_t maximum = 0;
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            first[lane] = plan.clause_literal_offsets[base + lane];
            count[lane] = plan.clause_literal_offsets[base + lane + 1U] -
                          first[lane];
            maximum = std::max(maximum, count[lane]);
        }
        auto output = valid;
        for (std::size_t step = 0; step < maximum; ++step) {
            alignas(32) std::array<std::int64_t, 4> indices{};
            alignas(32) std::array<std::uint64_t, 4> negated{};
            alignas(32) std::array<std::uint64_t, 4> active{};
            for (std::size_t lane = 0; lane < lane_count; ++lane) {
                if (step < count[lane]) {
                    const auto literal = first[lane] + step;
                    indices[lane] = plan.literal_features[literal];
                    negated[lane] = plan.literal_negated[literal] != 0
                                        ? ~std::uint64_t{0}
                                        : 0;
                    active[lane] = ~std::uint64_t{0};
                }
            }
            const auto index_vector = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(indices.data()));
            auto truth = _mm256_i64gather_epi64(
                reinterpret_cast<const long long*>(feature_words.data()),
                index_vector, 8);
            truth = _mm256_xor_si256(
                truth, _mm256_load_si256(
                           reinterpret_cast<const __m256i*>(negated.data())));
            const auto active_vector = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(active.data()));
            truth = _mm256_or_si256(
                _mm256_and_si256(truth, active_vector),
                _mm256_andnot_si256(active_vector, all_ones));
            output = _mm256_and_si256(output, truth);
        }
        alignas(32) std::array<std::uint64_t, 4> values{};
        _mm256_store_si256(reinterpret_cast<__m256i*>(values.data()), output);
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            const auto clause = base + lane;
            const auto value = values[lane] & valid_example_mask;
            feedback_clause_outputs[clause] = value;
            clause_outputs[clause] = count[lane] == 0 ? 0 : value;
        }
    }

    const auto& table = vote_table();
    for (std::size_t clause = 0; clause < plan.clause_count(); ++clause) {
        const auto word = clause_outputs[clause];
        if (word == 0) {
            continue;
        }
        const bool positive = (clause % 2U) == 0;
        for (std::size_t block = 0; block < 8; ++block) {
            const auto byte = static_cast<std::uint8_t>(word >> (block * 8U));
            auto current = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(scores.data() + block * 8U));
            const auto votes = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(table[byte].lanes.data()));
            current = positive ? _mm256_add_epi32(current, votes)
                               : _mm256_sub_epi32(current, votes);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(scores.data() + block * 8U), current);
        }
    }
}

}  // namespace ptm::detail
