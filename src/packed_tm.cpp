#include "ptm/packed_tm.hpp"

#include "packed_tm_backend.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

namespace ptm {

namespace {

[[nodiscard]] std::size_t checked_product(std::size_t left,
                                          std::size_t right,
                                          const char* message) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::invalid_argument(message);
    }
    return left * right;
}

}  // namespace

PackedTMModel64::PackedTMModel64(
    std::size_t number_of_clauses,
    std::size_t number_of_features,
    std::uint16_t states_per_action,
    int threshold,
    std::span<const std::uint16_t> states)
    : number_of_clauses_(number_of_clauses),
      number_of_features_(number_of_features),
      literal_count_(checked_product(number_of_features, 2,
                                     "TM literal count overflows size_t")),
      states_per_action_(states_per_action),
      threshold_(threshold) {
    if (number_of_clauses_ == 0 || number_of_features_ == 0 ||
        states_per_action_ == 0 ||
        states_per_action_ > std::numeric_limits<std::uint16_t>::max() / 2U ||
        threshold_ <= 0 ||
        number_of_clauses_ >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("packed TM configuration is outside its bounds");
    }
    const auto automaton_count = checked_product(
        number_of_clauses_, literal_count_,
        "TM automaton count overflows size_t");
    if (states.size() != automaton_count) {
        throw std::invalid_argument("packed TM state array has the wrong shape");
    }
    const auto maximum_state =
        static_cast<unsigned>(states_per_action_) * 2U;
    state_bit_count_ = std::bit_width(maximum_state);
    state_word_count_ = (automaton_count + 63U) / 64U;
    literal_word_count_ = (literal_count_ + 63U) / 64U;
    state_planes_.assign(
        checked_product(state_bit_count_, state_word_count_,
                        "bit-sliced TM state size overflows size_t"),
        0);
    include_masks_.assign(
        checked_product(number_of_clauses_, literal_word_count_,
                        "TM action-mask size overflows size_t"),
        0);
    clause_literal_offsets_.reserve(number_of_clauses_ + 1U);
    clause_literal_offsets_.push_back(0);

    for (std::size_t automaton = 0; automaton < states.size(); ++automaton) {
        const auto value = static_cast<unsigned>(states[automaton]);
        if (value == 0 || value > maximum_state) {
            throw std::invalid_argument("TA state lies outside its two action regions");
        }
        const auto state_word = automaton / 64U;
        const auto state_bit = std::uint64_t{1} << (automaton % 64U);
        for (std::size_t bit = 0; bit < state_bit_count_; ++bit) {
            if ((value & (1U << bit)) != 0) {
                state_planes_[bit * state_word_count_ + state_word] |= state_bit;
            }
        }
        if (value > states_per_action_) {
            const auto clause = automaton / literal_count_;
            const auto literal = automaton % literal_count_;
            include_masks_[clause * literal_word_count_ + literal / 64U] |=
                std::uint64_t{1} << (literal % 64U);
        }
    }
    for (std::size_t clause = 0; clause < number_of_clauses_; ++clause) {
        for (std::size_t word = 0; word < literal_word_count_; ++word) {
            auto literals = include_masks_[clause * literal_word_count_ + word];
            while (literals != 0) {
                const auto offset = static_cast<std::size_t>(
                    std::countr_zero(literals));
                const auto literal = word * 64U + offset;
                if (literal < literal_count_) {
                    literal_features_.push_back(literal / 2U);
                    literal_negated_.push_back(
                        static_cast<std::uint8_t>(literal % 2U));
                }
                literals &= literals - 1U;
            }
        }
        clause_literal_offsets_.push_back(literal_features_.size());
    }
}

PackedTMModel64 PackedTMModel64::from_scalar(const ScalarBinaryTM& machine) {
    const auto snapshot = machine.snapshot();
    return PackedTMModel64(snapshot.number_of_clauses,
                           snapshot.number_of_features,
                           snapshot.states_per_action,
                           snapshot.threshold,
                           snapshot.states);
}

std::size_t PackedTMModel64::automaton_index(std::size_t clause,
                                             std::size_t literal) const {
    if (clause >= number_of_clauses_ || literal >= literal_count_) {
        throw std::out_of_range("packed TM automaton index");
    }
    return clause * literal_count_ + literal;
}

std::uint16_t PackedTMModel64::state(std::size_t clause,
                                     std::size_t literal) const {
    const auto automaton = automaton_index(clause, literal);
    const auto word = automaton / 64U;
    const auto mask = std::uint64_t{1} << (automaton % 64U);
    unsigned result = 0;
    for (std::size_t bit = 0; bit < state_bit_count_; ++bit) {
        if ((state_planes_[bit * state_word_count_ + word] & mask) != 0) {
            result |= 1U << bit;
        }
    }
    return static_cast<std::uint16_t>(result);
}

bool PackedTMModel64::action_include(std::size_t clause,
                                     std::size_t literal) const {
    static_cast<void>(automaton_index(clause, literal));
    return (include_masks_[clause * literal_word_count_ + literal / 64U] &
            (std::uint64_t{1} << (literal % 64U))) != 0;
}

std::uint64_t PackedTMModel64::state_plane_word(std::size_t bit,
                                                std::size_t word) const {
    if (bit >= state_bit_count_ || word >= state_word_count_) {
        throw std::out_of_range("packed TM state plane word");
    }
    return state_planes_[bit * state_word_count_ + word];
}

void PackedTMModel64::evaluate_into(
    std::span<const std::uint64_t> feature_words,
    std::uint64_t valid_example_mask,
    std::span<std::uint64_t> clause_outputs,
    std::span<std::uint64_t> feedback_clause_outputs,
    std::span<std::int32_t, packed_tm_batch_width> scores,
    std::uint64_t& prediction_mask,
    PackedTMBackend backend) const {
    if (feature_words.size() != number_of_features_) {
        throw std::invalid_argument("packed TM feature plane has the wrong width");
    }
    if (clause_outputs.size() < number_of_clauses_) {
        throw std::invalid_argument("packed TM clause output buffer is too small");
    }
    if (feedback_clause_outputs.size() < number_of_clauses_) {
        throw std::invalid_argument(
            "packed TM feedback output buffer is too small");
    }
    if (backend == PackedTMBackend::automatic) {
        backend = selected_backend();
    }
    if (!packed_tm_backend_available(backend)) {
        throw std::invalid_argument(
            std::string("packed TM backend is unavailable: ") +
            packed_tm_backend_name(backend));
    }

    const detail::PackedTMPlanView plan{
        clause_literal_offsets_, literal_features_, literal_negated_};
    switch (backend) {
        case PackedTMBackend::scalar:
            detail::evaluate_scalar_backend(
                plan, feature_words, valid_example_mask, clause_outputs,
                feedback_clause_outputs, scores);
            break;
#if defined(PTM_COMPILED_AVX2)
        case PackedTMBackend::avx2:
            detail::evaluate_avx2_backend(
                plan, feature_words, valid_example_mask, clause_outputs,
                feedback_clause_outputs, scores);
            break;
#endif
#if defined(PTM_COMPILED_AVX512)
        case PackedTMBackend::avx512:
            detail::evaluate_avx512_backend(
                plan, feature_words, valid_example_mask, clause_outputs,
                feedback_clause_outputs, scores);
            break;
#endif
        case PackedTMBackend::automatic:
        default:
            throw std::logic_error("packed TM backend dispatch is invalid");
    }

    prediction_mask = 0;
    for (std::size_t lane = 0; lane < packed_tm_batch_width; ++lane) {
        const auto bit = std::uint64_t{1} << lane;
        if ((valid_example_mask & bit) == 0) {
            scores[lane] = 0;
            continue;
        }
        scores[lane] = std::clamp(scores[lane], -threshold_, threshold_);
        if (scores[lane] > 0) {
            prediction_mask |= bit;
        }
    }
}

PackedTMResult64 PackedTMModel64::evaluate(
    std::span<const std::uint64_t> feature_words,
    std::uint64_t valid_example_mask,
    PackedTMBackend backend) const {
    PackedTMResult64 result{};
    result.valid_example_mask = valid_example_mask;
    result.clause_outputs.resize(number_of_clauses_);
    result.feedback_clause_outputs.resize(number_of_clauses_);
    evaluate_into(feature_words,
                  valid_example_mask,
                  result.clause_outputs,
                  result.feedback_clause_outputs,
                  result.scores,
                  result.prediction_mask,
                  backend);
    return result;
}

PackedTMBackend PackedTMModel64::selected_backend() const noexcept {
    const auto sparse_enough =
        literal_features_.size() <= number_of_clauses_;
    if (number_of_clauses_ >= 16U && number_of_clauses_ <= 128U &&
        sparse_enough &&
        packed_tm_backend_available(PackedTMBackend::avx512)) {
        return PackedTMBackend::avx512;
    }
    if (number_of_clauses_ >= 16U && number_of_clauses_ <= 128U &&
        sparse_enough &&
        packed_tm_backend_available(PackedTMBackend::avx2)) {
        return PackedTMBackend::avx2;
    }
    return PackedTMBackend::scalar;
}

}  // namespace ptm
