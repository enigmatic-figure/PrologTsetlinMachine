#include "ptm/scalar_tm.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ptm {

ScalarBinaryTM::ScalarBinaryTM(std::size_t number_of_clauses,
                               std::size_t number_of_features,
                               std::uint16_t states_per_action,
                               double specificity,
                               int threshold,
                               std::uint64_t seed)
    : number_of_clauses_(number_of_clauses),
      number_of_features_(number_of_features),
      states_per_action_(states_per_action),
      specificity_(specificity),
      threshold_(threshold),
      states_(number_of_clauses * number_of_features * 2),
      rng_(seed) {
    if (number_of_clauses == 0 || number_of_features == 0) {
        throw std::invalid_argument("clause and feature counts must be positive");
    }
    if (states_per_action == 0 ||
        states_per_action > std::numeric_limits<std::uint16_t>::max() / 2) {
        throw std::invalid_argument("states_per_action is outside uint16 range");
    }
    if (specificity <= 1.0) {
        throw std::invalid_argument("specificity must be greater than one");
    }
    if (threshold <= 0) {
        throw std::invalid_argument("threshold must be positive");
    }
    for (auto& value : states_) {
        value = static_cast<std::uint16_t>(states_per_action_ + (rng_() & 1ULL));
    }
}

std::size_t ScalarBinaryTM::index(std::size_t clause,
                                  std::size_t literal) const {
    if (clause >= number_of_clauses_) {
        throw std::out_of_range("PTM clause index");
    }
    if (literal >= number_of_features_ * 2) {
        throw std::out_of_range("PTM literal index");
    }
    return clause * number_of_features_ * 2 + literal;
}

std::uint16_t ScalarBinaryTM::state(std::size_t clause,
                                    std::size_t literal) const {
    return states_[index(clause, literal)];
}

void ScalarBinaryTM::set_state(std::size_t clause,
                               std::size_t literal,
                               std::uint16_t value) {
    if (value == 0 || value > static_cast<unsigned>(states_per_action_) * 2U) {
        throw std::invalid_argument("TA state lies outside its two action regions");
    }
    states_[index(clause, literal)] = value;
}

bool ScalarBinaryTM::action_include(std::size_t clause,
                                    std::size_t literal) const {
    return state(clause, literal) > states_per_action_;
}

bool ScalarBinaryTM::literal_truth(std::span<const std::uint8_t> features,
                                   std::size_t literal) const {
    const bool feature = features[literal / 2] != 0;
    return literal % 2 == 0 ? feature : !feature;
}

bool ScalarBinaryTM::clause_output(
    std::size_t clause,
    std::span<const std::uint8_t> features,
    bool prediction) const {
    if (features.size() != number_of_features_) {
        throw std::invalid_argument("feature vector has the wrong width");
    }
    bool included = false;
    for (std::size_t literal = 0; literal < number_of_features_ * 2; ++literal) {
        if (action_include(clause, literal)) {
            included = true;
            if (!literal_truth(features, literal)) {
                return false;
            }
        }
    }
    return prediction ? included : true;
}

int ScalarBinaryTM::score(std::span<const std::uint8_t> features) const {
    int total = 0;
    for (std::size_t clause = 0; clause < number_of_clauses_; ++clause) {
        if (clause_output(clause, features, true)) {
            total += clause % 2 == 0 ? 1 : -1;
        }
    }
    return std::clamp(total, -threshold_, threshold_);
}

int ScalarBinaryTM::predict(std::span<const std::uint8_t> features) const {
    return score(features) > 0 ? 1 : 0;
}

void ScalarBinaryTM::increment(std::size_t clause, std::size_t literal) {
    auto& value = states_[index(clause, literal)];
    const auto maximum = static_cast<unsigned>(states_per_action_) * 2U;
    if (value < maximum) {
        ++value;
    }
}

void ScalarBinaryTM::decrement(std::size_t clause, std::size_t literal) {
    auto& value = states_[index(clause, literal)];
    if (value > 1) {
        --value;
    }
}

double ScalarBinaryTM::random_unit() {
    return std::generate_canonical<double, 53>(rng_);
}

void ScalarBinaryTM::update(std::span<const std::uint8_t> features, int target) {
    update(features, target, false);
}

void ScalarBinaryTM::update(std::span<const std::uint8_t> features,
                            int target,
                            bool boost_true_positive_feedback) {
    if (features.size() != number_of_features_) {
        throw std::invalid_argument("feature vector has the wrong width");
    }
    if (target != 0 && target != 1) {
        throw std::invalid_argument("binary target must be zero or one");
    }

    const int class_sum = score(features);
    const double reward_probability = boost_true_positive_feedback
                                          ? 1.0
                                          : (specificity_ - 1.0) / specificity_;
    const double penalty_probability = 1.0 / specificity_;

    for (std::size_t clause = 0; clause < number_of_clauses_; ++clause) {
        const int polarity = clause % 2 == 0 ? 1 : -1;
        const double feedback_probability =
            static_cast<double>(threshold_ +
                                (1 - 2 * target) * polarity * class_sum) /
            static_cast<double>(2 * threshold_);
        if (random_unit() > feedback_probability) {
            continue;
        }

        const bool output = clause_output(clause, features, false);
        const bool type_i = (target == 1 && polarity == 1) ||
                            (target == 0 && polarity == -1);
        if (type_i) {
            for (std::size_t literal = 0; literal < number_of_features_ * 2;
                 ++literal) {
                if (output && literal_truth(features, literal)) {
                    if (random_unit() <= reward_probability) {
                        increment(clause, literal);
                    }
                } else if (random_unit() <= penalty_probability) {
                    decrement(clause, literal);
                }
            }
        } else if (output) {
            for (std::size_t literal = 0; literal < number_of_features_ * 2;
                 ++literal) {
                if (!literal_truth(features, literal) &&
                    !action_include(clause, literal)) {
                    increment(clause, literal);
                }
            }
        }
    }
}

ScalarTMSnapshot ScalarBinaryTM::snapshot() const {
    return ScalarTMSnapshot{
        snapshot_schema_version,
        number_of_clauses_,
        number_of_features_,
        states_per_action_,
        specificity_,
        threshold_,
        states_,
        rng_,
    };
}

void ScalarBinaryTM::restore(const ScalarTMSnapshot& source) {
    if (source.schema_version != snapshot_schema_version ||
        source.number_of_clauses != number_of_clauses_ ||
        source.number_of_features != number_of_features_ ||
        source.states_per_action != states_per_action_ ||
        source.specificity != specificity_ || source.threshold != threshold_ ||
        source.states.size() != states_.size()) {
        throw std::invalid_argument(
            "snapshot configuration does not match this machine");
    }
    states_ = source.states;
    rng_ = source.rng;
}

}  // namespace ptm
