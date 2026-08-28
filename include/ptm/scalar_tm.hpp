#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace ptm {

inline constexpr std::uint32_t snapshot_schema_version = 1;

struct ScalarTMSnapshot {
    std::uint32_t schema_version{};
    std::size_t number_of_clauses{};
    std::size_t number_of_features{};
    std::uint16_t states_per_action{};
    double specificity{};
    int threshold{};
    std::vector<std::uint16_t> states{};
    std::mt19937_64 rng{};
};

class ScalarBinaryTM {
public:
    ScalarBinaryTM(std::size_t number_of_clauses,
                   std::size_t number_of_features,
                   std::uint16_t states_per_action = 100,
                   double specificity = 3.9,
                   int threshold = 15,
                   std::uint64_t seed = 1);

    [[nodiscard]] std::size_t number_of_clauses() const noexcept {
        return number_of_clauses_;
    }
    [[nodiscard]] std::size_t number_of_features() const noexcept {
        return number_of_features_;
    }
    [[nodiscard]] std::uint16_t states_per_action() const noexcept {
        return states_per_action_;
    }

    [[nodiscard]] std::uint16_t state(std::size_t clause,
                                      std::size_t literal) const;
    void set_state(std::size_t clause,
                   std::size_t literal,
                   std::uint16_t state);
    [[nodiscard]] bool action_include(std::size_t clause,
                                      std::size_t literal) const;

    [[nodiscard]] bool clause_output(std::size_t clause,
                                     std::span<const std::uint8_t> features,
                                     bool prediction = true) const;
    [[nodiscard]] int raw_vote(std::span<const std::uint8_t> features) const;
    [[nodiscard]] int score(std::span<const std::uint8_t> features) const;
    [[nodiscard]] int predict(std::span<const std::uint8_t> features) const;
    [[nodiscard]] double standard_feedback_probability(
        std::span<const std::uint8_t> features,
        int target) const;

    void update(std::span<const std::uint8_t> features, int target);
    void update(std::span<const std::uint8_t> features,
                int target,
                bool boost_true_positive_feedback);

    [[nodiscard]] ScalarTMSnapshot snapshot() const;
    void restore(const ScalarTMSnapshot& snapshot);

private:
    [[nodiscard]] std::size_t index(std::size_t clause,
                                    std::size_t literal) const;
    [[nodiscard]] bool literal_truth(std::span<const std::uint8_t> features,
                                     std::size_t literal) const;
    void increment(std::size_t clause, std::size_t literal);
    void decrement(std::size_t clause, std::size_t literal);
    [[nodiscard]] double random_unit();

    std::size_t number_of_clauses_{};
    std::size_t number_of_features_{};
    std::uint16_t states_per_action_{};
    double specificity_{};
    int threshold_{};
    std::vector<std::uint16_t> states_{};
    std::mt19937_64 rng_{};
};

}  // namespace ptm
