#include "ptm/logic_ir.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Dataset {
    std::size_t row_count{};
    std::size_t feature_count{};
    std::vector<std::uint8_t> features{};
    std::vector<std::uint8_t> labels{};
};

struct ModelDiagnostics {
    std::size_t state_count{};
    std::size_t moved_states{};
    std::uint64_t absolute_state_movement{};
    std::size_t included_actions{};
    std::size_t saturated_low{};
    std::size_t saturated_high{};
    std::size_t boundary_states{};
    std::size_t empty_clauses{};
    std::size_t contradictory_clauses{};
    std::size_t dead_clauses{};
    std::size_t always_firing_clauses{};
    std::size_t unique_clause_behaviors{};
    std::size_t train_ties{};
    std::size_t evaluation_ties{};
};

struct FailureDiagnostics {
    std::size_t total_errors{};
    std::size_t seen_evaluation_rows{};
    std::size_t unseen_evaluation_rows{};
    std::size_t conflicting_seen_rows{};
    std::size_t errors_on_seen_rows{};
    std::size_t errors_on_unseen_rows{};
    std::size_t train_signature_disagreements{};
    std::size_t model_errors_despite_matching_train_majority{};
};

Dataset load_binary_dataset(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open dataset: " + path.string());
    }

    Dataset result{};
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream row_stream(line);
        std::vector<std::uint8_t> row;
        int value = 0;
        while (row_stream >> value) {
            if (value != 0 && value != 1) {
                throw std::runtime_error("dataset is not binary: " +
                                         path.string());
            }
            row.push_back(static_cast<std::uint8_t>(value));
        }
        if (row.size() < 2) {
            throw std::runtime_error("dataset row is too short: " +
                                     path.string());
        }
        const auto feature_count = row.size() - 1;
        if (result.row_count == 0) {
            result.feature_count = feature_count;
        } else if (feature_count != result.feature_count) {
            throw std::runtime_error("dataset rows have inconsistent widths: " +
                                     path.string());
        }
        result.features.insert(
            result.features.end(), row.begin(), row.end() - 1);
        result.labels.push_back(row.back());
        ++result.row_count;
    }
    if (result.row_count == 0) {
        throw std::runtime_error("dataset is empty: " + path.string());
    }
    return result;
}

double accuracy(const std::vector<std::uint8_t>& predictions,
                const std::vector<std::uint8_t>& labels) {
    if (predictions.size() != labels.size() || labels.empty()) {
        throw std::invalid_argument("accuracy vectors have incompatible sizes");
    }
    std::size_t correct = 0;
    for (std::size_t index = 0; index < labels.size(); ++index) {
        correct += predictions[index] == labels[index];
    }
    return static_cast<double>(correct) / static_cast<double>(labels.size());
}

std::vector<std::uint8_t> source_predictions(
    const ptm::ScalarBinaryTM& machine,
    const Dataset& data) {
    std::vector<std::uint8_t> result(data.row_count, 0);
    for (std::size_t row = 0; row < data.row_count; ++row) {
        const auto inputs = std::span<const std::uint8_t>(
            data.features.data() + row * data.feature_count,
            data.feature_count);
        result[row] = static_cast<std::uint8_t>(machine.predict(inputs));
    }
    return result;
}

ModelDiagnostics diagnose_model(
    const ptm::ScalarBinaryTM& machine,
    const ptm::ScalarTMSnapshot& initial,
    const Dataset& train,
    const Dataset& evaluation) {
    const auto final = machine.snapshot();
    ModelDiagnostics result{};
    result.state_count = final.states.size();
    for (std::size_t index = 0; index < final.states.size(); ++index) {
        result.moved_states += final.states[index] != initial.states[index];
        result.absolute_state_movement += static_cast<std::uint64_t>(
            std::abs(static_cast<int>(final.states[index]) -
                     static_cast<int>(initial.states[index])));
        result.saturated_low += final.states[index] == 1;
        result.saturated_high +=
            final.states[index] == 2 * machine.states_per_action();
        result.boundary_states +=
            final.states[index] == machine.states_per_action() ||
            final.states[index] == machine.states_per_action() + 1;
    }

    std::set<std::vector<std::uint64_t>> behaviors;
    const auto behavior_words = (train.row_count + 63) / 64;
    for (std::size_t clause = 0; clause < machine.number_of_clauses(); ++clause) {
        std::size_t included = 0;
        bool contradictory = false;
        for (std::size_t feature = 0; feature < machine.number_of_features();
             ++feature) {
            const bool positive = machine.action_include(clause, feature * 2);
            const bool negative =
                machine.action_include(clause, feature * 2 + 1);
            included += positive + negative;
            contradictory = contradictory || (positive && negative);
        }
        result.included_actions += included;
        result.empty_clauses += included == 0;
        result.contradictory_clauses += contradictory;

        std::vector<std::uint64_t> behavior(behavior_words, 0);
        std::size_t firing = 0;
        for (std::size_t row = 0; row < train.row_count; ++row) {
            const auto features = std::span<const std::uint8_t>(
                train.features.data() + row * train.feature_count,
                train.feature_count);
            if (machine.clause_output(clause, features, true)) {
                behavior[row / 64] |= std::uint64_t{1} << (row % 64);
                ++firing;
            }
        }
        result.dead_clauses += firing == 0;
        result.always_firing_clauses += firing == train.row_count;
        behaviors.insert(std::move(behavior));
    }
    result.unique_clause_behaviors = behaviors.size();

    const auto count_ties = [&machine](const Dataset& data) {
        std::size_t ties = 0;
        for (std::size_t row = 0; row < data.row_count; ++row) {
            const auto features = std::span<const std::uint8_t>(
                data.features.data() + row * data.feature_count,
                data.feature_count);
            ties += machine.score(features) == 0;
        }
        return ties;
    };
    result.train_ties = count_ties(train);
    result.evaluation_ties = count_ties(evaluation);
    return result;
}

FailureDiagnostics diagnose_failures(
    const Dataset& train,
    const Dataset& evaluation,
    const std::vector<std::uint8_t>& predictions) {
    if (predictions.size() != evaluation.row_count) {
        throw std::invalid_argument("failure diagnostics prediction size differs");
    }
    std::map<std::vector<std::uint8_t>, std::array<std::size_t, 2>> signatures;
    for (std::size_t row = 0; row < train.row_count; ++row) {
        const auto begin = train.features.begin() +
                           static_cast<std::ptrdiff_t>(row * train.feature_count);
        const std::vector<std::uint8_t> signature(
            begin, begin + static_cast<std::ptrdiff_t>(train.feature_count));
        ++signatures[signature][train.labels[row]];
    }

    FailureDiagnostics result{};
    for (std::size_t row = 0; row < evaluation.row_count; ++row) {
        const auto begin =
            evaluation.features.begin() +
            static_cast<std::ptrdiff_t>(row * evaluation.feature_count);
        const std::vector<std::uint8_t> signature(
            begin,
            begin + static_cast<std::ptrdiff_t>(evaluation.feature_count));
        const auto found = signatures.find(signature);
        const bool error = predictions[row] != evaluation.labels[row];
        result.total_errors += error;
        if (found == signatures.end()) {
            ++result.unseen_evaluation_rows;
            result.errors_on_unseen_rows += error;
            continue;
        }
        ++result.seen_evaluation_rows;
        const auto& counts = found->second;
        result.conflicting_seen_rows += counts[0] != 0 && counts[1] != 0;
        result.errors_on_seen_rows += error;
        const auto train_majority = static_cast<std::uint8_t>(counts[1] > counts[0]);
        const bool signature_disagrees =
            train_majority != evaluation.labels[row];
        result.train_signature_disagreements += signature_disagrees;
        result.model_errors_despite_matching_train_majority +=
            error && !signature_disagrees;
    }
    return result;
}

std::uint64_t checksum_bytes(const std::vector<std::uint8_t>& values) {
    std::uint64_t result = 0;
    for (const auto value : values) {
        result = (result * 0x100000001b3ULL) ^ value;
    }
    return result;
}

std::uint64_t checksum_words(const std::vector<std::uint64_t>& values) {
    std::uint64_t result = 0;
    for (const auto value : values) {
        result = (result * 0x9e3779b97f4a7c15ULL) ^ value;
    }
    return result;
}

template <typename Function>
void benchmark(std::string_view name,
               std::size_t prediction_count,
               Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    const auto checksum = function();
    const auto finish = std::chrono::steady_clock::now();
    const auto seconds =
        std::chrono::duration<double>(finish - start).count();
    const auto rate = static_cast<double>(prediction_count) / seconds;
    std::cout << std::left << std::setw(29) << name << std::right
              << std::setw(14) << std::fixed << std::setprecision(0) << rate
              << " examples/s  checksum=" << checksum << '\n';
}

std::size_t parse_positive(const char* text, std::string_view name) {
    const auto value = std::stoull(text);
    if (value == 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto train_path = argc > 1
                                    ? std::filesystem::path(argv[1])
                                    : std::filesystem::path(
                                          "data/NoisyXOR/NoisyXORTrainingData.txt");
        const auto test_path = argc > 2
                                   ? std::filesystem::path(argv[2])
                                   : std::filesystem::path(
                                         "data/NoisyXOR/NoisyXORTestData.txt");
        const auto epochs = argc > 3 ? parse_positive(argv[3], "epochs") : 5;
        const auto repeats = argc > 4 ? parse_positive(argv[4], "repeats") : 25;

        const auto train = load_binary_dataset(train_path);
        const auto test = load_binary_dataset(test_path);
        if (train.feature_count != test.feature_count) {
            throw std::runtime_error("training and test feature counts differ");
        }

        const auto clause_count =
            argc > 5 ? parse_positive(argv[5], "clauses") : 100;
        ptm::ScalarBinaryTM machine(
            clause_count, train.feature_count, 100, 3.9, 15, 20260806);
        const auto initial = machine.snapshot();
        const auto training_start = std::chrono::steady_clock::now();
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            for (std::size_t row = 0; row < train.row_count; ++row) {
                machine.update(
                    std::span<const std::uint8_t>(
                        train.features.data() + row * train.feature_count,
                        train.feature_count),
                    train.labels[row]);
            }
        }
        const auto training_seconds = std::chrono::duration<double>(
                                          std::chrono::steady_clock::now() -
                                          training_start)
                                          .count();

        const auto train_predictions = source_predictions(machine, train);
        const auto source = source_predictions(machine, test);
        const auto diagnostics = diagnose_model(machine, initial, train, test);
        const auto failures = diagnose_failures(train, test, source);
        auto lowered = ptm::lower_scalar_tm(machine);
        const auto program =
            ptm::LogicProgram::compile(lowered.graph, lowered.root);
        const auto scalar = program.evaluate_scalar_rows(
            test.features, test.row_count, test.feature_count);
        const auto packed_input = ptm::PackedInputBatch::from_rows(
            test.features, test.row_count, test.feature_count);
        const auto packed = program.evaluate_packed_rows(packed_input);
        if (source != scalar || source != packed) {
            throw std::runtime_error(
                "compiled scalar/packed predictions differ from source TM");
        }

        const auto stats =
            lowered.graph.statistics(lowered.root, test.feature_count);
        const auto row_plan = ptm::LogicPlanner::choose(
            lowered.graph,
            lowered.root,
            ptm::LogicWorkload{test.row_count,
                               test.feature_count,
                               1,
                               ptm::LogicInputLayout::row_major_bytes});
        const auto packed_plan = ptm::LogicPlanner::choose(
            lowered.graph,
            lowered.root,
            ptm::LogicWorkload{test.row_count,
                               test.feature_count,
                               repeats,
                               ptm::LogicInputLayout::feature_major_packed});

        const auto train_positive =
            std::accumulate(train.labels.begin(), train.labels.end(), 0ULL);
        const auto test_positive =
            std::accumulate(test.labels.begin(), test.labels.end(), 0ULL);
        const auto majority_accuracy = [](std::size_t positives,
                                          std::size_t rows) {
            return static_cast<double>(std::max(positives, rows - positives)) /
                   static_cast<double>(rows);
        };
        std::cout << "PTM binary dataset compiler calibration\n"
                  << "train_file=" << train_path.string() << '\n'
                  << "evaluation_file=" << test_path.string() << '\n'
                  << "train=" << train.row_count << " test=" << test.row_count
                  << " features=" << test.feature_count
                  << " clauses=" << clause_count << '\n'
                  << "training_epochs=" << epochs << " training_seconds="
                  << std::fixed << std::setprecision(3) << training_seconds
                  << " train_accuracy=" << std::setprecision(4)
                  << accuracy(train_predictions, train.labels)
                  << " evaluation_accuracy=" << accuracy(source, test.labels)
                  << " majority_train="
                  << majority_accuracy(train_positive, train.row_count)
                  << " majority_evaluation="
                  << majority_accuracy(test_positive, test.row_count) << '\n'
                  << "state_moved_ratio="
                  << static_cast<double>(diagnostics.moved_states) /
                         diagnostics.state_count
                  << " mean_absolute_movement="
                  << static_cast<double>(diagnostics.absolute_state_movement) /
                         diagnostics.state_count
                  << " saturated_low_ratio="
                  << static_cast<double>(diagnostics.saturated_low) /
                         diagnostics.state_count
                  << " saturated_high_ratio="
                  << static_cast<double>(diagnostics.saturated_high) /
                         diagnostics.state_count
                  << " boundary_ratio="
                  << static_cast<double>(diagnostics.boundary_states) /
                         diagnostics.state_count
                  << '\n'
                  << "mean_included_per_clause="
                  << static_cast<double>(diagnostics.included_actions) /
                         clause_count
                  << " empty_clauses=" << diagnostics.empty_clauses
                  << " contradictory_clauses="
                  << diagnostics.contradictory_clauses
                  << " dead_clauses=" << diagnostics.dead_clauses
                  << " always_firing_clauses="
                  << diagnostics.always_firing_clauses
                  << " unique_clause_behaviors="
                  << diagnostics.unique_clause_behaviors << '\n'
                  << "train_tie_ratio="
                  << static_cast<double>(diagnostics.train_ties) /
                         train.row_count
                  << " evaluation_tie_ratio="
                  << static_cast<double>(diagnostics.evaluation_ties) /
                         test.row_count
                  << '\n'
                  << "evaluation_errors=" << failures.total_errors
                  << " seen_rows=" << failures.seen_evaluation_rows
                  << " unseen_rows=" << failures.unseen_evaluation_rows
                  << " conflicting_seen_rows="
                  << failures.conflicting_seen_rows
                  << " errors_seen=" << failures.errors_on_seen_rows
                  << " errors_unseen=" << failures.errors_on_unseen_rows
                  << " train_signature_disagreements="
                  << failures.train_signature_disagreements
                  << " model_errors_despite_matching_train_majority="
                  << failures.model_errors_despite_matching_train_majority
                  << '\n'
                  << "graph_nodes=" << stats.node_count
                  << " operators=" << stats.operator_count
                  << " edges=" << stats.edge_count << " depth=" << stats.depth
                  << " input_density=" << stats.input_density << '\n'
                  << "row_major_plan="
                  << ptm::logic_backend_name(row_plan.backend) << " ("
                  << row_plan.rationale << ")\n"
                  << "prepacked_plan="
                  << ptm::logic_backend_name(packed_plan.backend) << " ("
                  << packed_plan.rationale << ")\n\n";

        const auto prediction_count = test.row_count * repeats;
        benchmark("source scalar TM", prediction_count, [&] {
            std::uint64_t checksum = 0;
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                checksum ^= checksum_bytes(source_predictions(machine, test)) +
                            repeat;
            }
            return checksum;
        });
        benchmark("compiled scalar IR", prediction_count, [&] {
            std::uint64_t checksum = 0;
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                checksum ^=
                    checksum_bytes(program.evaluate_scalar_rows(
                        test.features, test.row_count, test.feature_count)) +
                    repeat;
            }
            return checksum;
        });
        benchmark("packed IR, transpose once", prediction_count, [&] {
            std::uint64_t checksum = 0;
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                checksum ^=
                    checksum_words(program.evaluate_packed_words(packed_input)) +
                    repeat;
            }
            return checksum;
        });
        benchmark("packed IR, incl. transpose", prediction_count, [&] {
            std::uint64_t checksum = 0;
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                const auto repacked = ptm::PackedInputBatch::from_rows(
                    test.features, test.row_count, test.feature_count);
                checksum ^=
                    checksum_words(program.evaluate_packed_words(repacked)) +
                    repeat;
            }
            return checksum;
        });
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PTM logic benchmark failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
