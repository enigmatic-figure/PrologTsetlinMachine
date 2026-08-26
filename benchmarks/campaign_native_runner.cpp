#include "ptm/packed_tm.hpp"
#include "ptm/scalar_tm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Dataset {
    std::size_t rows{};
    std::size_t features{};
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> labels;
};

struct PackedPage {
    std::vector<std::uint64_t> feature_words;
    std::uint64_t valid_mask{};
};

struct ScoreRun {
    std::filesystem::path input;
    std::filesystem::path output;
    Dataset dataset;
    std::vector<PackedPage> pages;
    std::vector<std::uint8_t> predictions;
    std::vector<int> vote_scores;
    std::vector<double> samples;
};

[[nodiscard]] std::size_t parse_positive(std::string_view text,
                                         std::string_view name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    std::size_t consumed = 0;
    const auto value = std::stoull(std::string(text), &consumed);
    if (consumed != text.size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::uint64_t parse_u64(std::string_view text,
                                      std::string_view name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be uint64");
    }
    std::size_t consumed = 0;
    const auto value = std::stoull(std::string(text), &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string(name) + " must be uint64");
    }
    return value;
}

[[nodiscard]] double parse_specificity(std::string_view text) {
    std::size_t consumed = 0;
    const auto value = std::stod(std::string(text), &consumed);
    if (consumed != text.size() || !std::isfinite(value) || value <= 1.0) {
        throw std::invalid_argument(
            "specificity must be finite and greater than one");
    }
    return value;
}

[[nodiscard]] Dataset load_dataset(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open dataset: " + path.string());
    }
    Dataset result;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            throw std::runtime_error("dataset contains an empty row");
        }
        std::vector<std::uint8_t> row;
        std::size_t offset = 0;
        while (offset < line.size()) {
            while (offset < line.size() && line[offset] == ' ') {
                ++offset;
            }
            if (offset == line.size()) {
                break;
            }
            if ((line[offset] != '0' && line[offset] != '1') ||
                (offset + 1 < line.size() && line[offset + 1] != ' ')) {
                throw std::runtime_error("dataset contains a non-binary field");
            }
            row.push_back(static_cast<std::uint8_t>(line[offset] - '0'));
            ++offset;
        }
        if (row.size() < 2) {
            throw std::runtime_error("dataset row is too short");
        }
        const auto width = row.size() - 1U;
        if (result.rows == 0) {
            result.features = width;
        } else if (width != result.features) {
            throw std::runtime_error("dataset rows have inconsistent widths");
        }
        result.values.insert(result.values.end(), row.begin(), row.end() - 1);
        result.labels.push_back(row.back());
        ++result.rows;
    }
    if (result.rows == 0) {
        throw std::runtime_error("dataset is empty");
    }
    return result;
}

[[nodiscard]] std::vector<PackedPage> pack_dataset(const Dataset& dataset) {
    const auto count = (dataset.rows + 63U) / 64U;
    std::vector<PackedPage> result;
    result.reserve(count);
    for (std::size_t page = 0; page < count; ++page) {
        const auto first = page * 64U;
        const auto lanes = std::min<std::size_t>(64U, dataset.rows - first);
        PackedPage packed;
        packed.feature_words.assign(dataset.features, 0);
        packed.valid_mask = lanes == 64U
                                ? ~std::uint64_t{0}
                                : (std::uint64_t{1} << lanes) - 1U;
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const auto row = first + lane;
            for (std::size_t feature = 0; feature < dataset.features; ++feature) {
                if (dataset.values[row * dataset.features + feature] != 0) {
                    packed.feature_words[feature] |= std::uint64_t{1} << lane;
                }
            }
        }
        result.push_back(std::move(packed));
    }
    return result;
}

[[nodiscard]] std::vector<std::uint8_t> evaluate_packed(
    const ptm::PackedTMModel64& model,
    const Dataset& dataset,
    const std::vector<PackedPage>& pages) {
    std::vector<std::uint8_t> result(dataset.rows, 0);
    std::size_t row = 0;
    for (const auto& page : pages) {
        const auto evaluated = model.evaluate(
            page.feature_words, page.valid_mask, ptm::PackedTMBackend::automatic);
        for (std::size_t lane = 0; lane < 64U && row < dataset.rows; ++lane, ++row) {
            result[row] = static_cast<std::uint8_t>(
                (evaluated.prediction_mask >> lane) & 1U);
        }
    }
    return result;
}

void verify_scalar(const ptm::ScalarBinaryTM& machine,
                   const Dataset& dataset,
                   const std::vector<std::uint8_t>& expected) {
    for (std::size_t row = 0; row < dataset.rows; ++row) {
        const auto features = std::span<const std::uint8_t>(
            dataset.values.data() + row * dataset.features, dataset.features);
        if (machine.predict(features) != expected[row]) {
            throw std::runtime_error("scalar and packed predictions differ");
        }
    }
}

void write_predictions(const std::filesystem::path& path,
                       const std::vector<std::uint8_t>& predictions) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open prediction output: " + path.string());
    }
    for (const auto prediction : predictions) {
        stream << static_cast<unsigned>(prediction) << '\n';
    }
    stream.flush();
    if (!stream) {
        throw std::runtime_error("could not write prediction output");
    }
}

void write_scores(const std::filesystem::path& path,
                  const std::vector<int>& scores) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open vote-score output: " + path.string());
    }
    for (const auto score : scores) {
        stream << score << '\n';
    }
    stream.flush();
    if (!stream) {
        throw std::runtime_error("could not write vote-score output");
    }
}

[[nodiscard]] double median(std::vector<std::size_t> values) {
    if (values.empty()) {
        throw std::invalid_argument("cannot compute the median of no values");
    }
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2U;
    if (values.size() % 2U != 0U) {
        return static_cast<double>(values[middle]);
    }
    return (static_cast<double>(values[middle - 1U]) +
            static_cast<double>(values[middle])) /
           2.0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 13) {
            throw std::invalid_argument(
                "usage: ptm_campaign_native_runner TRAIN CLAUSES STATES S T "
                "EPOCHS SEED REPEATS WARMUPS SCORE_COUNT INPUT OUTPUT "
                "[INPUT OUTPUT ...]");
        }
        const auto clauses = parse_positive(argv[2], "clauses");
        const auto raw_states = parse_positive(argv[3], "states per action");
        if (raw_states > std::numeric_limits<std::uint16_t>::max() / 2U) {
            throw std::invalid_argument("states per action exceeds native range");
        }
        const auto states = static_cast<std::uint16_t>(raw_states);
        const auto specificity = parse_specificity(argv[4]);
        const auto raw_threshold = parse_positive(argv[5], "threshold");
        if (raw_threshold >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("threshold exceeds native range");
        }
        const auto threshold = static_cast<int>(raw_threshold);
        const auto epochs = parse_positive(argv[6], "epochs");
        const auto seed = parse_u64(argv[7], "seed");
        const auto repeats = parse_positive(argv[8], "repeats");
        const auto warmups = parse_u64(argv[9], "warmups");
        const auto score_count = parse_positive(argv[10], "score count");
        if ((argc - 11) % 2 != 0 ||
            score_count != static_cast<std::size_t>((argc - 11) / 2)) {
            throw std::invalid_argument("score input/output arguments are incomplete");
        }

        const auto preprocessing_start = std::chrono::steady_clock::now();
        const auto train = load_dataset(argv[1]);
        std::vector<ScoreRun> scores;
        scores.reserve(score_count);
        for (std::size_t index = 0; index < score_count; ++index) {
            ScoreRun score;
            score.input = argv[11U + index * 2U];
            score.output = argv[12U + index * 2U];
            score.dataset = load_dataset(score.input);
            if (score.dataset.features != train.features) {
                throw std::runtime_error("training and scoring widths differ");
            }
            scores.push_back(std::move(score));
        }
        ptm::ScalarBinaryTM machine(
            clauses, train.features, states, specificity, threshold, seed);
        const auto initial_snapshot = machine.snapshot();
        auto preprocessing_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - preprocessing_start).count();

        const auto training_start = std::chrono::steady_clock::now();
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            for (std::size_t row = 0; row < train.rows; ++row) {
                machine.update(
                    std::span<const std::uint8_t>(
                        train.values.data() + row * train.features,
                        train.features),
                    train.labels[row]);
            }
        }
        const auto training_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - training_start).count();

        const auto materialization_start = std::chrono::steady_clock::now();
        const auto packed = ptm::PackedTMModel64::from_scalar(machine);
        for (auto& score : scores) {
            score.pages = pack_dataset(score.dataset);
        }
        preprocessing_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - materialization_start).count();

        for (auto& score : scores) {
            for (std::uint64_t warmup = 0; warmup < warmups; ++warmup) {
                static_cast<void>(
                    evaluate_packed(packed, score.dataset, score.pages));
            }
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                const auto started = std::chrono::steady_clock::now();
                auto current = evaluate_packed(packed, score.dataset, score.pages);
                score.samples.push_back(std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count());
                if (score.predictions.empty()) {
                    score.predictions = std::move(current);
                } else if (score.predictions != current) {
                    throw std::runtime_error(
                        "packed predictions changed without adaptation");
                }
            }
            verify_scalar(machine, score.dataset, score.predictions);
            write_predictions(score.output, score.predictions);
        }

        const auto diagnostic_start = std::chrono::steady_clock::now();
        const auto snapshot = machine.snapshot();
        std::size_t included = 0;
        std::size_t saturated_low = 0;
        std::size_t saturated_high = 0;
        std::size_t boundary_states = 0;
        std::size_t moved_states = 0;
        std::uint64_t absolute_state_movement = 0;
        std::size_t empty_clauses = 0;
        std::size_t contradictory_clauses = 0;
        std::size_t dead_clauses = 0;
        std::size_t low_support_clauses = 0;
        std::size_t singleton_support_clauses = 0;
        std::vector<std::size_t> included_per_clause;
        std::vector<std::size_t> support_per_clause;
        included_per_clause.reserve(clauses);
        support_per_clause.reserve(clauses);
        std::set<std::vector<std::uint64_t>> unsigned_behaviors;
        std::set<std::vector<std::uint64_t>> positive_behaviors;
        std::set<std::vector<std::uint64_t>> negative_behaviors;
        const auto behavior_words = (train.rows + 63U) / 64U;
        for (std::size_t clause = 0; clause < clauses; ++clause) {
            std::size_t clause_included = 0;
            bool contradictory = false;
            for (std::size_t feature = 0; feature < train.features; ++feature) {
                const auto positive = snapshot.states[
                    clause * train.features * 2U + feature * 2U] > states;
                const auto negative = snapshot.states[
                    clause * train.features * 2U + feature * 2U + 1U] > states;
                clause_included += positive + negative;
                contradictory = contradictory || (positive && negative);
            }
            for (std::size_t literal = 0; literal < train.features * 2U;
                 ++literal) {
                const auto index = clause * train.features * 2U + literal;
                const auto state = snapshot.states[
                    index];
                saturated_low += state == 1U;
                saturated_high += state == states * 2U;
                boundary_states += state == states || state == states + 1U;
                moved_states += state != initial_snapshot.states[index];
                absolute_state_movement += static_cast<std::uint64_t>(std::abs(
                    static_cast<int>(state) -
                    static_cast<int>(initial_snapshot.states[index])));
            }
            included += clause_included;
            empty_clauses += clause_included == 0;
            contradictory_clauses += contradictory;
            included_per_clause.push_back(clause_included);

            std::vector<std::uint64_t> behavior(behavior_words, 0);
            std::size_t support = 0;
            for (std::size_t row = 0; row < train.rows; ++row) {
                const auto features = std::span<const std::uint8_t>(
                    train.values.data() + row * train.features, train.features);
                if (machine.clause_output(clause, features, true)) {
                    behavior[row / 64U] |= std::uint64_t{1} << (row % 64U);
                    ++support;
                }
            }
            support_per_clause.push_back(support);
            dead_clauses += support == 0;
            singleton_support_clauses += support == 1;
            low_support_clauses +=
                static_cast<double>(support) / static_cast<double>(train.rows) <
                0.01;
            unsigned_behaviors.insert(behavior);
            (clause % 2U == 0U ? positive_behaviors : negative_behaviors)
                .insert(std::move(behavior));
        }
        std::size_t opposite_polarity_behaviors = 0;
        for (const auto& behavior : positive_behaviors) {
            opposite_polarity_behaviors += negative_behaviors.contains(behavior);
        }

        for (auto& score : scores) {
            score.vote_scores.reserve(score.dataset.rows);
            for (std::size_t row = 0; row < score.dataset.rows; ++row) {
                const auto features = std::span<const std::uint8_t>(
                    score.dataset.values.data() + row * score.dataset.features,
                    score.dataset.features);
                const auto value = machine.score(features);
                if (static_cast<std::uint8_t>(value > 0) !=
                    score.predictions[row]) {
                    throw std::runtime_error(
                        "vote score and retained prediction differ");
                }
                score.vote_scores.push_back(value);
            }
            auto score_path = score.output;
            score_path += ".scores";
            write_scores(score_path, score.vote_scores);
        }
        const auto diagnostic_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - diagnostic_start).count();

        std::cout << std::setprecision(17)
                  << "{\"schema\":\"ptm.native-campaign-runner.v2\","
                  << "\"preprocessing_materialization_s\":"
                  << preprocessing_seconds << ','
                  << "\"adaptive_training_s\":" << training_seconds << ','
                  << "\"diagnostic_collection_s\":" << diagnostic_seconds << ','
                  << "\"resident_inference_samples_s\":[";
        for (std::size_t index = 0; index < scores.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << '[';
            for (std::size_t sample = 0; sample < scores[index].samples.size();
                 ++sample) {
                if (sample != 0) {
                    std::cout << ',';
                }
                std::cout << scores[index].samples[sample];
            }
            std::cout << ']';
        }
        const auto state_count = snapshot.states.size();
        std::cout << "],\"backend\":\""
                  << ptm::packed_tm_backend_name(packed.selected_backend())
                  << "\",\"diagnostics\":{"
                  << "\"mean_included_literals_per_clause\":"
                  << static_cast<double>(included) / clauses << ','
                  << "\"median_included_literals_per_clause\":"
                  << median(included_per_clause) << ','
                  << "\"empty_clauses\":" << empty_clauses << ','
                  << "\"contradictory_clauses\":" << contradictory_clauses << ','
                  << "\"dead_clauses\":" << dead_clauses << ','
                  << "\"singleton_support_clauses\":"
                  << singleton_support_clauses << ','
                  << "\"low_support_clauses_below_1pct\":"
                  << low_support_clauses << ','
                  << "\"mean_clause_support_fraction\":"
                  << static_cast<double>(std::accumulate(
                         support_per_clause.begin(), support_per_clause.end(),
                         std::uint64_t{0})) /
                         static_cast<double>(clauses * train.rows)
                  << ','
                  << "\"median_clause_support_fraction\":"
                  << median(support_per_clause) / static_cast<double>(train.rows)
                  << ','
                  << "\"unique_clause_behaviors\":"
                  << unsigned_behaviors.size() << ','
                  << "\"unique_signed_clause_behaviors\":"
                  << positive_behaviors.size() + negative_behaviors.size() << ','
                  << "\"duplicate_clause_behaviors_within_polarity\":"
                  << clauses - positive_behaviors.size() - negative_behaviors.size()
                  << ','
                  << "\"opposite_polarity_shared_behaviors\":"
                  << opposite_polarity_behaviors << ','
                  << "\"ta_saturated_low_fraction\":"
                  << static_cast<double>(saturated_low) / state_count << ','
                  << "\"ta_saturated_high_fraction\":"
                  << static_cast<double>(saturated_high) / state_count << ','
                  << "\"ta_boundary_fraction\":"
                  << static_cast<double>(boundary_states) / state_count << ','
                  << "\"ta_moved_fraction\":"
                  << static_cast<double>(moved_states) / state_count << ','
                  << "\"mean_absolute_ta_state_movement\":"
                  << static_cast<double>(absolute_state_movement) / state_count
                  << ','
                  << "\"runtime_conformance_mismatches\":0}}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PTM native campaign runner failure: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
