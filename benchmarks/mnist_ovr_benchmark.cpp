#include "ptm/scalar_tm.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Dataset {
    std::uint32_t rows{};
    std::uint32_t features{};
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> labels;
};

std::uint32_t read_u32(std::ifstream& stream) {
    std::array<unsigned char, 4> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) throw std::runtime_error("truncated MNIST header");
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

Dataset load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open MNIST material");
    std::array<char, 8> magic{};
    stream.read(magic.data(), magic.size());
    if (std::string(magic.data(), magic.size()) != "PTMMNIST")
        throw std::runtime_error("MNIST material magic is invalid");
    if (read_u32(stream) != 1U) throw std::runtime_error("MNIST material version is unsupported");
    Dataset result;
    result.rows = read_u32(stream);
    result.features = read_u32(stream);
    if (result.rows == 0U || result.features != 784U)
        throw std::runtime_error("MNIST material shape is invalid");
    const auto cells = static_cast<std::size_t>(result.rows) * result.features;
    result.values.resize(cells);
    result.labels.resize(result.rows);
    stream.read(reinterpret_cast<char*>(result.values.data()), static_cast<std::streamsize>(cells));
    stream.read(reinterpret_cast<char*>(result.labels.data()), static_cast<std::streamsize>(result.labels.size()));
    if (!stream || stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("MNIST material payload is malformed");
    for (const auto value : result.values) if (value > 1U) throw std::runtime_error("MNIST feature is not binary");
    for (const auto label : result.labels) if (label > 9U) throw std::runtime_error("MNIST label is invalid");
    return result;
}

std::size_t positive(std::string_view text, std::string_view label) {
    std::size_t used = 0;
    const auto value = std::stoull(std::string(text), &used);
    if (used != text.size() || value == 0U) throw std::invalid_argument(std::string(label) + " must be positive");
    return static_cast<std::size_t>(value);
}

double finite_double(std::string_view text, std::string_view label) {
    std::size_t used = 0;
    const auto value = std::stod(std::string(text), &used);
    if (used != text.size() || !std::isfinite(value)) throw std::invalid_argument(std::string(label) + " is invalid");
    return value;
}

double accuracy(const std::array<ptm::ScalarBinaryTM, 10>& models, const Dataset& data) {
    std::array<std::size_t, 10> correct{};
    std::array<std::thread, 10> workers;
    for (std::size_t worker = 0; worker < workers.size(); ++worker) {
        workers[worker] = std::thread([&, worker] {
            const auto begin = data.rows * worker / workers.size();
            const auto end = data.rows * (worker + 1U) / workers.size();
            for (std::size_t row = begin; row < end; ++row) {
                const auto features = std::span<const std::uint8_t>(
                    data.values.data() + row * data.features, data.features);
                int best_score = std::numeric_limits<int>::min();
                std::uint8_t best_class = 0;
                for (std::uint8_t cls = 0; cls < 10U; ++cls) {
                    const auto score = models[cls].score(features);
                    if (score > best_score) {
                        best_score = score;
                        best_class = cls;
                    }
                }
                correct[worker] += best_class == data.labels[row];
            }
        });
    }
    for (auto& worker : workers) worker.join();
    return static_cast<double>(
               std::accumulate(correct.begin(), correct.end(), std::size_t{0})) /
           data.rows;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 9 || argc > 11) throw std::invalid_argument(
            "usage: ptm_mnist_ovr_benchmark TRAIN VALIDATION CLAUSES STATES S T EPOCHS SEED [paired|all] [standard|boost]");
        const auto train = load(argv[1]);
        const auto validation = load(argv[2]);
        if (train.features != validation.features) throw std::runtime_error("MNIST split widths differ");
        const auto clauses = positive(argv[3], "clauses");
        const auto states_raw = positive(argv[4], "states");
        const auto specificity = finite_double(argv[5], "specificity");
        const auto threshold_raw = positive(argv[6], "threshold");
        const auto epochs = positive(argv[7], "epochs");
        const auto seed = positive(argv[8], "seed");
        const std::string policy = argc >= 10 ? argv[9] : "paired";
        const std::string feedback = argc == 11 ? argv[10] : "standard";
        if (policy != "paired" && policy != "all")
            throw std::invalid_argument("training policy must be paired or all");
        if (feedback != "standard" && feedback != "boost")
            throw std::invalid_argument("feedback mode must be standard or boost");
        const bool boost_true_positive_feedback = feedback == "boost";
        if (states_raw > std::numeric_limits<std::uint16_t>::max() / 2U || threshold_raw > std::numeric_limits<int>::max())
            throw std::invalid_argument("native configuration exceeds its range");
        std::array<ptm::ScalarBinaryTM, 10> models = {
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 0U),
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 1U),
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 2U),
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 3U),
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 4U),
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 5U),
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 6U),
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 7U),
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 8U),
            ptm::ScalarBinaryTM(clauses, train.features, states_raw, specificity, threshold_raw, seed + 9U)};
        double cumulative = 0.0;
        std::mt19937_64 competition_rng(seed ^ 0x6a09e667f3bcc909ULL);
        std::mt19937_64 shuffle_rng(seed ^ 0xbb67ae8584caa73bULL);
        std::vector<std::size_t> training_order(train.rows);
        std::iota(training_order.begin(), training_order.end(), 0U);
        for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
            std::shuffle(training_order.begin(), training_order.end(), shuffle_rng);
            const auto started = std::chrono::steady_clock::now();
            if (policy == "all") {
                std::array<std::thread, 10> workers;
                for (std::uint8_t cls = 0; cls < 10U; ++cls) {
                    workers[cls] = std::thread([&, cls] {
                        for (const auto row : training_order) {
                            const auto features = std::span<const std::uint8_t>(
                                train.values.data() + row * train.features,
                                train.features);
                            models[cls].update(
                                features,
                                static_cast<int>(train.labels[row] == cls),
                                boost_true_positive_feedback);
                        }
                    });
                }
                for (auto& worker : workers) worker.join();
            } else {
                std::vector<std::uint8_t> competitors(train.rows);
                for (std::size_t position = 0; position < train.rows; ++position) {
                    const auto target = train.labels[training_order[position]];
                    competitors[position] = static_cast<std::uint8_t>(
                        (target + 1U + competition_rng() % 9U) % 10U);
                }
                std::array<std::thread, 10> workers;
                for (std::uint8_t cls = 0; cls < 10U; ++cls) {
                    workers[cls] = std::thread([&, cls] {
                        for (std::size_t position = 0; position < train.rows;
                             ++position) {
                            const auto row = training_order[position];
                            const auto target = train.labels[row];
                            if (target != cls && competitors[position] != cls) continue;
                            const auto features = std::span<const std::uint8_t>(
                                train.values.data() + row * train.features,
                                train.features);
                            models[cls].update(features,
                                               static_cast<int>(target == cls),
                                               boost_true_positive_feedback);
                        }
                    });
                }
                for (auto& worker : workers) worker.join();
            }
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            cumulative += elapsed;
            std::cout << std::setprecision(17)
                      << "{\"schema\":\"ptm.mnist-ovr-epoch.v1\",\"epoch\":" << epoch
                      << ",\"clauses_per_class\":" << clauses
                      << ",\"threshold\":" << threshold_raw
                      << ",\"specificity\":" << specificity
                      << ",\"training_policy\":\"" << policy << "\""
                      << ",\"epoch_shuffle\":true"
                      << ",\"parallel_class_training\":true"
                      << ",\"parallel_validation\":true"
                      << ",\"boost_true_positive_feedback\":"
                      << (boost_true_positive_feedback ? "true" : "false")
                      << ",\"training_seconds\":" << elapsed
                      << ",\"cumulative_training_seconds\":" << cumulative
                      << ",\"validation_accuracy\":" << accuracy(models, validation) << "}\n"
                      << std::flush;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PTM MNIST OVR benchmark failure: " << error.what() << '\n';
        return 1;
    }
}
