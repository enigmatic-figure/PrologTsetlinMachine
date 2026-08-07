#include "ptm/packed_tm.hpp"
#include "ptm/packed_tm_cuda.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void require_same_result(const ptm::PackedTMResult64& expected,
                         const ptm::PackedTMResult64& actual,
                         std::string_view context) {
    require(actual.valid_example_mask == expected.valid_example_mask,
            std::string(context) + " changed the valid-example mask");
    require(actual.prediction_mask == expected.prediction_mask,
            std::string(context) + " changed the prediction mask");
    require(actual.scores == expected.scores,
            std::string(context) + " changed signed/clamped scores");
    require(actual.clause_outputs == expected.clause_outputs,
            std::string(context) + " changed prediction clause words");
    require(actual.feedback_clause_outputs ==
                expected.feedback_clause_outputs,
            std::string(context) + " changed feedback clause words");
}

std::vector<std::uint64_t> random_pages(std::size_t page_count,
                                        std::size_t feature_count,
                                        std::mt19937_64& random) {
    std::vector<std::uint64_t> result(page_count * feature_count);
    for (auto& word : result) {
        word = random();
    }
    return result;
}

void compare_all_pages(const ptm::PackedTMModel64& model,
                       ptm::CudaPackedTMExecutor64& executor,
                       const std::vector<std::uint64_t>& feature_pages,
                       const std::vector<std::uint64_t>& valid_masks,
                       ptm::CudaPackedTMBackend backend,
                       ptm::CudaPackedTMVoteBackend vote_backend =
                           ptm::CudaPackedTMVoteBackend::two_stage) {
    ptm::CudaPackedTMTiming timing{};
    executor.upload_pages(feature_pages, valid_masks, &timing);
    executor.evaluate_resident(backend, 2, &timing, vote_backend);
    const auto actual = executor.download(&timing);
    require(actual.page_count == valid_masks.size(),
            "CUDA result returned the wrong page count");
    require(actual.clause_count == model.number_of_clauses(),
            "CUDA result returned the wrong clause count");
    require(timing.input_upload_ms >= 0.0F && timing.kernel_ms >= 0.0F &&
                timing.result_download_ms >= 0.0F,
            "CUDA timing contained a negative duration");
    for (std::size_t page = 0; page < valid_masks.size(); ++page) {
        const auto first = page * model.number_of_features();
        const auto features = std::span<const std::uint64_t>(
            feature_pages.data() + first, model.number_of_features());
        const auto expected = model.evaluate(
            features, valid_masks[page], ptm::PackedTMBackend::scalar);
        const auto context =
            std::string(ptm::cuda_packed_tm_backend_name(backend)) + "/" +
            ptm::cuda_packed_tm_vote_backend_name(vote_backend);
        require_same_result(expected, actual.page(page), context);
    }
}

void test_device_metadata() {
    const auto device = ptm::cuda_device_capabilities();
    require(device.available, "CUDA device unexpectedly became unavailable");
    require(!device.name.empty(), "CUDA device name is empty");
    require(device.compute_capability_major > 0,
            "CUDA compute capability is invalid");
    require(device.total_global_memory > 0,
            "CUDA global-memory capacity is invalid");
    require(device.multiprocessor_count > 0,
            "CUDA multiprocessor count is invalid");
    require(device.warp_size == 32, "CUDA warp size is not 32");
    require(device.driver_version > 0 && device.runtime_version > 0,
            "CUDA driver/runtime versions are invalid");
}

void test_empty_contradictory_negative_and_clipping() {
    constexpr std::size_t clauses = 11;
    constexpr std::size_t features = 65;
    constexpr std::uint16_t states_per_action = 31;
    std::vector<std::uint16_t> states(
        clauses * features * 2U, states_per_action);
    const auto include = [&](std::size_t clause, std::size_t literal) {
        states[clause * features * 2U + literal] =
            states_per_action + 1U;
    };
    // Clause 0 remains empty. Clause 1 is contradictory. Clause 2 contains a
    // negative literal. Remaining even clauses create threshold clipping.
    include(1, 0);
    include(1, 1);
    include(2, 1);
    include(3, 0);
    for (std::size_t clause = 4; clause < clauses; ++clause) {
        include(clause, (clause % 2U) == 0 ? 2U : 3U);
    }
    const ptm::PackedTMModel64 model(
        clauses, features, states_per_action, 2, states);
    ptm::CudaPackedTMExecutor64 executor(model);
    std::mt19937_64 random(0xc0da'2026U);
    auto pages = random_pages(4, features, random);
    pages[0] |= 1U;
    const std::vector<std::uint64_t> masks{
        1U,
        (std::uint64_t{1} << 17U) - 1U,
        (std::uint64_t{1} << 63U) - 1U,
        0xa55a'0ff0'f00f'1357ULL,
    };
    for (const auto backend : {ptm::CudaPackedTMBackend::sparse,
                               ptm::CudaPackedTMBackend::warp_tile,
                               ptm::CudaPackedTMBackend::dense_bitset}) {
        for (const auto vote_backend : {
                 ptm::CudaPackedTMVoteBackend::two_stage,
                 ptm::CudaPackedTMVoteBackend::fused_atomic}) {
            compare_all_pages(
                model, executor, pages, masks, backend, vote_backend);
        }
    }
}

void test_randomized_exact_equivalence() {
    std::mt19937_64 random(0x51a7'c0da'2026ULL);
    const std::array<std::size_t, 3> clause_counts{1, 33, 257};
    const std::array<std::size_t, 3> feature_counts{1, 65, 257};
    const std::vector<std::uint64_t> masks{
        1U,
        (std::uint64_t{1} << 17U) - 1U,
        (std::uint64_t{1} << 63U) - 1U,
        ~std::uint64_t{0},
    };
    for (const auto clauses : clause_counts) {
        for (const auto features : feature_counts) {
            constexpr std::uint16_t states_per_action = 100;
            std::vector<std::uint16_t> states(
                clauses * features * 2U, states_per_action);
            for (auto& state : states) {
                if ((random() % 100U) < 10U) {
                    state = static_cast<std::uint16_t>(
                        states_per_action + 1U + random() %
                            states_per_action);
                } else {
                    state = static_cast<std::uint16_t>(
                        1U + random() % states_per_action);
                }
            }
            const ptm::PackedTMModel64 model(
                clauses, features, states_per_action, 7, states);
            ptm::CudaPackedTMExecutor64 executor(model);
            const auto pages = random_pages(masks.size(), features, random);
            for (const auto backend : {ptm::CudaPackedTMBackend::sparse,
                                       ptm::CudaPackedTMBackend::warp_tile,
                                       ptm::CudaPackedTMBackend::dense_bitset}) {
                for (const auto vote_backend : {
                         ptm::CudaPackedTMVoteBackend::two_stage,
                         ptm::CudaPackedTMVoteBackend::fused_atomic}) {
                    compare_all_pages(
                        model, executor, pages, masks, backend, vote_backend);
                }
            }
        }
    }
}

void test_dense_bitset_high_density_equivalence() {
    constexpr std::size_t clauses = 65;
    constexpr std::size_t features = 129;
    constexpr std::uint16_t states_per_action = 37;
    std::mt19937_64 random(0xde05e'2026ULL);
    std::vector<std::uint16_t> states(
        clauses * features * 2U, states_per_action);
    for (auto& state : states) {
        if ((random() % 100U) < 75U) {
            state = static_cast<std::uint16_t>(
                states_per_action + 1U + random() % states_per_action);
        } else {
            state = static_cast<std::uint16_t>(
                1U + random() % states_per_action);
        }
    }
    const ptm::PackedTMModel64 model(
        clauses, features, states_per_action, 11, states);
    ptm::CudaPackedTMExecutor64 executor(model);
    const auto pages = random_pages(3, features, random);
    const std::vector<std::uint64_t> masks{
        ~std::uint64_t{0},
        (std::uint64_t{1} << 17U) - 1U,
        0xa55a'0ff0'f00f'1357ULL,
    };
    for (const auto vote_backend : {
             ptm::CudaPackedTMVoteBackend::two_stage,
             ptm::CudaPackedTMVoteBackend::fused_atomic}) {
        compare_all_pages(model, executor, pages, masks,
                          ptm::CudaPackedTMBackend::dense_bitset,
                          vote_backend);
    }
}

}  // namespace

int main() {
    try {
        const auto device = ptm::cuda_device_capabilities();
        if (!device.available) {
            std::cout << "CUDA tests skipped: " << device.status << '\n';
            return 0;
        }
        test_device_metadata();
        test_empty_contradictory_negative_and_clipping();
        test_randomized_exact_equivalence();
        test_dense_bitset_high_density_equivalence();
        std::cout << "CUDA packed TM tests passed on " << device.name
                  << " (compute " << device.compute_capability_major << '.'
                  << device.compute_capability_minor << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA packed TM test failure: " << error.what() << '\n';
        return 1;
    }
}
