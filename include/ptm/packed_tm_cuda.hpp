#pragma once

#include "ptm/packed_tm.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ptm {

enum class CudaPackedTMBackend : std::uint8_t {
    sparse,
    warp_tile,
    dense_bitset,
};

[[nodiscard]] const char* cuda_packed_tm_backend_name(
    CudaPackedTMBackend backend) noexcept;

enum class CudaPackedTMVoteBackend : std::uint8_t {
    two_stage,
    fused_atomic,
};

[[nodiscard]] const char* cuda_packed_tm_vote_backend_name(
    CudaPackedTMVoteBackend backend) noexcept;

struct CudaDeviceCapabilities {
    bool available{};
    int device_ordinal{};
    std::string name;
    int compute_capability_major{};
    int compute_capability_minor{};
    std::size_t total_global_memory{};
    int multiprocessor_count{};
    int warp_size{};
    int driver_version{};
    int runtime_version{};
    std::string status;
};

[[nodiscard]] CudaDeviceCapabilities cuda_device_capabilities(
    int device_ordinal = 0);

struct CudaPackedTMTiming {
    float input_upload_ms{};
    float kernel_ms{};
    float result_download_ms{};
};

// Results are page-major. Each page represents 64 feature-major examples.
struct CudaPackedTMResult64 {
    std::size_t page_count{};
    std::size_t clause_count{};
    std::vector<std::uint64_t> valid_example_masks;
    std::vector<std::uint64_t> prediction_masks;
    std::vector<std::int32_t> scores;
    std::vector<std::uint64_t> clause_outputs;
    std::vector<std::uint64_t> feedback_clause_outputs;

    [[nodiscard]] PackedTMResult64 page(std::size_t page_index) const;
};

// Native-only experimental CUDA execution object. Immutable clause plans are
// uploaded once; feature-major pages remain resident across repeated kernels.
// This type is deliberately absent from C ABI version 2.
class CudaPackedTMExecutor64 {
public:
    explicit CudaPackedTMExecutor64(const PackedTMModel64& model,
                                    int device_ordinal = 0);
    ~CudaPackedTMExecutor64();

    CudaPackedTMExecutor64(CudaPackedTMExecutor64&&) noexcept;
    CudaPackedTMExecutor64& operator=(CudaPackedTMExecutor64&&) noexcept;
    CudaPackedTMExecutor64(const CudaPackedTMExecutor64&) = delete;
    CudaPackedTMExecutor64& operator=(const CudaPackedTMExecutor64&) = delete;

    [[nodiscard]] std::size_t number_of_clauses() const noexcept;
    [[nodiscard]] std::size_t number_of_features() const noexcept;
    [[nodiscard]] std::size_t resident_page_count() const noexcept;
    [[nodiscard]] int device_ordinal() const noexcept;

    void upload_pages(std::span<const std::uint64_t> feature_words,
                      std::span<const std::uint64_t> valid_example_masks,
                      CudaPackedTMTiming* timing = nullptr);

    // Repeats execute against the same resident pages. Only the final outputs
    // are retained; kernel_ms covers all repeated clause and vote launches.
    void evaluate_resident(CudaPackedTMBackend backend,
                           std::size_t repeats = 1,
                           CudaPackedTMTiming* timing = nullptr,
                           CudaPackedTMVoteBackend vote_backend =
                               CudaPackedTMVoteBackend::two_stage);

    [[nodiscard]] CudaPackedTMResult64 download(
        CudaPackedTMTiming* timing = nullptr) const;

    [[nodiscard]] CudaPackedTMResult64 evaluate(
        std::span<const std::uint64_t> feature_words,
        std::span<const std::uint64_t> valid_example_masks,
        CudaPackedTMBackend backend,
        CudaPackedTMTiming* timing = nullptr,
        CudaPackedTMVoteBackend vote_backend =
            CudaPackedTMVoteBackend::two_stage);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ptm
