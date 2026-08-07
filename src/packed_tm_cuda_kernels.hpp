#pragma once

#include <cstddef>
#include <cstdint>

namespace ptm::cuda_detail {

inline constexpr std::size_t error_buffer_size = 512;

struct DeviceInfo {
    char name[256]{};
    int compute_capability_major{};
    int compute_capability_minor{};
    std::size_t total_global_memory{};
    int multiprocessor_count{};
    int warp_size{};
    int driver_version{};
    int runtime_version{};
};

struct ExecutorHandle;

int get_device_info(int device_ordinal,
                    DeviceInfo* info,
                    char* error) noexcept;

int create_executor(int device_ordinal,
                    std::uint32_t clause_count,
                    std::uint32_t feature_count,
                    int threshold,
                    const std::uint32_t* clause_literal_offsets,
                    const std::uint32_t* literal_features,
                    const std::uint8_t* literal_negated,
                    std::uint32_t included_literal_count,
                    const std::uint64_t* positive_include_masks,
                    const std::uint64_t* negative_include_masks,
                    std::uint32_t feature_word_count,
                    ExecutorHandle** handle,
                    char* error) noexcept;

void destroy_executor(ExecutorHandle* handle) noexcept;

int upload_pages(ExecutorHandle* handle,
                 const std::uint64_t* feature_words,
                 const std::uint64_t* valid_example_masks,
                 std::size_t page_count,
                 float* elapsed_ms,
                 char* error) noexcept;

int evaluate_resident(ExecutorHandle* handle,
                      int backend,
                      int vote_backend,
                      std::size_t repeats,
                      float* elapsed_ms,
                      char* error) noexcept;

int download_results(ExecutorHandle* handle,
                     std::uint64_t* prediction_masks,
                     std::int32_t* scores,
                     std::uint64_t* clause_outputs,
                     std::uint64_t* feedback_clause_outputs,
                     float* elapsed_ms,
                     char* error) noexcept;

}  // namespace ptm::cuda_detail
