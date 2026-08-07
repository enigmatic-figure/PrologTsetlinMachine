#include "packed_tm_cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace ptm::cuda_detail {

namespace {

void set_error(char* destination, const char* format, ...) noexcept {
    if (destination == nullptr) {
        return;
    }
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(destination, error_buffer_size, format, arguments);
    va_end(arguments);
}

int fail_cuda(cudaError_t status, const char* operation, char* error) noexcept {
    set_error(error, "%s failed: %s", operation, cudaGetErrorString(status));
    return static_cast<int>(status == cudaSuccess ? cudaErrorUnknown : status);
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] cudaError_t resize(std::size_t count) noexcept {
        if (count == count_) {
            return cudaSuccess;
        }
        reset();
        if (count == 0) {
            return cudaSuccess;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            return cudaErrorMemoryAllocation;
        }
        const auto status = cudaMalloc(
            reinterpret_cast<void**>(&data_), count * sizeof(T));
        if (status == cudaSuccess) {
            count_ = count;
        }
        return status;
    }

    void reset() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }
        data_ = nullptr;
        count_ = 0;
    }

    [[nodiscard]] T* get() noexcept { return data_; }
    [[nodiscard]] const T* get() const noexcept { return data_; }

private:
    T* data_{};
    std::size_t count_{};
};

template <typename Function>
cudaError_t measure_cuda(Function&& function, float* elapsed_ms) noexcept {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    auto status = cudaEventCreate(&start);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaEventCreate(&stop);
    if (status != cudaSuccess) {
        static_cast<void>(cudaEventDestroy(start));
        return status;
    }
    status = cudaEventRecord(start);
    if (status == cudaSuccess) {
        status = function();
    }
    if (status == cudaSuccess) {
        status = cudaEventRecord(stop);
    }
    if (status == cudaSuccess) {
        status = cudaEventSynchronize(stop);
    }
    float measured = 0.0F;
    if (status == cudaSuccess) {
        status = cudaEventElapsedTime(&measured, start, stop);
    }
    static_cast<void>(cudaEventDestroy(stop));
    static_cast<void>(cudaEventDestroy(start));
    if (status == cudaSuccess && elapsed_ms != nullptr) {
        *elapsed_ms = measured;
    }
    return status;
}

[[nodiscard]] bool product_overflows(std::size_t left,
                                     std::size_t right) noexcept {
    return left != 0 &&
           right > std::numeric_limits<std::size_t>::max() / left;
}

__device__ std::uint64_t evaluate_clause(
    std::uint32_t clause,
    std::size_t page,
    std::uint32_t feature_count,
    const std::uint32_t* offsets,
    const std::uint32_t* literal_features,
    const std::uint8_t* literal_negated,
    const std::uint64_t* feature_words,
    const std::uint64_t* valid_masks,
    bool* empty) {
    const auto first = offsets[clause];
    const auto last = offsets[clause + 1U];
    auto output = valid_masks[page];
    const auto page_offset = page * static_cast<std::size_t>(feature_count);
    for (auto literal = first; literal < last; ++literal) {
        auto truth = feature_words[
            page_offset + literal_features[literal]];
        if (literal_negated[literal] != 0) {
            truth = ~truth;
        }
        output &= truth;
    }
    *empty = first == last;
    return output & valid_masks[page];
}

template <bool FuseVotes>
__device__ void publish_clause_result(
    std::uint32_t clause,
    std::size_t page,
    std::uint32_t clause_count,
    bool empty,
    std::uint64_t output,
    std::uint64_t* clause_outputs,
    std::uint64_t* feedback_outputs,
    std::int32_t* scores) {
    const auto destination =
        page * static_cast<std::size_t>(clause_count) + clause;
    feedback_outputs[destination] = output;
    clause_outputs[destination] = empty ? 0 : output;
    if constexpr (FuseVotes) {
        if (empty) {
            return;
        }
        const auto contribution = (clause & 1U) == 0 ? 1 : -1;
        auto active_lanes = output;
        const auto score_offset = page * 64U;
        while (active_lanes != 0) {
            const auto lane = static_cast<unsigned>(__ffsll(active_lanes) - 1);
            atomicAdd(scores + score_offset + lane, contribution);
            active_lanes &= active_lanes - 1U;
        }
    }
}

template <bool FuseVotes>
__global__ void sparse_clause_kernel(
    std::uint32_t clause_count,
    std::uint32_t feature_count,
    const std::uint32_t* offsets,
    const std::uint32_t* literal_features,
    const std::uint8_t* literal_negated,
    const std::uint64_t* feature_words,
    const std::uint64_t* valid_masks,
    std::uint64_t* clause_outputs,
    std::uint64_t* feedback_outputs,
    std::int32_t* scores) {
    const auto clause = static_cast<std::uint32_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const auto page = static_cast<std::size_t>(blockIdx.y);
    if (clause >= clause_count) {
        return;
    }
    bool empty = false;
    const auto output = evaluate_clause(
        clause, page, feature_count, offsets, literal_features,
        literal_negated, feature_words, valid_masks, &empty);
    publish_clause_result<FuseVotes>(
        clause, page, clause_count, empty, output, clause_outputs,
        feedback_outputs, scores);
}

template <bool FuseVotes>
__global__ void warp_tile_clause_kernel(
    std::uint32_t clause_count,
    std::uint32_t feature_count,
    const std::uint32_t* offsets,
    const std::uint32_t* literal_features,
    const std::uint8_t* literal_negated,
    const std::uint64_t* feature_words,
    const std::uint64_t* valid_masks,
    std::uint64_t* clause_outputs,
    std::uint64_t* feedback_outputs,
    std::int32_t* scores) {
    const auto clause = static_cast<std::uint32_t>(
        blockIdx.x * warpSize + threadIdx.x);
    const auto page = static_cast<std::size_t>(blockIdx.y);
    if (clause >= clause_count) {
        return;
    }
    bool empty = false;
    const auto output = evaluate_clause(
        clause, page, feature_count, offsets, literal_features,
        literal_negated, feature_words, valid_masks, &empty);
    publish_clause_result<FuseVotes>(
        clause, page, clause_count, empty, output, clause_outputs,
        feedback_outputs, scores);
}

template <bool FuseVotes>
__global__ void dense_bitset_clause_kernel(
    std::uint32_t clause_count,
    std::uint32_t feature_count,
    std::uint32_t feature_word_count,
    const std::uint64_t* positive_include_masks,
    const std::uint64_t* negative_include_masks,
    const std::uint64_t* feature_words,
    const std::uint64_t* valid_masks,
    std::uint64_t* clause_outputs,
    std::uint64_t* feedback_outputs,
    std::int32_t* scores) {
    const auto clause = static_cast<std::uint32_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const auto page = static_cast<std::size_t>(blockIdx.y);
    if (clause >= clause_count) {
        return;
    }

    const auto valid = valid_masks[page];
    auto output = valid;
    bool empty = true;
    const auto page_offset = page * static_cast<std::size_t>(feature_count);
    for (std::uint32_t word = 0; word < feature_word_count; ++word) {
        const auto mask_index =
            static_cast<std::size_t>(word) * clause_count + clause;
        const auto positive = positive_include_masks[mask_index];
        const auto negative = negative_include_masks[mask_index];
        const auto included = positive | negative;
        empty = empty && included == 0;
        const auto feature_base = word * 64U;
        for (std::uint32_t offset = 0;
             offset < 64U && feature_base + offset < feature_count; ++offset) {
            const auto bit = std::uint64_t{1} << offset;
            if ((included & bit) == 0) {
                continue;
            }
            const auto truth = feature_words[
                page_offset + static_cast<std::size_t>(feature_base + offset)];
            if ((positive & bit) != 0) {
                output &= truth;
            }
            if ((negative & bit) != 0) {
                output &= ~truth;
            }
        }
    }
    output &= valid;
    publish_clause_result<FuseVotes>(
        clause, page, clause_count, empty, output, clause_outputs,
        feedback_outputs, scores);
}

template <bool SumClauseOutputs>
__global__ void vote_kernel(std::uint32_t clause_count,
                            int threshold,
                            const std::uint64_t* valid_masks,
                            const std::uint64_t* clause_outputs,
                            std::int32_t* scores,
                            std::uint64_t* prediction_masks) {
    const auto page = static_cast<std::size_t>(blockIdx.x);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    const auto lane_bit = std::uint64_t{1} << lane;
    int score = 0;
    if ((valid_masks[page] & lane_bit) != 0) {
        if constexpr (SumClauseOutputs) {
            const auto page_offset =
                page * static_cast<std::size_t>(clause_count);
            for (std::uint32_t clause = 0; clause < clause_count; ++clause) {
                if ((clause_outputs[page_offset + clause] & lane_bit) != 0) {
                    score += (clause & 1U) == 0 ? 1 : -1;
                }
            }
        } else {
            score = scores[page * 64U + lane];
        }
        score = score < -threshold ? -threshold : score;
        score = score > threshold ? threshold : score;
    }
    scores[page * 64U + lane] = score;

    __shared__ unsigned warp_predictions[2];
    const auto warp_lane = lane & 31U;
    const auto warp = lane >> 5U;
    const auto votes = __ballot_sync(
        0xffff'ffffU, (valid_masks[page] & lane_bit) != 0 && score > 0);
    if (warp_lane == 0) {
        warp_predictions[warp] = votes;
    }
    __syncthreads();
    if (lane == 0) {
        prediction_masks[page] =
            static_cast<std::uint64_t>(warp_predictions[0]) |
            (static_cast<std::uint64_t>(warp_predictions[1]) << 32U);
    }
}

}  // namespace

struct ExecutorHandle {
    int device_ordinal{};
    std::uint32_t clause_count{};
    std::uint32_t feature_count{};
    int threshold{};
    std::size_t page_count{};
    DeviceBuffer<std::uint32_t> clause_literal_offsets;
    DeviceBuffer<std::uint32_t> literal_features;
    DeviceBuffer<std::uint8_t> literal_negated;
    std::uint32_t feature_word_count{};
    DeviceBuffer<std::uint64_t> positive_include_masks;
    DeviceBuffer<std::uint64_t> negative_include_masks;
    DeviceBuffer<std::uint64_t> feature_words;
    DeviceBuffer<std::uint64_t> valid_masks;
    DeviceBuffer<std::uint64_t> prediction_masks;
    DeviceBuffer<std::int32_t> scores;
    DeviceBuffer<std::uint64_t> clause_outputs;
    DeviceBuffer<std::uint64_t> feedback_clause_outputs;
};

int get_device_info(int device_ordinal,
                    DeviceInfo* info,
                    char* error) noexcept {
    if (info == nullptr || device_ordinal < 0) {
        set_error(error, "invalid CUDA device query");
        return -1;
    }
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        return fail_cuda(status, "cudaGetDeviceCount", error);
    }
    if (device_ordinal >= count) {
        set_error(error, "CUDA device %d is unavailable; device count is %d",
                  device_ordinal, count);
        return -1;
    }
    status = cudaSetDevice(device_ordinal);
    if (status != cudaSuccess) {
        return fail_cuda(status, "cudaSetDevice", error);
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device_ordinal);
    if (status != cudaSuccess) {
        return fail_cuda(status, "cudaGetDeviceProperties", error);
    }
    std::snprintf(info->name, sizeof(info->name), "%s", properties.name);
    info->compute_capability_major = properties.major;
    info->compute_capability_minor = properties.minor;
    info->total_global_memory = properties.totalGlobalMem;
    info->multiprocessor_count = properties.multiProcessorCount;
    info->warp_size = properties.warpSize;
    status = cudaDriverGetVersion(&info->driver_version);
    if (status != cudaSuccess) {
        return fail_cuda(status, "cudaDriverGetVersion", error);
    }
    status = cudaRuntimeGetVersion(&info->runtime_version);
    if (status != cudaSuccess) {
        return fail_cuda(status, "cudaRuntimeGetVersion", error);
    }
    return 0;
}

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
                    char* error) noexcept {
    if (handle == nullptr || clause_count == 0 || feature_count == 0 ||
        threshold <= 0 || clause_literal_offsets == nullptr ||
        (included_literal_count != 0 &&
         (literal_features == nullptr || literal_negated == nullptr)) ||
        positive_include_masks == nullptr ||
        negative_include_masks == nullptr ||
        feature_word_count !=
            feature_count / 64U + (feature_count % 64U != 0 ? 1U : 0U) ||
        product_overflows(feature_word_count, clause_count)) {
        set_error(error, "invalid CUDA executor configuration");
        return -1;
    }
    DeviceInfo ignored{};
    if (const auto status = get_device_info(device_ordinal, &ignored, error);
        status != 0) {
        return status;
    }
    auto candidate = std::unique_ptr<ExecutorHandle>(
        new (std::nothrow) ExecutorHandle{});
    if (!candidate) {
        set_error(error, "host allocation for CUDA executor failed");
        return -1;
    }
    candidate->device_ordinal = device_ordinal;
    candidate->clause_count = clause_count;
    candidate->feature_count = feature_count;
    candidate->feature_word_count = feature_word_count;
    candidate->threshold = threshold;
    const auto dense_mask_word_count =
        static_cast<std::size_t>(feature_word_count) * clause_count;

    auto status = candidate->clause_literal_offsets.resize(
        static_cast<std::size_t>(clause_count) + 1U);
    if (status == cudaSuccess) {
        status = candidate->literal_features.resize(included_literal_count);
    }
    if (status == cudaSuccess) {
        status = candidate->literal_negated.resize(included_literal_count);
    }
    if (status == cudaSuccess) {
        status = candidate->positive_include_masks.resize(
            dense_mask_word_count);
    }
    if (status == cudaSuccess) {
        status = candidate->negative_include_masks.resize(
            dense_mask_word_count);
    }
    if (status != cudaSuccess) {
        return fail_cuda(status, "CUDA model allocation", error);
    }
    status = cudaMemcpy(
        candidate->clause_literal_offsets.get(), clause_literal_offsets,
        (static_cast<std::size_t>(clause_count) + 1U) *
            sizeof(std::uint32_t),
        cudaMemcpyHostToDevice);
    if (status == cudaSuccess && included_literal_count != 0) {
        status = cudaMemcpy(
            candidate->literal_features.get(), literal_features,
            static_cast<std::size_t>(included_literal_count) *
                sizeof(std::uint32_t),
            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(
            candidate->positive_include_masks.get(), positive_include_masks,
            dense_mask_word_count * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(
            candidate->negative_include_masks.get(), negative_include_masks,
            dense_mask_word_count * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess && included_literal_count != 0) {
        status = cudaMemcpy(
            candidate->literal_negated.get(), literal_negated,
            static_cast<std::size_t>(included_literal_count) *
                sizeof(std::uint8_t),
            cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) {
        return fail_cuda(status, "CUDA model upload", error);
    }
    *handle = candidate.release();
    return 0;
}

void destroy_executor(ExecutorHandle* handle) noexcept {
    if (handle != nullptr) {
        static_cast<void>(cudaSetDevice(handle->device_ordinal));
        delete handle;
    }
}

int upload_pages(ExecutorHandle* handle,
                 const std::uint64_t* feature_words,
                 const std::uint64_t* valid_example_masks,
                 std::size_t page_count,
                 float* elapsed_ms,
                 char* error) noexcept {
    if (handle == nullptr || feature_words == nullptr ||
        valid_example_masks == nullptr || page_count == 0 ||
        page_count > 65535U ||
        product_overflows(page_count, handle->feature_count) ||
        product_overflows(page_count, handle->clause_count) ||
        product_overflows(page_count, 64U)) {
        set_error(error, "invalid CUDA resident-page configuration");
        return -1;
    }
    auto status = cudaSetDevice(handle->device_ordinal);
    if (status != cudaSuccess) {
        return fail_cuda(status, "cudaSetDevice", error);
    }
    handle->page_count = 0;
    const auto feature_word_count =
        page_count * static_cast<std::size_t>(handle->feature_count);
    const auto clause_word_count =
        page_count * static_cast<std::size_t>(handle->clause_count);
    status = handle->feature_words.resize(feature_word_count);
    if (status == cudaSuccess) {
        status = handle->valid_masks.resize(page_count);
    }
    if (status == cudaSuccess) {
        status = handle->prediction_masks.resize(page_count);
    }
    if (status == cudaSuccess) {
        status = handle->scores.resize(page_count * 64U);
    }
    if (status == cudaSuccess) {
        status = handle->clause_outputs.resize(clause_word_count);
    }
    if (status == cudaSuccess) {
        status = handle->feedback_clause_outputs.resize(clause_word_count);
    }
    if (status != cudaSuccess) {
        return fail_cuda(status, "CUDA resident-page allocation", error);
    }
    status = measure_cuda([&]() noexcept {
        auto copy_status = cudaMemcpy(
            handle->feature_words.get(), feature_words,
            feature_word_count * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice);
        if (copy_status == cudaSuccess) {
            copy_status = cudaMemcpy(
                handle->valid_masks.get(), valid_example_masks,
                page_count * sizeof(std::uint64_t),
                cudaMemcpyHostToDevice);
        }
        return copy_status;
    }, elapsed_ms);
    if (status != cudaSuccess) {
        return fail_cuda(status, "CUDA input upload", error);
    }
    handle->page_count = page_count;
    return 0;
}

int evaluate_resident(ExecutorHandle* handle,
                      int backend,
                      int vote_backend,
                      std::size_t repeats,
                      float* elapsed_ms,
                      char* error) noexcept {
    if (handle == nullptr || handle->page_count == 0 || repeats == 0 ||
        (backend < 0 || backend > 2) ||
        (vote_backend < 0 || vote_backend > 1)) {
        set_error(error, "invalid CUDA resident evaluation");
        return -1;
    }
    auto status = cudaSetDevice(handle->device_ordinal);
    if (status != cudaSuccess) {
        return fail_cuda(status, "cudaSetDevice", error);
    }
    const auto sparse_blocks = dim3(
        (handle->clause_count + 255U) / 256U,
        static_cast<unsigned>(handle->page_count));
    const auto warp_blocks = dim3(
        (handle->clause_count + 31U) / 32U,
        static_cast<unsigned>(handle->page_count));
    status = measure_cuda([&]() noexcept {
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            const auto fused_votes = vote_backend == 1;
            auto launch_status = cudaSuccess;
            if (fused_votes) {
                launch_status = cudaMemsetAsync(
                    handle->scores.get(), 0,
                    handle->page_count * 64U * sizeof(std::int32_t));
            }
            if (launch_status != cudaSuccess) {
                return launch_status;
            }
            if (backend == 0) {
                if (fused_votes) {
                    sparse_clause_kernel<true><<<sparse_blocks, 256>>>(
                        handle->clause_count, handle->feature_count,
                        handle->clause_literal_offsets.get(),
                        handle->literal_features.get(),
                        handle->literal_negated.get(),
                        handle->feature_words.get(), handle->valid_masks.get(),
                        handle->clause_outputs.get(),
                        handle->feedback_clause_outputs.get(),
                        handle->scores.get());
                } else {
                    sparse_clause_kernel<false><<<sparse_blocks, 256>>>(
                        handle->clause_count, handle->feature_count,
                        handle->clause_literal_offsets.get(),
                        handle->literal_features.get(),
                        handle->literal_negated.get(),
                        handle->feature_words.get(), handle->valid_masks.get(),
                        handle->clause_outputs.get(),
                        handle->feedback_clause_outputs.get(),
                        handle->scores.get());
                }
            } else if (backend == 1) {
                if (fused_votes) {
                    warp_tile_clause_kernel<true><<<warp_blocks, 32>>>(
                        handle->clause_count, handle->feature_count,
                        handle->clause_literal_offsets.get(),
                        handle->literal_features.get(),
                        handle->literal_negated.get(),
                        handle->feature_words.get(), handle->valid_masks.get(),
                        handle->clause_outputs.get(),
                        handle->feedback_clause_outputs.get(),
                        handle->scores.get());
                } else {
                    warp_tile_clause_kernel<false><<<warp_blocks, 32>>>(
                        handle->clause_count, handle->feature_count,
                        handle->clause_literal_offsets.get(),
                        handle->literal_features.get(),
                        handle->literal_negated.get(),
                        handle->feature_words.get(), handle->valid_masks.get(),
                        handle->clause_outputs.get(),
                        handle->feedback_clause_outputs.get(),
                        handle->scores.get());
                }
            } else {
                if (fused_votes) {
                    dense_bitset_clause_kernel<true><<<sparse_blocks, 256>>>(
                        handle->clause_count, handle->feature_count,
                        handle->feature_word_count,
                        handle->positive_include_masks.get(),
                        handle->negative_include_masks.get(),
                        handle->feature_words.get(), handle->valid_masks.get(),
                        handle->clause_outputs.get(),
                        handle->feedback_clause_outputs.get(),
                        handle->scores.get());
                } else {
                    dense_bitset_clause_kernel<false><<<sparse_blocks, 256>>>(
                        handle->clause_count, handle->feature_count,
                        handle->feature_word_count,
                        handle->positive_include_masks.get(),
                        handle->negative_include_masks.get(),
                        handle->feature_words.get(), handle->valid_masks.get(),
                        handle->clause_outputs.get(),
                        handle->feedback_clause_outputs.get(),
                        handle->scores.get());
                }
            }
            launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                return launch_status;
            }
            if (fused_votes) {
                vote_kernel<false>
                    <<<static_cast<unsigned>(handle->page_count), 64>>>(
                        handle->clause_count, handle->threshold,
                        handle->valid_masks.get(),
                        handle->clause_outputs.get(), handle->scores.get(),
                        handle->prediction_masks.get());
            } else {
                vote_kernel<true>
                    <<<static_cast<unsigned>(handle->page_count), 64>>>(
                        handle->clause_count, handle->threshold,
                        handle->valid_masks.get(),
                        handle->clause_outputs.get(), handle->scores.get(),
                        handle->prediction_masks.get());
            }
            launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                return launch_status;
            }
        }
        return cudaSuccess;
    }, elapsed_ms);
    if (status != cudaSuccess) {
        return fail_cuda(status, "CUDA packed TM kernels", error);
    }
    return 0;
}

int download_results(ExecutorHandle* handle,
                     std::uint64_t* prediction_masks,
                     std::int32_t* scores,
                     std::uint64_t* clause_outputs,
                     std::uint64_t* feedback_clause_outputs,
                     float* elapsed_ms,
                     char* error) noexcept {
    if (handle == nullptr || handle->page_count == 0 ||
        prediction_masks == nullptr || scores == nullptr ||
        clause_outputs == nullptr || feedback_clause_outputs == nullptr) {
        set_error(error, "invalid CUDA result download");
        return -1;
    }
    auto status = cudaSetDevice(handle->device_ordinal);
    if (status != cudaSuccess) {
        return fail_cuda(status, "cudaSetDevice", error);
    }
    const auto clause_word_count =
        handle->page_count * static_cast<std::size_t>(handle->clause_count);
    status = measure_cuda([&]() noexcept {
        auto copy_status = cudaMemcpy(
            prediction_masks, handle->prediction_masks.get(),
            handle->page_count * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost);
        if (copy_status == cudaSuccess) {
            copy_status = cudaMemcpy(
                scores, handle->scores.get(),
                handle->page_count * 64U * sizeof(std::int32_t),
                cudaMemcpyDeviceToHost);
        }
        if (copy_status == cudaSuccess) {
            copy_status = cudaMemcpy(
                clause_outputs, handle->clause_outputs.get(),
                clause_word_count * sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost);
        }
        if (copy_status == cudaSuccess) {
            copy_status = cudaMemcpy(
                feedback_clause_outputs,
                handle->feedback_clause_outputs.get(),
                clause_word_count * sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost);
        }
        return copy_status;
    }, elapsed_ms);
    if (status != cudaSuccess) {
        return fail_cuda(status, "CUDA result download", error);
    }
    return 0;
}

}  // namespace ptm::cuda_detail
