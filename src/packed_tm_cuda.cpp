#include "ptm/packed_tm_cuda.hpp"

#include "packed_tm_cuda_kernels.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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

void require_u32(std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(message);
    }
}

void check_cuda(int status,
                const std::array<char, cuda_detail::error_buffer_size>& error,
                const char* operation) {
    if (status == 0) {
        return;
    }
    const auto detail = error.front() == '\0' ? "unknown CUDA error"
                                               : error.data();
    throw std::runtime_error(std::string(operation) + ": " + detail);
}

}  // namespace

const char* cuda_packed_tm_backend_name(CudaPackedTMBackend backend) noexcept {
    switch (backend) {
        case CudaPackedTMBackend::sparse:
            return "cuda_sparse";
        case CudaPackedTMBackend::warp_tile:
            return "cuda_warp_tile";
        case CudaPackedTMBackend::dense_bitset:
            return "cuda_dense_bitset";
    }
    return "cuda_unknown";
}

const char* cuda_packed_tm_vote_backend_name(
    CudaPackedTMVoteBackend backend) noexcept {
    switch (backend) {
        case CudaPackedTMVoteBackend::two_stage:
            return "two_stage";
        case CudaPackedTMVoteBackend::fused_atomic:
            return "fused_atomic";
    }
    return "vote_unknown";
}

CudaDeviceCapabilities cuda_device_capabilities(int device_ordinal) {
    CudaDeviceCapabilities result{};
    result.device_ordinal = device_ordinal;
    cuda_detail::DeviceInfo info{};
    std::array<char, cuda_detail::error_buffer_size> error{};
    const auto status = cuda_detail::get_device_info(
        device_ordinal, &info, error.data());
    if (status != 0) {
        result.status = error.front() == '\0' ? "unknown CUDA error"
                                              : error.data();
        return result;
    }
    result.available = true;
    result.name = info.name;
    result.compute_capability_major = info.compute_capability_major;
    result.compute_capability_minor = info.compute_capability_minor;
    result.total_global_memory = info.total_global_memory;
    result.multiprocessor_count = info.multiprocessor_count;
    result.warp_size = info.warp_size;
    result.driver_version = info.driver_version;
    result.runtime_version = info.runtime_version;
    result.status = "ok";
    return result;
}

PackedTMResult64 CudaPackedTMResult64::page(std::size_t page_index) const {
    if (page_index >= page_count) {
        throw std::out_of_range("CUDA packed TM page index");
    }
    const auto score_first = page_index * packed_tm_batch_width;
    const auto clause_first = page_index * clause_count;
    if (valid_example_masks.size() != page_count ||
        prediction_masks.size() != page_count ||
        scores.size() != page_count * packed_tm_batch_width ||
        clause_outputs.size() != page_count * clause_count ||
        feedback_clause_outputs.size() != page_count * clause_count) {
        throw std::logic_error("CUDA packed TM result has inconsistent storage");
    }
    PackedTMResult64 result{};
    result.valid_example_mask = valid_example_masks[page_index];
    result.prediction_mask = prediction_masks[page_index];
    std::copy_n(scores.begin() + static_cast<std::ptrdiff_t>(score_first),
                packed_tm_batch_width, result.scores.begin());
    result.clause_outputs.assign(
        clause_outputs.begin() + static_cast<std::ptrdiff_t>(clause_first),
        clause_outputs.begin() +
            static_cast<std::ptrdiff_t>(clause_first + clause_count));
    result.feedback_clause_outputs.assign(
        feedback_clause_outputs.begin() +
            static_cast<std::ptrdiff_t>(clause_first),
        feedback_clause_outputs.begin() +
            static_cast<std::ptrdiff_t>(clause_first + clause_count));
    return result;
}

class CudaPackedTMExecutor64::Impl {
public:
    Impl(std::size_t model_clause_count,
         std::size_t model_feature_count,
         int model_threshold,
         std::span<const std::uint32_t> offsets,
         std::span<const std::uint32_t> features,
         std::span<const std::uint8_t> negated,
         std::span<const std::uint64_t> positive_include_masks,
         std::span<const std::uint64_t> negative_include_masks,
         std::uint32_t feature_word_count,
         int device_ordinal)
        : clause_count(model_clause_count),
          feature_count(model_feature_count),
          threshold(model_threshold),
          device(device_ordinal) {
        std::array<char, cuda_detail::error_buffer_size> error{};
        check_cuda(cuda_detail::create_executor(
                       device,
                       static_cast<std::uint32_t>(clause_count),
                       static_cast<std::uint32_t>(feature_count), threshold,
                       offsets.data(), features.data(), negated.data(),
                       static_cast<std::uint32_t>(features.size()),
                       positive_include_masks.data(),
                       negative_include_masks.data(), feature_word_count,
                       &handle, error.data()),
                   error, "create CUDA packed TM executor");
    }

    ~Impl() { cuda_detail::destroy_executor(handle); }

    std::size_t clause_count{};
    std::size_t feature_count{};
    int threshold{};
    int device{};
    std::size_t page_count{};
    std::vector<std::uint64_t> valid_masks;
    cuda_detail::ExecutorHandle* handle{};
};

CudaPackedTMExecutor64::CudaPackedTMExecutor64(const PackedTMModel64& model,
                                               int device_ordinal) {
    require_u32(model.number_of_clauses_,
                "CUDA clause count exceeds uint32");
    require_u32(model.number_of_features_,
                "CUDA feature count exceeds uint32");
    require_u32(model.literal_features_.size(),
                "CUDA included-literal count exceeds uint32");

    std::vector<std::uint32_t> offsets;
    offsets.reserve(model.clause_literal_offsets_.size());
    for (const auto offset : model.clause_literal_offsets_) {
        require_u32(offset, "CUDA clause offset exceeds uint32");
        offsets.push_back(static_cast<std::uint32_t>(offset));
    }
    std::vector<std::uint32_t> features;
    features.reserve(model.literal_features_.size());
    for (const auto feature : model.literal_features_) {
        require_u32(feature, "CUDA literal feature exceeds uint32");
        features.push_back(static_cast<std::uint32_t>(feature));
    }
    const auto feature_word_count =
        (model.number_of_features_ + 63U) / 64U;
    require_u32(feature_word_count,
                "CUDA dense feature-word count exceeds uint32");
    const auto dense_mask_words = checked_product(
        feature_word_count, model.number_of_clauses_,
        "CUDA dense Include-mask size overflows size_t");
    std::vector<std::uint64_t> positive_include_masks(dense_mask_words, 0);
    std::vector<std::uint64_t> negative_include_masks(dense_mask_words, 0);
    for (std::size_t clause = 0; clause < model.number_of_clauses_; ++clause) {
        for (auto literal = model.clause_literal_offsets_[clause];
             literal < model.clause_literal_offsets_[clause + 1U]; ++literal) {
            const auto feature = model.literal_features_[literal];
            const auto destination =
                (feature / 64U) * model.number_of_clauses_ + clause;
            const auto bit = std::uint64_t{1} << (feature % 64U);
            if (model.literal_negated_[literal] != 0) {
                negative_include_masks[destination] |= bit;
            } else {
                positive_include_masks[destination] |= bit;
            }
        }
    }
    impl_ = std::make_unique<Impl>(
        model.number_of_clauses_, model.number_of_features_, model.threshold_,
        offsets, features, model.literal_negated_, positive_include_masks,
        negative_include_masks, static_cast<std::uint32_t>(feature_word_count),
        device_ordinal);
}

CudaPackedTMExecutor64::~CudaPackedTMExecutor64() = default;
CudaPackedTMExecutor64::CudaPackedTMExecutor64(
    CudaPackedTMExecutor64&&) noexcept = default;
CudaPackedTMExecutor64& CudaPackedTMExecutor64::operator=(
    CudaPackedTMExecutor64&&) noexcept = default;

std::size_t CudaPackedTMExecutor64::number_of_clauses() const noexcept {
    return impl_->clause_count;
}

std::size_t CudaPackedTMExecutor64::number_of_features() const noexcept {
    return impl_->feature_count;
}

std::size_t CudaPackedTMExecutor64::resident_page_count() const noexcept {
    return impl_->page_count;
}

int CudaPackedTMExecutor64::device_ordinal() const noexcept {
    return impl_->device;
}

void CudaPackedTMExecutor64::upload_pages(
    std::span<const std::uint64_t> feature_words,
    std::span<const std::uint64_t> valid_example_masks,
    CudaPackedTMTiming* timing) {
    if (valid_example_masks.empty()) {
        throw std::invalid_argument("CUDA packed TM requires at least one page");
    }
    const auto expected_words = checked_product(
        valid_example_masks.size(), impl_->feature_count,
        "CUDA feature-page size overflows size_t");
    if (feature_words.size() != expected_words) {
        throw std::invalid_argument(
            "CUDA feature words do not match page and feature counts");
    }
    std::array<char, cuda_detail::error_buffer_size> error{};
    float elapsed = 0.0F;
    check_cuda(cuda_detail::upload_pages(
                   impl_->handle, feature_words.data(),
                   valid_example_masks.data(), valid_example_masks.size(),
                   &elapsed, error.data()),
               error, "upload CUDA packed TM pages");
    impl_->page_count = valid_example_masks.size();
    impl_->valid_masks.assign(valid_example_masks.begin(),
                              valid_example_masks.end());
    if (timing != nullptr) {
        timing->input_upload_ms = elapsed;
    }
}

void CudaPackedTMExecutor64::evaluate_resident(CudaPackedTMBackend backend,
                                               std::size_t repeats,
                                               CudaPackedTMTiming* timing,
                                               CudaPackedTMVoteBackend vote_backend) {
    if (impl_->page_count == 0) {
        throw std::logic_error("CUDA packed TM has no resident input pages");
    }
    if (repeats == 0) {
        throw std::invalid_argument("CUDA repeat count must be positive");
    }
    std::array<char, cuda_detail::error_buffer_size> error{};
    float elapsed = 0.0F;
    check_cuda(cuda_detail::evaluate_resident(
                   impl_->handle, static_cast<int>(backend),
                   static_cast<int>(vote_backend), repeats, &elapsed,
                   error.data()),
               error, "evaluate CUDA packed TM pages");
    if (timing != nullptr) {
        timing->kernel_ms = elapsed;
    }
}

CudaPackedTMResult64 CudaPackedTMExecutor64::download(
    CudaPackedTMTiming* timing) const {
    if (impl_->page_count == 0) {
        throw std::logic_error("CUDA packed TM has no resident input pages");
    }
    CudaPackedTMResult64 result{};
    result.page_count = impl_->page_count;
    result.clause_count = impl_->clause_count;
    result.valid_example_masks = impl_->valid_masks;
    result.prediction_masks.resize(result.page_count);
    result.scores.resize(checked_product(
        result.page_count, packed_tm_batch_width,
        "CUDA result score size overflows size_t"));
    const auto clause_words = checked_product(
        result.page_count, result.clause_count,
        "CUDA result clause size overflows size_t");
    result.clause_outputs.resize(clause_words);
    result.feedback_clause_outputs.resize(clause_words);

    std::array<char, cuda_detail::error_buffer_size> error{};
    float elapsed = 0.0F;
    check_cuda(cuda_detail::download_results(
                   impl_->handle, result.prediction_masks.data(),
                   result.scores.data(), result.clause_outputs.data(),
                   result.feedback_clause_outputs.data(), &elapsed,
                   error.data()),
               error, "download CUDA packed TM results");
    if (timing != nullptr) {
        timing->result_download_ms = elapsed;
    }
    return result;
}

CudaPackedTMResult64 CudaPackedTMExecutor64::evaluate(
    std::span<const std::uint64_t> feature_words,
    std::span<const std::uint64_t> valid_example_masks,
    CudaPackedTMBackend backend,
    CudaPackedTMTiming* timing,
    CudaPackedTMVoteBackend vote_backend) {
    CudaPackedTMTiming local{};
    upload_pages(feature_words, valid_example_masks, &local);
    evaluate_resident(backend, 1, &local, vote_backend);
    auto result = download(&local);
    if (timing != nullptr) {
        *timing = local;
    }
    return result;
}

}  // namespace ptm
