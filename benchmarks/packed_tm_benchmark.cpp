#include "ptm/packed_tm.hpp"

#if defined(PTM_HAS_CUDA)
#include "ptm/packed_tm_cuda.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view schema = "ptm.runtime-benchmark.v1";

enum class BenchmarkBackend : std::uint8_t {
    automatic,
    scalar,
    avx2,
    avx512,
    cuda_sparse,
    cuda_warp_tile,
    cuda_dense_bitset,
    cuda_sparse_fused_vote,
    cuda_warp_tile_fused_vote,
    cuda_dense_bitset_fused_vote,
};

[[nodiscard]] const char* backend_name(BenchmarkBackend backend) noexcept {
    switch (backend) {
        case BenchmarkBackend::automatic: return "automatic";
        case BenchmarkBackend::scalar: return "scalar";
        case BenchmarkBackend::avx2: return "avx2";
        case BenchmarkBackend::avx512: return "avx512";
        case BenchmarkBackend::cuda_sparse: return "cuda_sparse";
        case BenchmarkBackend::cuda_warp_tile: return "cuda_warp_tile";
        case BenchmarkBackend::cuda_dense_bitset:
            return "cuda_dense_bitset";
        case BenchmarkBackend::cuda_sparse_fused_vote:
            return "cuda_sparse_fused_vote";
        case BenchmarkBackend::cuda_warp_tile_fused_vote:
            return "cuda_warp_tile_fused_vote";
        case BenchmarkBackend::cuda_dense_bitset_fused_vote:
            return "cuda_dense_bitset_fused_vote";
    }
    return "unknown";
}

[[nodiscard]] bool is_cuda_backend(BenchmarkBackend backend) noexcept {
    return backend == BenchmarkBackend::cuda_sparse ||
           backend == BenchmarkBackend::cuda_warp_tile ||
           backend == BenchmarkBackend::cuda_dense_bitset ||
           backend == BenchmarkBackend::cuda_sparse_fused_vote ||
           backend == BenchmarkBackend::cuda_warp_tile_fused_vote ||
           backend == BenchmarkBackend::cuda_dense_bitset_fused_vote;
}

[[nodiscard]] ptm::PackedTMBackend cpu_backend(
    BenchmarkBackend backend) {
    switch (backend) {
        case BenchmarkBackend::automatic:
            return ptm::PackedTMBackend::automatic;
        case BenchmarkBackend::scalar:
            return ptm::PackedTMBackend::scalar;
        case BenchmarkBackend::avx2:
            return ptm::PackedTMBackend::avx2;
        case BenchmarkBackend::avx512:
            return ptm::PackedTMBackend::avx512;
        case BenchmarkBackend::cuda_sparse:
        case BenchmarkBackend::cuda_warp_tile:
        case BenchmarkBackend::cuda_dense_bitset:
        case BenchmarkBackend::cuda_sparse_fused_vote:
        case BenchmarkBackend::cuda_warp_tile_fused_vote:
        case BenchmarkBackend::cuda_dense_bitset_fused_vote:
            break;
    }
    throw std::logic_error("CUDA backend has no CPU backend value");
}

#if defined(PTM_HAS_CUDA)
struct CudaBenchmarkBackend {
    ptm::CudaPackedTMBackend clauses{};
    ptm::CudaPackedTMVoteBackend votes{};
};

[[nodiscard]] CudaBenchmarkBackend cuda_backend(
    BenchmarkBackend backend) {
    switch (backend) {
        case BenchmarkBackend::cuda_sparse:
            return {ptm::CudaPackedTMBackend::sparse,
                    ptm::CudaPackedTMVoteBackend::two_stage};
        case BenchmarkBackend::cuda_warp_tile:
            return {ptm::CudaPackedTMBackend::warp_tile,
                    ptm::CudaPackedTMVoteBackend::two_stage};
        case BenchmarkBackend::cuda_dense_bitset:
            return {ptm::CudaPackedTMBackend::dense_bitset,
                    ptm::CudaPackedTMVoteBackend::two_stage};
        case BenchmarkBackend::cuda_sparse_fused_vote:
            return {ptm::CudaPackedTMBackend::sparse,
                    ptm::CudaPackedTMVoteBackend::fused_atomic};
        case BenchmarkBackend::cuda_warp_tile_fused_vote:
            return {ptm::CudaPackedTMBackend::warp_tile,
                    ptm::CudaPackedTMVoteBackend::fused_atomic};
        case BenchmarkBackend::cuda_dense_bitset_fused_vote:
            return {ptm::CudaPackedTMBackend::dense_bitset,
                    ptm::CudaPackedTMVoteBackend::fused_atomic};
        default:
            throw std::logic_error("CPU backend has no CUDA backend value");
    }
}
#endif

[[nodiscard]] std::vector<BenchmarkBackend> all_backends() {
    return {
        BenchmarkBackend::automatic,
        BenchmarkBackend::scalar,
        BenchmarkBackend::avx2,
        BenchmarkBackend::avx512,
        BenchmarkBackend::cuda_sparse,
        BenchmarkBackend::cuda_warp_tile,
        BenchmarkBackend::cuda_dense_bitset,
        BenchmarkBackend::cuda_sparse_fused_vote,
        BenchmarkBackend::cuda_warp_tile_fused_vote,
        BenchmarkBackend::cuda_dense_bitset_fused_vote,
    };
}

struct Options {
    std::vector<std::size_t> clause_counts{20};
    std::vector<std::size_t> feature_counts{256};
    std::vector<double> densities{0.02};
    std::vector<std::size_t> resident_page_counts{1};
    std::vector<BenchmarkBackend> backends{all_backends()};
    std::size_t repeats{5000};
    std::size_t warmup_repeats{100};
    std::size_t samples{5};
    std::uint64_t seed{20260806};
    bool json_lines{};
    bool repeats_was_set{};
};

struct Timing {
    double elapsed_seconds{};
    double examples_per_second{};
    std::uint64_t checksum{};
};

struct TimingSummary {
    double median_elapsed_seconds{};
    double median_examples_per_second{};
    double mad_examples_per_second{};
    std::uint64_t checksum{};
    std::size_t samples{};
};

[[nodiscard]] std::size_t checked_product(std::size_t left,
                                          std::size_t right,
                                          const char* message) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::invalid_argument(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t parse_positive(std::string_view value,
                                         std::string_view name) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    std::size_t consumed = 0;
    const auto parsed = std::stoull(std::string(value), &consumed);
    if (consumed != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(parsed);
}

[[nodiscard]] std::uint64_t parse_u64(std::string_view value,
                                      std::string_view name) {
    try {
        if (value.empty() || value.front() == '-') {
            throw std::invalid_argument("negative");
        }
        std::size_t consumed = 0;
        const auto parsed = std::stoull(std::string(value), &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be uint64");
    }
}

[[nodiscard]] double parse_density(std::string_view value) {
    std::size_t consumed = 0;
    const auto parsed = std::stod(std::string(value), &consumed);
    if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0 ||
        parsed > 1.0) {
        throw std::invalid_argument("density must lie between zero and one");
    }
    return parsed;
}

template <typename Value, typename Parser>
[[nodiscard]] std::vector<Value> parse_list(std::string_view text,
                                            Parser&& parser) {
    std::vector<Value> result;
    if (text.empty()) {
        throw std::invalid_argument("comma-separated option cannot be empty");
    }
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto token = text.substr(0, comma);
        if (token.empty()) {
            throw std::invalid_argument(
                "comma-separated option contains an empty value");
        }
        result.push_back(parser(token));
        if (comma == std::string_view::npos) {
            break;
        }
        text.remove_prefix(comma + 1U);
    }
    return result;
}

[[nodiscard]] BenchmarkBackend parse_backend(std::string_view value) {
    if (value == "auto" || value == "automatic") {
        return BenchmarkBackend::automatic;
    }
    if (value == "scalar") {
        return BenchmarkBackend::scalar;
    }
    if (value == "avx2") {
        return BenchmarkBackend::avx2;
    }
    if (value == "avx512" || value == "avx-512") {
        return BenchmarkBackend::avx512;
    }
    if (value == "cuda_sparse" || value == "cuda-sparse") {
        return BenchmarkBackend::cuda_sparse;
    }
    if (value == "cuda_warp_tile" || value == "cuda-warp-tile" ||
        value == "cuda_warp") {
        return BenchmarkBackend::cuda_warp_tile;
    }
    if (value == "cuda_dense_bitset" || value == "cuda-dense-bitset" ||
        value == "cuda_dense") {
        return BenchmarkBackend::cuda_dense_bitset;
    }
    if (value == "cuda_sparse_fused_vote" ||
        value == "cuda-sparse-fused-vote" ||
        value == "cuda_sparse_fused") {
        return BenchmarkBackend::cuda_sparse_fused_vote;
    }
    if (value == "cuda_warp_tile_fused_vote" ||
        value == "cuda-warp-tile-fused-vote" ||
        value == "cuda_warp_fused") {
        return BenchmarkBackend::cuda_warp_tile_fused_vote;
    }
    if (value == "cuda_dense_bitset_fused_vote" ||
        value == "cuda-dense-bitset-fused-vote" ||
        value == "cuda_dense_fused") {
        return BenchmarkBackend::cuda_dense_bitset_fused_vote;
    }
    throw std::invalid_argument("unknown backend: " + std::string(value));
}

[[nodiscard]] std::vector<BenchmarkBackend> parse_backends(
    std::string_view value) {
    if (value == "all") {
        return all_backends();
    }
    return parse_list<BenchmarkBackend>(value, parse_backend);
}

[[nodiscard]] std::string_view require_value(int argc,
                                             char** argv,
                                             int& index) {
    if (++index >= argc) {
        throw std::invalid_argument(
            std::string("missing value after ") + argv[index - 1]);
    }
    return argv[index];
}

void print_help() {
    std::cout
        << "PTM packed runtime benchmark\n\n"
        << "Usage: ptm_packed_tm_benchmark [clauses features repeats]\n"
        << "       ptm_packed_tm_benchmark [options]\n\n"
        << "  --sweep                 legacy CPU-oriented shape sweep\n"
        << "  --gpu-sweep             CUDA handoff matrix (large)\n"
        << "  --clauses LIST          comma-separated clause counts\n"
        << "  --features LIST         comma-separated feature counts\n"
        << "  --densities LIST        comma-separated Include densities [0,1]\n"
        << "  --resident-pages LIST   feature-major 64-example page counts\n"
        << "  --backend LIST|all      auto,scalar,avx2,avx512,cuda_sparse,"
           "cuda_warp_tile,cuda_dense_bitset and *_fused_vote variants\n"
        << "  --repeats N             timed launches per sample\n"
        << "  --warmup N              warm-up launches\n"
        << "  --samples N             timing samples used for median/MAD\n"
        << "  --seed N                deterministic model/input seed\n"
        << "  --jsonl                 versioned JSON Lines output\n";
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options result;
    if (argc > 1 && !std::string_view(argv[1]).starts_with('-')) {
        if (argc > 4) {
            throw std::invalid_argument(
                "positional form accepts only clauses, features, repeats");
        }
        result.clause_counts = {parse_positive(argv[1], "clauses")};
        if (argc > 2) {
            result.feature_counts = {parse_positive(argv[2], "features")};
        }
        if (argc > 3) {
            result.repeats = parse_positive(argv[3], "repeats");
            result.repeats_was_set = true;
        }
        return result;
    }

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            print_help();
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--jsonl") {
            result.json_lines = true;
        } else if (argument == "--sweep") {
            result.clause_counts = {4, 20, 100};
            result.feature_counts = {64, 256, 1024};
            result.densities = {0.005, 0.02, 0.10};
            result.resident_page_counts = {1};
            if (!result.repeats_was_set) {
                result.repeats = 1000;
            }
        } else if (argument == "--gpu-sweep") {
            result.clause_counts = {32, 64, 256, 1024};
            result.feature_counts = {64, 256, 1024, 4096};
            result.densities = {0.005, 0.02, 0.10, 0.50};
            result.resident_page_counts = {1, 16, 256, 4096};
            if (!result.repeats_was_set) {
                result.repeats = 25;
            }
        } else if (argument == "--clauses") {
            result.clause_counts = parse_list<std::size_t>(
                require_value(argc, argv, index), [](auto value) {
                    return parse_positive(value, "clauses");
                });
        } else if (argument == "--features") {
            result.feature_counts = parse_list<std::size_t>(
                require_value(argc, argv, index), [](auto value) {
                    return parse_positive(value, "features");
                });
        } else if (argument == "--densities") {
            result.densities = parse_list<double>(
                require_value(argc, argv, index), parse_density);
        } else if (argument == "--resident-pages") {
            result.resident_page_counts = parse_list<std::size_t>(
                require_value(argc, argv, index), [](auto value) {
                    return parse_positive(value, "resident pages");
                });
        } else if (argument == "--backend") {
            result.backends = parse_backends(require_value(argc, argv, index));
        } else if (argument == "--repeats") {
            result.repeats = parse_positive(
                require_value(argc, argv, index), "repeats");
            result.repeats_was_set = true;
        } else if (argument == "--warmup") {
            result.warmup_repeats = parse_positive(
                require_value(argc, argv, index), "warmup");
        } else if (argument == "--samples") {
            result.samples = parse_positive(
                require_value(argc, argv, index), "samples");
        } else if (argument == "--seed") {
            result.seed = parse_u64(require_value(argc, argv, index), "seed");
        } else {
            throw std::invalid_argument(
                "unknown option: " + std::string(argument));
        }
    }
    return result;
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 8U);
    for (const auto character : value) {
        switch (character) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += character; break;
        }
    }
    return result;
}

[[nodiscard]] bool same_result(const ptm::PackedTMResult64& left,
                               const ptm::PackedTMResult64& right) {
    return left.valid_example_mask == right.valid_example_mask &&
           left.prediction_mask == right.prediction_mask &&
           left.scores == right.scores &&
           left.clause_outputs == right.clause_outputs &&
           left.feedback_clause_outputs == right.feedback_clause_outputs;
}

template <typename Function>
[[nodiscard]] Timing time_examples(std::size_t evaluated_batches,
                                   Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    const auto checksum = function();
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return Timing{
        elapsed,
        static_cast<double>(evaluated_batches) * 64.0 / elapsed,
        checksum,
    };
}

[[nodiscard]] double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2U;
    return (values.size() & 1U) != 0
               ? values[middle]
               : (values[middle - 1U] + values[middle]) / 2.0;
}

[[nodiscard]] TimingSummary summarize(const std::vector<Timing>& timings) {
    if (timings.empty()) {
        throw std::logic_error("cannot summarize zero timings");
    }
    std::vector<double> elapsed;
    std::vector<double> rates;
    elapsed.reserve(timings.size());
    rates.reserve(timings.size());
    const auto checksum = timings.front().checksum;
    for (const auto& timing : timings) {
        if (timing.checksum != checksum) {
            throw std::runtime_error("timing samples produced different checksums");
        }
        elapsed.push_back(timing.elapsed_seconds);
        rates.push_back(timing.examples_per_second);
    }
    const auto rate_median = median(rates);
    std::vector<double> deviations;
    deviations.reserve(rates.size());
    for (const auto rate : rates) {
        deviations.push_back(std::abs(rate - rate_median));
    }
    return TimingSummary{
        median(elapsed), rate_median, median(deviations), checksum,
        timings.size()};
}

[[nodiscard]] std::uint64_t workload_seed(const Options& options,
                                          std::size_t clauses,
                                          std::size_t features,
                                          double density) {
    const auto density_key = static_cast<std::uint64_t>(
        std::llround(density * 1'000'000.0));
    return options.seed ^ (clauses * 0x9e3779b97f4a7c15ULL) ^
           (features * 0xbf58476d1ce4e5b9ULL) ^ density_key;
}

struct Workload {
    ptm::PackedTMModel64 model;
    std::vector<std::uint64_t> feature_words;
    std::vector<std::uint64_t> valid_masks;
    std::size_t included_literals{};
};

[[nodiscard]] Workload make_workload(const Options& options,
                                     std::size_t clauses,
                                     std::size_t features,
                                     double density,
                                     std::size_t resident_pages) {
    constexpr std::uint16_t states_per_action = 100;
    constexpr int threshold = 15;
    std::mt19937_64 random(workload_seed(options, clauses, features, density));
    std::bernoulli_distribution include(density);
    if (features > std::numeric_limits<std::size_t>::max() / clauses / 2U) {
        throw std::invalid_argument("benchmark TM dimensions overflow size_t");
    }
    std::vector<std::uint16_t> states(clauses * features * 2U);
    std::size_t included = 0;
    for (auto& state : states) {
        const bool selected = include(random);
        state = selected ? states_per_action + 1U : states_per_action;
        included += selected;
    }
    std::vector<std::uint64_t> feature_words(
        checked_product(resident_pages, features,
                        "benchmark feature pages overflow size_t"));
    for (auto& word : feature_words) {
        word = random();
    }
    return Workload{
        ptm::PackedTMModel64(
            clauses, features, states_per_action, threshold, states),
        std::move(feature_words),
        std::vector<std::uint64_t>(resident_pages, ~std::uint64_t{0}),
        included,
    };
}

void emit_capabilities(const Options& options) {
    const auto& cpu = ptm::cpu_capabilities();
#if defined(PTM_HAS_CUDA)
    const auto gpu = ptm::cuda_device_capabilities();
#endif
    if (options.json_lines) {
        std::cout << "{\"schema\":\"" << schema
                  << "\",\"event\":\"capabilities\",\"cpu_brand\":\""
                  << json_escape(cpu.brand)
                  << "\",\"hardware\":{\"x86\":" << std::boolalpha << cpu.x86
                  << ",\"os_xsave\":" << cpu.os_xsave
                  << ",\"avx\":" << cpu.avx
                  << ",\"avx2\":" << cpu.avx2
                  << ",\"avx512f\":" << cpu.avx512f
                  << "},\"compiled\":{\"scalar\":true,\"avx2\":"
                  << cpu.compiled_avx2 << ",\"avx512\":"
                  << cpu.compiled_avx512
#if defined(PTM_HAS_CUDA)
                  << ",\"cuda\":true},\"gpu\":{\"available\":"
                  << gpu.available << ",\"device_ordinal\":"
                  << gpu.device_ordinal << ",\"name\":\""
                  << json_escape(gpu.name) << "\",\"compute_capability\":\""
                  << gpu.compute_capability_major << '.'
                  << gpu.compute_capability_minor << "\",\"total_vram_bytes\":"
                  << gpu.total_global_memory << ",\"multiprocessors\":"
                  << gpu.multiprocessor_count << ",\"warp_size\":"
                  << gpu.warp_size << ",\"driver_version\":"
                  << gpu.driver_version << ",\"runtime_version\":"
                  << gpu.runtime_version
                  << ",\"supported_backends\":[\"cuda_sparse\","
                     "\"cuda_warp_tile\",\"cuda_dense_bitset\","
                     "\"cuda_sparse_fused_vote\","
                     "\"cuda_warp_tile_fused_vote\","
                     "\"cuda_dense_bitset_fused_vote\"],\"status\":\""
                  << json_escape(gpu.status) << "\"}}"
#else
                  << ",\"cuda\":false},\"gpu\":{\"available\":false,"
                     "\"supported_backends\":[],\"status\":\"not_compiled\"}}"
#endif
                  << std::endl;
    } else {
        std::cout << "cpu=\"" << cpu.brand << "\" avx2="
                  << (cpu.avx2 && cpu.compiled_avx2 ? "available" : "unavailable")
                  << " avx512="
                  << (cpu.avx512f && cpu.compiled_avx512 ? "available"
                                                        : "unavailable");
#if defined(PTM_HAS_CUDA)
        std::cout << " gpu=\"" << gpu.name << "\" cuda="
                  << (gpu.available ? "available" : "unavailable")
                  << " compute=" << gpu.compute_capability_major << '.'
                  << gpu.compute_capability_minor;
#else
        std::cout << " cuda=not_compiled";
#endif
        std::cout << '\n';
    }
}

void emit_skip(const Options& options,
               std::size_t clauses,
               std::size_t features,
               double target_density,
               std::size_t resident_pages,
               BenchmarkBackend backend) {
    if (options.json_lines) {
        std::cout << "{\"schema\":\"" << schema
                  << "\",\"event\":\"skip\",\"benchmark\":"
                     "\"packed_tm_inference\",\"backend_requested\":\""
                  << backend_name(backend) << "\",\"clauses\":" << clauses
                  << ",\"features\":" << features
                  << ",\"resident_pages\":" << resident_pages
                  << ",\"include_density_target\":" << target_density
                  << ",\"seed\":\"" << options.seed
                  << "\",\"reason\":\"unavailable\"}" << std::endl;
    } else {
        std::cout << "backend=" << backend_name(backend)
                  << " clauses=" << clauses << " features=" << features
                  << " resident_pages=" << resident_pages
                  << " density=" << target_density
                  << " skipped=unavailable\n";
    }
}

void emit_measurement(const Options& options,
                      std::size_t clauses,
                      std::size_t features,
                      double target_density,
                      double actual_density,
                      std::size_t resident_pages,
                      BenchmarkBackend requested,
                      std::string_view selected,
                      std::string_view timing_scope,
                      std::string_view timing_source,
                      std::size_t input_upload_bytes,
                      std::size_t result_download_bytes,
                      const TimingSummary& timing) {
    const auto batches_per_second = timing.median_examples_per_second / 64.0;
    const auto timed_batches = checked_product(
        checked_product(options.repeats, resident_pages,
                        "timed batch count overflows size_t"),
        std::size_t{1}, "timed batch count overflows size_t");
    if (options.json_lines) {
        std::cout << std::setprecision(12)
                  << "{\"schema\":\"" << schema
                  << "\",\"event\":\"measurement\",\"benchmark\":"
                     "\"packed_tm_inference\",\"backend_requested\":\""
                  << backend_name(requested)
                  << "\",\"backend_selected\":\"" << selected
                  << "\",\"input_layout\":\"feature_major_packed64\""
                  << ",\"correctness_gate\":\"pass\""
                  << ",\"timing_scope\":\"" << timing_scope
                  << "\",\"timing_source\":\"" << timing_source << "\""
                  << ",\"clauses\":" << clauses
                  << ",\"features\":" << features
                  << ",\"batch_width\":64"
                  << ",\"resident_pages\":" << resident_pages
                  << ",\"include_density_target\":" << target_density
                  << ",\"include_density_actual\":" << actual_density
                  << ",\"seed\":\"" << options.seed << "\""
                  << ",\"workload_seed\":\""
                  << workload_seed(options, clauses, features, target_density)
                  << "\",\"warmup_batches\":"
                  << options.warmup_repeats * resident_pages
                  << ",\"launch_repeats\":" << options.repeats
                  << ",\"timed_batches\":" << timed_batches
                  << ",\"samples\":" << timing.samples
                  << ",\"elapsed_seconds\":"
                  << timing.median_elapsed_seconds
                  << ",\"batches_per_second\":" << batches_per_second
                  << ",\"examples_per_second\":"
                  << timing.median_examples_per_second
                  << ",\"mad_examples_per_second\":"
                  << timing.mad_examples_per_second
                  << ",\"input_upload_bytes\":" << input_upload_bytes
                  << ",\"result_download_bytes\":" << result_download_bytes
                  << ",\"cuda_error_status\":\""
                  << (is_cuda_backend(requested) ? "ok" : "not_applicable")
                  << "\""
                  << ",\"checksum\":\"" << timing.checksum << "\"}"
                  << std::endl;
    } else {
        std::cout << std::fixed << std::setprecision(2)
                  << "backend_requested=" << backend_name(requested)
                  << " backend_selected=" << selected
                  << " timing_scope=" << timing_scope
                  << " correctness_gate=pass clauses=" << clauses
                  << " features=" << features
                  << " resident_pages=" << resident_pages
                  << " include_density=" << actual_density
                  << " examples_per_second="
                  << timing.median_examples_per_second
                  << " mad=" << timing.mad_examples_per_second
                  << " samples=" << timing.samples
                  << " checksum=" << timing.checksum << '\n';
    }
}

[[nodiscard]] std::vector<ptm::PackedTMResult64> scalar_oracle(
    const Workload& workload) {
    std::vector<ptm::PackedTMResult64> result;
    result.reserve(workload.valid_masks.size());
    const auto features = workload.model.number_of_features();
    for (std::size_t page = 0; page < workload.valid_masks.size(); ++page) {
        result.push_back(workload.model.evaluate(
            std::span<const std::uint64_t>(
                workload.feature_words.data() + page * features, features),
            workload.valid_masks[page], ptm::PackedTMBackend::scalar));
    }
    return result;
}

[[nodiscard]] std::uint64_t checksum_page(
    const ptm::PackedTMResult64& result,
    std::size_t key,
    int threshold) {
    return result.prediction_mask ^
           result.clause_outputs[key % result.clause_outputs.size()] ^
           static_cast<std::uint64_t>(
               result.scores[key % result.scores.size()] + threshold);
}

#if defined(PTM_HAS_CUDA)
[[nodiscard]] std::uint64_t checksum_cuda_result(
    const ptm::CudaPackedTMResult64& result,
    int threshold) {
    std::uint64_t checksum = 0;
    for (std::size_t page = 0; page < result.page_count; ++page) {
        checksum += checksum_page(result.page(page), page, threshold);
    }
    return checksum;
}
#endif

std::size_t benchmark_cpu_backend(
    const Options& options,
    Workload& workload,
    const std::vector<ptm::PackedTMResult64>& oracle,
    std::size_t clauses,
    std::size_t features,
    double density,
    double actual_density,
    BenchmarkBackend requested) {
    const auto native_backend = cpu_backend(requested);
    if (requested != BenchmarkBackend::automatic &&
        !ptm::packed_tm_backend_available(native_backend)) {
        emit_skip(options, clauses, features, density,
                  workload.valid_masks.size(), requested);
        return 0;
    }
    const auto selected = requested == BenchmarkBackend::automatic
                              ? workload.model.selected_backend()
                              : native_backend;
    const auto selected_name = ptm::packed_tm_backend_name(selected);
    std::vector<std::uint64_t> clause_outputs(clauses);
    std::vector<std::uint64_t> feedback_outputs(clauses);
    std::array<std::int32_t, 64> scores{};
    std::uint64_t predictions = 0;
    for (std::size_t page = 0; page < workload.valid_masks.size(); ++page) {
        const auto first = page * features;
        const auto candidate = workload.model.evaluate(
            std::span<const std::uint64_t>(
                workload.feature_words.data() + first, features),
            workload.valid_masks[page], native_backend);
        if (!same_result(oracle[page], candidate)) {
            throw std::runtime_error(
                std::string("correctness gate failed for ") +
                backend_name(requested));
        }
    }
    for (std::size_t repeat = 0; repeat < options.warmup_repeats; ++repeat) {
        for (std::size_t page = 0; page < workload.valid_masks.size(); ++page) {
            workload.model.evaluate_into(
                std::span<const std::uint64_t>(
                    workload.feature_words.data() + page * features, features),
                workload.valid_masks[page], clause_outputs, feedback_outputs,
                scores, predictions, native_backend);
        }
    }
    const auto evaluated_batches = checked_product(
        options.repeats, workload.valid_masks.size(),
        "CPU benchmark batch count overflows size_t");
    std::vector<Timing> timings;
    timings.reserve(options.samples);
    for (std::size_t sample = 0; sample < options.samples; ++sample) {
        timings.push_back(time_examples(evaluated_batches, [&] {
            std::uint64_t checksum = 0;
            for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
                for (std::size_t page = 0; page < workload.valid_masks.size();
                     ++page) {
                    workload.model.evaluate_into(
                        std::span<const std::uint64_t>(
                            workload.feature_words.data() + page * features,
                            features),
                        workload.valid_masks[page], clause_outputs,
                        feedback_outputs, scores, predictions, native_backend);
                    checksum += predictions ^
                                clause_outputs[page % clauses] ^
                                static_cast<std::uint64_t>(
                                    scores[page % scores.size()] +
                                    workload.model.threshold());
                }
            }
            return checksum;
        }));
    }
    emit_measurement(options, clauses, features, density, actual_density,
                     workload.valid_masks.size(), requested, selected_name,
                     "host_end_to_end", "host_monotonic", 0, 0,
                     summarize(timings));
    return 1;
}

#if defined(PTM_HAS_CUDA)
std::size_t benchmark_cuda_backend(
    const Options& options,
    Workload& workload,
    const std::vector<ptm::PackedTMResult64>& oracle,
    std::size_t clauses,
    std::size_t features,
    double density,
    double actual_density,
    BenchmarkBackend requested) {
    const auto device = ptm::cuda_device_capabilities();
    if (!device.available) {
        emit_skip(options, clauses, features, density,
                  workload.valid_masks.size(), requested);
        return 0;
    }
    const auto native_backend = cuda_backend(requested);
    ptm::CudaPackedTMExecutor64 executor(workload.model);
    ptm::CudaPackedTMTiming transfer_timing{};
    executor.upload_pages(
        workload.feature_words, workload.valid_masks, &transfer_timing);
    executor.evaluate_resident(
        native_backend.clauses, 1, &transfer_timing, native_backend.votes);
    auto candidate = executor.download(&transfer_timing);
    for (std::size_t page = 0; page < oracle.size(); ++page) {
        if (!same_result(oracle[page], candidate.page(page))) {
            throw std::runtime_error(
                std::string("correctness gate failed for ") +
                backend_name(requested));
        }
    }
    executor.evaluate_resident(
        native_backend.clauses, options.warmup_repeats, nullptr,
        native_backend.votes);

    const auto pages = workload.valid_masks.size();
    const auto evaluated_batches = checked_product(
        options.repeats, pages,
        "CUDA benchmark batch count overflows size_t");
    const auto input_bytes = checked_product(
        workload.feature_words.size() + workload.valid_masks.size(),
        sizeof(std::uint64_t), "CUDA input byte count overflows size_t");
    const auto output_bytes_per_page =
        sizeof(std::uint64_t) + 64U * sizeof(std::int32_t) +
        2U * clauses * sizeof(std::uint64_t);
    const auto output_bytes = checked_product(
        pages, output_bytes_per_page,
        "CUDA output byte count overflows size_t");

    std::vector<Timing> kernel_timings;
    kernel_timings.reserve(options.samples);
    for (std::size_t sample = 0; sample < options.samples; ++sample) {
        ptm::CudaPackedTMTiming timing{};
        executor.evaluate_resident(
            native_backend.clauses, options.repeats, &timing,
            native_backend.votes);
        const auto result = executor.download();
        const auto seconds = static_cast<double>(timing.kernel_ms) / 1000.0;
        kernel_timings.push_back(Timing{
            seconds,
            static_cast<double>(evaluated_batches) * 64.0 / seconds,
            checksum_cuda_result(result, workload.model.threshold()) *
                options.repeats,
        });
    }
    emit_measurement(options, clauses, features, density, actual_density,
                     pages, requested, backend_name(requested), "kernel_only",
                     "cuda_event", 0, 0, summarize(kernel_timings));

    std::vector<Timing> resident_timings;
    resident_timings.reserve(options.samples);
    for (std::size_t sample = 0; sample < options.samples; ++sample) {
        resident_timings.push_back(time_examples(evaluated_batches, [&] {
            std::uint64_t checksum = 0;
            for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
                executor.evaluate_resident(
                    native_backend.clauses, 1, nullptr, native_backend.votes);
                checksum += checksum_cuda_result(
                    executor.download(), workload.model.threshold());
            }
            return checksum;
        }));
    }
    emit_measurement(
        options, clauses, features, density, actual_density, pages, requested,
        backend_name(requested), "resident_device_end_to_end",
        "host_monotonic", 0,
        checked_product(output_bytes, options.repeats,
                        "CUDA resident output bytes overflow size_t"),
        summarize(resident_timings));

    std::vector<Timing> cold_timings;
    cold_timings.reserve(options.samples);
    for (std::size_t sample = 0; sample < options.samples; ++sample) {
        cold_timings.push_back(time_examples(evaluated_batches, [&] {
            std::uint64_t checksum = 0;
            for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
                checksum += checksum_cuda_result(
                    executor.evaluate(workload.feature_words,
                                      workload.valid_masks,
                                      native_backend.clauses, nullptr,
                                      native_backend.votes),
                    workload.model.threshold());
            }
            return checksum;
        }));
    }
    emit_measurement(
        options, clauses, features, density, actual_density, pages, requested,
        backend_name(requested), "cold_host_end_to_end", "host_monotonic",
        checked_product(input_bytes, options.repeats,
                        "CUDA cold input bytes overflow size_t"),
        checked_product(output_bytes, options.repeats,
                        "CUDA cold output bytes overflow size_t"),
        summarize(cold_timings));
    return 3;
}
#endif

std::size_t benchmark_workload(const Options& options,
                               std::size_t clauses,
                               std::size_t features,
                               double density,
                               std::size_t resident_pages) {
    auto workload = make_workload(
        options, clauses, features, density, resident_pages);
    const auto oracle = scalar_oracle(workload);
    const auto possible_literals = static_cast<double>(
        workload.model.number_of_clauses() *
        workload.model.number_of_features() * 2U);
    const auto actual_density =
        static_cast<double>(workload.included_literals) / possible_literals;
    std::size_t measurement_count = 0;
    for (const auto requested : options.backends) {
        if (is_cuda_backend(requested)) {
#if defined(PTM_HAS_CUDA)
            measurement_count += benchmark_cuda_backend(
                options, workload, oracle, clauses, features, density,
                actual_density, requested);
#else
            emit_skip(options, clauses, features, density, resident_pages,
                      requested);
#endif
        } else {
            measurement_count += benchmark_cpu_backend(
                options, workload, oracle, clauses, features, density,
                actual_density, requested);
        }
    }
    return measurement_count;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        emit_capabilities(options);
        std::size_t measurement_count = 0;
        for (const auto clauses : options.clause_counts) {
            for (const auto features : options.feature_counts) {
                for (const auto density : options.densities) {
                    for (const auto resident_pages :
                         options.resident_page_counts) {
                        measurement_count += benchmark_workload(
                            options, clauses, features, density,
                            resident_pages);
                    }
                }
            }
        }
        if (options.json_lines) {
            std::cout << "{\"schema\":\"" << schema
                      << "\",\"event\":\"run_end\",\"measurements\":"
                      << measurement_count << "}" << std::endl;
        } else {
            std::cout << "measurements=" << measurement_count << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PTM packed TM benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
