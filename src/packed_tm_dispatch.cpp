#include "ptm/packed_tm.hpp"

#include "packed_tm_backend.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define PTM_X86_HOST 1
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace ptm {

namespace {

#if defined(PTM_X86_HOST)
struct CpuRegisters {
    std::uint32_t eax{};
    std::uint32_t ebx{};
    std::uint32_t ecx{};
    std::uint32_t edx{};
};

[[nodiscard]] CpuRegisters cpuid(std::uint32_t leaf,
                                 std::uint32_t subleaf = 0) noexcept {
#if defined(_MSC_VER)
    std::array<int, 4> values{};
    __cpuidex(values.data(), static_cast<int>(leaf), static_cast<int>(subleaf));
    return CpuRegisters{
        static_cast<std::uint32_t>(values[0]),
        static_cast<std::uint32_t>(values[1]),
        static_cast<std::uint32_t>(values[2]),
        static_cast<std::uint32_t>(values[3]),
    };
#else
    CpuRegisters result{};
    __cpuid_count(leaf, subleaf,
                  result.eax, result.ebx, result.ecx, result.edx);
    return result;
#endif
}

[[nodiscard]] std::uint64_t xgetbv0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    std::uint32_t eax{};
    std::uint32_t edx{};
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32U) | eax;
#endif
}
#endif

[[nodiscard]] CpuCapabilities detect_cpu_capabilities() {
    CpuCapabilities result{};
#if defined(PTM_COMPILED_AVX2)
    result.compiled_avx2 = true;
#endif
#if defined(PTM_COMPILED_AVX512)
    result.compiled_avx512 = true;
#endif
#if defined(PTM_X86_HOST)
    result.x86 = true;
    const auto maximum_leaf = cpuid(0).eax;
    if (maximum_leaf >= 1) {
        const auto leaf1 = cpuid(1);
        result.os_xsave = (leaf1.ecx & (1U << 27U)) != 0;
        const bool hardware_avx = (leaf1.ecx & (1U << 28U)) != 0;
        const auto xcr0 = result.os_xsave ? xgetbv0() : 0;
        result.avx = hardware_avx && (xcr0 & 0x6U) == 0x6U;
        if (maximum_leaf >= 7) {
            const auto leaf7 = cpuid(7, 0);
            result.avx2 = result.avx && (leaf7.ebx & (1U << 5U)) != 0;
            const bool avx512_state = (xcr0 & 0xe0U) == 0xe0U;
            result.avx512f = result.avx && avx512_state &&
                             (leaf7.ebx & (1U << 16U)) != 0;
        }
    }
    const auto maximum_extended = cpuid(0x80000000U).eax;
    if (maximum_extended >= 0x80000004U) {
        std::array<char, 49> brand{};
        for (std::uint32_t leaf = 0; leaf < 3; ++leaf) {
            const auto values = cpuid(0x80000002U + leaf);
            const std::array<std::uint32_t, 4> words{
                values.eax, values.ebx, values.ecx, values.edx};
            std::memcpy(brand.data() + leaf * 16U,
                        words.data(), 16U);
        }
        result.brand = brand.data();
        const auto first = result.brand.find_first_not_of(' ');
        const auto last = result.brand.find_last_not_of(' ');
        result.brand = first == std::string::npos
                           ? std::string{}
                           : result.brand.substr(first, last - first + 1U);
    }
#endif
    if (result.brand.empty()) {
        result.brand = "unknown";
    }
    return result;
}

}  // namespace

const CpuCapabilities& cpu_capabilities() noexcept {
    static const auto capabilities = detect_cpu_capabilities();
    return capabilities;
}

const char* packed_tm_backend_name(PackedTMBackend backend) noexcept {
    switch (backend) {
        case PackedTMBackend::automatic:
            return "automatic";
        case PackedTMBackend::scalar:
            return "scalar";
        case PackedTMBackend::avx2:
            return "avx2";
        case PackedTMBackend::avx512:
            return "avx512";
    }
    return "unknown";
}

bool packed_tm_backend_available(PackedTMBackend backend) noexcept {
    const auto& capabilities = cpu_capabilities();
    switch (backend) {
        case PackedTMBackend::automatic:
        case PackedTMBackend::scalar:
            return true;
        case PackedTMBackend::avx2:
            return capabilities.compiled_avx2 && capabilities.avx2;
        case PackedTMBackend::avx512:
            return capabilities.compiled_avx512 && capabilities.avx512f;
    }
    return false;
}

namespace detail {

void evaluate_scalar_backend(
    PackedTMPlanView plan,
    std::span<const std::uint64_t> feature_words,
    std::uint64_t valid_example_mask,
    std::span<std::uint64_t> clause_outputs,
    std::span<std::uint64_t> feedback_clause_outputs,
    std::span<std::int32_t, 64> scores) noexcept {
    std::fill(scores.begin(), scores.end(), 0);
    for (std::size_t clause = 0; clause < plan.clause_count(); ++clause) {
        auto output = valid_example_mask;
        const auto first = plan.clause_literal_offsets[clause];
        const auto last = plan.clause_literal_offsets[clause + 1U];
        for (auto literal = first; literal < last; ++literal) {
            const auto feature = plan.literal_features[literal];
            const auto truth = plan.literal_negated[literal] != 0
                                   ? ~feature_words[feature]
                                   : feature_words[feature];
            output &= truth;
        }
        output &= valid_example_mask;
        feedback_clause_outputs[clause] = output;
        clause_outputs[clause] = first == last ? 0 : output;
        const auto polarity = (clause % 2U) == 0 ? 1 : -1;
        auto lanes = clause_outputs[clause];
        while (lanes != 0) {
            const auto lane = static_cast<std::size_t>(std::countr_zero(lanes));
            scores[lane] += polarity;
            lanes &= lanes - 1U;
        }
    }
}

}  // namespace detail

}  // namespace ptm
