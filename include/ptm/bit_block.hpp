#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ptm {

enum class PortSemantic : std::uint8_t {
    literal_truth,
    ta_action,
    literal_condition,
    clause_output,
};

const char* port_semantic_name(PortSemantic semantic) noexcept;

template <std::size_t Bits, PortSemantic Semantic>
struct alignas(64) TypedBitBlock {
    static_assert(Bits == 1024 || Bits == 4096,
                  "PTM PA blocks are fixed at 32x32 or 64x64 bits");
    static_assert(Bits % 64 == 0);

    static constexpr std::size_t bit_count = Bits;
    static constexpr std::size_t word_count = Bits / 64;
    static constexpr PortSemantic semantic = Semantic;

    std::array<std::uint64_t, word_count> words{};

    [[nodiscard]] bool get(std::size_t index) const {
        if (index >= Bits) {
            throw std::out_of_range("PTM bit index");
        }
        return ((words[index / 64] >> (index % 64)) & 1ULL) != 0;
    }

    void set(std::size_t index, bool value) {
        if (index >= Bits) {
            throw std::out_of_range("PTM bit index");
        }
        const auto mask = std::uint64_t{1} << (index % 64);
        if (value) {
            words[index / 64] |= mask;
        } else {
            words[index / 64] &= ~mask;
        }
    }

    void clear() noexcept { words.fill(0); }

    [[nodiscard]] std::size_t population() const noexcept {
        std::size_t result = 0;
        for (const auto word : words) {
            result += static_cast<std::size_t>(std::popcount(word));
        }
        return result;
    }

    friend bool operator==(const TypedBitBlock&, const TypedBitBlock&) = default;
};

template <PortSemantic Semantic>
using BitBlock32x32 = TypedBitBlock<1024, Semantic>;

template <PortSemantic Semantic>
using BitBlock64x64 = TypedBitBlock<4096, Semantic>;

using LiteralTruth32x32 = BitBlock32x32<PortSemantic::literal_truth>;
using LiteralTruth64x64 = BitBlock64x64<PortSemantic::literal_truth>;
using TAAction32x32 = BitBlock32x32<PortSemantic::ta_action>;
using TAAction64x64 = BitBlock64x64<PortSemantic::ta_action>;
using LiteralCondition32x32 = BitBlock32x32<PortSemantic::literal_condition>;
using LiteralCondition64x64 = BitBlock64x64<PortSemantic::literal_condition>;
using ClauseOutput32x32 = BitBlock32x32<PortSemantic::clause_output>;
using ClauseOutput64x64 = BitBlock64x64<PortSemantic::clause_output>;

static_assert(sizeof(TAAction32x32) == 128);
static_assert(sizeof(TAAction64x64) == 512);
static_assert(alignof(TAAction32x32) == 64);
static_assert(alignof(TAAction64x64) == 64);

}  // namespace ptm

