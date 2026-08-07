#pragma once

#include "ptm/bit_block.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ptm {

template <std::size_t Bits, PortSemantic Semantic>
struct ThresholdResult {
    bool value{};
    std::uint8_t reserved[3]{};
    std::uint32_t matched_count{};
    std::uint32_t selected_count{};
    std::uint8_t alignment_padding[52]{};
    TypedBitBlock<Bits, Semantic> matched{};
    TypedBitBlock<Bits, Semantic> missing{};
};

template <std::size_t Bits, PortSemantic Semantic>
class MaskedThresholdKernel {
public:
    using Block = TypedBitBlock<Bits, Semantic>;
    using Result = ThresholdResult<Bits, Semantic>;

    MaskedThresholdKernel(const Block& selection, std::uint32_t minimum_true)
        : selection_(selection),
          minimum_true_(minimum_true),
          selected_count_(static_cast<std::uint32_t>(selection.population())) {
        if (minimum_true > selected_count_) {
            throw std::invalid_argument(
                "minimum_true exceeds the selected PA slot count");
        }
    }

    [[nodiscard]] Result evaluate(const Block& input) const noexcept {
        Result result{};
        result.selected_count = selected_count_;
        for (std::size_t word = 0; word < Block::word_count; ++word) {
            result.matched.words[word] =
                input.words[word] & selection_.words[word];
            result.missing.words[word] =
                (~input.words[word]) & selection_.words[word];
            result.matched_count += static_cast<std::uint32_t>(
                std::popcount(result.matched.words[word]));
        }
        result.value = result.matched_count >= minimum_true_;
        return result;
    }

    [[nodiscard]] const Block& selection() const noexcept { return selection_; }
    [[nodiscard]] std::uint32_t minimum_true() const noexcept {
        return minimum_true_;
    }

private:
    Block selection_{};
    std::uint32_t minimum_true_{};
    std::uint32_t selected_count_{};
};

}  // namespace ptm
