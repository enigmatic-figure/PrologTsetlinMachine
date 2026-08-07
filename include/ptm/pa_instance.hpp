#pragma once

#include "ptm/bit_block.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ptm {

enum class SourceKind : std::uint8_t {
    literal,
    ta,
    literal_condition,
    clause,
    artifact_output,
};

struct SlotBinding {
    std::uint16_t slot{};
    SourceKind source_kind{};
    std::uint64_t source_id{};
};

template <std::size_t Bits, PortSemantic Semantic>
class PAInstance {
public:
    using Block = TypedBitBlock<Bits, Semantic>;

    PAInstance(std::string mapping_version,
               std::string restoration_handle,
               std::vector<SlotBinding> bindings)
        : mapping_version_(std::move(mapping_version)),
          restoration_handle_(std::move(restoration_handle)),
          bindings_(std::move(bindings)) {
        if (mapping_version_.empty()) {
            throw std::invalid_argument("mapping version cannot be empty");
        }
        if (restoration_handle_.empty()) {
            throw std::invalid_argument("restoration handle cannot be empty");
        }
        std::sort(bindings_.begin(), bindings_.end(),
                  [](const SlotBinding& left, const SlotBinding& right) {
                      return left.slot < right.slot;
                  });
        for (std::size_t index = 0; index < bindings_.size(); ++index) {
            if (bindings_[index].slot >= Bits) {
                throw std::invalid_argument("slot binding exceeds PA shape");
            }
            if (index > 0 && bindings_[index - 1].slot == bindings_[index].slot) {
                throw std::invalid_argument("PA slot has more than one binding");
            }
        }
    }

    [[nodiscard]] const std::string& mapping_version() const noexcept {
        return mapping_version_;
    }
    [[nodiscard]] const std::string& restoration_handle() const noexcept {
        return restoration_handle_;
    }
    [[nodiscard]] const std::vector<SlotBinding>& bindings() const noexcept {
        return bindings_;
    }
    [[nodiscard]] const Block& input() const noexcept { return input_; }
    [[nodiscard]] Block& input() noexcept { return input_; }

    void clear() noexcept { input_.clear(); }

    void write_slot(std::size_t slot, bool value) { input_.set(slot, value); }

    [[nodiscard]] std::size_t write_source(SourceKind kind,
                                           std::uint64_t source_id,
                                           bool value) {
        std::size_t written = 0;
        for (const auto& binding : bindings_) {
            if (binding.source_kind == kind && binding.source_id == source_id) {
                input_.set(binding.slot, value);
                ++written;
            }
        }
        return written;
    }

private:
    Block input_{};
    std::string mapping_version_;
    std::string restoration_handle_;
    std::vector<SlotBinding> bindings_;
};

}  // namespace ptm
