#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ptm {

using SourceHandle = std::uint32_t;
using ArtifactHandle = std::uint32_t;

class ConsolidationRegistry;

inline constexpr std::uint16_t maximum_pa_slot = 4095;
inline constexpr ArtifactHandle maximum_artifact_handle = (1U << 24U) - 1U;
inline constexpr std::uint32_t maximum_mapping_generation = (1U << 27U) - 1U;

struct MappingEntry {
    bool source_valid{};
    bool bound{};
    ArtifactHandle artifact{};
    std::uint16_t slot{};
    std::uint32_t generation{};
    std::uint64_t encoded{};

    friend bool operator==(const MappingEntry&, const MappingEntry&) = default;
};

// Fixed-capacity, dense source mapping. Hot reads are one bounds check and one
// acquire load. Generation-tagged CAS operations prevent stale release calls
// from clearing a newer binding.
class ConcurrentMappingTable {
public:
    explicit ConcurrentMappingTable(std::size_t source_capacity);

    ConcurrentMappingTable(const ConcurrentMappingTable&) = delete;
    ConcurrentMappingTable& operator=(const ConcurrentMappingTable&) = delete;

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool is_lock_free() const noexcept;
    [[nodiscard]] MappingEntry lookup(SourceHandle source) const noexcept;

    [[nodiscard]] bool try_bind(SourceHandle source,
                                ArtifactHandle artifact,
                                std::uint16_t slot,
                                std::uint32_t expected_generation) noexcept;

    [[nodiscard]] bool try_release(SourceHandle source,
                                   const MappingEntry& expected) noexcept;

    [[nodiscard]] bool try_rebind(SourceHandle source,
                                  const MappingEntry& expected,
                                  ArtifactHandle artifact,
                                  std::uint16_t slot) noexcept;

private:
    friend class ConsolidationRegistry;

    static constexpr std::uint64_t slot_mask = (1ULL << 12U) - 1ULL;
    static constexpr std::uint64_t artifact_mask = (1ULL << 24U) - 1ULL;
    static constexpr std::uint64_t generation_mask = (1ULL << 27U) - 1ULL;
    static constexpr unsigned artifact_shift = 12;
    static constexpr unsigned generation_shift = 36;
    static constexpr std::uint64_t bound_mask = 1ULL << 63U;

    [[nodiscard]] static std::uint64_t encode_bound(ArtifactHandle artifact,
                                                    std::uint16_t slot,
                                                    std::uint32_t generation) noexcept;
    [[nodiscard]] static std::uint64_t encode_unbound(
        std::uint32_t generation) noexcept;
    [[nodiscard]] static MappingEntry decode(bool source_valid,
                                             std::uint64_t encoded) noexcept;
    [[nodiscard]] bool restore_encoded(SourceHandle source,
                                       std::uint64_t encoded) noexcept;

    std::size_t capacity_{};
    std::unique_ptr<std::atomic<std::uint64_t>[]> entries_;
};

}  // namespace ptm
