#include "ptm/concurrent_mapping.hpp"

#include <stdexcept>

namespace ptm {

ConcurrentMappingTable::ConcurrentMappingTable(std::size_t source_capacity)
    : capacity_(source_capacity),
      entries_(std::make_unique<std::atomic<std::uint64_t>[]>(source_capacity)) {
    if (source_capacity == 0 || source_capacity > (1ULL << 32U)) {
        throw std::invalid_argument("mapping source capacity is outside uint32 range");
    }
    for (std::size_t index = 0; index < capacity_; ++index) {
        entries_[index].store(0, std::memory_order_relaxed);
    }
}

bool ConcurrentMappingTable::is_lock_free() const noexcept {
    return entries_[0].is_lock_free();
}

std::uint64_t ConcurrentMappingTable::encode_bound(
    ArtifactHandle artifact,
    std::uint16_t slot,
    std::uint32_t generation) noexcept {
    return bound_mask |
           ((static_cast<std::uint64_t>(generation) & generation_mask)
            << generation_shift) |
           ((static_cast<std::uint64_t>(artifact) & artifact_mask)
            << artifact_shift) |
           (static_cast<std::uint64_t>(slot) & slot_mask);
}

std::uint64_t ConcurrentMappingTable::encode_unbound(
    std::uint32_t generation) noexcept {
    return (static_cast<std::uint64_t>(generation) & generation_mask)
           << generation_shift;
}

MappingEntry ConcurrentMappingTable::decode(bool source_valid,
                                            std::uint64_t encoded) noexcept {
    return MappingEntry{
        source_valid,
        (encoded & bound_mask) != 0,
        static_cast<ArtifactHandle>((encoded >> artifact_shift) & artifact_mask),
        static_cast<std::uint16_t>(encoded & slot_mask),
        static_cast<std::uint32_t>((encoded >> generation_shift) & generation_mask),
        encoded,
    };
}

MappingEntry ConcurrentMappingTable::lookup(SourceHandle source) const noexcept {
    if (static_cast<std::size_t>(source) >= capacity_) {
        return MappingEntry{};
    }
    return decode(true, entries_[source].load(std::memory_order_acquire));
}

bool ConcurrentMappingTable::restore_encoded(SourceHandle source,
                                              std::uint64_t encoded) noexcept {
    if (static_cast<std::size_t>(source) >= capacity_) {
        return false;
    }
    auto expected = std::uint64_t{0};
    return entries_[source].compare_exchange_strong(
        expected, encoded, std::memory_order_release, std::memory_order_relaxed);
}

bool ConcurrentMappingTable::try_bind(SourceHandle source,
                                      ArtifactHandle artifact,
                                      std::uint16_t slot,
                                      std::uint32_t expected_generation) noexcept {
    if (static_cast<std::size_t>(source) >= capacity_ ||
        artifact > maximum_artifact_handle || slot > maximum_pa_slot ||
        expected_generation > maximum_mapping_generation) {
        return false;
    }
    auto expected = encode_unbound(expected_generation);
    const auto desired = encode_bound(artifact, slot, expected_generation);
    return entries_[source].compare_exchange_strong(
        expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
}

bool ConcurrentMappingTable::try_release(SourceHandle source,
                                         const MappingEntry& expected) noexcept {
    if (static_cast<std::size_t>(source) >= capacity_ || !expected.source_valid ||
        !expected.bound || expected.artifact > maximum_artifact_handle ||
        expected.slot > maximum_pa_slot ||
        expected.generation > maximum_mapping_generation) {
        return false;
    }
    const auto decoded_expected = decode(true, expected.encoded);
    if (!decoded_expected.bound || decoded_expected.artifact != expected.artifact ||
        decoded_expected.slot != expected.slot ||
        decoded_expected.generation != expected.generation) {
        return false;
    }
    auto expected_word = expected.encoded;
    const auto next_generation =
        (expected.generation + 1U) & maximum_mapping_generation;
    const auto desired = encode_unbound(next_generation);
    return entries_[source].compare_exchange_strong(
        expected_word, desired, std::memory_order_acq_rel,
        std::memory_order_acquire);
}

bool ConcurrentMappingTable::try_rebind(SourceHandle source,
                                        const MappingEntry& expected,
                                        ArtifactHandle artifact,
                                        std::uint16_t slot) noexcept {
    if (static_cast<std::size_t>(source) >= capacity_ || !expected.source_valid ||
        !expected.bound || expected.artifact > maximum_artifact_handle ||
        expected.slot > maximum_pa_slot || artifact > maximum_artifact_handle ||
        slot > maximum_pa_slot ||
        expected.generation > maximum_mapping_generation) {
        return false;
    }
    const auto decoded_expected = decode(true, expected.encoded);
    if (!decoded_expected.bound || decoded_expected.artifact != expected.artifact ||
        decoded_expected.slot != expected.slot ||
        decoded_expected.generation != expected.generation) {
        return false;
    }
    auto expected_word = expected.encoded;
    const auto next_generation =
        (expected.generation + 1U) & maximum_mapping_generation;
    const auto desired = encode_bound(artifact, slot, next_generation);
    return entries_[source].compare_exchange_strong(
        expected_word, desired, std::memory_order_acq_rel,
        std::memory_order_acquire);
}

}  // namespace ptm
