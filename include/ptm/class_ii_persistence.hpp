#pragma once

#include "ptm/consolidation_registry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ptm {

inline constexpr std::uint32_t class_ii_persistence_schema_version = 1;
using PersistenceDigest = std::array<std::uint8_t, 32>;

enum class PersistenceErrorCode : std::uint8_t {
    io_error,
    corrupt_data,
    unsupported_version,
    inconsistent_state,
    sequence_conflict,
};

class PersistenceError : public std::runtime_error {
public:
    PersistenceError(PersistenceErrorCode code, std::string message)
        : std::runtime_error(std::move(message)), code_(code) {}

    [[nodiscard]] PersistenceErrorCode code() const noexcept { return code_; }

private:
    PersistenceErrorCode code_;
};

struct DurableRegistryImage {
    std::uint32_t schema_version{class_ii_persistence_schema_version};
    std::uint64_t sequence{};
    PersistenceDigest last_event_digest{};
    ConsolidationRegistrySnapshot registry;
};

struct RegistryReplayResult {
    DurableRegistryImage image;
    std::size_t applied_events{};
    std::size_t ignored_tail_bytes{};
    std::size_t valid_log_bytes{};
};

// Persistence is deliberately a cold control-plane service. Each event is a
// complete immutable post-transaction image. This makes replay deterministic,
// bounds recovery work, and keeps persistence out of resolve()/audit hot paths.
class ClassIIPersistence {
public:
    [[nodiscard]] static DurableRegistryImage capture(
        const ConsolidationRegistry& registry);

    static void write_snapshot_atomic(
        const std::filesystem::path& path,
        const DurableRegistryImage& image);
    [[nodiscard]] static DurableRegistryImage read_snapshot(
        const std::filesystem::path& path);

    // The returned image is the durable commit token. A caller must not report
    // the associated registry mutation committed until this method succeeds.
    [[nodiscard]] static DurableRegistryImage append_event(
        const std::filesystem::path& event_log,
        const DurableRegistryImage& prior,
        const ConsolidationRegistry& registry,
        std::string_view event_name);

    [[nodiscard]] static RegistryReplayResult recover(
        const std::filesystem::path& snapshot,
        const std::filesystem::path& event_log);

    // Checkpoints the recovered image and atomically replaces the replay log
    // with an empty file. Recovery also tolerates a crash between those steps.
    static void compact(const std::filesystem::path& snapshot,
                        const std::filesystem::path& event_log,
                        const DurableRegistryImage& image);

    [[nodiscard]] static std::unique_ptr<ConsolidationRegistry> restore(
        const DurableRegistryImage& image);
};

[[nodiscard]] std::string persistence_digest_hex(
    const PersistenceDigest& digest);

}  // namespace ptm
