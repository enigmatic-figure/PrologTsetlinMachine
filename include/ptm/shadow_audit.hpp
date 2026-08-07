#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ptm {

struct AuditSnapshot {
    std::uint64_t sequence_end{};
    std::size_t observed{};
    std::size_t mismatches{};

    [[nodiscard]] double mismatch_rate() const noexcept {
        return observed == 0
                   ? 0.0
                   : static_cast<double>(mismatches) /
                         static_cast<double>(observed);
    }
};

struct AuditPolicy {
    std::size_t shadow_min_observations{256};
    double activation_max_mismatch_rate{0.001};
    std::size_t live_min_observations{256};
    std::size_t reopen_min_mismatches{4};
    double reopen_mismatch_rate{0.02};
};

enum class AuditPhase : std::uint8_t {
    shadow,
    live,
};

enum class AuditDecision : std::uint8_t {
    insufficient_data,
    activate,
    reject,
    healthy,
    reopen,
};

[[nodiscard]] AuditDecision decide_audit(AuditPhase phase,
                                         const AuditSnapshot& snapshot,
                                         const AuditPolicy& policy) noexcept;

// Recording uses atomic sequence reservation and CAS without a mutex.
// Snapshotting is a cold O(window_size) control-plane action.
class ShadowAuditor {
public:
    explicit ShadowAuditor(std::size_t window_size);

    ShadowAuditor(const ShadowAuditor&) = delete;
    ShadowAuditor& operator=(const ShadowAuditor&) = delete;

    [[nodiscard]] std::size_t window_size() const noexcept { return window_size_; }
    [[nodiscard]] std::uint64_t total_recorded() const noexcept {
        return cursor_.load(std::memory_order_acquire);
    }

    void record(bool expected, bool actual) noexcept;
    [[nodiscard]] AuditSnapshot snapshot() const noexcept;

    // Safe only while the owning artifact is outside shadow/live states.
    void reset() noexcept;

    // Rebuilds an equivalent aggregate window before its owner is published.
    // The order of matches inside the window is intentionally not preserved.
    void restore(const AuditSnapshot& snapshot);

private:
    [[nodiscard]] static std::uint64_t encode(std::uint64_t sequence,
                                              bool mismatch) noexcept;
    [[nodiscard]] static std::uint64_t decode_sequence(
        std::uint64_t encoded) noexcept;

    std::size_t window_size_{};
    std::unique_ptr<std::atomic<std::uint64_t>[]> observations_;
    std::atomic<std::uint64_t> cursor_{0};
};

}  // namespace ptm
