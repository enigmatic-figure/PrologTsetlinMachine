#include "ptm/shadow_audit.hpp"

#include <limits>
#include <stdexcept>

namespace ptm {

AuditDecision decide_audit(AuditPhase phase,
                           const AuditSnapshot& snapshot,
                           const AuditPolicy& policy) noexcept {
    if (phase == AuditPhase::shadow) {
        if (snapshot.observed < policy.shadow_min_observations) {
            return AuditDecision::insufficient_data;
        }
        return snapshot.mismatch_rate() <= policy.activation_max_mismatch_rate
                   ? AuditDecision::activate
                   : AuditDecision::reject;
    }
    if (snapshot.observed < policy.live_min_observations) {
        return AuditDecision::insufficient_data;
    }
    if (snapshot.mismatches >= policy.reopen_min_mismatches &&
        snapshot.mismatch_rate() >= policy.reopen_mismatch_rate) {
        return AuditDecision::reopen;
    }
    return AuditDecision::healthy;
}

ShadowAuditor::ShadowAuditor(std::size_t window_size)
    : window_size_(window_size),
      observations_(
          std::make_unique<std::atomic<std::uint64_t>[]>(window_size)) {
    if (window_size == 0) {
        throw std::invalid_argument("shadow audit window cannot be empty");
    }
    reset();
}

std::uint64_t ShadowAuditor::encode(std::uint64_t sequence,
                                    bool mismatch) noexcept {
    return ((sequence + 1U) << 1U) | static_cast<std::uint64_t>(mismatch);
}

std::uint64_t ShadowAuditor::decode_sequence(std::uint64_t encoded) noexcept {
    return (encoded >> 1U) - 1U;
}

void ShadowAuditor::record(bool expected, bool actual) noexcept {
    const auto sequence = cursor_.fetch_add(1, std::memory_order_relaxed);
    const auto encoded = encode(sequence, expected != actual);
    auto& destination = observations_[sequence % window_size_];
    auto current = destination.load(std::memory_order_acquire);
    for (;;) {
        if (current != 0 && decode_sequence(current) > sequence) {
            return;
        }
        if (destination.compare_exchange_weak(
                current, encoded, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return;
        }
    }
}

AuditSnapshot ShadowAuditor::snapshot() const noexcept {
    const auto end = cursor_.load(std::memory_order_acquire);
    const auto start = end > window_size_ ? end - window_size_ : 0;
    AuditSnapshot result{end, 0, 0};
    for (std::size_t index = 0; index < window_size_; ++index) {
        const auto encoded = observations_[index].load(std::memory_order_acquire);
        if (encoded == 0) {
            continue;
        }
        const auto sequence = decode_sequence(encoded);
        if (sequence >= start && sequence < end) {
            ++result.observed;
            result.mismatches += static_cast<std::size_t>(encoded & 1U);
        }
    }
    return result;
}

void ShadowAuditor::reset() noexcept {
    cursor_.store(0, std::memory_order_relaxed);
    for (std::size_t index = 0; index < window_size_; ++index) {
        observations_[index].store(0, std::memory_order_relaxed);
    }
}

void ShadowAuditor::restore(const AuditSnapshot& snapshot) {
    constexpr auto maximum_sequence_end =
        (std::numeric_limits<std::uint64_t>::max() >> 1U) - 1U;
    if (snapshot.observed > window_size_ ||
        snapshot.mismatches > snapshot.observed ||
        snapshot.sequence_end < snapshot.observed ||
        snapshot.sequence_end > maximum_sequence_end) {
        throw std::invalid_argument("audit checkpoint is inconsistent");
    }

    reset();
    const auto first = snapshot.sequence_end - snapshot.observed;
    for (std::size_t index = 0; index < snapshot.observed; ++index) {
        const auto sequence = first + index;
        const bool mismatch = index < snapshot.mismatches;
        observations_[sequence % window_size_].store(
            encode(sequence, mismatch), std::memory_order_relaxed);
    }
    cursor_.store(snapshot.sequence_end, std::memory_order_release);
}

}  // namespace ptm
