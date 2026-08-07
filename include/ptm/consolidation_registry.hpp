#pragma once

#include "ptm/concurrent_mapping.hpp"
#include "ptm/pa_instance.hpp"
#include "ptm/shadow_audit.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ptm {

struct MaturityMetrics {
    double precision{};
    std::uint64_t support{};
    double recent_state_movement{};
    double feedback_rate{};
    std::uint64_t reuse_count{};
    double perturbation_sensitivity{};
};

struct MaturityPolicy {
    double minimum_precision{0.98};
    std::uint64_t minimum_support{128};
    double maximum_recent_state_movement{0.01};
    double maximum_feedback_rate{0.02};
    std::uint64_t minimum_reuse_count{4};
    double maximum_perturbation_sensitivity{0.01};
};

[[nodiscard]] bool is_mature(const MaturityMetrics& metrics,
                             const MaturityPolicy& policy) noexcept;

enum class ConsolidationState : std::uint8_t {
    nominated,
    validated,
    compiled,
    shadowing,
    activating,
    active,
    reopening,
    dissolved,
    rejected,
};

[[nodiscard]] const char* consolidation_state_name(
    ConsolidationState state) noexcept;

struct ConsolidationSpec {
    std::string artifact_id;
    std::string mapping_version;
    std::string restoration_handle;
    std::size_t input_bits{};
    PortSemantic port_semantic{PortSemantic::ta_action};
    std::vector<SlotBinding> bindings;
    MaturityMetrics maturity;
};

enum class RegistryStatus : std::uint8_t {
    ok,
    invalid_handle,
    invalid_state,
    invalid_specification,
    immature,
    capacity_exhausted,
    audit_not_ready,
    audit_rejected,
    mapping_conflict,
};

struct NominationResult {
    RegistryStatus status{RegistryStatus::invalid_specification};
    ArtifactHandle handle{};
};

struct ResolvedConsolidation {
    ArtifactHandle artifact{};
    std::uint16_t slot{};
    std::uint32_t generation{};
};

inline constexpr std::uint32_t consolidation_snapshot_schema_version = 1;

struct ArtifactCheckpoint {
    ConsolidationSpec specification;
    ConsolidationState state{ConsolidationState::nominated};
    ConsolidationState audit_phase{ConsolidationState::nominated};
    AuditSnapshot audit;
};

struct ConsolidationRegistrySnapshot {
    std::uint32_t schema_version{consolidation_snapshot_schema_version};
    std::size_t source_capacity{};
    std::size_t artifact_capacity{};
    std::size_t audit_window_size{};
    MaturityPolicy maturity_policy;
    AuditPolicy audit_policy;
    std::vector<ArtifactCheckpoint> artifacts;
    // Sparse raw mapping words. Zero means an unbound source at generation 0.
    std::vector<std::pair<SourceHandle, std::uint64_t>> mapping_words;
};

// Registry mutations are serialized on the cold control path. resolve() and
// record_observation() use only atomics and stable pointers on the hot path.
class ConsolidationRegistry {
public:
    ConsolidationRegistry(std::size_t source_capacity,
                          std::size_t artifact_capacity,
                          std::size_t audit_window_size,
                          MaturityPolicy maturity_policy = {},
                          AuditPolicy audit_policy = {});
    ~ConsolidationRegistry();

    ConsolidationRegistry(const ConsolidationRegistry&) = delete;
    ConsolidationRegistry& operator=(const ConsolidationRegistry&) = delete;

    [[nodiscard]] NominationResult nominate(ConsolidationSpec specification);
    [[nodiscard]] RegistryStatus mark_validated(ArtifactHandle artifact);
    [[nodiscard]] RegistryStatus mark_compiled(ArtifactHandle artifact);
    [[nodiscard]] RegistryStatus begin_shadow(ArtifactHandle artifact);
    [[nodiscard]] RegistryStatus activate(ArtifactHandle artifact);
    [[nodiscard]] RegistryStatus replace_active(
        ArtifactHandle active_artifact,
        ArtifactHandle replacement_artifact);
    [[nodiscard]] RegistryStatus reopen(ArtifactHandle artifact);
    [[nodiscard]] RegistryStatus dissolve(ArtifactHandle artifact);
    [[nodiscard]] RegistryStatus reject(ArtifactHandle artifact);

    [[nodiscard]] bool record_observation(ArtifactHandle artifact,
                                          bool expected,
                                          bool actual) noexcept;
    [[nodiscard]] std::optional<AuditSnapshot> audit_snapshot(
        ArtifactHandle artifact) const noexcept;
    [[nodiscard]] std::optional<AuditDecision> audit_decision(
        ArtifactHandle artifact) const noexcept;

    [[nodiscard]] std::optional<ResolvedConsolidation> resolve(
        SourceHandle source) const noexcept;
    [[nodiscard]] std::optional<ConsolidationState> state(
        ArtifactHandle artifact) const noexcept;
    [[nodiscard]] std::optional<ConsolidationSpec> specification(
        ArtifactHandle artifact) const;
    [[nodiscard]] std::size_t artifact_count() const;

    // A cold, immutable control-plane checkpoint. Concurrent audit writers may
    // conservatively reduce the captured observation count, exactly as with a
    // normal audit snapshot; source routing and lifecycle state are serialized.
    [[nodiscard]] ConsolidationRegistrySnapshot checkpoint() const;
    [[nodiscard]] static std::unique_ptr<ConsolidationRegistry> restore(
        const ConsolidationRegistrySnapshot& snapshot);

    [[nodiscard]] const ConcurrentMappingTable& mappings() const noexcept {
        return mappings_;
    }

private:
    struct ArtifactRuntime;

    [[nodiscard]] ArtifactRuntime* runtime(ArtifactHandle artifact) const noexcept;
    [[nodiscard]] RegistryStatus transition(ArtifactHandle artifact,
                                            ConsolidationState expected,
                                            ConsolidationState desired);
    [[nodiscard]] bool valid_specification(
        const ConsolidationSpec& specification) const;
    void rotate_auditor(ArtifactRuntime& artifact,
                        ConsolidationState phase);
    [[nodiscard]] RegistryStatus release_mappings(ArtifactRuntime& artifact);

    std::size_t artifact_capacity_{};
    std::size_t audit_window_size_{};
    MaturityPolicy maturity_policy_{};
    AuditPolicy audit_policy_{};
    ConcurrentMappingTable mappings_;
    std::unique_ptr<std::atomic<ArtifactRuntime*>[]> published_;
    std::vector<std::unique_ptr<ArtifactRuntime>> owned_;
    mutable std::mutex writer_mutex_;
};

}  // namespace ptm
