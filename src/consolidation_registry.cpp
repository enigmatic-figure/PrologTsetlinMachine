#include "ptm/consolidation_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ptm {

namespace {

bool probability(double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

bool valid_semantic(PortSemantic semantic) noexcept {
    switch (semantic) {
        case PortSemantic::literal_truth:
        case PortSemantic::ta_action:
        case PortSemantic::literal_condition:
        case PortSemantic::clause_output:
            return true;
    }
    return false;
}

bool valid_source_kind(SourceKind kind) noexcept {
    switch (kind) {
        case SourceKind::literal:
        case SourceKind::ta:
        case SourceKind::literal_condition:
        case SourceKind::clause:
        case SourceKind::artifact_output:
            return true;
    }
    return false;
}

bool valid_state(ConsolidationState state) noexcept {
    switch (state) {
        case ConsolidationState::nominated:
        case ConsolidationState::validated:
        case ConsolidationState::compiled:
        case ConsolidationState::shadowing:
        case ConsolidationState::activating:
        case ConsolidationState::active:
        case ConsolidationState::reopening:
        case ConsolidationState::dissolved:
        case ConsolidationState::rejected:
            return true;
    }
    return false;
}

}  // namespace

bool is_mature(const MaturityMetrics& metrics,
               const MaturityPolicy& policy) noexcept {
    return probability(metrics.precision) &&
           probability(metrics.recent_state_movement) &&
           probability(metrics.feedback_rate) &&
           probability(metrics.perturbation_sensitivity) &&
           metrics.precision >= policy.minimum_precision &&
           metrics.support >= policy.minimum_support &&
           metrics.recent_state_movement <=
               policy.maximum_recent_state_movement &&
           metrics.feedback_rate <= policy.maximum_feedback_rate &&
           metrics.reuse_count >= policy.minimum_reuse_count &&
           metrics.perturbation_sensitivity <=
               policy.maximum_perturbation_sensitivity;
}

const char* consolidation_state_name(ConsolidationState state) noexcept {
    switch (state) {
        case ConsolidationState::nominated:
            return "nominated";
        case ConsolidationState::validated:
            return "validated";
        case ConsolidationState::compiled:
            return "compiled";
        case ConsolidationState::shadowing:
            return "shadowing";
        case ConsolidationState::activating:
            return "activating";
        case ConsolidationState::active:
            return "active";
        case ConsolidationState::reopening:
            return "reopening";
        case ConsolidationState::dissolved:
            return "dissolved";
        case ConsolidationState::rejected:
            return "rejected";
    }
    return "unknown";
}

struct ConsolidationRegistry::ArtifactRuntime {
    struct AuditGeneration {
        AuditGeneration(ConsolidationState value, std::size_t window_size)
            : phase(value), auditor(window_size) {}

        ConsolidationState phase;
        ShadowAuditor auditor;
    };

    explicit ArtifactRuntime(ConsolidationSpec value,
                             std::size_t audit_window_size)
        : specification(std::move(value)) {
        auditors.push_back(std::make_unique<AuditGeneration>(
            ConsolidationState::nominated, audit_window_size));
        current_auditor.store(auditors.back().get(), std::memory_order_relaxed);
    }

    ConsolidationSpec specification;
    std::atomic<ConsolidationState> state{ConsolidationState::nominated};
    std::vector<std::unique_ptr<AuditGeneration>> auditors;
    std::atomic<AuditGeneration*> current_auditor{nullptr};
    std::vector<std::pair<SourceHandle, MappingEntry>> active_mappings;
};

ConsolidationRegistry::ConsolidationRegistry(
    std::size_t source_capacity,
    std::size_t artifact_capacity,
    std::size_t audit_window_size,
    MaturityPolicy maturity_policy,
    AuditPolicy audit_policy)
    : artifact_capacity_(artifact_capacity),
      audit_window_size_(audit_window_size),
      maturity_policy_(maturity_policy),
      audit_policy_(audit_policy),
      mappings_(source_capacity),
      published_(
          std::make_unique<std::atomic<ArtifactRuntime*>[]>(artifact_capacity)) {
    if (artifact_capacity == 0 ||
        artifact_capacity > static_cast<std::size_t>(maximum_artifact_handle) + 1U) {
        throw std::invalid_argument("artifact capacity is outside mapping range");
    }
    if (audit_window_size == 0) {
        throw std::invalid_argument("audit window cannot be empty");
    }
    if (!probability(maturity_policy_.minimum_precision) ||
        !probability(maturity_policy_.maximum_recent_state_movement) ||
        !probability(maturity_policy_.maximum_feedback_rate) ||
        !probability(maturity_policy_.maximum_perturbation_sensitivity) ||
        !probability(audit_policy_.activation_max_mismatch_rate) ||
        !probability(audit_policy_.reopen_mismatch_rate) ||
        audit_policy_.shadow_min_observations == 0 ||
        audit_policy_.shadow_min_observations > audit_window_size_ ||
        audit_policy_.live_min_observations == 0 ||
        audit_policy_.live_min_observations > audit_window_size_ ||
        audit_policy_.reopen_min_mismatches > audit_window_size_) {
        throw std::invalid_argument("registry policy is inconsistent with its audit window");
    }
    for (std::size_t index = 0; index < artifact_capacity_; ++index) {
        published_[index].store(nullptr, std::memory_order_relaxed);
    }
    owned_.reserve(artifact_capacity_);
}

ConsolidationRegistry::~ConsolidationRegistry() = default;

ConsolidationRegistry::ArtifactRuntime* ConsolidationRegistry::runtime(
    ArtifactHandle artifact) const noexcept {
    if (static_cast<std::size_t>(artifact) >= artifact_capacity_) {
        return nullptr;
    }
    return published_[artifact].load(std::memory_order_acquire);
}

bool ConsolidationRegistry::valid_specification(
    const ConsolidationSpec& specification) const {
    if (specification.artifact_id.empty() ||
        specification.mapping_version.empty() ||
        specification.restoration_handle.empty() ||
        (specification.input_bits != 1024 && specification.input_bits != 4096) ||
        specification.bindings.empty() ||
        !valid_semantic(specification.port_semantic)) {
        return false;
    }
    std::unordered_set<std::uint64_t> sources;
    std::unordered_set<std::uint16_t> slots;
    for (const auto& binding : specification.bindings) {
        if (binding.source_id > std::numeric_limits<SourceHandle>::max() ||
            binding.source_id >= mappings_.capacity() ||
            binding.slot >= specification.input_bits ||
            !valid_source_kind(binding.source_kind) ||
            !sources.insert(binding.source_id).second ||
            !slots.insert(binding.slot).second) {
            return false;
        }
    }
    return true;
}

NominationResult ConsolidationRegistry::nominate(
    ConsolidationSpec specification) {
    std::lock_guard lock(writer_mutex_);
    if (!valid_specification(specification)) {
        return {RegistryStatus::invalid_specification, 0};
    }
    if (!is_mature(specification.maturity, maturity_policy_)) {
        return {RegistryStatus::immature, 0};
    }
    if (owned_.size() >= artifact_capacity_) {
        return {RegistryStatus::capacity_exhausted, 0};
    }
    const auto handle = static_cast<ArtifactHandle>(owned_.size());
    auto artifact =
        std::make_unique<ArtifactRuntime>(std::move(specification),
                                          audit_window_size_);
    auto* published = artifact.get();
    owned_.push_back(std::move(artifact));
    published_[handle].store(published, std::memory_order_release);
    return {RegistryStatus::ok, handle};
}

RegistryStatus ConsolidationRegistry::transition(
    ArtifactHandle artifact,
    ConsolidationState expected,
    ConsolidationState desired) {
    std::lock_guard lock(writer_mutex_);
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return RegistryStatus::invalid_handle;
    }
    if (value->state.load(std::memory_order_acquire) != expected) {
        return RegistryStatus::invalid_state;
    }
    value->state.store(desired, std::memory_order_release);
    return RegistryStatus::ok;
}

RegistryStatus ConsolidationRegistry::mark_validated(ArtifactHandle artifact) {
    return transition(artifact, ConsolidationState::nominated,
                      ConsolidationState::validated);
}

RegistryStatus ConsolidationRegistry::mark_compiled(ArtifactHandle artifact) {
    return transition(artifact, ConsolidationState::validated,
                      ConsolidationState::compiled);
}

void ConsolidationRegistry::rotate_auditor(ArtifactRuntime& artifact,
                                           ConsolidationState phase) {
    artifact.auditors.push_back(std::make_unique<ArtifactRuntime::AuditGeneration>(
        phase, audit_window_size_));
    artifact.current_auditor.store(artifact.auditors.back().get(),
                                   std::memory_order_release);
}

RegistryStatus ConsolidationRegistry::begin_shadow(ArtifactHandle artifact) {
    std::lock_guard lock(writer_mutex_);
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return RegistryStatus::invalid_handle;
    }
    const auto current = value->state.load(std::memory_order_acquire);
    if (current != ConsolidationState::compiled &&
        current != ConsolidationState::reopening) {
        return RegistryStatus::invalid_state;
    }
    if (!value->active_mappings.empty()) {
        return RegistryStatus::mapping_conflict;
    }
    rotate_auditor(*value, ConsolidationState::shadowing);
    value->state.store(ConsolidationState::shadowing,
                       std::memory_order_release);
    return RegistryStatus::ok;
}

bool ConsolidationRegistry::record_observation(ArtifactHandle artifact,
                                               bool expected,
                                               bool actual) noexcept {
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return false;
    }
    const auto before = value->state.load(std::memory_order_acquire);
    if (before != ConsolidationState::shadowing &&
        before != ConsolidationState::active) {
        return false;
    }
    auto* generation = value->current_auditor.load(std::memory_order_acquire);
    if (generation->phase != before ||
        value->state.load(std::memory_order_acquire) != before) {
        return false;
    }
    generation->auditor.record(expected, actual);
    return true;
}

std::optional<AuditSnapshot> ConsolidationRegistry::audit_snapshot(
    ArtifactHandle artifact) const noexcept {
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return std::nullopt;
    }
    auto* generation = value->current_auditor.load(std::memory_order_acquire);
    return generation->auditor.snapshot();
}

std::optional<AuditDecision> ConsolidationRegistry::audit_decision(
    ArtifactHandle artifact) const noexcept {
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return std::nullopt;
    }
    const auto current = value->state.load(std::memory_order_acquire);
    const auto snapshot = audit_snapshot(artifact);
    if (!snapshot.has_value()) {
        return std::nullopt;
    }
    if (current == ConsolidationState::shadowing) {
        return decide_audit(AuditPhase::shadow, *snapshot, audit_policy_);
    }
    if (current == ConsolidationState::active) {
        return decide_audit(AuditPhase::live, *snapshot, audit_policy_);
    }
    return std::nullopt;
}

RegistryStatus ConsolidationRegistry::activate(ArtifactHandle artifact) {
    std::lock_guard lock(writer_mutex_);
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return RegistryStatus::invalid_handle;
    }
    if (value->state.load(std::memory_order_acquire) !=
        ConsolidationState::shadowing) {
        return RegistryStatus::invalid_state;
    }
    const auto snapshot = value->current_auditor.load(
        std::memory_order_acquire)->auditor.snapshot();
    const auto decision = decide_audit(AuditPhase::shadow, snapshot, audit_policy_);
    if (decision == AuditDecision::insufficient_data) {
        return RegistryStatus::audit_not_ready;
    }
    if (decision != AuditDecision::activate) {
        return RegistryStatus::audit_rejected;
    }

    value->state.store(ConsolidationState::activating,
                       std::memory_order_release);
    value->active_mappings.clear();
    for (const auto& binding : value->specification.bindings) {
        const auto source = static_cast<SourceHandle>(binding.source_id);
        const auto current = mappings_.lookup(source);
        if (!current.source_valid || current.bound ||
            !mappings_.try_bind(source, artifact, binding.slot,
                                current.generation)) {
            const auto release_status = release_mappings(*value);
            value->state.store(
                release_status == RegistryStatus::ok
                    ? ConsolidationState::shadowing
                    : ConsolidationState::reopening,
                std::memory_order_release);
            return RegistryStatus::mapping_conflict;
        }
        value->active_mappings.emplace_back(source, mappings_.lookup(source));
    }

    rotate_auditor(*value, ConsolidationState::active);
    value->state.store(ConsolidationState::active, std::memory_order_release);
    return RegistryStatus::ok;
}

RegistryStatus ConsolidationRegistry::replace_active(
    ArtifactHandle active_artifact,
    ArtifactHandle replacement_artifact) {
    std::lock_guard lock(writer_mutex_);
    if (active_artifact == replacement_artifact) {
        return RegistryStatus::invalid_handle;
    }
    auto* active = runtime(active_artifact);
    auto* replacement = runtime(replacement_artifact);
    if (active == nullptr || replacement == nullptr) {
        return RegistryStatus::invalid_handle;
    }
    if (active->state.load(std::memory_order_acquire) !=
            ConsolidationState::active ||
        replacement->state.load(std::memory_order_acquire) !=
            ConsolidationState::shadowing) {
        return RegistryStatus::invalid_state;
    }
    if (!replacement->active_mappings.empty() ||
        active->specification.input_bits != replacement->specification.input_bits ||
        active->specification.port_semantic !=
            replacement->specification.port_semantic ||
        active->specification.bindings.size() !=
            replacement->specification.bindings.size()) {
        return RegistryStatus::invalid_specification;
    }

    using BindingValue = std::pair<SourceKind, std::uint16_t>;
    std::unordered_map<SourceHandle, BindingValue> active_bindings;
    active_bindings.reserve(active->specification.bindings.size());
    for (const auto& binding : active->specification.bindings) {
        active_bindings.emplace(
            static_cast<SourceHandle>(binding.source_id),
            BindingValue{binding.source_kind, binding.slot});
    }
    for (const auto& binding : replacement->specification.bindings) {
        const auto source = static_cast<SourceHandle>(binding.source_id);
        const auto found = active_bindings.find(source);
        if (found == active_bindings.end() ||
            found->second.first != binding.source_kind) {
            return RegistryStatus::invalid_specification;
        }
    }
    const auto capture_mappings = [this](ArtifactRuntime& value,
                                         ArtifactHandle handle) {
        value.active_mappings.clear();
        for (const auto& binding : value.specification.bindings) {
            const auto source = static_cast<SourceHandle>(binding.source_id);
            const auto current = mappings_.lookup(source);
            if (current.bound && current.artifact == handle) {
                value.active_mappings.emplace_back(source, current);
            }
        }
    };

    const auto snapshot = replacement->current_auditor.load(
        std::memory_order_acquire)->auditor.snapshot();
    const auto decision = decide_audit(AuditPhase::shadow, snapshot, audit_policy_);
    if (decision == AuditDecision::insufficient_data) {
        return RegistryStatus::audit_not_ready;
    }
    if (decision != AuditDecision::activate) {
        return RegistryStatus::audit_rejected;
    }

    replacement->state.store(ConsolidationState::activating,
                             std::memory_order_release);
    replacement->active_mappings.clear();
    for (const auto& binding : replacement->specification.bindings) {
        const auto source = static_cast<SourceHandle>(binding.source_id);
        const auto current = mappings_.lookup(source);
        if (!current.source_valid || !current.bound ||
            current.artifact != active_artifact ||
            !mappings_.try_rebind(source, current, replacement_artifact,
                                  binding.slot)) {
            bool rollback_ok = true;
            for (auto moved = replacement->active_mappings.rbegin();
                 moved != replacement->active_mappings.rend(); ++moved) {
                const auto parent_binding = active_bindings.at(moved->first);
                const auto replacement_mapping = mappings_.lookup(moved->first);
                rollback_ok =
                    mappings_.try_rebind(moved->first,
                                         replacement_mapping,
                                         active_artifact,
                                         parent_binding.second) &&
                    rollback_ok;
            }
            replacement->active_mappings.clear();
            capture_mappings(*active, active_artifact);
            capture_mappings(*replacement, replacement_artifact);
            if (!rollback_ok || active->active_mappings.size() !=
                                    active->specification.bindings.size()) {
                active->state.store(ConsolidationState::reopening,
                                    std::memory_order_release);
                replacement->state.store(ConsolidationState::reopening,
                                         std::memory_order_release);
                return RegistryStatus::mapping_conflict;
            }
            replacement->active_mappings.clear();
            replacement->state.store(ConsolidationState::shadowing,
                                     std::memory_order_release);
            return RegistryStatus::mapping_conflict;
        }
        replacement->active_mappings.emplace_back(
            source, mappings_.lookup(source));
    }

    active->active_mappings.clear();
    rotate_auditor(*replacement, ConsolidationState::active);
    replacement->state.store(ConsolidationState::active,
                             std::memory_order_release);
    active->state.store(ConsolidationState::reopening,
                        std::memory_order_release);
    return RegistryStatus::ok;
}

RegistryStatus ConsolidationRegistry::release_mappings(
    ArtifactRuntime& artifact) {
    std::vector<std::pair<SourceHandle, MappingEntry>> remaining;
    for (const auto& [source, mapping] : artifact.active_mappings) {
        if (!mappings_.try_release(source, mapping)) {
            remaining.emplace_back(source, mapping);
        }
    }
    artifact.active_mappings = std::move(remaining);
    return artifact.active_mappings.empty() ? RegistryStatus::ok
                                            : RegistryStatus::mapping_conflict;
}

RegistryStatus ConsolidationRegistry::reopen(ArtifactHandle artifact) {
    std::lock_guard lock(writer_mutex_);
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return RegistryStatus::invalid_handle;
    }
    const auto current = value->state.load(std::memory_order_acquire);
    if (current != ConsolidationState::active &&
        current != ConsolidationState::reopening) {
        return RegistryStatus::invalid_state;
    }
    if (current == ConsolidationState::active) {
        value->state.store(ConsolidationState::reopening,
                           std::memory_order_release);
    }
    return release_mappings(*value);
}

RegistryStatus ConsolidationRegistry::dissolve(ArtifactHandle artifact) {
    std::lock_guard lock(writer_mutex_);
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return RegistryStatus::invalid_handle;
    }
    auto current = value->state.load(std::memory_order_acquire);
    if (current == ConsolidationState::dissolved ||
        current == ConsolidationState::rejected ||
        current == ConsolidationState::activating) {
        return RegistryStatus::invalid_state;
    }
    RegistryStatus release_status = RegistryStatus::ok;
    if (current == ConsolidationState::active) {
        value->state.store(ConsolidationState::reopening,
                           std::memory_order_release);
        current = ConsolidationState::reopening;
    }
    if (current == ConsolidationState::reopening) {
        release_status = release_mappings(*value);
        if (release_status != RegistryStatus::ok) {
            return release_status;
        }
    }
    value->state.store(ConsolidationState::dissolved,
                       std::memory_order_release);
    return release_status;
}

RegistryStatus ConsolidationRegistry::reject(ArtifactHandle artifact) {
    std::lock_guard lock(writer_mutex_);
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return RegistryStatus::invalid_handle;
    }
    const auto current = value->state.load(std::memory_order_acquire);
    if (current == ConsolidationState::active ||
        current == ConsolidationState::activating ||
        current == ConsolidationState::reopening ||
        current == ConsolidationState::dissolved ||
        current == ConsolidationState::rejected) {
        return RegistryStatus::invalid_state;
    }
    value->state.store(ConsolidationState::rejected,
                       std::memory_order_release);
    return RegistryStatus::ok;
}

std::optional<ResolvedConsolidation> ConsolidationRegistry::resolve(
    SourceHandle source) const noexcept {
    const auto mapping = mappings_.lookup(source);
    if (!mapping.source_valid || !mapping.bound) {
        return std::nullopt;
    }
    auto* artifact = runtime(mapping.artifact);
    if (artifact == nullptr ||
        artifact->state.load(std::memory_order_acquire) !=
            ConsolidationState::active) {
        return std::nullopt;
    }
    return ResolvedConsolidation{
        mapping.artifact,
        mapping.slot,
        mapping.generation,
    };
}

std::optional<ConsolidationState> ConsolidationRegistry::state(
    ArtifactHandle artifact) const noexcept {
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return std::nullopt;
    }
    return value->state.load(std::memory_order_acquire);
}

std::optional<ConsolidationSpec> ConsolidationRegistry::specification(
    ArtifactHandle artifact) const {
    auto* value = runtime(artifact);
    if (value == nullptr) {
        return std::nullopt;
    }
    return value->specification;
}

std::size_t ConsolidationRegistry::artifact_count() const {
    std::lock_guard lock(writer_mutex_);
    return owned_.size();
}

ConsolidationRegistrySnapshot ConsolidationRegistry::checkpoint() const {
    std::lock_guard lock(writer_mutex_);
    ConsolidationRegistrySnapshot result{
        consolidation_snapshot_schema_version,
        mappings_.capacity(),
        artifact_capacity_,
        audit_window_size_,
        maturity_policy_,
        audit_policy_,
        {},
        {},
    };
    result.artifacts.reserve(owned_.size());
    for (const auto& artifact : owned_) {
        auto* generation = artifact->current_auditor.load(
            std::memory_order_acquire);
        result.artifacts.push_back(ArtifactCheckpoint{
            artifact->specification,
            artifact->state.load(std::memory_order_acquire),
            generation->phase,
            generation->auditor.snapshot(),
        });
    }
    for (std::size_t source = 0; source < mappings_.capacity(); ++source) {
        const auto mapping = mappings_.lookup(static_cast<SourceHandle>(source));
        if (mapping.encoded != 0) {
            result.mapping_words.emplace_back(
                static_cast<SourceHandle>(source), mapping.encoded);
        }
    }
    return result;
}

std::unique_ptr<ConsolidationRegistry> ConsolidationRegistry::restore(
    const ConsolidationRegistrySnapshot& snapshot) {
    if (snapshot.schema_version != consolidation_snapshot_schema_version) {
        throw std::invalid_argument("unsupported consolidation snapshot version");
    }
    if (snapshot.artifacts.size() > snapshot.artifact_capacity) {
        throw std::invalid_argument("snapshot exceeds its artifact capacity");
    }

    auto registry = std::make_unique<ConsolidationRegistry>(
        snapshot.source_capacity,
        snapshot.artifact_capacity,
        snapshot.audit_window_size,
        snapshot.maturity_policy,
        snapshot.audit_policy);

    for (const auto& checkpoint : snapshot.artifacts) {
        if (!registry->valid_specification(checkpoint.specification) ||
            !is_mature(checkpoint.specification.maturity,
                       snapshot.maturity_policy) ||
            !valid_state(checkpoint.state) ||
            !valid_state(checkpoint.audit_phase) ||
            checkpoint.state == ConsolidationState::activating ||
            (checkpoint.audit_phase != ConsolidationState::nominated &&
             checkpoint.audit_phase != ConsolidationState::shadowing &&
             checkpoint.audit_phase != ConsolidationState::active) ||
            (checkpoint.state == ConsolidationState::shadowing &&
             checkpoint.audit_phase != ConsolidationState::shadowing) ||
            (checkpoint.state == ConsolidationState::active &&
             checkpoint.audit_phase != ConsolidationState::active)) {
            throw std::invalid_argument("artifact checkpoint is inconsistent");
        }
        auto artifact = std::make_unique<ArtifactRuntime>(
            checkpoint.specification, snapshot.audit_window_size);
        auto* generation = artifact->current_auditor.load(
            std::memory_order_relaxed);
        generation->phase = checkpoint.audit_phase;
        generation->auditor.restore(checkpoint.audit);
        artifact->state.store(checkpoint.state, std::memory_order_relaxed);
        auto* published = artifact.get();
        const auto handle = static_cast<ArtifactHandle>(registry->owned_.size());
        registry->owned_.push_back(std::move(artifact));
        registry->published_[handle].store(published, std::memory_order_release);
    }

    SourceHandle prior_source{};
    bool have_prior = false;
    for (const auto& [source, encoded] : snapshot.mapping_words) {
        if (encoded == 0) {
            throw std::invalid_argument("snapshot contains a redundant zero mapping word");
        }
        if (static_cast<std::size_t>(source) >= snapshot.source_capacity) {
            throw std::invalid_argument(
                "snapshot mapping source " + std::to_string(source) +
                " exceeds capacity " +
                std::to_string(snapshot.source_capacity));
        }
        if (have_prior && source <= prior_source) {
            throw std::invalid_argument("snapshot mapping sources are not strictly ordered");
        }
        if (!registry->mappings_.restore_encoded(source, encoded)) {
            throw std::invalid_argument("snapshot mapping word could not be restored");
        }
        prior_source = source;
        have_prior = true;
        const auto mapping = registry->mappings_.lookup(source);
        if (!mapping.bound) {
            if (mapping.artifact != 0 || mapping.slot != 0) {
                throw std::invalid_argument(
                    "unbound snapshot mapping contains artifact data");
            }
            continue;
        }
        auto* artifact = registry->runtime(mapping.artifact);
        if (artifact == nullptr) {
            throw std::invalid_argument("snapshot maps to an unknown artifact");
        }
        const auto state = artifact->state.load(std::memory_order_relaxed);
        if (state != ConsolidationState::active &&
            state != ConsolidationState::reopening) {
            throw std::invalid_argument("snapshot maps to an unpublished artifact");
        }
        const auto binding = std::find_if(
            artifact->specification.bindings.begin(),
            artifact->specification.bindings.end(),
            [source, &mapping](const SlotBinding& candidate) {
                return candidate.source_id == source &&
                       candidate.slot == mapping.slot;
            });
        if (binding == artifact->specification.bindings.end()) {
            throw std::invalid_argument("snapshot mapping violates its artifact specification");
        }
        artifact->active_mappings.emplace_back(source, mapping);
    }

    for (std::size_t handle = 0; handle < registry->owned_.size(); ++handle) {
        const auto& artifact = *registry->owned_[handle];
        const auto state = artifact.state.load(std::memory_order_relaxed);
        if (state == ConsolidationState::active &&
            artifact.active_mappings.size() !=
                artifact.specification.bindings.size()) {
            throw std::invalid_argument("active snapshot artifact is only partially mapped");
        }
    }
    return registry;
}

}  // namespace ptm
