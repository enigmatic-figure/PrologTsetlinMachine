#include "ptm/class_ii_persistence.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto root = argc > 1
                              ? std::filesystem::path(argv[1])
                              : std::filesystem::path("out/class-ii-persistence");
        const auto run = root /
                         std::to_string(
                             std::chrono::high_resolution_clock::now()
                                 .time_since_epoch()
                                 .count());
        const auto snapshot_path = run / "registry.snapshot";
        const auto event_path = run / "registry.events";

        ptm::AuditPolicy audit_policy{};
        audit_policy.shadow_min_observations = 2;
        audit_policy.activation_max_mismatch_rate = 0.0;
        audit_policy.live_min_observations = 2;
        audit_policy.reopen_min_mismatches = 1;
        audit_policy.reopen_mismatch_rate = 0.5;
        ptm::ConsolidationRegistry registry(16, 4, 4, {}, audit_policy);
        const auto nomination = registry.nominate(ptm::ConsolidationSpec{
            "sha256:persistence-example",
            "mapping:persistence-example-v1",
            "snapshot:ta-state-before-example-absorption",
            1024,
            ptm::PortSemantic::ta_action,
            {
                {4, ptm::SourceKind::ta, 2},
                {7, ptm::SourceKind::ta, 3},
            },
            ptm::MaturityMetrics{0.999, 1000, 0.0, 0.0, 50, 0.0},
        });
        require(nomination.status == ptm::RegistryStatus::ok,
                "nomination failed");
        require(registry.mark_validated(nomination.handle) ==
                        ptm::RegistryStatus::ok &&
                    registry.mark_compiled(nomination.handle) ==
                        ptm::RegistryStatus::ok &&
                    registry.begin_shadow(nomination.handle) ==
                        ptm::RegistryStatus::ok,
                "shadow transition failed");
        require(registry.record_observation(nomination.handle, true, true) &&
                    registry.record_observation(nomination.handle, false, false),
                "shadow observations failed");
        require(registry.activate(nomination.handle) == ptm::RegistryStatus::ok,
                "activation failed");

        auto durable = ptm::ClassIIPersistence::capture(registry);
        ptm::ClassIIPersistence::write_snapshot_atomic(snapshot_path, durable);

        require(registry.record_observation(nomination.handle, true, false) &&
                    registry.record_observation(nomination.handle, false, false),
                "live observations failed");
        durable = ptm::ClassIIPersistence::append_event(
            event_path, durable, registry, "live-drift-detected");
        require(registry.audit_decision(nomination.handle) ==
                    ptm::AuditDecision::reopen,
                "live drift did not request reopening");
        require(registry.reopen(nomination.handle) == ptm::RegistryStatus::ok,
                "reopen failed");
        durable = ptm::ClassIIPersistence::append_event(
            event_path, durable, registry, "sources-released");

        const auto replay = ptm::ClassIIPersistence::recover(
            snapshot_path, event_path);
        auto recovered = ptm::ClassIIPersistence::restore(replay.image);
        const auto audit = recovered->audit_snapshot(nomination.handle);
        require(audit.has_value(), "recovered audit is missing");
        require(recovered->state(nomination.handle) ==
                    ptm::ConsolidationState::reopening &&
                    !recovered->resolve(2).has_value(),
                "recovered registry published a reopened artifact");

        ptm::ClassIIPersistence::compact(
            snapshot_path, event_path, replay.image);
        const auto compacted = ptm::ClassIIPersistence::recover(
            snapshot_path, event_path);

        std::cout << "snapshot=" << snapshot_path.string() << '\n'
                  << "event_sequence=" << replay.image.sequence << '\n'
                  << "event_digest="
                  << ptm::persistence_digest_hex(
                         replay.image.last_event_digest)
                  << '\n'
                  << "recovered_state="
                  << ptm::consolidation_state_name(
                         *recovered->state(nomination.handle))
                  << '\n'
                  << "recovered_audit=" << audit->observed << '/'
                  << audit->mismatches << '\n'
                  << "source_2_resolves=false\n"
                  << "compacted_replay_events=" << compacted.applied_events
                  << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Class II persistence example failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
