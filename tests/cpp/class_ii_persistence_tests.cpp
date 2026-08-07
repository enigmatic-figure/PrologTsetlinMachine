#include "ptm/class_ii_persistence.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function>
void require_persistence_error(Function&& function,
                               ptm::PersistenceErrorCode code,
                               std::string_view message) {
    try {
        function();
    } catch (const ptm::PersistenceError& error) {
        require(error.code() == code, message);
        return;
    }
    throw std::runtime_error(std::string(message));
}

struct TemporaryDirectory {
    TemporaryDirectory() {
        const auto stamp = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        path = std::filesystem::temp_directory_path() /
               ("ptm-class-ii-persistence-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    std::filesystem::path path;
};

ptm::MaturityMetrics mature_metrics() {
    return ptm::MaturityMetrics{0.995, 500, 0.001, 0.002, 20, 0.001};
}

ptm::ConsolidationSpec specification() {
    return ptm::ConsolidationSpec{
        "sha256:persistent-class-ii-artifact",
        "mapping:persistence-v1",
        "snapshot:adaptive-substrate-before-absorption",
        1024,
        ptm::PortSemantic::ta_action,
        {
            {4, ptm::SourceKind::ta, 2},
            {7, ptm::SourceKind::ta, 3},
        },
        mature_metrics(),
    };
}

void record_matches(ptm::ConsolidationRegistry& registry,
                    ptm::ArtifactHandle artifact,
                    std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        const bool expected = (index % 2U) == 0;
        require(registry.record_observation(artifact, expected, expected),
                "audit match was rejected");
    }
}

void write_bytes(const std::filesystem::path& path,
                 const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(output), "test could not write fixture bytes");
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    const auto size = std::filesystem::file_size(path);
    std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size()));
    require(static_cast<bool>(input), "test could not read fixture bytes");
    return result;
}

void append_torn_tail(const std::filesystem::path& path) {
    const std::array<std::uint8_t, 12> tail{
        'P', 'T', 'M', '2', 'E', 'V', 'T', '1', 1, 0, 0, 0};
    std::ofstream output(path, std::ios::binary | std::ios::app);
    output.write(reinterpret_cast<const char*>(tail.data()),
                 static_cast<std::streamsize>(tail.size()));
    require(static_cast<bool>(output), "test could not append torn event tail");
}

void test_snapshot_event_replay_and_compaction() {
    TemporaryDirectory temporary;
    const auto snapshot_path = temporary.path / "registry.snapshot";
    const auto log_path = temporary.path / "registry.events";

    ptm::AuditPolicy audit_policy{};
    audit_policy.shadow_min_observations = 2;
    audit_policy.activation_max_mismatch_rate = 0.0;
    audit_policy.live_min_observations = 2;
    audit_policy.reopen_min_mismatches = 1;
    audit_policy.reopen_mismatch_rate = 0.5;
    ptm::ConsolidationRegistry registry(16, 4, 4, {}, audit_policy);
    const auto nomination = registry.nominate(specification());
    require(nomination.status == ptm::RegistryStatus::ok,
            "persistent artifact nomination failed");
    require(registry.mark_validated(nomination.handle) == ptm::RegistryStatus::ok &&
                registry.mark_compiled(nomination.handle) == ptm::RegistryStatus::ok &&
                registry.begin_shadow(nomination.handle) == ptm::RegistryStatus::ok,
            "persistent artifact did not reach shadowing");
    record_matches(registry, nomination.handle, 2);
    require(registry.activate(nomination.handle) == ptm::RegistryStatus::ok,
            "persistent artifact did not activate");

    auto rejected_specification = specification();
    rejected_specification.artifact_id = "sha256:persistent-rejected-artifact";
    rejected_specification.restoration_handle = "snapshot:rejected-source-state";
    rejected_specification.bindings = {{9, ptm::SourceKind::ta, 10}};
    const auto rejected = registry.nominate(std::move(rejected_specification));
    require(rejected.status == ptm::RegistryStatus::ok &&
                registry.mark_validated(rejected.handle) ==
                    ptm::RegistryStatus::ok &&
                registry.reject(rejected.handle) == ptm::RegistryStatus::ok,
            "rejected persistence fixture did not reach its terminal state");

    auto dissolved_specification = specification();
    dissolved_specification.artifact_id = "sha256:persistent-dissolved-artifact";
    dissolved_specification.restoration_handle = "snapshot:dissolved-source-state";
    dissolved_specification.bindings = {{11, ptm::SourceKind::ta, 11}};
    const auto dissolved = registry.nominate(std::move(dissolved_specification));
    require(dissolved.status == ptm::RegistryStatus::ok &&
                registry.dissolve(dissolved.handle) == ptm::RegistryStatus::ok,
            "dissolved persistence fixture did not reach its terminal state");

    const auto origin = ptm::ClassIIPersistence::capture(registry);
    ptm::ClassIIPersistence::write_snapshot_atomic(snapshot_path, origin);
    const auto round_trip = ptm::ClassIIPersistence::read_snapshot(snapshot_path);
    require(round_trip.sequence == 0 &&
                round_trip.registry.artifacts.size() == 3,
            "snapshot envelope did not round-trip");
    auto initial_restore = ptm::ClassIIPersistence::restore(round_trip);
    require(initial_restore->resolve(2).has_value() &&
                initial_restore->resolve(2)->slot == 4,
            "snapshot did not restore active source routing");
    require(initial_restore->state(rejected.handle) ==
                ptm::ConsolidationState::rejected &&
                initial_restore->state(dissolved.handle) ==
                    ptm::ConsolidationState::dissolved,
            "snapshot did not restore terminal artifact states");

    const auto branch_a = temporary.path / "branch-a.events";
    const auto branch_b = temporary.path / "branch-b.events";
    const auto forked_log = temporary.path / "forked.events";
    static_cast<void>(ptm::ClassIIPersistence::append_event(
        branch_a, origin, *initial_restore, "branch-a"));
    static_cast<void>(ptm::ClassIIPersistence::append_event(
        branch_b, origin, *initial_restore, "branch-b"));
    auto forked_bytes = read_bytes(branch_a);
    const auto second_branch = read_bytes(branch_b);
    forked_bytes.insert(forked_bytes.end(), second_branch.begin(),
                        second_branch.end());
    write_bytes(forked_log, forked_bytes);
    require_persistence_error(
        [&] {
            static_cast<void>(ptm::ClassIIPersistence::recover(
                snapshot_path, forked_log));
        },
        ptm::PersistenceErrorCode::sequence_conflict,
        "forked event ancestry was not rejected");

    require(registry.record_observation(nomination.handle, true, false) &&
                registry.record_observation(nomination.handle, false, false),
            "live persistence observations were rejected");
    auto durable = ptm::ClassIIPersistence::append_event(
        log_path, origin, registry, "live-audit-checkpoint");
    const auto audit_replay = ptm::ClassIIPersistence::recover(
        snapshot_path, log_path);
    require(audit_replay.applied_events == 1 &&
                audit_replay.ignored_tail_bytes == 0,
            "live audit event did not replay");
    auto audit_restore = ptm::ClassIIPersistence::restore(audit_replay.image);
    const auto restored_audit = audit_restore->audit_snapshot(nomination.handle);
    require(restored_audit.has_value() && restored_audit->sequence_end == 2 &&
                restored_audit->observed == 2 &&
                restored_audit->mismatches == 1 &&
                audit_restore->audit_decision(nomination.handle) ==
                    ptm::AuditDecision::reopen,
            "live audit window was not restored exactly");

    require(registry.reopen(nomination.handle) == ptm::RegistryStatus::ok,
            "persistent artifact did not reopen");
    durable = ptm::ClassIIPersistence::append_event(
        log_path, durable, registry, "reopen");
    require(registry.begin_shadow(nomination.handle) == ptm::RegistryStatus::ok,
            "persistent artifact did not resume shadowing");
    record_matches(registry, nomination.handle, 2);
    require(registry.activate(nomination.handle) == ptm::RegistryStatus::ok,
            "persistent artifact did not reactivate");
    durable = ptm::ClassIIPersistence::append_event(
        log_path, durable, registry, "reactivate");

    auto replay = ptm::ClassIIPersistence::recover(snapshot_path, log_path);
    require(replay.image.sequence == 3 && replay.applied_events == 3,
            "event log did not reach the latest commit");
    auto recovered = ptm::ClassIIPersistence::restore(replay.image);
    const auto source_2 = recovered->resolve(2);
    const auto source_3 = recovered->resolve(3);
    require(source_2.has_value() && source_3.has_value() &&
                source_2->slot == 4 && source_3->slot == 7 &&
                source_2->generation == 1 && source_3->generation == 1,
            "replay did not preserve routing generations");
    const auto restored_spec = recovered->specification(nomination.handle);
    require(restored_spec.has_value() &&
                restored_spec->restoration_handle ==
                    "snapshot:adaptive-substrate-before-absorption",
            "replay lost the restoration handle");

    append_torn_tail(log_path);
    replay = ptm::ClassIIPersistence::recover(snapshot_path, log_path);
    require(replay.image.sequence == 3 && replay.ignored_tail_bytes == 12,
            "recovery did not isolate a torn event tail");
    durable = ptm::ClassIIPersistence::append_event(
        log_path, replay.image, registry, "post-recovery-checkpoint");
    replay = ptm::ClassIIPersistence::recover(snapshot_path, log_path);
    require(replay.image.sequence == 4 && replay.ignored_tail_bytes == 0,
            "append did not repair the torn event tail");

    const auto corrupt_log = temporary.path / "corrupt.events";
    auto log_bytes = read_bytes(log_path);
    log_bytes[log_bytes.size() / 2U] ^= 0x40U;
    write_bytes(corrupt_log, log_bytes);
    require_persistence_error(
        [&] {
            static_cast<void>(ptm::ClassIIPersistence::recover(
                snapshot_path, corrupt_log));
        },
        ptm::PersistenceErrorCode::corrupt_data,
        "event-log tampering was not rejected");

    const auto corrupt_snapshot = temporary.path / "corrupt.snapshot";
    auto snapshot_bytes = read_bytes(snapshot_path);
    snapshot_bytes.back() ^= 0x01U;
    write_bytes(corrupt_snapshot, snapshot_bytes);
    require_persistence_error(
        [&] {
            static_cast<void>(ptm::ClassIIPersistence::read_snapshot(
                corrupt_snapshot));
        },
        ptm::PersistenceErrorCode::corrupt_data,
        "snapshot tampering was not rejected");

    auto inconsistent = replay.image;
    inconsistent.registry.mapping_words.push_back(
        inconsistent.registry.mapping_words.front());
    require_persistence_error(
        [&] {
            static_cast<void>(ptm::ClassIIPersistence::restore(inconsistent));
        },
        ptm::PersistenceErrorCode::inconsistent_state,
        "inconsistent mapping snapshot was restored");

    ptm::ClassIIPersistence::compact(snapshot_path, log_path, replay.image);
    require(std::filesystem::file_size(log_path) == 0,
            "event-log compaction did not truncate the replay prefix");
    const auto compacted = ptm::ClassIIPersistence::recover(
        snapshot_path, log_path);
    require(compacted.image.sequence == 4 && compacted.applied_events == 0,
            "compacted snapshot did not recover independently");
    const auto compacted_registry =
        ptm::ClassIIPersistence::restore(compacted.image);
    require(compacted_registry->resolve(2).has_value() &&
                compacted_registry->resolve(2)->generation == 1,
            "compaction changed recovered routing state");
}

void test_log_only_origin_recovery() {
    TemporaryDirectory temporary;
    const auto missing_snapshot = temporary.path / "missing.snapshot";
    const auto log_path = temporary.path / "origin.events";
    ptm::AuditPolicy audit_policy{};
    audit_policy.shadow_min_observations = 1;
    audit_policy.live_min_observations = 1;
    audit_policy.reopen_min_mismatches = 1;
    ptm::ConsolidationRegistry registry(8, 2, 2, {}, audit_policy);
    const auto nomination = registry.nominate(specification());
    require(nomination.status == ptm::RegistryStatus::ok,
            "log-only nomination failed");
    const auto origin = ptm::ClassIIPersistence::capture(registry);
    const auto first = ptm::ClassIIPersistence::append_event(
        log_path, origin, registry, "registry-origin");
    const auto replay = ptm::ClassIIPersistence::recover(
        missing_snapshot, log_path);
    require(replay.image.sequence == first.sequence &&
                replay.applied_events == 1,
            "event log could not recover without a snapshot file");
    const auto restored = ptm::ClassIIPersistence::restore(replay.image);
    require(restored->state(nomination.handle) ==
                ptm::ConsolidationState::nominated,
            "log-only recovery restored the wrong lifecycle state");
}

}  // namespace

int main() {
    try {
        test_snapshot_event_replay_and_compaction();
        test_log_only_origin_recovery();
        std::cout << "PTM Class II persistence tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PTM Class II persistence test failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
