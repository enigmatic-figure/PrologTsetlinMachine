#include "ptm/concurrent_mapping.hpp"
#include "ptm/consolidation_registry.hpp"
#include "ptm/disjoint_set.hpp"
#include "ptm/logic_program.hpp"
#include "ptm/shadow_audit.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void test_candidate_union_find() {
    ptm::DisjointSet clusters(8);
    require(clusters.unite(1, 2), "first candidate union failed");
    require(clusters.unite(2, 3), "second candidate union failed");
    require(!clusters.unite(1, 3), "redundant candidate union must be detected");
    require(clusters.connected(1, 3), "candidate cluster connectivity failed");
    require(!clusters.connected(1, 4), "unrelated candidates were joined");
    require(clusters.component_size(2) == 3, "candidate cluster size is wrong");
}

void test_generation_tagged_mapping() {
    ptm::ConcurrentMappingTable mappings(8);
    const auto initial = mappings.lookup(2);
    require(initial.source_valid && !initial.bound && initial.generation == 0,
            "new mapping entry has the wrong state");
    require(mappings.try_bind(2, 7, 9, initial.generation),
            "mapping bind failed");
    const auto bound = mappings.lookup(2);
    require(bound.bound && bound.artifact == 7 && bound.slot == 9,
            "bound mapping decoded incorrectly");
    require(!mappings.try_bind(2, 8, 10, initial.generation),
            "mapping overwrite should require release");
    require(mappings.try_release(2, bound), "mapping release failed");
    require(!mappings.try_release(2, bound),
            "stale release defeated the generation tag");
    const auto released = mappings.lookup(2);
    require(!released.bound && released.generation == 1,
            "release did not advance the mapping generation");
    require(mappings.try_bind(2, 8, 10, released.generation),
            "mapping could not be rebound at its new generation");
    const auto second = mappings.lookup(2);
    require(mappings.try_rebind(2, second, 9, 11),
            "atomic artifact-to-artifact rebind failed");
    const auto rebound = mappings.lookup(2);
    require(rebound.bound && rebound.artifact == 9 && rebound.slot == 11 &&
                rebound.generation == 2,
            "atomic rebind published the wrong mapping");
    require(!mappings.try_release(2, second),
            "stale pre-rebind mapping defeated the generation tag");
    require(!mappings.try_rebind(2, second, 10, 12),
            "stale mapping performed a second artifact rebind");
    require(!mappings.lookup(99).source_valid,
            "out-of-range source must not resolve");
}

void test_mapping_concurrent_readers() {
    ptm::ConcurrentMappingTable mappings(1);
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> invalid_reads{0};
    std::vector<std::thread> readers;
    for (int thread = 0; thread < 4; ++thread) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                const auto value = mappings.lookup(0);
                if (!value.source_valid ||
                    (value.bound && (value.artifact != 3 || value.slot != 7))) {
                    invalid_reads.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::size_t iteration = 0; iteration < 100'000; ++iteration) {
        const auto available = mappings.lookup(0);
        require(!available.bound, "writer expected an unbound source");
        require(mappings.try_bind(0, 3, 7, available.generation),
                "concurrent stress bind failed");
        const auto bound = mappings.lookup(0);
        require(mappings.try_release(0, bound),
                "concurrent stress release failed");
    }
    stop.store(true, std::memory_order_release);
    for (auto& reader : readers) {
        reader.join();
    }
    require(invalid_reads.load(std::memory_order_relaxed) == 0,
            "reader observed a torn mapping word");
}

void test_shadow_window_and_policy() {
    ptm::ShadowAuditor auditor(4);
    auditor.record(true, true);
    auditor.record(true, false);
    auditor.record(false, true);
    auditor.record(false, false);
    auditor.record(true, true);
    auditor.record(true, false);
    const auto snapshot = auditor.snapshot();
    require(snapshot.observed == 4 && snapshot.mismatches == 2,
            "shadow auditor did not retain the last four observations");

    ptm::AuditPolicy policy{};
    policy.shadow_min_observations = 4;
    policy.activation_max_mismatch_rate = 0.5;
    policy.live_min_observations = 4;
    policy.reopen_min_mismatches = 2;
    policy.reopen_mismatch_rate = 0.5;
    require(ptm::decide_audit(ptm::AuditPhase::shadow, snapshot, policy) ==
                ptm::AuditDecision::activate,
            "shadow acceptance boundary is wrong");
    require(ptm::decide_audit(ptm::AuditPhase::live, snapshot, policy) ==
                ptm::AuditDecision::reopen,
            "live reopen boundary is wrong");
}

void test_shadow_window_concurrent_writers() {
    ptm::ShadowAuditor auditor(1024);
    std::vector<std::thread> writers;
    for (int thread = 0; thread < 4; ++thread) {
        writers.emplace_back([&auditor, thread] {
            for (std::size_t index = 0; index < 10'000; ++index) {
                const bool mismatch = ((index + static_cast<std::size_t>(thread)) % 5) == 0;
                auditor.record(true, !mismatch);
            }
        });
    }
    for (auto& writer : writers) {
        writer.join();
    }
    const auto snapshot = auditor.snapshot();
    require(snapshot.observed == 1024,
            "concurrent audit writers left holes in the committed window");
    require(snapshot.mismatches <= snapshot.observed,
            "concurrent audit mismatch count is impossible");
}

ptm::MaturityMetrics mature_metrics() {
    return ptm::MaturityMetrics{0.995, 500, 0.001, 0.002, 20, 0.001};
}

ptm::ConsolidationSpec specification(std::string id,
                                     std::vector<ptm::SlotBinding> bindings) {
    return ptm::ConsolidationSpec{
        std::move(id),
        "map-v1",
        "snapshot:before-absorption",
        1024,
        ptm::PortSemantic::ta_action,
        std::move(bindings),
        mature_metrics(),
    };
}

void advance_to_shadow(ptm::ConsolidationRegistry& registry,
                       ptm::ArtifactHandle artifact) {
    require(registry.mark_validated(artifact) == ptm::RegistryStatus::ok,
            "validation transition failed");
    require(registry.mark_compiled(artifact) == ptm::RegistryStatus::ok,
            "compilation transition failed");
    require(registry.begin_shadow(artifact) == ptm::RegistryStatus::ok,
            "shadow transition failed");
}

void record_matches(ptm::ConsolidationRegistry& registry,
                    ptm::ArtifactHandle artifact,
                    std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        require(registry.record_observation(artifact, index % 2 == 0,
                                            index % 2 == 0),
                "shadow observation was rejected");
    }
}

void test_registry_lifecycle_and_conflict_rollback() {
    ptm::AuditPolicy audit_policy{};
    audit_policy.shadow_min_observations = 4;
    audit_policy.activation_max_mismatch_rate = 0.0;
    audit_policy.live_min_observations = 4;
    audit_policy.reopen_min_mismatches = 2;
    audit_policy.reopen_mismatch_rate = 0.25;
    ptm::ConsolidationRegistry registry(64, 8, 8, {}, audit_policy);

    auto immature = specification(
        "artifact:immature", {{0, ptm::SourceKind::ta, 20}});
    immature.maturity.precision = 0.5;
    require(registry.nominate(std::move(immature)).status ==
                ptm::RegistryStatus::immature,
            "immature candidate entered the registry");

    auto first = registry.nominate(specification(
        "artifact:first",
        {{0, ptm::SourceKind::ta, 10}, {1, ptm::SourceKind::ta, 11}}));
    require(first.status == ptm::RegistryStatus::ok,
            "mature candidate was not nominated");
    advance_to_shadow(registry, first.handle);
    require(registry.activate(first.handle) == ptm::RegistryStatus::audit_not_ready,
            "activation ignored minimum shadow observations");
    record_matches(registry, first.handle, 4);
    require(registry.activate(first.handle) == ptm::RegistryStatus::ok,
            "validated shadow artifact did not activate");
    const auto source_10 = registry.resolve(10);
    require(source_10.has_value() && source_10->artifact == first.handle &&
                source_10->slot == 0 && source_10->generation == 0,
            "active source did not resolve in O(1) mapping path");

    auto second = registry.nominate(specification(
        "artifact:second",
        {{2, ptm::SourceKind::ta, 12}, {3, ptm::SourceKind::ta, 10}}));
    require(second.status == ptm::RegistryStatus::ok,
            "second candidate was not nominated");
    advance_to_shadow(registry, second.handle);
    record_matches(registry, second.handle, 4);
    require(registry.activate(second.handle) ==
                ptm::RegistryStatus::mapping_conflict,
            "overlapping artifacts must not activate concurrently");
    const auto rolled_back = registry.mappings().lookup(12);
    require(!rolled_back.bound && rolled_back.generation == 1,
            "partial activation was not rolled back");

    auto inaccurate = registry.nominate(specification(
        "artifact:inaccurate", {{4, ptm::SourceKind::ta, 20}}));
    require(inaccurate.status == ptm::RegistryStatus::ok,
            "inaccurate shadow candidate was not registered");
    advance_to_shadow(registry, inaccurate.handle);
    require(registry.record_observation(inaccurate.handle, true, false),
            "inaccurate shadow observation was rejected");
    record_matches(registry, inaccurate.handle, 3);
    require(registry.activate(inaccurate.handle) ==
                ptm::RegistryStatus::audit_rejected,
            "inaccurate shadow artifact was activated");
    require(registry.reject(inaccurate.handle) == ptm::RegistryStatus::ok,
            "inaccurate artifact was not rejectable");

    require(registry.record_observation(first.handle, true, false),
            "live mismatch observation was rejected");
    require(registry.record_observation(first.handle, false, true),
            "live mismatch observation was rejected");
    require(registry.record_observation(first.handle, true, true),
            "live match observation was rejected");
    require(registry.record_observation(first.handle, false, false),
            "live match observation was rejected");
    require(registry.audit_decision(first.handle) == ptm::AuditDecision::reopen,
            "live drift did not request reopening");
    require(registry.reopen(first.handle) == ptm::RegistryStatus::ok,
            "active artifact did not reopen");
    require(!registry.resolve(10).has_value(),
            "reopened source still resolved to its artifact");
    require(registry.mappings().lookup(10).generation == 1,
            "reopen did not advance source generation");

    require(registry.begin_shadow(first.handle) == ptm::RegistryStatus::ok,
            "reopened artifact could not return to shadowing");
    record_matches(registry, first.handle, 4);
    require(registry.activate(first.handle) == ptm::RegistryStatus::ok,
            "repaired artifact did not reactivate");
    require(registry.resolve(10)->generation == 1,
            "reactivation did not use the current mapping generation");
    require(registry.dissolve(first.handle) == ptm::RegistryStatus::ok,
            "active artifact did not dissolve");
    require(registry.state(first.handle) == ptm::ConsolidationState::dissolved,
            "dissolved artifact has the wrong state");
    require(!registry.resolve(10).has_value(),
            "dissolved artifact remains visible to hot readers");
    const auto retained = registry.specification(first.handle);
    require(retained.has_value() &&
                retained->restoration_handle == "snapshot:before-absorption",
            "dissolution discarded its restoration handle");
}

void test_registry_readers_during_republication() {
    ptm::AuditPolicy audit_policy{};
    audit_policy.shadow_min_observations = 1;
    audit_policy.activation_max_mismatch_rate = 0.0;
    audit_policy.live_min_observations = 1;
    audit_policy.reopen_min_mismatches = 1;
    ptm::ConsolidationRegistry registry(8, 2, 1, {}, audit_policy);
    const auto nomination = registry.nominate(specification(
        "artifact:republish", {{5, ptm::SourceKind::ta, 3}}));
    require(nomination.status == ptm::RegistryStatus::ok,
            "republish candidate nomination failed");
    advance_to_shadow(registry, nomination.handle);
    record_matches(registry, nomination.handle, 1);
    require(registry.activate(nomination.handle) == ptm::RegistryStatus::ok,
            "republish candidate activation failed");

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> invalid_resolutions{0};
    std::vector<std::thread> readers;
    for (int thread = 0; thread < 4; ++thread) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                const auto resolved = registry.resolve(3);
                if (resolved.has_value() &&
                    (resolved->artifact != nomination.handle ||
                     resolved->slot != 5)) {
                    invalid_resolutions.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::size_t iteration = 0; iteration < 1000; ++iteration) {
        require(registry.reopen(nomination.handle) == ptm::RegistryStatus::ok,
                "concurrent republish reopen failed");
        require(registry.begin_shadow(nomination.handle) == ptm::RegistryStatus::ok,
                "concurrent republish shadow transition failed");
        record_matches(registry, nomination.handle, 1);
        require(registry.activate(nomination.handle) == ptm::RegistryStatus::ok,
                "concurrent republish activation failed");
    }
    stop.store(true, std::memory_order_release);
    for (auto& reader : readers) {
        reader.join();
    }
    require(invalid_resolutions.load(std::memory_order_relaxed) == 0,
            "reader resolved a partially published artifact");
}

void test_fixed_logic_evaluator_activation() {
    ptm::FixedLogicProgram32 program{};
    program.instruction_count = 3;
    program.root_instruction = 2;
    program.instructions[0] = {0, ptm::FixedLogicOp::input, 0, 0};
    program.instructions[1] = {0, ptm::FixedLogicOp::input, 1, 0};
    program.instructions[2] = {
        (1U << 0U) | (1U << 1U), ptm::FixedLogicOp::exclusive_or, 0, 0};

    ptm::AuditPolicy audit_policy{};
    audit_policy.shadow_min_observations = 32;
    audit_policy.activation_max_mismatch_rate = 0.0;
    audit_policy.live_min_observations = 32;
    ptm::ConsolidationRegistry registry(8, 2, 32, {}, audit_policy);
    const auto nomination = registry.nominate(ptm::ConsolidationSpec{
        "sha256:logic-evaluator",
        "logic-ast-program32-v1",
        "snapshot:flat-logic-before-evaluator",
        1024,
        ptm::PortSemantic::literal_truth,
        {
            {0, ptm::SourceKind::literal, 0},
            {1, ptm::SourceKind::literal, 1},
            {2, ptm::SourceKind::literal, 2},
            {3, ptm::SourceKind::literal, 3},
            {4, ptm::SourceKind::literal, 4},
        },
        ptm::MaturityMetrics{1.0, 5000, 0.0, 0.0, 5000, 0.0},
    });
    require(nomination.status == ptm::RegistryStatus::ok,
            "fixed Logic evaluator nomination failed");
    advance_to_shadow(registry, nomination.handle);

    for (std::uint8_t bindings = 0; bindings < 32; ++bindings) {
        ptm::FixedLogicResult32 result{};
        require(ptm::evaluate_fixed_logic_program(program, bindings, result) ==
                    ptm::FixedLogicStatus::ok,
                "fixed Logic evaluator failed during shadowing");
        const bool expected = ((bindings & 1U) != 0) !=
                              ((bindings & 2U) != 0);
        require(registry.record_observation(
                    nomination.handle, expected, result.value != 0),
                "fixed Logic shadow observation was rejected");
    }
    require(registry.audit_decision(nomination.handle) ==
                ptm::AuditDecision::activate,
            "exact fixed Logic artifact did not pass shadow audit");
    require(registry.activate(nomination.handle) == ptm::RegistryStatus::ok,
            "exact fixed Logic artifact did not activate");
    for (ptm::SourceHandle source = 0; source < 5; ++source) {
        const auto resolved = registry.resolve(source);
        require(resolved.has_value() &&
                    resolved->artifact == nomination.handle &&
                    resolved->slot == source,
                "active fixed Logic input did not resolve through the registry");
    }
}

void test_transactional_morphology_replacement() {
    ptm::AuditPolicy audit_policy{};
    audit_policy.shadow_min_observations = 4;
    audit_policy.activation_max_mismatch_rate = 0.0;
    audit_policy.live_min_observations = 4;
    ptm::ConsolidationRegistry registry(32, 4, 8, {}, audit_policy);

    const auto parent = registry.nominate(specification(
        "artifact:morphology-parent",
        {{0, ptm::SourceKind::literal, 10},
         {1, ptm::SourceKind::literal, 11}}));
    require(parent.status == ptm::RegistryStatus::ok,
            "morphology parent nomination failed");
    advance_to_shadow(registry, parent.handle);
    record_matches(registry, parent.handle, 4);
    require(registry.activate(parent.handle) == ptm::RegistryStatus::ok,
            "morphology parent activation failed");

    const auto child = registry.nominate(specification(
        "artifact:morphology-child",
        {{4, ptm::SourceKind::literal, 10},
         {5, ptm::SourceKind::literal, 11}}));
    require(child.status == ptm::RegistryStatus::ok,
            "morphology child nomination failed");
    advance_to_shadow(registry, child.handle);
    record_matches(registry, child.handle, 4);
    require(registry.replace_active(parent.handle, child.handle) ==
                ptm::RegistryStatus::ok,
            "audited morphology child did not replace its parent");
    require(registry.state(parent.handle) ==
                ptm::ConsolidationState::reopening &&
                registry.state(child.handle) == ptm::ConsolidationState::active,
            "morphology replacement published the wrong lifecycle states");
    const auto first = registry.resolve(10);
    const auto second = registry.resolve(11);
    require(first.has_value() && second.has_value() &&
                first->artifact == child.handle && first->slot == 4 &&
                second->artifact == child.handle && second->slot == 5 &&
                first->generation == 1 && second->generation == 1,
            "morphology replacement did not atomically rebind parent sources");
    require(registry.reopen(parent.handle) == ptm::RegistryStatus::ok,
            "replaced parent could not complete its reopen transition");

    const auto rejected = registry.nominate(specification(
        "artifact:morphology-rejected",
        {{6, ptm::SourceKind::literal, 10},
         {7, ptm::SourceKind::literal, 11}}));
    require(rejected.status == ptm::RegistryStatus::ok,
            "inaccurate morphology child nomination failed");
    advance_to_shadow(registry, rejected.handle);
    require(registry.record_observation(rejected.handle, true, false),
            "morphology mismatch observation was rejected");
    record_matches(registry, rejected.handle, 3);
    require(registry.replace_active(child.handle, rejected.handle) ==
                ptm::RegistryStatus::audit_rejected,
            "inaccurate morphology child replaced an active artifact");
    require(registry.resolve(10)->artifact == child.handle &&
                registry.state(child.handle) == ptm::ConsolidationState::active,
            "rejected morphology replacement disturbed its active parent");
}

void test_concurrent_readers_during_morphology_handoff() {
    ptm::AuditPolicy audit_policy{};
    audit_policy.shadow_min_observations = 1;
    audit_policy.activation_max_mismatch_rate = 0.0;
    audit_policy.live_min_observations = 1;
    audit_policy.reopen_min_mismatches = 1;
    ptm::ConsolidationRegistry registry(16, 2, 1, {}, audit_policy);
    const auto first = registry.nominate(specification(
        "artifact:morphology-a", {{0, ptm::SourceKind::literal, 10}}));
    const auto second = registry.nominate(specification(
        "artifact:morphology-b", {{4, ptm::SourceKind::literal, 10}}));
    require(first.status == ptm::RegistryStatus::ok &&
                second.status == ptm::RegistryStatus::ok,
            "concurrent morphology artifacts were not nominated");
    advance_to_shadow(registry, first.handle);
    record_matches(registry, first.handle, 1);
    require(registry.activate(first.handle) == ptm::RegistryStatus::ok,
            "first concurrent morphology artifact did not activate");
    advance_to_shadow(registry, second.handle);
    record_matches(registry, second.handle, 1);

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> invalid_resolutions{0};
    std::vector<std::thread> readers;
    for (int thread = 0; thread < 4; ++thread) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                const auto resolved = registry.resolve(10);
                if (resolved.has_value() &&
                    !((resolved->artifact == first.handle &&
                       resolved->slot == 0) ||
                      (resolved->artifact == second.handle &&
                       resolved->slot == 4))) {
                    invalid_resolutions.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    auto active = first.handle;
    auto replacement = second.handle;
    for (std::size_t iteration = 0; iteration < 1000; ++iteration) {
        require(registry.replace_active(active, replacement) ==
                    ptm::RegistryStatus::ok,
                "concurrent morphology handoff failed");
        std::swap(active, replacement);
        require(registry.begin_shadow(replacement) == ptm::RegistryStatus::ok,
                "replaced morphology parent could not re-enter shadowing");
        record_matches(registry, replacement, 1);
    }
    stop.store(true, std::memory_order_release);
    for (auto& reader : readers) {
        reader.join();
    }
    require(invalid_resolutions.load(std::memory_order_relaxed) == 0,
            "reader observed an invalid morphology handoff");
}

}  // namespace

int main() {
    try {
        test_candidate_union_find();
        test_generation_tagged_mapping();
        test_mapping_concurrent_readers();
        test_shadow_window_and_policy();
        test_shadow_window_concurrent_writers();
        test_registry_lifecycle_and_conflict_rollback();
        test_registry_readers_during_republication();
        test_fixed_logic_evaluator_activation();
        test_transactional_morphology_replacement();
        test_concurrent_readers_during_morphology_handoff();
        std::cout << "PTM Class II tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PTM Class II test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
