#include "ptm/concurrent_mapping.hpp"
#include "ptm/consolidation_registry.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    const std::size_t iterations =
        argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 50'000'000;
    constexpr std::size_t source_count = 65'536;
    ptm::ConcurrentMappingTable mappings(source_count);
    for (std::size_t source = 0; source < source_count; ++source) {
        const auto current = mappings.lookup(static_cast<ptm::SourceHandle>(source));
        if (!mappings.try_bind(static_cast<ptm::SourceHandle>(source),
                               static_cast<ptm::ArtifactHandle>(source % 1024),
                               static_cast<std::uint16_t>(source % 4096),
                               current.generation)) {
            throw std::runtime_error("benchmark mapping setup failed");
        }
    }

    std::uint64_t checksum = 0;
    std::uint32_t source = 1;
    const auto mapping_start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        source = source * 1664525U + 1013904223U;
        const auto mapping = mappings.lookup(source & (source_count - 1));
        checksum += mapping.artifact + mapping.slot + mapping.generation;
    }
    const auto mapping_end = std::chrono::steady_clock::now();
    const auto mapping_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        mapping_end - mapping_start).count();
    const auto nanoseconds_per_lookup = static_cast<double>(mapping_elapsed) /
                                        static_cast<double>(iterations);
    std::cout << "mapping_lock_free=" << (mappings.is_lock_free() ? "true" : "false")
              << " iterations=" << iterations
              << " ns_per_lookup=" << nanoseconds_per_lookup
              << " checksum=" << checksum << '\n';

    ptm::AuditPolicy audit_policy{};
    audit_policy.shadow_min_observations = 1;
    audit_policy.activation_max_mismatch_rate = 0.0;
    audit_policy.live_min_observations = 1;
    audit_policy.reopen_min_mismatches = 1;
    ptm::ConsolidationRegistry registry(
        source_count, source_count / 4096, 1, {}, audit_policy);
    for (std::size_t artifact_index = 0; artifact_index < source_count / 4096;
         ++artifact_index) {
        std::vector<ptm::SlotBinding> bindings;
        bindings.reserve(4096);
        for (std::size_t slot = 0; slot < 4096; ++slot) {
            bindings.push_back(ptm::SlotBinding{
                static_cast<std::uint16_t>(slot),
                ptm::SourceKind::ta,
                artifact_index * 4096 + slot,
            });
        }
        auto nomination = registry.nominate(ptm::ConsolidationSpec{
            "benchmark:" + std::to_string(artifact_index),
            "benchmark-map-v1",
            "benchmark-snapshot",
            4096,
            ptm::PortSemantic::ta_action,
            std::move(bindings),
            ptm::MaturityMetrics{0.999, 1000, 0.0, 0.0, 100, 0.0},
        });
        if (nomination.status != ptm::RegistryStatus::ok ||
            registry.mark_validated(nomination.handle) != ptm::RegistryStatus::ok ||
            registry.mark_compiled(nomination.handle) != ptm::RegistryStatus::ok ||
            registry.begin_shadow(nomination.handle) != ptm::RegistryStatus::ok ||
            !registry.record_observation(nomination.handle, true, true) ||
            registry.activate(nomination.handle) != ptm::RegistryStatus::ok) {
            throw std::runtime_error("registry benchmark setup failed");
        }
    }

    checksum = 0;
    source = 1;
    const auto resolve_start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        source = source * 1664525U + 1013904223U;
        const auto resolved = registry.resolve(source & (source_count - 1));
        if (resolved.has_value()) {
            checksum += resolved->artifact + resolved->slot + resolved->generation;
        }
    }
    const auto resolve_end = std::chrono::steady_clock::now();
    const auto resolve_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        resolve_end - resolve_start).count();
    const auto nanoseconds_per_resolve = static_cast<double>(resolve_elapsed) /
                                         static_cast<double>(iterations);
    std::cout << "active_resolve iterations=" << iterations
              << " ns_per_resolve=" << nanoseconds_per_resolve
              << " checksum=" << checksum << '\n';
    return EXIT_SUCCESS;
}
