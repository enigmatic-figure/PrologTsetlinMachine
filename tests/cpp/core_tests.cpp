#include "ptm/bit_block.hpp"
#include "ptm/fredkin.hpp"
#include "ptm/pa_kernel.hpp"
#include "ptm/pa_instance.hpp"
#include "ptm/scalar_tm.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void test_fredkin() {
    std::array<bool, 8> seen{};
    for (unsigned encoded = 0; encoded < 8; ++encoded) {
        const bool control = (encoded & 4U) != 0;
        const bool first = (encoded & 2U) != 0;
        const bool second = (encoded & 1U) != 0;
        const auto result = ptm::fredkin_gate(control, first, second);
        const auto restored =
            ptm::fredkin_gate(result.control, result.first, result.second);
        require(restored == ptm::FredkinResult{control, first, second},
                "Fredkin gate must be its own inverse");
        const unsigned output = (static_cast<unsigned>(result.control) << 2U) |
                                (static_cast<unsigned>(result.first) << 1U) |
                                static_cast<unsigned>(result.second);
        seen[output] = true;
        require(static_cast<unsigned>(control) + static_cast<unsigned>(first) +
                    static_cast<unsigned>(second) ==
                    static_cast<unsigned>(result.control) +
                        static_cast<unsigned>(result.first) +
                        static_cast<unsigned>(result.second),
                "Fredkin gate must conserve Hamming weight");
    }
    for (const bool value : seen) {
        require(value, "Fredkin mapping must be bijective");
    }
    require(ptm::fredkin_literal_condition(false, false).first,
            "excluded literal must be neutral true");
    require(!ptm::fredkin_literal_condition(true, false).first,
            "included false literal must remain false");
}

void test_pa_kernel() {
    ptm::TAAction32x32 selection{};
    selection.set(1, true);
    selection.set(7, true);
    selection.set(70, true);
    ptm::TAAction32x32 inputs{};
    inputs.set(1, true);
    inputs.set(70, true);

    const ptm::MaskedThresholdKernel<1024, ptm::PortSemantic::ta_action> kernel(
        selection, 2);
    const auto result = kernel.evaluate(inputs);
    require(result.value, "two-of-three threshold should pass");
    require(result.matched_count == 2, "kernel matched count is incorrect");
    require(result.selected_count == 3, "kernel selected count is incorrect");
    require(result.missing.population() == 1,
            "kernel missing diagnostic is incorrect");
    require(std::string_view(ptm::port_semantic_name(
                ptm::PortSemantic::ta_action)) == "ta_action",
            "port semantic name is incorrect");
}

void test_pa_instance_mapping() {
    ptm::PAInstance<4096, ptm::PortSemantic::ta_action> instance(
        "map-17",
        "snapshot:before-compile",
        {{9, ptm::SourceKind::ta, 300}, {2, ptm::SourceKind::ta, 101}});
    require(instance.bindings()[0].slot == 2,
            "PA instance bindings must be ordered by slot");
    require(instance.write_source(ptm::SourceKind::ta, 300, true) == 1,
            "source synchronization did not find its slot");
    require(instance.input().get(9),
            "source synchronization did not update the aligned input block");
    require(instance.input().words.size() == 64,
            "64x64 PA must contain exactly 64 words");
}

ptm::ScalarBinaryTM configured_xor() {
    ptm::ScalarBinaryTM machine(4, 2, 4, 3.0, 5, 9);
    for (std::size_t clause = 0; clause < 4; ++clause) {
        for (std::size_t literal = 0; literal < 4; ++literal) {
            machine.set_state(clause, literal, 4);
        }
    }
    machine.set_state(0, 0, 5);
    machine.set_state(0, 3, 5);
    machine.set_state(2, 1, 5);
    machine.set_state(2, 2, 5);
    machine.set_state(1, 0, 5);
    machine.set_state(1, 2, 5);
    machine.set_state(3, 1, 5);
    machine.set_state(3, 3, 5);
    return machine;
}

void test_scalar_tm() {
    auto machine = configured_xor();
    const std::array<std::uint8_t, 2> x00{0, 0};
    const std::array<std::uint8_t, 2> x01{0, 1};
    const std::array<std::uint8_t, 2> x10{1, 0};
    const std::array<std::uint8_t, 2> x11{1, 1};
    require(machine.predict(x00) == 0, "XOR 00 prediction failed");
    require(machine.predict(x01) == 1, "XOR 01 prediction failed");
    require(machine.predict(x10) == 1, "XOR 10 prediction failed");
    require(machine.predict(x11) == 0, "XOR 11 prediction failed");

    const auto before = machine.snapshot();
    machine.update(x10, 1);
    const auto first = machine.snapshot();
    machine.restore(before);
    machine.update(x10, 1);
    const auto replay = machine.snapshot();
    require(first.states == replay.states,
            "restored TM state did not replay identically");
    require(first.rng == replay.rng,
            "restored random stream did not replay identically");
}

}  // namespace

int main() {
    try {
        test_fredkin();
        test_pa_kernel();
        test_pa_instance_mapping();
        test_scalar_tm();
        std::cout << "PTM native tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PTM native test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
