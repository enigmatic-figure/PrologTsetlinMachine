#include "ptm/logic_program.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

ptm::FixedLogicInstruction instruction(ptm::FixedLogicOp operation,
                                       std::uint32_t operands = 0,
                                       std::uint8_t argument = 0) {
    return ptm::FixedLogicInstruction{operands, operation, argument, 0};
}

ptm::FixedLogicProgram32 conditional_program() {
    ptm::FixedLogicProgram32 program{};
    program.instruction_count = 9;
    program.root_instruction = 8;
    program.instructions[0] = instruction(ptm::FixedLogicOp::input, 0, 0);  // A
    program.instructions[1] = instruction(ptm::FixedLogicOp::input, 0, 1);  // B
    program.instructions[2] = instruction(ptm::FixedLogicOp::input, 0, 2);  // C
    program.instructions[3] =
        instruction(ptm::FixedLogicOp::logical_not, 1U << 1U);
    program.instructions[4] = instruction(
        ptm::FixedLogicOp::conjunction, (1U << 0U) | (1U << 3U));
    program.instructions[5] =
        instruction(ptm::FixedLogicOp::logical_not, 1U << 2U);
    program.instructions[6] = instruction(
        ptm::FixedLogicOp::conjunction, (1U << 1U) | (1U << 5U));
    program.instructions[7] = instruction(
        ptm::FixedLogicOp::conjunction, (1U << 2U) | (1U << 4U));
    program.instructions[8] = instruction(
        ptm::FixedLogicOp::disjunction, (1U << 6U) | (1U << 7U));
    return program;
}

void test_conditional_program_and_diagnostics() {
    const auto program = conditional_program();
    require(ptm::validate_fixed_logic_program(program) ==
                ptm::FixedLogicStatus::ok,
            "valid fixed logic program was rejected");
    for (std::uint8_t bindings = 0; bindings < 32; ++bindings) {
        ptm::FixedLogicResult32 result{};
        require(ptm::evaluate_fixed_logic_program(program, bindings, result) ==
                    ptm::FixedLogicStatus::ok,
                "fixed logic evaluation failed");
        const bool a = (bindings & 1U) != 0;
        const bool b = (bindings & 2U) != 0;
        const bool c = (bindings & 4U) != 0;
        const bool expected = c ? (a && !b) : b;
        require((result.value != 0) == expected,
                "fixed logic conditional produced the wrong value");
        require(result.evaluated_instruction_mask == 0x1FFU,
                "fixed logic diagnostic mask is incomplete");
        require(((result.true_instruction_mask >> program.root_instruction) & 1U) ==
                    result.value,
                "root diagnostic disagrees with result");
    }
}

void test_malformed_program_is_rejected() {
    auto program = conditional_program();
    program.instructions[3].operand_mask = 1U << 8U;
    require(ptm::validate_fixed_logic_program(program) ==
                ptm::FixedLogicStatus::forward_reference,
            "forward instruction reference was accepted");

    program = conditional_program();
    program.instructions[8].operation = static_cast<ptm::FixedLogicOp>(99);
    require(ptm::validate_fixed_logic_program(program) ==
                ptm::FixedLogicStatus::invalid_opcode,
            "unknown fixed logic opcode was accepted");

    ptm::FixedLogicResult32 result{};
    program = conditional_program();
    require(ptm::evaluate_fixed_logic_program(program, 0x80U, result) ==
                ptm::FixedLogicStatus::invalid_bindings,
            "out-of-range binding bits were accepted");
}

}  // namespace

int main() {
    try {
        test_conditional_program_and_diagnostics();
        test_malformed_program_is_rejected();
        std::cout << "PTM fixed logic program tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PTM fixed logic program test failure: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
