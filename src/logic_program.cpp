#include "ptm/logic_program.hpp"

#include <bit>
#include <limits>

namespace ptm {
namespace {

constexpr std::uint32_t preceding_mask(std::size_t instruction) noexcept {
    if (instruction == 0) {
        return 0;
    }
    return (std::uint32_t{1} << instruction) - 1U;
}

constexpr std::uint32_t active_mask(std::size_t count) noexcept {
    return count == fixed_logic_program_capacity
               ? std::numeric_limits<std::uint32_t>::max()
               : (std::uint32_t{1} << count) - 1U;
}

}  // namespace

FixedLogicStatus validate_fixed_logic_program(
    const FixedLogicProgram32& program) noexcept {
    if (program.instruction_count == 0 ||
        program.instruction_count > fixed_logic_program_capacity) {
        return FixedLogicStatus::invalid_instruction_count;
    }
    if (program.root_instruction >= program.instruction_count ||
        program.root_instruction + 1U != program.instruction_count) {
        return FixedLogicStatus::invalid_root;
    }

    for (std::size_t index = 0; index < program.instruction_count; ++index) {
        const auto& instruction = program.instructions[index];
        if (instruction.reserved != 0) {
            return FixedLogicStatus::invalid_argument;
        }
        if ((instruction.operand_mask & ~preceding_mask(index)) != 0) {
            return FixedLogicStatus::forward_reference;
        }
        const auto operand_count = std::popcount(instruction.operand_mask);
        switch (instruction.operation) {
            case FixedLogicOp::constant:
                if (instruction.operand_mask != 0 || instruction.argument > 1) {
                    return FixedLogicStatus::invalid_argument;
                }
                break;
            case FixedLogicOp::input:
                if (instruction.operand_mask != 0 ||
                    instruction.argument >= fixed_logic_binding_count) {
                    return FixedLogicStatus::invalid_argument;
                }
                break;
            case FixedLogicOp::logical_not:
                if (operand_count != 1 || instruction.argument != 0) {
                    return FixedLogicStatus::invalid_operands;
                }
                break;
            case FixedLogicOp::conjunction:
            case FixedLogicOp::disjunction:
            case FixedLogicOp::exclusive_or:
                if (operand_count < 2 || instruction.argument != 0) {
                    return FixedLogicStatus::invalid_operands;
                }
                break;
            default:
                return FixedLogicStatus::invalid_opcode;
        }
    }
    return FixedLogicStatus::ok;
}

FixedLogicStatus evaluate_fixed_logic_program(
    const FixedLogicProgram32& program,
    std::uint8_t binding_bits,
    FixedLogicResult32& result) noexcept {
    result = {};
    const auto validation = validate_fixed_logic_program(program);
    if (validation != FixedLogicStatus::ok) {
        return validation;
    }
    if ((binding_bits >> fixed_logic_binding_count) != 0) {
        return FixedLogicStatus::invalid_bindings;
    }

    std::uint32_t values = 0;
    for (std::size_t index = 0; index < program.instruction_count; ++index) {
        const auto& instruction = program.instructions[index];
        bool value = false;
        switch (instruction.operation) {
            case FixedLogicOp::constant:
                value = instruction.argument != 0;
                break;
            case FixedLogicOp::input:
                value = ((binding_bits >> instruction.argument) & 1U) != 0;
                break;
            case FixedLogicOp::logical_not:
                value = (values & instruction.operand_mask) == 0;
                break;
            case FixedLogicOp::conjunction:
                value = (values & instruction.operand_mask) ==
                        instruction.operand_mask;
                break;
            case FixedLogicOp::disjunction:
                value = (values & instruction.operand_mask) != 0;
                break;
            case FixedLogicOp::exclusive_or:
                value = (std::popcount(values & instruction.operand_mask) & 1U) !=
                        0;
                break;
            default:
                return FixedLogicStatus::invalid_opcode;
        }
        if (value) {
            values |= std::uint32_t{1} << index;
        }
    }

    result.value = static_cast<std::uint8_t>(
        (values >> program.root_instruction) & 1U);
    result.true_instruction_mask = values;
    result.evaluated_instruction_mask = active_mask(program.instruction_count);
    return FixedLogicStatus::ok;
}

}  // namespace ptm
