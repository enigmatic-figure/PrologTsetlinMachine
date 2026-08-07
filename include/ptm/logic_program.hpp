#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ptm {

inline constexpr std::size_t fixed_logic_program_capacity = 32;
inline constexpr std::size_t fixed_logic_binding_count = 5;

enum class FixedLogicOp : std::uint8_t {
    constant = 0,
    input = 1,
    logical_not = 2,
    conjunction = 3,
    disjunction = 4,
    exclusive_or = 5,
};

struct FixedLogicInstruction {
    std::uint32_t operand_mask{};
    FixedLogicOp operation{FixedLogicOp::constant};
    std::uint8_t argument{};
    std::uint16_t reserved{};
};

struct alignas(64) FixedLogicProgram32 {
    std::uint32_t instruction_count{};
    std::uint32_t root_instruction{};
    std::array<FixedLogicInstruction, fixed_logic_program_capacity> instructions{};
    std::array<std::uint8_t, 56> alignment_padding{};
};

struct FixedLogicResult32 {
    std::uint8_t value{};
    std::array<std::uint8_t, 3> reserved{};
    std::uint32_t true_instruction_mask{};
    std::uint32_t evaluated_instruction_mask{};
    std::uint32_t alignment_padding{};
};

enum class FixedLogicStatus : std::uint8_t {
    ok,
    invalid_instruction_count,
    invalid_root,
    invalid_opcode,
    invalid_argument,
    invalid_operands,
    forward_reference,
    invalid_bindings,
};

[[nodiscard]] FixedLogicStatus validate_fixed_logic_program(
    const FixedLogicProgram32& program) noexcept;

[[nodiscard]] FixedLogicStatus evaluate_fixed_logic_program(
    const FixedLogicProgram32& program,
    std::uint8_t binding_bits,
    FixedLogicResult32& result) noexcept;

static_assert(sizeof(FixedLogicInstruction) == 8);
static_assert(alignof(FixedLogicProgram32) == 64);
static_assert(sizeof(FixedLogicProgram32) == 320);
static_assert(sizeof(FixedLogicResult32) == 16);

}  // namespace ptm
