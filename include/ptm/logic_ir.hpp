#pragma once

#include "ptm/scalar_tm.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace ptm {

using LogicNodeId = std::uint32_t;

enum class LogicOp : std::uint8_t {
    constant,
    input,
    logical_not,
    conjunction,
    disjunction,
    exclusive_or,
    weighted_threshold,
};

const char* logic_op_name(LogicOp operation) noexcept;

struct LogicNode {
    LogicOp operation{LogicOp::constant};
    bool constant_value{};
    std::uint32_t input_index{};
    std::int64_t threshold{};
    std::vector<LogicNodeId> operands{};
    std::vector<std::int32_t> weights{};
};

struct LogicGraphStats {
    std::size_t node_count{};
    std::size_t operator_count{};
    std::size_t edge_count{};
    std::size_t referenced_input_count{};
    std::size_t max_fan_in{};
    std::size_t depth{};
    std::size_t threshold_count{};
    double input_density{};
    double edge_density{};
};

class LogicGraph {
public:
    LogicGraph() = default;

    [[nodiscard]] LogicNodeId constant(bool value);
    [[nodiscard]] LogicNodeId input(std::uint32_t index);
    [[nodiscard]] LogicNodeId negate(LogicNodeId operand);
    [[nodiscard]] LogicNodeId all(std::span<const LogicNodeId> operands);
    [[nodiscard]] LogicNodeId all(std::initializer_list<LogicNodeId> operands);
    [[nodiscard]] LogicNodeId any(std::span<const LogicNodeId> operands);
    [[nodiscard]] LogicNodeId any(std::initializer_list<LogicNodeId> operands);
    [[nodiscard]] LogicNodeId parity(std::span<const LogicNodeId> operands);
    [[nodiscard]] LogicNodeId parity(
        std::initializer_list<LogicNodeId> operands);
    [[nodiscard]] LogicNodeId implies(LogicNodeId premise,
                                      LogicNodeId conclusion);
    [[nodiscard]] LogicNodeId equivalent(LogicNodeId left,
                                         LogicNodeId right);
    [[nodiscard]] LogicNodeId threshold(
        std::span<const LogicNodeId> operands,
        std::int64_t minimum_true);
    [[nodiscard]] LogicNodeId weighted_threshold(
        std::span<const LogicNodeId> operands,
        std::span<const std::int32_t> weights,
        std::int64_t minimum_score);

    [[nodiscard]] const LogicNode& node(LogicNodeId id) const;
    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }
    [[nodiscard]] LogicGraphStats statistics(
        LogicNodeId root,
        std::size_t available_input_count) const;

private:
    struct NodeKey {
        LogicOp operation{LogicOp::constant};
        bool constant_value{};
        std::uint32_t input_index{};
        std::int64_t threshold{};
        std::vector<LogicNodeId> operands{};
        std::vector<std::int32_t> weights{};

        friend bool operator<(const NodeKey& left, const NodeKey& right) {
            return std::tie(left.operation,
                            left.constant_value,
                            left.input_index,
                            left.threshold,
                            left.operands,
                            left.weights) <
                   std::tie(right.operation,
                            right.constant_value,
                            right.input_index,
                            right.threshold,
                            right.operands,
                            right.weights);
        }
    };

    [[nodiscard]] LogicNodeId intern(LogicNode node);
    void validate(LogicNodeId id) const;

    std::vector<LogicNode> nodes_{};
    std::map<NodeKey, LogicNodeId> interned_{};
};

class PackedInputBatch {
public:
    static PackedInputBatch from_rows(std::span<const std::uint8_t> rows,
                                      std::size_t row_count,
                                      std::size_t feature_count);

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::size_t feature_count() const noexcept {
        return feature_count_;
    }
    [[nodiscard]] std::size_t word_count() const noexcept { return word_count_; }
    [[nodiscard]] std::uint64_t word(std::size_t feature,
                                     std::size_t word_index) const;

private:
    std::size_t row_count_{};
    std::size_t feature_count_{};
    std::size_t word_count_{};
    std::vector<std::uint64_t> words_{};
};

enum class LogicBackend : std::uint8_t {
    cpu_scalar,
    cpu_packed_64,
};

const char* logic_backend_name(LogicBackend backend) noexcept;

enum class LogicInputLayout : std::uint8_t {
    row_major_bytes,
    feature_major_packed,
};

struct LogicWorkload {
    std::size_t batch_size{};
    std::size_t available_input_count{};
    std::size_t expected_reuse{1};
    LogicInputLayout input_layout{LogicInputLayout::row_major_bytes};
};

struct LogicExecutionPlan {
    LogicBackend backend{LogicBackend::cpu_scalar};
    LogicGraphStats graph{};
    std::string rationale{};
};

class LogicPlanner {
public:
    [[nodiscard]] static LogicExecutionPlan choose(const LogicGraph& graph,
                                                   LogicNodeId root,
                                                   const LogicWorkload& workload);
};

class LogicProgram {
public:
    static LogicProgram compile(const LogicGraph& graph, LogicNodeId root);

    [[nodiscard]] bool evaluate(std::span<const std::uint8_t> inputs) const;
    [[nodiscard]] std::vector<std::uint8_t> evaluate_scalar_rows(
        std::span<const std::uint8_t> rows,
        std::size_t row_count,
        std::size_t feature_count) const;
    [[nodiscard]] std::vector<std::uint64_t> evaluate_packed_words(
        const PackedInputBatch& inputs) const;
    [[nodiscard]] std::vector<std::uint8_t> evaluate_packed_rows(
        const PackedInputBatch& inputs) const;

    [[nodiscard]] std::size_t instruction_count() const noexcept {
        return instructions_.size();
    }
    [[nodiscard]] std::size_t required_input_count() const noexcept {
        return required_input_count_;
    }

private:
    struct Instruction {
        LogicOp operation{LogicOp::constant};
        bool constant_value{};
        std::uint32_t input_index{};
        std::int64_t threshold{};
        std::vector<std::uint32_t> operands{};
        std::vector<std::int32_t> weights{};
    };

    [[nodiscard]] bool evaluate_instruction(
        const Instruction& instruction,
        std::span<const std::uint8_t> values) const;

    std::vector<Instruction> instructions_{};
    std::uint32_t root_instruction_{};
    std::size_t required_input_count_{};
};

struct CompiledTsetlinGraph {
    LogicGraph graph{};
    LogicNodeId root{};
    std::vector<LogicNodeId> clause_roots{};
};

[[nodiscard]] CompiledTsetlinGraph lower_scalar_tm(
    const ScalarBinaryTM& machine);

}  // namespace ptm
