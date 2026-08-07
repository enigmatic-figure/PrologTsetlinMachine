#include "ptm/logic_ir.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace ptm {

namespace {

constexpr auto invalid_instruction = std::numeric_limits<std::uint32_t>::max();

bool is_operator(LogicOp operation) noexcept {
    return operation != LogicOp::constant && operation != LogicOp::input;
}

}  // namespace

const char* logic_op_name(LogicOp operation) noexcept {
    switch (operation) {
        case LogicOp::constant:
            return "constant";
        case LogicOp::input:
            return "input";
        case LogicOp::logical_not:
            return "not";
        case LogicOp::conjunction:
            return "and";
        case LogicOp::disjunction:
            return "or";
        case LogicOp::exclusive_or:
            return "xor";
        case LogicOp::weighted_threshold:
            return "weighted_threshold";
    }
    return "unknown";
}

void LogicGraph::validate(LogicNodeId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("logic node identifier");
    }
}

LogicNodeId LogicGraph::intern(LogicNode candidate) {
    const NodeKey key{
        candidate.operation,
        candidate.constant_value,
        candidate.input_index,
        candidate.threshold,
        candidate.operands,
        candidate.weights,
    };
    if (const auto found = interned_.find(key); found != interned_.end()) {
        return found->second;
    }
    if (nodes_.size() >= invalid_instruction) {
        throw std::length_error("logic graph exceeds 32-bit node space");
    }
    const auto id = static_cast<LogicNodeId>(nodes_.size());
    nodes_.push_back(std::move(candidate));
    interned_.emplace(key, id);
    return id;
}

LogicNodeId LogicGraph::constant(bool value) {
    LogicNode result{};
    result.operation = LogicOp::constant;
    result.constant_value = value;
    return intern(std::move(result));
}

LogicNodeId LogicGraph::input(std::uint32_t index) {
    LogicNode result{};
    result.operation = LogicOp::input;
    result.input_index = index;
    return intern(std::move(result));
}

LogicNodeId LogicGraph::negate(LogicNodeId operand) {
    validate(operand);
    const auto& source = node(operand);
    if (source.operation == LogicOp::constant) {
        return constant(!source.constant_value);
    }
    if (source.operation == LogicOp::logical_not) {
        return source.operands.front();
    }
    LogicNode result{};
    result.operation = LogicOp::logical_not;
    result.operands.push_back(operand);
    return intern(std::move(result));
}

LogicNodeId LogicGraph::all(std::initializer_list<LogicNodeId> operands) {
    return all(std::span<const LogicNodeId>(operands.begin(), operands.size()));
}

LogicNodeId LogicGraph::all(std::span<const LogicNodeId> source) {
    std::vector<LogicNodeId> operands;
    for (const auto id : source) {
        validate(id);
        const auto& child = node(id);
        if (child.operation == LogicOp::constant) {
            if (!child.constant_value) {
                return constant(false);
            }
            continue;
        }
        if (child.operation == LogicOp::conjunction) {
            operands.insert(
                operands.end(), child.operands.begin(), child.operands.end());
        } else {
            operands.push_back(id);
        }
    }

    std::sort(operands.begin(), operands.end());
    operands.erase(std::unique(operands.begin(), operands.end()), operands.end());
    for (const auto id : operands) {
        const auto& child = node(id);
        if (child.operation == LogicOp::logical_not &&
            std::binary_search(
                operands.begin(), operands.end(), child.operands.front())) {
            return constant(false);
        }
    }
    if (operands.empty()) {
        return constant(true);
    }
    if (operands.size() == 1) {
        return operands.front();
    }

    LogicNode result{};
    result.operation = LogicOp::conjunction;
    result.operands = std::move(operands);
    return intern(std::move(result));
}

LogicNodeId LogicGraph::any(std::initializer_list<LogicNodeId> operands) {
    return any(std::span<const LogicNodeId>(operands.begin(), operands.size()));
}

LogicNodeId LogicGraph::any(std::span<const LogicNodeId> source) {
    std::vector<LogicNodeId> operands;
    for (const auto id : source) {
        validate(id);
        const auto& child = node(id);
        if (child.operation == LogicOp::constant) {
            if (child.constant_value) {
                return constant(true);
            }
            continue;
        }
        if (child.operation == LogicOp::disjunction) {
            operands.insert(
                operands.end(), child.operands.begin(), child.operands.end());
        } else {
            operands.push_back(id);
        }
    }

    std::sort(operands.begin(), operands.end());
    operands.erase(std::unique(operands.begin(), operands.end()), operands.end());
    for (const auto id : operands) {
        const auto& child = node(id);
        if (child.operation == LogicOp::logical_not &&
            std::binary_search(
                operands.begin(), operands.end(), child.operands.front())) {
            return constant(true);
        }
    }
    if (operands.empty()) {
        return constant(false);
    }
    if (operands.size() == 1) {
        return operands.front();
    }

    // Factor a conjunction shared by every branch. This also implements
    // absorption: p OR (p AND q) becomes p.
    std::vector<LogicNodeId> common;
    const auto factors_for = [this](LogicNodeId id) {
        if (node(id).operation == LogicOp::conjunction) {
            return node(id).operands;
        }
        return std::vector<LogicNodeId>{id};
    };
    common = factors_for(operands.front());
    for (std::size_t index = 1; index < operands.size() && !common.empty();
         ++index) {
        const auto factors = factors_for(operands[index]);
        std::vector<LogicNodeId> intersection;
        std::set_intersection(common.begin(),
                              common.end(),
                              factors.begin(),
                              factors.end(),
                              std::back_inserter(intersection));
        common = std::move(intersection);
    }
    if (!common.empty()) {
        std::vector<LogicNodeId> remainders;
        for (const auto branch : operands) {
            const auto factors = factors_for(branch);
            std::vector<LogicNodeId> remainder;
            std::set_difference(factors.begin(),
                                factors.end(),
                                common.begin(),
                                common.end(),
                                std::back_inserter(remainder));
            if (remainder.empty()) {
                return all(common);
            }
            remainders.push_back(all(remainder));
        }
        common.push_back(any(remainders));
        return all(common);
    }

    LogicNode result{};
    result.operation = LogicOp::disjunction;
    result.operands = std::move(operands);
    return intern(std::move(result));
}

LogicNodeId LogicGraph::parity(std::initializer_list<LogicNodeId> operands) {
    return parity(
        std::span<const LogicNodeId>(operands.begin(), operands.size()));
}

LogicNodeId LogicGraph::parity(std::span<const LogicNodeId> source) {
    bool inverted = false;
    std::map<LogicNodeId, bool> odd;
    std::vector<LogicNodeId> pending(source.begin(), source.end());
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        validate(id);
        const auto& child = node(id);
        if (child.operation == LogicOp::constant) {
            inverted = inverted != child.constant_value;
        } else if (child.operation == LogicOp::exclusive_or) {
            pending.insert(
                pending.end(), child.operands.begin(), child.operands.end());
        } else {
            odd[id] = !odd[id];
        }
    }

    std::vector<LogicNodeId> operands;
    for (const auto& [id, present] : odd) {
        if (present) {
            operands.push_back(id);
        }
    }
    std::vector<bool> removed(operands.size(), false);
    for (std::size_t index = 0; index < operands.size(); ++index) {
        if (removed[index]) {
            continue;
        }
        const auto& child = node(operands[index]);
        if (child.operation != LogicOp::logical_not) {
            continue;
        }
        const auto found = std::lower_bound(
            operands.begin(), operands.end(), child.operands.front());
        if (found != operands.end() && *found == child.operands.front()) {
            const auto other = static_cast<std::size_t>(found - operands.begin());
            if (!removed[other]) {
                removed[index] = true;
                removed[other] = true;
                inverted = !inverted;
            }
        }
    }
    std::vector<LogicNodeId> reduced;
    for (std::size_t index = 0; index < operands.size(); ++index) {
        if (!removed[index]) {
            reduced.push_back(operands[index]);
        }
    }
    if (reduced.empty()) {
        return constant(inverted);
    }
    if (reduced.size() == 1) {
        return inverted ? negate(reduced.front()) : reduced.front();
    }
    LogicNode result{};
    result.operation = LogicOp::exclusive_or;
    result.operands = std::move(reduced);
    const auto id = intern(std::move(result));
    return inverted ? negate(id) : id;
}

LogicNodeId LogicGraph::implies(LogicNodeId premise,
                                LogicNodeId conclusion) {
    return any({negate(premise), conclusion});
}

LogicNodeId LogicGraph::equivalent(LogicNodeId left, LogicNodeId right) {
    return negate(parity({left, right}));
}

LogicNodeId LogicGraph::threshold(std::span<const LogicNodeId> operands,
                                  std::int64_t minimum_true) {
    std::vector<std::int32_t> weights(operands.size(), 1);
    return weighted_threshold(operands, weights, minimum_true);
}

LogicNodeId LogicGraph::weighted_threshold(
    std::span<const LogicNodeId> source,
    std::span<const std::int32_t> source_weights,
    std::int64_t minimum_score) {
    if (source.size() != source_weights.size()) {
        throw std::invalid_argument(
            "weighted threshold operands and weights differ in size");
    }

    std::map<LogicNodeId, std::int64_t> accumulated;
    auto adjusted_threshold = minimum_score;
    for (std::size_t index = 0; index < source.size(); ++index) {
        validate(source[index]);
        const auto weight = source_weights[index];
        const auto& child = node(source[index]);
        if (child.operation == LogicOp::constant) {
            if (child.constant_value) {
                adjusted_threshold -= weight;
            }
        } else {
            accumulated[source[index]] += weight;
        }
    }

    std::vector<LogicNodeId> operands;
    std::vector<std::int32_t> weights;
    std::int64_t minimum_possible = 0;
    std::int64_t maximum_possible = 0;
    for (const auto& [id, weight] : accumulated) {
        if (weight == 0) {
            continue;
        }
        if (weight < std::numeric_limits<std::int32_t>::min() ||
            weight > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("combined threshold weight exceeds int32");
        }
        operands.push_back(id);
        weights.push_back(static_cast<std::int32_t>(weight));
        minimum_possible += std::min<std::int64_t>(0, weight);
        maximum_possible += std::max<std::int64_t>(0, weight);
    }
    if (adjusted_threshold <= minimum_possible) {
        return constant(true);
    }
    if (adjusted_threshold > maximum_possible) {
        return constant(false);
    }

    if (operands.size() == 1) {
        const bool when_false = 0 >= adjusted_threshold;
        const bool when_true = weights.front() >= adjusted_threshold;
        if (when_false == when_true) {
            return constant(when_false);
        }
        return when_true ? operands.front() : negate(operands.front());
    }

    const bool unit_weights = std::all_of(
        weights.begin(), weights.end(), [](std::int32_t value) {
            return value == 1;
        });
    if (unit_weights && adjusted_threshold == 1) {
        return any(operands);
    }
    if (unit_weights &&
        adjusted_threshold == static_cast<std::int64_t>(operands.size())) {
        return all(operands);
    }

    LogicNode result{};
    result.operation = LogicOp::weighted_threshold;
    result.threshold = adjusted_threshold;
    result.operands = std::move(operands);
    result.weights = std::move(weights);
    return intern(std::move(result));
}

const LogicNode& LogicGraph::node(LogicNodeId id) const {
    validate(id);
    return nodes_[id];
}

LogicGraphStats LogicGraph::statistics(
    LogicNodeId root,
    std::size_t available_input_count) const {
    validate(root);
    std::vector<bool> reachable(nodes_.size(), false);
    std::vector<LogicNodeId> pending{root};
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        if (reachable[id]) {
            continue;
        }
        reachable[id] = true;
        pending.insert(
            pending.end(), nodes_[id].operands.begin(), nodes_[id].operands.end());
    }

    LogicGraphStats result{};
    std::set<std::uint32_t> inputs;
    std::vector<std::size_t> depth(nodes_.size(), 0);
    for (std::size_t id = 0; id < nodes_.size(); ++id) {
        if (!reachable[id]) {
            continue;
        }
        ++result.node_count;
        const auto& current = nodes_[id];
        if (current.operation == LogicOp::input) {
            if (current.input_index >= available_input_count) {
                throw std::invalid_argument(
                    "available input count excludes a referenced input");
            }
            inputs.insert(current.input_index);
        }
        if (is_operator(current.operation)) {
            ++result.operator_count;
        }
        if (current.operation == LogicOp::weighted_threshold) {
            ++result.threshold_count;
        }
        result.edge_count += current.operands.size();
        result.max_fan_in = std::max(result.max_fan_in, current.operands.size());
        std::size_t current_depth = 1;
        for (const auto operand : current.operands) {
            current_depth = std::max(current_depth, depth[operand] + 1);
        }
        depth[id] = current_depth;
        result.depth = std::max(result.depth, current_depth);
    }
    result.referenced_input_count = inputs.size();
    if (available_input_count != 0) {
        result.input_density = static_cast<double>(inputs.size()) /
                               static_cast<double>(available_input_count);
    }
    const auto possible_edges = result.operator_count * available_input_count;
    if (possible_edges != 0) {
        result.edge_density = std::min(
            1.0,
            static_cast<double>(result.edge_count) /
                static_cast<double>(possible_edges));
    }
    return result;
}

PackedInputBatch PackedInputBatch::from_rows(
    std::span<const std::uint8_t> rows,
    std::size_t row_count,
    std::size_t feature_count) {
    if (feature_count == 0) {
        throw std::invalid_argument("packed input feature count must be positive");
    }
    if (row_count > std::numeric_limits<std::size_t>::max() / feature_count ||
        rows.size() != row_count * feature_count) {
        throw std::invalid_argument("packed input rows have the wrong shape");
    }
    PackedInputBatch result{};
    result.row_count_ = row_count;
    result.feature_count_ = feature_count;
    result.word_count_ = (row_count + 63) / 64;
    result.words_.assign(feature_count * result.word_count_, 0);
    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t feature = 0; feature < feature_count; ++feature) {
            if (rows[row * feature_count + feature] != 0) {
                result.words_[feature * result.word_count_ + row / 64] |=
                    std::uint64_t{1} << (row % 64);
            }
        }
    }
    return result;
}

std::uint64_t PackedInputBatch::word(std::size_t feature,
                                     std::size_t word_index) const {
    if (feature >= feature_count_ || word_index >= word_count_) {
        throw std::out_of_range("packed input word");
    }
    return words_[feature * word_count_ + word_index];
}

const char* logic_backend_name(LogicBackend backend) noexcept {
    switch (backend) {
        case LogicBackend::cpu_scalar:
            return "cpu_scalar";
        case LogicBackend::cpu_packed_64:
            return "cpu_packed_64";
    }
    return "unknown";
}

LogicExecutionPlan LogicPlanner::choose(const LogicGraph& graph,
                                        LogicNodeId root,
                                        const LogicWorkload& workload) {
    if (workload.batch_size == 0 || workload.available_input_count == 0 ||
        workload.expected_reuse == 0) {
        throw std::invalid_argument(
            "logic workload dimensions and reuse must be positive");
    }
    LogicExecutionPlan result{};
    result.graph = graph.statistics(root, workload.available_input_count);
    const bool already_packed =
        workload.input_layout == LogicInputLayout::feature_major_packed;
    const bool amortized_pack = workload.expected_reuse >= 4;
    const bool dense_enough = result.graph.input_density >= 0.05 ||
                              result.graph.edge_density >= 0.02;

    if ((already_packed && workload.batch_size >= 64) ||
        (amortized_pack && workload.batch_size >= 64) ||
        (dense_enough && workload.batch_size >= 256)) {
        result.backend = LogicBackend::cpu_packed_64;
        result.rationale = already_packed
                               ? "feature-major input avoids transpose cost"
                           : amortized_pack
                               ? "packed input cost is amortized across reuse"
                               : "batch size and logical density amortize packing";
    } else {
        result.backend = LogicBackend::cpu_scalar;
        result.rationale = !dense_enough
                               ? "sparse low-volume graph favors scalar execution"
                               : "batch is below the portable packing crossover";
    }
    return result;
}

LogicProgram LogicProgram::compile(const LogicGraph& graph, LogicNodeId root) {
    (void)graph.node(root);
    std::vector<bool> reachable(graph.size(), false);
    std::vector<LogicNodeId> pending{root};
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        if (reachable[id]) {
            continue;
        }
        reachable[id] = true;
        const auto& source = graph.node(id);
        pending.insert(
            pending.end(), source.operands.begin(), source.operands.end());
    }

    LogicProgram result{};
    std::vector<std::uint32_t> remap(graph.size(), invalid_instruction);
    for (std::size_t old_id = 0; old_id < graph.size(); ++old_id) {
        if (!reachable[old_id]) {
            continue;
        }
        const auto& source = graph.node(static_cast<LogicNodeId>(old_id));
        Instruction instruction{};
        instruction.operation = source.operation;
        instruction.constant_value = source.constant_value;
        instruction.input_index = source.input_index;
        instruction.threshold = source.threshold;
        instruction.weights = source.weights;
        for (const auto operand : source.operands) {
            if (remap[operand] == invalid_instruction) {
                throw std::logic_error("logic graph is not topologically ordered");
            }
            instruction.operands.push_back(remap[operand]);
        }
        if (source.operation == LogicOp::input) {
            result.required_input_count_ = std::max(
                result.required_input_count_,
                static_cast<std::size_t>(source.input_index) + 1);
        }
        if (result.instructions_.size() >= invalid_instruction) {
            throw std::length_error("logic program exceeds 32-bit instruction space");
        }
        remap[old_id] =
            static_cast<std::uint32_t>(result.instructions_.size());
        result.instructions_.push_back(std::move(instruction));
    }
    result.root_instruction_ = remap[root];
    return result;
}

bool LogicProgram::evaluate_instruction(
    const Instruction& instruction,
    std::span<const std::uint8_t> values) const {
    switch (instruction.operation) {
        case LogicOp::constant:
            return instruction.constant_value;
        case LogicOp::input:
            throw std::logic_error("input instruction requires external storage");
        case LogicOp::logical_not:
            return values[instruction.operands.front()] == 0;
        case LogicOp::conjunction:
            return std::all_of(instruction.operands.begin(),
                               instruction.operands.end(),
                               [&values](std::uint32_t operand) {
                                   return values[operand] != 0;
                               });
        case LogicOp::disjunction:
            return std::any_of(instruction.operands.begin(),
                               instruction.operands.end(),
                               [&values](std::uint32_t operand) {
                                   return values[operand] != 0;
                               });
        case LogicOp::exclusive_or: {
            bool value = false;
            for (const auto operand : instruction.operands) {
                value = value != (values[operand] != 0);
            }
            return value;
        }
        case LogicOp::weighted_threshold: {
            std::int64_t score = 0;
            for (std::size_t index = 0; index < instruction.operands.size();
                 ++index) {
                if (values[instruction.operands[index]] != 0) {
                    score += instruction.weights[index];
                }
            }
            return score >= instruction.threshold;
        }
    }
    return false;
}

bool LogicProgram::evaluate(std::span<const std::uint8_t> inputs) const {
    if (inputs.size() < required_input_count_) {
        throw std::invalid_argument("logic program input vector is too short");
    }
    std::vector<std::uint8_t> values(instructions_.size(), 0);
    for (std::size_t index = 0; index < instructions_.size(); ++index) {
        const auto& instruction = instructions_[index];
        if (instruction.operation == LogicOp::input) {
            values[index] = inputs[instruction.input_index] != 0;
        } else {
            values[index] = evaluate_instruction(instruction, values);
        }
    }
    return values[root_instruction_] != 0;
}

std::vector<std::uint8_t> LogicProgram::evaluate_scalar_rows(
    std::span<const std::uint8_t> rows,
    std::size_t row_count,
    std::size_t feature_count) const {
    if (feature_count < required_input_count_ ||
        row_count > std::numeric_limits<std::size_t>::max() / feature_count ||
        rows.size() != row_count * feature_count) {
        throw std::invalid_argument("scalar logic rows have the wrong shape");
    }
    std::vector<std::uint8_t> result(row_count, 0);
    std::vector<std::uint8_t> values(instructions_.size(), 0);
    for (std::size_t row = 0; row < row_count; ++row) {
        const auto row_values = rows.subspan(row * feature_count, feature_count);
        for (std::size_t index = 0; index < instructions_.size(); ++index) {
            const auto& instruction = instructions_[index];
            if (instruction.operation == LogicOp::input) {
                values[index] = row_values[instruction.input_index] != 0;
            } else {
                values[index] = evaluate_instruction(instruction, values);
            }
        }
        result[row] = values[root_instruction_] != 0;
    }
    return result;
}

std::vector<std::uint64_t> LogicProgram::evaluate_packed_words(
    const PackedInputBatch& inputs) const {
    if (inputs.feature_count() < required_input_count_) {
        throw std::invalid_argument("packed logic input has too few features");
    }
    std::vector<std::uint64_t> result(inputs.word_count(), 0);
    std::vector<std::uint64_t> values(instructions_.size(), 0);
    for (std::size_t word_index = 0; word_index < inputs.word_count();
         ++word_index) {
        const auto remaining = inputs.row_count() - word_index * 64;
        const auto valid_bits = remaining >= 64
                                    ? ~std::uint64_t{0}
                                    : (std::uint64_t{1} << remaining) - 1;
        for (std::size_t index = 0; index < instructions_.size(); ++index) {
            const auto& instruction = instructions_[index];
            switch (instruction.operation) {
                case LogicOp::constant:
                    values[index] = instruction.constant_value ? valid_bits : 0;
                    break;
                case LogicOp::input:
                    values[index] =
                        inputs.word(instruction.input_index, word_index) & valid_bits;
                    break;
                case LogicOp::logical_not:
                    values[index] =
                        (~values[instruction.operands.front()]) & valid_bits;
                    break;
                case LogicOp::conjunction: {
                    auto value = valid_bits;
                    for (const auto operand : instruction.operands) {
                        value &= values[operand];
                    }
                    values[index] = value;
                    break;
                }
                case LogicOp::disjunction: {
                    std::uint64_t value = 0;
                    for (const auto operand : instruction.operands) {
                        value |= values[operand];
                    }
                    values[index] = value & valid_bits;
                    break;
                }
                case LogicOp::exclusive_or: {
                    std::uint64_t value = 0;
                    for (const auto operand : instruction.operands) {
                        value ^= values[operand];
                    }
                    values[index] = value & valid_bits;
                    break;
                }
                case LogicOp::weighted_threshold: {
                    const bool unit_weights = std::all_of(
                        instruction.weights.begin(),
                        instruction.weights.end(),
                        [](std::int32_t weight) { return weight == 1; });
                    if (unit_weights && instruction.threshold > 0 &&
                        instruction.threshold <=
                            static_cast<std::int64_t>(
                                instruction.operands.size())) {
                        const auto threshold =
                            static_cast<std::size_t>(instruction.threshold);
                        std::vector<std::uint64_t> at_least(threshold + 1, 0);
                        for (const auto operand : instruction.operands) {
                            for (std::size_t count = threshold; count > 1;
                                 --count) {
                                at_least[count] |=
                                    at_least[count - 1] & values[operand];
                            }
                            at_least[1] |= values[operand];
                        }
                        values[index] = at_least[threshold] & valid_bits;
                    } else {
                        std::uint64_t value = 0;
                        const auto lane_count = std::min<std::size_t>(64, remaining);
                        for (std::size_t lane = 0; lane < lane_count; ++lane) {
                            std::int64_t score = 0;
                            const auto bit = std::uint64_t{1} << lane;
                            for (std::size_t operand = 0;
                                 operand < instruction.operands.size(); ++operand) {
                                if ((values[instruction.operands[operand]] & bit) !=
                                    0) {
                                    score += instruction.weights[operand];
                                }
                            }
                            if (score >= instruction.threshold) {
                                value |= bit;
                            }
                        }
                        values[index] = value;
                    }
                    break;
                }
            }
        }
        result[word_index] = values[root_instruction_] & valid_bits;
    }
    return result;
}

std::vector<std::uint8_t> LogicProgram::evaluate_packed_rows(
    const PackedInputBatch& inputs) const {
    const auto words = evaluate_packed_words(inputs);
    std::vector<std::uint8_t> result(inputs.row_count(), 0);
    for (std::size_t row = 0; row < inputs.row_count(); ++row) {
        result[row] = static_cast<std::uint8_t>(
            (words[row / 64] >> (row % 64)) & std::uint64_t{1});
    }
    return result;
}

CompiledTsetlinGraph lower_scalar_tm(const ScalarBinaryTM& machine) {
    CompiledTsetlinGraph result{};
    result.clause_roots.reserve(machine.number_of_clauses());
    std::vector<std::int32_t> polarities;
    polarities.reserve(machine.number_of_clauses());

    for (std::size_t clause = 0; clause < machine.number_of_clauses(); ++clause) {
        std::vector<LogicNodeId> literals;
        for (std::size_t feature = 0; feature < machine.number_of_features();
             ++feature) {
            const auto positive_literal = feature * 2;
            const auto negative_literal = positive_literal + 1;
            if (machine.action_include(clause, positive_literal)) {
                literals.push_back(
                    result.graph.input(static_cast<std::uint32_t>(feature)));
            }
            if (machine.action_include(clause, negative_literal)) {
                literals.push_back(result.graph.negate(
                    result.graph.input(static_cast<std::uint32_t>(feature))));
            }
        }
        // An empty TM clause is suppressed during prediction.
        const auto clause_root = literals.empty()
                                     ? result.graph.constant(false)
                                     : result.graph.all(literals);
        result.clause_roots.push_back(clause_root);
        polarities.push_back(clause % 2 == 0 ? 1 : -1);
    }
    result.root = result.graph.weighted_threshold(
        result.clause_roots, polarities, 1);
    return result;
}

}  // namespace ptm
