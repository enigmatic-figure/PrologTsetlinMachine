#include "ptm/runtime.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<std::uint8_t, 8> artifact_magic{
    'P', 'T', 'M', 'O', 'D', 'E', 'L', 0};
constexpr std::uint32_t container_version = 1;
constexpr std::uint32_t packed_tm_model_kind = 1;
constexpr std::uint32_t logic_program_model_kind = 2;
constexpr std::uint32_t masked_threshold_model_kind = 3;
constexpr std::uint32_t packed_tm_payload_version = 1;
constexpr std::uint32_t logic_program_payload_version = 1;
constexpr std::uint32_t masked_threshold_payload_version = 1;
constexpr std::size_t container_header_size = 64;
constexpr std::size_t packed_tm_header_size = 32;
constexpr std::size_t logic_program_header_size = 32;
constexpr std::size_t masked_threshold_header_size = 32;
constexpr std::size_t logic_instruction_size = 8;
constexpr std::size_t digest_size = 32;
constexpr std::size_t maximum_artifact_size = 256U * 1024U * 1024U;
constexpr std::uint32_t maximum_dimension = 1U << 20U;
constexpr std::uint32_t maximum_conformance_cases = 16;
constexpr std::uint32_t logic_program_capacity = 32;
constexpr std::uint32_t logic_binding_count = 5;

using Digest = std::array<std::uint8_t, digest_size>;

struct ConformanceCase {
    std::uint64_t valid_mask{};
    std::uint64_t prediction_mask{};
    std::vector<std::uint64_t> feature_words;
    std::array<std::int32_t, 64> scores{};
};

struct LogicInstruction {
    std::uint32_t operand_mask{};
    std::uint8_t opcode{};
    std::uint8_t argument{};
};

struct LogicConformanceCase {
    std::uint64_t valid_mask{};
    std::uint64_t value_mask{};
    std::array<std::uint64_t, logic_binding_count> binding_words{};
    std::array<std::uint32_t, 64> true_instruction_masks{};
    std::array<std::uint32_t, 64> evaluated_instruction_masks{};
};

struct MaskedThresholdConformanceCase {
    std::uint64_t valid_mask{};
    std::uint64_t value_mask{};
    std::vector<std::uint64_t> selected_input_words;
    std::array<std::uint32_t, 64> matched_counts{};
    std::vector<std::uint64_t> matched_selected_words;
    std::vector<std::uint64_t> missing_selected_words;
};

[[nodiscard]] bool add_overflows(std::size_t left,
                                 std::size_t right) noexcept {
    return right > std::numeric_limits<std::size_t>::max() - left;
}

[[nodiscard]] bool product_overflows(std::size_t left,
                                     std::size_t right) noexcept {
    return left != 0 &&
           right > std::numeric_limits<std::size_t>::max() / left;
}

[[nodiscard]] std::uint32_t read_u32(
    std::span<const std::uint8_t> bytes,
    std::size_t offset) noexcept {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
           (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
           (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
}

[[nodiscard]] std::int32_t read_i32(
    std::span<const std::uint8_t> bytes,
    std::size_t offset) noexcept {
    return static_cast<std::int32_t>(read_u32(bytes, offset));
}

[[nodiscard]] std::uint64_t read_u64(
    std::span<const std::uint8_t> bytes,
    std::size_t offset) noexcept {
    std::uint64_t result = 0;
    for (unsigned shift = 0; shift < 64U; shift += 8U) {
        result |= static_cast<std::uint64_t>(bytes[offset + shift / 8U])
                  << shift;
    }
    return result;
}

constexpr std::array<std::uint32_t, 64> sha256_constants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

[[nodiscard]] Digest sha256(std::span<const std::uint8_t> input) {
    std::array<std::uint32_t, 8> state{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::vector<std::uint8_t> padded(input.begin(), input.end());
    const auto bit_length = static_cast<std::uint64_t>(input.size()) * 8U;
    padded.push_back(0x80U);
    while ((padded.size() % 64U) != 56U) {
        padded.push_back(0U);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        padded.push_back(static_cast<std::uint8_t>(bit_length >> shift));
    }

    for (std::size_t offset = 0; offset < padded.size(); offset += 64U) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const auto base = offset + index * 4U;
            words[index] =
                (static_cast<std::uint32_t>(padded[base]) << 24U) |
                (static_cast<std::uint32_t>(padded[base + 1U]) << 16U) |
                (static_cast<std::uint32_t>(padded[base + 2U]) << 8U) |
                static_cast<std::uint32_t>(padded[base + 3U]);
        }
        for (std::size_t index = 16U; index < 64U; ++index) {
            const auto left = words[index - 15U];
            const auto right = words[index - 2U];
            const auto sigma0 = std::rotr(left, 7) ^ std::rotr(left, 18) ^
                                (left >> 3U);
            const auto sigma1 = std::rotr(right, 17) ^
                                std::rotr(right, 19) ^ (right >> 10U);
            words[index] = words[index - 16U] + sigma0 +
                           words[index - 7U] + sigma1;
        }

        auto a = state[0];
        auto b = state[1];
        auto c = state[2];
        auto d = state[3];
        auto e = state[4];
        auto f = state[5];
        auto g = state[6];
        auto h = state[7];
        for (std::size_t index = 0; index < 64U; ++index) {
            const auto sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^
                              std::rotr(e, 25);
            const auto choice = (e & f) ^ (~e & g);
            const auto temporary1 = h + sum1 + choice +
                                    sha256_constants[index] + words[index];
            const auto sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^
                              std::rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    Digest result{};
    for (std::size_t index = 0; index < state.size(); ++index) {
        result[index * 4U] =
            static_cast<std::uint8_t>(state[index] >> 24U);
        result[index * 4U + 1U] =
            static_cast<std::uint8_t>(state[index] >> 16U);
        result[index * 4U + 2U] =
            static_cast<std::uint8_t>(state[index] >> 8U);
        result[index * 4U + 3U] =
            static_cast<std::uint8_t>(state[index]);
    }
    return result;
}

[[nodiscard]] std::string digest_id(const Digest& digest) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result = "sha256:";
    result.reserve(7U + digest.size() * 2U);
    for (const auto value : digest) {
        result.push_back(digits[value >> 4U]);
        result.push_back(digits[value & 0x0fU]);
    }
    return result;
}

void copy_text(char* destination,
               std::size_t capacity,
               std::string_view source) noexcept {
    const auto copied = std::min(capacity - 1U, source.size());
    std::memcpy(destination, source.data(), copied);
    destination[copied] = '\0';
}

void describe_port(ptmrt_port_description& port,
                   std::string_view name,
                   std::string_view semantic,
                   ptmrt_port_direction direction,
                   ptmrt_dtype dtype,
                   std::uint32_t rank,
                   std::uint64_t first_dimension = 0) noexcept {
    port = {};
    copy_text(port.name, sizeof(port.name), name);
    copy_text(port.semantic, sizeof(port.semantic), semantic);
    port.direction = static_cast<std::uint32_t>(direction);
    port.dtype = static_cast<std::uint32_t>(dtype);
    port.rank = rank;
    if (rank != 0) {
        port.shape[0] = first_dimension;
    }
}

[[nodiscard]] bool is_canonical_utf8_text(
    std::span<const std::uint8_t> bytes) noexcept {
    std::size_t index = 0;
    const auto continuation = [&bytes](std::size_t position) {
        return position < bytes.size() &&
               bytes[position] >= 0x80U && bytes[position] <= 0xbfU;
    };
    while (index < bytes.size()) {
        const auto lead = bytes[index];
        if (lead >= 0x20U && lead <= 0x7fU) {
            ++index;
        } else if (lead >= 0xc2U && lead <= 0xdfU &&
                   continuation(index + 1U)) {
            index += 2U;
        } else if (lead == 0xe0U && index + 2U < bytes.size() &&
                   bytes[index + 1U] >= 0xa0U &&
                   bytes[index + 1U] <= 0xbfU &&
                   continuation(index + 2U)) {
            index += 3U;
        } else if (((lead >= 0xe1U && lead <= 0xecU) ||
                    (lead >= 0xeeU && lead <= 0xefU)) &&
                   continuation(index + 1U) &&
                   continuation(index + 2U)) {
            index += 3U;
        } else if (lead == 0xedU && index + 2U < bytes.size() &&
                   bytes[index + 1U] >= 0x80U &&
                   bytes[index + 1U] <= 0x9fU &&
                   continuation(index + 2U)) {
            index += 3U;
        } else if (lead == 0xf0U && index + 3U < bytes.size() &&
                   bytes[index + 1U] >= 0x90U &&
                   bytes[index + 1U] <= 0xbfU &&
                   continuation(index + 2U) &&
                   continuation(index + 3U)) {
            index += 4U;
        } else if (lead >= 0xf1U && lead <= 0xf3U &&
                   continuation(index + 1U) &&
                   continuation(index + 2U) &&
                   continuation(index + 3U)) {
            index += 4U;
        } else if (lead == 0xf4U && index + 3U < bytes.size() &&
                   bytes[index + 1U] >= 0x80U &&
                   bytes[index + 1U] <= 0x8fU &&
                   continuation(index + 2U) &&
                   continuation(index + 3U)) {
            index += 4U;
        } else {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool manifest_contains(std::string_view manifest,
                                     std::string_view value) noexcept {
    return manifest.find(value) != std::string_view::npos;
}

[[nodiscard]] bool packed_manifest_matches(std::string_view manifest,
                                           std::uint32_t clauses,
                                           std::uint32_t features,
                                           std::int32_t threshold,
                                           std::uint32_t conformance_count) {
    return manifest_contains(
               manifest, "\"artifact_schema\":\"ptm.model.v1\"") &&
           manifest_contains(
               manifest,
               "\"artifact_kind\":\"packed_tm_binary_v1\"") &&
           manifest_contains(
               manifest,
               "\"container_digest\":\"sha256-trailer-v1\"") &&
           manifest_contains(
               manifest, "\"number_of_clauses\":" +
                             std::to_string(clauses)) &&
           manifest_contains(
               manifest, "\"number_of_features\":" +
                             std::to_string(features)) &&
           manifest_contains(
               manifest, "\"threshold\":" + std::to_string(threshold)) &&
           manifest_contains(
               manifest, "\"conformance_case_count\":" +
                             std::to_string(conformance_count)) &&
           manifest_contains(
               manifest, "\"dtype\":\"uint64\","
                         "\"layout\":\"feature_major_packed64\","
                         "\"name\":\"features\",\"shape\":[" +
                             std::to_string(features) + "]") &&
           manifest_contains(
               manifest,
               "\"dtype\":\"uint64\",\"name\":\"valid_mask\","
               "\"shape\":[]") &&
           manifest_contains(
               manifest,
               "\"dtype\":\"uint64\",\"name\":\"predictions\","
               "\"shape\":[]") &&
           manifest_contains(
               manifest, "\"dtype\":\"int32\",\"name\":\"scores\","
                         "\"shape\":[64]") &&
           manifest_contains(
               manifest, "\"kind\":\"binary_classification\"") &&
           manifest_contains(
               manifest, "\"materialization\":\"precomputed\"");
}

[[nodiscard]] bool logic_manifest_matches(std::string_view manifest,
                                          std::uint32_t instructions,
                                          std::uint32_t root,
                                          std::uint32_t conformance_count) {
    return manifest_contains(
               manifest, "\"artifact_schema\":\"ptm.model.v1\"") &&
           manifest_contains(
               manifest, "\"artifact_kind\":\"logic_program32_v1\"") &&
           manifest_contains(
               manifest,
               "\"container_digest\":\"sha256-trailer-v1\"") &&
           manifest_contains(
               manifest, "\"binding_count\":5") &&
           manifest_contains(
               manifest, "\"instruction_count\":" +
                             std::to_string(instructions)) &&
           manifest_contains(
               manifest, "\"root_instruction\":" +
                             std::to_string(root)) &&
           manifest_contains(
               manifest, "\"conformance_case_count\":" +
                             std::to_string(conformance_count)) &&
           manifest_contains(
               manifest, "\"dtype\":\"uint64\","
                         "\"layout\":\"binding_major_packed64\","
                         "\"name\":\"bindings\",\"shape\":[5]") &&
           manifest_contains(
               manifest, "\"dtype\":\"uint64\","
                         "\"name\":\"valid_mask\",\"shape\":[]") &&
           manifest_contains(
               manifest, "\"dtype\":\"uint64\","
                         "\"name\":\"values\",\"shape\":[]") &&
           manifest_contains(
               manifest, "\"dtype\":\"uint32\","
                         "\"name\":\"true_instruction_masks\","
                         "\"shape\":[64]") &&
           manifest_contains(
               manifest, "\"dtype\":\"uint32\","
                         "\"name\":\"evaluated_instruction_masks\","
                         "\"shape\":[64]") &&
           manifest_contains(
               manifest, "\"kind\":\"boolean_function\"") &&
           manifest_contains(
               manifest, "\"opcodes\":{\"and\":3,\"constant\":0,"
                         "\"input\":1,\"not\":2,\"or\":4,\"xor\":5}") &&
           manifest_contains(
               manifest, "\"materialization\":\"precomputed\"");
}

[[nodiscard]] bool pa_manifest_matches(std::string_view manifest,
                                       std::uint32_t slots,
                                       std::uint32_t minimum_true,
                                       std::uint32_t selected_count,
                                       std::uint32_t conformance_count) {
    const auto slot_shape = "\"shape\":[" + std::to_string(slots) + "]";
    return manifest_contains(
               manifest, "\"artifact_schema\":\"ptm.model.v1\"") &&
           manifest_contains(
               manifest, "\"artifact_kind\":\"masked_threshold_v1\"") &&
           manifest_contains(
               manifest,
               "\"container_digest\":\"sha256-trailer-v1\"") &&
           manifest_contains(
               manifest, "\"slot_count\":" + std::to_string(slots)) &&
           manifest_contains(
               manifest, "\"minimum_true\":" +
                             std::to_string(minimum_true)) &&
           manifest_contains(
               manifest, "\"selected_count\":" +
                             std::to_string(selected_count)) &&
           manifest_contains(
               manifest, "\"conformance_case_count\":" +
                             std::to_string(conformance_count)) &&
           manifest_contains(
               manifest, "\"dtype\":\"uint64\","
                         "\"layout\":\"slot_major_packed64\","
                         "\"name\":\"slots\"," + slot_shape) &&
           manifest_contains(
               manifest, "\"dtype\":\"uint64\","
                         "\"name\":\"valid_mask\",\"shape\":[]") &&
           manifest_contains(
               manifest, "\"dtype\":\"uint64\","
                         "\"name\":\"values\",\"shape\":[]") &&
           manifest_contains(
               manifest, "\"dtype\":\"uint32\","
                         "\"name\":\"matched_counts\",\"shape\":[64]") &&
           manifest_contains(
               manifest, "\"dtype\":\"uint64\","
                         "\"layout\":\"slot_major_packed64\","
                         "\"name\":\"matched_slots\"," + slot_shape) &&
           manifest_contains(
               manifest, "\"dtype\":\"uint64\","
                         "\"layout\":\"slot_major_packed64\","
                         "\"name\":\"missing_slots\"," + slot_shape) &&
           manifest_contains(
               manifest, "\"kind\":\"boolean_threshold\"") &&
           manifest_contains(
               manifest, "\"materialization\":\"precomputed\"");
}

}  // namespace

struct ptmrt_model {
    std::string manifest;
    ptmrt_model_description description{};
    std::uint32_t feature_word_count{};
    std::vector<std::uint64_t> positive_masks;
    std::vector<std::uint64_t> negative_masks;
    std::vector<ConformanceCase> conformance_cases;
    std::vector<LogicInstruction> logic_instructions;
    LogicConformanceCase logic_conformance{};
    std::vector<std::uint64_t> pa_selection_words;
    std::vector<std::uint32_t> pa_selected_slots;
    std::uint32_t pa_slot_count{};
    std::uint32_t pa_minimum_true{};
    MaskedThresholdConformanceCase pa_conformance{};
};

namespace {

[[nodiscard]] ptmrt_status evaluate_packed_model(
    const ptmrt_model& model,
    std::span<const std::uint64_t> feature_words,
    std::uint64_t valid_mask,
    std::uint64_t& prediction_mask,
    std::array<std::int32_t, 64>& scores) noexcept {
    if (feature_words.size() != model.description.number_of_features) {
        return PTMRT_STATUS_INVALID_ARGUMENT;
    }
    scores.fill(0);
    const auto clauses = model.description.number_of_clauses;
    const auto features = model.description.number_of_features;
    for (std::uint32_t clause = 0; clause < clauses; ++clause) {
        auto output = valid_mask;
        bool empty = true;
        const auto mask_base =
            static_cast<std::size_t>(clause) * model.feature_word_count;
        for (std::uint32_t word = 0; word < model.feature_word_count; ++word) {
            const auto positive = model.positive_masks[mask_base + word];
            const auto negative = model.negative_masks[mask_base + word];
            auto included = positive | negative;
            empty = empty && included == 0;
            while (included != 0) {
                const auto offset = static_cast<std::uint32_t>(
                    std::countr_zero(included));
                const auto feature = word * 64U + offset;
                if (feature < features) {
                    const auto bit = std::uint64_t{1} << offset;
                    const auto truth = feature_words[feature];
                    if ((positive & bit) != 0) {
                        output &= truth;
                    }
                    if ((negative & bit) != 0) {
                        output &= ~truth;
                    }
                }
                included &= included - 1U;
            }
        }
        output &= valid_mask;
        if (empty) {
            output = 0;
        }
        const auto contribution = (clause & 1U) == 0 ? 1 : -1;
        while (output != 0) {
            const auto lane = static_cast<std::size_t>(
                std::countr_zero(output));
            scores[lane] += contribution;
            output &= output - 1U;
        }
    }

    prediction_mask = 0;
    const auto threshold = model.description.threshold;
    for (std::size_t lane = 0; lane < scores.size(); ++lane) {
        const auto bit = std::uint64_t{1} << lane;
        if ((valid_mask & bit) == 0) {
            scores[lane] = 0;
            continue;
        }
        scores[lane] = std::clamp(scores[lane], -threshold, threshold);
        if (scores[lane] > 0) {
            prediction_mask |= bit;
        }
    }
    return PTMRT_STATUS_OK;
}

[[nodiscard]] ptmrt_status evaluate_logic_model(
    const ptmrt_model& model,
    std::span<const std::uint64_t> binding_words,
    std::uint64_t valid_mask,
    std::uint64_t& value_mask,
    std::array<std::uint32_t, 64>& true_masks,
    std::array<std::uint32_t, 64>& evaluated_masks) noexcept {
    if (binding_words.size() != logic_binding_count ||
        model.logic_instructions.empty()) {
        return PTMRT_STATUS_INVALID_ARGUMENT;
    }
    std::array<std::uint64_t, logic_program_capacity> words{};
    for (std::size_t index = 0; index < model.logic_instructions.size(); ++index) {
        const auto& instruction = model.logic_instructions[index];
        auto value = std::uint64_t{0};
        switch (instruction.opcode) {
            case 0:
                value = instruction.argument != 0 ? valid_mask : 0;
                break;
            case 1:
                value = binding_words[instruction.argument] & valid_mask;
                break;
            case 2: {
                const auto operand = static_cast<std::size_t>(
                    std::countr_zero(instruction.operand_mask));
                value = ~words[operand] & valid_mask;
                break;
            }
            case 3:
                value = valid_mask;
                for (auto operands = instruction.operand_mask;
                     operands != 0; operands &= operands - 1U) {
                    value &= words[std::countr_zero(operands)];
                }
                break;
            case 4:
                for (auto operands = instruction.operand_mask;
                     operands != 0; operands &= operands - 1U) {
                    value |= words[std::countr_zero(operands)];
                }
                value &= valid_mask;
                break;
            case 5:
                for (auto operands = instruction.operand_mask;
                     operands != 0; operands &= operands - 1U) {
                    value ^= words[std::countr_zero(operands)];
                }
                value &= valid_mask;
                break;
            default:
                return PTMRT_STATUS_INTERNAL_ERROR;
        }
        words[index] = value;
    }

    true_masks.fill(0);
    evaluated_masks.fill(0);
    const auto count = model.logic_instructions.size();
    const auto evaluated = count == logic_program_capacity
                               ? std::numeric_limits<std::uint32_t>::max()
                               : (std::uint32_t{1} << count) - 1U;
    for (std::size_t lane = 0; lane < 64; ++lane) {
        const auto lane_bit = std::uint64_t{1} << lane;
        if ((valid_mask & lane_bit) == 0) {
            continue;
        }
        for (std::size_t instruction = 0; instruction < count; ++instruction) {
            if ((words[instruction] & lane_bit) != 0) {
                true_masks[lane] |= std::uint32_t{1} << instruction;
            }
        }
        evaluated_masks[lane] = evaluated;
    }
    value_mask = words[model.description.instruction_count - 1U] & valid_mask;
    return PTMRT_STATUS_OK;
}

[[nodiscard]] ptmrt_status evaluate_pa_model(
    const ptmrt_model& model,
    std::span<const std::uint64_t> slot_words,
    std::uint64_t valid_mask,
    std::uint64_t& value_mask,
    std::array<std::uint32_t, 64>& matched_counts,
    std::span<std::uint64_t> matched_slots,
    std::span<std::uint64_t> missing_slots) noexcept {
    if (slot_words.size() != model.pa_slot_count ||
        matched_slots.size() != model.pa_slot_count ||
        missing_slots.size() != model.pa_slot_count) {
        return PTMRT_STATUS_INVALID_ARGUMENT;
    }
    matched_counts.fill(0);
    std::fill(matched_slots.begin(), matched_slots.end(), 0);
    std::fill(missing_slots.begin(), missing_slots.end(), 0);
    for (const auto slot : model.pa_selected_slots) {
        const auto matched = slot_words[slot] & valid_mask;
        const auto missing = ~slot_words[slot] & valid_mask;
        matched_slots[slot] = matched;
        missing_slots[slot] = missing;
        auto lanes = matched;
        while (lanes != 0) {
            const auto lane = static_cast<std::size_t>(std::countr_zero(lanes));
            ++matched_counts[lane];
            lanes &= lanes - 1U;
        }
    }
    value_mask = 0;
    for (std::size_t lane = 0; lane < 64; ++lane) {
        const auto bit = std::uint64_t{1} << lane;
        if ((valid_mask & bit) != 0 &&
            matched_counts[lane] >= model.pa_minimum_true) {
            value_mask |= bit;
        }
    }
    return PTMRT_STATUS_OK;
}

[[nodiscard]] ptmrt_status verify_model(const ptmrt_model& model) {
    if (model.description.model_kind == masked_threshold_model_kind) {
        const auto& expected = model.pa_conformance;
        std::vector<std::uint64_t> slots(model.pa_slot_count, 0);
        std::vector<std::uint64_t> matched(model.pa_slot_count, 0);
        std::vector<std::uint64_t> missing(model.pa_slot_count, 0);
        for (std::size_t index = 0; index < model.pa_selected_slots.size();
             ++index) {
            slots[model.pa_selected_slots[index]] =
                expected.selected_input_words[index];
        }
        std::array<std::uint32_t, 64> counts{};
        std::uint64_t values = 0;
        const auto status = evaluate_pa_model(
            model, slots, expected.valid_mask, values, counts, matched, missing);
        if (status != PTMRT_STATUS_OK || values != expected.value_mask ||
            counts != expected.matched_counts) {
            return PTMRT_STATUS_CONFORMANCE_FAILED;
        }
        for (std::size_t index = 0; index < model.pa_selected_slots.size();
             ++index) {
            const auto slot = model.pa_selected_slots[index];
            if (matched[slot] != expected.matched_selected_words[index] ||
                missing[slot] != expected.missing_selected_words[index]) {
                return PTMRT_STATUS_CONFORMANCE_FAILED;
            }
        }
        return PTMRT_STATUS_OK;
    }
    if (model.description.model_kind == logic_program_model_kind) {
        std::array<std::uint32_t, 64> true_masks{};
        std::array<std::uint32_t, 64> evaluated_masks{};
        std::uint64_t values = 0;
        const auto& expected = model.logic_conformance;
        const auto status = evaluate_logic_model(
            model, expected.binding_words, expected.valid_mask, values,
            true_masks, evaluated_masks);
        if (status != PTMRT_STATUS_OK || values != expected.value_mask ||
            true_masks != expected.true_instruction_masks ||
            evaluated_masks != expected.evaluated_instruction_masks) {
            return PTMRT_STATUS_CONFORMANCE_FAILED;
        }
        return PTMRT_STATUS_OK;
    }
    for (const auto& expected : model.conformance_cases) {
        std::array<std::int32_t, 64> scores{};
        std::uint64_t prediction = 0;
        const auto status = evaluate_packed_model(
            model, expected.feature_words, expected.valid_mask,
            prediction, scores);
        if (status != PTMRT_STATUS_OK ||
            prediction != expected.prediction_mask ||
            scores != expected.scores) {
            return PTMRT_STATUS_CONFORMANCE_FAILED;
        }
    }
    return PTMRT_STATUS_OK;
}

[[nodiscard]] ptmrt_status parse_logic_payload(
    std::span<const std::uint8_t> payload,
    std::string manifest,
    const Digest& digest,
    std::uint32_t version,
    std::uint32_t kind,
    std::unique_ptr<ptmrt_model>& result) {
    if (payload.size() < masked_threshold_header_size) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    const auto payload_version = read_u32(payload, 0);
    const auto instruction_count = read_u32(payload, 4);
    const auto root_instruction = read_u32(payload, 8);
    const auto binding_count = read_u32(payload, 12);
    const auto conformance_count = read_u32(payload, 16);
    const auto payload_flags = read_u32(payload, 20);
    const auto reserved0 = read_u32(payload, 24);
    const auto reserved1 = read_u32(payload, 28);
    if (payload_version != logic_program_payload_version) {
        return PTMRT_STATUS_UNSUPPORTED_VERSION;
    }
    if (instruction_count == 0 || instruction_count > logic_program_capacity ||
        root_instruction + 1U != instruction_count ||
        binding_count != logic_binding_count || conformance_count != 1 ||
        payload_flags != 0 || reserved0 != 0 || reserved1 != 0) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    constexpr auto case_size = 16U + logic_binding_count * 8U +
                               64U * sizeof(std::uint32_t) * 2U;
    const auto expected_size = logic_program_header_size +
                               static_cast<std::size_t>(instruction_count) *
                                   logic_instruction_size +
                               case_size;
    if (payload.size() != expected_size ||
        !logic_manifest_matches(manifest, instruction_count,
                                root_instruction, conformance_count)) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }

    auto model = std::make_unique<ptmrt_model>();
    model->manifest = std::move(manifest);
    model->logic_instructions.reserve(instruction_count);
    std::size_t offset = logic_program_header_size;
    for (std::uint32_t index = 0; index < instruction_count; ++index) {
        LogicInstruction instruction{};
        instruction.operand_mask = read_u32(payload, offset);
        instruction.opcode = payload[offset + 4U];
        instruction.argument = payload[offset + 5U];
        if (payload[offset + 6U] != 0 || payload[offset + 7U] != 0) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
        const auto preceding = index == 0
                                   ? 0U
                                   : (std::uint32_t{1} << index) - 1U;
        if ((instruction.operand_mask & ~preceding) != 0) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
        const auto operand_count = std::popcount(instruction.operand_mask);
        bool valid = false;
        switch (instruction.opcode) {
            case 0:
                valid = instruction.operand_mask == 0 &&
                        instruction.argument <= 1;
                break;
            case 1:
                valid = instruction.operand_mask == 0 &&
                        instruction.argument < logic_binding_count;
                break;
            case 2:
                valid = operand_count == 1 && instruction.argument == 0;
                break;
            case 3:
            case 4:
            case 5:
                valid = operand_count >= 2 && instruction.argument == 0;
                break;
            default:
                return PTMRT_STATUS_INVALID_FORMAT;
        }
        if (!valid) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
        model->logic_instructions.push_back(instruction);
        offset += logic_instruction_size;
    }

    auto& test_case = model->logic_conformance;
    test_case.valid_mask = read_u64(payload, offset);
    test_case.value_mask = read_u64(payload, offset + 8U);
    offset += 16U;
    if ((test_case.value_mask & ~test_case.valid_mask) != 0) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    for (auto& word : test_case.binding_words) {
        word = read_u64(payload, offset);
        offset += 8U;
        if ((word & ~test_case.valid_mask) != 0) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
    }
    const auto active_mask = instruction_count == logic_program_capacity
                                 ? std::numeric_limits<std::uint32_t>::max()
                                 : (std::uint32_t{1} << instruction_count) - 1U;
    for (std::size_t lane = 0; lane < 64; ++lane) {
        test_case.true_instruction_masks[lane] = read_u32(payload, offset);
        offset += 4U;
        if ((test_case.true_instruction_masks[lane] & ~active_mask) != 0 ||
            (((test_case.valid_mask >> lane) & 1U) == 0 &&
             test_case.true_instruction_masks[lane] != 0)) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
    }
    for (std::size_t lane = 0; lane < 64; ++lane) {
        test_case.evaluated_instruction_masks[lane] = read_u32(payload, offset);
        offset += 4U;
        const auto expected = ((test_case.valid_mask >> lane) & 1U) != 0
                                  ? active_mask
                                  : 0U;
        if (test_case.evaluated_instruction_masks[lane] != expected) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
    }

    auto& description = model->description;
    description = {};
    const auto id = digest_id(digest);
    copy_text(description.artifact_id, sizeof(description.artifact_id), id);
    copy_text(description.artifact_schema,
              sizeof(description.artifact_schema), "ptm.model.v1");
    description.container_version = version;
    description.model_kind = kind;
    description.payload_version = payload_version;
    description.input_count = 2;
    description.output_count = 3;
    description.conformance_case_count = conformance_count;
    description.instruction_count = instruction_count;
    description.binding_count = binding_count;
    describe_port(description.inputs[0], "bindings",
                  "binding_major_packed64", PTMRT_PORT_INPUT,
                  PTMRT_DTYPE_UINT64, 1, binding_count);
    describe_port(description.inputs[1], "valid_mask", "valid_lanes",
                  PTMRT_PORT_INPUT, PTMRT_DTYPE_UINT64, 0);
    describe_port(description.outputs[0], "values", "boolean_result",
                  PTMRT_PORT_OUTPUT, PTMRT_DTYPE_UINT64, 0);
    describe_port(description.outputs[1], "true_instruction_masks",
                  "true_instructions", PTMRT_PORT_OUTPUT,
                  PTMRT_DTYPE_UINT32, 1, 64);
    describe_port(description.outputs[2], "evaluated_instruction_masks",
                  "evaluated_instructions", PTMRT_PORT_OUTPUT,
                  PTMRT_DTYPE_UINT32, 1, 64);
    result = std::move(model);
    return PTMRT_STATUS_OK;
}

[[nodiscard]] ptmrt_status parse_pa_payload(
    std::span<const std::uint8_t> payload,
    std::string manifest,
    const Digest& digest,
    std::uint32_t version,
    std::uint32_t kind,
    std::unique_ptr<ptmrt_model>& result) {
    if (payload.size() < logic_program_header_size) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    const auto payload_version = read_u32(payload, 0);
    const auto slot_count = read_u32(payload, 4);
    const auto selection_word_count = read_u32(payload, 8);
    const auto minimum_true = read_u32(payload, 12);
    const auto selected_count = read_u32(payload, 16);
    const auto conformance_count = read_u32(payload, 20);
    const auto payload_flags = read_u32(payload, 24);
    const auto payload_reserved = read_u32(payload, 28);
    if (payload_version != masked_threshold_payload_version) {
        return PTMRT_STATUS_UNSUPPORTED_VERSION;
    }
    if ((slot_count != 1024 && slot_count != 4096) ||
        selection_word_count != slot_count / 64U ||
        minimum_true > selected_count || selected_count > slot_count ||
        conformance_count != 1 || payload_flags != 0 ||
        payload_reserved != 0) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    const auto case_size = 16U + static_cast<std::size_t>(selected_count) * 24U +
                           64U * sizeof(std::uint32_t);
    const auto expected_size = masked_threshold_header_size +
                               static_cast<std::size_t>(selection_word_count) * 8U +
                               case_size;
    if (payload.size() != expected_size ||
        !pa_manifest_matches(manifest, slot_count, minimum_true,
                             selected_count, conformance_count)) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }

    auto model = std::make_unique<ptmrt_model>();
    model->manifest = std::move(manifest);
    model->pa_slot_count = slot_count;
    model->pa_minimum_true = minimum_true;
    model->pa_selection_words.resize(selection_word_count);
    std::size_t offset = masked_threshold_header_size;
    for (std::uint32_t word = 0; word < selection_word_count; ++word) {
        model->pa_selection_words[word] = read_u64(payload, offset);
        auto selected = model->pa_selection_words[word];
        while (selected != 0) {
            model->pa_selected_slots.push_back(
                word * 64U + static_cast<std::uint32_t>(
                                 std::countr_zero(selected)));
            selected &= selected - 1U;
        }
        offset += 8U;
    }
    if (model->pa_selected_slots.size() != selected_count) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }

    auto& test_case = model->pa_conformance;
    test_case.valid_mask = read_u64(payload, offset);
    test_case.value_mask = read_u64(payload, offset + 8U);
    offset += 16U;
    if ((test_case.value_mask & ~test_case.valid_mask) != 0) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    test_case.selected_input_words.resize(selected_count);
    for (auto& word : test_case.selected_input_words) {
        word = read_u64(payload, offset);
        offset += 8U;
        if ((word & ~test_case.valid_mask) != 0) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
    }
    for (std::size_t lane = 0; lane < 64; ++lane) {
        test_case.matched_counts[lane] = read_u32(payload, offset);
        offset += 4U;
        if (test_case.matched_counts[lane] > selected_count ||
            (((test_case.valid_mask >> lane) & 1U) == 0 &&
             test_case.matched_counts[lane] != 0)) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
    }
    test_case.matched_selected_words.resize(selected_count);
    for (std::size_t index = 0; index < selected_count; ++index) {
        const auto word = read_u64(payload, offset);
        offset += 8U;
        if (word !=
            (test_case.selected_input_words[index] & test_case.valid_mask)) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
        test_case.matched_selected_words[index] = word;
    }
    test_case.missing_selected_words.resize(selected_count);
    for (std::size_t index = 0; index < selected_count; ++index) {
        const auto word = read_u64(payload, offset);
        offset += 8U;
        if (word !=
            (~test_case.selected_input_words[index] & test_case.valid_mask)) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
        test_case.missing_selected_words[index] = word;
    }

    auto& description = model->description;
    description = {};
    const auto id = digest_id(digest);
    copy_text(description.artifact_id, sizeof(description.artifact_id), id);
    copy_text(description.artifact_schema,
              sizeof(description.artifact_schema), "ptm.model.v1");
    description.container_version = version;
    description.model_kind = kind;
    description.payload_version = payload_version;
    description.input_count = 2;
    description.output_count = 4;
    description.conformance_case_count = conformance_count;
    description.slot_count = slot_count;
    description.minimum_true = minimum_true;
    description.selected_count = selected_count;
    describe_port(description.inputs[0], "slots", "slot_major_packed64",
                  PTMRT_PORT_INPUT, PTMRT_DTYPE_UINT64, 1, slot_count);
    describe_port(description.inputs[1], "valid_mask", "valid_lanes",
                  PTMRT_PORT_INPUT, PTMRT_DTYPE_UINT64, 0);
    describe_port(description.outputs[0], "values", "boolean_result",
                  PTMRT_PORT_OUTPUT, PTMRT_DTYPE_UINT64, 0);
    describe_port(description.outputs[1], "matched_counts", "matched_count",
                  PTMRT_PORT_OUTPUT, PTMRT_DTYPE_UINT32, 1, 64);
    describe_port(description.outputs[2], "matched_slots", "matched_slots",
                  PTMRT_PORT_OUTPUT, PTMRT_DTYPE_UINT64, 1, slot_count);
    describe_port(description.outputs[3], "missing_slots", "missing_slots",
                  PTMRT_PORT_OUTPUT, PTMRT_DTYPE_UINT64, 1, slot_count);
    result = std::move(model);
    return PTMRT_STATUS_OK;
}

[[nodiscard]] ptmrt_status parse_artifact(
    std::span<const std::uint8_t> bytes,
    std::unique_ptr<ptmrt_model>& result) {
    if (bytes.size() < container_header_size + packed_tm_header_size +
                           digest_size ||
        bytes.size() > maximum_artifact_size) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    if (!std::equal(artifact_magic.begin(), artifact_magic.end(),
                    bytes.begin())) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    const auto version = read_u32(bytes, 8);
    const auto header_size = read_u32(bytes, 12);
    const auto kind = read_u32(bytes, 16);
    const auto flags = read_u32(bytes, 20);
    const auto manifest_size_u64 = read_u64(bytes, 24);
    const auto payload_size_u64 = read_u64(bytes, 32);
    if (version != container_version || header_size != container_header_size) {
        return PTMRT_STATUS_UNSUPPORTED_VERSION;
    }
    if (kind != packed_tm_model_kind && kind != logic_program_model_kind &&
        kind != masked_threshold_model_kind) {
        return PTMRT_STATUS_UNSUPPORTED_MODEL;
    }
    if (flags != 0 || std::any_of(bytes.begin() + 40, bytes.begin() + 64,
                                  [](std::uint8_t value) {
                                      return value != 0;
                                  })) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    if (manifest_size_u64 > std::numeric_limits<std::size_t>::max() ||
        payload_size_u64 > std::numeric_limits<std::size_t>::max()) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    const auto manifest_size = static_cast<std::size_t>(manifest_size_u64);
    const auto payload_size = static_cast<std::size_t>(payload_size_u64);
    if (add_overflows(header_size, manifest_size) ||
        add_overflows(header_size + manifest_size, payload_size) ||
        add_overflows(header_size + manifest_size + payload_size,
                      digest_size) ||
        header_size + manifest_size + payload_size + digest_size !=
            bytes.size() ||
        payload_size < packed_tm_header_size) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }

    const auto content_size = bytes.size() - digest_size;
    const auto calculated = sha256(bytes.first(content_size));
    if (!std::equal(calculated.begin(), calculated.end(),
                    bytes.begin() + static_cast<std::ptrdiff_t>(content_size))) {
        return PTMRT_STATUS_INTEGRITY_ERROR;
    }

    const auto manifest_bytes = bytes.subspan(header_size, manifest_size);
    if (manifest_bytes.size() < 2U || manifest_bytes.front() != '{' ||
        manifest_bytes.back() != '}' ||
        !is_canonical_utf8_text(manifest_bytes)) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    std::string manifest(
        reinterpret_cast<const char*>(manifest_bytes.data()),
        manifest_bytes.size());
    const auto payload = bytes.subspan(header_size + manifest_size,
                                       payload_size);
    if (kind == logic_program_model_kind) {
        return parse_logic_payload(payload, std::move(manifest), calculated,
                                   version, kind, result);
    }
    if (kind == masked_threshold_model_kind) {
        return parse_pa_payload(payload, std::move(manifest), calculated,
                                version, kind, result);
    }
    const auto payload_version = read_u32(payload, 0);
    const auto clauses = read_u32(payload, 4);
    const auto features = read_u32(payload, 8);
    const auto feature_word_count = read_u32(payload, 12);
    const auto threshold = read_i32(payload, 16);
    const auto conformance_count = read_u32(payload, 20);
    const auto payload_flags = read_u32(payload, 24);
    const auto payload_reserved = read_u32(payload, 28);
    if (payload_version != packed_tm_payload_version) {
        return PTMRT_STATUS_UNSUPPORTED_VERSION;
    }
    if (clauses == 0 || features == 0 || clauses > maximum_dimension ||
        features > maximum_dimension || threshold <= 0 ||
        feature_word_count != (features + 63U) / 64U ||
        conformance_count == 0 ||
        conformance_count > maximum_conformance_cases ||
        payload_flags != 0 || payload_reserved != 0) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    if (!packed_manifest_matches(
            manifest, clauses, features, threshold, conformance_count)) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }

    if (product_overflows(clauses, feature_word_count)) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    const auto mask_count =
        static_cast<std::size_t>(clauses) * feature_word_count;
    if (product_overflows(mask_count, 16U) ||
        product_overflows(features, 8U)) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    const auto case_size = 16U + static_cast<std::size_t>(features) * 8U +
                           64U * sizeof(std::int32_t);
    if (product_overflows(conformance_count, case_size) ||
        add_overflows(packed_tm_header_size, mask_count * 16U) ||
        add_overflows(packed_tm_header_size + mask_count * 16U,
                      static_cast<std::size_t>(conformance_count) * case_size) ||
        packed_tm_header_size + mask_count * 16U +
                static_cast<std::size_t>(conformance_count) * case_size !=
            payload.size()) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }

    auto model = std::make_unique<ptmrt_model>();
    model->manifest = std::move(manifest);
    model->feature_word_count = feature_word_count;
    model->positive_masks.resize(mask_count);
    model->negative_masks.resize(mask_count);
    std::size_t offset = packed_tm_header_size;
    for (auto& mask : model->positive_masks) {
        mask = read_u64(payload, offset);
        offset += 8U;
    }
    for (auto& mask : model->negative_masks) {
        mask = read_u64(payload, offset);
        offset += 8U;
    }
    const auto tail = features % 64U;
    if (tail != 0) {
        const auto invalid = ~((std::uint64_t{1} << tail) - 1U);
        for (std::uint32_t clause = 0; clause < clauses; ++clause) {
            const auto tail_index =
                static_cast<std::size_t>(clause) * feature_word_count +
                feature_word_count - 1U;
            if (((model->positive_masks[tail_index] |
                  model->negative_masks[tail_index]) & invalid) != 0) {
                return PTMRT_STATUS_INVALID_FORMAT;
            }
        }
    }

    model->conformance_cases.reserve(conformance_count);
    for (std::uint32_t index = 0; index < conformance_count; ++index) {
        ConformanceCase test_case{};
        test_case.valid_mask = read_u64(payload, offset);
        test_case.prediction_mask = read_u64(payload, offset + 8U);
        offset += 16U;
        if ((test_case.prediction_mask & ~test_case.valid_mask) != 0) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
        test_case.feature_words.resize(features);
        for (auto& word : test_case.feature_words) {
            word = read_u64(payload, offset);
            offset += 8U;
        }
        for (std::size_t lane = 0; lane < test_case.scores.size(); ++lane) {
            test_case.scores[lane] = read_i32(payload, offset);
            offset += 4U;
            if (((test_case.valid_mask >> lane) & 1U) == 0 &&
                test_case.scores[lane] != 0) {
                return PTMRT_STATUS_INVALID_FORMAT;
            }
        }
        model->conformance_cases.push_back(std::move(test_case));
    }

    auto& description = model->description;
    description = {};
    const auto id = digest_id(calculated);
    copy_text(description.artifact_id, sizeof(description.artifact_id), id);
    copy_text(description.artifact_schema,
              sizeof(description.artifact_schema), "ptm.model.v1");
    description.container_version = version;
    description.model_kind = kind;
    description.payload_version = payload_version;
    description.input_count = 2;
    description.output_count = 2;
    description.conformance_case_count = conformance_count;
    description.number_of_clauses = clauses;
    description.number_of_features = features;
    description.threshold = threshold;
    describe_port(description.inputs[0], "features",
                  "feature_major_packed64", PTMRT_PORT_INPUT,
                  PTMRT_DTYPE_UINT64, 1, features);
    describe_port(description.inputs[1], "valid_mask", "valid_lanes",
                  PTMRT_PORT_INPUT, PTMRT_DTYPE_UINT64, 0);
    describe_port(description.outputs[0], "predictions", "binary_prediction",
                  PTMRT_PORT_OUTPUT, PTMRT_DTYPE_UINT64, 0);
    describe_port(description.outputs[1], "scores", "signed_clamped_score",
                  PTMRT_PORT_OUTPUT, PTMRT_DTYPE_INT32, 1, 64);

    result = std::move(model);
    return PTMRT_STATUS_OK;
}

[[nodiscard]] const ptmrt_tensor_view* find_tensor(
    const ptmrt_tensor_view* tensors,
    std::uint32_t count,
    std::string_view name) noexcept {
    const ptmrt_tensor_view* result = nullptr;
    for (std::uint32_t index = 0; index < count; ++index) {
        if (tensors[index].name != nullptr &&
            name == tensors[index].name) {
            if (result != nullptr) {
                return nullptr;
            }
            result = tensors + index;
        }
    }
    return result;
}

[[nodiscard]] bool valid_tensor(const ptmrt_tensor_view& tensor,
                                ptmrt_dtype dtype,
                                std::uint32_t rank,
                                std::uint64_t first_dimension,
                                std::uint64_t bytes) noexcept {
    return tensor.data != nullptr &&
           tensor.dtype == static_cast<std::uint32_t>(dtype) &&
           tensor.rank == rank && (rank == 0 || tensor.shape[0] == first_dimension) &&
           tensor.byte_size >= bytes;
}

}  // namespace

extern "C" {

std::uint32_t ptmrt_abi_version(void) {
    return PTMRT_ABI_VERSION;
}

const char* ptmrt_status_message(ptmrt_status status) {
    switch (status) {
        case PTMRT_STATUS_OK: return "ok";
        case PTMRT_STATUS_NULL_POINTER: return "null pointer";
        case PTMRT_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case PTMRT_STATUS_IO_ERROR: return "I/O error";
        case PTMRT_STATUS_INVALID_FORMAT: return "invalid artifact format";
        case PTMRT_STATUS_UNSUPPORTED_VERSION: return "unsupported version";
        case PTMRT_STATUS_UNSUPPORTED_MODEL: return "unsupported model";
        case PTMRT_STATUS_INTEGRITY_ERROR: return "integrity check failed";
        case PTMRT_STATUS_INSUFFICIENT_CAPACITY:
            return "insufficient capacity";
        case PTMRT_STATUS_CONFORMANCE_FAILED:
            return "embedded conformance check failed";
        case PTMRT_STATUS_INTERNAL_ERROR: return "internal error";
    }
    return "unknown status";
}

ptmrt_status ptmrt_model_open_file(const char* path, ptmrt_model** model) {
    if (path == nullptr || model == nullptr) {
        return PTMRT_STATUS_NULL_POINTER;
    }
    *model = nullptr;
    try {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            return PTMRT_STATUS_IO_ERROR;
        }
        const auto end = stream.tellg();
        if (end < 0 || static_cast<std::uint64_t>(end) > maximum_artifact_size) {
            return PTMRT_STATUS_INVALID_FORMAT;
        }
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        if (!bytes.empty() &&
            !stream.read(reinterpret_cast<char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size()))) {
            return PTMRT_STATUS_IO_ERROR;
        }
        std::unique_ptr<ptmrt_model> loaded;
        const auto status = parse_artifact(bytes, loaded);
        if (status == PTMRT_STATUS_OK) {
            *model = loaded.release();
        }
        return status;
    } catch (...) {
        return PTMRT_STATUS_INTERNAL_ERROR;
    }
}

ptmrt_status ptmrt_model_open_memory(const void* data,
                                     std::uint64_t size,
                                     ptmrt_model** model) {
    if (data == nullptr || model == nullptr) {
        return PTMRT_STATUS_NULL_POINTER;
    }
    *model = nullptr;
    if (size > maximum_artifact_size ||
        size > std::numeric_limits<std::size_t>::max()) {
        return PTMRT_STATUS_INVALID_FORMAT;
    }
    try {
        const auto* first = static_cast<const std::uint8_t*>(data);
        std::unique_ptr<ptmrt_model> loaded;
        const auto status = parse_artifact(
            std::span<const std::uint8_t>(first, static_cast<std::size_t>(size)),
            loaded);
        if (status == PTMRT_STATUS_OK) {
            *model = loaded.release();
        }
        return status;
    } catch (...) {
        return PTMRT_STATUS_INTERNAL_ERROR;
    }
}

void ptmrt_model_close(ptmrt_model* model) {
    delete model;
}

ptmrt_status ptmrt_model_describe(
    const ptmrt_model* model,
    ptmrt_model_description* description) {
    if (model == nullptr || description == nullptr) {
        return PTMRT_STATUS_NULL_POINTER;
    }
    *description = model->description;
    return PTMRT_STATUS_OK;
}

ptmrt_status ptmrt_model_manifest_json(const ptmrt_model* model,
                                       char* buffer,
                                       std::uint64_t capacity,
                                       std::uint64_t* required_size) {
    if (model == nullptr || required_size == nullptr) {
        return PTMRT_STATUS_NULL_POINTER;
    }
    const auto required = static_cast<std::uint64_t>(model->manifest.size()) + 1U;
    *required_size = required;
    if (buffer == nullptr) {
        return capacity == 0 ? PTMRT_STATUS_OK : PTMRT_STATUS_NULL_POINTER;
    }
    if (capacity < required) {
        return PTMRT_STATUS_INSUFFICIENT_CAPACITY;
    }
    std::memcpy(buffer, model->manifest.data(), model->manifest.size());
    buffer[model->manifest.size()] = '\0';
    return PTMRT_STATUS_OK;
}

ptmrt_status ptmrt_model_verify(const ptmrt_model* model) {
    if (model == nullptr) {
        return PTMRT_STATUS_NULL_POINTER;
    }
    try {
        return verify_model(*model);
    } catch (...) {
        return PTMRT_STATUS_INTERNAL_ERROR;
    }
}

ptmrt_status ptmrt_model_run(const ptmrt_model* model,
                             const ptmrt_tensor_view* inputs,
                             std::uint32_t input_count,
                             ptmrt_tensor_view* outputs,
                             std::uint32_t output_count) {
    if (model == nullptr || inputs == nullptr || outputs == nullptr) {
        return PTMRT_STATUS_NULL_POINTER;
    }
    if (model->description.model_kind == masked_threshold_model_kind) {
        if (input_count != 2 || output_count != 4) {
            return PTMRT_STATUS_INVALID_ARGUMENT;
        }
        const auto* slots = find_tensor(inputs, input_count, "slots");
        const auto* valid = find_tensor(inputs, input_count, "valid_mask");
        const auto* values = find_tensor(outputs, output_count, "values");
        const auto* matched_counts = find_tensor(
            outputs, output_count, "matched_counts");
        const auto* matched_slots = find_tensor(
            outputs, output_count, "matched_slots");
        const auto* missing_slots = find_tensor(
            outputs, output_count, "missing_slots");
        const auto slot_bytes =
            static_cast<std::uint64_t>(model->pa_slot_count) *
            sizeof(std::uint64_t);
        constexpr auto count_bytes = 64U * sizeof(std::uint32_t);
        if (slots == nullptr || valid == nullptr || values == nullptr ||
            matched_counts == nullptr || matched_slots == nullptr ||
            missing_slots == nullptr ||
            !valid_tensor(*slots, PTMRT_DTYPE_UINT64, 1,
                          model->pa_slot_count, slot_bytes) ||
            !valid_tensor(*valid, PTMRT_DTYPE_UINT64, 0, 0,
                          sizeof(std::uint64_t)) ||
            !valid_tensor(*values, PTMRT_DTYPE_UINT64, 0, 0,
                          sizeof(std::uint64_t)) ||
            !valid_tensor(*matched_counts, PTMRT_DTYPE_UINT32, 1, 64,
                          count_bytes) ||
            !valid_tensor(*matched_slots, PTMRT_DTYPE_UINT64, 1,
                          model->pa_slot_count, slot_bytes) ||
            !valid_tensor(*missing_slots, PTMRT_DTYPE_UINT64, 1,
                          model->pa_slot_count, slot_bytes)) {
            return PTMRT_STATUS_INVALID_ARGUMENT;
        }
        try {
            std::vector<std::uint64_t> slot_values(model->pa_slot_count);
            std::memcpy(slot_values.data(), slots->data,
                        static_cast<std::size_t>(slot_bytes));
            std::uint64_t valid_mask = 0;
            std::memcpy(&valid_mask, valid->data, sizeof(valid_mask));
            std::uint64_t value_mask = 0;
            std::array<std::uint32_t, 64> count_values{};
            std::vector<std::uint64_t> matched_values(model->pa_slot_count);
            std::vector<std::uint64_t> missing_values(model->pa_slot_count);
            const auto status = evaluate_pa_model(
                *model, slot_values, valid_mask, value_mask, count_values,
                matched_values, missing_values);
            if (status != PTMRT_STATUS_OK) {
                return status;
            }
            std::memcpy(values->data, &value_mask, sizeof(value_mask));
            std::memcpy(matched_counts->data, count_values.data(), count_bytes);
            std::memcpy(matched_slots->data, matched_values.data(),
                        static_cast<std::size_t>(slot_bytes));
            std::memcpy(missing_slots->data, missing_values.data(),
                        static_cast<std::size_t>(slot_bytes));
            return PTMRT_STATUS_OK;
        } catch (...) {
            return PTMRT_STATUS_INTERNAL_ERROR;
        }
    }
    if (model->description.model_kind == logic_program_model_kind) {
        if (input_count != 2 || output_count != 3) {
            return PTMRT_STATUS_INVALID_ARGUMENT;
        }
        const auto* bindings = find_tensor(inputs, input_count, "bindings");
        const auto* valid = find_tensor(inputs, input_count, "valid_mask");
        const auto* values = find_tensor(outputs, output_count, "values");
        const auto* true_masks = find_tensor(
            outputs, output_count, "true_instruction_masks");
        const auto* evaluated_masks = find_tensor(
            outputs, output_count, "evaluated_instruction_masks");
        constexpr auto binding_bytes =
            logic_binding_count * sizeof(std::uint64_t);
        constexpr auto diagnostic_bytes = 64U * sizeof(std::uint32_t);
        if (bindings == nullptr || valid == nullptr || values == nullptr ||
            true_masks == nullptr || evaluated_masks == nullptr ||
            !valid_tensor(*bindings, PTMRT_DTYPE_UINT64, 1,
                          logic_binding_count, binding_bytes) ||
            !valid_tensor(*valid, PTMRT_DTYPE_UINT64, 0, 0,
                          sizeof(std::uint64_t)) ||
            !valid_tensor(*values, PTMRT_DTYPE_UINT64, 0, 0,
                          sizeof(std::uint64_t)) ||
            !valid_tensor(*true_masks, PTMRT_DTYPE_UINT32, 1, 64,
                          diagnostic_bytes) ||
            !valid_tensor(*evaluated_masks, PTMRT_DTYPE_UINT32, 1, 64,
                          diagnostic_bytes)) {
            return PTMRT_STATUS_INVALID_ARGUMENT;
        }
        try {
            std::array<std::uint64_t, logic_binding_count> binding_values{};
            std::memcpy(binding_values.data(), bindings->data, binding_bytes);
            std::uint64_t valid_mask = 0;
            std::memcpy(&valid_mask, valid->data, sizeof(valid_mask));
            std::uint64_t value_mask = 0;
            std::array<std::uint32_t, 64> true_values{};
            std::array<std::uint32_t, 64> evaluated_values{};
            const auto status = evaluate_logic_model(
                *model, binding_values, valid_mask, value_mask,
                true_values, evaluated_values);
            if (status != PTMRT_STATUS_OK) {
                return status;
            }
            std::memcpy(values->data, &value_mask, sizeof(value_mask));
            std::memcpy(true_masks->data, true_values.data(), diagnostic_bytes);
            std::memcpy(evaluated_masks->data, evaluated_values.data(),
                        diagnostic_bytes);
            return PTMRT_STATUS_OK;
        } catch (...) {
            return PTMRT_STATUS_INTERNAL_ERROR;
        }
    }
    if (input_count != 2 || output_count != 2) {
        return PTMRT_STATUS_INVALID_ARGUMENT;
    }
    const auto* features = find_tensor(inputs, input_count, "features");
    const auto* valid = find_tensor(inputs, input_count, "valid_mask");
    const auto* predictions = find_tensor(outputs, output_count, "predictions");
    const auto* scores = find_tensor(outputs, output_count, "scores");
    const auto feature_bytes =
        static_cast<std::uint64_t>(model->description.number_of_features) *
        sizeof(std::uint64_t);
    if (features == nullptr || valid == nullptr || predictions == nullptr ||
        scores == nullptr ||
        !valid_tensor(*features, PTMRT_DTYPE_UINT64, 1,
                      model->description.number_of_features, feature_bytes) ||
        !valid_tensor(*valid, PTMRT_DTYPE_UINT64, 0, 0,
                      sizeof(std::uint64_t)) ||
        !valid_tensor(*predictions, PTMRT_DTYPE_UINT64, 0, 0,
                      sizeof(std::uint64_t)) ||
        !valid_tensor(*scores, PTMRT_DTYPE_INT32, 1, 64,
                      64U * sizeof(std::int32_t))) {
        return PTMRT_STATUS_INVALID_ARGUMENT;
    }

    try {
        std::vector<std::uint64_t> feature_words(
            model->description.number_of_features);
        std::memcpy(feature_words.data(), features->data,
                    static_cast<std::size_t>(feature_bytes));
        std::uint64_t valid_mask = 0;
        std::memcpy(&valid_mask, valid->data, sizeof(valid_mask));
        std::uint64_t prediction_mask = 0;
        std::array<std::int32_t, 64> score_values{};
        const auto status = evaluate_packed_model(
            *model, feature_words, valid_mask, prediction_mask, score_values);
        if (status != PTMRT_STATUS_OK) {
            return status;
        }
        std::memcpy(predictions->data, &prediction_mask,
                    sizeof(prediction_mask));
        std::memcpy(scores->data, score_values.data(),
                    score_values.size() * sizeof(score_values[0]));
        return PTMRT_STATUS_OK;
    } catch (...) {
        return PTMRT_STATUS_INTERNAL_ERROR;
    }
}

}  // extern "C"
