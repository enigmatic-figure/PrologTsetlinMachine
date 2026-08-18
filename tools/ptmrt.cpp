#include "ptm/runtime.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct ModelCloser {
    void operator()(ptmrt_model* model) const noexcept {
        ptmrt_model_close(model);
    }
};

using Model = std::unique_ptr<ptmrt_model, ModelCloser>;

[[noreturn]] void fail(std::string_view operation, ptmrt_status status) {
    std::cerr << "ptmrt: " << operation << ": "
              << ptmrt_status_message(status) << '\n';
    std::exit(2);
}

[[nodiscard]] Model open_model(const char* path) {
    ptmrt_model* raw = nullptr;
    const auto status = ptmrt_model_open_file(path, &raw);
    if (status != PTMRT_STATUS_OK) {
        fail("open", status);
    }
    return Model(raw);
}

[[nodiscard]] ptmrt_model_description describe(const Model& model) {
    ptmrt_model_description result{};
    const auto status = ptmrt_model_describe(model.get(), &result);
    if (status != PTMRT_STATUS_OK) {
        fail("describe", status);
    }
    return result;
}

[[nodiscard]] const char* model_kind_name(std::uint32_t kind) {
    switch (kind) {
        case PTMRT_MODEL_PACKED_TM_BINARY_V1:
            return "packed_tm_binary_v1";
        case PTMRT_MODEL_LOGIC_PROGRAM32_V1:
            return "logic_program32_v1";
        case PTMRT_MODEL_MASKED_THRESHOLD_V1:
            return "masked_threshold_v1";
        default:
            return "unknown";
    }
}

[[nodiscard]] std::string manifest_json(const Model& model) {
    std::uint64_t required = 0;
    auto status = ptmrt_model_manifest_json(
        model.get(), nullptr, 0, &required);
    if (status != PTMRT_STATUS_OK || required == 0) {
        fail("manifest query", status);
    }
    std::vector<char> buffer(static_cast<std::size_t>(required));
    status = ptmrt_model_manifest_json(
        model.get(), buffer.data(), buffer.size(), &required);
    if (status != PTMRT_STATUS_OK) {
        fail("manifest read", status);
    }
    return std::string(buffer.data());
}

void inspect(const char* path) {
    const auto model = open_model(path);
    const auto description = describe(model);
    std::cout << "artifact_id=" << description.artifact_id << '\n'
              << "artifact_schema=" << description.artifact_schema << '\n'
              << "model_kind=" << model_kind_name(description.model_kind)
              << '\n'
              << "conformance_cases="
              << description.conformance_case_count << '\n';
    if (description.model_kind == PTMRT_MODEL_PACKED_TM_BINARY_V1) {
        std::cout << "clauses=" << description.number_of_clauses << '\n'
                  << "features=" << description.number_of_features << '\n'
                  << "threshold=" << description.threshold << '\n';
    } else if (description.model_kind == PTMRT_MODEL_LOGIC_PROGRAM32_V1) {
        std::cout << "instructions=" << description.instruction_count << '\n'
                  << "bindings=" << description.binding_count << '\n';
    } else if (description.model_kind == PTMRT_MODEL_MASKED_THRESHOLD_V1) {
        std::cout << "slots=" << description.slot_count << '\n'
                  << "minimum_true=" << description.minimum_true << '\n'
                  << "selected_slots=" << description.selected_count << '\n';
    }
    for (std::uint32_t index = 0; index < description.input_count; ++index) {
        const auto& port = description.inputs[index];
        std::cout << "input=" << port.name << ':' << port.semantic << '\n';
    }
    for (std::uint32_t index = 0; index < description.output_count; ++index) {
        const auto& port = description.outputs[index];
        std::cout << "output=" << port.name << ':' << port.semantic << '\n';
    }
    std::cout << "manifest=" << manifest_json(model) << '\n';
}

void verify(const char* path) {
    const auto model = open_model(path);
    const auto status = ptmrt_model_verify(model.get());
    if (status != PTMRT_STATUS_OK) {
        fail("verify", status);
    }
    const auto description = describe(model);
    std::cout << "verified " << description.artifact_id << " ("
              << description.conformance_case_count
              << " conformance cases)\n";
}

[[nodiscard]] std::vector<std::uint8_t> parse_binary_inputs(
    std::string_view text,
    std::uint32_t expected) {
    std::vector<std::uint8_t> result;
    std::stringstream stream{std::string(text)};
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item == "0") {
            result.push_back(0);
        } else if (item == "1") {
            result.push_back(1);
        } else {
            std::cerr << "ptmrt: inputs must be comma-separated 0/1 values\n";
            std::exit(2);
        }
    }
    if (result.size() != expected) {
        std::cerr << "ptmrt: expected " << expected << " inputs, received "
                  << result.size() << '\n';
        std::exit(2);
    }
    return result;
}

[[nodiscard]] std::vector<std::uint64_t> parse_active_slots(
    std::string_view text,
    std::uint32_t slot_count) {
    std::vector<std::uint64_t> result(slot_count, 0);
    if (text == "none") {
        return result;
    }
    if (text.empty() || text.front() == ',' || text.back() == ',' ||
        text.find(",,") != std::string_view::npos) {
        std::cerr << "ptmrt: active slots must be unique indices in [0, "
                  << slot_count << ") or 'none'\n";
        std::exit(2);
    }
    std::stringstream stream{std::string(text)};
    std::string item;
    while (std::getline(stream, item, ',')) {
        std::uint32_t slot = 0;
        const auto parsed = std::from_chars(
            item.data(), item.data() + item.size(), slot);
        if (item.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != item.data() + item.size() || slot >= slot_count) {
            std::cerr << "ptmrt: active slots must be unique indices in [0, "
                      << slot_count << ") or 'none'\n";
            std::exit(2);
        }
        if (result[slot] != 0) {
            std::cerr << "ptmrt: active slot indices must be unique\n";
            std::exit(2);
        }
        result[slot] = 1;
    }
    return result;
}

void print_active_slots(std::span<const std::uint64_t> words) {
    std::cout << '[';
    bool first = true;
    for (std::size_t slot = 0; slot < words.size(); ++slot) {
        if ((words[slot] & 1U) == 0) {
            continue;
        }
        if (!first) {
            std::cout << ',';
        }
        std::cout << slot;
        first = false;
    }
    std::cout << ']';
}

void run(const char* path, std::string_view input_text) {
    const auto model = open_model(path);
    const auto description = describe(model);
    const auto width = description.inputs[0].shape[0];
    std::vector<std::uint64_t> packed_words;
    if (description.model_kind == PTMRT_MODEL_MASKED_THRESHOLD_V1) {
        packed_words = parse_active_slots(
            input_text, static_cast<std::uint32_t>(width));
    } else {
        const auto values = parse_binary_inputs(
            input_text, static_cast<std::uint32_t>(width));
        packed_words.assign(values.begin(), values.end());
    }
    std::uint64_t valid_mask = 1;

    std::array<ptmrt_tensor_view, 2> inputs{};
    inputs[0].name = description.inputs[0].name;
    inputs[0].data = packed_words.data();
    inputs[0].byte_size = packed_words.size() * sizeof(std::uint64_t);
    inputs[0].dtype = PTMRT_DTYPE_UINT64;
    inputs[0].rank = 1;
    inputs[0].shape[0] = packed_words.size();
    inputs[1].name = "valid_mask";
    inputs[1].data = &valid_mask;
    inputs[1].byte_size = sizeof(valid_mask);
    inputs[1].dtype = PTMRT_DTYPE_UINT64;
    inputs[1].rank = 0;

    if (description.model_kind == PTMRT_MODEL_MASKED_THRESHOLD_V1) {
        std::uint64_t value_mask = 0;
        std::array<std::uint32_t, 64> matched_counts{};
        std::vector<std::uint64_t> matched_slots(packed_words.size());
        std::vector<std::uint64_t> missing_slots(packed_words.size());
        std::array<ptmrt_tensor_view, 4> outputs{};
        outputs[0] = {"values", &value_mask, sizeof(value_mask),
                      PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
        outputs[1] = {"matched_counts", matched_counts.data(),
                      sizeof(matched_counts), PTMRT_DTYPE_UINT32, 1,
                      {64, 0, 0, 0}};
        outputs[2] = {"matched_slots", matched_slots.data(),
                      matched_slots.size() * sizeof(std::uint64_t),
                      PTMRT_DTYPE_UINT64, 1,
                      {matched_slots.size(), 0, 0, 0}};
        outputs[3] = {"missing_slots", missing_slots.data(),
                      missing_slots.size() * sizeof(std::uint64_t),
                      PTMRT_DTYPE_UINT64, 1,
                      {missing_slots.size(), 0, 0, 0}};
        const auto status = ptmrt_model_run(
            model.get(), inputs.data(), inputs.size(), outputs.data(),
            outputs.size());
        if (status != PTMRT_STATUS_OK) {
            fail("run", status);
        }
        std::cout << "{\"artifact_id\":\"" << description.artifact_id
                  << "\",\"value\":" << (value_mask & 1U)
                  << ",\"matched_count\":" << matched_counts[0]
                  << ",\"matched_slots\":";
        print_active_slots(matched_slots);
        std::cout << ",\"missing_slots\":";
        print_active_slots(missing_slots);
        std::cout << "}\n";
        return;
    }

    if (description.model_kind == PTMRT_MODEL_PACKED_TM_BINARY_V1) {
        std::uint64_t prediction_mask = 0;
        std::array<std::int32_t, 64> scores{};
        std::array<ptmrt_tensor_view, 2> outputs{};
        outputs[0] = {"predictions", &prediction_mask,
                      sizeof(prediction_mask), PTMRT_DTYPE_UINT64, 0,
                      {0, 0, 0, 0}};
        outputs[1] = {"scores", scores.data(), sizeof(scores),
                      PTMRT_DTYPE_INT32, 1, {64, 0, 0, 0}};
        const auto status = ptmrt_model_run(
            model.get(), inputs.data(), inputs.size(), outputs.data(),
            outputs.size());
        if (status != PTMRT_STATUS_OK) {
            fail("run", status);
        }
        std::cout << "{\"artifact_id\":\"" << description.artifact_id
                  << "\",\"prediction\":" << (prediction_mask & 1U)
                  << ",\"score\":" << scores[0] << "}\n";
        return;
    }

    if (description.model_kind == PTMRT_MODEL_LOGIC_PROGRAM32_V1) {
        std::uint64_t value_mask = 0;
        std::array<std::uint32_t, 64> true_masks{};
        std::array<std::uint32_t, 64> evaluated_masks{};
        std::array<ptmrt_tensor_view, 3> outputs{};
        outputs[0] = {"values", &value_mask, sizeof(value_mask),
                      PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
        outputs[1] = {"true_instruction_masks", true_masks.data(),
                      sizeof(true_masks), PTMRT_DTYPE_UINT32, 1,
                      {64, 0, 0, 0}};
        outputs[2] = {"evaluated_instruction_masks", evaluated_masks.data(),
                      sizeof(evaluated_masks), PTMRT_DTYPE_UINT32, 1,
                      {64, 0, 0, 0}};
        const auto status = ptmrt_model_run(
            model.get(), inputs.data(), inputs.size(), outputs.data(),
            outputs.size());
        if (status != PTMRT_STATUS_OK) {
            fail("run", status);
        }
        std::cout << "{\"artifact_id\":\"" << description.artifact_id
                  << "\",\"value\":" << (value_mask & 1U)
                  << ",\"true_instruction_mask\":" << true_masks[0]
                  << ",\"evaluated_instruction_mask\":"
                  << evaluated_masks[0] << "}\n";
        return;
    }
    fail("run", PTMRT_STATUS_UNSUPPORTED_MODEL);
}

struct OwnedRecordField {
    std::string name;
    std::string text;
    std::uint32_t kind{};
    std::uint32_t boolean_value{};
    std::int64_t integer_value{};
    double number_value{};
};

[[nodiscard]] OwnedRecordField parse_record_field(std::string_view argument) {
    const auto colon = argument.find(':');
    if (colon == std::string_view::npos || colon == 0) {
        std::cerr << "ptmrt: record fields use NAME:TYPE=VALUE\n";
        std::exit(2);
    }
    OwnedRecordField result{};
    result.name = argument.substr(0, colon);
    const auto specification = argument.substr(colon + 1);
    if (specification == "null") {
        result.kind = PTMRT_VALUE_NULL;
        return result;
    }
    const auto equals = specification.find('=');
    if (equals == std::string_view::npos) {
        std::cerr << "ptmrt: non-null fields require NAME:TYPE=VALUE\n";
        std::exit(2);
    }
    const auto type = specification.substr(0, equals);
    const auto value = specification.substr(equals + 1);
    if (type == "bool") {
        result.kind = PTMRT_VALUE_BOOL;
        if (value == "true") result.boolean_value = 1;
        else if (value == "false") result.boolean_value = 0;
        else {
            std::cerr << "ptmrt: bool values must be true or false\n";
            std::exit(2);
        }
    } else if (type == "int") {
        result.kind = PTMRT_VALUE_INT64;
        const auto parsed = std::from_chars(
            value.data(), value.data() + value.size(), result.integer_value);
        if (value.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != value.data() + value.size()) {
            std::cerr << "ptmrt: int values must be signed 64-bit integers\n";
            std::exit(2);
        }
    } else if (type == "float") {
        result.kind = PTMRT_VALUE_FLOAT64;
        const auto parsed = std::from_chars(
            value.data(), value.data() + value.size(), result.number_value);
        if (value.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != value.data() + value.size() ||
            !std::isfinite(result.number_value)) {
            std::cerr << "ptmrt: float values must be finite numbers\n";
            std::exit(2);
        }
    } else if (type == "string") {
        result.kind = PTMRT_VALUE_UTF8;
        result.text = value;
    } else {
        std::cerr << "ptmrt: field type must be bool, int, float, string, or null\n";
        std::exit(2);
    }
    return result;
}

void run_record(const char* path, int field_count, char** field_arguments) {
    const auto model = open_model(path);
    const auto description = describe(model);
    if (description.model_kind != PTMRT_MODEL_PACKED_TM_BINARY_V1 ||
        ptmrt_model_has_preprocessing(model.get()) == 0) {
        fail("run-record", PTMRT_STATUS_UNSUPPORTED_MODEL);
    }
    if (field_count < 0 || field_count > 4096) {
        fail("run-record", PTMRT_STATUS_INVALID_ARGUMENT);
    }
    std::vector<OwnedRecordField> owned;
    owned.reserve(static_cast<std::size_t>(field_count));
    for (int index = 0; index < field_count; ++index) {
        owned.push_back(parse_record_field(field_arguments[index]));
    }
    std::vector<ptmrt_record_field> fields;
    fields.reserve(owned.size());
    for (const auto& item : owned) {
        fields.push_back({
            item.name.c_str(), item.kind, item.boolean_value,
            item.integer_value, item.number_value,
            item.kind == PTMRT_VALUE_UTF8 ? item.text.data() : nullptr,
            item.kind == PTMRT_VALUE_UTF8 ? item.text.size() : 0U});
    }
    std::uint64_t required = 0;
    auto status = ptmrt_model_preprocess_record(
        model.get(), fields.data(), static_cast<std::uint32_t>(fields.size()),
        nullptr, 0, &required);
    if (status != PTMRT_STATUS_OK) fail("preprocess query", status);
    std::vector<std::uint64_t> words(static_cast<std::size_t>(required));
    status = ptmrt_model_preprocess_record(
        model.get(), fields.data(), static_cast<std::uint32_t>(fields.size()),
        words.data(), words.size(), &required);
    if (status != PTMRT_STATUS_OK) fail("preprocess", status);

    std::uint64_t valid_mask = 1;
    std::array<ptmrt_tensor_view, 2> inputs{};
    inputs[0] = {description.inputs[0].name, words.data(),
                 words.size() * sizeof(std::uint64_t), PTMRT_DTYPE_UINT64, 1,
                 {words.size(), 0, 0, 0}};
    inputs[1] = {"valid_mask", &valid_mask, sizeof(valid_mask),
                 PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    std::uint64_t prediction_mask = 0;
    std::array<std::int32_t, 64> scores{};
    std::array<ptmrt_tensor_view, 2> outputs{};
    outputs[0] = {"predictions", &prediction_mask, sizeof(prediction_mask),
                  PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    outputs[1] = {"scores", scores.data(), sizeof(scores),
                  PTMRT_DTYPE_INT32, 1, {64, 0, 0, 0}};
    status = ptmrt_model_run(
        model.get(), inputs.data(), inputs.size(), outputs.data(),
        outputs.size());
    if (status != PTMRT_STATUS_OK) fail("run", status);
    std::cout << "{\"artifact_id\":\"" << description.artifact_id
              << "\",\"features\":[";
    for (std::size_t index = 0; index < words.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << words[index];
    }
    std::cout << "],\"prediction\":" << (prediction_mask & 1U)
              << ",\"score\":" << scores[0] << "}\n";
}

void print_help() {
    std::cout << "PTM static inference runtime\n\n"
              << "Usage:\n"
              << "  ptmrt inspect MODEL.ptm\n"
              << "  ptmrt verify MODEL.ptm\n"
              << "  ptmrt run MODEL.ptm INPUT0,INPUT1,...\n"
              << "  ptmrt run-record MODEL.ptm [NAME:TYPE=VALUE ...]\n\n"
              << "Packed-TM and Logic inputs are binary values. Masked-threshold\n"
              << "inputs are active slot indices, or 'none'. Raw field TYPE is\n"
              << "bool, int, float, string, or null (written NAME:null).\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                      std::strcmp(argv[1], "-h") == 0)) {
        print_help();
        return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "inspect") == 0) {
        inspect(argv[2]);
        return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "verify") == 0) {
        verify(argv[2]);
        return 0;
    }
    if (argc == 4 && std::strcmp(argv[1], "run") == 0) {
        run(argv[2], argv[3]);
        return 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "run-record") == 0) {
        run_record(argv[2], argc - 3, argv + 3);
        return 0;
    }
    print_help();
    return 2;
}
