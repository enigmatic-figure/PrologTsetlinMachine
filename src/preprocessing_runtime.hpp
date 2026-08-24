#ifndef PTM_PREPROCESSING_RUNTIME_HPP
#define PTM_PREPROCESSING_RUNTIME_HPP

#include "ptm/runtime.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <variant>
#include <vector>

namespace ptm::runtime_detail {

struct JsonValue {
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue, std::less<>>;
    std::variant<std::nullptr_t, bool, std::int64_t, double,
                 std::string, Array, Object> value;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view source) : source_(source) {}

    [[nodiscard]] std::optional<JsonValue> parse() {
        auto result = value(0);
        if (!result || position_ != source_.size()) {
            return std::nullopt;
        }
        return result;
    }

private:
    [[nodiscard]] std::optional<JsonValue> value(unsigned depth) {
        if (depth > PTMRT_MODEL_MANIFEST_MAX_DEPTH ||
            position_ >= source_.size() ||
            ++nodes_ > PTMRT_MODEL_MANIFEST_MAX_NODES) {
            return std::nullopt;
        }
        const auto lead = source_[position_];
        if (lead == '"') {
            auto text = string();
            return text ? std::optional<JsonValue>{JsonValue{*text}}
                        : std::nullopt;
        }
        if (lead == '{') {
            return object(depth + 1);
        }
        if (lead == '[') {
            return array(depth + 1);
        }
        if (source_.substr(position_, 4) == "true") {
            position_ += 4;
            return JsonValue{true};
        }
        if (source_.substr(position_, 5) == "false") {
            position_ += 5;
            return JsonValue{false};
        }
        if (source_.substr(position_, 4) == "null") {
            position_ += 4;
            return JsonValue{nullptr};
        }
        return number();
    }

    [[nodiscard]] std::optional<JsonValue> object(unsigned depth) {
        ++position_;
        JsonValue::Object result;
        if (consume('}')) {
            return JsonValue{std::move(result)};
        }
        while (true) {
            auto key = string();
            if (!key || !consume(':')) {
                return std::nullopt;
            }
            auto item = value(depth);
            if (!item || !result.emplace(std::move(*key), std::move(*item)).second) {
                return std::nullopt;
            }
            if (consume('}')) {
                return JsonValue{std::move(result)};
            }
            if (!consume(',')) {
                return std::nullopt;
            }
        }
    }

    [[nodiscard]] std::optional<JsonValue> array(unsigned depth) {
        ++position_;
        JsonValue::Array result;
        if (consume(']')) {
            return JsonValue{std::move(result)};
        }
        while (true) {
            auto item = value(depth);
            if (!item) {
                return std::nullopt;
            }
            result.push_back(std::move(*item));
            if (consume(']')) {
                return JsonValue{std::move(result)};
            }
            if (!consume(',')) {
                return std::nullopt;
            }
        }
    }

    [[nodiscard]] std::optional<std::string> string() {
        if (!consume('"')) {
            return std::nullopt;
        }
        std::string result;
        while (position_ < source_.size()) {
            const auto value = source_[position_++];
            if (value == '"') {
                return result;
            }
            if (static_cast<unsigned char>(value) < 0x20U) {
                return std::nullopt;
            }
            if (value != '\\') {
                result.push_back(value);
                continue;
            }
            if (position_ >= source_.size()) {
                return std::nullopt;
            }
            switch (source_[position_++]) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default: return std::nullopt;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<JsonValue> number() {
        const auto first = position_;
        if (position_ < source_.size() && source_[position_] == '-') {
            ++position_;
        }
        if (position_ >= source_.size() || source_[position_] < '0' ||
            source_[position_] > '9') {
            return std::nullopt;
        }
        if (source_[position_] == '0') {
            ++position_;
        } else {
            while (position_ < source_.size() && source_[position_] >= '0' &&
                   source_[position_] <= '9') {
                ++position_;
            }
        }
        bool floating = false;
        if (position_ < source_.size() && source_[position_] == '.') {
            floating = true;
            ++position_;
            const auto fraction = position_;
            while (position_ < source_.size() && source_[position_] >= '0' &&
                   source_[position_] <= '9') {
                ++position_;
            }
            if (fraction == position_) {
                return std::nullopt;
            }
        }
        if (position_ < source_.size() &&
            (source_[position_] == 'e' || source_[position_] == 'E')) {
            floating = true;
            ++position_;
            if (position_ < source_.size() &&
                (source_[position_] == '+' || source_[position_] == '-')) {
                ++position_;
            }
            const auto exponent = position_;
            while (position_ < source_.size() && source_[position_] >= '0' &&
                   source_[position_] <= '9') {
                ++position_;
            }
            if (exponent == position_) {
                return std::nullopt;
            }
        }
        const auto text = source_.substr(first, position_ - first);
        if (!floating) {
            std::int64_t parsed = 0;
            const auto result = std::from_chars(
                text.data(), text.data() + text.size(), parsed);
            if (result.ec == std::errc{} && result.ptr == text.data() + text.size()) {
                return JsonValue{parsed};
            }
            return std::nullopt;
        }
        double parsed = 0;
        const auto result = std::from_chars(
            text.data(), text.data() + text.size(), parsed);
        if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
            !std::isfinite(parsed)) {
            return std::nullopt;
        }
        return JsonValue{parsed};
    }

    bool consume(char expected) {
        if (position_ >= source_.size() || source_[position_] != expected) {
            return false;
        }
        ++position_;
        return true;
    }

    std::string_view source_;
    std::size_t position_{};
    std::size_t nodes_{};
};

enum class FieldKind { number, category, boolean };
enum class Transform { numeric_ge, numeric_between, category_eq, category_in,
                       is_missing };
enum class NullPolicy { false_value, true_value, error };

struct CategoryValue {
    std::variant<bool, std::int64_t, std::string> value;
};

struct PreprocessingOutput {
    std::string field;
    std::string field_id;
    std::string literal_id;
    FieldKind field_kind{};
    Transform transform{};
    NullPolicy null_policy{};
    double threshold{};
    double lower{};
    double upper{};
    bool inclusive_lower{};
    bool inclusive_upper{};
    std::vector<CategoryValue> categories;
};

[[nodiscard]] inline bool portable_text(std::string_view value) {
    for (const auto character : value) {
        if (static_cast<unsigned char>(character) < 0x20U) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline bool decimal_u64(std::string_view value) {
    if (value.empty() || (value.size() > 1 && value.front() == '0')) {
        return false;
    }
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(
        value.data(), value.data() + value.size(), parsed);
    return result.ec == std::errc{} &&
           result.ptr == value.data() + value.size();
}

template <typename T>
[[nodiscard]] const T* as(const JsonValue& value) {
    return std::get_if<T>(&value.value);
}

[[nodiscard]] inline const JsonValue* member(
    const JsonValue::Object& object, std::string_view name) {
    const auto found = object.find(name);
    return found == object.end() ? nullptr : &found->second;
}

[[nodiscard]] inline std::optional<double> json_number(const JsonValue& value) {
    if (const auto* integer = as<std::int64_t>(value)) {
        if (*integer < -(std::int64_t{1} << 53) ||
            *integer > (std::int64_t{1} << 53)) {
            return std::nullopt;
        }
        return static_cast<double>(*integer);
    }
    if (const auto* number = as<double>(value); number && std::isfinite(*number)) {
        return *number;
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<CategoryValue> category_value(
    const JsonValue& value) {
    if (const auto* boolean = as<bool>(value)) {
        return CategoryValue{*boolean};
    }
    if (const auto* integer = as<std::int64_t>(value)) {
        return CategoryValue{*integer};
    }
    if (const auto* text = as<std::string>(value)) {
        return CategoryValue{*text};
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::size_t> top_level_member(
    std::string_view manifest, std::string_view marker) {
    unsigned depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t position = 0; position < manifest.size(); ++position) {
        const auto character = manifest[position];
        if (in_string) {
            if (escaped) escaped = false;
            else if (character == '\\') escaped = true;
            else if (character == '"') in_string = false;
            continue;
        }
        if (character == '"') {
            if (depth == 1 && manifest.substr(position, marker.size()) == marker) {
                return position;
            }
            in_string = true;
        } else if (character == '{') {
            ++depth;
        } else if (character == '}') {
            if (depth == 0) return std::nullopt;
            --depth;
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::string_view> member_object(
    std::string_view manifest, std::string_view marker) {
    const auto position = top_level_member(manifest, marker);
    if (!position) return std::nullopt;
    const auto start = *position + marker.size();
    if (start >= manifest.size() || manifest[start] != '{') {
        return std::nullopt;
    }
    std::size_t end = start;
    unsigned depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (; end < manifest.size(); ++end) {
        const auto character = manifest[end];
        if (in_string) {
            if (escaped) escaped = false;
            else if (character == '\\') escaped = true;
            else if (character == '"') in_string = false;
            continue;
        }
        if (character == '"') in_string = true;
        else if (character == '{') ++depth;
        else if (character == '}') {
            if (depth == 0) return std::nullopt;
            if (--depth == 0) {
                ++end;
                return manifest.substr(start, end - start);
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::string> feature_materialization(
    std::string_view manifest) {
    const auto source = member_object(manifest, "\"features\":");
    if (!source) return std::nullopt;
    auto parsed = JsonParser(*source).parse();
    const auto* object = parsed ? as<JsonValue::Object>(*parsed) : nullptr;
    const auto* raw = object ? member(*object, "materialization") : nullptr;
    const auto* value = raw ? as<std::string>(*raw) : nullptr;
    return value ? std::optional<std::string>{*value} : std::nullopt;
}

[[nodiscard]] inline std::optional<std::vector<std::string>> feature_literal_ids(
    std::string_view manifest) {
    const auto source = member_object(manifest, "\"features\":");
    if (!source) return std::nullopt;
    auto parsed = JsonParser(*source).parse();
    const auto* object = parsed ? as<JsonValue::Object>(*parsed) : nullptr;
    const auto* raw = object ? member(*object, "literal_ids") : nullptr;
    const auto* values = raw ? as<JsonValue::Array>(*raw) : nullptr;
    if (!values) return std::nullopt;
    std::vector<std::string> result;
    result.reserve(values->size());
    for (const auto& value : *values) {
        const auto* text = as<std::string>(value);
        if (!text || !decimal_u64(*text)) return std::nullopt;
        result.push_back(*text);
    }
    return result;
}

[[nodiscard]] inline bool parse_preprocessing(
    std::string_view manifest,
    std::uint32_t expected_outputs,
    std::vector<PreprocessingOutput>& result) {
    constexpr std::string_view marker = "\"preprocessing\":";
    const auto marker_position = top_level_member(manifest, marker);
    if (!marker_position) {
        result.clear();
        return true;
    }
    const auto source = member_object(manifest, marker);
    if (!source) return false;
    JsonParser contract_parser(*source);
    auto root = contract_parser.parse();
    const auto* object = root ? as<JsonValue::Object>(*root) : nullptr;
    if (!object || object->size() != 2) {
        return false;
    }
    const auto* schema_value = member(*object, "schema");
    const auto* outputs_value = member(*object, "outputs");
    const auto* schema = schema_value ? as<std::string>(*schema_value) : nullptr;
    const auto* outputs = outputs_value ? as<JsonValue::Array>(*outputs_value) : nullptr;
    if (!schema || *schema != "ptm.preprocessing.v1" || !outputs ||
        outputs->size() != expected_outputs || outputs->size() > 4096) {
        return false;
    }
    result.clear();
    result.reserve(outputs->size());
    for (const auto& raw_output : *outputs) {
        const auto* item = as<JsonValue::Object>(raw_output);
        if (!item || item->size() != 7) return false;
        const auto* field_value = member(*item, "field");
        const auto* kind_value = member(*item, "field_kind");
        const auto* transform_value = member(*item, "transform");
        const auto* null_value = member(*item, "null_policy");
        const auto* parameters_value = member(*item, "parameters");
        const auto* field_id_value = member(*item, "field_id");
        const auto* literal_id_value = member(*item, "literal_id");
        const auto* field = field_value ? as<std::string>(*field_value) : nullptr;
        const auto* kind = kind_value ? as<std::string>(*kind_value) : nullptr;
        const auto* transform = transform_value ? as<std::string>(*transform_value) : nullptr;
        const auto* null_policy = null_value ? as<std::string>(*null_value) : nullptr;
        const auto* parameters = parameters_value ? as<JsonValue::Object>(*parameters_value) : nullptr;
        const auto* field_id = field_id_value
                                   ? as<std::string>(*field_id_value) : nullptr;
        const auto* literal_id = literal_id_value
                                     ? as<std::string>(*literal_id_value) : nullptr;
        if (!field || field->empty() || !portable_text(*field) || !kind ||
            !transform || !null_policy || !parameters || !field_id ||
            !literal_id || !decimal_u64(*field_id) ||
            !decimal_u64(*literal_id)) {
            return false;
        }
        PreprocessingOutput output{};
        output.field = *field;
        output.field_id = *field_id;
        output.literal_id = *literal_id;
        if (*kind == "number") output.field_kind = FieldKind::number;
        else if (*kind == "category") output.field_kind = FieldKind::category;
        else if (*kind == "boolean") output.field_kind = FieldKind::boolean;
        else return false;
        if (*null_policy == "false") output.null_policy = NullPolicy::false_value;
        else if (*null_policy == "true") output.null_policy = NullPolicy::true_value;
        else if (*null_policy == "error") output.null_policy = NullPolicy::error;
        else return false;

        if (*transform == "numeric_ge") {
            output.transform = Transform::numeric_ge;
            const auto* threshold = member(*parameters, "threshold");
            const auto parsed = threshold ? json_number(*threshold) : std::nullopt;
            if (parameters->size() != 1 || !parsed) return false;
            output.threshold = *parsed;
        } else if (*transform == "numeric_between") {
            output.transform = Transform::numeric_between;
            const auto* lower = member(*parameters, "lower");
            const auto* upper = member(*parameters, "upper");
            const auto* include_lower = member(*parameters, "inclusive_lower");
            const auto* include_upper = member(*parameters, "inclusive_upper");
            if (parameters->size() != 4 || !lower || !upper ||
                !include_lower || !include_upper) return false;
            const auto parsed_lower = json_number(*lower);
            const auto parsed_upper = json_number(*upper);
            const auto* lower_flag = as<bool>(*include_lower);
            const auto* upper_flag = as<bool>(*include_upper);
            if (!parsed_lower.has_value() || !parsed_upper.has_value() ||
                !lower_flag || !upper_flag) return false;
            const auto lower_number = parsed_lower.value();
            const auto upper_number = parsed_upper.value();
            if (lower_number > upper_number) return false;
            output.lower = lower_number;
            output.upper = upper_number;
            output.inclusive_lower = *lower_flag;
            output.inclusive_upper = *upper_flag;
        } else if (*transform == "category_eq") {
            output.transform = Transform::category_eq;
            const auto* raw = member(*parameters, "value");
            auto parsed = raw ? category_value(*raw) : std::nullopt;
            if (parameters->size() != 1 || !parsed) return false;
            output.categories.push_back(std::move(*parsed));
        } else if (*transform == "category_in") {
            output.transform = Transform::category_in;
            const auto* raw = member(*parameters, "values");
            const auto* values = raw ? as<JsonValue::Array>(*raw) : nullptr;
            if (parameters->size() != 1 || !values || values->empty()) return false;
            for (const auto& value : *values) {
                auto parsed = category_value(value);
                if (!parsed) return false;
                output.categories.push_back(std::move(*parsed));
            }
        } else if (*transform == "is_missing") {
            output.transform = Transform::is_missing;
            if (!parameters->empty() ||
                output.null_policy != NullPolicy::false_value) return false;
        } else return false;
        const auto numeric_transform =
            output.transform == Transform::numeric_ge ||
            output.transform == Transform::numeric_between;
        const auto category_transform =
            output.transform == Transform::category_eq ||
            output.transform == Transform::category_in;
        if ((numeric_transform && output.field_kind != FieldKind::number) ||
            (category_transform && output.field_kind == FieldKind::number)) {
            return false;
        }
        for (const auto& category : output.categories) {
            if (output.field_kind == FieldKind::boolean &&
                !std::holds_alternative<bool>(category.value)) return false;
            if (const auto* text = std::get_if<std::string>(&category.value);
                text && !portable_text(*text)) return false;
        }
        for (const auto& existing : result) {
            if (existing.literal_id == output.literal_id) return false;
            if (existing.field == output.field &&
                (existing.field_id != output.field_id ||
                 existing.field_kind != output.field_kind)) return false;
        }
        result.push_back(std::move(output));
    }
    return true;
}

[[nodiscard]] inline bool category_matches(
    const CategoryValue& expected, const ptmrt_record_field& actual) {
    if (const auto* value = std::get_if<bool>(&expected.value)) {
        return actual.kind == PTMRT_VALUE_BOOL &&
               (actual.boolean_value != 0) == *value;
    }
    if (const auto* value = std::get_if<std::int64_t>(&expected.value)) {
        return actual.kind == PTMRT_VALUE_INT64 && actual.integer_value == *value;
    }
    const auto* value = std::get_if<std::string>(&expected.value);
    return value && actual.kind == PTMRT_VALUE_UTF8 &&
           actual.string_data != nullptr &&
           std::string_view(actual.string_data,
                            static_cast<std::size_t>(actual.string_size)) == *value;
}

[[nodiscard]] inline bool valid_utf8(std::string_view value) {
    for (std::size_t index = 0; index < value.size();) {
        const auto lead = static_cast<unsigned char>(value[index]);
        if (lead < 0x80U) {
            ++index;
            continue;
        }
        std::size_t length = 0;
        std::uint32_t point = 0;
        if ((lead & 0xE0U) == 0xC0U) { length = 2; point = lead & 0x1FU; }
        else if ((lead & 0xF0U) == 0xE0U) { length = 3; point = lead & 0x0FU; }
        else if ((lead & 0xF8U) == 0xF0U) { length = 4; point = lead & 0x07U; }
        else return false;
        if (index + length > value.size()) return false;
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto next = static_cast<unsigned char>(value[index + offset]);
            if ((next & 0xC0U) != 0x80U) return false;
            point = (point << 6U) | (next & 0x3FU);
        }
        if ((length == 2 && point < 0x80U) ||
            (length == 3 && point < 0x800U) ||
            (length == 4 && point < 0x10000U) ||
            point > 0x10FFFFU || (point >= 0xD800U && point <= 0xDFFFU)) {
            return false;
        }
        index += length;
    }
    return true;
}

[[nodiscard]] inline bool valid_field_value(
    FieldKind kind, const ptmrt_record_field& field) {
    if (kind == FieldKind::number) {
        return (field.kind == PTMRT_VALUE_INT64 &&
                field.integer_value >= -(std::int64_t{1} << 53) &&
                field.integer_value <= (std::int64_t{1} << 53)) ||
               (field.kind == PTMRT_VALUE_FLOAT64 &&
                std::isfinite(field.number_value));
    }
    if (kind == FieldKind::boolean) {
        return field.kind == PTMRT_VALUE_BOOL && field.boolean_value <= 1U;
    }
    if (field.kind == PTMRT_VALUE_BOOL) return field.boolean_value <= 1U;
    if (field.kind == PTMRT_VALUE_INT64) return true;
    if (field.kind != PTMRT_VALUE_UTF8 || field.string_data == nullptr ||
        field.string_size > static_cast<std::uint64_t>(SIZE_MAX)) return false;
    const std::string_view text(
        field.string_data, static_cast<std::size_t>(field.string_size));
    return valid_utf8(text) && portable_text(text);
}

[[nodiscard]] inline bool materialize_preprocessing(
    const std::vector<PreprocessingOutput>& outputs,
    const ptmrt_record_field* fields,
    std::uint32_t field_count,
    std::uint64_t* words) {
    if (field_count > 4096) return false;
    for (std::uint32_t left = 0; left < field_count; ++left) {
        if (fields[left].name == nullptr) return false;
        for (std::uint32_t right = left + 1; right < field_count; ++right) {
            if (fields[right].name == nullptr ||
                std::string_view(fields[left].name) == fields[right].name) return false;
        }
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        const auto& output = outputs[index];
        const ptmrt_record_field* field = nullptr;
        for (std::uint32_t item = 0; item < field_count; ++item) {
            if (std::string_view(fields[item].name) == output.field) {
                field = &fields[item];
                break;
            }
        }
        const auto missing = field == nullptr || field->kind == PTMRT_VALUE_NULL;
        if (!missing && !valid_field_value(output.field_kind, *field)) {
            return false;
        }
        if (output.transform == Transform::is_missing) {
            words[index] = missing ? 1U : 0U;
            continue;
        }
        if (missing) {
            if (output.null_policy == NullPolicy::error) return false;
            words[index] = output.null_policy == NullPolicy::true_value ? 1U : 0U;
            continue;
        }
        bool matched = false;
        if (output.field_kind == FieldKind::number) {
            double value = 0;
            if (field->kind == PTMRT_VALUE_INT64) {
                if (field->integer_value < -(std::int64_t{1} << 53) ||
                    field->integer_value > (std::int64_t{1} << 53)) return false;
                value = static_cast<double>(field->integer_value);
            } else if (field->kind == PTMRT_VALUE_FLOAT64 &&
                       std::isfinite(field->number_value)) {
                value = field->number_value;
            } else return false;
            if (output.transform == Transform::numeric_ge) {
                matched = value >= output.threshold;
            } else if (output.transform == Transform::numeric_between) {
                const auto lower = output.inclusive_lower ? value >= output.lower
                                                          : value > output.lower;
                const auto upper = output.inclusive_upper ? value <= output.upper
                                                          : value < output.upper;
                matched = lower && upper;
            } else return false;
        } else if (output.field_kind == FieldKind::boolean) {
            if (field->kind != PTMRT_VALUE_BOOL) return false;
            if (output.transform != Transform::category_eq &&
                output.transform != Transform::category_in) return false;
            for (const auto& expected : output.categories) {
                matched = matched || category_matches(expected, *field);
            }
        } else {
            if (field->kind != PTMRT_VALUE_BOOL &&
                field->kind != PTMRT_VALUE_INT64 &&
                field->kind != PTMRT_VALUE_UTF8) return false;
            if (output.transform != Transform::category_eq &&
                output.transform != Transform::category_in) return false;
            for (const auto& expected : output.categories) {
                matched = matched || category_matches(expected, *field);
            }
        }
        words[index] = matched ? 1U : 0U;
    }
    return true;
}

}  // namespace ptm::runtime_detail

#endif
