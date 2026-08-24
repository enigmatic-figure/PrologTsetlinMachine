#include "ptm/runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

// Minimal SHA-256 for rehashing hostile fixtures (duplicate of runtime, test-only)
[[nodiscard]] std::array<std::uint8_t, 32> sha256_test(std::span<const std::uint8_t> input) {
    constexpr std::array<std::uint32_t, 64> k{
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
    std::array<std::uint32_t, 8> state{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::vector<std::uint8_t> padded(input.begin(), input.end());
    const auto bit_len = static_cast<std::uint64_t>(input.size()) * 8U;
    padded.push_back(0x80U);
    while ((padded.size() % 64U) != 56U) padded.push_back(0U);
    for (int shift = 56; shift >= 0; shift -= 8) padded.push_back(static_cast<std::uint8_t>(bit_len >> shift));
    for (std::size_t off = 0; off < padded.size(); off += 64U) {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16U; ++i) {
            const auto b = off + i * 4U;
            w[i] = (static_cast<std::uint32_t>(padded[b]) << 24U) | (static_cast<std::uint32_t>(padded[b+1]) << 16U) | (static_cast<std::uint32_t>(padded[b+2]) << 8U) | static_cast<std::uint32_t>(padded[b+3]);
        }
        for (std::size_t i = 16U; i < 64U; ++i) {
            const auto s0 = std::rotr(w[i-15],7) ^ std::rotr(w[i-15],18) ^ (w[i-15]>>3U);
            const auto s1 = std::rotr(w[i-2],17) ^ std::rotr(w[i-2],19) ^ (w[i-2]>>10U);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        auto a=state[0], b=state[1], c=state[2], d=state[3], e=state[4], f=state[5], g=state[6], h=state[7];
        for (std::size_t i=0;i<64U;++i){
            const auto S1 = std::rotr(e,6) ^ std::rotr(e,11) ^ std::rotr(e,25);
            const auto ch = (e & f) ^ (~e & g);
            const auto temp1 = h + S1 + ch + k[i] + w[i];
            const auto S0 = std::rotr(a,2) ^ std::rotr(a,13) ^ std::rotr(a,22);
            const auto maj = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = S0 + maj;
            h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d; state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    std::array<std::uint8_t,32> out{};
    for (std::size_t i=0;i<8U;++i){
        out[i*4]=static_cast<std::uint8_t>(state[i]>>24U);
        out[i*4+1]=static_cast<std::uint8_t>(state[i]>>16U);
        out[i*4+2]=static_cast<std::uint8_t>(state[i]>>8U);
        out[i*4+3]=static_cast<std::uint8_t>(state[i]);
    }
    return out;
}

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::vector<std::uint8_t> read_fixture(std::string_view filename) {
    std::ifstream stream(
        std::string(PTM_TEST_DATA_DIR) + "/" + std::string(filename));
    require(static_cast<bool>(stream), "could not open model artifact fixture");
    std::string text((std::istreambuf_iterator<char>(stream)),
                     std::istreambuf_iterator<char>());
    std::string digits;
    for (const auto value : text) {
        if (!std::isspace(static_cast<unsigned char>(value))) {
            digits.push_back(value);
        }
    }
    require(digits.size() % 2U == 0, "artifact fixture has odd hex length");
    std::vector<std::uint8_t> result;
    result.reserve(digits.size() / 2U);
    const auto nibble = [](char value) -> unsigned {
        if (value >= '0' && value <= '9') {
            return static_cast<unsigned>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return 10U + static_cast<unsigned>(value - 'a');
        }
        if (value >= 'A' && value <= 'F') {
            return 10U + static_cast<unsigned>(value - 'A');
        }
        throw std::runtime_error("artifact fixture contains non-hex text");
    };
    for (std::size_t index = 0; index < digits.size(); index += 2U) {
        result.push_back(static_cast<std::uint8_t>(
            (nibble(digits[index]) << 4U) | nibble(digits[index + 1U])));
    }
    return result;
}

std::vector<std::uint8_t> read_golden() {
    return read_fixture("xor_packed_tm_v1.hex");
}

[[nodiscard]] std::uint64_t read_u64_test(
    const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    require(offset + sizeof(std::uint64_t) <= bytes.size(),
            "test fixture uint64 read is out of bounds");
    std::uint64_t result = 0;
    for (std::size_t byte = 0; byte < sizeof(result); ++byte) {
        result |= static_cast<std::uint64_t>(bytes[offset + byte])
                  << (byte * 8U);
    }
    return result;
}

void write_u64_test(std::vector<std::uint8_t>& bytes, std::size_t offset,
                    std::uint64_t value) {
    require(offset + sizeof(value) <= bytes.size(),
            "test fixture uint64 write is out of bounds");
    for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
        bytes[offset + byte] =
            static_cast<std::uint8_t>(value >> (byte * 8U));
    }
}

[[nodiscard]] std::string fixture_manifest(
    const std::vector<std::uint8_t>& artifact) {
    const auto manifest_size = read_u64_test(artifact, 24U);
    require(manifest_size <= artifact.size() - 64U - 32U,
            "test fixture manifest is out of bounds");
    return std::string(
        artifact.begin() + 64,
        artifact.begin() + 64 + static_cast<std::ptrdiff_t>(manifest_size));
}

[[nodiscard]] std::vector<std::uint8_t> replace_fixture_manifest(
    const std::vector<std::uint8_t>& artifact, std::string_view manifest) {
    const auto old_manifest_size = read_u64_test(artifact, 24U);
    const auto payload_size = read_u64_test(artifact, 32U);
    const auto payload_offset = 64U + old_manifest_size;
    require(payload_offset + payload_size + 32U == artifact.size(),
            "test fixture sections are inconsistent");

    std::vector<std::uint8_t> result;
    result.reserve(64U + manifest.size() +
                   static_cast<std::size_t>(payload_size) + 32U);
    result.insert(result.end(), artifact.begin(), artifact.begin() + 64);
    result.insert(result.end(), manifest.begin(), manifest.end());
    result.insert(
        result.end(),
        artifact.begin() + static_cast<std::ptrdiff_t>(payload_offset),
        artifact.begin() +
            static_cast<std::ptrdiff_t>(payload_offset + payload_size));
    result.resize(result.size() + 32U);
    write_u64_test(result, 24U, manifest.size());
    const auto digest = sha256_test(std::span<const std::uint8_t>(
        result.data(), result.size() - 32U));
    std::copy(digest.begin(), digest.end(), result.end() - 32);
    return result;
}

[[nodiscard]] std::string manifest_with_probe(
    std::string manifest, std::string_view probe) {
    require(!manifest.empty() && manifest.back() == '}',
            "test fixture manifest is not an object");
    manifest.pop_back();
    manifest.append(",\"zz_portability_probe\":");
    manifest.append(probe);
    manifest.push_back('}');
    return manifest;
}

[[nodiscard]] std::string nested_probe(std::size_t container_count) {
    std::string result;
    result.reserve(container_count * 6U + 4U);
    for (std::size_t index = 0; index < container_count; ++index) {
        result.append("{\"x\":");
    }
    result.append("null");
    result.append(container_count, '}');
    return result;
}

[[nodiscard]] std::string null_array_probe(std::size_t element_count) {
    std::string result;
    result.reserve(element_count * 5U + 2U);
    result.push_back('[');
    for (std::size_t index = 0; index < element_count; ++index) {
        if (index != 0) result.push_back(',');
        result.append("null");
    }
    result.push_back(']');
    return result;
}

void require_open_status(const std::vector<std::uint8_t>& artifact,
                         ptmrt_status expected, std::string_view message) {
    ptmrt_model* model = nullptr;
    const auto actual =
        ptmrt_model_open_memory(artifact.data(), artifact.size(), &model);
    require(actual == expected, message);
    require((actual == PTMRT_STATUS_OK) == (model != nullptr),
            "runtime returned an inconsistent model pointer");
    ptmrt_model_close(model);
}

struct ModelOwner {
    ptmrt_model* value{};
    ModelOwner() = default;
    ModelOwner(const ModelOwner&) = delete;
    ModelOwner& operator=(const ModelOwner&) = delete;
    ModelOwner(ModelOwner&& other) noexcept : value(other.value) {
        other.value = nullptr;
    }
    ModelOwner& operator=(ModelOwner&& other) noexcept {
        if (this != &other) {
            ptmrt_model_close(value);
            value = other.value;
            other.value = nullptr;
        }
        return *this;
    }
    ~ModelOwner() { ptmrt_model_close(value); }
};

ModelOwner open_golden(const std::vector<std::uint8_t>& bytes) {
    ModelOwner result{};
    const auto status =
        ptmrt_model_open_memory(bytes.data(), bytes.size(), &result.value);
    require(status == PTMRT_STATUS_OK,
            std::string("runtime rejected the golden model artifact: ") +
                ptmrt_status_message(status));
    require(result.value != nullptr, "runtime returned a null model");
    return result;
}

void test_description_manifest_and_conformance() {
    const auto bytes = read_golden();
    auto model = open_golden(bytes);
    require(ptmrt_abi_version() == PTMRT_ABI_VERSION,
            "runtime ABI version differs from its header");

    ptmrt_model_description description{};
    require(ptmrt_model_describe(model.value, &description) == PTMRT_STATUS_OK,
            "runtime model description failed");
    require(std::string_view(description.artifact_id) ==
                "sha256:2437f0e6ae6193c9eaf8081c8796c7482314985010e5bb08f5d72d0ddafa5129",
            "runtime artifact ID is wrong");
    require(description.model_kind == PTMRT_MODEL_PACKED_TM_BINARY_V1 &&
                description.number_of_clauses == 4 &&
                description.number_of_features == 2 &&
                description.threshold == 8,
            "runtime packed-TM description is wrong");
    require(description.input_count == 2 && description.output_count == 2 &&
                std::string_view(description.inputs[0].name) == "features" &&
                std::string_view(description.outputs[0].name) == "predictions",
            "runtime port description is wrong");

    std::uint64_t required = 0;
    require(ptmrt_model_manifest_json(model.value, nullptr, 0, &required) ==
                PTMRT_STATUS_OK &&
                required > 1,
            "runtime manifest size query failed");
    std::vector<char> manifest(static_cast<std::size_t>(required));
    require(ptmrt_model_manifest_json(
                model.value, manifest.data(), manifest.size(), &required) ==
                PTMRT_STATUS_OK,
            "runtime manifest read failed");
    require(std::string_view(manifest.data()).find("XOR Little Guy") !=
                std::string_view::npos,
            "runtime manifest lost the artifact title");
    require(ptmrt_model_verify(model.value) == PTMRT_STATUS_OK,
            "runtime conformance verification failed");
}

void test_generic_tensor_run() {
    const auto bytes = read_golden();
    auto model = open_golden(bytes);
    std::array<std::uint64_t, 2> features{0b1100U, 0b1010U};
    std::uint64_t valid = 0x0fU;
    std::uint64_t predictions = 0;
    std::array<std::int32_t, 64> scores{};

    std::array<ptmrt_tensor_view, 2> inputs{};
    inputs[0] = {"features", features.data(), sizeof(features),
                 PTMRT_DTYPE_UINT64, 1, {2, 0, 0, 0}};
    inputs[1] = {"valid_mask", &valid, sizeof(valid),
                 PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    std::array<ptmrt_tensor_view, 2> outputs{};
    outputs[0] = {"predictions", &predictions, sizeof(predictions),
                  PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    outputs[1] = {"scores", scores.data(), sizeof(scores),
                  PTMRT_DTYPE_INT32, 1, {64, 0, 0, 0}};
    require(ptmrt_model_run(model.value, inputs.data(), inputs.size(),
                            outputs.data(), outputs.size()) == PTMRT_STATUS_OK,
            "runtime generic tensor inference failed");
    require(predictions == 0b0110U,
            "runtime XOR predictions are wrong");
    require(scores[0] == -1 && scores[1] == 1 && scores[2] == 1 &&
                scores[3] == -1,
            "runtime XOR scores are wrong");
    for (std::size_t lane = 4; lane < scores.size(); ++lane) {
        require(scores[lane] == 0, "runtime did not suppress an invalid lane");
    }

    inputs[0].name = "unknown";
    require(ptmrt_model_run(model.value, inputs.data(), inputs.size(),
                            outputs.data(), outputs.size()) ==
                PTMRT_STATUS_INVALID_ARGUMENT,
            "runtime accepted a missing named tensor port");
}

void test_raw_record_preprocessing() {
    const auto bytes = read_fixture("raw_xor_packed_tm_v1.hex");
    auto model = open_golden(bytes);
    require(ptmrt_model_has_preprocessing(model.value) == 1,
            "runtime did not expose embedded preprocessing");

    std::array<ptmrt_record_field, 2> fields{};
    fields[0].name = "left";
    fields[0].kind = PTMRT_VALUE_BOOL;
    fields[0].boolean_value = 0;
    fields[1].name = "right";
    fields[1].kind = PTMRT_VALUE_BOOL;
    fields[1].boolean_value = 1;
    std::uint64_t required = 0;
    require(ptmrt_model_preprocess_record(
                model.value, fields.data(), fields.size(), nullptr, 0,
                &required) == PTMRT_STATUS_OK && required == 2,
            "runtime preprocessing size query failed");
    std::array<std::uint64_t, 2> features{9, 9};
    require(ptmrt_model_preprocess_record(
                model.value, fields.data(), fields.size(), features.data(),
                features.size(), &required) == PTMRT_STATUS_OK &&
                features[0] == 0 && features[1] == 1,
            "runtime raw Boolean materialization is wrong");

    std::uint64_t valid = 1;
    std::uint64_t predictions = 0;
    std::array<std::int32_t, 64> scores{};
    std::array<ptmrt_tensor_view, 2> inputs{};
    inputs[0] = {"features", features.data(), sizeof(features),
                 PTMRT_DTYPE_UINT64, 1, {2, 0, 0, 0}};
    inputs[1] = {"valid_mask", &valid, sizeof(valid),
                 PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    std::array<ptmrt_tensor_view, 2> outputs{};
    outputs[0] = {"predictions", &predictions, sizeof(predictions),
                  PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    outputs[1] = {"scores", scores.data(), sizeof(scores),
                  PTMRT_DTYPE_INT32, 1, {64, 0, 0, 0}};
    require(ptmrt_model_run(model.value, inputs.data(), inputs.size(),
                            outputs.data(), outputs.size()) ==
                PTMRT_STATUS_OK && predictions == 1 && scores[0] == 1,
            "runtime raw-record inference is wrong");

    auto invalid_fields = fields;
    invalid_fields[0].kind = PTMRT_VALUE_INT64;
    invalid_fields[0].integer_value = 0;
    features = {9, 9};
    require(ptmrt_model_preprocess_record(
                model.value, invalid_fields.data(), invalid_fields.size(),
                features.data(), features.size(), &required) ==
                PTMRT_STATUS_INVALID_ARGUMENT &&
                features[0] == 9 && features[1] == 9,
            "runtime accepted an untyped Boolean or partially wrote output");
    require(ptmrt_model_preprocess_record(
                model.value, fields.data(), fields.size(), features.data(), 1,
                &required) == PTMRT_STATUS_INSUFFICIENT_CAPACITY,
            "runtime accepted an undersized preprocessing buffer");

    const auto old_bytes = read_golden();
    auto old_model = open_golden(old_bytes);
    require(ptmrt_model_has_preprocessing(old_model.value) == 0 &&
                ptmrt_model_preprocess_record(
                    old_model.value, nullptr, 0, nullptr, 0, &required) ==
                    PTMRT_STATUS_UNSUPPORTED_MODEL,
            "precomputed-only artifacts advertised raw preprocessing");
}

void test_pta_materialized_threshold_artifact() {
    const auto bytes = read_fixture("pta_threshold_clause_v1.hex");
    auto model = open_golden(bytes);
    ptmrt_model_description description{};
    require(ptmrt_model_describe(model.value, &description) ==
                PTMRT_STATUS_OK,
            "PTA threshold fixture description failed");
    require(description.model_kind == PTMRT_MODEL_PACKED_TM_BINARY_V1 &&
                description.number_of_clauses == 1 &&
                description.number_of_features == 1 &&
                description.threshold == 1,
            "PTA threshold fixture has the wrong native model shape");
    require(ptmrt_model_verify(model.value) == PTMRT_STATUS_OK,
            "PTA threshold fixture failed native conformance verification");

    std::uint64_t manifest_size = 0;
    require(ptmrt_model_manifest_json(
                model.value, nullptr, 0, &manifest_size) == PTMRT_STATUS_OK &&
                manifest_size > 1,
            "PTA threshold manifest size query failed");
    std::vector<char> manifest(static_cast<std::size_t>(manifest_size));
    require(ptmrt_model_manifest_json(
                model.value, manifest.data(), manifest.size(),
                &manifest_size) == PTMRT_STATUS_OK,
            "PTA threshold manifest read failed");
    const std::string_view manifest_view(manifest.data());
    require(
        manifest_view.find("origin_proposal_provenance_id") !=
            std::string_view::npos &&
        manifest_view.find("observations_digest") != std::string_view::npos &&
        manifest_view.find("not a trained source-label classifier") !=
            std::string_view::npos,
        "PTA threshold artifact lost provenance or its semantic limitation");

    const auto evaluate_record = [&model](double temperature) {
        ptmrt_record_field field{};
        field.name = "temperature";
        field.kind = PTMRT_VALUE_FLOAT64;
        field.number_value = temperature;
        std::uint64_t required = 0;
        require(ptmrt_model_preprocess_record(
                    model.value, &field, 1, nullptr, 0, &required) ==
                    PTMRT_STATUS_OK &&
                    required == 1,
                "PTA threshold preprocessing size query failed");
        std::array<std::uint64_t, 1> features{};
        require(ptmrt_model_preprocess_record(
                    model.value, &field, 1, features.data(), features.size(),
                    &required) == PTMRT_STATUS_OK,
                "PTA threshold preprocessing failed");

        std::uint64_t valid = 1;
        std::uint64_t predictions = 0;
        std::array<std::int32_t, 64> scores{};
        std::array<ptmrt_tensor_view, 2> inputs{};
        inputs[0] = {"features", features.data(), sizeof(features),
                     PTMRT_DTYPE_UINT64, 1, {1, 0, 0, 0}};
        inputs[1] = {"valid_mask", &valid, sizeof(valid),
                     PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
        std::array<ptmrt_tensor_view, 2> outputs{};
        outputs[0] = {"predictions", &predictions, sizeof(predictions),
                      PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
        outputs[1] = {"scores", scores.data(), sizeof(scores),
                      PTMRT_DTYPE_INT32, 1, {64, 0, 0, 0}};
        require(ptmrt_model_run(model.value, inputs.data(), inputs.size(),
                                outputs.data(), outputs.size()) ==
                    PTMRT_STATUS_OK,
                "PTA threshold native inference failed");
        return std::array<std::int64_t, 3>{
            static_cast<std::int64_t>(features[0]),
            static_cast<std::int64_t>(predictions & 1U),
            static_cast<std::int64_t>(scores[0])};
    };

    require(evaluate_record(74.5) ==
                std::array<std::int64_t, 3>{0, 0, 0},
            "PTA threshold artifact activated below its boundary");
    require(evaluate_record(75.0) ==
                std::array<std::int64_t, 3>{1, 1, 1},
            "PTA threshold artifact rejected its inclusive boundary");
    require(evaluate_record(90.0) ==
                std::array<std::int64_t, 3>{1, 1, 1},
            "PTA threshold artifact rejected an above-boundary record");
}

void test_all_portable_preprocessing_transforms() {
    const auto bytes = read_fixture("preprocessing_demo_v1.hex");
    auto model = open_golden(bytes);
    const std::string status = "ready";
    std::array<ptmrt_record_field, 3> fields{};
    fields[0].name = "age";
    fields[0].kind = PTMRT_VALUE_FLOAT64;
    fields[0].number_value = 30.0;
    fields[1].name = "status";
    fields[1].kind = PTMRT_VALUE_UTF8;
    fields[1].string_data = status.data();
    fields[1].string_size = status.size();
    fields[2].name = "active";
    fields[2].kind = PTMRT_VALUE_BOOL;
    fields[2].boolean_value = 1;
    std::array<std::uint64_t, 6> features{};
    std::uint64_t required = 0;
    require(ptmrt_model_preprocess_record(
                model.value, fields.data(), fields.size(), features.data(),
                features.size(), &required) == PTMRT_STATUS_OK &&
                features == std::array<std::uint64_t, 6>{1, 1, 1, 1, 1, 0},
            "native preprocessing transforms disagree with the contract");

    fields[0].kind = PTMRT_VALUE_INT64;
    fields[0].integer_value = 19;
    fields[1].kind = PTMRT_VALUE_NULL;
    fields[2].kind = PTMRT_VALUE_NULL;
    require(ptmrt_model_preprocess_record(
                model.value, fields.data(), fields.size(), features.data(),
                features.size(), &required) == PTMRT_STATUS_OK &&
                features == std::array<std::uint64_t, 6>{1, 0, 1, 0, 0, 1},
            "native preprocessing null policies are wrong");

    require(ptmrt_model_preprocess_record(
                model.value, fields.data() + 1, fields.size() - 1,
                features.data(), features.size(), &required) ==
                PTMRT_STATUS_INVALID_ARGUMENT,
            "native preprocessing ignored a required missing field");
}

void test_logic_artifact_and_generic_tensor_run() {
    const auto bytes = read_fixture("conditional_logic_program_v1.hex");
    auto model = open_golden(bytes);
    ptmrt_model_description description{};
    require(ptmrt_model_describe(model.value, &description) == PTMRT_STATUS_OK,
            "runtime Logic model description failed");
    require(std::string_view(description.artifact_id) ==
                "sha256:dde64b31658c0d79c74ed7cf3b9b928b662fa32e10ff87b6b51e7296476879b2",
            "runtime Logic artifact ID is wrong");
    require(description.model_kind == PTMRT_MODEL_LOGIC_PROGRAM32_V1 &&
                description.instruction_count == 11 &&
                description.binding_count == 5 &&
                description.input_count == 2 && description.output_count == 3,
            "runtime Logic description is wrong");
    require(std::string_view(description.inputs[0].name) == "bindings" &&
                std::string_view(description.outputs[0].name) == "values" &&
                description.outputs[1].dtype == PTMRT_DTYPE_UINT32,
            "runtime Logic port description is wrong");
    require(ptmrt_model_verify(model.value) == PTMRT_STATUS_OK,
            "runtime Logic conformance verification failed");

    std::array<std::uint64_t, 5> bindings{
        0xaaaaaaaaU,
        0xccccccccU,
        0xf0f0f0f0U,
        0xff00ff00U,
        0xffff0000U,
    };
    std::uint64_t valid = 0xffffffffU;
    std::uint64_t values = 0;
    std::array<std::uint32_t, 64> true_masks{};
    std::array<std::uint32_t, 64> evaluated_masks{};
    std::array<ptmrt_tensor_view, 2> inputs{};
    inputs[0] = {"bindings", bindings.data(), sizeof(bindings),
                 PTMRT_DTYPE_UINT64, 1, {5, 0, 0, 0}};
    inputs[1] = {"valid_mask", &valid, sizeof(valid),
                 PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    std::array<ptmrt_tensor_view, 3> outputs{};
    outputs[0] = {"values", &values, sizeof(values),
                  PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    outputs[1] = {"true_instruction_masks", true_masks.data(),
                  sizeof(true_masks), PTMRT_DTYPE_UINT32, 1, {64, 0, 0, 0}};
    outputs[2] = {"evaluated_instruction_masks", evaluated_masks.data(),
                  sizeof(evaluated_masks), PTMRT_DTYPE_UINT32, 1,
                  {64, 0, 0, 0}};
    require(ptmrt_model_run(model.value, inputs.data(), inputs.size(),
                            outputs.data(), outputs.size()) == PTMRT_STATUS_OK,
            "runtime generic Logic inference failed");
    require(values == 0x2f2f2f20U,
            "runtime exhaustive Logic values are wrong");
    require(true_masks[0] == 0x108U && evaluated_masks[0] == 0x7ffU,
            "runtime Logic diagnostics are wrong");
    for (std::size_t lane = 32; lane < 64; ++lane) {
        require(true_masks[lane] == 0 && evaluated_masks[lane] == 0,
                "runtime Logic diagnostics use an invalid lane");
    }
}

void test_masked_threshold_artifact_and_generic_tensor_run() {
    const auto bytes = read_fixture("masked_threshold_v1.hex");
    auto model = open_golden(bytes);
    ptmrt_model_description description{};
    require(ptmrt_model_describe(model.value, &description) == PTMRT_STATUS_OK,
            "runtime masked-threshold model description failed");
    require(std::string_view(description.artifact_id) ==
                "sha256:93a80e4f6f6cf7a604db3ef56a8df1dbe7d951414d0fe69038f7fc77e0d15022",
            "runtime masked-threshold artifact ID is wrong");
    require(description.model_kind == PTMRT_MODEL_MASKED_THRESHOLD_V1 &&
                description.slot_count == 1024 &&
                description.minimum_true == 2 &&
                description.selected_count == 3 &&
                description.input_count == 2 && description.output_count == 4,
            "runtime masked-threshold description is wrong");
    require(std::string_view(description.inputs[0].name) == "slots" &&
                std::string_view(description.outputs[0].name) == "values" &&
                std::string_view(description.outputs[1].name) ==
                    "matched_counts" &&
                description.outputs[1].dtype == PTMRT_DTYPE_UINT32,
            "runtime masked-threshold port description is wrong");
    require(ptmrt_model_verify(model.value) == PTMRT_STATUS_OK,
            "runtime masked-threshold conformance verification failed");

    std::vector<std::uint64_t> slots(1024, 0);
    slots[1] = 0xaaU;
    slots[7] = 0xccU;
    slots[70] = 0xf0U;
    std::uint64_t valid = 0xffU;
    std::uint64_t values = 0;
    std::array<std::uint32_t, 64> matched_counts{};
    std::vector<std::uint64_t> matched_slots(1024);
    std::vector<std::uint64_t> missing_slots(1024);
    std::array<ptmrt_tensor_view, 2> inputs{};
    inputs[0] = {"slots", slots.data(), slots.size() * sizeof(slots[0]),
                 PTMRT_DTYPE_UINT64, 1, {slots.size(), 0, 0, 0}};
    inputs[1] = {"valid_mask", &valid, sizeof(valid),
                 PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    std::array<ptmrt_tensor_view, 4> outputs{};
    outputs[0] = {"values", &values, sizeof(values),
                  PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
    outputs[1] = {"matched_counts", matched_counts.data(),
                  sizeof(matched_counts), PTMRT_DTYPE_UINT32, 1,
                  {64, 0, 0, 0}};
    outputs[2] = {"matched_slots", matched_slots.data(),
                  matched_slots.size() * sizeof(matched_slots[0]),
                  PTMRT_DTYPE_UINT64, 1,
                  {matched_slots.size(), 0, 0, 0}};
    outputs[3] = {"missing_slots", missing_slots.data(),
                  missing_slots.size() * sizeof(missing_slots[0]),
                  PTMRT_DTYPE_UINT64, 1,
                  {missing_slots.size(), 0, 0, 0}};
    require(ptmrt_model_run(model.value, inputs.data(), inputs.size(),
                            outputs.data(), outputs.size()) == PTMRT_STATUS_OK,
            "runtime generic masked-threshold inference failed");
    require(values == 0xe8U,
            "runtime masked-threshold values are wrong");
    constexpr std::array<std::uint32_t, 8> expected_counts{
        0, 1, 1, 2, 1, 2, 2, 3};
    require(std::equal(expected_counts.begin(), expected_counts.end(),
                       matched_counts.begin()),
            "runtime masked-threshold match counts are wrong");
    require(matched_slots[1] == 0xaaU && matched_slots[7] == 0xccU &&
                matched_slots[70] == 0xf0U && missing_slots[1] == 0x55U &&
                missing_slots[7] == 0x33U && missing_slots[70] == 0x0fU,
            "runtime masked-threshold slot diagnostics are wrong");
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
        if (slot != 1 && slot != 7 && slot != 70) {
            require(matched_slots[slot] == 0 && missing_slots[slot] == 0,
                    "runtime emitted diagnostics for an unselected slot");
        }
    }
    for (std::size_t lane = 8; lane < matched_counts.size(); ++lane) {
        require(matched_counts[lane] == 0,
                "runtime counted an invalid masked-threshold lane");
    }
}

void test_integrity_and_argument_rejection() {
    auto bytes = read_golden();
    bytes[bytes.size() / 2U] ^= 1U;
    ptmrt_model* model = nullptr;
    require(ptmrt_model_open_memory(bytes.data(), bytes.size(), &model) ==
                PTMRT_STATUS_INTEGRITY_ERROR &&
                model == nullptr,
            "runtime accepted a corrupted artifact");
    require(ptmrt_model_open_memory(nullptr, 0, &model) ==
                PTMRT_STATUS_NULL_POINTER,
            "runtime accepted a null artifact buffer");

    bytes = read_golden();
    bytes[8] = 2;
    require(ptmrt_model_open_memory(bytes.data(), bytes.size(), &model) ==
                PTMRT_STATUS_UNSUPPORTED_VERSION,
            "runtime did not identify an unsupported container version");
    bytes = read_golden();
    bytes[16] = 99;
    require(ptmrt_model_open_memory(bytes.data(), bytes.size(), &model) ==
                PTMRT_STATUS_UNSUPPORTED_MODEL,
            "runtime did not identify an unsupported model kind");
    require(ptmrt_model_open_memory(bytes.data(), 12, &model) ==
                PTMRT_STATUS_INVALID_FORMAT,
            "runtime accepted a truncated artifact");
}

void test_bounded_hostile_artifact_corpus() {
    const auto golden = read_golden();
    ptmrt_model* model = nullptr;

    const auto reject = [&model](const std::uint8_t* data, std::uint64_t size,
                                 std::string_view message) {
        model = nullptr;
        const auto status = ptmrt_model_open_memory(data, size, &model);
        require(status != PTMRT_STATUS_OK && model == nullptr, message);
    };

    const auto prefix_limit = std::min<std::size_t>(golden.size(), 96U);
    for (std::size_t size = 0; size < prefix_limit; ++size) {
        reject(golden.data(), size,
               "runtime accepted a hostile truncated artifact prefix");
    }
    for (std::size_t removed = 1;
         removed <= std::min<std::size_t>(golden.size(), 64U); ++removed) {
        reject(golden.data(), golden.size() - removed,
               "runtime accepted a hostile truncated artifact suffix");
    }

    const auto stride = std::max<std::size_t>(1U, golden.size() / 128U);
    for (std::size_t offset = 0; offset < golden.size(); offset += stride) {
        auto mutated = golden;
        mutated[offset] ^= 0xa5U;
        reject(mutated.data(), mutated.size(),
               "runtime accepted a single-byte artifact mutation");
    }

    auto oversized_manifest = golden;
    const auto hostile_size = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t byte = 0; byte < sizeof(hostile_size); ++byte) {
        oversized_manifest[24U + byte] = static_cast<std::uint8_t>(
            hostile_size >> (byte * 8U));
    }
    reject(oversized_manifest.data(), oversized_manifest.size(),
           "runtime accepted an overflowing manifest size");
    reject(golden.data(), hostile_size,
           "runtime accepted an artifact size above its allocation ceiling");
}

void test_portable_manifest_complexity_contract() {
    static_assert(PTMRT_MODEL_MANIFEST_MAX_BYTES == 16U * 1024U * 1024U);
    static_assert(PTMRT_MODEL_MANIFEST_MAX_DEPTH == 8U);
    static_assert(PTMRT_MODEL_MANIFEST_MAX_NODES == 100000U);

    struct FixtureCase {
        std::string_view filename;
        std::size_t base_nodes;
    };
    constexpr std::array cases{
        FixtureCase{"xor_packed_tm_v1.hex", 69U},
        FixtureCase{"conditional_logic_program_v1.hex", 86U},
        FixtureCase{"masked_threshold_v1.hex", 93U},
    };

    for (const auto& test_case : cases) {
        const auto base = read_fixture(test_case.filename);
        const auto manifest = fixture_manifest(base);

        auto candidate = replace_fixture_manifest(
            base, manifest_with_probe(
                      manifest,
                      nested_probe(PTMRT_MODEL_MANIFEST_MAX_DEPTH - 1U)));
        require_open_status(candidate, PTMRT_STATUS_OK,
                            "runtime rejected a manifest at max depth");

        candidate = replace_fixture_manifest(
            base, manifest_with_probe(
                      manifest,
                      nested_probe(PTMRT_MODEL_MANIFEST_MAX_DEPTH)));
        require_open_status(candidate, PTMRT_STATUS_INVALID_FORMAT,
                            "runtime accepted a manifest above max depth");

        candidate = replace_fixture_manifest(
            base, manifest_with_probe(
                      manifest, "\"control\\u0001character\""));
        require(fixture_manifest(candidate).find("\\u0001") !=
                    std::string::npos,
                "Unicode escape probe was not serialized as expected");
        require_open_status(candidate, PTMRT_STATUS_OK,
                            "runtime rejected a canonical Unicode escape");

        candidate = replace_fixture_manifest(
            base, manifest_with_probe(manifest, "\"rocket\\ud83d\\ude80\""));
        require_open_status(candidate, PTMRT_STATUS_OK,
                            "runtime rejected a Unicode surrogate pair");

        candidate = replace_fixture_manifest(
            base, manifest_with_probe(manifest, "\"lone\\ud800surrogate\""));
        require_open_status(candidate, PTMRT_STATUS_INVALID_FORMAT,
                            "runtime accepted a lone Unicode surrogate");

        const auto at_limit_elements =
            PTMRT_MODEL_MANIFEST_MAX_NODES - test_case.base_nodes - 1U;
        candidate = replace_fixture_manifest(
            base, manifest_with_probe(
                      manifest, null_array_probe(at_limit_elements)));
        require_open_status(candidate, PTMRT_STATUS_OK,
                            "runtime rejected a manifest at max nodes");

        candidate = replace_fixture_manifest(
            base, manifest_with_probe(
                      manifest, null_array_probe(at_limit_elements + 1U)));
        require_open_status(candidate, PTMRT_STATUS_INVALID_FORMAT,
                            "runtime accepted a manifest above max nodes");
    }
}

void test_file_loader() {
    const auto bytes = read_golden();
    const auto path = std::filesystem::path(PTM_TEST_BINARY_DIR) /
                      "ptmrt-xor-file-loader-test.ptm";
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(stream), "could not create runtime file fixture");
        stream.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        require(static_cast<bool>(stream), "could not write runtime file fixture");
    }

    ModelOwner model{};
    const auto status = ptmrt_model_open_file(path.string().c_str(), &model.value);
    std::error_code removal_error;
    std::filesystem::remove(path, removal_error);
    require(status == PTMRT_STATUS_OK,
            std::string("runtime file loader rejected the golden artifact: ") +
                ptmrt_status_message(status));
    require(ptmrt_model_verify(model.value) == PTMRT_STATUS_OK,
            "file-loaded runtime model failed conformance verification");
    require(!removal_error, "could not remove runtime file fixture");
}

void test_concurrent_read_only_inference() {
    const auto bytes = read_golden();
    auto model = open_golden(bytes);
    std::atomic<bool> succeeded{true};
    std::vector<std::thread> workers;
    for (std::uint32_t worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&model, &succeeded, worker] {
            for (std::uint32_t iteration = 0; iteration < 100; ++iteration) {
                const auto value = (worker + iteration) & 3U;
                std::array<std::uint64_t, 2> features{
                    static_cast<std::uint64_t>((value >> 1U) & 1U),
                    static_cast<std::uint64_t>(value & 1U),
                };
                std::uint64_t valid = 1;
                std::uint64_t prediction = 0;
                std::array<std::int32_t, 64> scores{};
                std::array<ptmrt_tensor_view, 2> inputs{};
                inputs[0] = {"features", features.data(), sizeof(features),
                             PTMRT_DTYPE_UINT64, 1, {2, 0, 0, 0}};
                inputs[1] = {"valid_mask", &valid, sizeof(valid),
                             PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
                std::array<ptmrt_tensor_view, 2> outputs{};
                outputs[0] = {"predictions", &prediction, sizeof(prediction),
                              PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
                outputs[1] = {"scores", scores.data(), sizeof(scores),
                              PTMRT_DTYPE_INT32, 1, {64, 0, 0, 0}};
                const auto status = ptmrt_model_run(
                    model.value, inputs.data(), inputs.size(), outputs.data(),
                    outputs.size());
                const auto expected =
                    ((features[0] != 0) != (features[1] != 0)) ? 1U : 0U;
                if (status != PTMRT_STATUS_OK ||
                    (prediction & 1U) != expected ||
                    scores[0] != (expected != 0 ? 1 : -1)) {
                    succeeded.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    require(succeeded.load(std::memory_order_relaxed),
            "read-only runtime inference was not thread-safe");
}

void test_concurrent_logic_inference() {
    const auto bytes = read_fixture("conditional_logic_program_v1.hex");
    auto model = open_golden(bytes);
    std::atomic<bool> succeeded{true};
    std::vector<std::thread> workers;
    for (std::uint32_t worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&model, &succeeded, worker] {
            for (std::uint32_t iteration = 0; iteration < 100; ++iteration) {
                const auto bits = (worker + iteration) & 31U;
                std::array<std::uint64_t, 5> bindings{};
                for (std::size_t index = 0; index < bindings.size(); ++index) {
                    bindings[index] = (bits >> index) & 1U;
                }
                std::uint64_t valid = 1;
                std::uint64_t values = 0;
                std::array<std::uint32_t, 64> true_masks{};
                std::array<std::uint32_t, 64> evaluated_masks{};
                std::array<ptmrt_tensor_view, 2> inputs{};
                inputs[0] = {"bindings", bindings.data(), sizeof(bindings),
                             PTMRT_DTYPE_UINT64, 1, {5, 0, 0, 0}};
                inputs[1] = {"valid_mask", &valid, sizeof(valid),
                             PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
                std::array<ptmrt_tensor_view, 3> outputs{};
                outputs[0] = {"values", &values, sizeof(values),
                              PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
                outputs[1] = {"true_instruction_masks", true_masks.data(),
                              sizeof(true_masks), PTMRT_DTYPE_UINT32, 1,
                              {64, 0, 0, 0}};
                outputs[2] = {"evaluated_instruction_masks",
                              evaluated_masks.data(), sizeof(evaluated_masks),
                              PTMRT_DTYPE_UINT32, 1, {64, 0, 0, 0}};
                const auto status = ptmrt_model_run(
                    model.value, inputs.data(), inputs.size(), outputs.data(),
                    outputs.size());
                const auto a = bindings[0] != 0;
                const auto b = bindings[1] != 0;
                const auto c = bindings[2] != 0;
                const auto d = bindings[3] != 0;
                const auto e = bindings[4] != 0;
                const auto expected = c ? (a && !b) : (d || e);
                if (status != PTMRT_STATUS_OK ||
                    ((values & 1U) != 0) != expected ||
                    evaluated_masks[0] != 0x7ffU) {
                    succeeded.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    require(succeeded.load(std::memory_order_relaxed),
            "read-only Logic runtime inference was not thread-safe");
}

void test_graph_artifact_load_describe_verify_and_unsupported_run() {
    const auto bytes = read_fixture("graph_tm_v1.hex");
    ModelOwner model{};
    const auto open_status = ptmrt_model_open_memory(bytes.data(), bytes.size(), &model.value);
    require(open_status == PTMRT_STATUS_OK && model.value != nullptr, "runtime rejected the graph fixture");
    ptmrt_model_description desc{};
    require(ptmrt_model_describe(model.value, &desc) == PTMRT_STATUS_OK, "graph describe failed");
    require(desc.model_kind == PTMRT_MODEL_GRAPH_TM_V1, "graph model kind wrong");
    require(desc.graph_depth == 2 && desc.graph_clauses == 2 && desc.graph_hv_dim == 256, "graph dims wrong");
    require(desc.input_count == 1 && desc.output_count == 1, "graph port counts wrong");
    require(std::string_view(desc.inputs[0].name) == "graph" && std::string_view(desc.outputs[0].name) == "prediction", "graph port names wrong");
    require(desc.conformance_case_count == 2, "graph conformance count wrong");
    // Manifest should contain graph description
    std::uint64_t req = 0;
    require(ptmrt_model_manifest_json(model.value, nullptr, 0, &req) == PTMRT_STATUS_OK && req > 1, "graph manifest size query failed");
    std::vector<char> manifest(static_cast<std::size_t>(req));
    require(ptmrt_model_manifest_json(model.value, manifest.data(), manifest.size(), &req) == PTMRT_STATUS_OK, "graph manifest read failed");
    require(std::string_view(manifest.data()).find("graph_tm_v1") != std::string_view::npos, "graph manifest lost kind");
    require(std::string_view(manifest.data()).find("graph-fixture-v1") != std::string_view::npos, "graph manifest lost title");
    // Native verify and run are explicitly unsupported for graph
    require(ptmrt_model_verify(model.value) == PTMRT_STATUS_UNSUPPORTED_MODEL, "graph verify should be UNSUPPORTED_MODEL");
    std::array<ptmrt_tensor_view, 1> dummy_inputs{};
    std::array<ptmrt_tensor_view, 1> dummy_outputs{};
    require(ptmrt_model_run(model.value, dummy_inputs.data(), 0, dummy_outputs.data(), 0) == PTMRT_STATUS_UNSUPPORTED_MODEL, "graph run should be UNSUPPORTED_MODEL");
    // Mutated graph must be rejected (rehashed with wrong weights or corrupt graph JSON)
    auto mutated = bytes;
    // Flip a byte in the weight section (offset after container header + manifest; approximate payload header 32 then weights 16 bytes)
    // Instead flip a byte near middle of file (payload weights area)
    if (mutated.size() > 200) {
        mutated[150] ^= 0x01;
        ptmrt_model* bad = nullptr;
        const auto bad_status = ptmrt_model_open_memory(mutated.data(), mutated.size(), &bad);
        // Should be integrity error (SHA-256 mismatch) before parse; if we rehash it would be invalid_format weight mismatch — both are rejection
        require(bad_status != PTMRT_STATUS_OK && bad == nullptr, "runtime accepted mutated graph artifact");
    }
    // Rehashed hostile: valid SHA but manifest weight != payload weight must be INVALID_FORMAT (not OK)
    {
        auto rehashed = bytes;
        if (rehashed.size() >= 64 + 32) {
            auto manifest_size = static_cast<std::size_t>(rehashed[24] | (static_cast<std::size_t>(rehashed[25])<<8) | (static_cast<std::size_t>(rehashed[26])<<16) | (static_cast<std::size_t>(rehashed[27])<<24)
                                            | (static_cast<std::size_t>(rehashed[28])<<32) | (static_cast<std::size_t>(rehashed[29])<<40) | (static_cast<std::size_t>(rehashed[30])<<48) | (static_cast<std::size_t>(rehashed[31])<<56));
            std::size_t payload_off = 64 + manifest_size;
            if (payload_off + 32 + 8 < rehashed.size() - 32) {
                rehashed[payload_off + 32] ^= 0xFF; // corrupt first weight int32 byte
                // Also corrupt the second weight byte to stay within -1e6..1e6 but different from manifest
                // Flip low bit then recompute SHA-256 trailer to make container integrity valid
                auto content = std::span<const std::uint8_t>(rehashed.data(), rehashed.size() - 32);
                auto digest = sha256_test(content);
                std::copy(digest.begin(), digest.end(), rehashed.end() - 32);
                ptmrt_model* bad = nullptr;
                auto st2 = ptmrt_model_open_memory(rehashed.data(), rehashed.size(), &bad);
                require(st2 == PTMRT_STATUS_INVALID_FORMAT && bad == nullptr, "rehashed weight-mismatch graph must be rejected as INVALID_FORMAT");
            }
        }
    }
    // Corrupt a graph JSON byte (inside conformance graph) and ensure rejection via integrity error
    auto mutated2 = bytes;
    if (mutated2.size() > 400) {
        mutated2[400] ^= 0xFF;
        ptmrt_model* bad2 = nullptr;
        require(ptmrt_model_open_memory(mutated2.data(), mutated2.size(), &bad2) != PTMRT_STATUS_OK, "runtime accepted graph with corrupt JSON");
    }
}

void test_concurrent_masked_threshold_inference() {
    const auto bytes = read_fixture("masked_threshold_v1.hex");
    auto model = open_golden(bytes);
    std::atomic<bool> succeeded{true};
    std::vector<std::thread> workers;
    for (std::uint32_t worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&model, &succeeded, worker] {
            for (std::uint32_t iteration = 0; iteration < 100; ++iteration) {
                const auto bits = (worker + iteration) & 7U;
                std::vector<std::uint64_t> slots(1024, 0);
                slots[1] = bits & 1U;
                slots[7] = (bits >> 1U) & 1U;
                slots[70] = (bits >> 2U) & 1U;
                std::uint64_t valid = 1;
                std::uint64_t values = 0;
                std::array<std::uint32_t, 64> counts{};
                std::vector<std::uint64_t> matched(1024);
                std::vector<std::uint64_t> missing(1024);
                std::array<ptmrt_tensor_view, 2> inputs{};
                inputs[0] = {"slots", slots.data(),
                             slots.size() * sizeof(slots[0]),
                             PTMRT_DTYPE_UINT64, 1,
                             {slots.size(), 0, 0, 0}};
                inputs[1] = {"valid_mask", &valid, sizeof(valid),
                             PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
                std::array<ptmrt_tensor_view, 4> outputs{};
                outputs[0] = {"values", &values, sizeof(values),
                              PTMRT_DTYPE_UINT64, 0, {0, 0, 0, 0}};
                outputs[1] = {"matched_counts", counts.data(), sizeof(counts),
                              PTMRT_DTYPE_UINT32, 1, {64, 0, 0, 0}};
                outputs[2] = {"matched_slots", matched.data(),
                              matched.size() * sizeof(matched[0]),
                              PTMRT_DTYPE_UINT64, 1,
                              {matched.size(), 0, 0, 0}};
                outputs[3] = {"missing_slots", missing.data(),
                              missing.size() * sizeof(missing[0]),
                              PTMRT_DTYPE_UINT64, 1,
                              {missing.size(), 0, 0, 0}};
                const auto status = ptmrt_model_run(
                    model.value, inputs.data(), inputs.size(), outputs.data(),
                    outputs.size());
                const auto expected_count = static_cast<std::uint32_t>(
                    ((bits & 1U) != 0) + ((bits & 2U) != 0) +
                    ((bits & 4U) != 0));
                if (status != PTMRT_STATUS_OK ||
                    ((values & 1U) != 0) != (expected_count >= 2) ||
                    counts[0] != expected_count ||
                    matched[1] != (bits & 1U) ||
                    missing[70] != (((bits & 4U) == 0) ? 1U : 0U)) {
                    succeeded.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    require(succeeded.load(std::memory_order_relaxed),
            "read-only masked-threshold inference was not thread-safe");
}

}  // namespace

int main() {
    try {
        test_description_manifest_and_conformance();
        test_generic_tensor_run();
        test_raw_record_preprocessing();
        test_pta_materialized_threshold_artifact();
        test_all_portable_preprocessing_transforms();
        test_logic_artifact_and_generic_tensor_run();
        test_masked_threshold_artifact_and_generic_tensor_run();
        test_graph_artifact_load_describe_verify_and_unsupported_run();
        test_integrity_and_argument_rejection();
        test_bounded_hostile_artifact_corpus();
        test_portable_manifest_complexity_contract();
        test_file_loader();
        test_concurrent_read_only_inference();
        test_concurrent_logic_inference();
        test_concurrent_masked_threshold_inference();
        std::cout << "PTM static model runtime tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PTM static model runtime tests failed: "
                  << error.what() << '\n';
        return 1;
    }
}
