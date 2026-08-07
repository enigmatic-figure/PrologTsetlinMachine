#include "ptm/class_ii_persistence.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ptm {

namespace {

using Bytes = std::vector<std::uint8_t>;

constexpr std::array<std::uint8_t, 8> snapshot_magic{
    'P', 'T', 'M', '2', 'S', 'N', 'P', '1'};
constexpr std::array<std::uint8_t, 8> event_magic{
    'P', 'T', 'M', '2', 'E', 'V', 'T', '1'};
constexpr std::size_t digest_size = 32;
constexpr std::size_t event_fixed_size =
    event_magic.size() + sizeof(std::uint32_t) + sizeof(std::uint64_t) +
    digest_size + sizeof(std::uint64_t) + sizeof(std::uint32_t) + digest_size;
constexpr std::uint64_t maximum_persistence_file_size = 1ULL << 30U;
constexpr std::uint32_t maximum_string_size = 1U << 20U;

[[noreturn]] void corrupt(std::string message) {
    throw PersistenceError(PersistenceErrorCode::corrupt_data,
                           std::move(message));
}

[[noreturn]] void io_failure(std::string message) {
    throw PersistenceError(PersistenceErrorCode::io_error,
                           std::move(message));
}

class ByteWriter {
public:
    void u8(std::uint8_t value) { bytes_.push_back(value); }

    void u16(std::uint16_t value) {
        for (unsigned shift = 0; shift < 16; shift += 8) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void u32(std::uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void u64(std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void f64(double value) { u64(std::bit_cast<std::uint64_t>(value)); }

    void string(std::string_view value) {
        if (value.size() > maximum_string_size) {
            throw PersistenceError(PersistenceErrorCode::inconsistent_state,
                                   "persistence string exceeds its bound");
        }
        u32(static_cast<std::uint32_t>(value.size()));
        raw(std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(value.data()), value.size()));
    }

    template <std::size_t Size>
    void raw(const std::array<std::uint8_t, Size>& value) {
        raw(std::span<const std::uint8_t>(value));
    }

    void raw(std::span<const std::uint8_t> value) {
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }

    [[nodiscard]] const Bytes& bytes() const noexcept { return bytes_; }
    [[nodiscard]] Bytes take() && { return std::move(bytes_); }

private:
    Bytes bytes_;
};

class ByteReader {
public:
    explicit ByteReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::uint8_t u8() {
        require(1);
        return bytes_[position_++];
    }

    [[nodiscard]] std::uint16_t u16() {
        std::uint16_t result{};
        for (unsigned shift = 0; shift < 16; shift += 8) {
            result |= static_cast<std::uint16_t>(u8()) << shift;
        }
        return result;
    }

    [[nodiscard]] std::uint32_t u32() {
        std::uint32_t result{};
        for (unsigned shift = 0; shift < 32; shift += 8) {
            result |= static_cast<std::uint32_t>(u8()) << shift;
        }
        return result;
    }

    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t result{};
        for (unsigned shift = 0; shift < 64; shift += 8) {
            result |= static_cast<std::uint64_t>(u8()) << shift;
        }
        return result;
    }

    [[nodiscard]] double f64() {
        return std::bit_cast<double>(u64());
    }

    [[nodiscard]] std::string string() {
        const auto size = u32();
        if (size > maximum_string_size) {
            corrupt("persistence string exceeds its bound");
        }
        const auto value = raw(size);
        return std::string(reinterpret_cast<const char*>(value.data()),
                           value.size());
    }

    [[nodiscard]] std::span<const std::uint8_t> raw(std::size_t size) {
        require(size);
        const auto result = bytes_.subspan(position_, size);
        position_ += size;
        return result;
    }

    [[nodiscard]] std::size_t remaining() const noexcept {
        return bytes_.size() - position_;
    }

    void require_end() const {
        if (position_ != bytes_.size()) {
            corrupt("persistence payload has trailing bytes");
        }
    }

private:
    void require(std::size_t size) const {
        if (size > remaining()) {
            corrupt("persistence payload is truncated");
        }
    }

    std::span<const std::uint8_t> bytes_;
    std::size_t position_{};
};

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

[[nodiscard]] PersistenceDigest sha256(
    std::span<const std::uint8_t> input) {
    std::array<std::uint32_t, 8> state{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    Bytes padded(input.begin(), input.end());
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
        for (std::size_t index = 0; index < 16; ++index) {
            const auto base = offset + index * 4U;
            words[index] = (static_cast<std::uint32_t>(padded[base]) << 24U) |
                           (static_cast<std::uint32_t>(padded[base + 1U]) << 16U) |
                           (static_cast<std::uint32_t>(padded[base + 2U]) << 8U) |
                           static_cast<std::uint32_t>(padded[base + 3U]);
        }
        for (std::size_t index = 16; index < 64; ++index) {
            const auto left = words[index - 15U];
            const auto right = words[index - 2U];
            const auto sigma0 = std::rotr(left, 7) ^ std::rotr(left, 18) ^
                                (left >> 3U);
            const auto sigma1 = std::rotr(right, 17) ^ std::rotr(right, 19) ^
                                (right >> 10U);
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
        for (std::size_t index = 0; index < 64; ++index) {
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

    PersistenceDigest result{};
    for (std::size_t index = 0; index < state.size(); ++index) {
        result[index * 4U] = static_cast<std::uint8_t>(state[index] >> 24U);
        result[index * 4U + 1U] =
            static_cast<std::uint8_t>(state[index] >> 16U);
        result[index * 4U + 2U] =
            static_cast<std::uint8_t>(state[index] >> 8U);
        result[index * 4U + 3U] = static_cast<std::uint8_t>(state[index]);
    }
    return result;
}

[[nodiscard]] std::size_t to_size(std::uint64_t value,
                                  std::string_view field) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        corrupt(std::string(field) + " exceeds this platform's size range");
    }
    return static_cast<std::size_t>(value);
}

void write_metrics(ByteWriter& writer, const MaturityMetrics& value) {
    writer.f64(value.precision);
    writer.u64(value.support);
    writer.f64(value.recent_state_movement);
    writer.f64(value.feedback_rate);
    writer.u64(value.reuse_count);
    writer.f64(value.perturbation_sensitivity);
}

[[nodiscard]] MaturityMetrics read_metrics(ByteReader& reader) {
    return MaturityMetrics{
        reader.f64(), reader.u64(), reader.f64(), reader.f64(),
        reader.u64(), reader.f64(),
    };
}

[[nodiscard]] Bytes serialize_registry(
    const ConsolidationRegistrySnapshot& snapshot) {
    ByteWriter writer;
    writer.u32(snapshot.schema_version);
    writer.u64(snapshot.source_capacity);
    writer.u64(snapshot.artifact_capacity);
    writer.u64(snapshot.audit_window_size);
    writer.f64(snapshot.maturity_policy.minimum_precision);
    writer.u64(snapshot.maturity_policy.minimum_support);
    writer.f64(snapshot.maturity_policy.maximum_recent_state_movement);
    writer.f64(snapshot.maturity_policy.maximum_feedback_rate);
    writer.u64(snapshot.maturity_policy.minimum_reuse_count);
    writer.f64(snapshot.maturity_policy.maximum_perturbation_sensitivity);
    writer.u64(snapshot.audit_policy.shadow_min_observations);
    writer.f64(snapshot.audit_policy.activation_max_mismatch_rate);
    writer.u64(snapshot.audit_policy.live_min_observations);
    writer.u64(snapshot.audit_policy.reopen_min_mismatches);
    writer.f64(snapshot.audit_policy.reopen_mismatch_rate);

    writer.u64(snapshot.artifacts.size());
    for (const auto& artifact : snapshot.artifacts) {
        writer.string(artifact.specification.artifact_id);
        writer.string(artifact.specification.mapping_version);
        writer.string(artifact.specification.restoration_handle);
        writer.u64(artifact.specification.input_bits);
        writer.u8(static_cast<std::uint8_t>(artifact.specification.port_semantic));
        writer.u64(artifact.specification.bindings.size());
        for (const auto& binding : artifact.specification.bindings) {
            writer.u16(binding.slot);
            writer.u8(static_cast<std::uint8_t>(binding.source_kind));
            writer.u64(binding.source_id);
        }
        write_metrics(writer, artifact.specification.maturity);
        writer.u8(static_cast<std::uint8_t>(artifact.state));
        writer.u8(static_cast<std::uint8_t>(artifact.audit_phase));
        writer.u64(artifact.audit.sequence_end);
        writer.u64(artifact.audit.observed);
        writer.u64(artifact.audit.mismatches);
    }

    writer.u64(snapshot.mapping_words.size());
    for (const auto& [source, encoded] : snapshot.mapping_words) {
        writer.u32(source);
        writer.u64(encoded);
    }
    return std::move(writer).take();
}

[[nodiscard]] ConsolidationRegistrySnapshot deserialize_registry(
    std::span<const std::uint8_t> bytes) {
    ByteReader reader(bytes);
    ConsolidationRegistrySnapshot result;
    result.schema_version = reader.u32();
    result.source_capacity = to_size(reader.u64(), "source capacity");
    result.artifact_capacity = to_size(reader.u64(), "artifact capacity");
    result.audit_window_size = to_size(reader.u64(), "audit window size");
    result.maturity_policy = MaturityPolicy{
        reader.f64(), reader.u64(), reader.f64(), reader.f64(),
        reader.u64(), reader.f64(),
    };
    result.audit_policy = AuditPolicy{
        to_size(reader.u64(), "shadow minimum observations"),
        reader.f64(),
        to_size(reader.u64(), "live minimum observations"),
        to_size(reader.u64(), "reopen minimum mismatches"),
        reader.f64(),
    };

    const auto artifact_count = to_size(reader.u64(), "artifact count");
    if (artifact_count > result.artifact_capacity) {
        corrupt("artifact count exceeds registry capacity");
    }
    result.artifacts.reserve(artifact_count);
    for (std::size_t index = 0; index < artifact_count; ++index) {
        ConsolidationSpec specification;
        specification.artifact_id = reader.string();
        specification.mapping_version = reader.string();
        specification.restoration_handle = reader.string();
        specification.input_bits = to_size(reader.u64(), "PA input bits");
        specification.port_semantic =
            static_cast<PortSemantic>(reader.u8());
        const auto binding_count = to_size(reader.u64(), "binding count");
        if (binding_count > specification.input_bits) {
            corrupt("binding count exceeds PA input shape");
        }
        specification.bindings.reserve(binding_count);
        for (std::size_t binding = 0; binding < binding_count; ++binding) {
            specification.bindings.push_back(SlotBinding{
                reader.u16(), static_cast<SourceKind>(reader.u8()), reader.u64()});
        }
        specification.maturity = read_metrics(reader);
        result.artifacts.push_back(ArtifactCheckpoint{
            std::move(specification),
            static_cast<ConsolidationState>(reader.u8()),
            static_cast<ConsolidationState>(reader.u8()),
            AuditSnapshot{
                reader.u64(),
                to_size(reader.u64(), "audit observations"),
                to_size(reader.u64(), "audit mismatches"),
            },
        });
    }

    const auto mapping_count = to_size(reader.u64(), "mapping count");
    if (mapping_count > result.source_capacity) {
        corrupt("mapping count exceeds source capacity");
    }
    result.mapping_words.reserve(mapping_count);
    for (std::size_t index = 0; index < mapping_count; ++index) {
        const auto source = reader.u32();
        const auto encoded = reader.u64();
        result.mapping_words.emplace_back(source, encoded);
    }
    reader.require_end();
    return result;
}

[[nodiscard]] FILE* open_file(const std::filesystem::path& path,
                              const char* mode) {
#ifdef _WIN32
    std::wstring wide_mode;
    while (*mode != '\0') {
        wide_mode.push_back(static_cast<wchar_t>(*mode++));
    }
    FILE* file = nullptr;
    if (_wfopen_s(&file, path.c_str(), wide_mode.c_str()) != 0) {
        return nullptr;
    }
    return file;
#else
    return std::fopen(path.c_str(), mode);
#endif
}

void ensure_parent(const std::filesystem::path& path) {
    const auto parent = path.parent_path();
    if (parent.empty()) {
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error) {
        io_failure("could not create persistence directory: " + error.message());
    }
}

void write_file_synced(const std::filesystem::path& path,
                       std::span<const std::uint8_t> bytes,
                       const char* mode) {
    auto* file = open_file(path, mode);
    if (file == nullptr) {
        io_failure("could not open persistence file: " + path.string());
    }
    const auto written = bytes.empty()
                             ? std::size_t{0}
                             : std::fwrite(bytes.data(), 1, bytes.size(), file);
    bool ok = written == bytes.size() && std::fflush(file) == 0;
#ifdef _WIN32
    ok = ok && _commit(_fileno(file)) == 0;
#else
    ok = ok && ::fsync(::fileno(file)) == 0;
#endif
    ok = std::fclose(file) == 0 && ok;
    if (!ok) {
        io_failure("could not durably write persistence file: " + path.string());
    }
}

void replace_file(const std::filesystem::path& temporary,
                  const std::filesystem::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        io_failure("could not atomically replace persistence file: " +
                   destination.string());
    }
#else
    if (::rename(temporary.c_str(), destination.c_str()) != 0) {
        io_failure("could not atomically replace persistence file: " +
                   destination.string());
    }
    const auto parent = destination.parent_path().empty()
                            ? std::filesystem::path{"."}
                            : destination.parent_path();
    const auto directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
    if (directory >= 0) {
        static_cast<void>(::fsync(directory));
        static_cast<void>(::close(directory));
    }
#endif
}

void atomic_write(const std::filesystem::path& path,
                  std::span<const std::uint8_t> bytes) {
    ensure_parent(path);
    auto temporary = path;
    temporary += ".tmp";
    try {
        write_file_synced(temporary, bytes, "wb");
        replace_file(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

[[nodiscard]] Bytes read_file(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        io_failure("could not inspect persistence file: " + path.string());
    }
    if (size > maximum_persistence_file_size) {
        corrupt("persistence file exceeds the configured safety bound");
    }
    Bytes result(static_cast<std::size_t>(size));
    std::ifstream input(path, std::ios::binary);
    if (!input ||
        (!result.empty() &&
         !input.read(reinterpret_cast<char*>(result.data()),
                     static_cast<std::streamsize>(result.size())))) {
        io_failure("could not read persistence file: " + path.string());
    }
    return result;
}

[[nodiscard]] bool all_zero(const PersistenceDigest& digest) noexcept {
    return std::all_of(digest.begin(), digest.end(),
                       [](std::uint8_t value) { return value == 0; });
}

void validate_image(const DurableRegistryImage& image) {
    if (image.schema_version != class_ii_persistence_schema_version) {
        throw PersistenceError(PersistenceErrorCode::unsupported_version,
                               "unsupported Class II persistence version");
    }
    if ((image.sequence == 0 && !all_zero(image.last_event_digest)) ||
        (image.sequence != 0 && all_zero(image.last_event_digest))) {
        throw PersistenceError(PersistenceErrorCode::inconsistent_state,
                               "event sequence and digest anchor disagree");
    }
    try {
        static_cast<void>(ConsolidationRegistry::restore(image.registry));
    } catch (const std::invalid_argument& error) {
        throw PersistenceError(PersistenceErrorCode::inconsistent_state,
                               error.what());
    }
}

[[nodiscard]] Bytes serialize_snapshot_envelope(
    const DurableRegistryImage& image) {
    const auto payload = serialize_registry(image.registry);
    ByteWriter writer;
    writer.raw(snapshot_magic);
    writer.u32(image.schema_version);
    writer.u64(image.sequence);
    writer.raw(image.last_event_digest);
    writer.u64(payload.size());
    writer.raw(payload);
    const auto digest = sha256(writer.bytes());
    writer.raw(digest);
    return std::move(writer).take();
}

[[nodiscard]] DurableRegistryImage deserialize_snapshot_envelope(
    std::span<const std::uint8_t> bytes) {
    constexpr auto header_size = snapshot_magic.size() + sizeof(std::uint32_t) +
                                 sizeof(std::uint64_t) + digest_size +
                                 sizeof(std::uint64_t);
    if (bytes.size() < header_size + digest_size) {
        corrupt("snapshot envelope is truncated");
    }
    ByteReader reader(bytes);
    const auto magic = reader.raw(snapshot_magic.size());
    if (!std::equal(magic.begin(), magic.end(), snapshot_magic.begin())) {
        corrupt("snapshot magic is invalid");
    }
    const auto version = reader.u32();
    if (version != class_ii_persistence_schema_version) {
        throw PersistenceError(PersistenceErrorCode::unsupported_version,
                               "unsupported Class II snapshot version");
    }
    const auto sequence = reader.u64();
    PersistenceDigest anchor{};
    const auto anchor_bytes = reader.raw(anchor.size());
    std::copy(anchor_bytes.begin(), anchor_bytes.end(), anchor.begin());
    const auto payload_size = reader.u64();
    if (payload_size > reader.remaining() - digest_size ||
        payload_size != reader.remaining() - digest_size) {
        corrupt("snapshot payload size is inconsistent");
    }
    const auto payload = reader.raw(static_cast<std::size_t>(payload_size));
    PersistenceDigest stored_digest{};
    const auto stored = reader.raw(stored_digest.size());
    std::copy(stored.begin(), stored.end(), stored_digest.begin());
    reader.require_end();
    const auto calculated = sha256(bytes.first(bytes.size() - digest_size));
    if (calculated != stored_digest) {
        corrupt("snapshot SHA-256 integrity check failed");
    }
    DurableRegistryImage result{
        version, sequence, anchor, deserialize_registry(payload)};
    validate_image(result);
    return result;
}

struct ParsedEvent {
    std::uint64_t sequence{};
    PersistenceDigest previous{};
    PersistenceDigest digest{};
    std::string name;
    ConsolidationRegistrySnapshot registry;
    std::size_t end_offset{};
};

struct ParsedLog {
    std::vector<ParsedEvent> events;
    std::size_t valid_bytes{};
    std::size_t ignored_tail_bytes{};
};

[[nodiscard]] ParsedLog parse_log(std::span<const std::uint8_t> bytes) {
    ParsedLog result;
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto remaining = bytes.size() - offset;
        if (remaining < event_fixed_size) {
            result.ignored_tail_bytes = remaining;
            break;
        }
        ByteReader reader(bytes.subspan(offset));
        const auto magic = reader.raw(event_magic.size());
        if (!std::equal(magic.begin(), magic.end(), event_magic.begin())) {
            corrupt("event-log magic is invalid");
        }
        const auto version = reader.u32();
        if (version != class_ii_persistence_schema_version) {
            throw PersistenceError(PersistenceErrorCode::unsupported_version,
                                   "unsupported Class II event version");
        }
        const auto sequence = reader.u64();
        PersistenceDigest previous{};
        const auto previous_bytes = reader.raw(previous.size());
        std::copy(previous_bytes.begin(), previous_bytes.end(), previous.begin());
        const auto payload_size = reader.u64();
        const auto name_size = reader.u32();
        if (name_size == 0 || name_size > maximum_string_size ||
            payload_size > maximum_persistence_file_size) {
            corrupt("event-log field exceeds its safety bound");
        }
        const auto variable_size = static_cast<std::uint64_t>(name_size) +
                                   payload_size + digest_size;
        if (variable_size > reader.remaining()) {
            result.ignored_tail_bytes = remaining;
            break;
        }
        const auto name_bytes = reader.raw(name_size);
        const auto payload = reader.raw(static_cast<std::size_t>(payload_size));
        PersistenceDigest stored_digest{};
        const auto stored = reader.raw(stored_digest.size());
        std::copy(stored.begin(), stored.end(), stored_digest.begin());
        const auto frame_size = event_fixed_size + name_size +
                                static_cast<std::size_t>(payload_size);
        const auto calculated = sha256(bytes.subspan(offset,
            frame_size - digest_size));
        if (calculated != stored_digest) {
            corrupt("event-log SHA-256 integrity check failed");
        }
        const auto end = offset + frame_size;
        result.events.push_back(ParsedEvent{
            sequence,
            previous,
            stored_digest,
            std::string(reinterpret_cast<const char*>(name_bytes.data()),
                        name_bytes.size()),
            deserialize_registry(payload),
            end,
        });
        offset = end;
        result.valid_bytes = end;
    }
    for (std::size_t index = 1; index < result.events.size(); ++index) {
        if (result.events[index].sequence !=
                result.events[index - 1].sequence + 1U ||
            result.events[index].previous != result.events[index - 1].digest) {
            throw PersistenceError(PersistenceErrorCode::sequence_conflict,
                                   "event-log hash chain is discontinuous");
        }
    }
    return result;
}

[[nodiscard]] Bytes serialize_event(std::uint64_t sequence,
                                    const PersistenceDigest& previous,
                                    std::string_view event_name,
                                    const ConsolidationRegistrySnapshot& registry,
                                    PersistenceDigest& digest) {
    if (event_name.empty() || event_name.size() > maximum_string_size) {
        throw PersistenceError(PersistenceErrorCode::inconsistent_state,
                               "event name is empty or too large");
    }
    const auto payload = serialize_registry(registry);
    ByteWriter writer;
    writer.raw(event_magic);
    writer.u32(class_ii_persistence_schema_version);
    writer.u64(sequence);
    writer.raw(previous);
    writer.u64(payload.size());
    writer.u32(static_cast<std::uint32_t>(event_name.size()));
    writer.raw(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(event_name.data()),
        event_name.size()));
    writer.raw(payload);
    digest = sha256(writer.bytes());
    writer.raw(digest);
    return std::move(writer).take();
}

[[nodiscard]] bool path_exists(const std::filesystem::path& path) {
    std::error_code error;
    const bool result = std::filesystem::exists(path, error);
    if (error) {
        io_failure("could not inspect persistence path: " + error.message());
    }
    return result;
}

}  // namespace

DurableRegistryImage ClassIIPersistence::capture(
    const ConsolidationRegistry& registry) {
    return DurableRegistryImage{
        class_ii_persistence_schema_version,
        0,
        {},
        registry.checkpoint(),
    };
}

void ClassIIPersistence::write_snapshot_atomic(
    const std::filesystem::path& path,
    const DurableRegistryImage& image) {
    validate_image(image);
    const auto bytes = serialize_snapshot_envelope(image);
    atomic_write(path, bytes);
}

DurableRegistryImage ClassIIPersistence::read_snapshot(
    const std::filesystem::path& path) {
    return deserialize_snapshot_envelope(read_file(path));
}

DurableRegistryImage ClassIIPersistence::append_event(
    const std::filesystem::path& event_log,
    const DurableRegistryImage& prior,
    const ConsolidationRegistry& registry,
    std::string_view event_name) {
    validate_image(prior);
    if (prior.sequence == std::numeric_limits<std::uint64_t>::max()) {
        throw PersistenceError(PersistenceErrorCode::sequence_conflict,
                               "event sequence is exhausted");
    }

    Bytes existing;
    if (path_exists(event_log)) {
        existing = read_file(event_log);
    }
    const auto parsed = parse_log(existing);
    if (!parsed.events.empty()) {
        const auto& last = parsed.events.back();
        if (last.sequence != prior.sequence ||
            last.digest != prior.last_event_digest) {
            throw PersistenceError(PersistenceErrorCode::sequence_conflict,
                                   "event log does not end at the supplied commit token");
        }
    }
    if (parsed.ignored_tail_bytes != 0) {
        atomic_write(event_log,
                     std::span<const std::uint8_t>(existing).first(
                         parsed.valid_bytes));
    } else {
        ensure_parent(event_log);
    }

    auto checkpoint = registry.checkpoint();
    PersistenceDigest digest{};
    const auto frame = serialize_event(prior.sequence + 1U,
                                       prior.last_event_digest,
                                       event_name,
                                       checkpoint,
                                       digest);
    write_file_synced(event_log, frame, "ab");
    DurableRegistryImage result{
        class_ii_persistence_schema_version,
        prior.sequence + 1U,
        digest,
        std::move(checkpoint),
    };
    validate_image(result);
    return result;
}

RegistryReplayResult ClassIIPersistence::recover(
    const std::filesystem::path& snapshot,
    const std::filesystem::path& event_log) {
    const bool have_snapshot = path_exists(snapshot);
    const bool have_log = path_exists(event_log);
    if (!have_snapshot && !have_log) {
        io_failure("neither a Class II snapshot nor event log exists");
    }

    DurableRegistryImage current;
    if (have_snapshot) {
        current = read_snapshot(snapshot);
    }
    const auto log_bytes = have_log ? read_file(event_log) : Bytes{};
    const auto parsed = parse_log(log_bytes);
    std::size_t start = parsed.events.size();

    if (!have_snapshot) {
        if (parsed.events.empty() || parsed.events.front().sequence != 1U ||
            !all_zero(parsed.events.front().previous)) {
            throw PersistenceError(PersistenceErrorCode::sequence_conflict,
                                   "event log has no recoverable origin");
        }
        current = DurableRegistryImage{
            class_ii_persistence_schema_version, 0, {}, {}};
        start = 0;
    } else {
        for (std::size_t index = 0; index < parsed.events.size(); ++index) {
            const auto& event = parsed.events[index];
            if (event.sequence == current.sequence &&
                event.digest != current.last_event_digest) {
                throw PersistenceError(PersistenceErrorCode::sequence_conflict,
                                       "snapshot anchor disagrees with event log");
            }
            if (event.sequence == current.sequence + 1U &&
                event.previous == current.last_event_digest) {
                start = index;
                break;
            }
            if (event.sequence > current.sequence) {
                throw PersistenceError(PersistenceErrorCode::sequence_conflict,
                                       "event log does not continue the snapshot anchor");
            }
        }
    }

    std::size_t applied = 0;
    for (std::size_t index = start; index < parsed.events.size(); ++index) {
        const auto& event = parsed.events[index];
        if (event.sequence != current.sequence + 1U ||
            event.previous != current.last_event_digest) {
            throw PersistenceError(PersistenceErrorCode::sequence_conflict,
                                   "event replay encountered a discontinuity");
        }
        current = DurableRegistryImage{
            class_ii_persistence_schema_version,
            event.sequence,
            event.digest,
            event.registry,
        };
        validate_image(current);
        ++applied;
    }
    if (!have_snapshot && applied == 0) {
        throw PersistenceError(PersistenceErrorCode::sequence_conflict,
                               "event log contains no complete initial event");
    }
    return RegistryReplayResult{
        std::move(current),
        applied,
        parsed.ignored_tail_bytes,
        parsed.valid_bytes,
    };
}

void ClassIIPersistence::compact(
    const std::filesystem::path& snapshot,
    const std::filesystem::path& event_log,
    const DurableRegistryImage& image) {
    write_snapshot_atomic(snapshot, image);
    const Bytes empty;
    atomic_write(event_log, empty);
}

std::unique_ptr<ConsolidationRegistry> ClassIIPersistence::restore(
    const DurableRegistryImage& image) {
    validate_image(image);
    try {
        return ConsolidationRegistry::restore(image.registry);
    } catch (const std::invalid_argument& error) {
        throw PersistenceError(PersistenceErrorCode::inconsistent_state,
                               error.what());
    }
}

std::string persistence_digest_hex(const PersistenceDigest& digest) {
    constexpr char alphabet[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2U);
    for (const auto value : digest) {
        result.push_back(alphabet[value >> 4U]);
        result.push_back(alphabet[value & 0x0fU]);
    }
    return result;
}

}  // namespace ptm
