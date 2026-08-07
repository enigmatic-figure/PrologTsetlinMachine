#include "ptm/c_api.h"

#include "ptm/logic_program.hpp"
#include "ptm/packed_tm.hpp"
#include "ptm/pa_kernel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <string_view>

static_assert(sizeof(ptm_bitblock_1024) == 128);
static_assert(sizeof(ptm_bitblock_4096) == 512);
static_assert(alignof(ptm_bitblock_1024) == 64);
static_assert(alignof(ptm_bitblock_4096) == 64);
static_assert(offsetof(ptm_threshold_result_1024, matched) == 64);
static_assert(offsetof(ptm_threshold_result_1024, missing) == 192);
static_assert(sizeof(ptm_threshold_result_1024) == 320);
static_assert(offsetof(ptm_threshold_result_4096, matched) == 64);
static_assert(offsetof(ptm_threshold_result_4096, missing) == 576);
static_assert(sizeof(ptm_threshold_result_4096) == 1088);
static_assert(sizeof(ptm_logic_instruction32) == 8);
static_assert(offsetof(ptm_logic_program32, instructions) == 8);
static_assert(alignof(ptm_logic_program32) == 64);
static_assert(sizeof(ptm_logic_program32) == 320);
static_assert(sizeof(ptm_logic_result32) == 16);
static_assert(sizeof(ptm_tm_model_config) == 16);
static_assert(sizeof(ptm_cpu_capabilities) == 128);
static_assert(offsetof(ptm_tm_batch64_result, scores) == 16);
static_assert(alignof(ptm_tm_batch64_result) == 64);
static_assert(sizeof(ptm_tm_batch64_result) == 320);

struct ptm_tm_model {
    ptm_tm_model(std::size_t number_of_clauses,
                 std::size_t number_of_features,
                 std::uint16_t states_per_action,
                 int threshold,
                 std::span<const std::uint16_t> states)
        : value(number_of_clauses,
                number_of_features,
                states_per_action,
                threshold,
                states) {}

    ptm::PackedTMModel64 value;
};

namespace {

template <std::size_t Bits,
          ptm::PortSemantic Semantic,
          typename CBlock,
          typename CResult>
ptm_status evaluate_typed(const CBlock* input,
                          const CBlock* selection,
                          std::uint32_t minimum_true,
                          CResult* output) {
    using Block = ptm::TypedBitBlock<Bits, Semantic>;
    Block typed_input{};
    Block typed_selection{};
    std::copy_n(input->words, Block::word_count, typed_input.words.begin());
    std::copy_n(selection->words, Block::word_count, typed_selection.words.begin());

    if (minimum_true > typed_selection.population()) {
        return PTM_STATUS_INVALID_THRESHOLD;
    }

    const ptm::MaskedThresholdKernel<Bits, Semantic> kernel(
        typed_selection, minimum_true);
    const auto result = kernel.evaluate(typed_input);
    output->value = result.value ? 1U : 0U;
    output->reserved[0] = 0;
    output->reserved[1] = 0;
    output->reserved[2] = 0;
    output->matched_count = result.matched_count;
    output->selected_count = result.selected_count;
    std::fill_n(output->alignment_padding, 52, static_cast<std::uint8_t>(0));
    std::copy(result.matched.words.begin(), result.matched.words.end(),
              output->matched.words);
    std::copy(result.missing.words.begin(), result.missing.words.end(),
              output->missing.words);
    return PTM_STATUS_OK;
}

template <std::size_t Bits, typename CBlock, typename CResult>
ptm_status evaluate(ptm_port_semantic semantic,
                    const CBlock* input,
                    const CBlock* selection,
                    std::uint32_t minimum_true,
                    CResult* result) noexcept {
    if (input == nullptr || selection == nullptr || result == nullptr) {
        return PTM_STATUS_NULL_POINTER;
    }
    try {
        switch (semantic) {
            case PTM_LITERAL_TRUTH:
                return evaluate_typed<Bits, ptm::PortSemantic::literal_truth>(
                    input, selection, minimum_true, result);
            case PTM_TA_ACTION:
                return evaluate_typed<Bits, ptm::PortSemantic::ta_action>(
                    input, selection, minimum_true, result);
            case PTM_LITERAL_CONDITION:
                return evaluate_typed<Bits, ptm::PortSemantic::literal_condition>(
                    input, selection, minimum_true, result);
            case PTM_CLAUSE_OUTPUT:
                return evaluate_typed<Bits, ptm::PortSemantic::clause_output>(
                    input, selection, minimum_true, result);
        }
        return PTM_STATUS_INVALID_SEMANTIC;
    } catch (const std::exception&) {
        return PTM_STATUS_INTERNAL_ERROR;
    } catch (...) {
        return PTM_STATUS_INTERNAL_ERROR;
    }
}

ptm_status map_logic_status(ptm::FixedLogicStatus status) noexcept {
    if (status == ptm::FixedLogicStatus::ok) {
        return PTM_STATUS_OK;
    }
    if (status == ptm::FixedLogicStatus::invalid_bindings) {
        return PTM_STATUS_INVALID_BINDINGS;
    }
    return PTM_STATUS_INVALID_PROGRAM;
}

ptm::FixedLogicProgram32 copy_logic_program(
    const ptm_logic_program32& source) noexcept {
    ptm::FixedLogicProgram32 result{};
    result.instruction_count = source.instruction_count;
    result.root_instruction = source.root_instruction;
    for (std::size_t index = 0; index < result.instructions.size(); ++index) {
        result.instructions[index] = ptm::FixedLogicInstruction{
            source.instructions[index].operand_mask,
            static_cast<ptm::FixedLogicOp>(source.instructions[index].opcode),
            source.instructions[index].argument,
            source.instructions[index].reserved,
        };
    }
    return result;
}

void copy_logic_result(const ptm::FixedLogicResult32& source,
                       ptm_logic_result32& result) noexcept {
    result.value = source.value;
    result.reserved[0] = 0;
    result.reserved[1] = 0;
    result.reserved[2] = 0;
    result.true_instruction_mask = source.true_instruction_mask;
    result.evaluated_instruction_mask = source.evaluated_instruction_mask;
    result.alignment_padding = 0;
}

bool copy_backend(ptm_tm_backend source,
                  ptm::PackedTMBackend& destination) noexcept {
    switch (source) {
        case PTM_TM_BACKEND_AUTOMATIC:
            destination = ptm::PackedTMBackend::automatic;
            return true;
        case PTM_TM_BACKEND_SCALAR:
            destination = ptm::PackedTMBackend::scalar;
            return true;
        case PTM_TM_BACKEND_AVX2:
            destination = ptm::PackedTMBackend::avx2;
            return true;
        case PTM_TM_BACKEND_AVX512:
            destination = ptm::PackedTMBackend::avx512;
            return true;
    }
    return false;
}

ptm_tm_backend copy_backend(ptm::PackedTMBackend source) noexcept {
    switch (source) {
        case ptm::PackedTMBackend::automatic:
            return PTM_TM_BACKEND_AUTOMATIC;
        case ptm::PackedTMBackend::scalar:
            return PTM_TM_BACKEND_SCALAR;
        case ptm::PackedTMBackend::avx2:
            return PTM_TM_BACKEND_AVX2;
        case ptm::PackedTMBackend::avx512:
            return PTM_TM_BACKEND_AVX512;
    }
    return PTM_TM_BACKEND_SCALAR;
}

}  // namespace

extern "C" {

std::uint32_t ptm_abi_version(void) { return PTM_ABI_VERSION; }

const char* ptm_status_message(ptm_status status) {
    switch (status) {
        case PTM_STATUS_OK:
            return "ok";
        case PTM_STATUS_NULL_POINTER:
            return "null pointer";
        case PTM_STATUS_INVALID_SEMANTIC:
            return "invalid port semantic";
        case PTM_STATUS_INVALID_THRESHOLD:
            return "minimum_true exceeds selected slot count";
        case PTM_STATUS_INTERNAL_ERROR:
            return "internal error";
        case PTM_STATUS_INVALID_PROGRAM:
            return "invalid fixed logic program";
        case PTM_STATUS_INVALID_BINDINGS:
            return "binding bits lie outside A through E";
        case PTM_STATUS_INVALID_DIMENSIONS:
            return "TM dimensions or packed input width are invalid";
        case PTM_STATUS_INVALID_STATE:
            return "TA state lies outside its configured action regions";
        case PTM_STATUS_INSUFFICIENT_CAPACITY:
            return "caller-owned output buffer is too small";
        case PTM_STATUS_BACKEND_UNAVAILABLE:
            return "requested TM backend is unavailable";
    }
    return "unknown status";
}

ptm_status ptm_cpu_capabilities_query(
    ptm_cpu_capabilities* capabilities) {
    if (capabilities == nullptr) {
        return PTM_STATUS_NULL_POINTER;
    }
    const auto& native = ptm::cpu_capabilities();
    capabilities->hardware_flags = 0;
    capabilities->compiled_flags = 0;
    capabilities->reserved = 0;
    if (native.x86) {
        capabilities->hardware_flags |= PTM_CPU_CAP_X86;
    }
    if (native.os_xsave) {
        capabilities->hardware_flags |= PTM_CPU_CAP_OS_XSAVE;
    }
    if (native.avx) {
        capabilities->hardware_flags |= PTM_CPU_CAP_AVX;
    }
    if (native.avx2) {
        capabilities->hardware_flags |= PTM_CPU_CAP_AVX2;
    }
    if (native.avx512f) {
        capabilities->hardware_flags |= PTM_CPU_CAP_AVX512F;
    }
    if (native.compiled_avx2) {
        capabilities->compiled_flags |= PTM_CPU_CAP_AVX2;
    }
    if (native.compiled_avx512) {
        capabilities->compiled_flags |= PTM_CPU_CAP_AVX512F;
    }
    const auto preferred =
        ptm::packed_tm_backend_available(ptm::PackedTMBackend::avx512)
            ? ptm::PackedTMBackend::avx512
            : ptm::packed_tm_backend_available(ptm::PackedTMBackend::avx2)
                  ? ptm::PackedTMBackend::avx2
                  : ptm::PackedTMBackend::scalar;
    capabilities->preferred_backend = copy_backend(preferred);
    std::fill(std::begin(capabilities->brand),
              std::end(capabilities->brand), '\0');
    const auto brand = std::string_view(native.brand);
    std::copy_n(brand.begin(),
                std::min(brand.size(), sizeof(capabilities->brand) - 1U),
                capabilities->brand);
    return PTM_STATUS_OK;
}

ptm_status ptm_threshold_1024_eval(ptm_port_semantic semantic,
                                   const ptm_bitblock_1024* input,
                                   const ptm_bitblock_1024* selection,
                                   std::uint32_t minimum_true,
                                   ptm_threshold_result_1024* result) {
    return evaluate<1024>(semantic, input, selection, minimum_true, result);
}

ptm_status ptm_threshold_4096_eval(ptm_port_semantic semantic,
                                   const ptm_bitblock_4096* input,
                                   const ptm_bitblock_4096* selection,
                                   std::uint32_t minimum_true,
                                   ptm_threshold_result_4096* result) {
    return evaluate<4096>(semantic, input, selection, minimum_true, result);
}

ptm_status ptm_logic_program32_eval(const ptm_logic_program32* program,
                                    std::uint8_t binding_bits,
                                    ptm_logic_result32* result) {
    if (program == nullptr || result == nullptr) {
        return PTM_STATUS_NULL_POINTER;
    }
    try {
        const auto typed_program = copy_logic_program(*program);
        ptm::FixedLogicResult32 typed_result{};
        const auto status = ptm::evaluate_fixed_logic_program(
            typed_program, binding_bits, typed_result);
        if (status != ptm::FixedLogicStatus::ok) {
            return map_logic_status(status);
        }
        copy_logic_result(typed_result, *result);
        return PTM_STATUS_OK;
    } catch (const std::exception&) {
        return PTM_STATUS_INTERNAL_ERROR;
    } catch (...) {
        return PTM_STATUS_INTERNAL_ERROR;
    }
}

ptm_status ptm_logic_program32_eval_batch(
    const ptm_logic_program32* programs,
    const std::uint8_t* binding_bits,
    std::uint32_t program_count,
    ptm_logic_result32* results) {
    if (program_count == 0) {
        return PTM_STATUS_OK;
    }
    if (programs == nullptr || binding_bits == nullptr || results == nullptr) {
        return PTM_STATUS_NULL_POINTER;
    }
    for (std::uint32_t index = 0; index < program_count; ++index) {
        const auto status = ptm_logic_program32_eval(
            &programs[index], binding_bits[index], &results[index]);
        if (status != PTM_STATUS_OK) {
            return status;
        }
    }
    return PTM_STATUS_OK;
}

ptm_status ptm_tm_model_create(const ptm_tm_model_config* config,
                               const std::uint16_t* states,
                               std::uint64_t state_count,
                               ptm_tm_model** model) {
    if (config == nullptr || states == nullptr || model == nullptr) {
        return PTM_STATUS_NULL_POINTER;
    }
    *model = nullptr;
    if (config->number_of_clauses == 0 || config->number_of_features == 0 ||
        config->states_per_action == 0 ||
        config->states_per_action >
            std::numeric_limits<std::uint16_t>::max() / 2U ||
        config->threshold <= 0 ||
        state_count > std::numeric_limits<std::size_t>::max()) {
        return PTM_STATUS_INVALID_DIMENSIONS;
    }
    const auto clauses = static_cast<std::uint64_t>(config->number_of_clauses);
    const auto features = static_cast<std::uint64_t>(config->number_of_features);
    if (features > std::numeric_limits<std::uint64_t>::max() / clauses / 2U) {
        return PTM_STATUS_INVALID_DIMENSIONS;
    }
    const auto expected = clauses * features * 2U;
    if (state_count != expected) {
        return PTM_STATUS_INVALID_DIMENSIONS;
    }
    for (std::uint64_t index = 0; index < state_count; ++index) {
        if (states[index] == 0 ||
            states[index] > config->states_per_action * 2U) {
            return PTM_STATUS_INVALID_STATE;
        }
    }
    try {
        auto created = std::make_unique<ptm_tm_model>(
            config->number_of_clauses,
            config->number_of_features,
            static_cast<std::uint16_t>(config->states_per_action),
            config->threshold,
            std::span<const std::uint16_t>(
                states, static_cast<std::size_t>(state_count)));
        *model = created.release();
        return PTM_STATUS_OK;
    } catch (const std::invalid_argument&) {
        return PTM_STATUS_INVALID_DIMENSIONS;
    } catch (const std::exception&) {
        return PTM_STATUS_INTERNAL_ERROR;
    } catch (...) {
        return PTM_STATUS_INTERNAL_ERROR;
    }
}

void ptm_tm_model_destroy(ptm_tm_model* model) {
    delete model;
}

ptm_status ptm_tm_model_selected_backend(
    const ptm_tm_model* model,
    ptm_tm_backend* backend) {
    if (model == nullptr || backend == nullptr) {
        return PTM_STATUS_NULL_POINTER;
    }
    *backend = copy_backend(model->value.selected_backend());
    return PTM_STATUS_OK;
}

ptm_status ptm_tm_model_eval_packed64(
    const ptm_tm_model* model,
    const std::uint64_t* feature_words,
    std::uint32_t feature_word_count,
    std::uint64_t valid_example_mask,
    std::uint64_t* clause_outputs,
    std::uint64_t* feedback_clause_outputs,
    std::uint32_t clause_output_capacity,
    ptm_tm_batch64_result* result) {
    return ptm_tm_model_eval_packed64_backend(
        model, feature_words, feature_word_count, valid_example_mask,
        clause_outputs, feedback_clause_outputs, clause_output_capacity,
        PTM_TM_BACKEND_AUTOMATIC, result);
}

ptm_status ptm_tm_model_eval_packed64_backend(
    const ptm_tm_model* model,
    const std::uint64_t* feature_words,
    std::uint32_t feature_word_count,
    std::uint64_t valid_example_mask,
    std::uint64_t* clause_outputs,
    std::uint64_t* feedback_clause_outputs,
    std::uint32_t clause_output_capacity,
    ptm_tm_backend backend,
    ptm_tm_batch64_result* result) {
    if (model == nullptr || feature_words == nullptr ||
        clause_outputs == nullptr || feedback_clause_outputs == nullptr ||
        result == nullptr) {
        return PTM_STATUS_NULL_POINTER;
    }
    if (feature_word_count != model->value.number_of_features()) {
        return PTM_STATUS_INVALID_DIMENSIONS;
    }
    if (clause_output_capacity < model->value.number_of_clauses()) {
        return PTM_STATUS_INSUFFICIENT_CAPACITY;
    }
    ptm::PackedTMBackend native_backend{};
    if (!copy_backend(backend, native_backend)) {
        return PTM_STATUS_BACKEND_UNAVAILABLE;
    }
    if (native_backend != ptm::PackedTMBackend::automatic &&
        !ptm::packed_tm_backend_available(native_backend)) {
        return PTM_STATUS_BACKEND_UNAVAILABLE;
    }
    try {
        std::uint64_t prediction_mask = 0;
        model->value.evaluate_into(
            std::span<const std::uint64_t>(feature_words, feature_word_count),
            valid_example_mask,
            std::span<std::uint64_t>(
                clause_outputs, model->value.number_of_clauses()),
            std::span<std::uint64_t>(
                feedback_clause_outputs, model->value.number_of_clauses()),
            std::span<std::int32_t, ptm::packed_tm_batch_width>(result->scores),
            prediction_mask,
            native_backend);
        result->valid_example_mask = valid_example_mask;
        result->prediction_mask = prediction_mask;
        std::fill_n(result->alignment_padding, 48, std::uint8_t{0});
        return PTM_STATUS_OK;
    } catch (const std::invalid_argument&) {
        return PTM_STATUS_INVALID_DIMENSIONS;
    } catch (const std::exception&) {
        return PTM_STATUS_INTERNAL_ERROR;
    } catch (...) {
        return PTM_STATUS_INTERNAL_ERROR;
    }
}

}  // extern "C"
