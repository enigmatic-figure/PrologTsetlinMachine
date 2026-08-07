#ifndef PTM_C_API_H
#define PTM_C_API_H

#include <stdint.h>

#define PTM_ABI_VERSION 2u

#if defined(_WIN32)
#if defined(PTM_C_API_BUILD)
#define PTM_API __declspec(dllexport)
#else
#define PTM_API __declspec(dllimport)
#endif
#else
#define PTM_API
#endif

#if defined(__cplusplus)
#define PTM_ALIGNAS(bytes) alignas(bytes)
#elif defined(_MSC_VER)
#define PTM_ALIGNAS(bytes) __declspec(align(bytes))
#else
#define PTM_ALIGNAS(bytes) _Alignas(bytes)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ptm_status {
    PTM_STATUS_OK = 0,
    PTM_STATUS_NULL_POINTER = 1,
    PTM_STATUS_INVALID_SEMANTIC = 2,
    PTM_STATUS_INVALID_THRESHOLD = 3,
    PTM_STATUS_INTERNAL_ERROR = 4,
    PTM_STATUS_INVALID_PROGRAM = 5,
    PTM_STATUS_INVALID_BINDINGS = 6,
    PTM_STATUS_INVALID_DIMENSIONS = 7,
    PTM_STATUS_INVALID_STATE = 8,
    PTM_STATUS_INSUFFICIENT_CAPACITY = 9,
    PTM_STATUS_BACKEND_UNAVAILABLE = 10
} ptm_status;

typedef enum ptm_tm_backend {
    PTM_TM_BACKEND_AUTOMATIC = 0,
    PTM_TM_BACKEND_SCALAR = 1,
    PTM_TM_BACKEND_AVX2 = 2,
    PTM_TM_BACKEND_AVX512 = 3
} ptm_tm_backend;

typedef enum ptm_cpu_capability_flag {
    PTM_CPU_CAP_X86 = 1u << 0,
    PTM_CPU_CAP_OS_XSAVE = 1u << 1,
    PTM_CPU_CAP_AVX = 1u << 2,
    PTM_CPU_CAP_AVX2 = 1u << 3,
    PTM_CPU_CAP_AVX512F = 1u << 4
} ptm_cpu_capability_flag;

typedef struct ptm_cpu_capabilities {
    uint64_t hardware_flags;
    uint64_t compiled_flags;
    uint32_t preferred_backend;
    uint32_t reserved;
    char brand[104];
} ptm_cpu_capabilities;

typedef enum ptm_port_semantic {
    PTM_LITERAL_TRUTH = 0,
    PTM_TA_ACTION = 1,
    PTM_LITERAL_CONDITION = 2,
    PTM_CLAUSE_OUTPUT = 3
} ptm_port_semantic;

typedef struct ptm_bitblock_1024 {
    PTM_ALIGNAS(64) uint64_t words[16];
} ptm_bitblock_1024;

typedef struct ptm_bitblock_4096 {
    PTM_ALIGNAS(64) uint64_t words[64];
} ptm_bitblock_4096;

typedef struct ptm_threshold_result_1024 {
    uint8_t value;
    uint8_t reserved[3];
    uint32_t matched_count;
    uint32_t selected_count;
    uint8_t alignment_padding[52];
    ptm_bitblock_1024 matched;
    ptm_bitblock_1024 missing;
} ptm_threshold_result_1024;

typedef struct ptm_threshold_result_4096 {
    uint8_t value;
    uint8_t reserved[3];
    uint32_t matched_count;
    uint32_t selected_count;
    uint8_t alignment_padding[52];
    ptm_bitblock_4096 matched;
    ptm_bitblock_4096 missing;
} ptm_threshold_result_4096;

typedef enum ptm_logic_opcode {
    PTM_LOGIC_CONSTANT = 0,
    PTM_LOGIC_INPUT = 1,
    PTM_LOGIC_NOT = 2,
    PTM_LOGIC_AND = 3,
    PTM_LOGIC_OR = 4,
    PTM_LOGIC_XOR = 5
} ptm_logic_opcode;

typedef struct ptm_logic_instruction32 {
    uint32_t operand_mask;
    uint8_t opcode;
    uint8_t argument;
    uint16_t reserved;
} ptm_logic_instruction32;

typedef struct ptm_logic_program32 {
    PTM_ALIGNAS(64) uint32_t instruction_count;
    uint32_t root_instruction;
    ptm_logic_instruction32 instructions[32];
    uint8_t alignment_padding[56];
} ptm_logic_program32;

typedef struct ptm_logic_result32 {
    uint8_t value;
    uint8_t reserved[3];
    uint32_t true_instruction_mask;
    uint32_t evaluated_instruction_mask;
    uint32_t alignment_padding;
} ptm_logic_result32;

typedef struct ptm_tm_model_config {
    uint32_t number_of_clauses;
    uint32_t number_of_features;
    uint32_t states_per_action;
    int32_t threshold;
} ptm_tm_model_config;

typedef struct ptm_tm_batch64_result {
    PTM_ALIGNAS(64) uint64_t valid_example_mask;
    uint64_t prediction_mask;
    int32_t scores[64];
    uint8_t alignment_padding[48];
} ptm_tm_batch64_result;

typedef struct ptm_tm_model ptm_tm_model;

PTM_API uint32_t ptm_abi_version(void);
PTM_API const char* ptm_status_message(ptm_status status);
PTM_API ptm_status ptm_cpu_capabilities_query(
    ptm_cpu_capabilities* capabilities);

PTM_API ptm_status ptm_threshold_1024_eval(
    ptm_port_semantic semantic,
    const ptm_bitblock_1024* input,
    const ptm_bitblock_1024* selection,
    uint32_t minimum_true,
    ptm_threshold_result_1024* result);

PTM_API ptm_status ptm_threshold_4096_eval(
    ptm_port_semantic semantic,
    const ptm_bitblock_4096* input,
    const ptm_bitblock_4096* selection,
    uint32_t minimum_true,
    ptm_threshold_result_4096* result);

PTM_API ptm_status ptm_logic_program32_eval(
    const ptm_logic_program32* program,
    uint8_t binding_bits,
    ptm_logic_result32* result);

PTM_API ptm_status ptm_logic_program32_eval_batch(
    const ptm_logic_program32* programs,
    const uint8_t* binding_bits,
    uint32_t program_count,
    ptm_logic_result32* results);

// states are clause-major, with positive/negative literals interleaved for
// every feature. The immutable native model owns its packed copy.
PTM_API ptm_status ptm_tm_model_create(
    const ptm_tm_model_config* config,
    const uint16_t* states,
    uint64_t state_count,
    ptm_tm_model** model);

PTM_API void ptm_tm_model_destroy(ptm_tm_model* model);

PTM_API ptm_status ptm_tm_model_selected_backend(
    const ptm_tm_model* model,
    ptm_tm_backend* backend);

// feature_words[f] contains feature f for up to 64 example lanes. Prediction
// and feedback words are written per clause; scores are clamped to the TM limit.
PTM_API ptm_status ptm_tm_model_eval_packed64(
    const ptm_tm_model* model,
    const uint64_t* feature_words,
    uint32_t feature_word_count,
    uint64_t valid_example_mask,
    uint64_t* clause_outputs,
    uint64_t* feedback_clause_outputs,
    uint32_t clause_output_capacity,
    ptm_tm_batch64_result* result);

PTM_API ptm_status ptm_tm_model_eval_packed64_backend(
    const ptm_tm_model* model,
    const uint64_t* feature_words,
    uint32_t feature_word_count,
    uint64_t valid_example_mask,
    uint64_t* clause_outputs,
    uint64_t* feedback_clause_outputs,
    uint32_t clause_output_capacity,
    ptm_tm_backend backend,
    ptm_tm_batch64_result* result);

#ifdef __cplusplus
}
#endif

#undef PTM_ALIGNAS

#endif
