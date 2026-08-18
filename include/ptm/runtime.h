#ifndef PTM_RUNTIME_H
#define PTM_RUNTIME_H

#include <stdint.h>

#define PTMRT_ABI_VERSION 2u
#define PTMRT_MAX_PORTS 4u
#define PTMRT_MAX_RANK 4u

#if defined(_WIN32) && defined(PTMRT_SHARED)
#if defined(PTMRT_BUILD)
#define PTMRT_API __declspec(dllexport)
#else
#define PTMRT_API __declspec(dllimport)
#endif
#else
#define PTMRT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ptmrt_status {
    PTMRT_STATUS_OK = 0,
    PTMRT_STATUS_NULL_POINTER = 1,
    PTMRT_STATUS_INVALID_ARGUMENT = 2,
    PTMRT_STATUS_IO_ERROR = 3,
    PTMRT_STATUS_INVALID_FORMAT = 4,
    PTMRT_STATUS_UNSUPPORTED_VERSION = 5,
    PTMRT_STATUS_UNSUPPORTED_MODEL = 6,
    PTMRT_STATUS_INTEGRITY_ERROR = 7,
    PTMRT_STATUS_INSUFFICIENT_CAPACITY = 8,
    PTMRT_STATUS_CONFORMANCE_FAILED = 9,
    PTMRT_STATUS_INTERNAL_ERROR = 10
} ptmrt_status;

typedef enum ptmrt_dtype {
    PTMRT_DTYPE_UINT64 = 1,
    PTMRT_DTYPE_INT32 = 2,
    PTMRT_DTYPE_UINT32 = 3
} ptmrt_dtype;

typedef enum ptmrt_port_direction {
    PTMRT_PORT_INPUT = 1,
    PTMRT_PORT_OUTPUT = 2
} ptmrt_port_direction;

typedef enum ptmrt_model_kind {
    PTMRT_MODEL_PACKED_TM_BINARY_V1 = 1,
    PTMRT_MODEL_LOGIC_PROGRAM32_V1 = 2,
    PTMRT_MODEL_MASKED_THRESHOLD_V1 = 3
} ptmrt_model_kind;

typedef enum ptmrt_record_value_kind {
    PTMRT_VALUE_NULL = 0,
    PTMRT_VALUE_BOOL = 1,
    PTMRT_VALUE_INT64 = 2,
    PTMRT_VALUE_FLOAT64 = 3,
    PTMRT_VALUE_UTF8 = 4
} ptmrt_record_value_kind;

typedef struct ptmrt_record_field {
    const char* name;
    uint32_t kind;
    uint32_t boolean_value;
    int64_t integer_value;
    double number_value;
    const char* string_data;
    uint64_t string_size;
} ptmrt_record_field;

typedef struct ptmrt_port_description {
    char name[32];
    char semantic[32];
    uint32_t direction;
    uint32_t dtype;
    uint32_t rank;
    uint32_t reserved;
    uint64_t shape[PTMRT_MAX_RANK];
} ptmrt_port_description;

typedef struct ptmrt_model_description {
    char artifact_id[72];
    char artifact_schema[32];
    uint32_t container_version;
    uint32_t model_kind;
    uint32_t payload_version;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t conformance_case_count;
    uint32_t number_of_clauses;
    uint32_t number_of_features;
    int32_t threshold;
    uint32_t instruction_count;
    uint32_t binding_count;
    uint32_t slot_count;
    uint32_t minimum_true;
    uint32_t selected_count;
    ptmrt_port_description inputs[PTMRT_MAX_PORTS];
    ptmrt_port_description outputs[PTMRT_MAX_PORTS];
} ptmrt_model_description;

typedef struct ptmrt_tensor_view {
    const char* name;
    void* data;
    uint64_t byte_size;
    uint32_t dtype;
    uint32_t rank;
    uint64_t shape[PTMRT_MAX_RANK];
} ptmrt_tensor_view;

typedef struct ptmrt_model ptmrt_model;

PTMRT_API uint32_t ptmrt_abi_version(void);
PTMRT_API const char* ptmrt_status_message(ptmrt_status status);

PTMRT_API ptmrt_status ptmrt_model_open_file(
    const char* path,
    ptmrt_model** model);

PTMRT_API ptmrt_status ptmrt_model_open_memory(
    const void* data,
    uint64_t size,
    ptmrt_model** model);

PTMRT_API void ptmrt_model_close(ptmrt_model* model);

PTMRT_API ptmrt_status ptmrt_model_describe(
    const ptmrt_model* model,
    ptmrt_model_description* description);

// Query with buffer=NULL and capacity=0 to obtain required_size, including the
// trailing NUL. A non-NULL buffer always receives UTF-8 JSON on success.
PTMRT_API ptmrt_status ptmrt_model_manifest_json(
    const ptmrt_model* model,
    char* buffer,
    uint64_t capacity,
    uint64_t* required_size);

PTMRT_API ptmrt_status ptmrt_model_verify(const ptmrt_model* model);

// Returns whether the artifact embeds ptm.preprocessing.v1.
PTMRT_API uint32_t ptmrt_model_has_preprocessing(const ptmrt_model* model);

// Materialize one typed raw record into one Boolean word per model input.
// Query with output_words=NULL and capacity=0 to obtain required_words.
// Every produced word is either 0 or 1 and can be passed to ptmrt_model_run
// as lane zero of the model's normal feature-major input.
PTMRT_API ptmrt_status ptmrt_model_preprocess_record(
    const ptmrt_model* model,
    const ptmrt_record_field* fields,
    uint32_t field_count,
    uint64_t* output_words,
    uint64_t capacity,
    uint64_t* required_words);

// Callers discover model-specific named tensor ports with ptmrt_model_describe.
// Every v1 executor consumes feature/binding/slot-major 64-lane words; output
// names and diagnostic tensors remain model-specific.
PTMRT_API ptmrt_status ptmrt_model_run(
    const ptmrt_model* model,
    const ptmrt_tensor_view* inputs,
    uint32_t input_count,
    ptmrt_tensor_view* outputs,
    uint32_t output_count);

#ifdef __cplusplus
}
#endif

#endif
