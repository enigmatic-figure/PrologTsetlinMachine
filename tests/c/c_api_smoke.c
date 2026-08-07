#include "ptm/c_api.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#if defined(_MSC_VER)
#define PTM_TEST_ALIGNOF(type) __alignof(type)
#else
#define PTM_TEST_ALIGNOF(type) _Alignof(type)
#endif

static void set_bit_1024(ptm_bitblock_1024* block, size_t slot) {
    block->words[slot / 64] |= (uint64_t)1 << (slot % 64);
}

int main(void) {
    ptm_bitblock_1024 input = {{0}};
    ptm_bitblock_1024 selection = {{0}};
    ptm_threshold_result_1024 result;
    ptm_cpu_capabilities capabilities;

    if (ptm_abi_version() != PTM_ABI_VERSION) {
        fprintf(stderr, "unexpected PTM ABI version\n");
        return 1;
    }
    if (sizeof(ptm_bitblock_1024) != 128 || sizeof(ptm_bitblock_4096) != 512 ||
        PTM_TEST_ALIGNOF(ptm_bitblock_1024) != 64 ||
        PTM_TEST_ALIGNOF(ptm_bitblock_4096) != 64) {
        fprintf(stderr, "fixed PA payload has the wrong size\n");
        return 2;
    }
    if (sizeof(ptm_cpu_capabilities) != 128 ||
        ptm_cpu_capabilities_query(&capabilities) != PTM_STATUS_OK ||
        capabilities.brand[0] == '\0' ||
        capabilities.preferred_backend < PTM_TM_BACKEND_SCALAR ||
        capabilities.preferred_backend > PTM_TM_BACKEND_AVX512) {
        fprintf(stderr, "C ABI CPU capability query failed\n");
        return 15;
    }
    if (offsetof(ptm_threshold_result_1024, matched) != 64 ||
        offsetof(ptm_threshold_result_1024, missing) != 192 ||
        sizeof(ptm_threshold_result_1024) != 320 ||
        offsetof(ptm_threshold_result_4096, matched) != 64 ||
        offsetof(ptm_threshold_result_4096, missing) != 576 ||
        sizeof(ptm_threshold_result_4096) != 1088) {
        fprintf(stderr, "C ABI result layout has changed\n");
        return 6;
    }
    if (sizeof(ptm_logic_instruction32) != 8 ||
        offsetof(ptm_logic_program32, instructions) != 8 ||
        PTM_TEST_ALIGNOF(ptm_logic_program32) != 64 ||
        sizeof(ptm_logic_program32) != 320 ||
        sizeof(ptm_logic_result32) != 16) {
        fprintf(stderr, "fixed logic program has the wrong ABI layout\n");
        return 7;
    }
    if (sizeof(ptm_tm_model_config) != 16 ||
        offsetof(ptm_tm_batch64_result, scores) != 16 ||
        PTM_TEST_ALIGNOF(ptm_tm_batch64_result) != 64 ||
        sizeof(ptm_tm_batch64_result) != 320) {
        fprintf(stderr, "packed TM result has the wrong ABI layout\n");
        return 10;
    }

    set_bit_1024(&selection, 1);
    set_bit_1024(&selection, 7);
    set_bit_1024(&selection, 70);
    set_bit_1024(&input, 1);
    set_bit_1024(&input, 70);

    if (ptm_threshold_1024_eval(
            PTM_TA_ACTION, &input, &selection, 2, &result) != PTM_STATUS_OK) {
        fprintf(stderr, "C ABI threshold evaluation failed\n");
        return 3;
    }
    if (!result.value || result.matched_count != 2 ||
        result.selected_count != 3 || (result.missing.words[0] & (1ULL << 7)) == 0) {
        fprintf(stderr, "C ABI threshold result is incorrect\n");
        return 4;
    }
    if (ptm_threshold_1024_eval(
            PTM_TA_ACTION, &input, &selection, 4, &result) !=
        PTM_STATUS_INVALID_THRESHOLD) {
        fprintf(stderr, "C ABI did not reject an invalid threshold\n");
        return 5;
    }

    {
        ptm_logic_program32 program = {0};
        ptm_logic_result32 logic_result;
        program.instruction_count = 3;
        program.root_instruction = 2;
        program.instructions[0].opcode = PTM_LOGIC_INPUT;
        program.instructions[0].argument = 0;
        program.instructions[1].opcode = PTM_LOGIC_INPUT;
        program.instructions[1].argument = 1;
        program.instructions[2].opcode = PTM_LOGIC_XOR;
        program.instructions[2].operand_mask = (1U << 0) | (1U << 1);
        if (ptm_logic_program32_eval(&program, 1U, &logic_result) !=
                PTM_STATUS_OK ||
            !logic_result.value || logic_result.evaluated_instruction_mask != 7U) {
            fprintf(stderr, "C ABI fixed logic evaluation failed\n");
            return 8;
        }
        program.instructions[2].operand_mask |= 1U << 4;
        if (ptm_logic_program32_eval(&program, 1U, &logic_result) !=
            PTM_STATUS_INVALID_PROGRAM) {
            fprintf(stderr, "C ABI accepted a malformed fixed logic program\n");
            return 9;
        }
    }

    {
        ptm_tm_model_config config = {4, 2, 3, 8};
        uint16_t states[16];
        uint64_t features[2] = {0xC, 0xA};
        uint64_t clauses[4] = {0};
        uint64_t feedback_clauses[4] = {0};
        ptm_tm_batch64_result tm_result;
        ptm_tm_backend selected_backend = PTM_TM_BACKEND_AUTOMATIC;
        ptm_tm_model* model = NULL;
        size_t index;
        for (index = 0; index < 16; ++index) {
            states[index] = 3;
        }
        states[0 * 4 + 0] = 4;
        states[0 * 4 + 3] = 4;
        states[1 * 4 + 0] = 4;
        states[1 * 4 + 2] = 4;
        states[2 * 4 + 1] = 4;
        states[2 * 4 + 2] = 4;
        states[3 * 4 + 1] = 4;
        states[3 * 4 + 3] = 4;
        if (ptm_tm_model_create(&config, states, 16, &model) !=
                PTM_STATUS_OK ||
            model == NULL) {
            fprintf(stderr, "C ABI packed TM creation failed\n");
            return 11;
        }
        if (ptm_tm_model_eval_packed64(
                model, features, 2, 0xF, clauses, feedback_clauses, 3,
                &tm_result) !=
            PTM_STATUS_INSUFFICIENT_CAPACITY) {
            fprintf(stderr, "C ABI packed TM ignored clause capacity\n");
            ptm_tm_model_destroy(model);
            return 12;
        }
        if (ptm_tm_model_eval_packed64(
                model, features, 2, 0xF, clauses, feedback_clauses, 4,
                &tm_result) !=
                PTM_STATUS_OK ||
            tm_result.valid_example_mask != 0xF ||
            tm_result.prediction_mask != 0x6 ||
            clauses[0] != 0x4 || clauses[1] != 0x8 ||
            clauses[2] != 0x2 || clauses[3] != 0x1 ||
            tm_result.scores[0] != -1 || tm_result.scores[1] != 1 ||
            tm_result.scores[2] != 1 || tm_result.scores[3] != -1) {
            fprintf(stderr, "C ABI packed TM result is incorrect\n");
            ptm_tm_model_destroy(model);
            return 13;
        }
        if (ptm_tm_model_selected_backend(model, &selected_backend) !=
                PTM_STATUS_OK ||
            selected_backend == PTM_TM_BACKEND_AUTOMATIC) {
            fprintf(stderr, "C ABI packed TM backend selection failed\n");
            ptm_tm_model_destroy(model);
            return 16;
        }
        if (ptm_tm_model_eval_packed64_backend(
                model, features, 2, 0xF, clauses, feedback_clauses, 4,
                PTM_TM_BACKEND_SCALAR, &tm_result) != PTM_STATUS_OK ||
            tm_result.prediction_mask != 0x6) {
            fprintf(stderr, "C ABI forced scalar evaluation failed\n");
            ptm_tm_model_destroy(model);
            return 17;
        }
        if ((capabilities.hardware_flags & PTM_CPU_CAP_AVX512F) == 0 ||
            (capabilities.compiled_flags & PTM_CPU_CAP_AVX512F) == 0) {
            if (ptm_tm_model_eval_packed64_backend(
                    model, features, 2, 0xF, clauses, feedback_clauses, 4,
                    PTM_TM_BACKEND_AVX512, &tm_result) !=
                PTM_STATUS_BACKEND_UNAVAILABLE) {
                fprintf(stderr, "C ABI accepted unavailable AVX-512\n");
                ptm_tm_model_destroy(model);
                return 18;
            }
        }
        ptm_tm_model_destroy(model);
        states[0] = 0;
        model = NULL;
        if (ptm_tm_model_create(&config, states, 16, &model) !=
                PTM_STATUS_INVALID_STATE ||
            model != NULL) {
            fprintf(stderr, "C ABI packed TM accepted an invalid TA state\n");
            return 14;
        }
    }

    puts("PTM C ABI smoke test passed");
    return 0;
}
