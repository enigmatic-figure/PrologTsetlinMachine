#include "ptm/runtime.h"

#include <stdint.h>

int main(void) {
    ptmrt_model_description description = {0};
    ptmrt_tensor_view tensor = {0};
    if (ptmrt_abi_version() != PTMRT_ABI_VERSION) {
        return 1;
    }
    if (ptmrt_model_describe((const ptmrt_model*)0, &description) !=
        PTMRT_STATUS_NULL_POINTER) {
        return 2;
    }
    tensor.dtype = PTMRT_DTYPE_UINT64;
    tensor.rank = 1;
    tensor.shape[0] = 2;
    if (PTMRT_DTYPE_UINT32 != 3 || PTMRT_MODEL_LOGIC_PROGRAM32_V1 != 2 ||
        PTMRT_MODEL_MASKED_THRESHOLD_V1 != 3) {
        return 3;
    }
    return tensor.shape[0] == 2 ? 0 : 4;
}
