#include "ptm/runtime.h"

#include <stdio.h>

int main(void) {
    if (ptmrt_abi_version() != PTMRT_ABI_VERSION) {
        fputs("PTM runtime ABI mismatch\n", stderr);
        return 1;
    }
    printf("PTM runtime ABI %u: %s\n",
           ptmrt_abi_version(), ptmrt_status_message(PTMRT_STATUS_OK));
    return 0;
}
