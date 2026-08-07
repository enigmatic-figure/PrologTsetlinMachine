# PTM C ABI version 2

`include/ptm/c_api.h` is the first language-neutral execution boundary. It
builds as `ptm.dll` on Windows and `libptm.so` on Linux, and exposes only
fixed-width C types and `extern "C"` functions.

Version 2 adds CPU capability discovery and model-specific scalar/AVX2/AVX-512
dispatch. All version 1 payload layouts remain unchanged.

## Payloads

| C type | Logical shape | Word count | Payload size | Alignment |
| --- | ---: | ---: | ---: | ---: |
| `ptm_bitblock_1024` | 32x32 | 16 | 128 bytes | 64 bytes |
| `ptm_bitblock_4096` | 64x64 | 64 | 512 bytes | 64 bytes |

Bit `i` is stored in word `i / 64` at offset `i % 64`. Callers must honor the
declared alignment. The bundled Python binding over-allocates and aligns ctypes
objects explicitly.

The alignment is attached to each block's word-array member. That form is valid
for C11 `_Alignas`, C++ `alignas`, and MSVC `__declspec(align)`, while forcing
the containing structure to the same 64-byte alignment. Both the C and C++
smoke checks assert the resulting sizes, alignments, and nested result offsets.

The compiled Logic path adds an aligned 320-byte `ptm_logic_program32`. It holds
32 eight-byte instructions and has a fixed 64-byte-aligned stride, so prepared
program arrays remain aligned. Each instruction uses a 32-bit backward operand
mask plus an opcode and argument. `ptm_logic_result32` is 16 bytes.

## Evaluation

`ptm_threshold_1024_eval` and `ptm_threshold_4096_eval` receive:

- an explicit port semantic;
- an input block;
- a selected-slot mask;
- an unsigned `minimum_true` threshold;
- caller-owned result storage.

The result contains the Boolean value, matched and selected counts, and matched
and missing diagnostic masks. The ABI returns a status code and never permits a
C++ exception to cross the boundary.

The current portable implementation copies ABI words into strongly typed C++
blocks before invoking the shared kernel. That makes aliasing and semantic
boundaries unambiguous. A future zero-copy fast path may be added under a new
function or ABI version after alignment and lifetime behavior are benchmarked.

`ptm_logic_program32_eval` evaluates one compiled Boolean program from five
binding bits. `ptm_logic_program32_eval_batch` evaluates a resident array of
program states and binding bytes in one call. Both validate instruction count,
root placement, opcodes, arguments, arity, and topological references. Their
diagnostic result reports the root value, true-instruction mask, and
evaluated-instruction mask.

The adaptive TM path adds an opaque immutable `ptm_tm_model`. Creation copies a
clause-major `uint16_t` TA-state matrix described by `ptm_tm_model_config` and
prepares exact state bit planes plus Include masks. `ptm_tm_model_eval_packed64`
accepts one feature-major `uint64_t` per represented feature and an explicit
valid-lane mask. Callers provide one prediction clause word and one feedback
clause word per model clause. Its fixed 320-byte result returns the valid mask,
prediction mask, and 64 clamped signed scores. The two clause vectors preserve
the empty-clause semantic difference required by prediction and feedback.

Prepared TM models are read-only after creation. Concurrent evaluations are
safe with caller-private buffers. Training creates a new prepared image rather
than mutating one that may be in use.

The adaptive path supports runtime ISA selection without changing the fixed
result layout. `ptm_cpu_capabilities_query` fills a fixed 128-byte descriptor
with hardware/OS flags, compiled backend flags, a preferred backend, and a
bounded CPU brand string. `ptm_tm_model_selected_backend` returns the
model-specific automatic decision. The existing evaluation entry point uses
automatic dispatch; `ptm_tm_model_eval_packed64_backend` permits an explicit
scalar, AVX2, or AVX-512 request. Unsupported and invalid requests return
`PTM_STATUS_BACKEND_UNAVAILABLE` before evaluation.

SIMD object files are compiled separately from the baseline dispatcher. CPU
feature bits alone are insufficient: AVX and AVX-512 are reported usable only
when OSXSAVE and the corresponding XCR0 register-state bits are also enabled.
This prevents unsupported instructions from entering the common ABI path.

## Compatibility rules

- `ptm_abi_version()` must equal the consumer's `PTM_ABI_VERSION`.
- Existing structure fields and enum values are immutable within ABI version 2.
- Reserved bytes are zeroed and must not be interpreted by callers.
- New behavior requiring a changed layout or meaning increments the ABI.
