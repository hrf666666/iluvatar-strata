# The Iluvatar CoreX port — engineering log

Environment: 8× Iluvatar BI-V150 (32 GiB, cc 7.1 as reported, warpSize 64,
16 SMs, 128 KiB smem/block), CoreX 4.4.0 (CUDA 10.2 API surface,
clang 18.1.8 targeting `bi`/`ivcore11`, cmake 3.31.8+corex), Ubuntu 24.04,
gcc 13.3. Two engines were ported: Strata 0.1.41 (this tree) and llama.cpp
(`iluvatar-port/llama.cpp-iluvatar.patch`).

## 1. XID 22 — device calls and the wedged card

Symptom: any GPU-offloaded run hung forever inside `cudaMalloc` at context
creation; `dmesg` carried `iluvatar … XID: 22 Receive wa illegal
instruction interrupt!` from an earlier run. A single bad kernel wedges the
*card* (not just the process) until the node reboots — during bring-up two
of eight cards were lost this way. Reproduce safely by compiling a kernel
that calls a non-inlined device helper at `-O0`:

* CoreX's `cuda_fp16.h` declares its helpers `static __host__ __device__
  __inline__` — a plain hint, which clang ignores at `-O0`.
* The bi backend lowers a device→device call into
  `RPCMOV S[6:7] PC+8 / IADD.CC / PCSWAP SRF[32:33]…` — visible in
  `ixobjdump --sass`, absent from a working `-O1` build.
* `ivcc` therefore forces `-O1` (dropping incoming `-O*` and
  `--use_fast_math`). At `-O1` the helpers inline to native `F2F.F16…`
  conversions and 5.5M lines of ggml-cuda SASS contain zero `PCSWAP`.

A second trap: the *old* workaround for this (macro-shimming byval
`__half2`/`__nv_bfloat162` conversions through raw-struct pointers)
miscompiles at `-O1` — the optimizer folds the reinterpret chain into an
identity, returning raw f16 bit patterns as floats (`1.0f` → `15360.0f`).
The shims are gone; direct intrinsics at `-O1` are bit-correct except for
1-ULP tie-rounding of f32→f16 at ≥2^10 boundaries (harmless for inference).

## 2. FP64 is silently FP32

`0.1 + 0.2` on device prints `0.300000011921…` (the f32 number);
`sqrt(-1.0)` returns `6.805646933e+38` instead of NaN. Every kernel that
reduced in `double` (GDN l2/out norms, the router softmax, sampler top_p)
computed garbage or NaN. All device-side `double`s in the kernel files are
now `float`; host code keeps FP64. (Parity tests that demanded bit-equality
with a float64 reference now report benign tolerance diffs instead of NaN.)

## 3. 64-bit warp shuffles

`__shfl_*_sync` on `unsigned long long` hangs the kernel in a spin
(verified in isolation); `long long`/`double` happened to work but are
untrusted. `corex_compat.h` rewrites every 64-bit shuffle into two 32-bit
shuffles through a variadic-macro dispatch (`bi_shfl_*`), scoped to Strata
builds (`STRATA_CUDA_ARCH_NUM`) because ggml-cuda's wider type usage needs
the raw builtins.

## 4. warpSize 64 vs 32-lane kernels

`cudaDeviceProp.warpSize` is 64 on BI-V150. Host launch math that derives
block shapes from it (ggml's MMVQ `calc_launch_params`) disagreed with the
kernels compiled for 32 lanes → `cudaGetLastError(): invalid argument` at
the first decode. The backend info now clamps the reported warp size to 32.
32-lane `__shfl_down/xor/sync` semantics themselves verified correct
(per-group sums exact at blockDim 256).

## 5. Rejected instructions

* `dp4a` (`__dp4a` → `dp4a.s32…` asm): assembler error `invalid operand`.
  ggml gets the scalar byte-product emulation
  (`GGML_ILUVATAR_NO_DP4A`); Strata's `dp4a.hpp` vmad fallback is replaced
  with plain arithmetic for the same reason (`vmad` byte-selectors are
  rejected too).
* mma.sync / ldmatrix / wmma / cp.async (`>= 800` paths): routed off via
  the arch overrides. Strata's `STRATA_CUDA_ARCH_NUM=600` also flips its
  SM60-experimental fallbacks on; ggml uses `GGML_CUDA_ARCH_NUM=710`
  (710 = Pascal-class kernel bodies, no Volta-mma (`== 700` would light
  `VOLTA_MMA_AVAILABLE`), no dp4a hardware path).
* `%%globaltimer`/`mov.u64` + `"=l"`-constraint asm (profiler stamps):
  register allocator fails; replaced with constants (profiler-only values).
* `mov.f32 %0, %1` anti-fusion barriers (`"=f"`): allocator fails; dropped.
* `__threadfence_system()` (`nvvm.membar.sys`): llc `Cannot select`;
  downgraded to `__threadfence()` + the existing volatile stores.
* `__trap()` / `trap;` asm: replaced with no-ops (assertion-only sites).
* `#pragma unroll` before an `#if` (HIP/CUDA split bodies) breaks clang;
  the pragma moved inside the branch.
* `__shared__ alignas(16) uint32_t x[...]` (alignas-on-type) is rejected;
  declarator-side `__attribute__((aligned(16)))` is used instead.

## 6. API gaps (CUDA 11/12/13 → 10.2), in `corex_compat.h`

`cudaStreamWaitEvent` 2-arg; `cudaDevAttrReservedSharedMemoryPerBlock`;
`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` (enum rename);
`cudaGetDriverEntryPointByVersion` + the 4-arg plain form and
`cudaDriverEntryPointQueryResult` (int typedef);
`cudaGraphGetEdges_v2`; `cudaGraphInstantiate` 3-arg (overload → legacy
5-arg); `cudaStreamGetCaptureInfo_v3` (→ v2, deps zeroed);
`cudaHostRegisterReadOnly`; `cudaInitDevice` (→ set device + flags);
`cudaOccupancyMaxActiveClusters` (→ 1); `__ldcs` scalar overloads (CoreX
only overloads half/bf16); batched-memcpy types (guarded off by
CUDART_VERSION anyway). The final-link needs `-lcudart -lcublas -lcurand
-lcuda -lixthunk -lixml` appended (`libcuda.so.1`'s `Thunk_*` symbols live
in `libixthunk.so`).

## 7. Build oddities

* The CoreX cmake refuses numeric `CMAKE_CUDA_ARCHITECTURES` ("must be
  ivcore10/11/20") while Strata's own checks are numeric — the port
  exempts `ivcore*` names from Strata's floors and passes `ivcore11`
  (the wrapper drops the codegen flags it implies).
* Strata's `cublasGemmEx` calls pass `CUBLAS_COMPUTE_32F`; the 10.2
  signature wants a `cudaDataType` there — `CUDA_R_32F` is the value the
  10.2-era API means.
* At `-O1`+dp4a-emulation, `mmq-instance-q2_k.cu` makes the bi `llc` run
  56+ CPU-minutes without converging; Q2_K's MMQ case is therefore not
  instantiated (dispatcher falls back to dequant + cuBLAS; Q2_K is unused
  by the deployment's models).
* Host passes of `.cu` TUs parse `__device__` bodies: any
  `#if !defined(__CUDA_ARCH__) || >= 800` arrangement makes the host parse
  sm80-only asm — those guards were inverted to device-only.
* CMake's `--use_fast_math` is dropped by the wrapper (bi codegen with
  fast-math is untested and the engines do not need it).

## 8. Parity status (GPU, after the fixes)

Passing: `iq_parity` (all IQ dequant + MMVQ, real shard layer-0 experts),
`s_gemv_parity`, `mmvq_il_parity`, `verify_parity`, `kv_q8_parity`,
`cvec_parity`, `sampler_parity`, `gdn_parity`, `native_expert_parity`
(real-shard CPU+GPU experts), 32-lane shuffle semantics, fp16/bf16
conversions, device math (powf/sinf/cosf/rsqrt/expf/logf), double→float
kernel reductions.

Remaining, benign: fused `rms_norm+rope` bit-exactness (FMA-order),
`elementwise` embedding-gather FMA-order diffs, `rope_parity`'s
float64-reference comparisons after the F64→F32 change.

Remaining, real: `gr_parity`'s hyper-connection read (11 failures,
`mixed` errors ~0.5 with adjacent-value duplication in the output) — the
active lead for the Flash-Next e2e token quality; `qsa_parity`'s
spare-slot rms_norm (128/128) which may be the same root cause.

## 9. Running notes

* `CUDA_VISIBLE_DEVICES=<healthy card>` everywhere; check
  `dmesg | grep XID` when a run hangs, `ixsmi` for card state
  (`LD_LIBRARY_PATH=/usr/local/iluvatar/lib64`).
* Strata's expert arena loads 33 GiB through the file cache at ~0.7 GiB/s
  from the GGUF; the PLE table (320M rows) is read directly from shard 2.
* llama.cpp on the same card runs the 27B Q4_K_M at ~11 tok/s decode /
  ~35 tok/s prompt with full offload, and supports `qwen4exp` (Flash-Next
  GGUFs) natively for cross-checking.

## 10. Flash-Next Strata-engine defect: precise characterization (2026-10-11 night session)

The engine's constant-token output was driven down to its mechanism:

* **The verify-window graph replay produces all-NaN logits.** Hooked
  `Verifier::run` post-replay: every one of 248320 logits is NaN for every
  round → argmax 0 → the constant token. (Debug hooks live in the tree,
  env-gated: `STRATA_DBG_R` prints residual/logits stats, `STRATA_DBG_DUMP_R`
  dumps R after replay, `STRATA_DBG_LE` truncates the verify layer range,
  `STRATA_DBG_PLAIN` routes the last prompt token through the plain loop,
  `STRATA_DBG_ALLOW_DUMP` relaxes the fused-GR/dump exclusivity.)
* **R is exactly zero entering generation** — prefill for native packs never
  writes `ss.R` by design (the verify window embeds its own tokens); ruled
  out as a defect.
* **Ruled out by microtest**: mapped-host-memory kernel read/write
  (`cudaHostAllocMapped`, both directions, bit-exact); per-kernel parity for
  every family the window runs (gdn/qsa/gr/indexer/sampler/mmvf/experts all
  0-fail); graph lifecycle (the window graph is captured once per T and
  instantiated — no per-round `cudaGraphExecUpdate`, so the CoreX update bug
  documented for llama.cpp cannot apply).
* **Capture-time probes are inert** (capture records without executing), and
  truncating the window's layer range (`STRATA_DBG_LE`) stalls the replay —
  the doorbell/pool machinery outside the truncated graph never completes —
  so layer-level bisection of the live NaN needs device-side single-step,
  which CoreX does not expose. The NaN originates at or after layer 0 of the
  first replayed window.

Next session's entry point: run with `STRATA_DBG_R=1` and bisect by
selectively replacing window sub-kernels with their scalar twins (the
`STRATA_HC_SPLIT=0` knob already exists for the GR read; a similar knob for
`gdn_step_multi`/`native_qsa` would halve the search space per run).


## 11. Breakthrough: the Strata engine emits faithful text (2026-10-12)

The all-NaN characterization from §10 was driven to TWO concrete defects with
the engine's own `STRATA_DBG_NAN` probe plus per-layer residual ladders:

1. **Prefill MMQ path**: layer 0's MoE experts (IQ2_S gate/up + Q2_0 down via
   llama.cpp MMQ templates) emit non-finite activations from the first layer
   (`non-finite GU 519 H 1144 Dm 42085 bo 7715 of T 4`). Every constituent
   passes parity standalone (`mmq_il_parity` bitwise, `iq_parity`, bf16 GEMM
   rel 1e-7), and the pinned-tree MMQ kernels see `__CUDA_ARCH__=300` with
   `GGML_CUDA_ARCH_NUM` undefined (Strata compiles them with only
   `STRATA_CUDA_ARCH_NUM=600`), so the generic fp32 paths are taken — the
   divergence is in the engine's multi-expert/grouping driver, not the
   kernels. **`STRATA_PREFILL_MMQ=0`** sidesteps it: prompt-end residual 0
   non-finite (was 40960/40960).
2. **Multi-token verify graphs**: only the T=1 window graph computes
   correctly on replay; T≥2 graphs emit all-NaN logits (real token from the
   first T=1 window, then NaN). **`STRATA_VERIFY_T1=1`** pins every window
   to one token.

With both knobs: *"Paris. The capital of Germany is Berlin. The capital…"*
from `The capital of France is` — the first faithful output from the Strata
engine on this hardware (~4 tok/s, one token per round). Remaining: output
degrades to the constant token after ~11 positions (NaN onset at position 15,
suspiciously near the QSA 16-cell block boundary or an adaptive-tier swap);
the T≥2 graph parameter-passing root cause.
