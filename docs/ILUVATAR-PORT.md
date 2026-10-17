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
engine on this hardware (~4 tok/s, one token per round). With
`--adapt-swaps 0` added: *"Paris. The capital of Germany is Berlin. The
capital of Italy"* (13 tokens). Remaining: the NaN onset window drifts with
prompt length (4/13/15 for prompts of 4/7/13 tokens — roughly 12 verify
windows in, NOT a fixed cell boundary; `--adapt-swaps 0` only delays it by
one token, ruling the adaptive tier out as the sole trigger). The T≥2 graph
parameter-passing root cause and the ~12-window lifetime remain open.


## 12. T=2 graph facts (same session, follow-up)

* The T≥2 window graph is NaN **from its very first replay** (deterministic:
  first window T=1 real, first T=2 window all-NaN) — not a lifetime issue.
* Node composition (`STRATA_VERIFY_NODES=1`): T=1 = 2077 nodes (2076 kernel +
  1 memcpy), T=2 = 2096 nodes (2096 kernel, 0 memcpy). Only ~19 nodes apart —
  not a driver node-count limit; `cudaFuncGetName` needs CUDART ≥ 12.3 so
  kernel names are unavailable on this 10.2 runtime.
* `--adapt-swaps 0` extends the T=1-only run from 11 to 13 faithful tokens
  (*"Paris. The capital of Germany is Berlin. The capital of Italy"*), so the
  adaptive tier contributes to the T=1 lifetime but is not the sole trigger.
* The remaining suspects for T≥2 are the multi-token kernels inside the
  captured graph (gdn multi / qsa decode multi / 2-row grouped experts /
  2-column head) — all parity-green OUTSIDE graphs — or their parameter
  capture. Next lever: capture-free window execution (needs a driver without
  the doorbell-in-graph constraint) or per-node parameter comparison between
  capture and replay.


## 13. Direct-window experiment result (same session)

`STRATA_DBG_DIRECT_WINDOW=1` (in the tree, env-gated) replaces the graph
launch with a live `record_window` call — the clean A/B for "graph machinery
vs kernel". It **hangs**: the live path's doorbell wait never completes
because the CPU expert pool is only released by the GPU ring that the
*replay* protocol drives. A working variant needs the pool disabled end-to-end,
which the native-pack `--spec` path refuses. The experiment knob stays in the
tree for a follow-up with a pool-free window; the T≥2 root cause remains
between the ~19 multi-token nodes and their capture parameters.


## 14. Probing boundary found (same session, final)

Any instrumentation between the graph launch and the commit/doorbell machinery
(a `cudaStreamSynchronize` + host copy of R_, or the live-window knob) stalls
the CPU expert pool at its doorbell futex — the pool is only released by the
GPU ring that the replay protocol itself drives. Black-box probing of the
window interior is therefore **structurally blocked** on this engine. The three
open levers, in order of expected yield:

1. **Device-side NaN sentinels**: append a per-layer check kernel into the
   captured graph writing to a persistent counter buffer (adds nodes but does
   not touch the pool protocol).
2. **Node parameter diff**: `cudaGraphKernelNodeGetParams` over
   `STRATA_VERIFY_NODES`-dumped nodes, comparing captured params against the
   post-`refresh_ar`/`capture_commit` state for T=2 vs T=1.
3. **Pool-free window**: a `--spec` path that refuses the residency table is
   already escaped by `STRATA_DBG_PLAIN`; wiring it through the verify window
   (instead of the token loop) would allow live execution.

## 15. Performance pass: dp4a restored, MMQ back, 18.2x aggregate (2026-10-12)

**dp4a.** The blanket disable (`-DGGML_ILUVATAR_NO_DP4A` in ivcc, plus the
guard change in `common.cuh`) was only half right: the bi assembler rejects
the *inline PTX* form (`asm("dp4a.s32.s32.s32.s32 …")` — "invalid operand"),
but the `__dp4a()` **intrinsic** lowers to working hardware INT8 dot-product
instructions, verified bit-exact against a scalar reference on device. With
the intrinsic path enabled everywhere:

- decode (batch=1, Q4_K_M): 9.2 → **11.4 tok/s** (GPU util 0% → 91–100%)
- prefill: 39.6 → 199.5–232.5 tok/s once MMQ came back (below)

**MMQ.** The original reason MMQ was compiled out ("llc pathological
compile times with dp4a emulation") vanished with hardware dp4a: `mmq.cu`
and all 22 `template-instances/mmq-instance-*.cu` are now **byte-identical
to upstream** (pure revert of the port's compile-out). Prompt-time quantized
matmul no longer detours through dequant+cuBLAS: prefill 39.6 → **232.5 tok/s**
(5.9x).

**Bandwidth wall, measured.** A 16 GB device-scan kernel gives ~245 GB/s
usable per card. 27B Q4_K_M = 16.2 GB read per token at batch=1 → hard
ceiling ~15.3 tok/s; 11.4 achieved = 75% efficiency. `-O2/-O3` on
representative kernels: no measurable difference (launch/memory bound).
4-GPU `--tensor-split` at batch=1: **no decode gain** (11.2 tok/s — the
layer pipeline is sequential, per-card bandwidth does not aggregate).

**Batch aggregation.** `-np N` concurrent sequences read the weights once
per step and emit N tokens:

| np | aggregate tok/s (1 card) | per-seq |
|---|---|---|
| 1 | 11.4 | 11.4 |
| 8 | 18.5 | 2.31 |
| 16 | 22.7 | 1.42 |
| 32 | 32.5 | 1.02 |
| 64 | **46.3** | 0.73 |

np=64 with `-c 32768` (512/slot) fits in 30.1/32.7 GB.

**Multi-instance.** One server per card (healthy 3,4,6,7), each `-np 64`:
256 concurrent requests × 64 tokens → 16384 tokens in 97.9 s =
**167.4 tok/s aggregate** (41.8 per card, perfectly linear). That is
**18.2x** the 9.2 tok/s single-stream baseline. Launcher:
`iluvatar-port/serve-parallel.sh`. Raw log:
`docs/evidence/perf-optimization.log`.

## 16. JEV-35B: decision model deployed end-to-end (2026-10-15)

AutoTrust's JEV-35B (System 1 decision model on the Qwen3.5-35B-A3B MoE base,
released on ModelScope/HF) went from HF safetensors to faithful single-card
decisions on BI-V150:

- **Conversion**: `architectures: Qwen3_5MoeForConditionalGeneration` maps to
  the in-tree `Qwen3_5MoeTextModel` (arch `qwen35moe`, registered for exactly
  this class). 753 tensors, bf16 GGUF 67 GB, no unknown-tensor errors.
- **Quantization**: `llama-quantize --leave-output-tensor Q4_K_M` — the lm_head
  (248320×2048, bf16) carries the merged decision head in the verbalizer token
  rows (A..JU, false/true, 0..5 per `adapter_vllm/decision_head.json`), so it
  stays unquantized for exact label logits. Result: 21.3 GB, one card.
  Note: llama-quantize crashes enumerating wedged GPUs (CUDA_VISIBLE_DEVICES
  must name a healthy card; empty string is ignored by this runtime).
- **Read-out**: the model's own serve_decide.py expects vLLM's per-token-id
  logprobs; llama.cpp has no such endpoint, and `logit_bias` turned out to be a
  NO-OP in this CoreX build (verified: identical sampled tokens and reported
  logprobs with and without bias). Working path: `/v1/completions` with
  `"logprobs": N` in **integer** form returns full-vocab top-N logprobs
  (pre-bias, exact); labels outside top-64 are negligible. `tools/jev_decide.py`
  then applies `(logprob + bias) / T_kind` exactly like the reference.
- **Prompt**: bare text `[kind] K\n[state] S\n[question] Q\n[options]\nA) ..\n
  [decision]:` with fixed option lines for noul (false/true) and score (0..5) —
  getting that wrong yields a uniform 0.5/0.5 (both labels fall out of top-N).
- **Results**: 11/11 validation battery (choice incl. arithmetic, noul, score),
  warm decision latency 105 ms on one BI-V150 (B200 reference: 204 ms median).
  Raw log: `docs/evidence/jev35b-decisions.log`.

## 18. The decode NaN, root-caused and fixed: adaptive swaps poison the captured windows (2026-10-17)

The §11/§12 open front is closed. The engine's constant-token death was **not** a kernel and
**not** a capture defect: the **adaptive tier's swap-in tears the captured verify windows on
the CoreX runtime** — the first replay after a swap computes from torn residency/weights and
every later logits row is `0x7fffffff`.

* **Reproduced** unmitigated: `22525 369 279 1067 0 0 …` (4 faithful tokens, then dead).
* **Kernels excluded wholesale**: the whole parity battery (gdn multi, gr_multi incl. staged
  cp.async + graph replay, route_window, native_multi, mmvq_multi, verify_batch, spec_verify)
  green on the same card.
* **Boundary located** by per-row dump: rows 0..12 finite, rows 13+ all-NaN — 13 = 1+4+4+4,
  exactly the default 4-round swap point ("adaptive tier: 2 experts swapped (every 4
  rounds)"). Old puzzles dissolve: cache 12288 died after 1 token, cache 1 after 4 — the swap
  point counts **rounds, not VRAM hits**; the §11 "~12-window lifetime" was the same 4/8/12
  cadence under `STRATA_VERIFY_T1`.
* **Decisive A/B**: `--adapt-every 0`, everything else stock and no `STRATA_VERIFY_T1` →
  canonical text 16/16, twice; **61 × 248320 logits, 0 non-finite** across verification runs
  (48-token long-form run all-faithful as well).
* **Fix, default-on**: ivcore builds now define `STRATA_IVCORE_PORT`, under which
  `adapt_every` defaults to 0 with a one-line explanation (an explicit `--adapt-every N`
  re-enables at the operator's risk). `STRATA_VERIFY_T1=1` is no longer needed.

Carried in the same commit: the stale `-ffast-math` leftover for ivcore was dropped from the
CMake (the ivcc wrapper never had it; the flag changed results, see §2/§15).

## 19. CUDA API gaps vs LLM inference: what matters, what is shimmed, what cannot be (2026-10-17)

A systematic pass over both engines' CUDA surface against the CoreX 10.2-era headers
(`CUDART_VERSION 10020`). Verdicts, in the order the three headline gaps appear:

| API / feature | introduced | what inference uses it for | CoreX status | action |
|---|---|---|---|---|
| `cudaGraphInstantiate` 3-arg | 12.0 | decode fast path: the captured verify windows replay through it | legacy 5-arg form present | **shimmed** (`corex_compat.h` overload) — windows replay |
| thread-block clusters (`cudaLaunchKernelEx` template, `cudaOccupancyMaxActiveClusters`, cluster attrs) | 11.8, sm_90 | sm_90-only top-k / sampler kernels | API absent/partial; **hardware has no clusters anyway** (cc 7.1) | **compile-guarded out** — correct end state, nothing to fix |
| `__ldcs` streaming loads (int/float widths) | sm_32 intrinsics | expert/weight streaming reads | only half/bf16 overloads exist | **shimmed** as plain volatile loads (same values; only the L2 eviction hint lost) |
| `cudaGetDriverEntryPoint` (+Version, result enum) | 11.0 / 12.5 | VMM / driver symbol lookup (elastic KV, cache segments) | legacy 3-arg form only | **adapted** in `vmm.cpp` / `expert_cache.cpp` |
| `cudaInitDevice` | 12.0 | spin-scheduling hint before context creation | absent | **guarded off** (hint only) |
| PDL — programmatic dependent launch (`cudaGridDependencySynchronize`, `cudaTriggerProgrammaticLaunchCompletion`, serialization attr) | 11.8–12.x device intrinsics | overlaps launch latency across the verify window's kernel chain — a real decode-latency lever | **host headers mention it; the device-side intrinsics do not exist for ivcore11** (`use of undeclared identifier`, `cudaLaunchKernelExC` signature is the 10.2 one) | **cannot be shimmed in software** (needs griddepcontrol in the backend); `pdl_supported()` stays false; reopened only if Iluvatar ships the device intrinsics |
| `cudaGraphExecUpdate` | 10.2 (present) | refresh a captured graph in place | present but **defective on CoreX** (llama.cpp §10 evidence) | engines avoid it; avoidance is the fix |
| `cudaMallocAsync` / `cudaFreeAsync` | 11.2 | stream-ordered allocations | present | available; engines do not require it |
| sm_80 `mma.sync` (m16n8k*) for INT8 MMQ | 11.0 | prompt-time quantized matmul | backend compiles generic paths only | dp4a hardware path already covers decode; MMQ stays on the FP16 fallback here |

Net: of the three headline gaps, **graph instantiation and streaming loads are shimmed and
working** (every decode window replays through them), **clusters are hardware-gated and
correctly compiled out**, and PDL is the one genuine loss — it would shave launch gaps in the
decode chain but requires backend support that CoreX does not yet expose for ivcore11.
