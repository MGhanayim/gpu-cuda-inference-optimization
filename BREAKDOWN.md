# BREAKDOWN.md: Effort & Knowledge by Block

> Organized by the conceptual blocks in [CLAUDE.md](CLAUDE.md). Each block maps
> to a slice of [SPEC.md](SPEC.md). The source assigns **no explicit points**, so
> the weight columns are *indicative* (planning only) and HW2 is graded by a
> **speedup tier**, not points.

## Block → Assignment Mapping

> Simplified one-to-one view. See the many-to-many table below for the honest picture.

| Block | Theme | Maps to SPEC.md | Weight |
|-------|-------|-----------------|--------|
| 1A | CUDA-event timing | 1.3 | 6% |
| 1B | Roofline coordinates | 1.4 | 6% |
| 1C | Workloads + roofline run | 1.1, 1.2, 1.5 | 8% |
| 1D | HW1 writeup | 1.6, 1.7 | 5% |
| 2A | Profiling | 2.1 | 5% |
| 2B | KV-cache decode (correctness) | 2.2 | 15% |
| 2C | Build + time (speedup tier) | 2.3 | 15% |
| 2D | HW2 writeup | 2.4-2.7 | 5% |
| 3A | Launch overhead | 3.1 | 8% |
| 3B | Graph breaks | 3.2 | 9% |
| 3C | Manual CUDA graphs | 3.3 | 13% |
| 3D | HW3 writeup | 3.4-3.6 | 5% |

### Grading Line-Item → Blocks (the honest many-to-many view)

The "deliverables" are features, not files: several span multiple blocks, and a
few blocks feed multiple deliverables.

| Deliverable | Graded via | Built across blocks |
|-------------|-----------|---------------------|
| HW1 self-checks pass | asserts | **1A + 1B + 1C** (all four functions must exist before the self-check cell passes) |
| HW1 roofline correct shape | visual + json | **1A + 1B + 1C** (the run cell uses *all* of them) |
| HW1 writeup | prose | **1D**, but only answerable after 1C produced the plot |
| HW2 trace written | assert | **2A** |
| HW2 exact-token correctness | `torch.equal` | **2B** (and 2A's `profile` is reused by the run cell) |
| HW2 speedup ≥ 4× | tier | **2B + 2C** (cache from 2B is the bulk; 2C adds bf16/compile/graphs) |
| HW2 writeup | prose | **2D**, depends on traces from 2A and numbers from 2B/2C |
| HW3 launch-overhead plot | self-check + png | **3A** |
| HW3 0 graph breaks | `explain` | **3B** |
| HW3 CUDA-graph correctness | `allclose` | **3C** |
| HW3 final 4-way benchmark | png | **3B + 3C** (compares eager vs compiled-fixed vs manual graph vs reduce-overhead) |
| HW3 writeup | prose | **3D**, synthesizes 3A-3C **and** HW1 (roofline) + HW2 (profiler) |

**Key implications:**
- **HW1 self-checks are all-or-nothing across 1A-1C**: the single self-check cell
  exercises all four functions, so none of HW1's "code" credit lands until the
  trio is done. Build them together, test once.
- **The HW2 speedup is split:** the KV cache (2B) is most of it; bf16/compile/
  graphs (2C) top it up to the ≥4× tier. Measure them separately for Q2 (2D).
- **`profile` (2A) is dual-purpose:** a graded deliverable *and* the tool that
  produces the traces the 2D writeup analyzes.
- **HW3's writeup reaches back two notebooks**: Q3 asks you to *detect* compile
  pitfalls with the HW1 roofline and HW2 profiler. 3D depends on HW1+HW2, not just 3A-3C.

## Effort Estimation by Block

*Manual* = a learner fluent in Python but new to these GPU libraries, typing it
themselves. *With AI* = same learner with Claude/Cursor doing the typing, where
the bottleneck shifts to reviewing output, understanding the concept, and (for
HW2/HW3) debugging on the GPU. Ratio is typically ~3-4×, smaller where GPU
iteration and correctness debugging dominate (you can't AI-away a wrong KV cache).

| Block | What gets built | Weight | Manual | With AI | Complexity |
|-------|-----------------|--------|--------|---------|------------|
| 1A | `benchmark_fn` | 6% | 1.0 h | 0.3 h | Low-Med (async timing gotchas) |
| 1B | `compute_metrics` | 6% | 0.5 h | 0.15 h | Low (unit arithmetic) |
| 1C | `lowest_ai_fn`, `make_compute_fn`, run sweep | 8% | 1.5 h | 0.5 h | Med (eager-vs-compiled intuition) |
| 1D | HW1 writeup Q1-Q2 | 5% | 0.75 h | 0.4 h | Low (interpretation) |
| 2A | `profile` | 5% | 1.0 h | 0.3 h | Low-Med (profiler API) |
| 2B | `optimized_loop` (KV cache, no sync) | 15% | 3.0 h | 1.25 h | **High** (cache API + exact tokens) |
| 2C | `generate_optimized` (bf16/compile/graphs) | 15% | 2.5 h | 1.0 h | Med-High (hit ≥4×, keep correctness) |
| 2D | HW2 writeup Q1-Q4 | 5% | 1.25 h | 0.6 h | Med (ablation + trace reading) |
| 3A | launch-overhead sweep | 8% | 1.0 h | 0.4 h | Low-Med (sweep + estimate) |
| 3B | `fixed_decode_step` | 9% | 1.25 h | 0.5 h | Med (dynamo + `torch.where`) |
| 3C | `make_graphed_callable` | 13% | 2.5 h | 1.0 h | **High** (capture/replay, static buffers) |
| 3D | HW3 writeup Q1-Q3 | 5% | 1.0 h | 0.5 h | Med (synthesis across HW1-HW3) |

### Totals

| Scope | Blocks | Weight | Manual | With AI | Run (machine, per run-all) |
|-------|--------|--------|--------|---------|----------------------------|
| HW1 | 1A-1D | 25% | 3.75 h | 1.35 h | ~3 min |
| HW2 | 2A-2D | 40% | 7.75 h | 3.15 h | ~5 min |
| HW3 | 3A-3D | 35% | 5.75 h | 2.40 h | ~5 min |
| **Everything** | 12 blocks | **100%** | **≈ 17.25 h** | **≈ 6.9 h** | **≈ 13 min/pass** |

> **Human effort and machine time are separate budgets.** *Manual* / *With AI* are
> **your** hours (reading, writing, reviewing, debugging). **Run (machine)** is
> wall-clock for one clean top-to-bottom GPU pass: it overlaps your time only
> while you sit waiting on it, and is **not** added into the human-effort totals.

### Execution / Wall-Clock (machine time)

The "With AI" hours assume the code already runs; they do **not** include the
machine cost of executing the notebooks. Budget that separately:

| Cost | Rough wall-clock | Notes |
|------|------------------|-------|
| Colab session spin-up + `pip install transformers` | ~1-2 min | once per session; repeats after every disconnect/restart |
| First `torch.compile` trace (HW1 compiled series; HW3 default + `reduce-overhead`) | ~10-60 s each | one-time per process: warmup, not steady state |
| CUDA-graph capture + side-stream warmup (HW3) | a few seconds | per capture |
| Slow O(n²) baseline V0 (HW2, ×warmup+iters) | folded into HW2's ~5 min | the baseline is *meant* to be slow |
| Re-runs to fix correctness / hit the ≥4× tier | **×2-4** the per-pass time | the real multiplier in practice |

**Realistic project machine time ≈ 1-2 h** across all three notebooks (session
restarts + compile warmups + retries): entirely separate from the ~7 h "With AI"
human effort. No bonuses in the source to add on top.

## Knowledge by Block

### Block 1A: Async timing with CUDA events
**AI/ML Concepts & Architecture:**
- **GPU asynchronous execution**: kernels are queued; the CPU returns before the
  GPU finishes, so wall-clock timing needs an explicit device sync.
- **CUDA events & synchronization**: `Event(enable_timing=True)` records device
  timestamps; `synchronize()` makes the CPU wait so `elapsed_time` is real.
- **Warmup**: first calls pay one-time costs (cuBLAS autotune, allocator, JIT);
  measure steady state.

**Libraries & Tools:**
| What you need to know | Library / Tool |
|---|---|
| Event timing, `synchronize`, `is_available` | `torch.cuda` |
| CPU fallback timer | `time.perf_counter` |

### Block 1B: Roofline coordinates
**AI/ML Concepts & Architecture:**
- **Arithmetic intensity (FLOP/byte)**: the x-axis of the roofline.
- **Achieved compute vs bandwidth**: derive TFLOP/s and GB/s from one timing.
- **Ridge point**: `peak_FLOP/s ÷ peak_BW`; left = memory-bound, right = compute-bound.

**Libraries & Tools:**
| What you need to know | Library / Tool |
|---|---|
| Unit conversions (`1e12`, `1e9`) | plain Python |

### Block 1C: Workloads on the roofline
**AI/ML Concepts & Architecture:**
- **Memory-bound elementwise ops**: ~1 FLOP/element, dominated by HBM traffic.
- **Kernel fusion**: `torch.compile` fuses a chain so `x` is read once; eager
  pays an HBM round-trip per step (keeps achieved AI low).
- **Constant folding**: chain real data dependencies so steps aren't optimized away.
- **Matmul as the compute-bound anchor**: `O(N³)` FLOPs over `O(N²)` bytes.

**Libraries & Tools:**
| What you need to know | Library / Tool |
|---|---|
| Elementwise ops, `@` matmul, dtypes | `torch` |
| Tracing/fusing a callable | `torch.compile` |

### Block 1D: HW1 writeup
**AI/ML Concepts & Architecture:**
- **Reading a roofline**: slope region (BW-limited) vs flat region (compute-limited).
- **Matmul scaling**: FLOPs `~2N³`, bytes `~3N²·dbytes`, so `AI ∝ N`.

**Libraries & Tools:** *(prose: none)*

### Block 2A: Profiling a decode loop
**AI/ML Concepts & Architecture:**
- **`torch.profiler`**: record CPU+CUDA activity; self vs total time.
- **Trace visualization**: Chrome trace → Perfetto; spot launch gaps, tiny-kernel
  runs, host syncs.

**Libraries & Tools:**
| What you need to know | Library / Tool |
|---|---|
| `profile`, `ProfilerActivity`, `key_averages`, `export_chrome_trace` | `torch.profiler` |
| Trace UI | [ui.perfetto.dev](https://ui.perfetto.dev) |

### Block 2B: Fast, correct greedy decode
**AI/ML Concepts & Architecture:**
- **Prefill vs decode**: process the prompt once, then one token per step.
- **KV cache**: store/reuse past keys & values so each step is O(1) in sequence
  length, not O(n); the single biggest decode optimization.
- **Host syncs on the hot path**: `.item()` serializes CPU↔GPU every step; remove it.
- **Greedy determinism**: argmax decoding is reproducible → exact-token match required.

**Libraries & Tools:**
| What you need to know | Library / Tool |
|---|---|
| `use_cache=True`, `past_key_values`, `Cache` API | `transformers` (`LlamaForCausalLM`) |
| `argmax`, `cat`, `no_grad`, on-device token collection | `torch` |

### Block 2C: Build + time the optimized run
**AI/ML Concepts & Architecture:**
- **BF16 inference**: half the bytes on a bandwidth-bound loop; numerics differ
  (allowed on the timed path, not the correctness path).
- **Stacking optimizations**: cache → bf16 → `torch.compile`/CUDA graphs.
- **Warmup vs steady state**: measure after compile/capture has happened.

**Libraries & Tools:**
| What you need to know | Library / Tool |
|---|---|
| `torch.bfloat16`, `torch.compile`, `mode="reduce-overhead"` | `torch` |
| Warmed-up timing harness | `time_loop` (given) |

### Block 2D: HW2 writeup
**AI/ML Concepts & Architecture:**
- **Bottleneck attribution from a trace**: name the dominant symptom.
- **Ablation**: measure each optimization one at a time.
- **Memory-traffic vs launch-overhead**: classify which wins applied where.

**Libraries & Tools:** *(prose: uses traces from 2A)*

### Block 3A: Cost of one kernel launch
**AI/ML Concepts & Architecture:**
- **Kernel-launch overhead**: fixed µs CPU cost per launch, work-independent.
- **Launch-bound regime**: below a size knee, time ≈ launch overhead; the
  motivation for graphs.

**Libraries & Tools:**
| What you need to know | Library / Tool |
|---|---|
| Reusing the HW1-style timer (`bench`) | `torch.cuda` events |
| Plotting latency vs size | `matplotlib` |

### Block 3B: Eliminating graph breaks
**AI/ML Concepts & Architecture:**
- **TorchDynamo tracing**: Python → graph; breaks on values needed from tensor
  data (`.item()`, branch on tensor, `print`).
- **Graph breaks**: each splits the graph (no cross-break fusion); `.item()` also
  forces a host sync.
- **Branch-free GPU code**: `torch.where` computes both sides, stays on device;
  keep `scale` a 0-dim tensor.

**Libraries & Tools:**
| What you need to know | Library / Tool |
|---|---|
| `explain`, `reset`, `fullgraph=True` | `torch._dynamo`, `torch.compile` |
| `torch.where`, `rsqrt`, keeping 0-dim tensors | `torch` |

### Block 3C: CUDA graphs by hand
**AI/ML Concepts & Architecture:**
- **CUDA graph capture/replay**: record a kernel sequence once, replay in one
  launch; collapses N launches → 1 (what `reduce-overhead` automates).
- **Static buffers**: fixed input/output addresses; `copy_` in, `clone()` out.
- **Side-stream warmup**: capture fails on a stream with pending work.
- **Replay constraints**: fixed shape/dtype/addresses, no host sync/alloc inside.

**Libraries & Tools:**
| What you need to know | Library / Tool |
|---|---|
| `CUDAGraph`, `torch.cuda.graph`, `Stream`, `wait_stream`, `replay` | `torch.cuda` |
| `tensor.copy_`, `tensor.clone` | `torch` |

### Block 3D: HW3 writeup
**AI/ML Concepts & Architecture:**
- **Why `.item()` is doubly bad**: graph break *and* host sync.
- **Why replay helps batch-1 decode**: launch-bound regime (3A) dominates.
- **When `torch.compile` doesn't pay**: compile latency vs short jobs; recompiles
  from dynamic shapes; already-bandwidth-bound kernels: detect with roofline
  (HW1) and profiler traces (HW2).

**Libraries & Tools:** *(prose: synthesizes HW1-HW3 tools)*

## Dataset / API Quick Reference

No external dataset, API, or secret. Inputs are generated in-notebook:
- **HW1:** `torch.randn` vectors (`ELEM_N=1<<24`) + square matrices (`N∈{1024,2048,4096}`).
- **HW2:** synthetic 2-layer `LlamaForCausalLM` (fixed `SEED`, `hidden=512`,
  `layers=2`, `heads=8`), random `input_ids`, `PROMPT_LEN=64`, `MAX_NEW_TOKENS=128`.
- **HW3:** random fp32 tensors over `SIZES=[2**k for k in range(0,25,2)]`; toy
  4-block MLP `TinyDecoder` (`d=256`, `hidden=1024`), 256-step decode.

## Model / Tech Choice

- **PyTorch ≥ 2.4** for `torch.compile`, `mode="reduce-overhead"`, `CUDAGraph`,
  BF16 tensor cores. **transformers ≥ 4.42** for the toy Llama + stable KV-cache.
- **Runtime: paid Colab GPU.** Reported numbers must come from the GPU; local
  macOS runs are CPU smoke tests only.
- **Roofline ceilings** are hard-coded for **H100 SXM** and **L40S**. On a Colab
  A100/T4 the harness falls back to H100 ceilings (printed warning): a known
  caveat to note in the HW1 writeup, since the spec cell is DO-NOT-EDIT.
