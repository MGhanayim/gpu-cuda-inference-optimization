# SPEC.md: Requirements & Acceptance Criteria (as built)

> **Scope:** three notebooks: `01_roofline.ipynb`, `02_decode_optimization.ipynb`,
> `03_compile_cuda_graphs.ipynb` (roofline analysis → decode-loop optimization →
> launch-overhead elimination).
> **Deliverable:** each notebook executed top-to-bottom **on a GPU**, plus the
> files written under `results/`.
> **Status:** complete. Verified on a Colab **A100-SXM4-40GB**
> (torch 2.11.0+cu128); every checklist line at the bottom is measured, not
> assumed.

## Overview

Three Jupyter notebooks that teach GPU inference optimization hands-on. Each
notebook ships a **fixed harness** (cells marked *HARNESS*), a set of
**implemented functions** (cells marked *IMPLEMENTATION*), **self-checks**
(asserts that must pass), and an **analysis** written in markdown against the
measured numbers. The arc builds on itself:

1. **01, Roofline & Arithmetic Intensity:** measure real kernels and place
   them on the roofline of your card (memory-bound vs compute-bound).
2. **02, Profile & Optimize a Decode Loop:** profile a deliberately slow
   autoregressive decode loop and write a fast, numerically identical
   replacement (KV cache, kill host syncs, dtype ablation).
3. **03, `torch.compile` & CUDA Graphs:** quantify per-launch overhead and
   remove it two ways: by eliminating graph breaks, and by capturing/replaying
   a decode step as a CUDA graph by hand.

## Dataset / Inputs

There is **no external dataset, API, or secret.** All inputs are generated
in-notebook:

| Notebook | Inputs |
|----------|--------|
| 01 | Random fp32/bf16 tensors (`torch.randn`); square matrices for the matmul sweep. |
| 02 | A tiny **synthetic** 2-layer Llama (`LlamaForCausalLM` with fixed `SEED`); random `input_ids`. No pretrained weights, no tokenizer text. |
| 03 | Random fp32 tensors of swept sizes; a toy 4-block MLP `TinyDecoder`. |

Hardware spec sheets (peak FLOP/s, HBM bandwidth) for **H100 SXM** and **L40S**
are hard-coded in the notebook 01 harness; a CPU fallback exists for smoke
tests only. The A100 run falls back to H100 ceilings (documented in the
analysis and re-plotted against honest A100 ceilings in `roofline_a100.png`).

## Constraints (Apply to All Notebooks)

| ID | Constraint | Where it bites |
|----|-----------|----------------|
| C1 | **Harness cells stay fixed.** Editing the notebook 02 harness would invalidate the speedup comparison. | All notebooks |
| C2 | **PyTorch + `requirements.txt` only.** No vLLM / TensorRT-LLM / SGLang. | 02 esp. |
| C3 | **Reported numbers must come from a GPU run.** CPU is a correctness smoke test only; it prints warnings and trims work. | All |
| C4 | **Self-checks must pass** before the analysis counts. They run on CPU and GPU. | All |
| C5 | **Match the stub contracts exactly**: return types, dict keys, tensor shapes, "must not mutate input." | All |
| C6 | **Keep the stub code quality:** type hints on signatures and docstrings (preserved, extended, never stripped). | All |
| C7 | **Determinism where required:** notebook 02 greedy tokens must match the baseline *exactly* (`torch.equal`); notebook 03 results match eager within stated tolerances (`atol`). | 02, 03 |
| C8 | **Deliverable = executed notebook + `results/`.** Each notebook writes JSON/PNG/trace artifacts under its `results/` subfolder. | All |

---

## Notebook 01: Roofline & Arithmetic Intensity

> Run cell produces `results/01_roofline/roofline_data.json` and
> `results/01_roofline/roofline.png`; an added cell re-draws the same points
> against honest A100 ceilings (`roofline_a100.png`).

### 1.1 `lowest_ai_fn`: lowest-arithmetic-intensity op (Part 1a)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.1.1 | Return a **no-argument** function doing one cheap elementwise pass over `x`. | Calling the returned fn returns a tensor; `_y.numel() == _x.numel()`. |
| 1.1.2 | Aim for lowest AI: a few FLOPs/element, one read + one write → memory-bound (left of ridge). | On the plotted roofline the `elementwise` point sits low-left (memory-bound). |
| 1.1.3 | Self-check passes. | `lowest_ai_fn      PASS` prints. |

### 1.2 `make_compute_fn`: tunable-K kernel (Part 1b)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.2.1 | Return a no-arg fn applying `k` **chained** elementwise ops to `x`, each feeding the next. | Returns a tensor. |
| 1.2.2 | **Must not mutate `x`.** | `torch.allclose(_x, _x0)` after building fns. |
| 1.2.3 | `k` must really change the result (real data dependency; not constant-foldable). | `not torch.allclose(_g1, _g8)`. |
| 1.2.4 | Self-check passes. | `make_compute_fn   PASS` prints. |

**Run-cell behavior to satisfy:** eager `compute-K` stays at low achieved
intensity (each step a separate HBM round-trip); `torch.compile` fuses the
chain so the compiled series sits **above** eager at the same AI.
Measured: at K=16, compiled 0.82 vs eager 0.60 TFLOP/s.

### 1.3 `benchmark_fn`: CUDA-event timing (Part 2)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.3.1 | Run `n_warmup` un-timed calls first (let cuBLAS pick kernels / `torch.compile` finish). | n/a |
| 1.3.2 | On CUDA, time `n_iters` calls with `torch.cuda.Event(enable_timing=True)` + `synchronize()` before reading elapsed. | n/a |
| 1.3.3 | On CPU (no CUDA) fall back to `time.perf_counter`. | Runs off-GPU. |
| 1.3.4 | Return **average seconds per call** (`elapsed_total / n_iters`) as a positive `float`. | `isinstance(t, float) and t > 0`; a much heavier op is not much faster (`_t_heavy >= _t_light * 0.5`). |
| 1.3.5 | Self-check passes. | `benchmark_fn      PASS` prints. |

### 1.4 `compute_metrics`: measurements → roofline coordinates (Part 3)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.4.1 | Return a dict with **exactly** keys `{"ai", "achieved_tflops", "achieved_gbps"}`. | `set(m) == {...}`. |
| 1.4.2 | `ai = flops / bytes_moved`. | `isclose(m["ai"], 2000)` for the self-check inputs. |
| 1.4.3 | `achieved_tflops = flops / seconds / 1e12`. | `isclose(m["achieved_tflops"], 2000)`. |
| 1.4.4 | `achieved_gbps = bytes_moved / seconds / 1e9`. | `isclose(m["achieved_gbps"], 1000)`. |
| 1.4.5 | Self-check passes. | `compute_metrics   PASS` prints. |

### 1.5 Sweep + artifacts (driven by the harness run cell)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.5.1 | Run cell completes on GPU using the four functions. | `results/01_roofline/roofline_data.json` written. |
| 1.5.2 | Roofline rendered. | `results/01_roofline/roofline.png` written; **all points below the roofline**. |
| 1.5.3 | Shape is correct. | `elementwise` + small-`K` low-left (memory-bound); `matmul-N` climbs up-right toward the compute ceiling as `N` grows; compiled `compute-K` above eager. |

**Test points the sweep uses (verbatim from harness):** `ELEM_N = 1<<24` on GPU;
`COMPUTE_KS = [1, 4, 16, 64]`; `MATMUL_NS = [1024, 2048, 4096]`.

**Measured (A100):** `elementwise` at AI 0.25 reaches ~1257 GB/s (~81% of the
A100's 1555 GB/s HBM peak); `matmul N=4096` reaches ~270 TFLOP/s (~87% of the
A100's 312 TFLOP/s BF16 peak).

### 1.6 Analysis §1: reading the roofline

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.6.1 | Explain which points are memory-bound vs compute-bound **from the plot alone**. | Markdown answer references slope region (under the BW line) vs flat region (under the compute ceiling). |

### 1.7 Analysis §2: why AI rises with matmul `N`

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.7.1 | Explain why AI rises with `N` for an `N×N×N` matmul, with rough FLOPs and bytes. | Answer gives FLOPs ≈ `2N³`, bytes ≈ `3N²·dbytes`, so `AI ≈ (2/3)·N/dbytes` → grows linearly in `N`. |

---

## Notebook 02: Profile & Optimize an Autoregressive Decode Loop

> Baseline **V0** (fixed harness): greedy decode, **no KV cache** (O(n²)
> recompute), fp32 eager, a host sync (`.item()`) every step. Run cell writes
> `results/02_decode_optimization/trace_baseline.json` and
> `trace_optimized.json`.

### 2.1 `profile`: wrap a loop in `torch.profiler` (Part 1)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 2.1.1 | Profile **both CPU and CUDA** activity (`ProfilerActivity`). | n/a |
| 2.1.2 | Print a short table sorted by **self CUDA time** (`key_averages().table(...)`); on CPU-only sort by self CPU time. | Table prints. |
| 2.1.3 | Export a **Chrome trace** to `trace_path`. | `os.path.exists(trace) and os.path.getsize(trace) > 0`. |
| 2.1.4 | Self-check passes. | `profile() writes a Chrome trace   PASS` prints. |

### 2.2 `optimized_loop`: fast, correct greedy decode (Part 2)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 2.2.1 | Greedy-decode the **same fp32 model** and return shape `(1, max_new_tokens)`. | `tuple(_cand.shape) == (1, _n)`. |
| 2.2.2 | Reproduce the baseline greedy token ids **exactly**. | `check_correctness(_ref, _cand)` → `torch.equal`. |
| 2.2.3 | Use a **KV cache** (`use_cache=True`, feed `past_key_values`): stop the O(n²) recompute. | Reflected in profiler + speedup; primary lever. |
| 2.2.4 | Remove the **per-step host sync** (no `.item()` on the critical path). | Reflected in optimized trace (no per-step sync gaps). |
| 2.2.5 | Self-check passes. | `optimized_loop matches baseline   PASS` prints. |

### 2.3 `generate_optimized`: build, warm up, time (Part 3)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 2.3.1 | Return `(elapsed_seconds, generated_token_ids)`. | Tuple shape correct; ids shape `(1, max_new_tokens)`. |
| 2.3.2 | `elapsed_seconds` is a **warmed-up, timed** measurement via `time_loop`. | Used as the speedup denominator vs V0. |
| 2.3.3 | May change numerics (bf16) and/or add `torch.compile` / CUDA graphs for the timed run; correctness is checked on the fp32 path. | As built: **fp32 ships**. bf16 was measured one-at-a-time and *dropped* (432.0 ms vs 409.7 ms, a 0.95× regression on this launch-bound decode). |
| 2.3.4 | Hit the speedup target vs V0: **≥ 4.0×**, counting only if 2.2.2 passes and the run is on GPU. | **Met: 4.21×** (V0 1723.4 ms → 409.7 ms, A100). |

### 2.4-2.7 Analysis §1-§4

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 2.4 (§1) | From the **baseline trace**, name the specific symptom that dominates (kernel-launch gaps / long run of tiny kernels / host syncs) and explain it. | Concrete, references the actual trace (measured: `aten::mm` at 56.13% of self-CUDA time + per-step sync gaps). |
| 2.5 (§2) | List each optimization and the speedup it contributed, measured **one at a time** (baseline → +KV → +dtype → …). | A small table with per-step measured numbers. |
| 2.6 (§3) | Which single change had the biggest impact, and why for **this** workload (128 short decode steps, tiny model)? | Identifies the measured winner (KV cache) with reasoning. |
| 2.7 (§4) | Which optimizations attack memory traffic vs CPU/launch overhead, and which mattered more here? | Classifies each opt; concludes with evidence (the bf16 regression as proof the residual bottleneck is launch overhead). |

---

## Notebook 03: `torch.compile` & CUDA Graphs

> Run cells write `results/03_compile_cuda_graphs/launch_overhead.png` and
> `results/03_compile_cuda_graphs/decode_step_latency.png`. Parts 1-2 run
> anywhere; Part 3 + final benchmark need a GPU.

### 3.1 `sweep_elementwise_times` + `estimate_launch_overhead_us` (Part 1)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 3.1.1 | For each `n` in `sizes`, build a 1-D **fp32** tensor of `n` elements on `DEVICE` and time **one** out-of-place elementwise op with `bench()`. | `len(times) == len(SIZES)`. |
| 3.1.2 | Use the **same op** for every size (so time stops depending on `n` once launch-bound). | Monotone-ish flat region at small `n`. |
| 3.1.3 | `estimate_launch_overhead_us`: average seconds/call over launch-bound sizes (`n ≤ 2**12`), convert to µs. | Returns a positive float. **Measured: ~11.1 µs/launch**, flat out to ~10⁶ elements (knee near 2²²). |
| 3.1.4 | Plot written. | `results/03_compile_cuda_graphs/launch_overhead.png` written. |

### 3.2 `fixed_decode_step`: same math, zero graph breaks (Part 2)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 3.2.1 | Same math as `broken_decode_step` **minus the print**. | `allclose` on `h` and `state` across all 6 self-check cases (`atol=1e-6`). |
| 3.2.2 | Keep `scale` as a **0-dim tensor**: never `.item()` on the hot path. | No host sync; compiles. |
| 3.2.3 | Replace the Python `if` with `torch.where` (both branches computed; stays on GPU). | Branch-free trace. |
| 3.2.4 | Compiles under `torch.compile(..., fullgraph=True)`. | `compiles with fullgraph=True       PASS`. |
| 3.2.5 | **Zero graph breaks.** | `_ex.graph_break_count == 0` → `0 graph breaks                     PASS`. |

### 3.3 `make_graphed_callable`: manual CUDA-graph capture (Part 3, GPU)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 3.3.1 | Capture `fn(example_input)` into a `torch.cuda.CUDAGraph`; return `graphed(x)` that replays for same-shape/dtype inputs. | n/a |
| 3.3.2 | Use a **static input buffer** (`static_in = example.clone()`); warm up on a **side stream**; capture once under `torch.cuda.graph(g)`. | Capture succeeds (no pending-work error). |
| 3.3.3 | `graphed(x)`: `copy_` x into `static_in`, `g.replay()`, return `static_out.clone()`. | n/a |
| 3.3.4 | Matches eager on fresh inputs. | `allclose(_ref, _out, atol=1e-5)` for 3 inputs → `matches eager on fresh inputs      PASS`. |
| 3.3.5 | Static buffers handled (different inputs → different outputs; output cloned). | `not torch.allclose(_a, _b)` → `static buffers handled correctly   PASS`. |
| 3.3.6 | Final 4-way benchmark runs; graph replay beats eager by the largest factor. | `results/03_compile_cuda_graphs/decode_step_latency.png` written. **Measured: eager 435.6 µs/step, manual CUDA graph 80.9 µs/step = 5.38×; `torch.compile` default 446.1 µs/step = 0.98×; `reduce-overhead` errored on the static-buffer constraint (analyzed, not worked around).** |

### 3.4-3.6 Analysis §1-§3

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 3.4 (§1) | For `.item()`, the Python `if`, and `print`: why can't dynamo trace each, what replaced it, and why is `.item()` **doubly** bad (host sync)? | Addresses all three + the extra sync cost. |
| 3.5 (§2) | Why does graph replay help **small-batch decode** specifically (tie to Part 1 launch overhead), and which constraints (shapes, addresses, host syncs) were respected? | Connects launch-bound regime to replay. |
| 3.6 (§3) | Name situations where `torch.compile` is slower/unhelpful (compile latency vs short jobs; recompiles from changing shapes; already hardware-limited kernels) and how to **detect** each with roofline/profiler. | Concrete cases + detection method; includes the measured 0.98× default-compile case. |

---

## Scorecard

| Component | Verified via | Result |
|-----------|-------------|--------|
| 01 implementations (1.1-1.4) | self-checks | all PASS |
| 01 roofline run + plot (1.5) | artifacts + shape | correct; ~81% of BW peak, ~87% of compute peak (A100) |
| 01 analysis (1.6-1.7) | markdown | written from the measured plot |
| 02 `profile` (2.1) | self-check (trace written) | PASS |
| 02 `optimized_loop` (2.2) | exact-token correctness | PASS (`torch.equal`) |
| 02 `generate_optimized` (2.3) | speedup ≥ 4× on GPU | **4.21×** (fp32 KV cache; bf16 ablated and dropped) |
| 02 analysis (2.4-2.7) | markdown | written; §2 ablation table |
| 03 launch overhead (3.1) | self-check + plot | ~11.1 µs/launch |
| 03 `fixed_decode_step` (3.2) | 0 graph breaks + correctness | PASS |
| 03 `make_graphed_callable` (3.3) | matches eager + static buffers | PASS; replay **5.38×** over eager |
| 03 analysis (3.4-3.6) | markdown | written; includes measured 0.98× default-compile case |

## Verification Checklist

Each notebook was run **top-to-bottom on a GPU** (Colab A100); every line below
held.

**01**
- [x] `compute_metrics   PASS`
- [x] `lowest_ai_fn      PASS`
- [x] `make_compute_fn   PASS`
- [x] `benchmark_fn      PASS` → `All checks passed`
- [x] `results/01_roofline/roofline_data.json` and `roofline.png` exist
- [x] All points below the roofline; matmul-N climbs up-right; compiled > eager
- [x] Analysis §1 and §2 written

**02**
- [x] `optimized_loop matches baseline   PASS`
- [x] `profile() writes a Chrome trace   PASS` → `All checks passed`
- [x] `optimized_loop correctness (fp32): PASS`
- [x] speedup ≥ 4.0× on GPU (measured 4.21×)
- [x] `results/02_decode_optimization/trace_baseline.json` and `trace_optimized.json` exist
- [x] Analysis §1-§4 written (§2 has the per-optimization table)

**03**
- [x] `results/03_compile_cuda_graphs/launch_overhead.png` exists; overhead estimate printed (~11.1 µs)
- [x] `same math as the broken step       PASS`
- [x] `compiles with fullgraph=True       PASS`
- [x] `0 graph breaks                     PASS`
- [x] `matches eager on fresh inputs      PASS`
- [x] `static buffers handled correctly   PASS`
- [x] `results/03_compile_cuda_graphs/decode_step_latency.png` exists; replay fastest (5.38×)
- [x] Analysis §1-§3 written

**Cross-cutting**
- [x] No harness cell modified
- [x] No banned engine used (vLLM / TensorRT-LLM / SGLang)
- [x] All reported numbers come from a GPU run (not the CPU fallback)
- [x] Type hints + docstrings preserved on all implemented functions
