# CLAUDE.md

## Project

Three Jupyter notebooks that teach **GPU inference optimization** by
measurement: roofline analysis, profiling/optimizing an autoregressive decode
loop, and removing kernel-launch overhead with `torch.compile` and CUDA graphs.
Code is written **in-notebook**; the deliverable is each notebook executed on a
GPU plus the artifacts under `results/`.

- Requirements & acceptance criteria → [SPEC.md](SPEC.md)
- Architecture (cell layers, concept stack, diagrams) → [PLAN.md](PLAN.md)

**Status: all blocks complete**, executed and verified on a Colab A100.

### Tech Stack
- **PyTorch ≥ 2.4**: CUDA-event timing, `torch.compile`, `CUDAGraph`, BF16.
- **transformers ≥ 4.42**: toy `LlamaForCausalLM` + KV-cache API (notebook 02).
- **matplotlib / numpy**: roofline and latency plots.
- **Runtime:** paid Google Colab GPU. Local macOS = CPU smoke tests only.
- **Banned:** vLLM / TensorRT-LLM / SGLang (PyTorch + `requirements.txt` only).

## Why: concepts this project exercises

- Reason about a kernel as **memory-bound vs compute-bound** and place it on a
  roofline from real measurements.
- Time GPU code **correctly** (async execution, CUDA events, warmup): and know
  why `time.time()` lies.
- Understand the **KV cache** and why a naive decode loop is O(n²), and remove
  host syncs from a hot loop.
- Read a **profiler trace** (Perfetto) and name the bottleneck.
- Understand **graph breaks**, `torch.compile`, and **CUDA graphs**: and when
  compilation does *not* help.

## How to Work With Me

### Teaching Mode (default)
- **Do NOT write full implementations.** Guide me step by step.
- **Explain the concept BEFORE any code**: the GPU mechanism behind the task.
- **Show small snippets (≤20 lines)**, then let me write/extend the function.
- **Hints before answers** when I'm stuck. Escalate gradually.
- **Ask leading questions** to check understanding ("what does `.item()` force
  the CPU to do here?").
- **After each step, explain WHY** and connect it to the diagram in
  [PLAN.md](PLAN.md) and the concept stack (§2b).
- **Reference [SPEC.md](SPEC.md)** acceptance criteria so we know a step is done.
- **Tie back to earlier notebooks**: notebook 02's timer is notebook 01's idea;
  notebook 03 detects with the tools from notebooks 01/02.

### When I say "just do it" / "implement this"
- Switch to implementation mode: write the full function.
- Still explain non-obvious design decisions (why the dtype of the timed run is
  a measured choice; why `clone()` the graph output).
- Mark the relevant [SPEC.md](SPEC.md) acceptance criteria as met.

### Code Standards
- **Never edit a `HARNESS` cell.** (Editing the notebook 02 harness invalidates
  the speedup numbers.) Work only in `IMPLEMENTATION` cells.
- Keep the stubs' **type hints and docstrings**: extend, don't strip.
- Match the contract exactly: return types, dict keys, tensor shapes, "no mutate".
- Meaningful names; no leftover `print` on a hot path (notebook 03) or debug cruft.
- Run the **self-check on CPU first** (free) before paying for a GPU run.
- All reported numbers come from a **GPU** run.

## Implementation Steps

Work the blocks **in order**: they follow the concept stack in PLAN.md §2b.
Each block is one concept and ships one capability you can self-check before
moving on. Block ids are `<notebook><letter>`. All blocks below are done.

---

### BLOCK 1A: Async timing with CUDA events (Layer 1) · `01_roofline.ipynb`
**Concept:** GPU launches are asynchronous; `time.time()` measures Python queuing,
not GPU work. CUDA events + `synchronize()` give real device time; warmup hides
one-time costs (cuBLAS autotune, `torch.compile`).
**Outcome:** `benchmark_fn` returns avg seconds/call; self-check `benchmark_fn PASS`.

- [x] **1A.1** Explain why `time.time()` is wrong for GPU; what `synchronize` does.
- [x] **1A.2** Run `n_warmup` un-timed calls.
- [x] **1A.3** CUDA path: `Event(enable_timing=True)` start/end + sync, read `elapsed_time` (ms → s).
- [x] **1A.4** CPU fallback: `time.perf_counter`.
- [x] **1A.5** Return `elapsed_total / n_iters` as a positive float; pass the self-check.
- **Learn:** CUDA streams & async exec; `torch.cuda.Event`; warmup; ms vs s.

---

### BLOCK 1B: Roofline coordinates (Layer 1) · `01_roofline.ipynb`
**Concept:** Arithmetic intensity = FLOP/byte. Achieved compute (TFLOP/s) and
bandwidth (GB/s) come from the same measurement; the ridge point separates
memory- from compute-bound.
**Outcome:** `compute_metrics` returns exactly `{ai, achieved_tflops, achieved_gbps}`;
self-check `compute_metrics PASS`.

- [x] **1B.1** Define AI, achieved TFLOP/s, achieved GB/s from `(flops, bytes, sec)`.
- [x] **1B.2** Return the dict with the exact three keys (unit factors `1e12`, `1e9`).
- [x] **1B.3** Pass the three `isclose` asserts.
- **Learn:** arithmetic intensity; roofline ridge; FLOP/s vs byte/s units.

---

### BLOCK 1C: Workloads on the roofline (Layer 2) · `01_roofline.ipynb`
**Concept:** A memory-bound op (1 read, 1 write, ~1 FLOP) sits far left; chaining
`k` ops raises *ideal* AI but eager keeps it low (a kernel + HBM round-trip per
step) until `torch.compile` fuses the chain. Real data dependencies prevent
constant-folding.
**Outcome:** `lowest_ai_fn` + `make_compute_fn` pass self-checks; run cell writes
`results/01_roofline/roofline.png` + `roofline_data.json` with correctly-placed points.

- [x] **1C.1** `lowest_ai_fn`: return a no-arg fn doing one cheap elementwise pass.
- [x] **1C.2** `make_compute_fn`: chain `k` ops through `x`; **don't mutate** `x`; ensure `k` changes the result.
- [x] **1C.3** Run the sweep on GPU; confirm eager vs compiled separation and matmul climb.
- **Learn:** memory- vs compute-bound ops; kernel fusion via `torch.compile`; why eager AI stays low.

---

### BLOCK 1D: Roofline analysis (Layer 3) · `01_roofline.ipynb`
**Concept:** Read the roofline you produced.
**Outcome:** analysis §1 + §2 written.

- [x] **1D.1** §1: which points are memory- vs compute-bound, from the plot alone.
- [x] **1D.2** §2: why AI rises with `N` for `N×N×N` matmul (`~2N³` FLOPs, `~3N²·dbytes` bytes → `AI ∝ N`).
- **Learn:** interpreting slope vs flat regions; matmul scaling.

---

### BLOCK 2A: Profiling a decode loop (Layer 2) · `02_decode_optimization.ipynb`
**Concept:** `torch.profiler` records CPU+CUDA ops; the self-CUDA-time table and a
Chrome trace (Perfetto) reveal launch gaps, tiny-kernel runs, and host syncs.
**Outcome:** `profile` writes a non-empty trace; self-check `profile() writes a Chrome trace PASS`.

- [x] **2A.1** Use `profiler.profile` with `ProfilerActivity.CPU` + `.CUDA`.
- [x] **2A.2** Print `key_averages().table(sort_by="self_cuda_time_total")` (self CPU on CPU-only).
- [x] **2A.3** `export_chrome_trace(trace_path)`; pass the self-check.
- **Learn:** `torch.profiler`; reading a Perfetto trace; self vs total time.

---

### BLOCK 2B: Fast, correct greedy decode (Layer 2) · `02_decode_optimization.ipynb`
**Concept:** The baseline recomputes the whole sequence every step (no KV cache,
O(n²)) and forces a host sync (`.item()`) each step. A KV cache makes each step
process one token; greedy decoding is deterministic so tokens must match exactly.
**Outcome:** `optimized_loop` returns `(1, max_new_tokens)` and matches the
baseline tokens exactly; self-check `optimized_loop matches baseline PASS`.

- [x] **2B.1** Explain prefill vs decode and what `past_key_values` stores (PLAN §7).
- [x] **2B.2** Prefill the prompt with `use_cache=True`; keep the returned cache.
- [x] **2B.3** Decode loop: feed the **single** last token + cache; `argmax` the last logit.
- [x] **2B.4** **No `.item()`** on the hot path; collect tokens on-device, concat at the end.
- [x] **2B.5** Confirm exact-match correctness vs `baseline_loop` (fp32).
- **Learn:** KV cache; prefill/decode split; eliminating host syncs; greedy determinism.

---

### BLOCK 2C: Build + time the optimized run (Layer 2) · `02_decode_optimization.ipynb`
**Concept:** The timed path may change numerics and add `torch.compile` / CUDA
graphs; correctness is judged separately on the fp32 path. Warmup matters.
**Outcome:** `generate_optimized` returns `(elapsed_seconds, token_ids)`; speedup
≥ 4× on GPU. As built: fp32 KV cache ships (4.21×); bf16 was measured one-at-a-time
and dropped (0.95× regression on this launch-bound decode).

- [x] **2C.1** Build the model; reuse `optimized_loop` (fp32 shipped; bf16 measured and dropped).
- [x] **2C.2** Measure with `time_loop` (warmed up); return `(seconds, ids)`.
- [x] **2C.3** Ablate the dtype one-at-a-time; keep the measured winner.
- [x] **2C.4** Confirm the ≥4× target and that correctness still passes.
- **Learn:** dtype tradeoffs are measured, not assumed; warmup vs steady state; stacking optimizations.

---

### BLOCK 2D: Decode analysis (Layer 3) · `02_decode_optimization.ipynb`
**Concept:** Explain the trace and attribute speedups.
**Outcome:** analysis §1-§4 written (§2 has a per-optimization table).

- [x] **2D.1** §1: dominant symptom in the baseline trace (name it concretely).
- [x] **2D.2** §2: per-optimization speedup measured one at a time (table).
- [x] **2D.3** §3: biggest single change and why for this workload.
- [x] **2D.4** §4: memory-traffic vs CPU/launch optimizations; which mattered more.
- **Learn:** ablation; bandwidth-bound vs launch-bound reasoning.

---

### BLOCK 3A: Cost of one kernel launch (Layer 1) · `03_compile_cuda_graphs.ipynb`
**Concept:** A launch costs the CPU a few µs regardless of work; for tiny tensors
wall time is nearly pure overhead until size makes the work dominate (the knee).
**Outcome:** sweep + `estimate_launch_overhead_us`; `results/03_compile_cuda_graphs/launch_overhead.png`.

- [x] **3A.1** `sweep_elementwise_times`: same out-of-place op over fp32 tensors per size, timed with `bench()`.
- [x] **3A.2** `estimate_launch_overhead_us`: average the launch-bound sizes (`n ≤ 2**12`), → µs.
- [x] **3A.3** Run the sweep; read the flat region in the plot.
- **Learn:** launch overhead; launch-bound regime; finding the knee.

---

### BLOCK 3B: Eliminating graph breaks (Layer 2) · `03_compile_cuda_graphs.ipynb`
**Concept:** Dynamo breaks on values it needs from tensor data (`.item()`, branch
on a tensor, `print`). Each break splits the graph and `.item()` also syncs.
**Outcome:** `fixed_decode_step` same math, **0 graph breaks**, `fullgraph=True`.

- [x] **3B.1** Run `dynamo.explain` on `broken_decode_step`; read the breaks.
- [x] **3B.2** Keep `scale` a **0-dim tensor** (no `.item()`).
- [x] **3B.3** Replace the `if` with `torch.where`; drop the `print`.
- [x] **3B.4** Match the broken math (`allclose`); compile `fullgraph=True`; 0 breaks.
- **Learn:** TorchDynamo trace; graph breaks; `torch.where`; `fullgraph=True`.

---

### BLOCK 3C: CUDA graphs by hand (Layer 2) · `03_compile_cuda_graphs.ipynb`
**Concept:** A CUDA graph records a kernel sequence once and replays it in one
launch: what `reduce-overhead` automates. Cost: static buffers, fixed shapes, no
host sync inside.
**Outcome:** `make_graphed_callable` matches eager + handles static buffers; final
4-way benchmark writes `results/03_compile_cuda_graphs/decode_step_latency.png`.

- [x] **3C.1** Clone a `static_in`; warm up on a **side stream**.
- [x] **3C.2** Capture once under `torch.cuda.graph(g)` → `static_out`.
- [x] **3C.3** `graphed(x)`: `static_in.copy_(x)`, `g.replay()`, `return static_out.clone()`.
- [x] **3C.4** Pass both self-checks; run the final benchmark (replay wins: 5.38×).
- **Learn:** `CUDAGraph` capture/replay; static buffers; side-stream warmup; `copy_`/`clone`.

---

### BLOCK 3D: Compile/graphs analysis (Layer 3) · `03_compile_cuda_graphs.ipynb`
**Concept:** Explain the breaks, why replay helps batch-1 decode, and compile's limits.
**Outcome:** analysis §1-§3 written.

- [x] **3D.1** §1: why each construct breaks; what replaced it; why `.item()` is doubly bad.
- [x] **3D.2** §2: why replay helps small-batch decode (tie to 3A); constraints respected.
- [x] **3D.3** §3: two-plus cases where `compile` is slower/unhelpful + how to detect with roofline/profiler.
- **Learn:** synthesizing all three notebooks; when not to compile.

---

## Architecture Rules (from PLAN.md)

1. **Only write in `IMPLEMENTATION` cells.** Never touch `HARNESS` cells.
2. **Layer discipline:** your Layer-2 functions may call your Layer-1 primitives,
   never the reverse; depend on harness *contracts*, not its internals.
3. **Notebook order is a dependency:** finish 01 → 02 → 03 (the concept stack,
   PLAN §2b). Don't reach forward.
4. **Self-check on CPU first**, then the real GPU run. Reported numbers are GPU.
5. **Match contracts exactly** (shapes, keys, no-mutate, exact-token correctness).

## Quick Commands

```bash
# Local (macOS, CPU): smoke test only: run self-check cells to catch contract bugs.
pip install -r requirements.txt
jupyter lab            # or: jupyter notebook

# Real run: open each notebook in paid Colab on a GPU runtime, Run All.
#   Runtime ▸ Change runtime type ▸ GPU, then Runtime ▸ Run all.
#   Colab preinstalls torch/numpy/matplotlib; you may only need:  pip install -q "transformers>=4.42,<5"
# Inspect traces: upload results/02_decode_optimization/trace_*.json to https://ui.perfetto.dev
```

> **Colab runtime caveat:** notebook 01's `detect_hardware()` only recognizes
> `H100` / `L40S`. On an A100/T4 it falls back to **H100 ceilings** (with a
> printed warning), so the points sit artificially far below the roof. The spec
> cell is fixed harness: the mismatch is noted in notebook 01's analysis and the
> plot is re-drawn against honest A100 ceilings (`roofline_a100.png`) instead of
> editing the harness. Prefer an L40S/H100 runtime if your Colab tier offers one.
