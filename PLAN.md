# PLAN.md: Architecture

> Companion to [SPEC.md](SPEC.md) (requirements) and [CLAUDE.md](CLAUDE.md)
> (implementation blocks).

## 1. Context

This project is three Jupyter notebooks that teach GPU inference optimization by
measurement: roofline analysis (01) → profiling and optimizing an
autoregressive decode loop (02) → killing kernel-launch overhead with
`torch.compile` and CUDA graphs (03). The stack is **PyTorch only** (plus
`transformers` for the toy Llama, `matplotlib`/`numpy` for plots): external
inference engines are banned. Code is written **in-notebook** (the deliverable is
the executed notebook + `results/` artifacts), so "architecture" here means the
**dependency discipline inside each notebook** and the **concept stack across the
three**, not a layered source package.

## 2. Architecture & Dependency Rules

There is no import hierarchy of files to enforce. Instead two dependency
contracts govern the work.

### 2a. The in-notebook cell contract (per notebook)

Every notebook is the same four conceptual layers, top to bottom. **A lower
layer never depends on a higher one, and you only ever write in Layer 2.**

```
 Layer 0  HARNESS (fixed)         specs, model/data builders, timers,
          ────────────────────    plotters, self-checks, run + analysis cells
                  │ provides fixtures & contracts (calls DOWN into Layer 2)
                  ▼
 Layer 1  PRIMITIVES (yours)      one concept each, no cross-deps:
          ──────────────────       01: benchmark_fn, compute_metrics
                  │                 03: sweep_elementwise_times
                  ▼
 Layer 2  COMPOSITION (yours)     builds on Layer 1 primitives:
          ───────────────────      01: lowest_ai_fn / make_compute_fn
                  │                 02: optimized_loop, generate_optimized
                  ▼                 03: fixed_decode_step, make_graphed_callable
 Layer 3  ANALYSIS (yours, prose) analyses: read artifacts Layer 0 produced
```

Rule of thumb: **the harness calls down into your functions; your functions
never reach up into harness internals** (don't depend on a `HARNESS` cell's
locals: depend only on its documented inputs/outputs). Your Layer-2 functions
may call your Layer-1 functions, never the reverse.

### 2b. The cross-notebook concept stack

The notebooks are ordered; each reuses the prior one's *concepts* (you re-type
the code, but the idea is assumed learned):

```
01   CUDA-event timing ──────────────┐ "the same CUDA-event timer built in notebook 01"
     roofline / arithmetic intensity ─┼──────────────┐
                                       ▼              ▼
02   time_loop (same timer idea)   bandwidth-bound decode (roofline lens, §4)
     torch.profiler ────────────────────────────────┐
     KV cache, kill host syncs                       │
     torch.compile / CUDA graphs (applied) ──┐       │
                                              ▼       ▼
03   bench() (same timer idea)        profiler/roofline used to DETECT
     graph breaks (why .item() is bad: seen in 02)   when compile won't help
     CUDA graphs by hand (what reduce-overhead automates)
```

**Dependency direction:** 03 assumes 01+02; 02 assumes 01. Never the
reverse. Finish notebooks in order.

## 3. Project Structure

```
gpu-cuda-inference-optimization/
├── 01_roofline.ipynb             # roofline & arithmetic intensity
├── 02_decode_optimization.ipynb  # profile + optimize a decode loop
├── 03_compile_cuda_graphs.ipynb  # torch.compile & CUDA graphs
├── results/                      # generated artifacts (git-ignored except demo plots)
│   ├── 01_roofline/
│   │   ├── roofline_data.json    #   sweep coordinates
│   │   ├── roofline.png          #   the roofline plot
│   │   └── roofline_a100.png     #   re-drawn against honest A100 ceilings
│   ├── 02_decode_optimization/
│   │   ├── trace_baseline.json   #   Chrome trace, V0 baseline
│   │   ├── trace_optimized.json  #   Chrome trace, optimized loop
│   │   └── trace_check.json      #   self-check trace
│   └── 03_compile_cuda_graphs/
│       ├── launch_overhead.png   #   latency vs tensor size
│       └── decode_step_latency.png  # 4-way decode benchmark
├── SPEC.md                       # requirements & acceptance criteria
├── PLAN.md                       # this file
├── CLAUDE.md                     # implementation blocks + working rules
├── README.md                     # portfolio front door
├── requirements.txt              # pinned deps (reproducibility)
└── .gitignore
```

> No `.env` / `.env.example`: the project has **no secrets** (no API keys, no
> remote services). Runtime selection is the Colab/GPU runtime, not a config var.

## 4. Call Graph (per notebook: the "import graph" analogue)

In-notebook there are no module imports, so the contract is the **call graph**.
Each must be acyclic; arrows mean "calls".

**Notebook 01**
```
                 ┌─────────────── benchmark_fn ◄── (timer primitive)
 run cell ──────►│                    ▲
 (harness)       ├─ lowest_ai_fn      │ (timed by)
                 ├─ make_compute_fn ──┘
                 └─ compute_metrics ◄── (metrics primitive)
                          │
                          ▼
                 plot_roofline (harness)
```

**Notebook 02**
```
 run cell ──► optimized_loop ──► model.forward(use_cache=True, past_key_values)
 (harness)     generate_optimized ──► build model (bf16?) ──► time_loop (harness)
              profile ──► torch.profiler ──► Chrome trace file
              (correctness: check_correctness vs baseline_loop)   [harness]
```

**Notebook 03**
```
 run cell ──► sweep_elementwise_times ──► bench (harness)
 (harness)     estimate_launch_overhead_us ──► (consumes sweep output)
              fixed_decode_step ──► torch.compile(fullgraph=True)
              make_graphed_callable ──► CUDAGraph capture/replay
```

No cycles in any notebook. Your Layer-2 functions fan out to Layer-1 primitives
and harness fixtures only.

## 5. High-Level System Architecture

If a reader sees one diagram, this is the pipeline every notebook instantiates:

```
   ┌────────────┐   build    ┌──────────────┐   measure   ┌─────────────┐
   │  WORKLOAD  │──────────► │   EXECUTE    │ ──────────► │   TIMER /   │
   │ (tensors / │  on DEVICE │ (eager /     │  CUDA evts  │  PROFILER   │
   │  toy model)│            │  compiled /  │             │             │
   └────────────┘            │  graphed)    │             └──────┬──────┘
                             └──────────────┘                    │
                                                                 ▼
                            ┌───────────────┐  derive   ┌──────────────────┐
                            │   ANALYSIS    │◄──────────│  RAW MEASUREMENT │
                            │ roofline AI / │  metrics  │ (seconds, traces)│
                            │ speedup tiers │           └──────────────────┘
                            └───────┬───────┘
                                    ▼
                         results/ : .json · .png · trace
                                    │
                                    ▼
                              FINDINGS (prose)
```

Notebook 01 instruments the **whole roofline** (many workloads, one timer). Notebook 02 swaps the
workload for a **decode loop** and adds the **profiler** as the measurement.
Notebook 03 zooms into **one decode step** and attacks the **launch overhead** the timer
exposes.

## 6. Core Mechanic Diagrams

### 6a. The roofline (notebook 01)

```
 TFLOP/s
 (log)        compute ceiling (peak FLOP/s)
   peak ─────────────────────────────────────────  ← compute-bound up here
        │                      ╱──────────
        │  memory ceiling   ╱   ▲ matmul-N climbs up-right as N grows
        │  (slope = BW)  ╱      │
        │             ╱   • compiled compute-K (fused → higher)
        │          ╱    • eager compute-K
        │       ╱   • elementwise (lowest AI, memory-bound)
        └────╱──────────┬──────────────────────────── AI = FLOP/byte (log)
                     ridge AI = peak_FLOP/s ÷ peak_BW   (≈295 on H100)
            left of ridge: MEMORY-BOUND   right: COMPUTE-BOUND
```
`achieved = min(peak_compute, peak_BW × AI)`. A point's vertical gap to the roof
is unrealized performance.

### 6b. Autoregressive decode: V0 vs optimized (notebook 02)

```
 BASELINE V0 (no KV cache, O(n²)):           OPTIMIZED (KV cache, O(n)):
  step t feeds the WHOLE seq back in          step t feeds ONE token + cache
  ┌──────────────────────────┐                ┌────┐  reuse cached K,V
  │ [p0..p63, g0..g_{t-1}] →  │ recompute      │ g_{t-1} │ → model(use_cache)
  │ model over t+64 tokens    │ everything     └────┘      ▲   │
  └──────────────────────────┘                  past_key_values (grows by 1)
            │ argmax last logit                          │   ▼
            │ .item()  ◄── HOST SYNC every step      next_tok (no .item())
            ▼                                            ▼
        append g_t                                   append g_t
  cost per step ∝ (t+64)   → quadratic total   cost per step ∝ 1 → linear total
```
Two independent wins: **KV cache** removes redundant FLOPs/bytes; **dropping
`.item()`** removes a CPU↔GPU serialization each step. bf16 + `compile`/graphs in
`generate_optimized` add launch-overhead and bandwidth wins on top.

### 6c. Graph breaks & CUDA-graph capture (notebook 03)

```
 GRAPH BREAK (what NOT to do)            CUDA GRAPH (capture once, replay N×)
  fixed_decode_step compiled as 1 graph   ┌─ warm up on SIDE STREAM ─┐
   ✗ .item()  → host sync + break          │  s.wait_stream(cur)      │
   ✗ if scale>3 → branch on tensor         │  run fn(static_in)       │
   ✗ print()  → break                      └──────────┬───────────────┘
   ↓ replace with                                     ▼  capture
   ✓ keep scale as 0-dim tensor            g = CUDAGraph()
   ✓ torch.where(cond, a, b)               with cuda.graph(g):
   ✓ drop the print                           static_out = fn(static_in)
   → fullgraph=True, 0 breaks               ─────────────────────────────
                                            graphed(x):  N launches → 1 replay
                                              static_in.copy_(x); g.replay()
                                              return static_out.clone()
```

## 7. State Shape (where the project is stateful)

**Notebook 02: KV cache.** Per layer, `past_key_values` holds K and V of shape
`(batch=1, n_kv_heads=8, seq_len, head_dim=64)` and grows by **one** position
each decode step. The optimized loop carries this forward instead of rebuilding
it; the baseline discards it (`use_cache=False`).

**Notebook 03: static graph buffers.** A captured graph owns fixed-address tensors:
`static_in` (shape `(1, 256)`, fp32) you `copy_` into, and `static_out` the
replay overwrites. Replay is only valid for **identical shape/dtype/addresses**:
hence `copy_` in, `clone()` out.

## 8. Example Walkthroughs

**W1: one roofline point (notebook 01, `elementwise` on H100, bf16).**
`ELEM_N = 1<<24` elements. Convention: ~1 FLOP/elem, read x + write y →
`flops = 1.68e7`, `bytes = 2·1.68e7·2 = 6.7e7`. Say `benchmark_fn` returns
`t = 21 µs`. Then `ai = flops/bytes ≈ 0.25 FLOP/byte` (deep memory-bound),
`achieved_gbps = bytes/t/1e9 ≈ 3.2 TB/s` (near HBM peak). Point lands far
left, high on the memory ceiling. ✔ matches 1.5.3.

**W2: one decode step (notebook 02).** Baseline step `t`: forward over `64+t` tokens,
`argmax`, `.item()` (CPU waits for GPU), `cat`. Optimized step `t`: forward over
**1** token reusing `past_key_values`, `argmax`, append (no `.item()`). Over 128
steps the baseline does ~`Σ(64+t)` token-forwards; the optimized does 64 (prefill)
+ 128 (one each). Same greedy tokens (`torch.equal`), ≈4-10× faster on GPU.

**W3: graph replay (notebook 03).** First `graphed(x)`: `static_in.copy_(x)` →
`g.replay()` fires the whole recorded kernel sequence in **one** launch →
`static_out.clone()`. At batch 1 each kernel is launch-bound (Part 1 showed
~few-µs floor), so collapsing dozens of launches into one is the biggest single
speedup in the 4-way benchmark.

## 9. Artifacts / Persistence

No database. Persistence = files under the notebook's `results/` subfolder (git-ignored: they're
reproducible and can be large):

| Artifact | Producer | Consumed by |
|----------|----------|-------------|
| `01_roofline/roofline_data.json` | notebook 01 run cell | inspection; the analysis |
| `01_roofline/roofline.png` | `plot_roofline` | README demo; analysis §1/§2 |
| `02_decode_optimization/trace_baseline.json` | `profile(baseline_loop,…)` | Perfetto; analysis §1 |
| `02_decode_optimization/trace_optimized.json` | `profile(optimized_loop,…)` | Perfetto; analysis §1 |
| `03_compile_cuda_graphs/launch_overhead.png` | notebook 03 Part 1 run cell | analysis §2 |
| `03_compile_cuda_graphs/decode_step_latency.png` | notebook 03 final benchmark | README demo; analysis §2 |

Traces open at [ui.perfetto.dev](https://ui.perfetto.dev).

## 10. Component Organization (functions you implement)

| Notebook | Function | Layer | Purpose |
|----------|----------|-------|---------|
| 01 | `benchmark_fn` | 1 | CUDA-event timer; avg seconds/call |
| 01 | `compute_metrics` | 1 | (flops, bytes, sec) → {ai, tflops, gbps} |
| 01 | `lowest_ai_fn` | 2 | lowest-AI elementwise workload |
| 01 | `make_compute_fn` | 2 | tunable-K chained elementwise workload |
| 02 | `profile` | 2 | run a loop under `torch.profiler` + trace |
| 02 | `optimized_loop` | 2 | fast greedy decode, exact tokens |
| 02 | `generate_optimized` | 2 | build + warm up + time the fast path |
| 03 | `sweep_elementwise_times` | 1 | latency vs tensor size |
| 03 | `estimate_launch_overhead_us` | 1 | per-launch floor from the sweep |
| 03 | `fixed_decode_step` | 2 | break-free rewrite of the decode step |
| 03 | `make_graphed_callable` | 2 | manual CUDA-graph capture/replay |

## 11. API Surface (signatures you must match exactly)

| Function | Signature | Returns |
|----------|-----------|---------|
| `lowest_ai_fn` | `(x: Tensor) -> Callable[[], Tensor]` | no-arg fn → same-size tensor |
| `make_compute_fn` | `(x: Tensor, k: int) -> Callable[[], Tensor]` | no-arg fn → tensor (no mutate) |
| `benchmark_fn` | `(fn, n_warmup=10, n_iters=50) -> float` | avg seconds/call |
| `compute_metrics` | `(flops, bytes_moved, seconds) -> dict` | `{"ai","achieved_tflops","achieved_gbps"}` |
| `profile` | `(loop_fn, model, input_ids, max_new_tokens, trace_path) -> None` | writes a Chrome trace |
| `optimized_loop` | `(model, input_ids, max_new_tokens) -> Tensor` | `(1, max_new_tokens)` token ids |
| `generate_optimized` | `(max_new_tokens=MAX_NEW_TOKENS) -> (float, Tensor)` | `(elapsed_seconds, token_ids)` |
| `sweep_elementwise_times` | `(sizes: list) -> list` | seconds/call per size |
| `estimate_launch_overhead_us` | `(sizes, times) -> float` | µs per launch |
| `fixed_decode_step` | `(h: Tensor, state: Tensor) -> (Tensor, Tensor)` | `(h, state)` |
| `make_graphed_callable` | `(fn, example_input: Tensor) -> Callable` | `graphed(x) -> Tensor` |

## 12. Implementation Order

Mirrors the blocks in [CLAUDE.md](CLAUDE.md). Foundation → features → polish,
within and across notebooks:

1. **1A** `benchmark_fn` (timer) → **1B** `compute_metrics` → **1C**
   `lowest_ai_fn` + `make_compute_fn` + run sweep → **1D** notebook 01 analysis.
2. **2A** `profile` → **2B** `optimized_loop` (KV cache, no sync) → **2C**
   `generate_optimized` (dtype ablation, hit ≥4×) → **2D** notebook 02 analysis.
3. **3A** `sweep_elementwise_times` + `estimate_launch_overhead_us` → **3B**
   `fixed_decode_step` → **3C** `make_graphed_callable` + final benchmark →
   **3D** notebook 03 analysis.

## 13. Key Dependencies

| Package | Constraint | Why |
|---------|-----------|-----|
| `torch` | `>=2.4` | `torch.compile`, `mode="reduce-overhead"`, `CUDAGraph`, BF16 tensor cores. On Colab, use the preinstalled CUDA build. |
| `transformers` | `>=4.42,<5` | `LlamaConfig` / `LlamaForCausalLM` + a stable KV-cache (`Cache`) API. |
| `matplotlib` | `>=3.7` | roofline + latency plots. |
| `numpy` | `>=1.24` | roofline grid math. |

GPU runtime: **Colab (paid)**. See README for the runtime caveat: the notebook 01 spec
table only recognizes `H100`/`L40S`; on an A100/T4 it falls back to H100 ceilings
(documented in the analysis, since the cell is fixed harness).

## 14. Verification

- **Source of truth:** [SPEC.md](SPEC.md) acceptance criteria + checklist.
- **Smoke test (CPU, local macOS):** run each notebook's self-check cell: they
  run anywhere and catch contract violations before you pay for GPU time.
  (Notebook 03's Part 3 self-check `SKIP`s on CPU.)
- **Real run (GPU, Colab):** top-to-bottom; confirm every checklist line and that
  `results/` is populated. Reported numbers must be the GPU numbers.
