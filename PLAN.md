# PLAN.md — Architecture

> Companion to [SPEC.md](SPEC.md) (requirements) and [CLAUDE.md](CLAUDE.md)
> (learning blocks). Effort + knowledge breakdown in [BREAKDOWN.md](BREAKDOWN.md).

## 1. Context

This project is three Jupyter notebooks that teach GPU inference optimization by
measurement: roofline analysis (HW1) → profiling and optimizing an
autoregressive decode loop (HW2) → killing kernel-launch overhead with
`torch.compile` and CUDA graphs (HW3). The stack is **PyTorch only** (plus
`transformers` for the toy Llama, `matplotlib`/`numpy` for plots) — external
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
 Layer 0  HARNESS (DO NOT EDIT)   specs, model/data builders, timers,
          ────────────────────    plotters, self-checks, run + writeup cells
                  │ provides fixtures & contracts (calls DOWN into Layer 2)
                  ▼
 Layer 1  PRIMITIVES (yours)      one concept each, no cross-deps:
          ──────────────────       HW1 benchmark_fn, compute_metrics
                  │                 HW3 sweep_elementwise_times
                  ▼
 Layer 2  COMPOSITION (yours)     builds on Layer 1 primitives:
          ───────────────────      HW1 lowest_ai_fn / make_compute_fn
                  │                 HW2 optimized_loop, generate_optimized
                  ▼                 HW3 fixed_decode_step, make_graphed_callable
 Layer 3  ANALYSIS (yours, prose) writeups — read artifacts Layer 0 produced
```

Rule of thumb: **the harness calls down into your functions; your functions
never reach up into harness internals** (don't depend on a `DO NOT EDIT` cell's
locals — depend only on its documented inputs/outputs). Your Layer-2 functions
may call your Layer-1 functions, never the reverse.

### 2b. The cross-notebook concept stack

The notebooks are ordered; each reuses the prior one's *concepts* (you re-type
the code, but the idea is assumed learned):

```
HW1  CUDA-event timing ──────────────┐ "the timer you built yourself in HW1"
     roofline / arithmetic intensity ─┼──────────────┐
                                       ▼              ▼
HW2  time_loop (same timer idea)   bandwidth-bound decode (roofline lens, Q4)
     torch.profiler ────────────────────────────────┐
     KV cache, kill host syncs                       │
     torch.compile / CUDA graphs (applied) ──┐       │
                                              ▼       ▼
HW3  bench() (same timer idea)        profiler/roofline used to DETECT
     graph breaks (why .item() is bad — seen in HW2)  when compile won't help
     CUDA graphs by hand (what reduce-overhead automates)
```

**Dependency direction:** HW3 assumes HW1+HW2; HW2 assumes HW1. Never the
reverse. Finish notebooks in order.

## 3. Project Structure

```
gpu-cuda-inference-optimization/
├── hw1_roofline.ipynb            # HW1: roofline & arithmetic intensity
├── hw2_decode_optimization.ipynb # HW2: profile + optimize a decode loop
├── hw3_compile_cuda_graphs.ipynb # HW3: torch.compile & CUDA graphs
├── results/                      # generated artifacts (git-ignored)
│   ├── hw1/
│   │   ├── roofline_data.json    #   sweep coordinates
│   │   └── roofline.png          #   the roofline plot
│   ├── hw2/
│   │   ├── trace_baseline.json   #   Chrome trace, V0 baseline
│   │   ├── trace_optimized.json  #   Chrome trace, optimized loop
│   │   └── trace_check.json      #   self-check trace
│   └── hw3/
│       ├── launch_overhead.png   #   latency vs tensor size
│       └── decode_step_latency.png  # 4-way decode benchmark
├── SPEC.md                       # requirements & acceptance criteria
├── PLAN.md                       # this file
├── CLAUDE.md                     # learning blocks + teaching-mode rules
├── BREAKDOWN.md                  # effort + knowledge per block
├── README.md                     # portfolio front door
├── requirements.txt              # pinned deps (reproducibility)
└── .gitignore
```

> No `.env` / `.env.example`: the project has **no secrets** (no API keys, no
> remote services). Runtime selection is the Colab/GPU runtime, not a config var.

## 4. Call Graph (per notebook — the "import graph" analogue)

In-notebook there are no module imports, so the contract is the **call graph**.
Each must be acyclic; arrows mean "calls".

**HW1**
```
                 ┌─────────────── benchmark_fn ◄── (timer primitive)
 run cell ──────►│                    ▲
 (DO NOT EDIT)   ├─ lowest_ai_fn      │ (timed by)
                 ├─ make_compute_fn ──┘
                 └─ compute_metrics ◄── (metrics primitive)
                          │
                          ▼
                 plot_roofline (DO NOT EDIT)
```

**HW2**
```
 run cell ──► optimized_loop ──► model.forward(use_cache=True, past_key_values)
 (DO NOT EDIT) generate_optimized ──► build model (bf16?) ──► time_loop (harness)
              profile ──► torch.profiler ──► Chrome trace file
              (correctness: check_correctness vs baseline_loop)   [harness]
```

**HW3**
```
 run cell ──► sweep_elementwise_times ──► bench (harness)
 (DO NOT EDIT) estimate_launch_overhead_us ──► (consumes sweep output)
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
                              WRITEUP (prose)
```

HW1 instruments the **whole roofline** (many workloads, one timer). HW2 swaps the
workload for a **decode loop** and adds the **profiler** as the measurement.
HW3 zooms into **one decode step** and attacks the **launch overhead** the timer
exposes.

## 6. Core Mechanic Diagrams

### 6a. The roofline (HW1)

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

### 6b. Autoregressive decode — V0 vs optimized (HW2)

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

### 6c. Graph breaks & CUDA-graph capture (HW3)

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

**HW2 — KV cache.** Per layer, `past_key_values` holds K and V of shape
`(batch=1, n_kv_heads=8, seq_len, head_dim=64)` and grows by **one** position
each decode step. The optimized loop carries this forward instead of rebuilding
it; the baseline discards it (`use_cache=False`).

**HW3 — static graph buffers.** A captured graph owns fixed-address tensors:
`static_in` (shape `(1, 256)`, fp32) you `copy_` into, and `static_out` the
replay overwrites. Replay is only valid for **identical shape/dtype/addresses** —
hence `copy_` in, `clone()` out.

## 8. Example Walkthroughs

**W1 — one roofline point (HW1, `elementwise` on H100, bf16).**
`ELEM_N = 1<<24` elements. Convention: ~1 FLOP/elem, read x + write y →
`flops = 1.68e7`, `bytes = 2·1.68e7·2 = 6.7e7`. Say `benchmark_fn` returns
`t = 21 µs`. Then `ai = flops/bytes ≈ 0.25 FLOP/byte` (deep memory-bound),
`achieved_gbps = bytes/t/1e9 ≈ 3.2 TB/s` (near HBM peak). Point lands far
left, high on the memory ceiling. ✔ matches 1.5.3.

**W2 — one decode step (HW2).** Baseline step `t`: forward over `64+t` tokens,
`argmax`, `.item()` (CPU waits for GPU), `cat`. Optimized step `t`: forward over
**1** token reusing `past_key_values`, `argmax`, append (no `.item()`). Over 128
steps the baseline does ~`Σ(64+t)` token-forwards; the optimized does 64 (prefill)
+ 128 (one each). Same greedy tokens (`torch.equal`), ≈4–10× faster on GPU.

**W3 — graph replay (HW3).** First `graphed(x)`: `static_in.copy_(x)` →
`g.replay()` fires the whole recorded kernel sequence in **one** launch →
`static_out.clone()`. At batch 1 each kernel is launch-bound (Part 1 showed
~few-µs floor), so collapsing dozens of launches into one is the biggest single
speedup in the 4-way benchmark.

## 9. Artifacts / Persistence

No database. Persistence = files under `results/hwN/` (git-ignored — they're
reproducible and can be large):

| Artifact | Producer | Consumed by |
|----------|----------|-------------|
| `hw1/roofline_data.json` | HW1 run cell | inspection; the writeup |
| `hw1/roofline.png` | `plot_roofline` | README demo; Q1/Q2 |
| `hw2/trace_baseline.json` | `profile(baseline_loop,…)` | Perfetto; Q1 |
| `hw2/trace_optimized.json` | `profile(optimized_loop,…)` | Perfetto; Q1 |
| `hw3/launch_overhead.png` | HW3 Part 1 run cell | Q2 |
| `hw3/decode_step_latency.png` | HW3 final benchmark | README demo; Q2 |

Traces open at [ui.perfetto.dev](https://ui.perfetto.dev).

## 10. Component Organization (functions you implement)

| Notebook | Function | Layer | Purpose |
|----------|----------|-------|---------|
| HW1 | `benchmark_fn` | 1 | CUDA-event timer; avg seconds/call |
| HW1 | `compute_metrics` | 1 | (flops, bytes, sec) → {ai, tflops, gbps} |
| HW1 | `lowest_ai_fn` | 2 | lowest-AI elementwise workload |
| HW1 | `make_compute_fn` | 2 | tunable-K chained elementwise workload |
| HW2 | `profile` | 2 | run a loop under `torch.profiler` + trace |
| HW2 | `optimized_loop` | 2 | fast greedy decode, exact tokens |
| HW2 | `generate_optimized` | 2 | build + warm up + time the fast path |
| HW3 | `sweep_elementwise_times` | 1 | latency vs tensor size |
| HW3 | `estimate_launch_overhead_us` | 1 | per-launch floor from the sweep |
| HW3 | `fixed_decode_step` | 2 | break-free rewrite of the decode step |
| HW3 | `make_graphed_callable` | 2 | manual CUDA-graph capture/replay |

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
   `lowest_ai_fn` + `make_compute_fn` + run sweep → **1D** HW1 writeup.
2. **2A** `profile` → **2B** `optimized_loop` (KV cache, no sync) → **2C**
   `generate_optimized` (bf16/compile/graphs, hit ≥4×) → **2D** HW2 writeup.
3. **3A** `sweep_elementwise_times` + `estimate_launch_overhead_us` → **3B**
   `fixed_decode_step` → **3C** `make_graphed_callable` + final benchmark →
   **3D** HW3 writeup.

## 13. Key Dependencies

| Package | Constraint | Why |
|---------|-----------|-----|
| `torch` | `>=2.4` | `torch.compile`, `mode="reduce-overhead"`, `CUDAGraph`, BF16 tensor cores. On Colab, use the preinstalled CUDA build. |
| `transformers` | `>=4.42,<5` | `LlamaConfig` / `LlamaForCausalLM` + a stable KV-cache (`Cache`) API. |
| `matplotlib` | `>=3.7` | roofline + latency plots. |
| `numpy` | `>=1.24` | roofline grid math. |

GPU runtime: **Colab (paid)**. See README for the runtime caveat — the HW1 spec
table only recognizes `H100`/`L40S`; on an A100/T4 it falls back to H100 ceilings
(documented, not edited, since the cell is DO-NOT-EDIT).

## 14. Verification

- **Source of truth:** [SPEC.md](SPEC.md) acceptance criteria + checklist.
- **Smoke test (CPU, local macOS):** run each notebook's self-check cell — they
  run anywhere and catch contract violations before you pay for GPU time.
  (HW3 Part 3 self-check `SKIP`s on CPU.)
- **Real run (GPU, Colab):** top-to-bottom; confirm every checklist line and that
  `results/` is populated. Reported numbers must be the GPU numbers.
