# GPU & CUDA Inference Optimization

> Three hands-on notebooks that profile real GPU kernels and make a transformer
> decode loop dramatically faster: roofline analysis, KV-cache optimization, and
> kernel-launch elimination with `torch.compile` and CUDA graphs.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20required-76b900)
![License](https://img.shields.io/badge/license-MIT-green)

## Demo

<!-- TODO: drop the generated plots here after a GPU run -->
| Roofline (HW1) | Decode-step latency (HW3) |
|---|---|
| `results/hw1/roofline.png` | `results/hw3/decode_step_latency.png` |

<!-- e.g. "optimized decode: 4.8× faster than the naive baseline on an L40S" -->

## What This Project Demonstrates

- **Roofline analysis**: placing real kernels on the memory-bound/compute-bound
  roofline of a specific GPU from first-principles measurements.
- **Correct GPU timing**: CUDA-event timing with warmup, accounting for
  asynchronous execution (no `time.time()` traps).
- **Autoregressive decode optimization**: KV cache, removing per-step host
  syncs, bf16, for a **≥4×** end-to-end speedup with bit-exact greedy tokens.
- **Profiling**: reading `torch.profiler` Chrome traces in Perfetto to find
  launch gaps, tiny-kernel runs, and host syncs.
- **`torch.compile` & CUDA graphs**: eliminating graph breaks (`fullgraph=True`)
  and capturing/replaying a decode step by hand to kill launch overhead.

## Quick Start

```bash
git clone <repo-url>
cd gpu-cuda-inference-optimization
pip install -r requirements.txt

# Local (macOS / no GPU): smoke-test the self-checks only.
jupyter lab

# Real run (numbers that count): open each notebook on a GPU.
#   Google Colab ▸ Runtime ▸ Change runtime type ▸ GPU ▸ Run all.
#   Colab preinstalls torch; you may only need:  pip install -q "transformers>=4.42,<5"
```

> **GPU required for real results.** The notebooks run on CPU as a correctness
> smoke test but print warnings and trim work: all reported numbers must come
> from a GPU (H100 / L40S / T4).

## Notebooks

| Notebook | Topic | You build | Run |
|----------|-------|-----------|-----|
| [`hw1_roofline.ipynb`](hw1_roofline.ipynb) | Roofline & arithmetic intensity | CUDA-event timer, roofline metrics, memory-/compute-bound workloads | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MGhanayim/gpu-cuda-inference-optimization/blob/main/hw1_roofline.ipynb) |
| [`hw2_decode_optimization.ipynb`](hw2_decode_optimization.ipynb) | Profile & optimize a decode loop | profiler wrapper, KV-cache greedy decode, timed bf16 run | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MGhanayim/gpu-cuda-inference-optimization/blob/main/hw2_decode_optimization.ipynb) |
| [`hw3_compile_cuda_graphs.ipynb`](hw3_compile_cuda_graphs.ipynb) | `torch.compile` & CUDA graphs | launch-overhead sweep, break-free compiled step, manual CUDA graph | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MGhanayim/gpu-cuda-inference-optimization/blob/main/hw3_compile_cuda_graphs.ipynb) |

Each notebook: a fixed harness, functions you implement, self-checks that must
pass, and a short writeup. They build on each other: do them in order.

## Architecture

A single measure → analyze → optimize pipeline runs through all three notebooks:
build a workload, execute it (eager / compiled / graphed), time or profile it,
derive metrics, and write the result. Full layered breakdown, call graphs, and
diagrams are in **[PLAN.md](PLAN.md)**.

## Tech Stack

- **PyTorch ≥ 2.4**: CUDA events, `torch.compile`, `CUDAGraph`, BF16 tensor cores.
- **transformers ≥ 4.42**: a tiny synthetic Llama + KV-cache API (HW2).
- **matplotlib / numpy**: roofline and latency plots.
- **No external inference engines** (vLLM / TensorRT-LLM / SGLang) by design:
  every optimization is built from PyTorch primitives.

## Example Results

<!-- TODO after GPU run, e.g.:
- HW1: matmul-N marches toward the compute ceiling; elementwise sits on the BW roof.
- HW2: baseline V0 → +KV cache → +bf16 → +compile = 4.8× end-to-end, tokens bit-exact.
- HW3: manual CUDA-graph replay ~6× over eager at batch 1.
-->

## Project Structure

```
.
├── hw1_roofline.ipynb            # roofline & arithmetic intensity
├── hw2_decode_optimization.ipynb # profile + optimize a decode loop
├── hw3_compile_cuda_graphs.ipynb # torch.compile & CUDA graphs
├── results/                      # generated plots, JSON, traces (git-ignored)
├── SPEC.md   PLAN.md   CLAUDE.md   BREAKDOWN.md   # planning & learning docs
└── requirements.txt
```

## Key Concepts

- **Arithmetic intensity (AI):** FLOPs performed per byte moved to and from HBM
  (unit: FLOP/byte). It is the x-axis of the roofline and a property of the
  *workload*, not the hardware. Low AI (few ops per byte loaded) means
  memory-bound; high AI means compute-bound. An elementwise op is about
  0.25 FLOP/byte; an N×N×N matmul is about (2/3)·N/dbytes, which grows with N.
- **Roofline:** a log-log plot of achieved performance (TFLOP/s) against
  arithmetic intensity. Two ceilings bound it: a sloped bandwidth limit
  (peak_BW × AI) on the left and a flat compute limit (peak FLOP/s) on the right.
  The ridge point where they meet (peak FLOP/s ÷ peak BW) separates memory-bound
  from compute-bound.
- **Memory-bound vs compute-bound:** memory-bound work is limited by how fast
  bytes move to and from HBM (it sits under the sloped ceiling, left of the
  ridge); compute-bound work is limited by how fast the cores do math (under the
  flat ceiling, right of the ridge).
- **Kernel:** a single function that runs on the GPU, executed in parallel by
  many threads. Each PyTorch op (`sin`, `add`, `matmul`) launches one or more
  kernels, and every launch costs the CPU a fixed few microseconds regardless of
  the work (the launch overhead measured in HW3).
- **HBM (High Bandwidth Memory):** the GPU's main memory (its VRAM, for example
  40 GB on an A100-40GB) with very high bandwidth (about 1.5 TB/s). Reading
  inputs from HBM and writing results back (an "HBM round-trip") is what limits
  memory-bound ops; the roofline's sloped ceiling *is* the HBM bandwidth.
- **Eager:** PyTorch's default execution mode. Each op runs immediately as its
  own kernel launch with its own HBM round-trip. Simple and easy to debug, but a
  chain of K ops pays K launches and K round-trips.
- **Compiled (`torch.compile`):** traces the function into a graph and generates
  *fused* kernels. For an op chain it fuses many ops into one kernel (read from
  HBM once, compute in registers, write once), cutting launches and HBM
  round-trips. It costs a one-time compile (so timing needs warmup) and can
  "graph break" on some Python constructs (the subject of HW3).

## Docs

- **[SPEC.md](SPEC.md)**: requirements as acceptance criteria + verification checklist.
- **[PLAN.md](PLAN.md)**: architecture, dependency rules, diagrams, walkthroughs.
- **[CLAUDE.md](CLAUDE.md)**: the learning blocks and how the project was built.
- **[BREAKDOWN.md](BREAKDOWN.md)**: effort + ML concepts per block.

## License

MIT
