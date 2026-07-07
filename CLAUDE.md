# CLAUDE.md

Working notes for AI-assisted sessions in this repo. The project is complete
and published; treat every change as maintenance of a finished, measured
artifact.

## Project

Three Jupyter notebooks that teach GPU inference optimization by measurement:
roofline analysis (`01`), profiling and optimizing an autoregressive decode
loop (`02`), and removing kernel-launch overhead with `torch.compile` and CUDA
graphs (`03`). Code lives in-notebook; the deliverable is each notebook
executed on a GPU plus the artifacts under `results/`.

- Requirements & measured acceptance criteria → [SPEC.md](SPEC.md)
- Architecture (cell layers, concept stack, diagrams) → [ARCHITECTURE.md](ARCHITECTURE.md)

**Headline results (Colab A100-SXM4-40GB):** 4.21× decode speedup with
bit-exact greedy tokens (fp32 KV cache; bf16 was measured one-at-a-time and
dropped as a 0.95× regression), 5.38× manual CUDA-graph replay at batch 1,
~11 µs per kernel launch.

## Tech Stack

- **PyTorch ≥ 2.4**: CUDA-event timing, `torch.compile`, `CUDAGraph`, BF16.
- **transformers ≥ 4.42**: toy `LlamaForCausalLM` + KV-cache API (notebook 02).
- **matplotlib / numpy**: roofline and latency plots.
- **Runtime:** paid Google Colab GPU. Local macOS = CPU smoke tests only.
- **Banned:** vLLM / TensorRT-LLM / SGLang (PyTorch + `requirements.txt` only).

## Rules

1. **Never edit a `HARNESS` cell.** Editing the notebook 02 harness invalidates
   the speedup comparison. Authored code lives in `IMPLEMENTATION` cells only.
2. **Match contracts exactly**: return types, dict keys, tensor shapes,
   no-mutate, exact-token correctness. Keep type hints and docstrings.
3. **Layer discipline** (ARCHITECTURE.md §2a): Layer-2 functions may call
   Layer-1 primitives, never the reverse; depend on harness contracts, not
   harness internals.
4. **Self-check on CPU first** (free), then the real GPU run. All reported
   numbers come from a GPU run.
5. No leftover `print` on a hot path; no debug cruft.

## Quick Commands

```bash
# Local (macOS, CPU): smoke test only: run self-check cells to catch contract bugs.
pip install -r requirements.txt
jupyter lab            # or: jupyter notebook

# Real run: open each notebook in paid Colab on a GPU runtime, Run All.
#   Runtime ▸ Change runtime type ▸ GPU, then Runtime ▸ Run all.
#   Colab preinstalls torch/numpy/matplotlib; you may only need:  pip install -q "transformers>=4.42,<5"
# Inspect traces: re-generate under results/02_decode_optimization/, open at https://ui.perfetto.dev
```

> **Colab runtime caveat:** notebook 01's `detect_hardware()` only recognizes
> `H100` / `L40S`. On an A100/T4 it falls back to **H100 ceilings** (with a
> printed warning), so the points sit artificially far below the roof. The spec
> cell is fixed harness: the mismatch is noted in notebook 01's analysis and the
> plot is re-drawn against honest A100 ceilings (`roofline_a100.png`) instead of
> editing the harness. Prefer an L40S/H100 runtime if available.
