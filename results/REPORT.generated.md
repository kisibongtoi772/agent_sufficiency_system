# Token-Min Agent — Rapid Context-Efficiency Benchmark

## Executive result

This benchmark uses a real installed Python codebase (`networkx`) with **580 files** and **6,462,763 characters**. It generated **40 deterministic symbol-navigation tasks** from AST ground truth and compared five context policies under the same task stream.

**Main result:** the `token_min_hybrid` policy used **3,151 estimated input tokens/turn on average** versus **25,953** for the realistic raw grep/read baseline — a **87.9% reduction (8.24× smaller)** while retaining **100.0% target-file coverage** and **100.0% target-symbol coverage** in this benchmark.

> Token counting mode: `estimate_4chars`. If `tiktoken` is installed, the script automatically switches to exact tokenizer counts. In this sandbox it may fall back to the transparent `ceil(chars/4)` estimator, so treat absolute token counts as estimates; the character/token reduction ratios are still directly auditable.

## Results

| Policy | Mean tokens/turn | P95 tokens | Final-turn tokens | vs raw grep | Mean retrieval tokens | Mean tool tokens | Mean memory tokens | File hit | Mean build ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full repo upper bound | 1,639,383 | 1,647,190 | 1,648,083 | -6216.8% | 1,622,184 | 8,456 | 8,650 | 100.0% | 1.692 |
| Raw grep agent | 25,953 | 42,366 | 42,305 | baseline | 8,754 | 8,456 | 8,650 | 100.0% | 2.456 |
| Structural graph retrieval only | 18,295 | 25,656 | 27,227 | 29.5% | 1,096 | 8,456 | 8,650 | 100.0% | 0.312 |
| Compact state + lazy tools only | 8,984 | 17,173 | 5,298 | 65.4% | 6,802 | 1,013 | 1,066 | 100.0% | 2.302 |
| **Token-min hybrid** | **3,151** | **3,786** | **3,222** | **87.9%** | **970** | **1,013** | **1,066** | **100.0%** | **0.295** |

## What each policy means

- **Full repo upper bound:** whole repository + every tool schema + growing transcript. This is intentionally an upper bound, not a recommended agent.
- **Raw grep agent:** exact-symbol grep, then inject up to five whole matching files + every tool schema + growing transcript. This approximates a simple coding agent that repeatedly searches/reads raw files.
- **Structural graph retrieval only:** Graphify-like idea implemented locally with Python AST: return the exact symbol, file/line, imports, snippet, and a few resolvable call neighbors. Tool schemas and transcript remain unoptimized so retrieval savings are isolated.
- **Compact state + lazy tools only:** keep raw-file retrieval, but replace growing history with an external checkpoint + two recent turns and load only the code tools. This shows that orchestration/state design alone can save substantial tokens.
- **Token-min hybrid:** structural symbol retrieval + lazy tool loading + deterministic compact memory + a short stable system prompt.

## Graphify vs LangGraph

They solve different layers, so a direct “which uses fewer tokens?” comparison is category-confused:

- **Graphify** is a code knowledge graph/context-retrieval layer. Its job is to stop the agent from rereading large raw files/repositories.
- **LangGraph** is an orchestration/state-machine framework. It can store short/long-term state and lets you choose what each node sends to the model, but it does not automatically make context small. A poorly designed LangGraph agent can still send huge message histories; a well-designed one can implement the same compact-state/lazy-retrieval policy used here.

The practical architecture is therefore **LangGraph (or an even smaller custom runtime) + Graphify-like structural retrieval + explicit token-budgeted context compiler**.

## Important limitation

This is a **context-efficiency benchmark, not an end-to-end coding-quality benchmark**. Ground-truth coverage verifies that the target file/symbol survives compression, but no LLM was available in the sandbox to judge whether a generated patch is correct. The next serious benchmark should hold the model fixed and compare pass@1/task success, total billed input/output tokens, tool calls, latency, and dollars per solved task.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/run_context_benchmark.py --tasks 40 --out results
```

Optional exact OpenAI-style tokenization:

```bash
pip install tiktoken
PYTHONPATH=src python benchmarks/run_context_benchmark.py --tasks 40 --out results
```

Use another Python repo:

```bash
PYTHONPATH=src python benchmarks/run_context_benchmark.py --repo /path/to/repo --tasks 40 --out results
```

## Files

- `src/agent_sufficiency_system/core.py` — structural index, lazy tools, memory policy, context compiler.
- `benchmarks/run_context_benchmark.py` — deterministic benchmark runner.
- `results/results.json` — full machine-readable run.
- `results/results.csv` — per-turn measurements.
- `results/REPORT.generated.md` — report generated directly from the latest benchmark run.
