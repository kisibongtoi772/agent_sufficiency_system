# Agent Sufficiency System

A compact research prototype for measuring how much context an agent actually needs to complete a task.

The project focuses on four token-efficiency mechanisms:

- structural symbol-level retrieval instead of whole-file context,
- lazy tool-schema loading,
- external checkpoint memory instead of replaying full history,
- explicit context budgets with a stable system prefix.

The orchestration layer is intentionally minimal. The same context compiler can be embedded in LangGraph or another runtime without depending on it.

## Current benchmark

On a deterministic 40-task symbol-navigation benchmark over the installed NetworkX package, the hybrid policy reduced estimated input context from roughly **25.9k** to **3.15k tokens per turn** versus a raw grep/read baseline while retaining **100% target-file and target-symbol coverage** in this benchmark.

See [REPORT.md](REPORT.md) for methodology, limitations, and the complete result table.

## Quick start

```bash
python -m pip install -e .
python -m pip install networkx
PYTHONPATH=src python benchmarks/run_context_benchmark.py --tasks 40 --out results
```

For tokenizer-accurate counts with the configured tokenizer:

```bash
python -m pip install tiktoken
PYTHONPATH=src python benchmarks/run_context_benchmark.py --tasks 40 --out results
```

To benchmark another Python repository:

```bash
PYTHONPATH=src python benchmarks/run_context_benchmark.py \
  --repo /path/to/repository \
  --tasks 40 \
  --out results
```

## Repository layout

```text
src/agent_sufficiency_system/   core context compiler and retrieval policies
benchmarks/                     deterministic benchmark runner
results/                        machine-readable benchmark output
tests/                          small regression tests
REPORT.md                       benchmark report and research notes
```

## Scope

This repository benchmarks **context efficiency**, not end-to-end coding correctness. A compressed context is useful only if task success remains stable. The next benchmark layer should hold an LLM fixed and measure task success, billed input/output tokens, latency, tool calls, and cost per solved task.
