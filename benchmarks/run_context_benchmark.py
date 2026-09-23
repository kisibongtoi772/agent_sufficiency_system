from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

from agent_sufficiency_system import (
    SymbolGraphIndex,
    TokenCounter,
    TokenMinAgent,
    ToolRegistry,
    make_synthetic_turn,
)


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def resolve_default_repo() -> Path:
    import networkx
    return Path(networkx.__file__).resolve().parent


def run(repo: Path, outdir: Path, tasks_n: int, seed: int) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter()

    t0 = time.perf_counter()
    index = SymbolGraphIndex(repo)
    tools = ToolRegistry(n_distractors=44)
    agent = TokenMinAgent(index, tools)
    setup_s = time.perf_counter() - t0

    symbols = index.unique_symbols(min_lines=4, max_lines=100)
    rng = random.Random(seed)
    if len(symbols) < tasks_n:
        tasks_n = len(symbols)
    chosen = rng.sample(symbols, tasks_n)

    policies = [
        "full_repo_upper_bound",
        "raw_grep_agent",
        "graph_retrieval_only",
        "compact_state_only",
        "token_min_hybrid",
    ]

    rows = []
    history = []
    for turn_idx, sym in enumerate(chosen, start=1):
        query = (
            f"Find the definition of `{sym.name}`, identify its source file, and show the implementation "
            "and structural context needed to make a safe edit. Do not inspect unrelated files."
        )
        for policy in policies:
            pkg = agent.compile_context(policy, query, history, sym.file)
            comps = pkg.components()
            comp_chars = {k: len(v) for k, v in comps.items()}
            comp_tokens = {k: counter.count(v) for k, v in comps.items()}
            rows.append({
                "turn": turn_idx,
                "symbol": sym.name,
                "expected_file": sym.file,
                "policy": policy,
                "chars": sum(comp_chars.values()),
                "tokens": sum(comp_tokens.values()),
                "system_tokens": comp_tokens["system"],
                "tool_tokens": comp_tokens["tools"],
                "memory_tokens": comp_tokens["memory"],
                "retrieval_tokens": comp_tokens["retrieval"],
                "query_tokens": comp_tokens["query"],
                "build_ms": pkg.build_ms,
                "file_hit": int(pkg.expected_file_hit),
                "symbol_hit": int(pkg.expected_symbol_hit),
            })
        history.append(make_synthetic_turn(query, sym.file, turn_idx))

    summary = {}
    for policy in policies:
        pr = [r for r in rows if r["policy"] == policy]
        toks = [r["tokens"] for r in pr]
        times = [r["build_ms"] for r in pr]
        summary[policy] = {
            "mean_tokens": round(statistics.mean(toks), 1),
            "median_tokens": round(statistics.median(toks), 1),
            "p95_tokens": round(percentile(toks, 0.95), 1),
            "final_turn_tokens": pr[-1]["tokens"],
            "mean_build_ms": round(statistics.mean(times), 4),
            "p95_build_ms": round(percentile(times, 0.95), 4),
            "file_hit_rate": round(statistics.mean(r["file_hit"] for r in pr), 4),
            "symbol_hit_rate": round(statistics.mean(r["symbol_hit"] for r in pr), 4),
            "mean_tool_tokens": round(statistics.mean(r["tool_tokens"] for r in pr), 1),
            "mean_memory_tokens": round(statistics.mean(r["memory_tokens"] for r in pr), 1),
            "mean_retrieval_tokens": round(statistics.mean(r["retrieval_tokens"] for r in pr), 1),
        }

    raw_mean = summary["raw_grep_agent"]["mean_tokens"]
    for policy in policies:
        m = summary[policy]["mean_tokens"]
        summary[policy]["savings_vs_raw_grep"] = round(1.0 - m / raw_mean, 4)
        summary[policy]["reduction_x_vs_raw_grep"] = round(raw_mean / m, 3)

    metadata = {
        "repo": str(repo),
        "python_files_indexed": len(index.files),
        "corpus_chars": index.corpus_chars,
        "index_build_seconds": round(index.build_seconds, 4),
        "setup_seconds": round(setup_s, 4),
        "tasks": tasks_n,
        "seed": seed,
        "token_counter": counter.mode,
        "tool_count": len(tools.specs),
        "benchmark_note": "Context-efficiency benchmark; no LLM generation/judge was used.",
    }
    result = {"metadata": metadata, "summary": summary, "rows": rows}

    (outdir / "results.json").write_text(\n        json.dumps({"metadata": metadata, "summary": summary}, indent=2),\n        encoding="utf-8",\n    )
    with (outdir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tm = summary["token_min_hybrid"]
    gr = summary["graph_retrieval_only"]
    cs = summary["compact_state_only"]
    raw = summary["raw_grep_agent"]
    full = summary["full_repo_upper_bound"]
    report = f"""# Token-Min Agent — Rapid Context-Efficiency Benchmark

## Executive result

This benchmark uses a real installed Python codebase (`networkx`) with **{metadata['python_files_indexed']} files** and **{metadata['corpus_chars']:,} characters**. It generated **{tasks_n} deterministic symbol-navigation tasks** from AST ground truth and compared five context policies under the same task stream.

**Main result:** the `token_min_hybrid` policy used **{tm['mean_tokens']:,.0f} estimated input tokens/turn on average** versus **{raw['mean_tokens']:,.0f}** for the realistic raw grep/read baseline — a **{100*tm['savings_vs_raw_grep']:.1f}% reduction ({tm['reduction_x_vs_raw_grep']:.2f}× smaller)** while retaining **{100*tm['file_hit_rate']:.1f}% target-file coverage** and **{100*tm['symbol_hit_rate']:.1f}% target-symbol coverage** in this benchmark.

> Token counting mode: `{metadata['token_counter']}`. If `tiktoken` is installed, the script automatically switches to exact tokenizer counts. In this sandbox it may fall back to the transparent `ceil(chars/4)` estimator, so treat absolute token counts as estimates; the character/token reduction ratios are still directly auditable.

## Results

| Policy | Mean tokens/turn | P95 tokens | Final-turn tokens | vs raw grep | Mean retrieval tokens | Mean tool tokens | Mean memory tokens | File hit | Mean build ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full repo upper bound | {full['mean_tokens']:,.0f} | {full['p95_tokens']:,.0f} | {full['final_turn_tokens']:,.0f} | {100*full['savings_vs_raw_grep']:.1f}% | {full['mean_retrieval_tokens']:,.0f} | {full['mean_tool_tokens']:,.0f} | {full['mean_memory_tokens']:,.0f} | {100*full['file_hit_rate']:.1f}% | {full['mean_build_ms']:.3f} |
| Raw grep agent | {raw['mean_tokens']:,.0f} | {raw['p95_tokens']:,.0f} | {raw['final_turn_tokens']:,.0f} | baseline | {raw['mean_retrieval_tokens']:,.0f} | {raw['mean_tool_tokens']:,.0f} | {raw['mean_memory_tokens']:,.0f} | {100*raw['file_hit_rate']:.1f}% | {raw['mean_build_ms']:.3f} |
| Structural graph retrieval only | {gr['mean_tokens']:,.0f} | {gr['p95_tokens']:,.0f} | {gr['final_turn_tokens']:,.0f} | {100*gr['savings_vs_raw_grep']:.1f}% | {gr['mean_retrieval_tokens']:,.0f} | {gr['mean_tool_tokens']:,.0f} | {gr['mean_memory_tokens']:,.0f} | {100*gr['file_hit_rate']:.1f}% | {gr['mean_build_ms']:.3f} |
| Compact state + lazy tools only | {cs['mean_tokens']:,.0f} | {cs['p95_tokens']:,.0f} | {cs['final_turn_tokens']:,.0f} | {100*cs['savings_vs_raw_grep']:.1f}% | {cs['mean_retrieval_tokens']:,.0f} | {cs['mean_tool_tokens']:,.0f} | {cs['mean_memory_tokens']:,.0f} | {100*cs['file_hit_rate']:.1f}% | {cs['mean_build_ms']:.3f} |
| **Token-min hybrid** | **{tm['mean_tokens']:,.0f}** | **{tm['p95_tokens']:,.0f}** | **{tm['final_turn_tokens']:,.0f}** | **{100*tm['savings_vs_raw_grep']:.1f}%** | **{tm['mean_retrieval_tokens']:,.0f}** | **{tm['mean_tool_tokens']:,.0f}** | **{tm['mean_memory_tokens']:,.0f}** | **{100*tm['file_hit_rate']:.1f}%** | **{tm['mean_build_ms']:.3f}** |

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
PYTHONPATH=src python benchmarks/run_context_benchmark.py --tasks {tasks_n} --out results
```

Optional exact OpenAI-style tokenization:

```bash
pip install tiktoken
PYTHONPATH=src python benchmarks/run_context_benchmark.py --tasks {tasks_n} --out results
```

Use another Python repo:

```bash
PYTHONPATH=src python benchmarks/run_context_benchmark.py --repo /path/to/repo --tasks {tasks_n} --out results
```

## Files

- `src/agent_sufficiency_system/core.py` — structural index, lazy tools, memory policy, context compiler.
- `benchmarks/run_context_benchmark.py` — deterministic benchmark runner.
- `results/results.json` — full machine-readable run.
- `results/results.csv` — per-turn measurements.
- `results/REPORT.generated.md` — report generated directly from the latest benchmark run.
"""
    (outdir / "REPORT.generated.md").write_text(report, encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    repo = args.repo.resolve() if args.repo else resolve_default_repo()
    result = run(repo, args.out.resolve(), args.tasks, args.seed)
    print(json.dumps({"metadata": result["metadata"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
