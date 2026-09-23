# Capability-Reuse Benchmark

## Executive result

This experiment adds a **search-first capability resolver** in front of the context compiler:

`cache -> repo -> package -> MCP -> skill -> GitHub/OSS -> synthesize`

The deterministic benchmark contains **30 unique capability requests**, repeated **4 times** for **120 total requests**. Of the unique capabilities, **22** are intentionally available in one of the search layers and **8** require synthesis on first encounter.

The strongest policy, `search_first_learning`, synthesized only **8** capabilities versus **120** for build-every-time: **93.3% fewer net-new implementations**. It served **90 / 120 requests from the local capability cache**.

Using the benchmark's transparent prompt+artifact proxy, total model-visible work fell from **233,180** to **30,811 tokens**, a **86.8% reduction**. This is not billed LLM usage; it estimates work avoided when implementation, tests, and debug artifacts are reused.

## Policies

| Policy | Synthesis events | Reuse rate | Cache hit rate | Search probes | Mean proxy tokens/request | Total proxy tokens |
|---|---:|---:|---:|---:|---:|---:|
| Build every time | 120 | 0.0% | 0.0% | 0 | 1,943 | 233,180 |
| Search-first stateless | 32 | 73.3% | 0.0% | 532 | 633 | 75,940 |
| Search-first + generated cache | 8 | 93.3% | 20.0% | 412 | 261 | 31,366 |
| **Search-first learning** | **8** | **93.3%** | **75.0%** | **223** | **257** | **30,811** |

## What this proves

1. **Search before build avoids unnecessary implementations.**
2. **Remember after use amortizes search across later requests.**

The strongest invariant is implementation count: **120 -> 8**.

## Important limitation

Package/MCP/skill/GitHub catalogs are deterministic fixtures, not live network searches. No LLM judge is used. `artifact_proxy_tokens` count a deterministic plan/code/test/debug artifact and are an amortized-work proxy, not observed API billing.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/run_capability_benchmark.py --out results/capability_reuse
```
