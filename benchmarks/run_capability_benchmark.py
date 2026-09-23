from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from agent_sufficiency_system import (
    Capability,
    CapabilityCatalog,
    CapabilityRequest,
    SearchFirstResolver,
    TokenCounter,
    adoption_artifact_proxy,
    synthesis_artifact_proxy,
)


def capability(name: str, description: str, alias: str, layer: str, ref: str) -> Capability:
    return Capability(
        name=name,
        description=description,
        aliases=(alias,),
        source_layer=layer,
        implementation_ref=ref,
    )


def build_fixture() -> tuple[list[CapabilityCatalog], list[CapabilityRequest]]:
    repo = [
        capability("retry_http", "Retry transient HTTP requests.", "retry http requests", "repo", "utils/retry.py"),
        capability("atomic_write", "Write files atomically.", "atomic file write", "repo", "utils/files.py"),
        capability("parse_env", "Parse environment variables.", "parse environment config", "repo", "config/env.py"),
        capability("slugify_text", "Normalize text into URL-safe slugs.", "slugify text", "repo", "text/slug.py"),
        capability("batch_items", "Chunk an iterable into bounded batches.", "batch iterable items", "repo", "utils/batch.py"),
    ]
    package = [
        capability("csv_to_parquet", "Convert CSV data to parquet.", "convert csv to parquet", "package", "pyarrow.parquet"),
        capability("validate_json_schema", "Validate JSON against a schema.", "validate json schema", "package", "jsonschema.validate"),
        capability("resize_image", "Resize raster images.", "resize image", "package", "PIL.Image.resize"),
        capability("parse_yaml", "Parse YAML documents.", "parse yaml", "package", "yaml.safe_load"),
        capability("http_client", "Perform HTTP requests with timeouts.", "call http api", "package", "httpx.Client"),
    ]
    mcp = [
        capability("github_issue_search", "Search repository issues.", "search github issues", "mcp", "github.search_issues"),
        capability("calendar_lookup", "Read calendar availability.", "check calendar availability", "mcp", "calendar.free_busy"),
        capability("drive_search", "Search connected drive files.", "search drive documents", "mcp", "drive.search"),
        capability("database_query", "Execute a bounded database query.", "query connected database", "mcp", "database.query"),
    ]
    skill = [
        capability("pdf_extract", "Extract structured text from PDFs.", "extract pdf tables", "skill", "skills/pdf-extract"),
        capability("release_notes", "Draft release notes from changes.", "write release notes", "skill", "skills/release-notes"),
        capability("dependency_audit", "Audit dependency risks.", "audit dependencies", "skill", "skills/dependency-audit"),
        capability("benchmark_report", "Summarize benchmark artifacts.", "summarize benchmark results", "skill", "skills/benchmark-report"),
    ]
    github = [
        capability("mermaid_export", "Render Mermaid diagrams to SVG.", "render mermaid to svg", "github", "mermaid-cli"),
        capability("openapi_codegen", "Generate a client from OpenAPI.", "generate openapi client", "github", "openapi-generator"),
        capability("tree_sitter_index", "Index code with tree-sitter.", "build tree sitter index", "github", "tree-sitter/tree-sitter"),
        capability("semantic_diff", "Compute syntax-aware code diffs.", "semantic code diff", "github", "ast-grep/ast-grep"),
    ]
    missing = [
        ("normalize_proprietary_telemetry", "normalize proprietary telemetry format"),
        ("merge_domain_snapshots", "merge domain specific snapshots"),
        ("decode_legacy_packet", "decode legacy packet format"),
        ("score_custom_route", "score custom route candidates"),
        ("reconcile_vendor_export", "reconcile vendor specific export"),
        ("compact_agent_checkpoint", "compact custom agent checkpoint"),
        ("translate_internal_dsl", "translate internal dsl"),
        ("inspect_binary_trace", "inspect proprietary binary trace"),
    ]

    layers = [
        CapabilityCatalog("repo", repo),
        CapabilityCatalog("package", package),
        CapabilityCatalog("mcp", mcp),
        CapabilityCatalog("skill", skill),
        CapabilityCatalog("github", github),
    ]
    requests = [
        CapabilityRequest(item.aliases[0], item.name)
        for item in [*repo, *package, *mcp, *skill, *github]
    ]
    requests.extend(CapabilityRequest(intent, name) for name, intent in missing)
    assert len(requests) == 30
    return layers, requests


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] if lo == hi else xs[lo] * (hi - k) + xs[hi] * (k - lo)


def run_policy(
    policy: str,
    layers: list[CapabilityCatalog],
    stream: list[CapabilityRequest],
    counter: TokenCounter,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if policy == "build_every_time":
        resolver = None
    elif policy == "search_first_stateless":
        resolver = SearchFirstResolver(layers, cache_found=False, cache_synthesized=False)
    elif policy == "search_first_build_cache":
        resolver = SearchFirstResolver(layers, cache_found=False, cache_synthesized=True)
    elif policy == "search_first_learning":
        resolver = SearchFirstResolver(layers, cache_found=True, cache_synthesized=True)
    else:
        raise ValueError(policy)

    rows: list[dict[str, object]] = []
    for turn, request in enumerate(stream, start=1):
        started = time.perf_counter()
        if resolver is None:
            cap = Capability(
                name=request.expected_name or request.key.replace(" ", "_"),
                description=f"Build from scratch for: {request.intent}",
                aliases=(request.intent,),
                source_layer="synthesized",
                implementation_ref=f"generated/{request.key.replace(' ', '_')}.py",
            )
            trace = f"REQUEST intent={request.intent}\nDECISION build_from_scratch"
            artifact = synthesis_artifact_proxy(cap, request)
            source, synthesized, cache_hit, probes = "synthesized", True, False, 0
        else:
            resolution = resolver.resolve(request)
            cap = resolution.capability
            trace = resolution.trace_text()
            artifact = (
                synthesis_artifact_proxy(cap, request)
                if resolution.synthesized
                else adoption_artifact_proxy(cap, request)
            )
            source = resolution.source
            synthesized = resolution.synthesized
            cache_hit = resolution.cache_hit
            probes = len(resolution.probes)

        request_text = (
            "Resolve this capability before implementing new code. "
            f"Need: {request.intent}. Prefer reuse when compatible."
        )
        request_tokens = counter.count(request_text)
        resolution_tokens = counter.count(trace)
        artifact_tokens = counter.count(artifact)
        rows.append(
            {
                "turn": turn,
                "policy": policy,
                "intent": request.intent,
                "capability": cap.name,
                "source": source,
                "synthesized": int(synthesized),
                "cache_hit": int(cache_hit),
                "probes": probes,
                "request_tokens": request_tokens,
                "resolution_tokens": resolution_tokens,
                "artifact_proxy_tokens": artifact_tokens,
                "total_proxy_tokens": request_tokens + resolution_tokens + artifact_tokens,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
        )

    builds = sum(int(r["synthesized"]) for r in rows)
    cache_hits = sum(int(r["cache_hit"]) for r in rows)
    total_tokens = sum(int(r["total_proxy_tokens"]) for r in rows)
    search_probes = sum(int(r["probes"]) for r in rows)
    latencies = [float(r["elapsed_ms"]) for r in rows]
    sources = Counter(str(r["source"]) for r in rows)

    summary = {
        "requests": len(rows),
        "synthesis_events": builds,
        "reuse_rate": round(1.0 - builds / len(rows), 4),
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / len(rows), 4),
        "search_probes": search_probes,
        "mean_probes": round(search_probes / len(rows), 4),
        "total_proxy_tokens": total_tokens,
        "mean_proxy_tokens": round(total_tokens / len(rows), 1),
        "mean_resolution_tokens": round(statistics.mean(int(r["resolution_tokens"]) for r in rows), 1),
        "mean_artifact_proxy_tokens": round(statistics.mean(int(r["artifact_proxy_tokens"]) for r in rows), 1),
        "mean_elapsed_ms": round(statistics.mean(latencies), 4),
        "p95_elapsed_ms": round(percentile(latencies, 0.95), 4),
        "source_counts": dict(sorted(sources.items())),
    }
    return rows, summary


def render_report(metadata: dict[str, object], summary: dict[str, dict[str, object]]) -> str:
    baseline = summary["build_every_time"]
    stateless = summary["search_first_stateless"]
    build_cache = summary["search_first_build_cache"]
    learning = summary["search_first_learning"]

    def reduction(metric: str, candidate: dict[str, object]) -> float:
        return 1.0 - float(candidate[metric]) / float(baseline[metric])

    return f"""# Capability-Reuse Benchmark

## Executive result

This experiment adds a **search-first capability resolver** in front of the context compiler:

`cache -> repo -> package -> MCP -> skill -> GitHub/OSS -> synthesize`

The deterministic benchmark contains **{metadata['unique_capabilities']} unique capability requests**, repeated **{metadata['repeats']} times** for **{metadata['requests']} total requests**. Of the unique capabilities, **{metadata['ecosystem_hits']}** are intentionally available in one of the search layers and **{metadata['true_misses']}** require synthesis on first encounter.

The strongest policy, `search_first_learning`, synthesized only **{learning['synthesis_events']}** capabilities versus **{baseline['synthesis_events']}** for build-every-time: **{100*reduction('synthesis_events', learning):.1f}% fewer net-new implementations**. It served **{learning['cache_hits']} / {learning['requests']} requests from the local capability cache**.

Using the benchmark's transparent prompt+artifact proxy, total model-visible work fell from **{baseline['total_proxy_tokens']:,}** to **{learning['total_proxy_tokens']:,} tokens**, a **{100*reduction('total_proxy_tokens', learning):.1f}% reduction**. This is not billed LLM usage; it estimates work avoided when implementation, tests, and debug artifacts are reused.

| Policy | Synthesis events | Reuse rate | Cache hit rate | Search probes | Mean proxy tokens/request | Total proxy tokens |
|---|---:|---:|---:|---:|---:|---:|
| Build every time | {baseline['synthesis_events']} | {100*baseline['reuse_rate']:.1f}% | {100*baseline['cache_hit_rate']:.1f}% | {baseline['search_probes']} | {baseline['mean_proxy_tokens']:,.0f} | {baseline['total_proxy_tokens']:,} |
| Search-first stateless | {stateless['synthesis_events']} | {100*stateless['reuse_rate']:.1f}% | {100*stateless['cache_hit_rate']:.1f}% | {stateless['search_probes']} | {stateless['mean_proxy_tokens']:,.0f} | {stateless['total_proxy_tokens']:,} |
| Search-first + generated cache | {build_cache['synthesis_events']} | {100*build_cache['reuse_rate']:.1f}% | {100*build_cache['cache_hit_rate']:.1f}% | {build_cache['search_probes']} | {build_cache['mean_proxy_tokens']:,.0f} | {build_cache['total_proxy_tokens']:,} |
| **Search-first learning** | **{learning['synthesis_events']}** | **{100*learning['reuse_rate']:.1f}%** | **{100*learning['cache_hit_rate']:.1f}%** | **{learning['search_probes']}** | **{learning['mean_proxy_tokens']:,.0f}** | **{learning['total_proxy_tokens']:,}** |

## What this proves

1. **Search before build avoids unnecessary implementations.**
2. **Remember after use amortizes search across later requests.**

The strongest invariant is implementation count: **{baseline['synthesis_events']} -> {learning['synthesis_events']}**.

## Important limitation

Package/MCP/skill/GitHub catalogs are deterministic fixtures, not live network searches. No LLM judge is used. `artifact_proxy_tokens` count a deterministic plan/code/test/debug artifact and are an amortized-work proxy, not observed API billing.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/run_capability_benchmark.py --out results/capability_reuse
```
"""


def run(outdir: Path, seed: int = 17, repeats: int = 4) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter()
    layers, unique_requests = build_fixture()

    rng = random.Random(seed)
    stream: list[CapabilityRequest] = []
    for _ in range(repeats):
        round_requests = list(unique_requests)
        rng.shuffle(round_requests)
        stream.extend(round_requests)

    policies = [
        "build_every_time",
        "search_first_stateless",
        "search_first_build_cache",
        "search_first_learning",
    ]
    all_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for policy in policies:
        rows, policy_summary = run_policy(policy, layers, stream, counter)
        all_rows.extend(rows)
        summaries[policy] = policy_summary

    ecosystem_hits = sum(len(layer) for layer in layers)
    metadata = {
        "unique_capabilities": len(unique_requests),
        "repeats": repeats,
        "requests": len(stream),
        "ecosystem_hits": ecosystem_hits,
        "true_misses": len(unique_requests) - ecosystem_hits,
        "layers": [layer.layer for layer in layers],
        "seed": seed,
        "token_counter": counter.mode,
        "benchmark_note": (
            "Offline deterministic capability-resolution benchmark. External catalogs are fixtures; "
            "artifact tokens are a transparent implementation-work proxy, not billed LLM tokens."
        ),
    }

    baseline_total = summaries["build_every_time"]["total_proxy_tokens"]
    baseline_builds = summaries["build_every_time"]["synthesis_events"]
    for policy in policies:
        summaries[policy]["proxy_token_reduction_vs_build_every_time"] = round(
            1.0 - summaries[policy]["total_proxy_tokens"] / baseline_total, 4
        )
        summaries[policy]["implementation_reduction_vs_build_every_time"] = round(
            1.0 - summaries[policy]["synthesis_events"] / baseline_builds, 4
        )

    result = {"metadata": metadata, "summary": summaries}
    (outdir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (outdir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    for policy in policies:
        row = {"policy": policy, **summaries[policy]}
        row["source_counts"] = json.dumps(row["source_counts"], sort_keys=True)
        summary_rows.append(row)
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    (outdir / "REPORT.md").write_text(render_report(metadata, summaries), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/capability_reuse"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.out, seed=args.seed, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
