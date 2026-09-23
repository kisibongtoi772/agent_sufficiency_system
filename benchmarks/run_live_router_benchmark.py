from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ENDPOINT = "https://blockrun.ai/api/v1/chat/completions"

MODELS = {
    "nano": "nvidia/nemotron-3-nano-30b",
    "worker": "nvidia/gpt-oss-20b",
    "planner": "nvidia/nemotron-3.5-lightning",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    difficulty: str
    prompt: str
    expected: str


TASKS = [
    Task("arithmetic", "easy", "Compute exactly: (37 * 24) - (18 * 7) + 45.", "807"),
    Task(
        "python_trace",
        "easy",
        """What does this Python code return?

def f(xs):
    out = 0
    for i, x in enumerate(xs):
        if i % 2 == 0:
            out += x * (i + 1)
        else:
            out -= x
    return out

f([3, 5, 2, 7, 4])""",
        "17",
    ),
    Task(
        "sql_count",
        "easy",
        """Tables:
users(id): 1,2,3,4
orders(id,user_id,amount):
(1,1,30),(2,1,15),(3,2,40),(4,4,10),(5,4,60)

What integer is returned by:
SELECT COUNT(DISTINCT u.id)
FROM users u JOIN orders o ON u.id=o.user_id
WHERE o.amount >= 40;""",
        "2",
    ),
    Task(
        "shortest_path",
        "medium",
        """Directed weighted edges:
A->B 4, A->C 2, B->C 1, B->D 5,
C->B 1, C->D 8, C->E 10, D->E 2.
What is the shortest-path distance from A to E?""",
        "10",
    ),
    Task(
        "interval_scheduling",
        "medium",
        """Intervals are:
(1,4),(3,5),(0,6),(5,7),(3,9),(5,9),
(6,10),(8,11),(8,12),(2,14),(12,16).
Using non-overlapping intervals where end <= next start is allowed,
what is the maximum number of intervals that can be selected?""",
        "4",
    ),
    Task(
        "knapsack",
        "medium",
        """0/1 knapsack with capacity 10.
Items are (weight,value):
(6,30),(3,14),(4,16),(2,9).
What is the maximum total value?""",
        "46",
    ),
    Task(
        "grid_paths",
        "hard",
        """Count monotone paths in a 5x5 grid from (0,0) to (4,4).
You may move only right or down.
Blocked cells are (1,1),(1,3),(2,1),(3,3).
Neither start nor destination is blocked.
How many valid paths exist?""",
        "6",
    ),
    Task(
        "recursive_trace",
        "hard",
        """Given:
def mystery(n):
    if n <= 1:
        return n + 1
    return mystery(n-1) + 2*mystery(n-2)

What integer is returned by mystery(8)?""",
        "256",
    ),
    Task(
        "matrix_chain",
        "hard",
        """Three matrices have dimensions:
A1 = 10x30, A2 = 30x5, A3 = 5x60.
What is the minimum number of scalar multiplications needed
to compute A1*A2*A3 by choosing the best parenthesization?""",
        "4500",
    ),
]


def normalize_answer(value: str) -> str:
    value = value.strip().lower().replace(",", "")
    return re.sub(r"\s+", " ", value)


def parse_answer(content: str) -> str:
    content = content.strip()
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"])
    except Exception:
        pass
    match = re.search(r'\{[^{}]*"answer"\s*:\s*"?(.*?)"?(?:\s*[,}])', content, re.S)
    if match:
        return match.group(1).strip().strip('"')
    numbers = re.findall(r"-?\d+(?:\.\d+)?", content)
    return numbers[-1] if numbers else content


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def call_model(model: str, messages: list[dict[str, str]], retries: int = 4) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 700,
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "agent-sufficiency-benchmark/0.1",
        },
        method="POST",
    )
    last_error = None
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=150) as response:
                raw = response.read().decode("utf-8")
            elapsed = time.perf_counter() - started
            data = json.loads(raw)
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if prompt_tokens is None:
                prompt_tokens = estimate_tokens(json.dumps(messages, separators=(",", ":")))
            if completion_tokens is None:
                completion_tokens = estimate_tokens(content)
            return {
                "content": content,
                "requested_model": model,
                "served_model": data.get("model", "unknown"),
                "latency_s": elapsed,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = f"HTTP {exc.code}: {detail}"
            time.sleep(min(20, 2 ** attempt))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(20, 2 ** attempt))
    return {
        "content": "",
        "requested_model": model,
        "served_model": "error",
        "latency_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": last_error or "unknown error",
    }


def planner_call(task: Task) -> dict:
    return call_model(
        MODELS["planner"],
        [
            {
                "role": "system",
                "content": (
                    "You are the planner in a two-model system. Produce a short, concrete reasoning "
                    "plan for the worker. Focus on operations and checks that prevent mistakes. "
                    "Do not wrap the answer in JSON. Keep the plan under 180 words."
                ),
            },
            {"role": "user", "content": task.prompt},
        ],
    )


def worker_call(task: Task, model_key: str, plan: str | None = None) -> dict:
    user = task.prompt
    if plan:
        user += "\n\nPlanner notes:\n" + plan
    return call_model(
        MODELS[model_key],
        [
            {
                "role": "system",
                "content": (
                    'Solve the task carefully. Return ONLY a JSON object of the form '
                    '{"answer":"VALUE"}. No markdown, no explanation, no extra keys.'
                ),
            },
            {"role": "user", "content": user},
        ],
    )


def evaluate(task: Task, call: dict) -> bool:
    return normalize_answer(parse_answer(call["content"])) == normalize_answer(task.expected)


def combine_calls(calls: list[dict]) -> dict:
    return {
        "calls": len(calls),
        "latency_s": sum(c["latency_s"] for c in calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "requested_models": [c["requested_model"] for c in calls],
        "served_models": [c["served_model"] for c in calls],
        "errors": [c["error"] for c in calls if c["error"]],
    }


def run_direct(task: Task, model_key: str) -> tuple[dict, dict]:
    call = worker_call(task, model_key)
    return call, combine_calls([call])


def run_planner_worker(task: Task) -> tuple[dict, dict]:
    plan = planner_call(task)
    worker = worker_call(task, "worker", plan["content"])
    return worker, combine_calls([plan, worker])


def run_complexity_gated(task: Task) -> tuple[dict, dict]:
    if task.difficulty == "easy":
        return run_direct(task, "nano")
    if task.difficulty == "medium":
        return run_direct(task, "worker")
    return run_planner_worker(task)


def run_test_driven_cascade(task: Task) -> tuple[dict, dict]:
    calls: list[dict] = []
    first = worker_call(task, "nano")
    calls.append(first)
    if evaluate(task, first):
        return first, combine_calls(calls)

    second = worker_call(task, "worker")
    calls.append(second)
    if evaluate(task, second):
        return second, combine_calls(calls)

    plan = planner_call(task)
    calls.append(plan)
    final = worker_call(task, "worker", plan["content"])
    calls.append(final)
    return final, combine_calls(calls)


POLICIES: dict[str, Callable[[Task], tuple[dict, dict]]] = {
    "nano_only": lambda task: run_direct(task, "nano"),
    "gpt_oss_only": lambda task: run_direct(task, "worker"),
    "lightning_only": lambda task: run_direct(task, "planner"),
    "lightning_plan_gpt_worker": run_planner_worker,
    "complexity_gated": run_complexity_gated,
    "test_driven_cascade": run_test_driven_cascade,
}


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for policy in POLICIES:
        subset = [r for r in rows if r["policy"] == policy]
        successes = sum(r["success"] for r in subset)
        calls = sum(r["calls"] for r in subset)
        prompt_tokens = sum(r["prompt_tokens"] for r in subset)
        completion_tokens = sum(r["completion_tokens"] for r in subset)
        latency = sum(r["latency_s"] for r in subset)
        served = Counter()
        for row in subset:
            for model in row["served_models"].split("|"):
                if model:
                    served[model] += 1

        by_difficulty = {}
        for difficulty in ("easy", "medium", "hard"):
            part = [r for r in subset if r["difficulty"] == difficulty]
            wins = sum(r["success"] for r in part)
            by_difficulty[difficulty] = {
                "successes": wins,
                "tasks": len(part),
                "pass_rate": round(wins / len(part), 4),
            }

        total_tokens = prompt_tokens + completion_tokens
        summary[policy] = {
            "tasks": len(subset),
            "successes": successes,
            "pass_rate": round(successes / len(subset), 4),
            "total_calls": calls,
            "mean_calls_per_task": round(calls / len(subset), 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_solved_task": round(total_tokens / successes, 1) if successes else None,
            "total_latency_s": round(latency, 3),
            "mean_latency_s": round(latency / len(subset), 3),
            "latency_per_solved_task_s": round(latency / successes, 3) if successes else None,
            "served_model_counts": dict(served),
            "by_difficulty": by_difficulty,
        }
    return summary


def render_report(metadata: dict, summary: dict) -> str:
    lines = [
        "# Live Model-Routing Benchmark",
        "",
        "Keyless free OpenAI-compatible inference with objective-answer tasks.",
        "Free upstream capacity may substitute the requested model; served-model counts record what actually answered.",
        "",
        "| Policy | Pass rate | Calls/task | Total tokens | Tokens/solved | Mean latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in summary.items():
        per_solved = (
            "n/a"
            if stats["tokens_per_solved_task"] is None
            else f'{stats["tokens_per_solved_task"]:,.0f}'
        )
        lines.append(
            f'| {name} | {100*stats["pass_rate"]:.1f}% | '
            f'{stats["mean_calls_per_task"]:.2f} | {stats["total_tokens"]:,} | '
            f'{per_solved} | {stats["mean_latency_s"]:.2f}s |'
        )
    lines += ["", "## Served models", ""]
    for name, stats in summary.items():
        lines.append(f'- **{name}**: {json.dumps(stats["served_model_counts"], sort_keys=True)}')
    lines += [
        "",
        "## Scope",
        "",
        f'- Tasks: {metadata["tasks"]} ({metadata["difficulty_counts"]})',
        "- Evaluator: exact objective answer after JSON extraction.",
        "- Planner: concise reasoning plan; worker receives task plus planner notes.",
        "- Complexity-gated: easy -> Nano; medium -> GPT-OSS; hard -> Lightning planner + GPT-OSS worker.",
        "- Test-driven cascade: Nano -> GPT-OSS -> Lightning planner + GPT-OSS, escalating only after objective failure.",
        "",
        "This is a small routing experiment, not a general model ranking.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/model_routing"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.perf_counter()
    for policy, runner in POLICIES.items():
        for task in TASKS:
            final_call, aggregate = runner(task)
            actual = parse_answer(final_call["content"])
            success = evaluate(task, final_call)
            rows.append(
                {
                    "policy": policy,
                    "task_id": task.task_id,
                    "difficulty": task.difficulty,
                    "expected": task.expected,
                    "actual": actual,
                    "success": int(success),
                    "calls": aggregate["calls"],
                    "prompt_tokens": aggregate["prompt_tokens"],
                    "completion_tokens": aggregate["completion_tokens"],
                    "total_tokens": aggregate["prompt_tokens"] + aggregate["completion_tokens"],
                    "latency_s": round(aggregate["latency_s"], 4),
                    "requested_models": "|".join(aggregate["requested_models"]),
                    "served_models": "|".join(aggregate["served_models"]),
                    "errors": " | ".join(aggregate["errors"]),
                    "raw_final": final_call["content"].replace("\n", "\\n")[:2000],
                }
            )
            print(
                f'{policy:28s} {task.task_id:22s} '
                f'{"PASS" if success else "FAIL"} '
                f'calls={aggregate["calls"]} served={"|".join(aggregate["served_models"])}',
                flush=True,
            )

    summary = summarize(rows)
    metadata = {
        "endpoint": ENDPOINT,
        "requested_models": MODELS,
        "tasks": len(TASKS),
        "difficulty_counts": dict(Counter(task.difficulty for task in TASKS)),
        "policies": list(POLICIES),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "note": (
            "Live keyless free-tier benchmark. Free upstreams may fall back to another healthy "
            "model; requested and served models are both recorded."
        ),
    }
    result = {"metadata": metadata, "summary": summary, "rows": rows}
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "REPORT.md").write_text(render_report(metadata, summary), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
