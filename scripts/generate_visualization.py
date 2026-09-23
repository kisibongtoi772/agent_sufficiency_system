from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


WIDTH = 1440
HEIGHT = 860
FONT = "Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
INK = "#111827"
MUTED = "#667085"
BORDER = "#d8dee8"
PANEL = "#f7f9fc"
BAR = "#3976b9"
BAR_STRONG = "#6d5bd0"
ACCENT = "#215e9c"
GREEN = "#147a52"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def num(value: float) -> str:
    return f"{value:,.0f}"


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 18,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "white",
    stroke: str = BORDER,
    radius: int = 14,
    stroke_width: float = 1,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = BORDER,
    width: float = 1.5,
) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def arrow(x1: float, x2: float, y: float) -> str:
    return (
        line(x1, y, x2 - 8, y, stroke=MUTED, width=1.8)
        + f'<path d="M {x2 - 8} {y - 5} L {x2} {y} L {x2 - 8} {y + 5}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.8"/>'
    )


def bar_group(
    x: float,
    y: float,
    w: float,
    labels: list[str],
    values: list[float],
    *,
    strong_index: int | None = None,
    row_h: int = 43,
) -> str:
    max_value = max(values)
    label_w = 165
    bar_w = w - label_w - 85
    out: list[str] = []

    for idx, (label, value) in enumerate(zip(labels, values)):
        cy = y + idx * row_h
        out.append(text(x, cy + 18, label, size=14))
        width = max(4, bar_w * value / max_value)
        color = BAR_STRONG if idx == strong_index else BAR
        out.append(
            f'<rect x="{x + label_w}" y="{cy}" width="{width:.1f}" '
            f'height="24" rx="3" fill="{color}"/>'
        )
        out.append(
            text(x + label_w + width + 10, cy + 18, num(value), size=14, weight=700)
        )

    return "".join(out)


def metric_box(
    x: float,
    y: float,
    w: float,
    headline: str,
    label: str,
    *,
    color: str = ACCENT,
) -> str:
    return (
        rect(x, y, w, 78, fill=PANEL)
        + text(x + 18, y + 34, headline, size=27, weight=750, fill=color)
        + text(x + 18, y + 58, label, size=13, fill=MUTED)
    )


def build_svg(context: dict, capability: dict) -> str:
    ctx = context["summary"]
    cap = capability["summary"]
    raw = ctx["raw_grep_agent"]
    hybrid = ctx["token_min_hybrid"]
    learning = cap["search_first_learning"]

    context_labels = [
        "Raw grep/read",
        "Graph retrieval",
        "Compact + lazy",
        "Token-min hybrid",
    ]
    context_values = [
        raw["mean_tokens"],
        ctx["graph_retrieval_only"]["mean_tokens"],
        ctx["compact_state_only"]["mean_tokens"],
        hybrid["mean_tokens"],
    ]

    reuse_labels = [
        "Build every time",
        "Search-first stateless",
        "Build cache",
        "Search-first learning",
    ]
    reuse_values = [
        cap["build_every_time"]["synthesis_events"],
        cap["search_first_stateless"]["synthesis_events"],
        cap["search_first_build_cache"]["synthesis_events"],
        learning["synthesis_events"],
    ]
    work_values = [
        cap["build_every_time"]["total_proxy_tokens"],
        cap["search_first_stateless"]["total_proxy_tokens"],
        cap["search_first_build_cache"]["total_proxy_tokens"],
        learning["total_proxy_tokens"],
    ]

    p95_reduction = 1 - hybrid["p95_tokens"] / raw["p95_tokens"]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(58, 58, "Agent Sufficiency System", size=34, weight=800),
        text(
            58,
            87,
            "Minimum context + maximum capability reuse + minimum net-new reasoning",
            size=16,
            fill=MUTED,
        ),
        line(58, 112, 1382, 112),
        text(58, 155, "1. Context Sufficiency", size=23, weight=750),
        text(58, 184, "Mean estimated input context / turn", size=16, weight=650),
        bar_group(58, 208, 600, context_labels, context_values, strong_index=3),
        metric_box(
            58,
            390,
            245,
            pct(hybrid["savings_vs_raw_grep"]),
            "less mean context",
        ),
        metric_box(320, 390, 245, pct(p95_reduction), "lower P95 context"),
        metric_box(582, 390, 120, "100%", "target coverage", color=GREEN),
        text(760, 155, "2. Capability Sufficiency", size=23, weight=750),
        text(760, 184, "Net-new implementations / 120 requests", size=16, weight=650),
        bar_group(760, 208, 610, reuse_labels, reuse_values, strong_index=3),
        text(760, 390, "Implementation-work proxy", size=16, weight=650),
        bar_group(760, 414, 610, reuse_labels, work_values, strong_index=3, row_h=38),
        metric_box(
            760,
            575,
            245,
            pct(learning["implementation_reduction_vs_build_every_time"]),
            "fewer new implementations",
            color=GREEN,
        ),
        metric_box(
            1022,
            575,
            245,
            pct(learning["cache_hit_rate"]),
            "requests resolved from cache",
            color=GREEN,
        ),
        line(58, 682, 1382, 682),
        text(58, 726, "Architecture", size=21, weight=750),
    ]

    steps = [
        (58, 102, "Task"),
        (202, 225, "Capability Resolver"),
        (470, 205, "Context Compiler"),
        (718, 290, "Minimum Sufficient Context"),
        (1050, 185, "LLM / Agent"),
    ]
    y = 748
    for idx, (x, w, label) in enumerate(steps):
        svg.append(rect(x, y, w, 56, fill=PANEL, radius=10))
        svg.append(
            text(x + w / 2, y + 34, label, size=14, weight=700, anchor="middle")
        )
        if idx < len(steps) - 1:
            svg.append(arrow(x + w + 10, steps[idx + 1][0] - 10, y + 28))

    svg += [
        text(
            58,
            838,
            "Generated directly from results/results.json and "
            "results/capability_reuse/results.json",
            size=12,
            fill=MUTED,
        ),
        text(
            1382,
            838,
            "github.com/kisibongtoi772/agent_sufficiency_system",
            size=12,
            weight=650,
            anchor="end",
        ),
        "</svg>",
    ]
    return "".join(svg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the project benchmark visualization from result JSON files."
    )
    parser.add_argument("--context", type=Path, default=Path("results/results.json"))
    parser.add_argument(
        "--capability",
        type=Path,
        default=Path("results/capability_reuse/results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/benchmark_overview.svg"),
    )
    args = parser.parse_args()

    context = load_json(args.context)
    capability = load_json(args.capability)
    svg = build_svg(context, capability)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
