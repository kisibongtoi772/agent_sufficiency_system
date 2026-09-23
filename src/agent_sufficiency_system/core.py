from __future__ import annotations

import ast
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# -----------------------------
# Token accounting
# -----------------------------
class TokenCounter:
    """Use tiktoken when available; otherwise a transparent 4-chars/token estimate.

    The fallback is intentionally simple and deterministic. The benchmark also records
    character counts, so reduction ratios remain auditable independent of tokenizer.
    """

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        self.mode = "estimate_4chars"
        self._enc = None
        try:
            import tiktoken  # type: ignore

            self._enc = tiktoken.get_encoding(encoding_name)
            self.mode = f"tiktoken:{encoding_name}"
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        # Graphify itself documents ~4 chars/token for traversal budgeting.
        return math.ceil(len(text) / 4.0)


# -----------------------------
# Structural code index
# -----------------------------
@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    file: str
    lineno: int
    end_lineno: int
    calls: tuple[str, ...] = ()


@dataclass
class FileRecord:
    relpath: str
    text: str
    lines: list[str]
    imports: tuple[str, ...]
    symbols: list[Symbol]


class SymbolGraphIndex:
    """Dependency-free Graphify-like AST index for Python code.

    It does not claim to reproduce Graphify. It implements the same design idea:
    precompute structural facts once, then retrieve compact symbol-level context.
    """

    def __init__(self, root: Path, max_files: int | None = None) -> None:
        self.root = root.resolve()
        self.files: dict[str, FileRecord] = {}
        self.symbols_by_name: dict[str, list[Symbol]] = {}
        self.build_seconds = 0.0
        self._build(max_files=max_files)

    def _build(self, max_files: int | None) -> None:
        t0 = time.perf_counter()
        paths = sorted(self.root.rglob("*.py"))
        if max_files:
            paths = paths[:max_files]
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(text)
            except Exception:
                continue
            rel = str(path.relative_to(self.root))
            lines = text.splitlines()
            imports: list[str] = []
            symbols: list[Symbol] = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    calls: list[str] = []
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call):
                            fn = sub.func
                            if isinstance(fn, ast.Name):
                                calls.append(fn.id)
                            elif isinstance(fn, ast.Attribute):
                                calls.append(fn.attr)
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    sym = Symbol(
                        name=node.name,
                        kind=kind,
                        file=rel,
                        lineno=int(getattr(node, "lineno", 1)),
                        end_lineno=int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
                        calls=tuple(dict.fromkeys(calls)),
                    )
                    symbols.append(sym)
                    self.symbols_by_name.setdefault(sym.name, []).append(sym)
            self.files[rel] = FileRecord(
                relpath=rel,
                text=text,
                lines=lines,
                imports=tuple(dict.fromkeys(imports)),
                symbols=symbols,
            )
        self.build_seconds = time.perf_counter() - t0

    @property
    def corpus_text(self) -> str:
        parts = []
        for rel, rec in self.files.items():
            parts.append(f"\n### FILE {rel}\n{rec.text}")
        return "".join(parts)

    @property
    def corpus_chars(self) -> int:
        return sum(len(r.text) for r in self.files.values())

    def unique_symbols(self, min_lines: int = 3, max_lines: int = 120) -> list[Symbol]:
        out: list[Symbol] = []
        for name, syms in self.symbols_by_name.items():
            if len(syms) != 1:
                continue
            s = syms[0]
            span = s.end_lineno - s.lineno + 1
            if min_lines <= span <= max_lines:
                out.append(s)
        return sorted(out, key=lambda s: (s.file, s.lineno, s.name))

    def _snippet(self, sym: Symbol, pad: int = 2, max_chars: int = 5000) -> str:
        rec = self.files[sym.file]
        a = max(1, sym.lineno - pad)
        b = min(len(rec.lines), sym.end_lineno + pad)
        body = "\n".join(f"{i:04d}: {rec.lines[i-1]}" for i in range(a, b + 1))
        if len(body) > max_chars:
            body = body[:max_chars] + "\n...<snippet truncated>"
        return body

    def graph_context(self, symbol_name: str, neighbor_limit: int = 3, char_budget: int = 16000) -> str:
        syms = self.symbols_by_name.get(symbol_name, [])
        if not syms:
            return "NO_STRUCTURAL_MATCH"
        sym = syms[0]
        rec = self.files[sym.file]
        out = [
            f"TARGET {sym.kind} {sym.name} @ {sym.file}:{sym.lineno}-{sym.end_lineno}",
            f"FILE_IMPORTS: {', '.join(rec.imports[:12]) or '(none)'}",
            "TARGET_SNIPPET:\n" + self._snippet(sym),
        ]

        neighbors: list[Symbol] = []
        for call in sym.calls:
            hits = self.symbols_by_name.get(call, [])
            if len(hits) == 1 and hits[0] != sym:
                neighbors.append(hits[0])
            if len(neighbors) >= neighbor_limit:
                break
        if neighbors:
            out.append("STRUCTURAL_NEIGHBORS:")
            for n in neighbors:
                out.append(f"- calls -> {n.name} @ {n.file}:{n.lineno}-{n.end_lineno}")
                out.append(self._snippet(n, pad=1, max_chars=1800))
        result = "\n".join(out)
        return result[:char_budget]

    def grep_context(self, symbol_name: str, max_files: int = 5, char_budget: int = 120000) -> str:
        # Rank definition-containing files first, then any lexical occurrence.
        pat_def = re.compile(rf"^(?:async\s+def|def|class)\s+{re.escape(symbol_name)}\b", re.M)
        scored: list[tuple[int, int, str]] = []
        for rel, rec in self.files.items():
            if symbol_name not in rec.text:
                continue
            score = 100 if pat_def.search(rec.text) else 10
            occurrences = rec.text.count(symbol_name)
            scored.append((score, occurrences, rel))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        parts: list[str] = []
        used = 0
        for _, _, rel in scored[:max_files]:
            block = f"\n### GREP_MATCH_FILE {rel}\n{self.files[rel].text}\n"
            if used + len(block) > char_budget:
                block = block[: max(0, char_budget - used)]
            parts.append(block)
            used += len(block)
            if used >= char_budget:
                break
        return "".join(parts) or "NO_GREP_MATCH"


# -----------------------------
# Lazy tool registry
# -----------------------------
@dataclass(frozen=True)
class ToolSpec:
    name: str
    domain: str
    schema: str


class ToolRegistry:
    def __init__(self, n_distractors: int = 44) -> None:
        core = [
            ("search_code", "code"),
            ("read_symbol", "code"),
            ("trace_dependency", "code"),
            ("read_file", "code"),
        ]
        specs: list[ToolSpec] = []
        for name, domain in core:
            specs.append(ToolSpec(name, domain, self._schema(name, domain, 0)))
        domains = ["calendar", "email", "crm", "billing", "browser", "cloud", "database", "docs"]
        for i in range(n_distractors):
            domain = domains[i % len(domains)]
            name = f"{domain}_operation_{i:02d}"
            specs.append(ToolSpec(name, domain, self._schema(name, domain, i + 1)))
        self.specs = specs

    @staticmethod
    def _schema(name: str, domain: str, idx: int) -> str:
        desc = (
            f"Tool {name} performs a {domain} operation. Use only when the task requires {domain}. "
            "Returns structured JSON. Validate required identifiers before calling. "
            "Do not infer missing IDs. Prefer narrow queries and request only required fields."
        )
        obj = {
            "type": "function",
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Precise operation or search query."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "fields": {"type": "array", "items": {"type": "string"}},
                    "include_metadata": {"type": "boolean"},
                    "mode": {"type": "string", "enum": ["fast", "balanced", "deep"]},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "x_benchmark_id": idx,
        }
        return json.dumps(obj, separators=(",", ":"))

    def all_schemas(self) -> str:
        return "\n".join(s.schema for s in self.specs)

    def lazy_code_schemas(self) -> str:
        wanted = {"search_code", "read_symbol", "trace_dependency", "read_file"}
        return "\n".join(s.schema for s in self.specs if s.name in wanted)

    def index_only(self) -> str:
        return "\n".join(f"{s.name}: {s.domain}" for s in self.specs)


# -----------------------------
# Memory policies
# -----------------------------
@dataclass
class TurnRecord:
    user: str
    assistant: str
    tool_result: str


class FullHistory:
    def render(self, turns: Sequence[TurnRecord]) -> str:
        parts: list[str] = []
        for i, t in enumerate(turns):
            parts.append(f"TURN {i+1} USER:\n{t.user}\nASSISTANT:\n{t.assistant}\nTOOL_RESULT:\n{t.tool_result}")
        return "\n".join(parts)


class CompactHistory:
    """Deterministic, LLM-free checkpoint + recent window."""

    def __init__(self, recent: int = 2, checkpoint_chars: int = 1400) -> None:
        self.recent = recent
        self.checkpoint_chars = checkpoint_chars

    def render(self, turns: Sequence[TurnRecord]) -> str:
        if not turns:
            return ""
        old = turns[:-self.recent] if len(turns) > self.recent else []
        recent = turns[-self.recent :]
        checkpoint_lines: list[str] = []
        for i, t in enumerate(old[-24:]):
            # Store a tiny delta, not the transcript/tool payload.
            symbol = extract_symbol_name(t.user) or "unknown"
            checkpoint_lines.append(f"#{len(turns)-len(old)+i}: inspected={symbol}; result=located")
        checkpoint = "CHECKPOINT:\n" + "\n".join(checkpoint_lines)
        checkpoint = checkpoint[-self.checkpoint_chars :]
        recent_text = FullHistory().render(recent)
        return checkpoint + "\nRECENT:\n" + recent_text


# -----------------------------
# Context compiler / agent shell
# -----------------------------
def extract_symbol_name(query: str) -> str | None:
    m = re.search(r"`([^`]+)`", query)
    if m:
        return m.group(1)
    m = re.search(r"symbol\s+([A-Za-z_][A-Za-z0-9_]*)", query, re.I)
    return m.group(1) if m else None


@dataclass
class ContextPackage:
    policy: str
    system: str
    tool_context: str
    memory_context: str
    retrieval_context: str
    user_query: str
    build_ms: float
    expected_file_hit: bool
    expected_symbol_hit: bool

    def components(self) -> dict[str, str]:
        return {
            "system": self.system,
            "tools": self.tool_context,
            "memory": self.memory_context,
            "retrieval": self.retrieval_context,
            "query": self.user_query,
        }


class TokenMinAgent:
    """Minimal context compiler. Replace the model adapter with any LLM SDK."""

    STABLE_SYSTEM = (
        "You are a code agent. Use retrieved evidence only. Prefer symbol-level context. "
        "Do not request full files unless the symbol snippet is insufficient. Keep state external. "
        "Return concise edits with file and line references."
    )

    RAW_SYSTEM = (
        "You are a general coding agent. Inspect repository files and available tools to answer the request. "
        "Use conversation history for continuity and preserve prior tool outputs when useful."
    )

    def __init__(self, index: SymbolGraphIndex, tools: ToolRegistry) -> None:
        self.index = index
        self.tools = tools
        self.full_history = FullHistory()
        self.compact_history = CompactHistory()

    def compile_context(
        self,
        policy: str,
        query: str,
        turns: Sequence[TurnRecord],
        expected_file: str,
    ) -> ContextPackage:
        symbol = extract_symbol_name(query) or ""
        t0 = time.perf_counter()

        if policy == "full_repo_upper_bound":
            system = self.RAW_SYSTEM
            tool_context = self.tools.all_schemas()
            memory = self.full_history.render(turns)
            retrieval = self.index.corpus_text
        elif policy == "raw_grep_agent":
            system = self.RAW_SYSTEM
            tool_context = self.tools.all_schemas()
            memory = self.full_history.render(turns)
            retrieval = self.index.grep_context(symbol)
        elif policy == "graph_retrieval_only":
            system = self.RAW_SYSTEM
            tool_context = self.tools.all_schemas()
            memory = self.full_history.render(turns)
            retrieval = self.index.graph_context(symbol)
        elif policy == "compact_state_only":
            system = self.STABLE_SYSTEM
            tool_context = self.tools.index_only() + "\nLOADED:\n" + self.tools.lazy_code_schemas()
            memory = self.compact_history.render(turns)
            retrieval = self.index.grep_context(symbol, max_files=3, char_budget=60000)
        elif policy == "token_min_hybrid":
            system = self.STABLE_SYSTEM
            tool_context = self.tools.index_only() + "\nLOADED:\n" + self.tools.lazy_code_schemas()
            memory = self.compact_history.render(turns)
            retrieval = self.index.graph_context(symbol, neighbor_limit=2, char_budget=12000)
        else:
            raise ValueError(f"unknown policy: {policy}")

        ms = (time.perf_counter() - t0) * 1000.0
        return ContextPackage(
            policy=policy,
            system=system,
            tool_context=tool_context,
            memory_context=memory,
            retrieval_context=retrieval,
            user_query=query,
            build_ms=ms,
            expected_file_hit=expected_file in retrieval,
            expected_symbol_hit=symbol in retrieval,
        )


def make_synthetic_turn(query: str, expected_file: str, turn_id: int) -> TurnRecord:
    symbol = extract_symbol_name(query) or "symbol"
    assistant = (
        f"Located {symbol} and inspected its implementation. The definition is in {expected_file}. "
        f"Turn {turn_id} completed with a concise structural explanation."
    )
    # Deliberately resembles verbose raw compiler/search output that agents often retain.
    tool_result = (
        f"search_result turn={turn_id} symbol={symbol} file={expected_file}\n"
        + ("match metadata path line symbol import dependency candidate status=ok; " * 18)
    )
    return TurnRecord(user=query, assistant=assistant, tool_result=tool_result)
