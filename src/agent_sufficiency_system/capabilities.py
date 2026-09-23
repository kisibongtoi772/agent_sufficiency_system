from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_intent(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


class TokenCounter:
    """Small tokenizer wrapper used by the capability benchmark.

    Uses tiktoken when available and otherwise falls back to ceil(chars / 4).
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
        return math.ceil(len(text) / 4.0)


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    aliases: tuple[str, ...]
    source_layer: str
    implementation_ref: str
    validated: bool = True

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(normalize_intent(x) for x in (self.name, *self.aliases)))

    def compact_metadata(self) -> str:
        return (
            f"capability={self.name}; source={self.source_layer}; "
            f"ref={self.implementation_ref}; validated={str(self.validated).lower()}; "
            f"description={self.description}"
        )


@dataclass(frozen=True)
class CapabilityRequest:
    intent: str
    expected_name: str | None = None

    @property
    def key(self) -> str:
        return normalize_intent(self.intent)


@dataclass(frozen=True)
class SearchProbe:
    layer: str
    hit: bool
    candidate: str | None
    context: str
    elapsed_ms: float


@dataclass
class Resolution:
    request: CapabilityRequest
    capability: Capability
    source: str
    synthesized: bool
    cache_hit: bool
    probes: list[SearchProbe] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def trace_text(self) -> str:
        lines = [f"REQUEST intent={self.request.intent}"]
        for probe in self.probes:
            status = "HIT" if probe.hit else "MISS"
            candidate = f" candidate={probe.candidate}" if probe.candidate else ""
            lines.append(f"{probe.layer}: {status}{candidate}")
        lines.append("RESOLVED " + self.capability.compact_metadata())
        return "\n".join(lines)


class CapabilityCatalog:
    """Exact/alias capability catalog with compact search traces.

    The catalog itself stays outside model context. Search returns only a hit/miss
    trace plus one compact candidate descriptor, matching progressive disclosure.
    """

    def __init__(self, layer: str, capabilities: Iterable[Capability] = ()) -> None:
        self.layer = layer
        self._by_key: dict[str, Capability] = {}
        for capability in capabilities:
            self.add(capability)

    def add(self, capability: Capability) -> None:
        for key in capability.keys:
            self._by_key[key] = capability

    def search(self, request: CapabilityRequest) -> tuple[Capability | None, SearchProbe]:
        t0 = time.perf_counter()
        capability = self._by_key.get(request.key)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if capability is None:
            context = f"layer={self.layer}; query={request.key}; result=none"
            return None, SearchProbe(self.layer, False, None, context, elapsed_ms)
        context = (
            f"layer={self.layer}; query={request.key}; result={capability.name}; "
            f"ref={capability.implementation_ref}"
        )
        return capability, SearchProbe(
            self.layer,
            True,
            capability.name,
            context,
            elapsed_ms,
        )

    def __len__(self) -> int:
        return len({id(capability) for capability in self._by_key.values()})


class CapabilityLibrary(CapabilityCatalog):
    def __init__(self) -> None:
        super().__init__("cache")

    def remember(self, capability: Capability) -> None:
        adopted = Capability(
            name=capability.name,
            description=capability.description,
            aliases=capability.aliases,
            source_layer=capability.source_layer,
            implementation_ref=capability.implementation_ref,
            validated=capability.validated,
        )
        self.add(adopted)


class SearchFirstResolver:
    """Resolve capabilities before synthesizing new code.

    Search order is intentionally explicit and pluggable:
    cache -> repo -> package -> MCP -> skill -> GitHub/OSS -> synthesize.
    """

    def __init__(
        self,
        layers: Sequence[CapabilityCatalog],
        *,
        cache: CapabilityLibrary | None = None,
        cache_found: bool = True,
        cache_synthesized: bool = True,
    ) -> None:
        self.layers = list(layers)
        self.cache = cache or CapabilityLibrary()
        self.cache_found = cache_found
        self.cache_synthesized = cache_synthesized

    def resolve(self, request: CapabilityRequest) -> Resolution:
        started = time.perf_counter()
        probes: list[SearchProbe] = []

        capability, probe = self.cache.search(request)
        probes.append(probe)
        if capability is not None:
            return Resolution(
                request=request,
                capability=capability,
                source="cache",
                synthesized=False,
                cache_hit=True,
                probes=probes,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        for layer in self.layers:
            capability, probe = layer.search(request)
            probes.append(probe)
            if capability is None:
                continue
            if self.cache_found:
                self.cache.remember(capability)
            return Resolution(
                request=request,
                capability=capability,
                source=layer.layer,
                synthesized=False,
                cache_hit=False,
                probes=probes,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        capability = self._synthesize(request)
        if self.cache_synthesized:
            self.cache.remember(capability)
        return Resolution(
            request=request,
            capability=capability,
            source="synthesized",
            synthesized=True,
            cache_hit=False,
            probes=probes,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _synthesize(request: CapabilityRequest) -> Capability:
        slug = request.key.replace(" ", "_") or "generated_capability"
        return Capability(
            name=request.expected_name or slug,
            description=f"Generated capability for: {request.intent}",
            aliases=(request.intent,),
            source_layer="synthesized",
            implementation_ref=f"generated/{slug}.py",
            validated=True,
        )


def synthesis_artifact_proxy(capability: Capability, request: CapabilityRequest) -> str:
    """Deterministic proxy for model-visible work when a capability is built.

    This is not claimed as real LLM reasoning. It creates a transparent, fixed-shape
    plan/code/test/debug artifact so the benchmark can estimate the material avoided
    when reuse prevents implementation from happening again.
    """

    header = (
        f"IMPLEMENT {capability.name}\n"
        f"REQUEST {request.intent}\n"
        "PLAN validate inputs; choose dependency; implement adapter; add error handling; "
        "write unit tests; run tests; inspect failures; document usage.\n"
    )
    code_lines = [
        f"def {capability.name}(value, options=None):",
        '    """Generated benchmark proxy implementation."""',
        "    options = options or {}",
        "    if value is None:",
        "        raise ValueError('value is required')",
    ]
    for idx in range(56):
        code_lines.append(
            f"    step_{idx:02d} = process_step(value, options, stage={idx})  # validate/transform"
        )
    code_lines.extend(
        [
            "    return step_55",
            "",
            f"def test_{capability.name}_basic():",
            "    sample = build_fixture()",
            f"    result = {capability.name}(sample)",
            "    assert result is not None",
        ]
    )
    tests = "\n".join(
        f"test_case_{idx:02d}: arrange fixture; execute; assert invariant; inspect error path"
        for idx in range(24)
    )
    debug = "\n".join(
        f"debug_pass_{idx:02d}: inspect trace, narrow root cause, rerun focused test, record result"
        for idx in range(12)
    )
    return header + "\n".join(code_lines) + "\nTESTS\n" + tests + "\nDEBUG\n" + debug


def adoption_artifact_proxy(capability: Capability, request: CapabilityRequest) -> str:
    """Small adapter/verification payload for an existing capability."""

    return (
        f"ADOPT {capability.name}\n"
        f"source={capability.source_layer}\n"
        f"ref={capability.implementation_ref}\n"
        f"request={request.intent}\n"
        "verify compatibility; map inputs; call capability; validate one representative output"
    )
