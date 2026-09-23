from __future__ import annotations

import unittest

from agent_sufficiency_system import (
    Capability,
    CapabilityCatalog,
    CapabilityRequest,
    SearchFirstResolver,
)


class CapabilityResolverTests(unittest.TestCase):
    def test_searches_in_order_and_stops_on_hit(self) -> None:
        package = Capability(
            name="csv_to_parquet",
            description="Convert CSV data to parquet.",
            aliases=("convert csv to parquet",),
            source_layer="package",
            implementation_ref="pyarrow.parquet",
        )
        resolver = SearchFirstResolver(
            [
                CapabilityCatalog("repo"),
                CapabilityCatalog("package", [package]),
                CapabilityCatalog("mcp"),
            ]
        )

        result = resolver.resolve(CapabilityRequest("convert csv to parquet"))

        self.assertEqual(result.source, "package")
        self.assertEqual([p.layer for p in result.probes], ["cache", "repo", "package"])
        self.assertFalse(result.synthesized)

    def test_synthesized_capability_is_reused_from_cache(self) -> None:
        resolver = SearchFirstResolver([CapabilityCatalog("repo"), CapabilityCatalog("package")])
        request = CapabilityRequest("normalize obscure telemetry format", "normalize_telemetry")

        first = resolver.resolve(request)
        second = resolver.resolve(request)

        self.assertTrue(first.synthesized)
        self.assertFalse(second.synthesized)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.source, "cache")

    def test_existing_capability_can_be_adopted_into_cache(self) -> None:
        repo_capability = Capability(
            name="retry_http",
            description="Retry transient HTTP failures.",
            aliases=("retry http requests",),
            source_layer="repo",
            implementation_ref="utils/retry.py",
        )
        resolver = SearchFirstResolver([CapabilityCatalog("repo", [repo_capability])])
        request = CapabilityRequest("retry http requests")

        first = resolver.resolve(request)
        second = resolver.resolve(request)

        self.assertEqual(first.source, "repo")
        self.assertTrue(second.cache_hit)
        self.assertEqual(len(second.probes), 1)


if __name__ == "__main__":
    unittest.main()
