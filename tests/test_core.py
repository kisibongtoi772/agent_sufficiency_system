from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_sufficiency_system import (
    CompactHistory,
    SymbolGraphIndex,
    TurnRecord,
    extract_symbol_name,
)


class CoreTests(unittest.TestCase):
    def test_extract_symbol_name_prefers_backticks(self) -> None:
        self.assertEqual(extract_symbol_name("Inspect `target_fn` safely."), "target_fn")

    def test_structural_index_returns_target_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.py").write_text(
                "def helper():\n    return 1\n\ndef target_fn():\n    return helper()\n",
                encoding="utf-8",
            )
            index = SymbolGraphIndex(root)
            context = index.graph_context("target_fn")

            self.assertIn("TARGET function target_fn", context)
            self.assertIn("sample.py", context)
            self.assertIn("helper", context)

    def test_compact_history_drops_old_tool_payloads(self) -> None:
        turns = [
            TurnRecord(
                user=f"Find `symbol_{idx}`",
                assistant="located",
                tool_result=(f"payload_{idx} " * 200),
            )
            for idx in range(6)
        ]
        rendered = CompactHistory(recent=1).render(turns)

        self.assertIn("CHECKPOINT", rendered)
        self.assertIn("symbol_5", rendered)
        self.assertIn("symbol_0", rendered)
        self.assertNotIn("payload_0", rendered)
        self.assertIn("payload_5", rendered)


if __name__ == "__main__":
    unittest.main()
