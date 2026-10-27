from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.build_current_rushing_windows import (
    DEFAULT_SHARP_RUSH_DEF_RANK,
    resolve_sharp_rush_def_rank,
)


class ResolveSharpRushDefRankTests(unittest.TestCase):
    def test_prefers_current_rank(self) -> None:
        rank, source = resolve_sharp_rush_def_rank(
            pd.Series({"sharp_rush_eff_rank": 7.0, "sharp_def_rank": 12.0})
        )
        self.assertEqual((rank, source), (7, "sharp_rush_eff_rank"))

    def test_uses_legacy_rank_when_current_is_missing(self) -> None:
        rank, source = resolve_sharp_rush_def_rank(
            pd.Series({"sharp_rush_eff_rank": np.nan, "sharp_def_rank": 12.0})
        )
        self.assertEqual((rank, source), (12, "sharp_def_rank"))

    def test_uses_neutral_midpoint_when_all_sources_are_invalid(self) -> None:
        invalid_values = [np.nan, None, "", "not-a-rank", np.inf, 0, 33]
        for value in invalid_values:
            with self.subTest(value=value):
                rank, source = resolve_sharp_rush_def_rank(
                    pd.Series({"sharp_rush_eff_rank": value, "sharp_def_rank": np.nan})
                )
                self.assertEqual(rank, DEFAULT_SHARP_RUSH_DEF_RANK)
                self.assertEqual(source, "default_midpoint")


if __name__ == "__main__":
    unittest.main()
