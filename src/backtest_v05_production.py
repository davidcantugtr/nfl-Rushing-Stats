from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import backtest_v04_scoring as bt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

# Conservative production calibration: moves materially toward the walk-forward
# result without letting one 2025 sample fully erase forward-looking team/matchup context.
PRODUCTION_WEIGHTS = {
    "team_env": 0.15,
    "share": 0.30,
    "defense": 0.10,
    "history": 0.30,
    "consistency": 0.10,
    "confidence": 0.05,
}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    rows = bt.build_rows()
    out = []
    weekly_out = []
    for split_name, weeks in [
        ("train_wk5_12", bt.TRAIN_WEEKS),
        ("test_wk13_18", bt.TEST_WEEKS),
        ("all_wk5_18", bt.BACKTEST_WEEKS),
    ]:
        agg, weekly = bt.metrics(rows, PRODUCTION_WEIGHTS, weeks)
        out.append({"model": "v0.5 production calibrated", "split": split_name, **agg})
        for row in weekly:
            weekly_out.append({"model": "v0.5 production calibrated", "split": split_name, **row})

    buckets = []
    test = [dict(r, score=bt.score(r, PRODUCTION_WEIGHTS)) for r in rows if r["week"] in bt.TEST_WEEKS]
    for lo, hi in [(0,40),(40,50),(50,60),(60,70),(70,80),(80,101)]:
        b = [r for r in test if lo <= r["score"] < hi]
        if not b:
            continue
        buckets.append({
            "model": "v0.5 production calibrated",
            "score_band": f"{lo}-{hi-1}",
            "n": len(b),
            "avg_actual_yards": mean([r["actual_yards"] for r in b]),
            "median_actual_yards": float(np.median([r["actual_yards"] for r in b])),
            "hit20_rate": mean([r["actual_yards"] >= 20 for r in b]),
            "hit25_rate": mean([r["actual_yards"] >= 25 for r in b]),
        })

    weights = [{"component": k, "production_weight": v} for k, v in PRODUCTION_WEIGHTS.items()]
    write_csv(DATA / "backtest_v05_production_summary_2025.csv", list(out[0]), out)
    write_csv(DATA / "backtest_v05_production_weekly_2025.csv", list(weekly_out[0]), weekly_out)
    write_csv(DATA / "backtest_v05_production_buckets_2025.csv", list(buckets[0]), buckets)
    write_csv(DATA / "backtest_v05_production_weights.csv", list(weights[0]), weights)

    test_metrics, _ = bt.metrics(rows, PRODUCTION_WEIGHTS, bt.TEST_WEEKS)
    print("Production test:", {k: round(v,4) if isinstance(v,float) else v for k,v in test_metrics.items()})
    print("Production weights:", PRODUCTION_WEIGHTS)


if __name__ == "__main__":
    main()
