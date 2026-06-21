"""Lightweight experiment tracking (Phase 9).

Every backtest/tuning run is appended as one JSON line to
``reports/experiments.jsonl`` so the full optimization history is reproducible
and auditable. A small helper summarizes gains over the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentTracker:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def log(self, name: str, params: dict, metrics: dict, tags: dict | None = None) -> None:
        record: dict[str, Any] = {
            "experiment": name,
            "params": params,
            "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
            "tags": tags or {},
        }
        self.records.append(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def gains_over_baseline(self, baseline_name: str, metric: str, lower_is_better: bool = True) -> dict:
        by_name = {r["experiment"]: r["metrics"].get(metric) for r in self.records
                   if metric in r["metrics"]}
        base = by_name.get(baseline_name)
        if base is None:
            return {}
        gains = {}
        for name, val in by_name.items():
            if val is None:
                continue
            improvement = (base - val) if lower_is_better else (val - base)
            gains[name] = {
                "value": val,
                "abs_gain_vs_baseline": improvement,
                "pct_gain_vs_baseline": (improvement / abs(base) * 100.0) if base else float("nan"),
            }
        return gains
