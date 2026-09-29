"""Run the health assessment for a vehicle and print the maintenance report.

    python predict.py                          # a few test-fleet vehicles
    python predict.py --vehicle V0012          # a specific vehicle (latest reading)
    python predict.py --vehicle V0012 --hour 400 --model lstm
    python predict.py --csv my_readings.csv    # your own sensor log (one vehicle)
    python predict.py --fleet                  # summary table for the whole test fleet
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from pdm.config import ALL_MODELS, PRIMARY_MODEL, RAW_DATA_PATH
from pdm.inference import VehicleHealthMonitor, format_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle")
    ap.add_argument("--hour", type=int, help="assess as of this hour (default: latest)")
    ap.add_argument("--csv", help="CSV with the nine sensor columns, oldest reading first")
    ap.add_argument("--model", default=PRIMARY_MODEL, choices=ALL_MODELS)
    ap.add_argument("--fleet", action="store_true")
    ap.add_argument("--json", action="store_true", help="print raw JSON result")
    args = ap.parse_args()

    monitor = VehicleHealthMonitor()

    if args.csv:
        histories = [pd.read_csv(args.csv)]
    else:
        df = pd.read_csv(RAW_DATA_PATH)
        test_ids = json.loads((monitor.models_dir / "meta.json").read_text())["test_vehicles"]
        if args.vehicle:
            ids = [args.vehicle]
        elif args.fleet:
            ids = test_ids
        else:
            ids = []
        rng = np.random.default_rng(7)
        histories = []
        for vid in ids:
            h = df[df.vehicle_id == vid]
            if h.empty:
                raise SystemExit(f"Unknown vehicle {vid}")
            if args.hour is not None:
                cutoff = args.hour
            elif args.fleet:  # snapshot at a random point late in each vehicle's history
                cutoff = h.timestamp.max() - int(rng.integers(0, 150))
            else:
                cutoff = h.timestamp.max()
            histories.append(h[h.timestamp <= cutoff])
        if not ids:
            # Demo on held-out vehicles: one healthy vehicle, then one failing vehicle
            # observed 150, 60 and 20 hours before its breakdown.
            modes = df[df.vehicle_id.isin(test_ids)].groupby("vehicle_id").failure_mode.first()
            healthy = df[df.vehicle_id == modes[modes == "none"].index[0]]
            failing = df[df.vehicle_id == modes[modes != "none"].index[0]]
            end = failing.timestamp.max()
            histories = [healthy] + [failing[failing.timestamp <= end - k] for k in (150, 60, 20)]

    results = [monitor.assess(h, model=args.model) for h in histories]

    if args.fleet:
        table = pd.DataFrame([{
            "vehicle": r["vehicle_id"], "health_%": round(r["health_score"]),
            "failure_prob_%": round(100 * r["failure_probability"]), "status": r["status"],
            "top_factor": r["factors"][0]["description"] if r["factors"] else "-",
        } for r in results]).sort_values("failure_prob_%", ascending=False)
        print(table.to_string(index=False))
        print("\n" + table.status.value_counts().to_string())
        return

    for r in results:
        print(json.dumps(r, indent=2) if args.json else format_report(r))
        print("-" * 60)


if __name__ == "__main__":
    main()
