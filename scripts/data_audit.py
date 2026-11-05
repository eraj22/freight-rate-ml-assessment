from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible data-quality audit.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_audit.json"))
    return parser.parse_args()


def profile(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_load_ids": int(frame["load_id"].duplicated().sum()),
        "date_min": str(pd.to_datetime(frame["date"]).min().date()),
        "date_max": str(pd.to_datetime(frame["date"]).max().date()),
        "missing_by_column": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if count > 0
        },
        "negative_weight": int((pd.to_numeric(frame["weight"], errors="coerce") < 0).sum()),
    }


def main() -> None:
    args = parse_args()
    training = pd.read_csv(args.data_dir / "train_test.csv")
    validation = pd.read_csv(args.data_dir / "validation.csv")
    train_cities = set(training["pickup"]) | set(training["delivery"])
    validation_cities = set(validation["pickup"]) | set(validation["delivery"])
    train_lanes = set(zip(training["pickup"], training["delivery"]))
    validation_lanes = set(zip(validation["pickup"], validation["delivery"]))
    target = training["posted_rate"]
    payload = {
        "training": profile(training),
        "validation": profile(validation),
        "unseen_validation_cities": sorted(validation_cities - train_cities),
        "unseen_validation_lane_count": len(validation_lanes - train_lanes),
        "target": {
            "mean": float(target.mean()),
            "median": float(target.median()),
            "p99": float(target.quantile(0.99)),
            "maximum": float(target.max()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
