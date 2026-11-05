from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.freight_model import make_features, regression_metrics, validate_input


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "load_id": ["TR-1", "TR-2"],
                "pickup": ["Lexington", "Chicago"],
                "delivery": ["Fort Wayne", "Mobile"],
                "pickup_lat": [38.0, 41.8],
                "pickup_lon": [-84.5, -87.6],
                "delivery_lat": [41.1, 30.7],
                "delivery_lon": [-85.1, -88.0],
                "distance": [360.0, 900.0],
                "equipment": ["Dry Van", "Reefer"],
                "weight": [-32000.0, np.nan],
                "date": ["2025-10-01", "2025-10-04"],
                "market_index": [1.0, np.nan],
                "quote_signal": [2.0, 2.2],
                "posted_rate": [800.0, 2200.0],
            }
        )

    def test_negative_weight_is_repaired_and_flagged(self) -> None:
        features = make_features(self.frame)
        self.assertEqual(features.loc[0, "weight"], 32000.0)
        self.assertEqual(features.loc[0, "weight_was_negative"], 1)
        self.assertEqual(features.loc[1, "weight_missing"], 1)

    def test_date_and_lane_features_are_deterministic(self) -> None:
        with_placeholder = self.frame.assign(predicted_rate=np.nan)
        features = make_features(with_placeholder)
        self.assertEqual(features.loc[0, "lane"], "Lexington -> Fort Wayne")
        self.assertEqual(features.loc[0, "month"], 10)
        self.assertEqual(features.loc[1, "is_weekend"], 1)
        self.assertNotIn("predicted_rate", features.columns)

    def test_schema_validation_rejects_duplicate_ids(self) -> None:
        bad = pd.concat([self.frame.iloc[[0]], self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "load_id"):
            validate_input(bad, require_target=True)

    def test_metrics_are_exact_for_perfect_prediction(self) -> None:
        metrics = regression_metrics([100.0, 200.0], [100.0, 200.0])
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
