import json
import tempfile
import unittest
from pathlib import Path

from tsad_v2.evaluation import evaluate_predictions
from tsad_v2.io import write_jsonl


class EvaluationTests(unittest.TestCase):
    def test_one_prediction_is_scored_for_each_series_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "series.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "sample_id": "ucr_001",
                        "series_id": "ucr_001",
                        "length": 15,
                        "eval_start": 5,
                        "eval_end": 20,
                        "intervals": [{"start": 10, "end": 15}],
                    }
                ],
            )
            write_jsonl(
                predictions,
                [
                    {
                        "sample_id": "ucr_001",
                        "series_id": "ucr_001",
                        "parse_valid": True,
                        "intervals": [{"start": 10, "end": 15}],
                    },
                ],
            )
            summary = evaluate_predictions(manifest, predictions, root / "report")
            self.assertEqual(summary["point_f1"], 1.0)
            self.assertEqual(summary["event_f1"], 1.0)
            with (root / "report" / "summary.json").open() as handle:
                self.assertEqual(json.load(handle)["missing_samples"], 0)

    def test_window_prediction_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "series.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "sample_id": "ucr_001",
                        "length": 15,
                        "eval_start": 5,
                        "eval_end": 20,
                        "intervals": [{"start": 10, "end": 15}],
                    }
                ],
            )
            write_jsonl(
                predictions,
                [
                    {
                        "sample_id": "ucr_001_w0000",
                        "series_id": "ucr_001",
                        "parse_valid": True,
                        "intervals": [{"start": 10, "end": 15}],
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "not present in the series manifest"):
                evaluate_predictions(manifest, predictions, root / "report")


if __name__ == "__main__":
    unittest.main()
