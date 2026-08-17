import json
import tempfile
import unittest
from pathlib import Path

from tsad_v2.evaluation import evaluate_predictions
from tsad_v2.io import write_jsonl


class EvaluationTests(unittest.TestCase):
    def test_window_predictions_are_merged_by_series(self):
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
                        "length": 20,
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
                        "sample_id": "w0",
                        "series_id": "ucr_001",
                        "parse_valid": True,
                        "intervals": [{"start": 10, "end": 13}],
                    },
                    {
                        "sample_id": "w1",
                        "series_id": "ucr_001",
                        "parse_valid": True,
                        "intervals": [{"start": 12, "end": 15}],
                    },
                ],
            )
            summary = evaluate_predictions(manifest, predictions, root / "report")
            self.assertEqual(summary["point_f1"], 1.0)
            self.assertEqual(summary["event_f1"], 1.0)
            with (root / "report" / "summary.json").open() as handle:
                self.assertEqual(json.load(handle)["missing_samples"], 0)


if __name__ == "__main__":
    unittest.main()

