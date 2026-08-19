import unittest

from tsad_v2.metrics import aggregate, evaluate_sample


class MetricsTests(unittest.TestCase):
    def test_partial_overlap(self):
        metrics, counts = evaluate_sample([(2, 6)], [(4, 8)], 0, 10)
        self.assertAlmostEqual(metrics["point_precision"], 0.5)
        self.assertAlmostEqual(metrics["point_recall"], 0.5)
        self.assertAlmostEqual(metrics["point_f1"], 0.5)
        self.assertAlmostEqual(metrics["mean_matched_iou"], 2 / 6)
        self.assertEqual(counts.point_tp, 2)

    def test_empty_prediction_and_target_are_correct(self):
        metrics, counts = evaluate_sample([], [], 0, 10)
        self.assertEqual(metrics["point_f1"], 1.0)
        self.assertEqual(metrics["event_f1"], 1.0)
        summary = aggregate([(metrics, counts)])
        self.assertEqual(summary["point_f1"], 1.0)
        self.assertEqual(summary["normal_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
