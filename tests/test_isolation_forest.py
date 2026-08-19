import unittest

import numpy as np

from tsad_v2.baselines.isolation_forest import extract_features, mask_to_intervals


class IsolationForestTests(unittest.TestCase):
    def test_features_keep_one_row_per_time_point(self):
        features = extract_features(np.arange(10, dtype=np.float64), [3, 5])
        self.assertEqual(features.shape, (10, 8))
        self.assertTrue(np.isfinite(features).all())

    def test_mask_uses_half_open_global_intervals(self):
        mask = np.array([False, True, True, False, True])
        self.assertEqual(mask_to_intervals(mask, 20), [(21, 23), (24, 25)])


if __name__ == "__main__":
    unittest.main()
