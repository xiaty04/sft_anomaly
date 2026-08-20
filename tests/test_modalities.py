import tempfile
import unittest
from pathlib import Path

import numpy as np

from tsad_v2.modalities import extrema_summary, serialize_indexed_series, text_prompt


class ModalityTests(unittest.TestCase):
    def test_indexed_serialization_uses_global_coordinates(self):
        text = serialize_indexed_series(np.array([1.0, 2.5]), 10, precision=3)
        self.assertEqual(text, "[(10,1),(11,2.5)]")

    def test_text_prompt_loads_the_declared_input_range(self):
        with tempfile.TemporaryDirectory() as directory:
            series_path = Path(directory) / "series.npy"
            np.save(series_path, np.arange(8, dtype=np.float32))
            prompt = text_prompt(
                {
                    "series_path": str(series_path),
                    "length": 3,
                    "input_start": 2,
                    "input_end": 5,
                }
            )
            self.assertIn("(2,2)", prompt)
            self.assertIn("(4,4)", prompt)
            self.assertNotIn("(5,5)", prompt)

    def test_extrema_summary_keeps_global_endpoints_and_spike(self):
        values = np.zeros(20, dtype=np.float64)
        values[7] = 100.0
        indices, selected = extrema_summary(values, start_index=100, max_points=6)
        self.assertLessEqual(len(indices), 6)
        self.assertEqual(indices[0], 100)
        self.assertEqual(indices[-1], 119)
        self.assertIn(107, indices)
        self.assertEqual(selected[list(indices).index(107)], 100.0)


if __name__ == "__main__":
    unittest.main()
