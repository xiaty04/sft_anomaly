import tempfile
import unittest
from pathlib import Path

import numpy as np

from tsad_v2.modalities import serialize_indexed_series, text_prompt


class ModalityTests(unittest.TestCase):
    def test_indexed_serialization_uses_global_coordinates(self):
        text = serialize_indexed_series(np.array([1.0, 2.5]), 10, precision=3)
        self.assertEqual(text, "[(10,1),(11,2.5)]")

    def test_text_prompt_loads_the_same_window_as_vision(self):
        with tempfile.TemporaryDirectory() as directory:
            series_path = Path(directory) / "series.npy"
            np.save(series_path, np.arange(8, dtype=np.float32))
            prompt = text_prompt(
                {
                    "series_path": str(series_path),
                    "length": 3,
                    "window_start": 2,
                    "window_end": 5,
                }
            )
            self.assertIn("(2,2)", prompt)
            self.assertIn("(4,4)", prompt)
            self.assertNotIn("(5,5)", prompt)


if __name__ == "__main__":
    unittest.main()
