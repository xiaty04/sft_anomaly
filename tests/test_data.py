import tempfile
import unittest
from pathlib import Path

import numpy as np

from tsad_v2.data.synthetic import generate_sample
from tsad_v2.data.ucr import parse_ucr_filename, prepare_ucr
from tsad_v2.io import read_jsonl


class DataTests(unittest.TestCase):
    def test_ucr_filename_parser(self):
        metadata = parse_ucr_filename(
            Path("001_UCR_Anomaly_DISTORTED_demo_name_5_10_12.txt")
        )
        self.assertEqual(metadata["archive_id"], "001")
        self.assertEqual(metadata["name"], "DISTORTED_demo_name")
        self.assertEqual(metadata["anomaly_end_raw"], 12)

    def test_prepare_ucr_creates_series_and_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "001_UCR_Anomaly_demo_5_10_12.txt"
            np.savetxt(source, np.linspace(0, 1, 20))
            series_path, windows_path = prepare_ucr(
                {
                    "raw_dir": str(raw),
                    "output_dir": str(root / "processed"),
                    "filename_index_base": 0,
                    "anomaly_end_inclusive": True,
                    "train_end_is_count": True,
                    "window_size": 10,
                    "stride": 5,
                },
                {"width": 300, "height": 120, "dpi": 60},
            )
            series = read_jsonl(series_path)
            windows = read_jsonl(windows_path)
            self.assertEqual(series[0]["intervals"], [{"start": 10, "end": 13}])
            self.assertEqual(len(windows), 2)
            self.assertTrue(Path(windows[0]["image_path"]).exists())

    def test_synthetic_generation_is_reproducible(self):
        args = dict(
            length=256,
            anomaly_types=["point", "range", "frequency", "trend"],
            normal_probability=0.0,
            max_intervals=2,
            seed=7,
            forced_type="range",
        )
        first = generate_sample(**args)
        second = generate_sample(**args)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])


if __name__ == "__main__":
    unittest.main()

