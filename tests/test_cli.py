"""Input contracts and portable execution, without rerunning the full study."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run


class InputTests(unittest.TestCase):
    def test_npz_loads_valid_timeseries_without_changing_values(self):
        x = np.random.default_rng(10).normal(size=(97, 7))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            np.savez(path, timeseries=x)
            np.testing.assert_array_equal(run.load_timeseries(path), x)

    def test_invalid_shapes_lengths_types_and_nonfinite_values_are_rejected(self):
        examples = [
            np.zeros(97), np.zeros((97, 6)), np.zeros((97, 7, 1)),
            np.zeros((63, 7)), np.zeros((97, 7), dtype=complex),
            np.zeros((97, 7), dtype=bool), np.zeros((97, 7), dtype=object),
            np.full((97, 7), "text"), np.full((97, 7), np.nan),
            np.full((97, 7), np.inf),
        ]
        for i, x in enumerate(examples):
            with self.subTest(example=i), self.assertRaises(ValueError):
                run.validate_timeseries(x)

    def test_numeric_inputs_use_float64_without_unsigned_difference_underflow(self):
        x = np.tile(np.array([0, 255, 1, 254], dtype=np.uint8), 25)
        field = np.repeat(x[:, None], 7, axis=1)
        normalized = run.validate_timeseries(field)
        self.assertEqual(normalized.dtype, np.dtype("float64"))
        np.testing.assert_array_equal(normalized, field)
        self.assertEqual(np.diff(normalized[:, 0])[1], -254.0)
        for dtype in (np.int16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                self.assertEqual(run.validate_timeseries(field.astype(dtype)).dtype, np.dtype("float64"))

    def test_missing_array_and_pickle_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            np.savez(path, wrong_key=np.ones((97, 7)))
            with self.assertRaisesRegex(ValueError, "timeseries"):
                run.load_timeseries(path)
            np.savez(path, timeseries=np.ones((97, 7), dtype=object))
            with self.assertRaises(ValueError):
                run.load_timeseries(path)

    def test_json_serialization_handles_unsupported_numeric_diagnostics(self):
        payload = run.plain({"x": np.array([1.0, np.nan, np.inf]), "rank": np.int64(3)})
        encoded = json.dumps(payload, allow_nan=False)
        self.assertEqual(json.loads(encoded), {"x": [1.0, None, None], "rank": 3})


class CommandLineTests(unittest.TestCase):
    def call(self, directory, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / "run.py"), *map(str, arguments)],
            cwd=directory, text=True, capture_output=True, timeout=30,
        )

    def test_summary_runs_from_another_directory_and_writes_requested_output(self):
        data_files = sorted((ROOT / "data").glob("*.json"))
        self.assertEqual(
            {path.name for path in data_files},
            {"iaaft_aggregate.json", "iaaft_calibration_aggregate.json", "common_phase_aggregate.json"},
            "The three saved aggregate JSON files must be bundled",
        )
        expected = {path.name: json.loads(path.read_text()) for path in data_files}
        with tempfile.TemporaryDirectory() as directory:
            output = Path("nested") / "summary.json"
            result = self.call(directory, "--mode", "summary", "--output", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((Path(directory) / output).read_text()), expected)
            self.assertEqual(json.loads(result.stdout)["mode"], "summary")

    def test_summary_stdout_is_json(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.call(directory, "--mode", "summary")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), run.summary())

    def test_cli_rejects_invalid_inputs_before_running_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            malformed = Path(directory) / "short.npz"
            np.savez(malformed, timeseries=np.zeros((12, 7)))
            cases = [
                (["--mode", "unknown"], "invalid choice"),
                (["--mode", "summary", "--input", malformed], "only available in demo mode"),
                (["--input", malformed], "at least 64"),
                (["--input", Path(directory) / "absent.npz"], "No such file"),
            ]
            for arguments, message in cases:
                with self.subTest(arguments=arguments):
                    result = self.call(directory, *arguments)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_unwritable_output_is_reported_as_a_command_line_error(self):
        with tempfile.TemporaryDirectory() as directory:
            # A directory cannot be opened as a JSON file on any supported OS.
            result = self.call(directory, "--mode", "summary", "--output", directory)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
