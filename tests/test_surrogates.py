"""Scientific invariants for the paper's small, deterministic routines."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import surrogates as m


class SurrogateConstraintTests(unittest.TestCase):
    def test_iaaft_preserves_exact_marginals_and_25_step_result(self):
        for n in (97, 128):
            # Repeated values also exercise stable rank assignment.
            x = np.round(np.random.default_rng(n).normal(size=n), 1)
            with self.subTest(n=n):
                y, error = m.iaaft(x, m.rng_for(310, n))
                pair = m.iaaft_pair(x, m.rng_for(310, n))
                self.assertEqual(set(pair), {25, 250})
                np.testing.assert_array_equal(y, pair[25][0])
                self.assertEqual(error, pair[25][1])
                for draw, reported in pair.values():
                    np.testing.assert_array_equal(np.sort(draw), np.sort(x))
                    amp = np.abs(np.fft.rfft(x))
                    actual = np.linalg.norm(np.abs(np.fft.rfft(draw)) - amp)
                    actual /= np.linalg.norm(amp)
                    self.assertAlmostEqual(reported, actual, places=14)

    def test_common_phase_preserves_full_complex_cross_spectrum_and_fc(self):
        for n in (97, 128):
            rng = np.random.default_rng(n)
            x = rng.normal(size=(n, 7)) @ rng.normal(size=(7, 7))
            x += np.arange(7)[None, :]
            with self.subTest(n=n):
                y, cross_error, fc_error, drift = m.phase_draw(x, m.rng_for(410, n))
                # Check every signed frequency and channel pair independently.
                f = np.fft.fft(x, axis=0)
                g = np.fft.fft(y, axis=0)
                original = f[:, :, None] * f[:, None, :].conj()
                shifted = g[:, :, None] * g[:, None, :].conj()
                relative_error = np.linalg.norm(shifted - original) / np.linalg.norm(original)
                self.assertLess(relative_error, 1e-12)
                np.testing.assert_allclose(
                    np.corrcoef(y, rowvar=False), np.corrcoef(x, rowvar=False),
                    rtol=0, atol=1e-12,
                )
                np.testing.assert_allclose(y.mean(axis=0), x.mean(axis=0), rtol=0, atol=1e-12)
                self.assertLess(cross_error, 1e-12)
                self.assertLess(fc_error, 1e-12)
                self.assertEqual(np.shape(drift), (7,))
                self.assertTrue(np.isfinite(drift).all())
                self.assertFalse(np.allclose(y, x))

    def test_latent_paths_share_samples_at_common_physical_times(self):
        _, _, paths = m.latent_paths(2026102201)
        a, b = paths[(197, 1500)], paths[(97, 3000)]
        np.testing.assert_array_equal(a[::2][:97], b)


class DescriptorTests(unittest.TestCase):
    def test_fast_entropy_counts_agree_with_brute_force(self):
        rng = np.random.default_rng(20261022)
        examples = [
            rng.normal(size=197),
            np.round(rng.normal(size=97), 1),
            np.tile([0.0, 1.0, 0.0, -1.0], 32),
        ]
        for i, x in enumerate(examples):
            with self.subTest(example=i):
                slow, fast = m.sampen(x), m.sampen_fast(x)
                self.assertEqual(slow["status"], "defined")
                self.assertEqual(fast["status"], "defined")
                self.assertEqual((slow["A"], slow["B"]), (fast["A"], fast["B"]))
                self.assertAlmostEqual(slow["value"], fast["value"], places=14)
                self.assertEqual(slow["tolerance"], fast["tolerance"])

    def test_entropy_uses_common_starts_and_excludes_self_matches(self):
        x = np.tile([0.0, 1.0, 0.0, -1.0], 16)
        tolerance = 0.2 * np.std(x)
        a = b = 0
        for i in range(len(x) - 2):
            for j in range(i + 1, len(x) - 2):
                if max(abs(x[i:i + 2] - x[j:j + 2])) <= tolerance:
                    b += 1
                    if abs(x[i + 2] - x[j + 2]) <= tolerance:
                        a += 1
        result = m.sampen_fast(x)
        self.assertEqual((result["A"], result["B"]), (a, b))
        self.assertAlmostEqual(result["value"], -np.log(a / b), places=14)

    def test_short_constant_and_nonfinite_signals_are_unsupported(self):
        examples = [
            np.arange(63, dtype=float), np.ones(97),
            np.r_[np.nan, np.arange(96)], np.r_[np.inf, np.arange(96)],
        ]
        for function in (m.sampen, m.sampen_fast, m.mfdfa, m.higuchi):
            for i, x in enumerate(examples):
                with self.subTest(function=function.__name__, example=i), np.errstate(invalid="ignore"):
                    self.assertEqual(function(x)["status"], "unsupported")

    def test_vector_dimensions_and_unsupported_propagation(self):
        for n in (97, 197):
            x = np.random.default_rng(n).normal(size=n)
            with self.subTest(n=n):
                result = m.vectors(m.measurement(x))
                self.assertEqual(result["SampEn"].shape, (1,))
                self.assertEqual(result["MFDFA_shape"].shape, (12,))
                self.assertEqual(result["Higuchi_shape"].shape, (11,))
                self.assertTrue(all(np.isfinite(v).all() for v in result.values()))
        self.assertTrue(all(v is None for v in m.vectors(m.measurement(np.ones(97))).values()))
        short = m.measurement(np.random.default_rng(64).normal(size=64))
        self.assertIsNone(m.mfshape(short["MFDFA"]))
        self.assertIsNone(m.hshape(short["Higuchi"]))

    def test_shape_descriptors_are_invariant_to_positive_amplitude_units(self):
        x = np.random.default_rng(144).normal(size=197)
        before, after = m.vectors(m.measurement(x)), m.vectors(m.measurement(8 * x))
        for name in before:
            with self.subTest(descriptor=name):
                np.testing.assert_allclose(before[name], after[name], rtol=0, atol=1e-12)

    def test_rank_ties_are_conservative_and_nonfinite_values_are_unsupported(self):
        self.assertEqual(m.rank_stat(np.ones((21, 12))), (21, 0.0))
        self.assertEqual(m.rank_stat(np.array([[0.0], [np.nan]])), (None, None))


if __name__ == "__main__":
    unittest.main()
