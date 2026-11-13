"""Synthetic integration checks for the optional scientific fitting interface."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCIPY = importlib.util.find_spec('scipy') is not None


@unittest.skipUnless(SCIPY, 'Install weightloop[scientific] for static fitting tests')
class StaticFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'programs/static_reference.py'
        spec = importlib.util.spec_from_file_location('static_reference', path)
        cls.program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.program)

    def run_fitting_contract(self, optimizer):
        p = self.program
        rng = np.random.default_rng(11)
        jac = rng.normal(size=(12, 1, 3))
        w0 = np.array([.2, -.1, .3])
        original = w0.copy()
        baseline = (jac @ w0).reshape(12, 1)
        target = baseline + .5 * jac[:, :, 1]
        result, report = p.fit(baseline, jac, w0, np.ones(3), target,
            jac[:3], np.array([1]), p.specification(), output_scale=1.,
            optimizer=optimizer)
        np.testing.assert_array_equal(w0, original)
        np.testing.assert_array_equal(result[[0, 2]], original[[0, 2]])
        self.assertEqual(result[1], float(np.float32(result[1])))
        self.assertLess(report['objective_after_native_rounding'], report['objective_before'])
        self.assertTrue(np.isfinite(result).all())

    def test_analytic_fit_changes_only_authorized_native_coordinate(self):
        self.run_fitting_contract('lbfgs')

    def test_adam_fit_changes_only_authorized_native_coordinate(self):
        self.run_fitting_contract('adam')
