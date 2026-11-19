"""Portable checks for native editing, feedback isolation and model persistence."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from weightloop import NativeMLPEditor, choose_update, functional_utility, recipes


ROOT = Path(__file__).resolve().parents[1]


def fixture(n=30, seed=31):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 128))
    # Include exact and near-zero activations to exercise native ReLU changes.
    z[:, ::17] = 0.
    z[:, 3::23] = 1e-12
    w = rng.normal(0., .05, 128)
    hidden = rng.normal(0., .02, 128)
    return z, w, hidden, .07


class UtilityTests(unittest.TestCase):
    def test_oracle_and_reversed_ranking(self):
        y = np.array([0., 1., 2., 3.])
        self.assertEqual(functional_utility(y, y, 2), 1.)
        self.assertLess(functional_utility(-y, y, 2), 0.)

    def test_all_tied_scores_are_random_expectation(self):
        y = np.array([-2., 1., 4., 7., 11.])
        self.assertAlmostEqual(functional_utility(np.ones(5), y, 2), 0., places=14)

    def test_budget_boundary_tie_uses_whole_group(self):
        # Rank one is fixed. Only half the second/third tied group is included.
        scores = np.array([5., 3., 3., 0.])
        y = np.array([6., 0., 4., -2.])
        d = 1./np.log2(np.array([2., 3.])); d /= d.sum()
        expected_value = d[0]*6. + d[1]*(0.+4.)/2.
        oracle = d @ np.sort(y)[::-1][:2]
        expected = (expected_value-y.mean())/(oracle-y.mean())
        self.assertAlmostEqual(functional_utility(scores, y, 2), expected, places=14)
        order = np.array([0, 2, 1, 3])
        self.assertEqual(functional_utility(scores[order], y[order], 2), functional_utility(scores, y, 2))

    def test_constant_outcomes_are_undefined(self):
        self.assertIsNone(functional_utility(np.arange(4), np.ones(4), 2))

    def test_invalid_inputs_are_rejected(self):
        for budget in [0, -1, 5, True, 1.5]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                functional_utility(np.arange(4), np.arange(4), budget)
        for scores, y in [([1., np.nan], [1., 2.]), ([1., 2.], [1., np.inf]), ([1.], [1., 2.]), ([[1., 2.]], [1., 2.])]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                functional_utility(scores, y, 1)


class NativeEditorTests(unittest.TestCase):
    def test_constructor_rejects_invalid_native_shapes_and_values(self):
        z, w, hidden, bias = fixture()
        for zz, ww, hh in [(z[:, :127], w, hidden), (z, w[:127], hidden),
                           (z, w, hidden[:1]), (z * np.nan, w, hidden)]:
            with self.subTest(shape=zz.shape), self.assertRaises(ValueError):
                NativeMLPEditor(zz, ww, bias, hh, 17)

    def test_prediction_rejects_broadcastable_incomplete_vector(self):
        z, w, hidden, bias = fixture()
        editor = NativeMLPEditor(z, w, bias, hidden, 17)
        for vector in [np.r_[w, hidden[:1]], editor.base[:-1],
                       np.r_[editor.base, 0.], editor.base * np.nan]:
            with self.subTest(size=len(vector)), self.assertRaises(ValueError):
                editor.predict(vector)

    def test_proposal_rejects_broadcastable_targets_and_empty_feedback(self):
        z, w, hidden, bias = fixture()
        editor = NativeMLPEditor(z, w, bias, hidden, 17)
        for ix, y in [(np.arange(12), np.array([1.])),
                      (np.arange(12), np.full(12, np.nan)),
                      (np.array([], dtype=int), np.array([]))]:
            with self.subTest(rows=len(ix)), self.assertRaises(ValueError):
                editor.propose(ix, y, recipes()[0])

    def test_proposal_rejects_invalid_recipe(self):
        z, w, hidden, bias = fixture()
        editor = NativeMLPEditor(z, w, bias, hidden, 17)
        for changes in [{'scope': 'unknown'}, {'lambda_rel': 0.},
                        {'lambda_rel': np.nan}, {'radius': -1.}, {'radius': np.inf}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                editor.propose(np.arange(12), np.zeros(12), {**recipes()[0], **changes})

    def test_native_formula_and_relu_threshold_changes(self):
        z, w, hidden, bias = fixture()
        editor = NativeMLPEditor(z, w, bias, hidden, 17)
        vector = editor.base.copy()
        vector[:128] *= .9
        vector[128:] += .04
        expected = np.maximum(z + .04, 0.) @ (.9*w) + bias
        np.testing.assert_allclose(editor.predict(vector), expected, rtol=0., atol=1e-14)
        ix = np.array([2, 0, 9])
        np.testing.assert_allclose(editor.predict(vector, ix), editor.predict(vector)[ix], rtol=0., atol=1e-14)

    def test_proposals_do_not_use_withheld_feedback_rows(self):
        z, w, hidden, bias = fixture(n=18)
        rows = np.arange(12)
        y = np.maximum(z[rows], 0.) @ (.85*w) + bias
        altered = z.copy(); altered[12:] = 1e6
        first = NativeMLPEditor(z, w, bias, hidden, 17)
        second = NativeMLPEditor(altered, w, bias, hidden, 17)
        before = first.base.copy()
        for recipe in recipes():
            with self.subTest(recipe=recipe['id']):
                a = first.propose(rows, y, recipe)
                b = second.propose(rows, y, recipe)
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(a, first.propose(rows, y, recipe))
                np.testing.assert_array_equal(first.base, before)

    def test_packaged_code_matches_recorded_generated_program(self):
        source = ROOT/'programs/hepg2/fixed_rules.py'
        original = ast.parse(source.read_text())
        # This is the versioned, local program shipped with this repository.
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(original)))
        namespace = {'np':np}
        exec(compile(original, str(source), 'exec'), namespace)
        self.assertEqual(recipes(), namespace['recipes']())
        z, w, hidden, bias = fixture(n=18, seed=73)
        rows = np.arange(12)
        y = np.maximum(z[rows]+.02, 0.) @ (.9*w) + bias
        packaged = NativeMLPEditor(z, w, bias, hidden, 53)
        recorded = namespace['NativeMLPEditor'](z, w, bias, hidden, 53)
        for recipe in recipes():
            with self.subTest(recipe=recipe['id']):
                np.testing.assert_array_equal(packaged.propose(rows, y, recipe), recorded.propose(rows, y, recipe))

    def test_native_model_save_reload_and_query_order(self):
        z, w, hidden, bias = fixture()
        editor = NativeMLPEditor(z, w, bias, hidden, 17)
        y = np.maximum(z[:18], 0.) @ (.8*w) + bias
        vector = editor.propose(np.arange(18), y, recipes()[0])
        output_root = ROOT/'outputs'; output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root, prefix='.native-test-') as directory:
            path = Path(directory)/'updated.npz'
            np.savez(path, w=vector[:128], hidden_bias=vector[128:], bias=np.asarray(bias))
            with np.load(path, allow_pickle=False) as checkpoint:
                native = np.maximum(z+(checkpoint['hidden_bias']-hidden), 0.) @ checkpoint['w'] + float(checkpoint['bias'])
            np.testing.assert_array_equal(editor.predict(vector), native)
            np.testing.assert_allclose(native[::-1], editor.predict(vector, np.arange(len(z)-1,-1,-1)), rtol=0., atol=1e-14)
            for row in [0, 13, 29]:
                np.testing.assert_allclose(native[row:row+1], editor.predict(vector, np.array([row])), rtol=0., atol=1e-14)


class FeedbackSelectionTests(unittest.TestCase):
    def test_perfect_original_is_kept(self):
        z, w, hidden, bias = fixture()
        y = np.maximum(z, 0.) @ w+bias
        vector, report = choose_update(z, y, w, bias, hidden, np.arange(len(y))%3, budget=3)
        self.assertEqual(report['status'], 'base-kept')
        self.assertEqual(report['recipe'], 'unchanged')
        np.testing.assert_array_equal(vector, np.r_[w,hidden])

    def test_undefined_validation_keeps_original(self):
        z, w, hidden, bias = fixture()
        vector, report = choose_update(z, np.ones(len(z)), w, bias, hidden, np.arange(len(z))%3, budget=3)
        self.assertEqual(report['status'], 'base-kept')
        self.assertEqual(report['reason'], 'undefined-validation-utility')
        np.testing.assert_array_equal(vector, np.r_[w,hidden])

    def test_invalid_feedback_folds_and_shapes_are_rejected(self):
        z, w, hidden, bias = fixture()
        y = np.maximum(z, 0.) @ w+bias
        cases = [dict(folds=np.zeros(len(y),dtype=int)), dict(folds=np.arange(len(y))), dict(folds=np.arange(len(y))%3,budget=10)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                choose_update(z, y, w, bias, hidden, **case)
        with self.assertRaises(ValueError):
            choose_update(z[:, :127], y, w, bias, hidden, np.arange(len(y))%3)


if __name__ == '__main__':
    unittest.main()
