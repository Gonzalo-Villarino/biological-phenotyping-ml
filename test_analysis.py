"""Regression tests for the synthetic biological phenotyping workflow."""

import unittest

import numpy as np

from analysis import FEATURES, calibration_table, make_synthetic_data


class TestSyntheticWorkflow(unittest.TestCase):
    def test_generation_is_deterministic(self):
        first = make_synthetic_data(n_batches=4, observations_per_batch=5)
        second = make_synthetic_data(n_batches=4, observations_per_batch=5)
        self.assertTrue(first.equals(second))

    def test_expected_shape_and_batch_structure(self):
        data = make_synthetic_data(n_batches=6, observations_per_batch=7)
        self.assertEqual(len(data), 42)
        self.assertEqual(data["batch_id"].nunique(), 6)
        self.assertEqual(list(data.columns[2:8]), FEATURES)
        self.assertTrue(set(data["phenotype_outcome"]).issubset({0, 1}))

    def test_calibration_counts_every_observation(self):
        outcomes = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.3, 0.7, 0.9])
        table = calibration_table(outcomes, probabilities)
        self.assertEqual(int(table["n"].sum()), 4)


if __name__ == "__main__":
    unittest.main()
