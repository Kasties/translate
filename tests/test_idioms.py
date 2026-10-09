import math
import unittest

from source2synth.idioms import mcnemar_exact, mean_nll, summarise, wilson


def row(correct, challenge=False):
    return {"correct": correct, "challenge": challenge}


class MeanNllTest(unittest.TestCase):
    def test_scores_every_token_after_bos(self):
        logprobs = [
            None,
            {"7": {"logprob": -1.0}, "9": {"logprob": -0.1}},
            {"8": {"logprob": -3.0}},
        ]
        self.assertAlmostEqual(mean_nll([2, 7, 8], logprobs), 2.0)

    def test_rejects_misaligned_response(self):
        with self.assertRaises(ValueError):
            mean_nll([2, 7], [None])
        with self.assertRaises(ValueError):
            mean_nll([2, 7], [None, {"9": {"logprob": -1.0}}])


class StatisticsTest(unittest.TestCase):
    def test_wilson_matches_reference_formula(self):
        low, high = wilson(90, 100)
        self.assertAlmostEqual(low, 0.8256, places=4)
        self.assertAlmostEqual(high, 0.9448, places=4)

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 2 / 64)

    def test_summary_splits_challenge_and_pairs_models(self):
        summary = summarise(
            {
                "a": [row(True, True), row(True), row(False)],
                "b": [row(False, True), row(True), row(True)],
            }
        )
        self.assertEqual(summary["models"]["a"]["all"]["correct"], 2)
        self.assertEqual(summary["models"]["a"]["challenge"]["total"], 1)
        self.assertEqual(summary["paired"]["all"], {"only_a": 1, "only_b": 1, "mcnemar_exact_p": 1.0})
        self.assertTrue(math.isclose(summary["models"]["b"]["challenge"]["accuracy"], 0.0))


if __name__ == "__main__":
    unittest.main()
