"""LLM grades, the ridge scorer and best-first selection."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from source2synth.common import read_jsonl, write_jsonl
from source2synth.quality_scoring import grade_passages, ridge, score_passages, select_scored, spearman
from source2synth.register_sampling import select
from source2synth.register_sources import prepare_registers
from test_register_data import fixture_pool, source


def completion(content: str) -> mock.Mock:
    response = mock.Mock()
    response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                  "usage": {"completion_tokens": 20}}
    return response


def fake_grader():
    """Grade fiction 1, everything else 4; one passage returns broken JSON."""
    calls = []

    def post(url, json, **kwargs):
        calls.append(json)
        text = json["messages"][1]["content"]
        if "teater under år 1903" in text:
            return completion("inte json")
        score = 1 if "fiction" in text else 4
        return completion(f'{{"reason": "Kort motivering.", "score": {score}}}')
    return post, calls


class QualityScoringTests(unittest.TestCase):
    def test_quality_selection_keeps_balance_and_takes_the_best_first(self):
        # Two passages per document, documents alternating fiction and debate, one cluster.
        rows = [source(i, register="debate" if (i // 2) % 2 else "fiction", quality_score=float(i),
                       document_id=f"doc{i // 2}", cluster=0) for i in range(12)]
        picked = select(rows, "quality", 4, 42, {})
        self.assertEqual(sorted(r["register"] for r in picked), ["debate", "debate", "fiction", "fiction"])
        # Each document is its better passage (odd scores); the best two per register win.
        self.assertEqual(sorted(r["quality_score"] for r in picked), [5.0, 7.0, 9.0, 11.0])

    def test_ridge_recovers_a_linear_score(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(200, 5))
        targets = features @ np.array([1.0, -2.0, 0.5, 0.0, 0.0]) + 3.0
        weights = ridge(features, targets, penalty=1e-6)
        np.testing.assert_allclose(weights, [1.0, -2.0, 0.5, 0.0, 0.0, 3.0], atol=1e-4)
        self.assertAlmostEqual(spearman(np.arange(10.0), np.arange(10.0) ** 3), 1.0)

    def test_grades_are_journaled_resumable_and_skip_unusable_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = fixture_pool(path)
            prepare_registers(config, path / "data")
            post, calls = fake_grader()
            with mock.patch("requests.Session.post", side_effect=post):
                report = grade_passages(config, path / "data", 30)
                first = len(calls)
                again = grade_passages(config, path / "data", 30)
            grades = read_jsonl(path / "data/nlu/grades/grades.jsonl")
        self.assertEqual(report["graded"] + report["skipped"], 30)
        self.assertLessEqual(report["skipped"], 1)
        self.assertEqual(len(calls), first)  # the second run replayed every journaled call
        self.assertEqual(again["graded"], report["graded"])
        self.assertEqual({g["score"] for g in grades if g["register"] == "fiction"}, {1})
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertIn("Du bedömer svenska texter", calls[0]["messages"][0]["content"])

    def test_scores_rank_held_out_passages_and_selection_respects_the_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = fixture_pool(path)
            prepare_registers(config, path / "data")
            nlu = path / "data/nlu"
            pool = read_jsonl(nlu / "pool.jsonl")
            # Grades follow the register, which the hash-test embedding can see in the text.
            write_jsonl(nlu / "grades/grades.jsonl", [
                {"id": r["id"], "register": r["register"], "reason": "", "score": 1 if r["register"] == "fiction" else 4}
                for r in pool if r["split"] != "eval"])
            report = score_passages(config, path / "data", penalty=0.1)
            scores = {s["id"]: s["quality_score"] for s in read_jsonl(nlu / "scores.jsonl")}
            selection = select_scored(config, path / "data", floor=2.5)
            passages = read_jsonl(nlu / "passages.jsonl")
        self.assertGreater(report["agreement"]["spearman"], 0.5)  # two grade levels, many ties
        self.assertLess(report["agreement"]["mean_absolute_error"],
                        report["agreement"]["baseline_mean_absolute_error"])
        self.assertEqual(len(scores), len(pool))
        self.assertTrue(passages)
        self.assertTrue(all(scores[p["id"]] >= 2.5 for p in passages))
        self.assertNotIn("fiction", {p["register"] for p in passages})
        self.assertIn("slice0/random_above_floor", selection["comparison"])


if __name__ == "__main__":
    unittest.main()
