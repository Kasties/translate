"""LLM grades, the ridge scorer and quality selection in prepare_registers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from source2synth.common import read_jsonl, write_jsonl
from source2synth.quality_scoring import grade_passages, ridge, spearman
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
            config["model"]["id"] = "org/grader-1"
            prepare_registers(config, path / "data", path / "cache")
            pool = {r["id"]: r for r in read_jsonl(path / "data/nlu/pool.jsonl")}
            post, calls = fake_grader()
            with mock.patch("requests.Session.post", side_effect=post):
                report = grade_passages(config, path / "data", 30)
                first = len(calls)
                again = grade_passages(config, path / "data", 30)
            grades = read_jsonl(path / "data/nlu/grades/org_grader-1/grades.jsonl")
        self.assertEqual(report["graded"] + report["skipped"], 30)
        self.assertLessEqual(report["skipped"], 1)
        self.assertEqual(len(calls), first)  # the second run replayed every journaled call
        self.assertEqual(again["graded"], report["graded"])
        self.assertEqual({g["score"] for g in grades if g["register"] == "fiction"}, {1})
        self.assertTrue(all(g["text"] == pool[g["id"]]["text"] for g in grades))
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertIn("Du bedömer svenska texter", calls[0]["messages"][0]["content"])

    def test_quality_strategy_learns_grades_from_another_build_and_keeps_the_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = fixture_pool(path)
            prepare_registers(config, path / "other", path / "cache")
            # Grades follow the register, which the hash-test embedding can see in the text.
            # Their ids belong to no pool: grades are matched by text.
            write_jsonl(path / "grades.jsonl", [
                {"id": f"elsewhere-{i}", "register": r["register"], "text": r["text"], "reason": "",
                 "score": 1 if r["register"] == "fiction" else 4}
                for i, r in enumerate(read_jsonl(path / "other/nlu/pool.jsonl")) if r["split"] != "eval"])
            config["registers"]["sampling"]["strategy"] = "quality"
            config["registers"]["scoring"].update(grades=[str(path / "grades.jsonl")], floor=2.5, penalty=0.1)
            built = {}
            for order in ("balanced", "best_first"):
                config["registers"]["scoring"]["order"] = order
                manifest = prepare_registers(config, path / order, path / "cache")
                scores = {s["id"]: s["quality_score"] for s in read_jsonl(path / order / "nlu/scores.jsonl")}
                built[order] = manifest, scores, read_jsonl(path / order / "nlu/passages.jsonl")
        for order, (manifest, scores, passages) in built.items():
            agreement = manifest["scoring"]["agreement"]
            self.assertGreater(agreement["spearman"], 0.5)  # two grade levels, many ties
            self.assertLess(agreement["mean_absolute_error"], agreement["baseline_mean_absolute_error"])
            self.assertEqual(manifest["scoring"]["floor"], 2.5)
            self.assertEqual(len(scores), manifest["pool_passages"])
            self.assertTrue(passages)
            self.assertEqual(len(passages), manifest["passages"])
            self.assertTrue(all(scores[p["id"]] >= 2.5 for p in passages))
            self.assertNotIn("fiction", {p["register"] for p in passages})
            for name in ("balanced", "random", "random_above_floor", f"quality_{order}"):
                self.assertIn("mean_quality_score", manifest["comparison"][f"slice0/{name}"])

    def test_quality_strategy_needs_grades_with_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = fixture_pool(path)
            config["registers"]["sampling"]["strategy"] = "quality"
            with self.assertRaisesRegex(ValueError, "scoring.grades"):
                prepare_registers(config, path / "data", path / "cache")
            write_jsonl(path / "old.jsonl", [{"id": "p", "register": "debate", "reason": "", "score": 3}])
            config["registers"]["scoring"]["grades"] = [str(path / "old.jsonl")]
            with self.assertRaisesRegex(ValueError, "lack text"):
                prepare_registers(config, path / "data", path / "cache")


if __name__ == "__main__":
    unittest.main()
