"""Offline behavioral tests. Fictional QA here is a test fixture, never research data."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from jinja2 import StrictUndefined, Template

from source2synth.common import exact_answer, read_config, read_jsonl
from source2synth.curation import curation_reasons, flatten_generation, generation_reasons
from source2synth.curation import quote_in_source
from source2synth.korp import assert_disjoint, document_split, entity_cqp, passage_from_hit
from source2synth.pipeline import curation_builder, parse_json_response, MergedQuestion
from source2synth.__main__ import curate, evaluate, generate, parser, refilter, training_provenance


def sample() -> dict:
    return {
        "seed_id": "fixture", "bridge_entity": "Anna Exempel", "split": "slice1",
        "d1_document_id": "doc1", "d2_document_id": "doc2",
        "d1_title": "Testverket", "d2_title": "Testbiografi",
        "d1_text": "Testverket skrevs av Anna Exempel .",
        "d2_text": "Anna Exempel föddes i Testköping .",
        "hop1": {"possible": True, "question": "Vem skrev Testverket?",
                 "evidence": "Testverket skrevs av Anna Exempel ."},
        "hop2": {"possible": True, "question": "Var föddes Anna Exempel?", "answer": "Testköping",
                 "evidence": "Anna Exempel föddes i Testköping ."},
        "merged": {"possible": True, "question": "Var föddes författaren till Testverket?"},
    }


def curated_sample() -> dict:
    row = flatten_generation(sample())
    return {**row, "imputed_hop1": row["hop1"], "recomposed": row["merged"],
            "solve_1": {"answerable": True, "answer": "Felstad"},
            "solve_2": {"answerable": True, "answer": "Testköping"},
            "solve_3": {"answerable": False, "answer": ""},
            "quality": {"grounded": True, "requires_both_documents": True,
                        "subquestions_consistent": True, "swedish": True, "reason": "Test"}}


class CurationTests(unittest.TestCase):
    def test_plain_and_fenced_json_are_both_valid(self):
        expected = {"possible": True, "question": "Vilken?"}
        raw = json.dumps(expected)
        self.assertEqual(parse_json_response(raw, MergedQuestion), expected)
        self.assertEqual(parse_json_response("```json\n" + raw + "\n```", MergedQuestion), expected)
        with self.assertRaises(ValueError):
            parse_json_response("Not JSON", MergedQuestion)
        with self.assertRaises(ValueError):
            parse_json_response('{"possible": true}', MergedQuestion)

    def test_quote_spacing_is_allowed_but_reordering_is_not(self):
        self.assertTrue(quote_in_source("Anna skrevs in.", "Anna skrevs in ."))
        self.assertFalse(quote_in_source("B. A.", "A . B ."))
        self.assertFalse(quote_in_source("Anna skrev in.", "Anna skrevs in ."))

    def test_valid_generation(self):
        self.assertEqual(generation_reasons(sample()), [])

    def test_hallucinated_quote_rejected(self):
        row = sample()
        row["hop2"]["evidence"] = "En påhittad uppgift om Testköping"
        self.assertIn("invalid_d2_evidence", generation_reasons(row))

    def test_bridge_leak_rejected(self):
        row = sample()
        row["merged"]["question"] = "Var föddes Anna Exempel?"
        self.assertIn("bridge_leaked_in_question", generation_reasons(row))

    def test_same_article_rejected(self):
        row = sample()
        row["d2_document_id"] = row["d1_document_id"]
        self.assertIn("same_article", generation_reasons(row))

    def test_any_one_of_k_can_succeed(self):
        self.assertEqual(curation_reasons(curated_sample(), 3), [])

    def test_all_k_wrong_rejected(self):
        row = curated_sample()
        row["solve_2"]["answer"] = "Felstad"
        self.assertIn("unanswerable_in_k_attempts", curation_reasons(row, 3))

    def test_single_hop_rejected(self):
        row = curated_sample()
        row["quality"]["requires_both_documents"] = False
        self.assertIn("judge_failed_requires_both_documents", curation_reasons(row, 3))

    def test_answer_matching_is_not_substring(self):
        self.assertTrue(exact_answer(" TESTKÖPING. ", "Testköping"))
        self.assertFalse(exact_answer("Inte Testköping utan Felstad", "Testköping"))
        self.assertFalse(exact_answer("", ""))
        self.assertFalse(exact_answer("Aland", "Åland"))

    def test_optional_swedish_date_prefix(self):
        self.assertTrue(exact_answer("20 juli 1969", "den 20 juli 1969"))
        self.assertFalse(exact_answer("21 juli 1969", "den 20 juli 1969"))
        self.assertFalse(exact_answer("Den gamle", "Gamle"))

    def test_imputation_preserves_grounding(self):
        row = curated_sample()
        row["imputed_hop1"] = {**row["imputed_hop1"], "evidence": "Hittade på detta"}
        self.assertIn("invalid_imputed_evidence", curation_reasons(row, 3))

    def test_no_target_in_solver_or_original_q1_in_imputation(self):
        row = curated_sample()
        row["answer"] = "SECRET_TARGET"
        row["q1"] = "SECRET_ORIGINAL_Q1"
        with tempfile.TemporaryDirectory() as temp:
            seed = Path(temp) / "seed.parquet"
            pd.DataFrame([row]).to_parquet(seed, index=False)
            builder = curation_builder(seed, read_config())
        for column in builder.get_column_configs():
            if not hasattr(column, "prompt"):
                continue
            rendered = Template(column.prompt, undefined=StrictUndefined).render(**row)
            if column.name.startswith("solve_") or column.name in ("imputed_hop1_raw", "recomposed_raw"):
                self.assertNotIn("SECRET_TARGET", rendered)
                self.assertNotIn("SECRET_ORIGINAL_Q1", rendered)


class SourceTests(unittest.TestCase):
    def test_disjointness_detects_leakage(self):
        row = sample()
        with self.assertRaisesRegex(ValueError, "leaks"):
            assert_disjoint({"slice0": [row], "eval": [row]})

    def test_partition_is_article_stable(self):
        self.assertEqual(document_split("same article", 42), document_split("same article", 42))
        self.assertEqual({document_split(str(i), 42) for i in range(100)}, {"slice0", "slice1", "eval"})

    def test_missing_article_identity_is_not_invented(self):
        cfg = read_config()["corpus"]
        hit = {"structs": {}, "tokens": [{"word": "Nobel"}] * 30}
        self.assertIsNone(passage_from_hit(hit, "Nobel", cfg, {}, 42))

    def test_cqp_regex_escaped(self):
        self.assertEqual(entity_cqp("Apollo 11"), '[word="Apollo"] [word="11"]')
        self.assertIn(r"\.", entity_cqp("A.B"))
        self.assertIn(r'\"', entity_cqp('A"B'))

    def test_paper_mode_requires_training(self):
        args = parser().parse_args(["curate", "--mode", "paper", "--run-name", "test"])
        with self.assertRaisesRegex(ValueError, "requires"):
            training_provenance(read_config(), args)


def mock_completion(payload: dict) -> dict:
    """Replace only the HTTP boundary; run the installed DataDesigner engine."""
    prompt = str(payload["messages"])
    fixture = sample()
    if "Skapa Q1" in prompt or "Rekonstruera Q1" in prompt:
        answer = fixture["hop1"]
    elif "Skapa Q2" in prompt:
        answer = fixture["hop2"]
    elif "Slå samman" in prompt:
        answer = fixture["merged"]
    elif "Granska kritiskt" in prompt:
        answer = curated_sample()["quality"]
    elif "Besvara frågan" in prompt:
        answer = {"answerable": True, "answer": "Testköping"}
    else:
        answer = "OK"  # Provider health check, if requested by DataDesigner.
    return {"id": "mock", "object": "chat.completion", "created": 0, "model": "mock",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(answer)},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class DataDesignerIntegrationTests(unittest.TestCase):
    def test_generation_curation_and_export_with_mocked_http(self):
        from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient

        async def apost(self, route, payload, *args, **kwargs):
            return mock_completion(payload)

        def post(self, route, payload, *args, **kwargs):
            return mock_completion(payload)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "data/prepared"
            prepared.mkdir(parents=True)
            fixture = sample()
            seed = {k: v for k, v in fixture.items() if k not in {"hop1", "hop2", "merged"}}
            pd.DataFrame([seed]).to_parquet(prepared / "slice1.parquet", index=False)
            prefix = ["--data-dir", str(root / "data"), "--output-dir", str(root / "out")]
            with patch.dict("os.environ", {"S2S_ENDPOINT": "http://mock.invalid/v1", "S2S_API_KEY": "test"}), \
                 patch.object(OpenAICompatibleClient, "_apost", apost), \
                 patch.object(OpenAICompatibleClient, "_post_sync", post):
                report = generate(read_config(), parser().parse_args(prefix + [
                    "generate", "--split", "slice1", "--records", "1", "--run-name", "generated"]))
                self.assertEqual(report["accepted"], 1)
                report = curate(read_config(), parser().parse_args(prefix + [
                    "curate", "--mode", "baseline", "--records", "1", "--run-name", "curated"]))
                self.assertEqual(report["accepted"], 1)
                messages = read_jsonl(root / "out/runs/curated/train.jsonl")
                self.assertEqual(len(messages), 1)
                self.assertIn("Svar: Testköping", messages[0]["messages"][-1]["content"])
                checked = refilter(read_config(), parser().parse_args(prefix + [
                    "refilter", "--source-run", str(root / "out/runs/curated"), "--run-name", "refiltered"]))
                self.assertEqual(checked["accepted"], 1)
                self.assertEqual(checked["llm_calls"], 0)
                evaluation_seed = {**seed, "seed_id": "eval-fixture", "split": "eval"}
                pd.DataFrame([evaluation_seed]).to_parquet(prepared / "eval.parquet", index=False)
                generate(read_config(), parser().parse_args(prefix + [
                    "generate", "--split", "eval", "--records", "1", "--run-name", "eval-reference"]))
                score = evaluate(read_config(), parser().parse_args(prefix + [
                    "evaluate", "--model", "mock", "--records", "1", "--run-name", "eval-result"]))
                self.assertEqual(score["normalized_exact_match"], 1.0)
                frame = pd.read_parquet(root / "out/runs/eval-result/evaluation_input.parquet")
                self.assertNotIn("answer", frame.columns)
                self.assertNotIn("bridge_entity", frame.columns)


if __name__ == "__main__":
    unittest.main()
