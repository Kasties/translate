"""Checks for the NLU format boundaries that can silently corrupt labels."""
from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from jinja2 import StrictUndefined, Template
import pandas as pd

from source2synth.backend import resolved_config
from source2synth.common import read_config, write_json, write_jsonl
from source2synth.nlu_sources import TASKS, assert_source_disjoint, schedule
from source2synth.nlu_tasks import Prediction, build_example, entity_tags, move_verb, training_record
from source2synth.__main__ import nlu_generate, nlu_curate, nlu_merge, nlu_build, parser
from source2synth.nlu_pipeline import curation_builder


def seed(task: str, target: str) -> dict:
    return {"seed_id": "fixture", "task": task, "target": target, "style": "saklig",
            "difficulty": "medel", "split": "slice1",
            "source": {"text": "Astrid Lindgren skrev böcker i Stockholm . Hon blev berömd .",
                       "title": "Astrid Lindgren", "document_id": "one", "id": "passage"}}


class FormatTests(unittest.TestCase):
    def test_ner_bio_is_exact_and_repeated_entities_are_tagged(self):
        words, tags = entity_tags("Astrid Lindgren mötte Astrid Lindgren .",
                                  [{"text": "Astrid Lindgren", "type": "per"}])
        self.assertEqual(len(words), len(tags))
        self.assertEqual(tags, ["b-per", "i-per", "o", "b-per", "i-per", "o"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            entity_tags("Astrid Lindgren", [{"text": "Astrid Lindgren", "type": "per"},
                                             {"text": "Astrid", "type": "per"}])

    def test_ged_moves_only_a_verb_and_marks_its_new_position(self):
        draft = {"correct_text": "Forskaren undersöker ofta materialet med en kamera .",
                 "verb": "undersöker", "move_after": "materialet"}
        original, correct_tags = move_verb(draft, False)
        moved, tags = move_verb(draft, True)
        self.assertEqual(sorted(original), sorted(moved))
        self.assertEqual(correct_tags, ["o"] * len(original))
        self.assertEqual(tags.count("b-err"), 1)
        self.assertEqual(moved[tags.index("b-err")], "undersöker")
        with self.assertRaisesRegex(ValueError, "phrase_boundary"):
            move_verb({**draft, "correct_text": "Forskaren undersöker materialet , och gör anteckningar .",
                       "move_after": "anteckningar"}, True)

    def test_reading_answer_must_be_an_exact_span_and_stay_short(self):
        good = {"possible": True, "evidence": "Astrid Lindgren skrev böcker i Stockholm .",
                "question": "Var skrev Astrid Lindgren böcker?", "answer": "Stockholm"}
        item = build_example(seed("reading-comprehension", "place"), good, [])
        self.assertEqual(item["context"][item["answers"]["answer_start"][0]:][:9], "Stockholm")
        self.assertEqual(training_record(item)["messages"][-1]["content"], "Stockholm")
        with self.assertRaisesRegex(ValueError, "source_span"):
            build_example(seed("reading-comprehension", "place"), {**good, "answer": "Göteborg"}, [])
        with self.assertRaisesRegex(ValueError, "leaked"):
            build_example(seed("reading-comprehension", "place"), {**good,
                "question": "Var ligger Stockholm?"}, [])
        with self.assertRaisesRegex(ValueError, "named_answer_contains_digits"):
            row = seed("reading-comprehension", "place")
            row["source"]["text"] += " År 2014 öppnade biblioteket ."
            build_example(row, {**good, "answer": "2014", "evidence": "År 2014 öppnade biblioteket .",
                                "question": "Var öppnade biblioteket?"}, [])

    def test_repeated_identical_ner_entities_are_annotated_once(self):
        excerpt = "Astrid Lindgren skrev böcker i Stockholm ."
        item = build_example(seed("named-entity-recognition", "per"),
                             {"possible": True, "evidence": excerpt, "text": excerpt,
                              "entities": [{"text": "Astrid Lindgren", "type": "per"},
                                           {"text": "Astrid Lindgren", "type": "per"},
                                           {"text": "Stockholm", "type": "loc"}]}, [])
        self.assertEqual(item["tags"], ["b-per", "i-per", "o", "o", "o", "b-loc", "o"])
        self.assertIn("deduplicate_identical_entities", item["normalizations"])

    def test_nli_does_not_accept_a_copied_premise_as_hypothesis(self):
        same = "Astrid Lindgren skrev böcker i Stockholm."
        draft = {"possible": True, "evidence": "Astrid Lindgren skrev böcker i Stockholm .",
                 "premise": same, "hypothesis": same, "label": "entailment"}
        with self.assertRaisesRegex(ValueError, "trivial_nli_copy"):
            build_example(seed("natural-language-inference", "entailment"), draft, [])

    def test_unacceptable_sentence_gets_a_controlled_grammar_error(self):
        correct = "Astrid Lindgren är en författare som skrev böcker i Stockholm ."
        item = build_example(seed("linguistic-acceptability", "incorrect"),
                             {"possible": True, "evidence": "Astrid Lindgren skrev böcker i Stockholm .",
                              "correct_text": correct, "text": correct.replace("är", "var", 1),
                              "label": "incorrect"}, [])
        self.assertIn("är är", item["text"])
        self.assertEqual(item["output"], "incorrect")
        self.assertIn("controlled_grammar_edit:duplicated_finite_verb", item["normalizations"])

    def test_local_model_can_be_switched_without_cloud_provider(self):
        cfg = resolved_config(read_config(), "other-model", "http://127.0.0.1:9999/v1")
        self.assertEqual(cfg["model"]["id"], "other-model")
        self.assertEqual(cfg["model"]["endpoint"], "http://127.0.0.1:9999/v1")
        self.assertEqual(cfg["model"]["provider"], "local")

    def test_independent_solver_prompt_excludes_reference_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = {"seed_id": "x", "task": "classification", "target": "SECRET_TARGET",
                   "input": "Text: EN ANNAN TEXT",
                   "source_text": "SECRET_SOURCE", "expected_json": '"SECRET_LABEL"',
                   "draft_json": "SECRET_DRAFT"}
            path = Path(tmp) / "seed.parquet"
            pd.DataFrame([row]).to_parquet(path)
            builder = curation_builder(path, read_config())
            column = next(c for c in builder.get_column_configs() if c.name == "prediction_raw")
            prompt = Template(column.prompt, undefined=StrictUndefined).render(**row)
            for secret in ("SECRET_SOURCE", "SECRET_LABEL", "SECRET_DRAFT", "SECRET_TARGET"):
                self.assertNotIn(secret, prompt)

    def test_token_and_label_predictions_share_one_parquet_type(self):
        token_answer = Prediction.model_validate({"answer": ["o", "b-per"]}).model_dump()
        label_answer = Prediction.model_validate({"answer": "positive"}).model_dump()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.parquet"
            pd.DataFrame([{"prediction": token_answer}, {"prediction": label_answer}]).to_parquet(path)
            actual = pd.read_parquet(path)["prediction"].tolist()
            self.assertEqual(json.loads(actual[0]["answer"]), ["o", "b-per"])
            self.assertEqual(actual[1]["answer"], "positive")

    def test_all_tasks_cycle_requested_labels_and_sources_stay_disjoint(self):
        cfg = read_config()
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for i, topic in enumerate(cfg["nlu"]["topics"]):
                rows.append({"id": str(i), "document_id": str(i), "source_topic": topic,
                             "split": "slice1", "text": topic, "title": topic})
            write_jsonl(Path(tmp) / "nlu/passages.jsonl", rows)
            batch = schedule(cfg, Path(tmp), "slice1", list(TASKS), 12, 0)
            self.assertEqual(len(batch), 84)
            self.assertEqual(len({x["target"] for x in batch if x["task"] == "classification"}), 6)
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                assert_source_disjoint(rows + [{**rows[0], "split": "eval"}])


class DataDesignerNLUTests(unittest.TestCase):
    def test_target_build_deduplicates_and_resumes_to_exact_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = read_config()
            passages = [{"id": str(i), "document_id": str(i), "source_topic": topic,
                         "split": "slice1", "text": "Exempeltext", "title": topic}
                        for i, topic in enumerate(cfg["nlu"]["topics"])]
            write_jsonl(root / "data/nlu/passages.jsonl", passages)
            prefix = ["--data-dir", str(root / "data"), "--output-dir", str(root / "out")]
            calls = []

            def fake_generate(config, args):
                calls.append(args.offset)
                path = root / "out/runs" / args.run_name
                path.mkdir()
                write_json(path / "report.json", {"stage": "nlu_generation", "accepted": 1})

            def fake_curate(config, args):
                offset = int(args.source_run[0].name.split("-gen-")[1])
                path = root / "out/runs" / args.run_name
                path.mkdir()
                row = {"id": f"seed-{offset}", "task": "classification", "split": "slice1",
                       "target": "kultur", "input": "duplicate" if offset < 2 else "unique",
                       "output": "kultur", "source": passages[0],
                       "generation_run": str(args.source_run[0])}
                accepted = [row]
                if args.run_name.startswith("overshoot"):
                    accepted.append({**row, "id": "extra", "input": "second"})
                write_jsonl(path / "accepted.jsonl", accepted)
                write_json(path / "report.json", {"stage": "nlu_curation", "accepted": len(accepted)})

            base = prefix + ["nlu-build", "--target", "2", "--batch-size", "1",
                             "--max-batches", "2", "--tasks", "classification",
                             "--run-name", "target-build"]
            with patch("source2synth.__main__.check_backend", return_value={"model": "mock"}), \
                 patch("source2synth.__main__.nlu_generate", side_effect=fake_generate), \
                 patch("source2synth.__main__.nlu_curate", side_effect=fake_curate):
                partial = nlu_build(cfg, parser().parse_args(base))
                self.assertFalse(partial["complete"])
                self.assertEqual(partial["accepted"], 1)
                self.assertEqual(partial["next_offset"], 2)
                final = nlu_build(cfg, parser().parse_args(base + ["--resume"]))
                self.assertTrue(final["complete"])
                self.assertEqual(final["accepted"], 2)
                self.assertEqual(final["rejection_counts"], {"duplicate_across_batches": 1})
                self.assertEqual(calls, [0, 1, 2])
                self.assertEqual(len((root / "out/runs/target-build/train.jsonl").read_text().splitlines()), 2)
                with self.assertRaisesRegex(ValueError, "Resume settings"):
                    nlu_build(cfg, parser().parse_args(prefix + ["nlu-build", "--target", "3",
                        "--batch-size", "1", "--tasks", "classification", "--run-name",
                        "target-build", "--resume"]))
                overshoot = nlu_build(cfg, parser().parse_args(prefix + ["nlu-build",
                    "--target", "1", "--batch-size", "1", "--max-batches", "1",
                    "--tasks", "classification", "--run-name", "overshoot"]))
                self.assertTrue(overshoot["complete"])
                self.assertEqual(overshoot["accepted"], 1)
                self.assertEqual(overshoot["surplus"], 1)
                self.assertEqual(len((root / "out/runs/overshoot/surplus.jsonl").read_text().splitlines()), 1)

    def test_generation_and_independent_curation_use_the_real_engine(self):
        from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient

        source = "Astrid Lindgren skrev böcker i Stockholm . Hon blev berömd ."

        def completion(payload):
            prompt = str(payload["messages"])
            if "Föreslaget facit" in prompt:
                answer = {"source_supported": True, "unambiguous": True,
                          "language_valid": True, "task_valid": True, "reason": "Källstött"}
            elif "Lös följande NLU-uppgift" in prompt:
                answer = {"answer": "Astrid Lindgren"}
            elif "Uppgift: linguistic-acceptability" in prompt:
                answer = {"possible": True,
                          "evidence": "Astrid Lindgren skrev böcker i Stockholm . Hon blev berömd .",
                          "correct_text": "Astrid Lindgren skrev böcker i Stockholm och blev berömd .",
                          "text": "Astrid Lindgren skrev böcker i Stockholm och blev berömd .",
                          "label": "incorrect"}
            elif "Skapa en svensk fråga" in prompt:
                answer = {"possible": True, "evidence": "Astrid Lindgren skrev böcker i Stockholm .",
                          "question": "Vem skrev böcker i Stockholm?", "answer": "Astrid Lindgren"}
            else:
                answer = "OK"
            return {"id": "mock", "object": "chat.completion", "created": 0, "model": "mock",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(answer)},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        async def apost(self, route, payload, *args, **kwargs):
            return completion(payload)

        def post(self, route, payload, *args, **kwargs):
            return completion(payload)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"id": str(i), "document_id": str(i), "source_topic": topic,
                     "split": "slice1", "text": source, "title": "Biografi", "url": "https://example.org"}
                    for i, topic in enumerate(read_config()["nlu"]["topics"])]
            write_jsonl(root / "data/nlu/passages.jsonl", rows)
            prefix = ["--data-dir", str(root / "data"), "--output-dir", str(root / "out")]
            with patch("source2synth.__main__.check_backend", return_value={"model": "mock"}), \
                 patch.object(OpenAICompatibleClient, "_apost", apost), \
                 patch.object(OpenAICompatibleClient, "_post_sync", post):
                generated = nlu_generate(read_config(), parser().parse_args(prefix + [
                    "nlu-generate", "--tasks", "reading-comprehension", "--records", "1",
                    "--run-name", "generated"]))
                self.assertEqual(generated["accepted"], 1)
                curated = nlu_curate(read_config(), parser().parse_args(prefix + [
                    "nlu-curate", "--source-run", str(root / "out/runs/generated"),
                    "--run-name", "curated"]))
                self.assertEqual(curated["accepted"], 1)
                self.assertEqual(curated["prefilter_rejected"], 0)
                lines = (root / "out/runs/curated/train.jsonl").read_text().splitlines()
                self.assertEqual(json.loads(lines[0])["messages"][-1]["content"], "Astrid Lindgren")

                generated = nlu_generate(read_config(), parser().parse_args(prefix + [
                    "nlu-generate", "--tasks", "linguistic-acceptability", "--offset", "1",
                    "--records", "1", "--run-name", "grammar-generated"]))
                self.assertEqual(generated["accepted"], 1)
                curated = nlu_curate(read_config(), parser().parse_args(prefix + [
                    "nlu-curate", "--source-run", str(root / "out/runs/grammar-generated"),
                    "--run-name", "grammar-curated"]))
                self.assertEqual(curated["accepted"], 1)
                self.assertEqual(curated["solver_disagreements_on_controlled_tasks"], 1)
                lines = (root / "out/runs/grammar-curated/train.jsonl").read_text().splitlines()
                self.assertEqual(json.loads(lines[0])["messages"][-1]["content"], "incorrect")

                merged = nlu_merge(read_config(), parser().parse_args(prefix + [
                    "nlu-merge", "--source-run", str(root / "out/runs/curated"),
                    str(root / "out/runs/grammar-curated"), "--run-name", "merged"]))
                self.assertEqual(merged["accepted"], 2)
                self.assertEqual(merged["llm_calls"], 0)
                self.assertEqual(len((root / "out/runs/merged/train.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
