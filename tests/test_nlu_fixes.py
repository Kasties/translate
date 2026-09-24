"""Regression checks for nlu-1000 curation, resume, and inference controls."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from jinja2 import StrictUndefined, Template

from source2synth.__main__ import _build_output, _nlu_curator_responses, nlu_build, parser
from source2synth.common import ROOT, read_config, write_json, write_jsonl
from source2synth.nlu_curation import prediction_matches, reading_answer_match
from source2synth.nlu_pipeline import curation_builder
from source2synth.nlu_sources import schedule
from source2synth.nlu_tasks import acceptability_variant, build_example, canonical_source_span
from source2synth.pipeline import parse_json_response
from test_nlu import seed


class CurationFixTests(unittest.TestCase):
    def test_parser_accepts_gemma_channel_wrapper_but_not_extra_json(self):
        from source2synth.nlu_tasks import Draft
        wrapped = '<|channel>thought\n<channel|>{"possible": true, "evidence": "x"}'
        self.assertEqual(parse_json_response(wrapped, Draft)["evidence"], "x")
        with self.assertRaises(ValueError):
            parse_json_response('{"possible": true, "evidence": "x"}{"extra": true}', Draft)

    def test_canonical_source_spans_restore_spacing_without_partial_word_matches(self):
        text = "Stockholm , Göteborg och Uppsala har 1989 invånare ."
        self.assertEqual(canonical_source_span(text, "Stockholm, Göteborg"), "Stockholm , Göteborg")
        self.assertEqual(canonical_source_span(text, "stockholm"), "Stockholm")
        self.assertIsNone(canonical_source_span(text, "9"))
        self.assertIsNone(canonical_source_span(text, "Stockhol"))
        self.assertIsNone(canonical_source_span(text, ""))
        row = seed("reading-comprehension", "quantity")
        row["source"]["text"] = text
        with self.assertRaisesRegex(ValueError, "answer_is_not_exact_source_span"):
            build_example(row, {"possible": True, "evidence": text,
                                "question": "Hur många invånare finns?", "answer": "9"}, [])

    def test_acceptability_does_not_label_valid_subordinate_negation_as_an_error(self):
        changed, operation = acceptability_variant("Det som är viktigt för universitetet ligger nära staden .", 1)
        self.assertNotIn("som inte är", changed)
        self.assertEqual(operation, "duplicated_finite_verb")

    def test_sparse_ged_predictions_preserve_error_position_and_reject_bad_indices(self):
        item = {"task": "grammatical-error-detection", "output": ["o", "o", "b-err", "o"]}
        self.assertTrue(prediction_matches(item, {"error_indices": [2]}))
        for indices in [[1], [], [-1], [4], [True], ["2"]]:
            self.assertFalse(prediction_matches(item, {"error_indices": indices}))

    def ner_item(self):
        excerpt = "Astrid Lindgren skrev böcker i Stockholm ."
        return build_example(seed("named-entity-recognition", "per"), {
            "possible": True, "text": excerpt, "evidence": excerpt,
            "entities": [{"text": "Astrid Lindgren", "type": "per"},
                         {"text": "Stockholm", "type": "loc"}]}, [])

    def test_ner_agrees_by_entities_and_rejects_missing_wrong_or_invented_entities(self):
        item = self.ner_item()
        entities = list(reversed(item["draft"]["entities"]))
        self.assertTrue(prediction_matches(item, {"entities": entities}))
        self.assertFalse(prediction_matches(item, {"entities": entities[:1]}))
        self.assertFalse(prediction_matches(item, {"entities": [
            {"text": "Astrid Lindgren", "type": "org"}, entities[0]]}))
        self.assertFalse(prediction_matches(item, {"entities": entities + [
            {"text": "Göteborg", "type": "loc"}]}))
        self.assertFalse(prediction_matches(item, {"answer": json.dumps(item["output"])}))

    def test_ner_solver_sees_only_excerpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = {"seed_id": "x", "ner_text": "Astrid i Stockholm .", "task": "named-entity-recognition",
                   "target": "SECRET_TARGET", "input": "SECRET_INPUT", "source_text": "SECRET_SOURCE",
                   "expected_json": "SECRET_EXPECTED", "draft_json": "SECRET_DRAFT"}
            path = Path(tmp) / "seed.parquet"
            pd.DataFrame([row]).to_parquet(path)
            builder = curation_builder(path, read_config(), ner=True)
            prompt = next(c.prompt for c in builder.get_column_configs() if c.name == "prediction_raw")
            rendered = Template(prompt, undefined=StrictUndefined).render(**row)
            self.assertNotIn("SECRET", rendered)
            self.assertIn(row["ner_text"], rendered)

    def test_rc_allows_small_prefix_variants_but_not_containment_or_changed_facts(self):
        self.assertTrue(reading_answer_match("I Stockholm.", "Stockholm", "place"))
        self.assertTrue(reading_answer_match("År 1980", "1980", "time"))
        self.assertTrue(reading_answer_match("den 12 maj 1980", "12 maj 1980", "time"))
        for prediction in ["Stockholm eller Göteborg", "inte Stockholm", "Stockholms län"]:
            self.assertFalse(reading_answer_match(prediction, "Stockholm", "place"))
        self.assertFalse(reading_answer_match("1980 1981 1982", "1981", "time"))
        self.assertFalse(reading_answer_match("för att få pengar", "få pengar", "cause"))
        self.assertFalse(reading_answer_match("", "", "place"))

    def test_question_variants_and_answer_token_boundaries(self):
        row = seed("reading-comprehension", "quantity")
        row["source"]["text"] = "Biblioteket har 9 böcker sedan år 1989 ."
        item = build_example(row, {"possible": True, "evidence": row["source"]["text"],
            "question": "Hur stort var antalet böcker år 1989?", "answer": "9"}, [])
        self.assertEqual(item["output"], "9")
        self.assertIn("question_type_requires_semantic_review", item["normalizations"])

    def test_retry_only_mismatches_with_passing_review_and_stop_after_agreement(self):
        rows = [{"id": name, "task": "classification", "target": "kultur", "input": name,
                 "source": {"text": "Source"}, "draft": {}, "output": "kultur"}
                for name in ["retry", "good", "bad_review"]]
        cfg = read_config()
        cfg["nlu"]["curation_attempts"] = 3
        calls = []

        class Engine:
            def create(self, builder, **kwargs):
                data = pd.read_parquet(builder.get_seed_dataset_config().source.path)
                calls.append(data["seed_id"].tolist())
                first = len(calls) == 1
                output = []
                for seed_id in data["seed_id"]:
                    result = {"seed_id": seed_id, "prediction": {
                        "answer": "natur" if first and seed_id != "good" else "kultur"}}
                    if first:
                        result["review"] = {key: seed_id != "bad_review" for key in
                                            ["source_supported", "unambiguous", "language_valid", "task_valid"]}
                    output.append(result)
                return type("Result", (), {"load_dataset": lambda self: pd.DataFrame(output)})()

        with tempfile.TemporaryDirectory() as tmp, patch("source2synth.__main__.designer", return_value=Engine()):
            # Inspect the actual seed files instead of relying on DataDesigner's internals.
            def builder(path, *args, **kwargs):
                return type("Builder", (), {
                    "write_config": lambda self, target: None,
                    "get_seed_dataset_config": lambda self: type("Seed", (), {
                        "source": type("Source", (), {"path": path})()})()})()
            with patch("source2synth.__main__.nlu_curation_builder", side_effect=builder):
                results = _nlu_curator_responses(rows, Path(tmp), cfg)
        self.assertEqual(calls, [["retry", "good", "bad_review"], ["retry"]])
        retried = next(x for x in results if x["seed_id"] == "retry")
        self.assertEqual(len(retried["prediction_attempts"]), 2)
        self.assertTrue(retried["review"]["task_valid"])


class BuildFixTests(unittest.TestCase):
    def test_ner_schedule_replays_offsets_and_uses_unseen_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = read_config()
            sources = [{"id": f"{topic}-{i}", "document_id": f"{topic}-{i}", "source_topic": topic,
                        "split": "slice1", "text": "Staden har en författare vid ett universitet och en film ."}
                       for topic in cfg["nlu"]["topics"] for i in range(10)]
            write_jsonl(Path(tmp) / "nlu/passages.jsonl", sources)
            tasks = ["named-entity-recognition"]
            full = schedule(cfg, Path(tmp), "slice1", tasks, 30, 0)
            chunks = schedule(cfg, Path(tmp), "slice1", tasks, 15, 0) + schedule(cfg, Path(tmp), "slice1", tasks, 15, 15)
            self.assertEqual(full, chunks)
            self.assertEqual(len({x["source"]["id"] for x in full}), 30)

    def test_balanced_export_reserves_slots_for_rare_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "child"
            rows = [{"id": str(i), "task": "classification" if i < 4 else "named-entity-recognition",
                     "target": "x", "input": str(i), "output": "x", "source": {"document_id": str(i)}}
                    for i in range(5)]
            write_jsonl(child / "accepted.jsonl", rows)
            state = {"tasks": ["classification", "named-entity-recognition"], "target": 4,
                     "split": "slice1", "batch_size": 5, "offset": 0, "balance_tasks": True,
                     "batches": [{"curation_run": str(child), "curated": 5}]}
            report = _build_output(root / "run", state, read_config())
            self.assertFalse(report["complete"])
            self.assertEqual(report["by_task"], {"classification": 2, "named-entity-recognition": 1})
            self.assertEqual(report["surplus"], 2)

    def test_expanded_sources_require_explicit_refresh_and_keep_old_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = read_config()
            source = [{"id": str(i), "document_id": str(i), "source_topic": topic,
                       "split": "slice1", "text": topic} for i, topic in enumerate(cfg["nlu"]["topics"])]
            passages = root / "data/nlu/passages.jsonl"
            write_jsonl(passages, source)
            base = ["--data-dir", str(root / "data"), "--output-dir", str(root / "out"),
                    "nlu-build", "--run-name", "run", "--target", "1", "--max-batches", "1"]

            def empty_generate(config, args):
                path = root / "out/runs" / args.run_name
                write_json(path / "report.json", {"stage": "nlu_generation", "accepted": 0})

            with patch("source2synth.__main__.check_backend", return_value={}), \
                 patch("source2synth.__main__.nlu_generate", side_effect=empty_generate):
                nlu_build(cfg, parser().parse_args(base))
                write_jsonl(passages, source + [{**source[0], "id": "extra", "document_id": "extra"}])
                with self.assertRaisesRegex(ValueError, "Resume settings"):
                    nlu_build(cfg, parser().parse_args(base + ["--resume"]))
                nlu_build(cfg, parser().parse_args(base + ["--resume", "--refresh-inputs"]))
            state = json.loads((root / "out/runs/run/state.json").read_text())
            self.assertEqual(len(state["batches"]), 2)
            self.assertEqual(len(state["input_revisions"]), 1)
            self.assertIn("run-v1-gen-0001", state["batches"][1]["generation_run"])


class LauncherTests(unittest.TestCase):
    def test_default_mtp_profile_loads_the_matching_assistant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "gemma-4-12b-it"
            model_dir.mkdir()
            target = model_dir / "gemma-4-12B-it-Q8_0.gguf"
            assistant = model_dir / "mtp-gemma-4-12B-it-Q8_0.gguf"
            target.write_text("fixture")
            assistant.write_text("fixture")
            server = root / "server"
            server.write_text('#!/bin/bash\nif [[ "$1" == --help ]]; then\n'
                              'echo "--spec-type draft-mtp --spec-draft-n-max"\n'
                              'else printf "%s\\n" "$@"; fi\n')
            server.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if not key.startswith("S2S_")}
            env.update(LLAMA_SERVER=str(server), S2S_MODELS_DIR=str(root))
            result = subprocess.run(["bash", str(ROOT / "scripts/serve-local.sh")],
                                    env=env, text=True, capture_output=True, check=True)
            args = result.stdout.splitlines()
            self.assertEqual(args[args.index("--model") + 1], str(target))
            self.assertEqual(args[args.index("--model-draft") + 1], str(assistant))
            self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")
            self.assertNotIn("--chat-template-file", args)
            self.assertEqual(json.loads(args[args.index("--chat-template-kwargs") + 1]),
                             {"enable_thinking": False})

    def test_speculation_flags_and_context_per_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = root / "server"
            server.write_text('#!/bin/bash\nif [[ "$1" == --help ]]; then\n'
                              'echo "--spec-type draft-mtp ngram-simple --spec-draft-n-max"\n'
                              'else printf "%s\\n" "$@"; fi\n')
            server.chmod(0o755)
            gguf = root / "model.gguf"
            gguf.write_text("fixture")
            env = {**os.environ, "LLAMA_SERVER": str(server), "S2S_GGUF": str(gguf),
                   "S2S_MODEL_PROFILE": "base",
                   "S2S_SPEC_TYPE": "draft-mtp", "S2S_PARALLEL": "2", "S2S_DRAFT_TOKENS": "4"}
            env.pop("S2S_CONTEXT", None)
            result = subprocess.run(["bash", str(ROOT / "scripts/serve-local.sh")],
                                    env=env, text=True, capture_output=True, check=True)
            args = result.stdout.splitlines()
            self.assertEqual(args[args.index("--ctx-size") + 1], "8192")
            self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")
            self.assertEqual(args[args.index("--spec-draft-n-max") + 1], "4")
            result = subprocess.run(["bash", str(ROOT / "scripts/serve-local.sh")],
                                    env={**env, "S2S_PARALLEL": "bad"}, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_default_gemma_rejects_mtp_without_matching_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gguf = root / "gemma-4-12b/gemma-4-12b-q8_0.gguf"
            gguf.parent.mkdir()
            gguf.write_text("fixture")
            env = {**os.environ, "LLAMA_SERVER": "/bin/true", "S2S_MODELS_DIR": str(root),
                   "S2S_MODEL_PROFILE": "base", "S2S_SPEC_TYPE": "draft-mtp"}
            for key in ["S2S_GGUF", "S2S_MTP_DRAFT", "S2S_PARALLEL", "S2S_CONTEXT"]:
                env.pop(key, None)
            result = subprocess.run(["bash", str(ROOT / "scripts/serve-local.sh")],
                                    env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no MTP heads", result.stderr)
