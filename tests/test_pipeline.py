"""Experiment runner configuration, stage commands and non-Korp source passages."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from source2synth import run as pipeline
from source2synth.common import ROOT, read_config, read_jsonl
from source2synth.finetune import DEFAULTS, load_training_data, normalize_row
from source2synth.nlu_sources import schedule
from source2synth.sources import korp_tokens, paragraphs, prepare_sources

SWEDISH = (
    "Riksdagen beslutade i går om en ny lag för skolan . Beslutet togs efter en lång debatt "
    "där flera partier deltog och många ledamöter talade om utbildning och kommunernas ansvar ."
)


_environment = mock.patch.dict(os.environ)


def setUpModule():
    # A machine-wide S2S_PIPELINE_PROFILE must not change the defaults these tests expect.
    _environment.start()
    os.environ.pop("S2S_PIPELINE_PROFILE", None)


def tearDownModule():
    _environment.stop()


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


class ConfigTests(unittest.TestCase):
    def test_experiment_overrides_defaults_and_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp) / "exp.yaml", {"sft": {"model": "org/model", "epochs": 3}})
            config = pipeline.load_experiment(str(path), ["sft.epochs=2", "eval.idioms.enabled=false"])
        self.assertEqual(config["name"], "exp")
        self.assertEqual(config["sft"]["model"], "org/model")
        self.assertEqual(config["sft"]["epochs"], 2)
        self.assertIs(config["eval"]["idioms"]["enabled"], False)
        self.assertEqual(config["sft"]["lora_rank"], 16)  # untouched default

    def test_extends_chain_and_free_form_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yaml(Path(tmp) / "base.yaml", {"name": "base", "sft": {"epochs": 5}})
            child = write_yaml(Path(tmp) / "child.yaml", {
                "extends": "base.yaml", "datagen": {"overrides": {"nlu": {"anything": 1}}}})
            config = pipeline.load_experiment(str(child))
        self.assertEqual(config["name"], "child")
        self.assertEqual(config["sft"]["epochs"], 5)
        self.assertEqual(config["datagen"]["overrides"], {"nlu": {"anything": 1}})

    def test_typos_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp) / "exp.yaml", {"sft": {"learning_rat": 1}})
            with self.assertRaisesRegex(pipeline.PipelineError, "sft.learning_rat"):
                pipeline.load_experiment(str(path))
        with self.assertRaisesRegex(pipeline.PipelineError, "Unknown setting"):
            pipeline.load_experiment(None, ["gpu=1"], name="x")
        with self.assertRaisesRegex(pipeline.PipelineError, "letters"):
            pipeline.load_experiment(None, [], name="../escape")

    def test_hardware_profile_layers_over_experiment_before_set(self):
        with mock.patch.dict(os.environ, {"S2S_PIPELINE_PROFILE": "rtx4090-pilot"}):
            config = pipeline.load_experiment("register-native", ["sft.batch_size=2"])
            off = pipeline.load_experiment("register-native", profile="none")
        self.assertEqual((config["generator"]["backend"], config["sft"]["quantization"],
                          config["eval"]["quantization"]), ("llamacpp", "nf4", "bitsandbytes"))
        self.assertEqual(config["sft"]["batch_size"], 2)  # --set still wins
        self.assertEqual((config["name"], config["datagen"]["mode"]), ("register-native", "instructions"))
        self.assertEqual(off["generator"]["backend"], "vllm")
        self.assertEqual(pipeline.load_experiment("register-native", profile="rtx4090-pilot")["_profile"],
                         "rtx4090-pilot")

    def test_example_configs_are_valid(self):
        for path in sorted((ROOT / "configs/pipeline/examples").glob("*.yaml")):
            with self.subTest(path.name):
                pipeline.load_experiment(str(path))

    def test_experiment_references(self):
        config = pipeline.load_experiment(None, ["sft.train_file=[experiment:old, hf:org/data:train]",
                                                 "eval.adapter=experiment:old"], name="new")
        paths = pipeline.Paths("new")
        self.assertEqual(pipeline.train_files(config, paths),
                         ["artifacts/source2synth/runs/old/train.jsonl", "hf:org/data:train"])
        self.assertEqual(pipeline.eval_target(config, paths)["adapter"],
                         str(ROOT / "artifacts/finetuning/old/adapter"))
        config = pipeline.load_experiment(None, ["eval.adapter=none"], name="new")
        self.assertIsNone(pipeline.eval_target(config, paths)["adapter"])
        self.assertEqual(pipeline.train_files(config, paths), ["artifacts/source2synth/runs/new/train.jsonl"])

    def test_finished_stage_with_changed_settings_is_refused(self):
        config = pipeline.load_experiment(None, [], name="guard")
        paths = pipeline.Paths("guard")
        state = {"stages": {"sft": {"settings": pipeline.stage_settings("sft", config, paths)}}}
        pipeline.guard_changed("sft", config, paths, state)
        changed = pipeline.load_experiment(None, ["sft.epochs=2"], name="guard")
        with self.assertRaisesRegex(pipeline.PipelineError, "epochs"):
            pipeline.guard_changed("sft", changed, paths, state)
        # Serving knobs and newly enabled benchmarks do not invalidate a finished eval.
        state = {"stages": {"eval": {"settings": pipeline.stage_settings("eval", config, paths)}}}
        served = pipeline.load_experiment(None, ["eval.port=9000", "eval.idioms.enabled=false"], name="guard")
        pipeline.guard_changed("eval", served, paths, state)
        # Neither does where the adapter is uploaded.
        state = {"stages": {"sft": {"settings": pipeline.stage_settings("sft", config, paths)}}}
        uploaded = pipeline.load_experiment(None, ["sft.upload_repo=me/models"], name="guard")
        pipeline.guard_changed("sft", uploaded, paths, state)

    def test_finished_build_data_is_uploaded_and_reuploaded_when_extended(self):
        config = pipeline.load_experiment(None, ["datagen.upload_repo=me/models"], name="data")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "ROOT", Path(tmp)), \
                mock.patch("huggingface_hub.HfApi") as api, contextlib.redirect_stdout(io.StringIO()):
            paths = pipeline.Paths("data")
            paths.build.mkdir(parents=True)
            for name in ("train.jsonl", "accepted.jsonl", "state.json"):
                (paths.build / name).write_text("{}\n")
            (paths.build / "report.json").write_text('{"complete": true, "accepted": 2, "target": 2}')
            api.return_value.create_commit.return_value = mock.Mock(oid="abc", commit_url="https://hf.test/c/abc")
            state = {"stages": {}}
            pipeline.upload_build(config, paths, state, dry_run=False)
            pipeline.upload_build(config, paths, state, dry_run=False)
            operations = api.return_value.create_commit.call_args.kwargs["operations"]
            self.assertEqual(sorted(op.path_in_repo for op in operations),
                             ["data/data/accepted.jsonl", "data/data/report.json", "data/data/state.json",
                              "data/data/train.jsonl"])
            self.assertEqual(api.return_value.create_commit.call_count, 1)
            (paths.build / "report.json").write_text('{"complete": true, "accepted": 3, "target": 3}')
            pipeline.upload_build(config, paths, state, dry_run=False)
            self.assertEqual(api.return_value.create_commit.call_count, 2)

    def test_finished_adapter_is_uploaded_once_and_checkpoints_deleted(self):
        config = pipeline.load_experiment(None, ["sft.upload_repo=me/models"], name="up")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "ROOT", Path(tmp)), \
                mock.patch("huggingface_hub.HfApi") as api, contextlib.redirect_stdout(io.StringIO()):
            paths = pipeline.Paths("up")
            for name, text in [("manifest.json", '{"status": "complete"}'), ("adapter/adapter_config.json", "{}"),
                               ("checkpoint-61/optimizer.pt", "state")]:
                (paths.sft / name).parent.mkdir(parents=True, exist_ok=True)
                (paths.sft / name).write_text(text)
            paths.root.mkdir(parents=True)
            paths.finetune_config.write_text("model: m\n")
            api.return_value.create_commit.return_value = mock.Mock(oid="abc", commit_url="https://hf.test/c/abc")
            state = {"stages": {}}
            pipeline.upload_sft(config, paths, state, dry_run=False)
            pipeline.upload_sft(config, paths, state, dry_run=False)
            api.return_value.create_repo.assert_called_once_with("me/models", private=True, exist_ok=True)
            operations = api.return_value.create_commit.call_args.kwargs["operations"]
            self.assertEqual(sorted(op.path_in_repo for op in operations),
                             ["up/adapter/adapter_config.json", "up/finetune.yaml", "up/manifest.json"])
            self.assertEqual(api.return_value.create_commit.call_count, 1)
            self.assertFalse((paths.sft / "checkpoint-61").exists())
            self.assertEqual(state["stages"]["sft"]["upload"]["commit"], "abc")


class DryRunTests(unittest.TestCase):
    def dry_run(self, *sets: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            config = pipeline.load_experiment(
                None, [f"datagen.config={ROOT / 'configs/source2synth.yaml'}", *sets], name="dry")
            output = io.StringIO()
            with mock.patch.object(pipeline, "ROOT", Path(tmp)), \
                    mock.patch.object(pipeline, "is_multimodal", return_value=True), \
                    contextlib.redirect_stdout(output):
                pipeline.run(config, "sources", "eval", dry_run=True)
            written = yaml.safe_load((Path(tmp) / "artifacts/pipeline/dry/source2synth.yaml").read_text())
            self.assertEqual(written["model"]["id"], config["generator"]["model"])
            return output.getvalue()

    def test_default_commands(self):
        text = self.dry_run()
        self.assertIn("prepare-nlu --cache-dir data/raw", text)
        self.assertIn("vllm serve google/gemma-4-12B-it --served-model-name google/gemma-4-12B-it", text)
        self.assertIn("--language-model-only", text)
        self.assertIn("nlu-build --target 1000", text)
        self.assertIn("--run-name dry --balance-tasks", text)
        self.assertIn("source2synth.finetune --config artifacts/pipeline/dry/finetune.yaml", text)
        self.assertIn("--enable-lora --max-lora-rank 16 --lora-modules tuned=", text)
        self.assertIn("source2synth.euroeval", text)
        self.assertIn("source2synth.idioms", text)

    def test_alternative_backends(self):
        text = self.dry_run("data.source=hf", "data.hf.dataset=org/texts", "generator.backend=llamacpp",
                            "eval.adapter=none", "eval.quantization=bitsandbytes", "eval.idioms.enabled=false")
        self.assertIn("prepare-sources --source hf --dataset org/texts", text)
        self.assertIn("scripts/serve-local.sh", text)
        self.assertIn("--quantization bitsandbytes", text)
        self.assertNotIn("--enable-lora", text)
        self.assertIn("--models base --tokenizer google/gemma-4-12B", text)
        self.assertNotIn("source2synth.idioms", text)


class SourceTests(unittest.TestCase):
    def test_korp_tokenization(self):
        self.assertEqual(korp_tokens("Han föddes bl.a. i Sverige, t.ex. 1 000 (SVT:s) musik- och teater."),
                         ["Han", "föddes", "bl.a.", "i", "Sverige", ",", "t.ex.", "1", "000", "(", "SVT:s",
                          ")", "musik-", "och", "teater", "."])

    def test_long_paragraphs_are_split_on_sentences(self):
        text = "Rubrik\n\n" + "Ett två tre fyra fem sex . " * 20 + "\n* punkt i en lista"
        chunks = list(paragraphs(text, 25, 50))
        self.assertTrue(chunks)
        self.assertTrue(all(25 <= len(c.split()) <= 50 for c in chunks))
        self.assertTrue(all(c.endswith(".") for c in chunks))

    def test_local_documents_become_schedulable_passages(self):
        config = read_config()
        config["nlu"]["topics"] = {"samhälle": ["riksdagen"], "kultur": ["teater"]}
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs.jsonl"
            rows = [{"id": str(i), "title": f"Artikel {i}", "url": f"https://x/{i}",
                     "text": SWEDISH.replace("Riksdagen", "Riksdagen" if i % 2 else "Teater")
                     + f" Artikel nummer {i} ."}
                    for i in range(120)]
            docs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            manifest = prepare_sources(config, Path(tmp) / "out", source="local", path=str(docs))
            passages = read_jsonl(Path(tmp) / "out/nlu/passages.jsonl")
            self.assertEqual(manifest["passages"], len(passages))
            self.assertEqual({p["source_topic"] for p in passages}, {"samhälle", "kultur"})
            self.assertEqual(len({(p["document_id"], p["source_topic"]) for p in passages}), len(passages))
            for p in passages:
                keyword = config["nlu"]["topics"][p["source_topic"]][0]
                self.assertIn(keyword, p["text"].lower())
                self.assertEqual(p["url"], f"https://x/{p['corpus_text_id']}")
            seeds = schedule(config, Path(tmp) / "out", "slice1", ["classification"], 4, 0)
            self.assertEqual(len(seeds), 4)
            # A prepared passages file is reused unchanged.
            prepare_sources(config, Path(tmp) / "copy", source="local",
                            path=str(Path(tmp) / "out/nlu/passages.jsonl"))
            self.assertEqual(read_jsonl(Path(tmp) / "copy/nlu/passages.jsonl"), passages)

    def test_topic_field_labels_and_unknown_topics(self):
        config = read_config()
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs.jsonl"
            docs.write_text(json.dumps({"id": "1", "text": SWEDISH, "label": "natur"}) + "\n"
                            + json.dumps({"id": "2", "text": SWEDISH, "label": "okänd"}) + "\n")
            manifest = prepare_sources(config, Path(tmp) / "out", source="local", path=str(docs),
                                       topic_field="label")
        self.assertEqual(manifest["passages"], 1)
        self.assertEqual(manifest["skipped"], {"topic_not_in_nlu_topics": 1})


class TrainingDataTests(unittest.TestCase):
    def test_common_layouts_are_normalized(self):
        expected = [{"role": "user", "content": "Fråga"}, {"role": "assistant", "content": "Svar"}]
        for row in ({"prompt": "Fråga", "completion": "Svar"},
                    {"instruction": "Fråga", "output": "Svar"},
                    {"question": "Fråga", "answer": "Svar"},
                    {"conversations": [{"from": "human", "value": "Fråga"}, {"from": "gpt", "value": "Svar"}]}):
            with self.subTest(row=row):
                self.assertEqual(normalize_row(row, 1)["messages"], expected)
        self.assertEqual(normalize_row({"prompt": "Fråga", "completion": "Svar", "system": "S"}, 1)
                         ["messages"][0], {"role": "system", "content": "S"})
        with self.assertRaisesRegex(ValueError, "end with assistant"):
            normalize_row({"messages": [{"role": "user", "content": "x"}]}, 1)
        with self.assertRaisesRegex(ValueError, "no messages"):
            normalize_row({"text": "x"}, 1)

    def test_files_without_provenance_are_deduplicated_and_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
            first.write_text(json.dumps({"prompt": "p", "completion": "c"}) + "\n"
                             + json.dumps({"prompt": "p", "completion": "d"}) + "\n")
            second.write_text(json.dumps({"question": "q", "answer": "a"}) + "\n")
            rows, info = load_training_data({**DEFAULTS, "train_file": [str(first), str(second)]})
            self.assertEqual(len(rows), 2)
            self.assertEqual(info["sources"][0]["duplicates_dropped"], 1)
            self.assertEqual(info["sources"][0]["provenance"], "none")
            with self.assertRaisesRegex(ValueError, "require_provenance"):
                load_training_data({**DEFAULTS, "train_file": str(first), "require_provenance": True})

    def test_checked_in_export_keeps_strict_provenance(self):
        path = "artifacts/source2synth/runs/nlu-1000/train.jsonl"
        if not (ROOT / path).exists():
            self.skipTest("nlu-1000 export not present")
        rows, info = load_training_data({**DEFAULTS, "train_file": path})
        self.assertEqual(info["provenance"], "curated")
        self.assertEqual(info["rows"], len(rows))
        self.assertIn("by_task", info)



class EmptyBatchTests(unittest.TestCase):
    def test_all_failed_batch_yields_no_rows_unless_server_is_down(self):
        from data_designer.interface.errors import DataDesignerGenerationError

        from source2synth import __main__ as cli

        failing = mock.Mock()
        failing.return_value.create.side_effect = DataDesignerGenerationError("all rows failed")
        with mock.patch.object(cli, "designer", failing), mock.patch.object(cli, "check_backend"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.create_records(Path("/tmp/x"), {}, None, 3, "task"), [])
        with mock.patch.object(cli, "designer", failing), \
                mock.patch.object(cli, "check_backend", side_effect=ValueError("Cannot reach")):
            with self.assertRaisesRegex(ValueError, "Cannot reach"):
                cli.create_records(Path("/tmp/x"), {}, None, 3, "task")


if __name__ == "__main__":
    unittest.main()
