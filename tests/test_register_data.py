"""Native targets, source isolation, sampling fairness and resumed cost accounting."""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import requests

from source2synth import run as pipeline
from source2synth.common import read_config, read_jsonl, write_jsonl
from source2synth.finetune import load_training_rows, tokenize_rows
from source2synth.instruction_data import (
    CompletionClient, Grounded, Instruction, Review, build_instructions, compare_runs,
    make_example, quotas, register_quotas, review_request, training_record,
)
from source2synth.register_sampling import cluster_documents, diversity, efficiency, select
from source2synth.register_sources import (
    korp_cqp, korp_quotas, korp_records, korp_text, native_passages, prepare_registers,
)

TEXT = ("Det är ingen ko på isen, sade hon. Vi får ta en sak i taget och hjälpas åt, "
        "för det ordnar sig nog till slut. Han hade haft en lång dag men nu kändes "
        "det som om en sten föll från hans hjärta.")


def source(index=0, register="debate", **kwargs):
    return {"id": f"p{index}", "document_id": f"doc{index}", "text": TEXT,
            "register": register, "cluster": index % 3, "split": "slice1", "native_text": True,
            **kwargs}


class SamplingTests(unittest.TestCase):
    def test_balances_registers_even_with_unequal_cluster_counts(self):
        rows = [source(i, "debate", cluster=i % 6) for i in range(30)]
        rows += [source(30 + i, "fiction", cluster=0) for i in range(30)]
        selected = select(rows, "balanced", 18, 42, {})
        self.assertEqual(sum(r["register"] == "fiction" for r in selected), 9)
        self.assertEqual(len({r["document_id"] for r in selected}), 18)
        self.assertEqual(selected, select(list(reversed(rows)), "balanced", 18, 42, {}))

    def test_no_keyword_filter_in_balanced_or_random(self):
        rows = [source(i) for i in range(20)]
        topics = {"kultur": ["teater"]}
        self.assertEqual(len(select(rows, "balanced", 8, 42, topics)), 8)
        self.assertEqual(len(select(rows, "random", 8, 42, topics)), 8)
        self.assertEqual(select(rows, "keyword", 8, 42, topics), [])
        rows += [source(99, "encyclopedia", text="Teater är ett ord.")]
        self.assertEqual([r["id"] for r in select(rows, "keyword", 8, 42, topics)], ["p99"])

    def test_deterministic_clusters_and_exact_mean_distance(self):
        vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
        labels = cluster_documents(vectors, 2, 42)
        self.assertEqual(labels, cluster_documents(vectors, 2, 42))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        report = diversity([source(0), source(1)], {"doc0": [1, 0], "doc1": [0, 1]})
        self.assertAlmostEqual(report["mean_document_cosine_distance"], 1)

    def test_cost_includes_rejects_and_never_invents_missing_usage(self):
        metrics = diversity([source()])
        cost = efficiency(metrics, [{"usage": {"completion_tokens": 20}},
                                    {"usage": {"completion_tokens": 30}}])
        self.assertEqual(cost["completion_tokens"], 50)
        self.assertEqual(cost["per_1000_generated_tokens"]["documents"], 20)
        missing = efficiency(metrics, [{"usage": {}}, {"usage": {"completion_tokens": 30}}])
        self.assertIsNone(missing["completion_tokens"])
        self.assertIsNone(missing["per_1000_generated_tokens"]["documents"])


def korp_tokens(text, head=""):
    """Tokens as Korp returns them: whitespace escaped in _head/_tail."""
    words = text.split(" ")
    return [{"word": word, "_head": head if i == 0 else None,
             "_tail": "\\s" if i < len(words) - 1 else "\\n"} for i, word in enumerate(words)]


class FakeKorp:
    """Answers like KorpClient.get; texts are listed in the server's seeded random order."""
    def __init__(self, corpora, scrambled=(), whitespace=True):
        self.corpora, self.scrambled, self.whitespace = corpora, set(scrambled), whitespace
        self.calls = []

    def get(self, command, params):
        self.calls.append((command, dict(params)))
        names = params["corpus"].split(",")
        if command == "corpus_info":
            p = ["word", "_head", "_tail"] if self.whitespace else ["word"]
            response = {"corpora": {c: {"attrs": {"p": p, "s": ["text", "text__id", "text_title", "thread_id",
                                                             "text_username", "paragraph"]},
                                        "info": {"Updated": "2026-08-14"}} for c in names}}
        elif params["cqp"] == "<sentence> []":
            # Corpus order: coherent texts form runs; scrambled ones alternate.
            ids = ["a", "b"] * 50 if names[0] in self.scrambled else ["a"] * 50 + ["b"] * 50
            response = {"kwic": [{"structs": {"text__id": i}} for i in ids]}
        elif params["end"] == 0:
            response = {"corpus_hits": {c: len(self.corpora[c]) for c in names}}
        else:
            texts = self.corpora[names[0]][params["start"]:params["end"] + 1]
            response = {"kwic": [{"tokens": korp_tokens(text), "match": {"position": 1000 * params["start"] + i},
                                  "structs": {"text__id": f"t{i}", "text_title": "Tråd",
                                              "thread_id": f"th{i // 2}" if i != 3 else ""}}
                                 for i, text in enumerate(texts, params["start"])]}
        return {"response": response, "response_url": "https://korp.test/query",
                "retrieved_at": "2026-10-03T00:00:00+00:00", "response_sha256": "abc"}


def korp_spec(**kwargs):
    return {"name": "test", "kind": "korp", "corpora": ["BIG", "SMALL"], "unit": "text",
            "attributes": {"title": "text_title"}, "group_by": "thread_id", "max_documents": 8,
            "min_tokens": 5, **kwargs}


class SourceTests(unittest.TestCase):
    def test_korp_text_rebuilds_whitespace_and_counts_lost_separators(self):
        tokens = [{"word": "Fru", "_head": "\\s\\s", "_tail": "\\s"}, {"word": "talman", "_tail": ""},
                  {"word": "!", "_tail": "\\n\\n"}, {"word": "Det", "_tail": "\\t"}, {"word": "går", "_tail": "\\n"}]
        self.assertEqual(korp_text(tokens), ("Fru talman!\n\nDet\tgår", 0))
        glued = [{"word": "vara", "_tail": ""}, {"word": "rätt", "_tail": "\\s"}, {"word": "nu"}]
        self.assertEqual(korp_text(glued), ("vararätt nu", 1))
        # Whitespace kept as its own escaped token is whitespace, not a word to glue.
        spaced = [{"word": "Om", "_tail": "\\s"}, {"word": "jobbet", "_tail": ""}, {"word": "\\n\\n", "_tail": ""},
                  {"word": "Ditt", "_tail": "\\s"}, {"word": "arbete"}]
        self.assertEqual(korp_text(spaced), ("Om jobbet\n\nDitt arbete", 0))

    def test_korp_cqp_filters_dates_and_escapes_values(self):
        spec = {"name": "lb", "unit": "page", "date_from": datetime.date(2010, 1, 1),
                "where": {"text_mediatype": 'e"text.'}}
        self.assertEqual(korp_cqp(spec), '<page> [(int(_.text_datefrom) >= 20100101) & '
                                         '(_.text_mediatype = "e\\"text\\.")]')
        self.assertEqual(korp_cqp({"name": "wiki", "unit": "text"}), "<text> []")
        with self.assertRaisesRegex(ValueError, "attribute"):
            korp_cqp({"name": "bad", "unit": "text] []"})

    def test_korp_quotas_split_budget_by_hits_or_equally(self):
        self.assertEqual(korp_quotas({"A": 300, "B": 100, "C": 0}, 8, "proportional"), {"A": 6, "B": 2})
        self.assertEqual(korp_quotas({"A": 300, "B": 100}, 5, "equal"), {"A": 3, "B": 2})

    def test_korp_records_sample_per_corpus_and_skip_short_units(self):
        long = "Det är ingen ko på isen sade hon"
        corpora = {"BIG": [long if i % 2 else "Kort." for i in range(300)], "SMALL": [long] * 100}
        client, rejected = FakeKorp(corpora), Counter()
        rows = list(korp_records(korp_spec(), client, 42, rejected))
        self.assertEqual([r["corpus"] for r in rows], ["BIG"] * 6 + ["SMALL"] * 2)
        self.assertTrue(all(r["text"] == long and r["title"] == "Tråd" for r in rows))
        self.assertEqual(rows[0]["group_id"], "BIG:th0")
        self.assertIsNone(rows[1]["group_id"])  # empty value: grouped by its own document
        self.assertEqual(rows[0]["corpus_updated"], "2026-08-14")
        page = next(p for c, p in client.calls if c == "query" and p["corpus"] == "BIG" and p["end"] > 0
                    and p["cqp"] != "<sentence> []")
        self.assertEqual((page["sort"], page["random_seed"], page["cqp"]), ("random", 42, "<text> []"))
        self.assertEqual(page["show_struct"], "text__id,text_title,thread_id")  # no usernames requested
        self.assertEqual(rejected, Counter())

    def test_korp_refuses_scrambled_or_whitespace_less_corpora(self):
        corpora = {"BIG": ["Det är ingen ko på isen"] * 10, "SMALL": []}
        with self.assertRaisesRegex(ValueError, "scrambled"):
            list(korp_records(korp_spec(), FakeKorp(corpora, scrambled={"SMALL"}), 42, Counter()))
        with self.assertRaisesRegex(ValueError, "whitespace"):
            list(korp_records(korp_spec(), FakeKorp(corpora, whitespace=False), 42, Counter()))
        with self.assertRaisesRegex(ValueError, "structural"):
            list(korp_records(korp_spec(group_by="forum_id"), FakeKorp(corpora), 42, Counter()))

    def test_native_substrings_preserve_punctuation_and_dialogue(self):
        text = "Rubrik\n\n— ”Det går väl an”, sade hon; så gick de.\n" + TEXT * 3
        passages = list(native_passages(text, 8, 60))
        self.assertTrue(passages)
        self.assertTrue(passages[0][0].startswith("— ”Det"))
        for passage, start, end in passages:
            self.assertEqual(text[start:end], passage)
            self.assertLessEqual(len(passage.split()), 60)

    def test_local_registers_keep_native_targets_and_group_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = fixture_pool(path)
            manifest = prepare_registers(config, path / "data")
            passages = read_jsonl(path / "data/nlu/passages.jsonl")
            documents = {d["id"]: d for d in read_jsonl(path / "data/nlu/documents.jsonl")}
            groups = {}
            for row in passages:
                original = documents[row["document_id"]]["text"]
                self.assertEqual(original[row["char_start"]:row["char_end"]], row["text"])
                self.assertEqual(groups.setdefault(row["group_id"], row["split"]), row["split"])
            self.assertEqual(manifest["embedding_kind"], "test lexical hashing")
            self.assertTrue(passages)
            self.assertIn("slice1/keyword", manifest["comparison"])
            self.assertGreater(manifest["rejected"].get("duplicate_passage", 0), 0)
            config["registers"]["sources"][0]["coherent"] = False
            with self.assertRaisesRegex(ValueError, "shuffled"):
                prepare_registers(config, path / "refused")


def fixture_pool(path):
    config = read_config()
    config["registers"]["embedding"]["model"] = "hash-test"
    config["registers"]["sampling"].update(per_split=100, clusters=3)
    config["instruction_data"]["register_weights"] = None  # equal shares of the fixture registers
    specs = []
    for register in ("debate", "fiction", "encyclopedia"):
        file = path / f"{register}.jsonl"
        docs = [{"id": str(i), "group_id": f"group{i // 2}", "title": f"Text {i}",
                 "text": TEXT + f" Det handlade om {register} och teater under år {1900 + i}."}
                for i in range(30)]
        docs.append({**docs[0], "id": "duplicate", "group_id": "duplicate"})
        write_jsonl(file, docs)
        specs.append({"name": register, "kind": "local", "register": register,
                      "path": str(file), "max_documents": 40, "coherent": True,
                      "license": "test-fixture", "license_url": "https://example.test/license"})
    config["registers"]["sources"] = specs
    return config


class FakeClient:
    """Journal a deterministic response through the actual CompletionClient."""
    def __init__(self, config, directory):
        self.actual = CompletionClient(config, directory)
        self.current_source = None

    def call(self, key, prompt, schema, limit, temperature=0.7):
        if key.endswith("-review"):
            content = {"swedish": True, "self_contained": True, "source_faithful": True,
                       "task_fulfilled": True, "reason": "Alla krav uppfylls."}
        elif schema is Instruction:
            content = {"possible": True, "instruction": f"Skriv en kort berättelse om oro och lättnad i situation {key}."}
        else:
            content = {"possible": True, "literal": "Det finns inget skäl till oro. " + prompt.split("Original:\n")[-1]}
        response = mock.Mock()
        response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(content, ensure_ascii=False)}}], "usage": {"completion_tokens": 25}}
        with mock.patch.object(self.actual.session, "post", return_value=response):
            return self.actual.call(key, prompt, schema, limit, temperature)

    def calls(self):
        return self.actual.calls()


class GenerationTests(unittest.TestCase):
    def test_native_targets_are_unmodified_and_generated_text_is_masked(self):
        s = source()
        drafts = {"backtranslation": {"possible": True, "instruction": "Skriv en kort berättelse om oro och lättnad."},
                  "rewrite": {"possible": True, "literal": "Ingen fara finns just nu. " + TEXT}}
        for task, draft in drafts.items():
            item = make_example(task, s, draft, 0, [])
            self.assertEqual(item["output"], TEXT)
            self.assertEqual(item["target_origin"], "human_verbatim")
            row = training_record(item)
            class Tokenizer:
                def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                    text = "".join(f"<{m['role']}>" + m["content"] for m in messages)
                    return text + ("<assistant>" if add_generation_prompt else "<eos>")

                def __call__(self, text, **kwargs):
                    return {"input_ids": list(text.encode())}
            tokenized, _ = tokenize_rows([row], Tokenizer(), 10000)
            labels = tokenized[0]["labels"]
            supervised = bytes(x for x in labels if x != -100).decode()
            self.assertEqual(supervised, TEXT + "<eos>")

    def test_grounded_evidence_and_exact_quotas(self):
        q = quotas(read_config()["instruction_data"]["weights"], 1000)
        self.assertEqual(q["backtranslation"], 600)
        self.assertEqual(q["rewrite"], 250)
        self.assertEqual(sum(q.values()), 1000)
        draft = {"possible": True, "instruction": "Vad sade hon?", "answer": "Hon sade att allt ordnar sig.",
                 "evidence": "ett citat som inte finns"}
        with self.assertRaisesRegex(ValueError, "evidence"):
            make_example("grounded-qa", source(), draft, 0, [])

    def test_default_register_weights_split_every_task(self):
        config = read_config()["instruction_data"]
        cells = register_quotas(quotas(config["weights"], 1000), config["register_weights"],
                                {"debate", "encyclopedia", "job-ad"})
        self.assertEqual(cells["backtranslation"], {"debate": 240, "encyclopedia": 240, "job-ad": 120})
        self.assertEqual(sum(c["job-ad"] for c in cells.values()), 200)
        with self.assertRaisesRegex(ValueError, "without a register_weights entry"):
            register_quotas({"rewrite": 10}, config["register_weights"], {"debate", "encyclopedia", "job-ad", "fiction"})
        with self.assertRaisesRegex(ValueError, "no passages"):
            register_quotas({"rewrite": 10}, config["register_weights"], {"debate", "encyclopedia"})

    def test_register_weights_give_exact_shares_and_check_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture_pool(root)
            config["instruction_data"]["weights"] = {"backtranslation": 0.5, "rewrite": 0.5}
            config["instruction_data"]["register_weights"] = {"debate": 0.5, "fiction": 0.25, "encyclopedia": 0.25}
            prepare_registers(config, root / "data")
            args = argparse.Namespace(data_dir=root / "data", output_dir=root / "output", split="slice1",
                run_name="shares", target=8, batch_size=4, max_batches=10, resume=False)
            report = build_instructions(config, args, FakeClient)
            self.assertTrue(report["complete"])
            self.assertEqual(report["by_register"], {"debate": 4, "fiction": 2, "encyclopedia": 2})
            self.assertEqual(report["register_quotas"]["rewrite"], {"debate": 2, "fiction": 1, "encyclopedia": 1})
            config["instruction_data"]["register_weights"] = {"debate": 1, "fiction": 0, "encyclopedia": 0}
            args.run_name, args.target = "too-many", 200
            with self.assertRaisesRegex(ValueError, "debate passages"):
                build_instructions(config, args, FakeClient)

    def test_resume_matches_uninterrupted_build_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture_pool(root)
            config["instruction_data"]["weights"] = {"backtranslation": 0.6, "rewrite": 0.4}
            prepare_registers(config, root / "data")
            args = argparse.Namespace(data_dir=root / "data", output_dir=root / "output", split="slice1",
                run_name="resumed", target=10, batch_size=2, max_batches=1, resume=False)
            partial = build_instructions(config, args, FakeClient)
            self.assertFalse(partial["complete"])
            args.resume, args.max_batches = True, 10
            complete = build_instructions(config, args, FakeClient)
            self.assertTrue(complete["complete"])
            args.run_name, args.resume = "full", False
            full = build_instructions(config, args, FakeClient)
            left = root / "output/runs/resumed"
            right = root / "output/runs/full"
            self.assertEqual(read_jsonl(left / "train.jsonl"), read_jsonl(right / "train.jsonl"))
            self.assertEqual(complete["efficiency"], full["efficiency"])
            rows, info = load_training_rows(left / "train.jsonl")
            self.assertEqual(len(rows), 10)
            self.assertEqual(info["source_splits"], ["slice1"])
            self.assertEqual(complete["human_target_rows"], 10)
            self.assertTrue(compare_runs([left, right])["comparable"])
            args.resume = True
            changed = copy.deepcopy(config)
            changed["instruction_data"]["review_max_tokens"] += 1
            with self.assertRaisesRegex(ValueError, "changed"):
                build_instructions(changed, args, FakeClient)
            args.split = "eval"
            with self.assertRaisesRegex(ValueError, "Held-out"):
                build_instructions(config, args, FakeClient)

    def test_parallel_build_matches_sequential_and_reruns_unfinished_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture_pool(root)
            config["instruction_data"]["weights"] = {"backtranslation": 0.6, "rewrite": 0.4}
            prepare_registers(config, root / "data")
            args = argparse.Namespace(data_dir=root / "data", output_dir=root / "output", split="slice1",
                run_name="sequential", target=10, batch_size=2, max_batches=10, resume=False)
            config["model"]["max_parallel_requests"] = 1
            build_instructions(config, args, FakeClient)
            config["model"]["max_parallel_requests"] = 4
            args.run_name = "parallel"
            build_instructions(config, args, FakeClient)
            sequential, parallel = root / "output/runs/sequential", root / "output/runs/parallel"
            self.assertEqual(read_jsonl(sequential / "train.jsonl"), read_jsonl(parallel / "train.jsonl"))

            # Interrupt a backtranslation attempt after its draft call: no attempt record, no review.
            attempt_id = next(path.stem for path in sorted((parallel / "dispatched").glob("*.json"))
                              if json.loads(path.read_text())["task"] == "backtranslation")
            (parallel / f"attempts/{attempt_id}.json").unlink()
            (parallel / f"calls/{attempt_id}-review.json").unlink()
            draft = json.loads((parallel / f"calls/{attempt_id}-draft.json").read_text())
            message = draft["response"]["choices"][0]["message"]
            message["content"] = json.dumps({"possible": True, "instruction": "Berätta om en återspelad dag."})
            (parallel / f"calls/{attempt_id}-draft.json").write_text(json.dumps(draft))
            args.resume = True
            report = build_instructions(config, args, FakeClient)
            self.assertTrue(report["complete"])
            rerun = json.loads((parallel / f"attempts/{attempt_id}.json").read_text())
            # Same task, register and source as dispatched, so the same prompt: the saved
            # draft is replayed, not requested again.
            self.assertEqual(rerun["draft"]["instruction"], "Berätta om en återspelad dag.")
            self.assertTrue(rerun["accepted"])
            self.assertEqual(len(list((parallel / "attempts").glob("*.json"))), 10)

    def test_native_review_shows_target_once_and_schema_is_enforced(self):
        # Regressions: a duplicated source block read as "copied from the source", and
        # unconstrained JSON echoed schema keys such as "title" that strict parsing rejects.
        draft = {"possible": True, "instruction": "Skriv en kort berättelse om oro och lättnad."}
        prompt = review_request(make_example("backtranslation", source(), draft, 0, []))
        self.assertEqual(prompt.count(TEXT), 1)
        self.assertIn("avsiktligt", prompt)
        with tempfile.TemporaryDirectory() as tmp:
            client = CompletionClient(read_config(), Path(tmp))
            response = mock.Mock()
            review = {"swedish": True, "self_contained": True, "source_faithful": True,
                      "task_fulfilled": True, "reason": "Bra."}
            response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(review)}}], "usage": {"completion_tokens": 5}}
            with mock.patch.object(client.session, "post", return_value=response) as post:
                self.assertEqual(client.call("review", prompt, Review, 30), review)
            fmt = post.call_args.kwargs["json"]["response_format"]
            self.assertEqual((fmt["type"], fmt["json_schema"]["name"]), ("json_schema", "Review"))
            self.assertIs(fmt["json_schema"]["schema"]["additionalProperties"], False)

    def test_malformed_completion_is_charged_once_on_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = CompletionClient(read_config(), Path(tmp))
            response = mock.Mock()
            response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": "{"}}],
                                          "usage": {"completion_tokens": 9}}
            with mock.patch.object(client.session, "post", return_value=response) as post:
                for _ in range(2):
                    with self.assertRaises(ValueError):
                        client.call("draft", "prompt", Grounded, 30)
                self.assertEqual(post.call_count, 1)
            self.assertEqual(client.calls()[0]["usage"]["completion_tokens"], 9)

    def test_transport_failure_records_unknown_cost_and_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = CompletionClient(read_config(), Path(tmp))
            with mock.patch.object(client.session, "post", side_effect=requests.Timeout()) as post:
                with self.assertRaisesRegex(RuntimeError, "usage is unknown"):
                    client.call("draft", "prompt", Grounded, 30)
                with self.assertRaisesRegex(ValueError, "previous_request"):
                    client.call("draft", "prompt", Grounded, 30)
                self.assertEqual(post.call_count, 1)
            self.assertFalse(efficiency(diversity([]), client.calls())["usage_complete"])

    def test_pipeline_routes_to_native_builder(self):
        config = pipeline.load_experiment("register-native")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "ROOT", Path(tmp)), \
                mock.patch.object(pipeline, "run_step") as step:
            paths = pipeline.Paths("native-test")
            config["generator"]["backend"] = "external"
            config["generator"]["endpoint"] = "http://localhost:8080/v1"
            # Config file stays in the real workspace while output is isolated.
            config["datagen"]["config"] = str(Path(__file__).resolve().parents[1] / "configs/source2synth.yaml")
            pipeline.run_sources(config, paths, True)
            self.assertIn("prepare-registers", step.call_args.args[0])
            pipeline.run_generate(config, paths, True)
            self.assertIn("instruction-build", step.call_args.args[0])
            self.assertNotIn("--tasks", step.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
