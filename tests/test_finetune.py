"""Protect the supervision and provenance boundaries of the training pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jinja2 import Template

from source2synth.euroeval import benchmark_commands
from source2synth.finetune import BASE_CHAT_TEMPLATE, load_training_rows, tokenize_rows


class CharacterTokenizer:
    """Transparent tokenization makes masked content directly inspectable."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return Template(BASE_CHAT_TEMPLATE).render(
            messages=messages,
            bos_token="<bos>",
            eos_token="<eos>",
            add_generation_prompt=add_generation_prompt,
        )

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode())}


def row():
    return {
        "messages": [
            {"role": "system", "content": "Only return a label."},
            {"role": "user", "content": "This is a lovely film."},
            {"role": "assistant", "content": "positive"},
        ]
    }


class TrainingTests(unittest.TestCase):
    def test_prompt_is_excluded_but_answer_and_eos_are_supervised(self):
        tokenized, info = tokenize_rows([row()], CharacterTokenizer(), 2048)
        item = tokenized[0]
        supervised = bytes(label for label in item["labels"] if label != -100).decode()
        self.assertEqual(supervised, "positive<eos>\n\n")
        self.assertEqual(len(item["labels"]), len(item["input_ids"]))
        self.assertEqual(info["truncated_rows"], 0)

    def test_overlong_examples_fail_instead_of_losing_labels(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            tokenize_rows([row()], CharacterTokenizer(), 10)

    def test_export_must_match_curated_non_eval_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(json.dumps(row()) + "\n")
            accepted = {
                "input": row()["messages"][1]["content"],
                "output": "positive",
                "task": "sentiment-classification",
                "split": "slice1",
                "source": {"split": "slice1"},
            }
            provenance = path.with_name("accepted.jsonl")
            provenance.write_text(json.dumps(accepted) + "\n")
            self.assertEqual(load_training_rows(path)[1]["rows"], 1)
            accepted["source"]["split"] = "eval"
            provenance.write_text(json.dumps(accepted) + "\n")
            with self.assertRaisesRegex(ValueError, "Held-out"):
                load_training_rows(path)
            accepted["source"]["split"] = "slice1"
            accepted["output"] = "negative"
            provenance.write_text(json.dumps(accepted) + "\n")
            with self.assertRaisesRegex(ValueError, "no matching"):
                load_training_rows(path)

    def test_comparison_uses_identical_validation_protocol(self):
        commands = benchmark_commands(
            Path("/venv/bin/python"),
            "http://127.0.0.1:8000/v1",
            ["swerec", "scala-sv"],
            1,
            Path("/cache"),
            "google/gemma-4-12B",
            "rev",
        )
        self.assertEqual(commands[0][1][:-1], commands[1][1][:-1])
        self.assertEqual(commands[0][1][1:3], ["-m", "source2synth.euroeval_cli"])
        self.assertEqual(commands[0][1][commands[0][1].index("--") - 1], "rev")
        self.assertIn("--evaluate-val-split", commands[0][1])
        self.assertIn("--few-shot", commands[0][1])
        self.assertNotIn("--evaluate-test-split", commands[0][1])


if __name__ == "__main__":
    unittest.main()
