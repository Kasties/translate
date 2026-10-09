"""Run the EuroEval CLI with the vLLM backend's prompt stripping applied to API models.

EuroEval's native vLLM backend strips few-shot prompts for base models whose
tokenizer attaches the space to the label token (": ja" -> ":", " ja"). Its
LiteLLM/API backend, which we use against `vllm serve`, does not, so every prompt
ends in "...: " and the model sees a bare space token it never produced in the
few-shot examples. Gemma then emits "<strong>", "0", "\\n\\n" etc. instead of a label
and EuroEval's closest-label fallback hides the damage. This applies the same
`should_prompts_be_stripped` rule, with the served model's tokenizer, before
delegating to the unmodified EuroEval CLI.
"""

from __future__ import annotations

import argparse
import sys


def install_prompt_stripping(tokenizer) -> None:
    from euroeval.benchmark_modules.litellm import LiteLLMModel
    from euroeval.enums import GenerativeType
    from euroeval.tokenisation_utils import should_prompts_be_stripped

    original = LiteLLMModel.generate
    decisions: dict[tuple[str, ...], bool] = {}

    def generate(self, inputs: dict):
        if self.generative_type == GenerativeType.BASE and "text" in inputs:
            # Same label fallback as euroeval/benchmark_modules/vllm.py.
            labels = tuple(self.dataset_config.prompt_label_mapping.values()) or (
                "negative",
                "positive",
            )
            if labels not in decisions:
                decisions[labels] = should_prompts_be_stripped(labels, tokenizer)
                print(
                    f"Prompt stripping for labels {list(labels)}: {decisions[labels]}",
                    file=sys.stderr,
                    flush=True,
                )
            if decisions[labels]:
                inputs = {**inputs, "text": [p.strip() for p in inputs["text"]]}
        return original(self, inputs)

    LiteLLMModel.generate = generate


def load_tokenizer(model: str, revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, usage="%(prog)s --tokenizer M --tokenizer-revision R -- EUROEVAL_ARGS"
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", default="main")
    args, euroeval_args = parser.parse_known_args()
    if euroeval_args[:1] == ["--"]:
        euroeval_args = euroeval_args[1:]

    from euroeval.cli import benchmark

    install_prompt_stripping(load_tokenizer(args.tokenizer, args.tokenizer_revision))
    benchmark.main(args=euroeval_args, prog_name="euroeval")


if __name__ == "__main__":
    main()
