import importlib.util
import unittest
from types import SimpleNamespace

from source2synth.euroeval_cli import install_prompt_stripping


class SpaceInLabelTokenizer:
    """SentencePiece-like: ': ja' -> [':', ' ja'], so ': ' alone is [':', ' ']."""

    def __call__(self, text, add_special_tokens=False):
        head, _, label = text.partition(": ")
        return SimpleNamespace(input_ids=[head + ":", " " + label])


@unittest.skipUnless(importlib.util.find_spec("euroeval"), "needs .venv-eval")
class PromptStrippingTest(unittest.TestCase):
    def setUp(self):
        from euroeval.benchmark_modules.litellm import LiteLLMModel

        self.cls = LiteLLMModel
        self.saved = LiteLLMModel.generate
        self.seen = []
        LiteLLMModel.generate = lambda _self, inputs: self.seen.append(inputs)
        install_prompt_stripping(SpaceInLabelTokenizer())

    def tearDown(self):
        self.cls.generate = self.saved

    def model(self, generative_type):
        return SimpleNamespace(
            generative_type=generative_type,
            dataset_config=SimpleNamespace(
                prompt_label_mapping={"correct": "ja", "incorrect": "nej"}
            ),
        )

    def test_base_prompts_lose_trailing_space(self):
        from euroeval.enums import GenerativeType

        prompt = "Mening: X\nGrammatisk korrekt: "
        self.cls.generate(self.model(GenerativeType.BASE), {"text": [prompt]})
        self.assertEqual(self.seen[-1]["text"], [prompt.strip()])

    def test_chat_prompts_untouched(self):
        from euroeval.enums import GenerativeType

        inputs = {"messages": [[{"role": "user", "content": "x "}]]}
        self.cls.generate(self.model(GenerativeType.INSTRUCTION_TUNED), inputs)
        self.assertIs(self.seen[-1], inputs)


if __name__ == "__main__":
    unittest.main()
