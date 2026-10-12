"""Minimal-pair suite: datasets, the scorer, and the one function that loads a model.

Scoring follows Kunz (2026): each sentence is scored alone as raw text with the
tokenizer's default special tokens, and a pair is right when the better sentence has
the strictly higher log-probability. See README.md, "Scoring".
"""

import ast
import math
import string
from typing import NamedTuple

import torch
import transformers
from datasets import load_dataset

BATCH_SIZE = 32


class MinimalPair(NamedTuple):
    better: str
    worse: str
    # Every subset the pair counts toward. Each gets result records of its own. An Idioms pair counts toward
    # "all" and maybe "challenge". A dataset without "all", such as DaLAJ, says why in its loader.
    subsets: tuple[str, ...]


def load_model(reference, revision=None):
    """Load a checkpoint from a Hugging Face repo id or a local path.

    The only place that loads a model. The class is the one named in the checkpoint
    config, so causal LMs and multimodal ...ForConditionalGeneration models both load.
    """
    config = transformers.AutoConfig.from_pretrained(reference, revision=revision)
    model_class = getattr(transformers, config.architectures[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_class.from_pretrained(
        reference,
        revision=revision,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(reference, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_idioms():
    rows = load_dataset(
        "liu-nlp/swedish-idioms", revision="3d33f55db6c45a82a6a237f3cbee6490af08327d", split="train"
    )
    return [
        MinimalPair(
            better=row["Positive (Minimal Pairs Setup)"],
            worse=row["Negative (Minimal Pairs Setup)"],
            subsets=("all", "challenge") if row["Challenge Set"] == "x" else ("all",),
        )
        for row in rows
    ]


# The CSVs of Kunz (2026), pinned to one commit of jekunz/idiomatic-language.
IDIOMATIC_LANGUAGE_URL = (
    "https://raw.githubusercontent.com/jekunz/idiomatic-language/"
    "a84eaaff73a4f887e2bc95a0d8d66ef5bc36d808/data/{name}_{split}.csv"
)


def load_idiomatic_language_csv(name):
    """The rows of the train, validation and test CSVs of one dataset, together, as the paper's loaders read them."""
    return load_dataset(
        "csv",
        data_files=[IDIOMATIC_LANGUAGE_URL.format(name=name, split=split) for split in ("train", "validation", "test")],
        split="train",
    )


DALAJ_ERROR_TYPES = {"M": "morphology", "S": "syntax", "O": "orthography", "L": "lexical", "P": "punctuation"}


def load_dalaj():
    """DaLAJ pairs, one subset per error type. As in the paper's loader,
    rows with more than one error label, such as "L, S", are left out."""
    rows = load_idiomatic_language_csv("dalaj")
    pairs = []
    for row in rows:
        error_label = ast.literal_eval(row["meta"])["error_label"]
        if error_label in DALAJ_ERROR_TYPES:
            subsets = (DALAJ_ERROR_TYPES[error_label],)
            pairs.append(MinimalPair(better=row["corrected_sentence"], worse=row["sentence"], subsets=subsets))
    return pairs


SCALA_CORRUPTION_TYPES = {"flip_neighbours": "neighbouring words swapped", "delete": "word deleted"}


def load_scala():
    """ScaLA pairs, one subset per corruption type. There is no "all": the paper reports only ScaLA (average),
    without saying how it combines the two types."""
    rows = load_idiomatic_language_csv("scala_swedish")
    return [
        MinimalPair(better=row["correct"], worse=row["incorrect"], subsets=(SCALA_CORRUPTION_TYPES[row["type"]],))
        for row in rows
    ]


TRANSLATIONESE_TRANSLATIONS = {"swedish_opus": "human vs OPUS", "swedish_gpt": "human vs GPT"}


def strip_trailing_punctuation(sentence):
    """As the paper's code does, so a missing final full stop does not decide a Translationese pair."""
    return sentence.strip().rstrip(string.punctuation + " ")


def load_translationese():
    """Translationese pairs in the raw setup (no prompt): two per row, the human version against the OPUS-MT
    and against the GPT-4 translation, one subset each. There is no "all": the paper reports the two apart.
    After the strip, 86 human vs GPT pairs are identical and count as wrong."""
    rows = load_dataset(
        "liu-nlp/translationese-opensubtitles", revision="ea3b71f8e776cdb1a9a03ea21d3122b019168d09", split="train"
    )
    return [
        MinimalPair(
            better=strip_trailing_punctuation(row["swedish_human"]),
            worse=strip_trailing_punctuation(row[column]),
            subsets=(subset,),
        )
        for row in rows
        for column, subset in TRANSLATIONESE_TRANSLATIONS.items()
    ]


# The minimal-pair suite: the loader of each dataset, by the name its result records carry.
MINIMAL_PAIR_DATASETS = {
    "Idioms": load_idioms,
    "DaLAJ": load_dalaj,
    "ScaLA": load_scala,
    "Translationese": load_translationese,
}


@torch.no_grad()
def score_sentences(model, tokenizer, sentences):
    """Return (summed log-probability, token count) per sentence.

    Every token is scored given the tokens before it, so the first token is scored only
    when the tokenizer adds a BOS token in front of it. Padding is masked. Surrounding
    whitespace is stripped, as in the paper's code.
    """
    sentences = [sentence.strip() for sentence in sentences]
    scores = []
    for start in range(0, len(sentences), BATCH_SIZE):
        batch = tokenizer(
            sentences[start : start + BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            padding_side="right",
        ).to(model.device)
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        log_probs = logits[:, :-1].float().log_softmax(-1)
        targets = batch.input_ids[:, 1:]
        mask = batch.attention_mask[:, 1:]
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask
        scores += zip(token_log_probs.sum(1).tolist(), mask.sum(1).tolist())
    return scores


def wilson_interval(successes, n):
    """95% Wilson score interval for a proportion."""
    z = 1.96
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - margin, centre + margin


def evaluate(model, tokenizer, pairs):
    """Score the pairs and return preference accuracy (mean) and preference accuracy (sum),
    with Wilson intervals, for every subset a pair counts toward, and the scores of each pair."""
    # Each distinct sentence is scored once, so identical sentences get identical scores.
    sentences = list(dict.fromkeys(s for pair in pairs for s in (pair.better, pair.worse)))
    scores = dict(zip(sentences, score_sentences(model, tokenizer, sentences)))

    def mean(sentence):
        total, count = scores[sentence]
        # A one-token sentence has no scored token without a BOS token (DaLAJ has "om"). NaN makes
        # its pair wrong, as the paper's code does.
        return total / count if count else math.nan

    def summed(sentence):
        return scores[sentence][0]

    subsets = {}
    for pair in pairs:
        for subset in pair.subsets:
            subsets.setdefault(subset, []).append(pair)

    results = []
    for subset, subset_pairs in subsets.items():
        for metric, score in [("preference accuracy (mean)", mean), ("preference accuracy (sum)", summed)]:
            # Strict comparison: a tie counts as wrong.
            right = sum(score(pair.better) > score(pair.worse) for pair in subset_pairs)
            n = len(subset_pairs)
            lower, upper = wilson_interval(right, n)
            results.append(
                {"subset": subset, "metric": metric, "value": right / n, "lower": lower, "upper": upper, "items": n}
            )
    pair_scores = [
        {
            "subsets": list(pair.subsets),
            "better": pair.better,
            "worse": pair.worse,
            "better_summed_log_probability": scores[pair.better][0],
            "better_token_count": scores[pair.better][1],
            "worse_summed_log_probability": scores[pair.worse][0],
            "worse_token_count": scores[pair.worse][1],
        }
        for pair in pairs
    ]
    return results, pair_scores
