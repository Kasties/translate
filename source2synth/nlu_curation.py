"""Independent answer comparison; never infer NER correctness from BIO validity."""
from __future__ import annotations

import json
import re

from .common import exact_answer, normalize
from .nlu_tasks import NERPrediction, canonical_source_span, entity_tags

REVIEW_FIELDS = ("source_supported", "unambiguous", "language_valid", "task_valid")
CONTROLLED_TASKS = {"linguistic-acceptability", "grammatical-error-detection"}


def reading_answer_match(actual: str, expected: str, target: str) -> bool:
    """Allow only task-specific introductory words, never arbitrary containment."""
    if exact_answer(actual, expected):
        return True
    prefixes = {
        "place": r"^(?:i|på|vid|från|till) ",
        "time": r"^(?:år|året|under år|under året) ",
        "quantity": r"^(?:antalet är) ",
    }
    pattern = prefixes.get(target)
    if not pattern:
        return False
    # Compare both variants so canonical answers may themselves include a prefix.
    return exact_answer(re.sub(pattern, "", normalize(actual)),
                        re.sub(pattern, "", normalize(expected)))


def prediction_matches(item: dict, prediction: dict) -> bool:
    expected = item["output"]
    if item["task"] == "named-entity-recognition":
        try:
            entities = NERPrediction.model_validate(prediction).model_dump()["entities"]
            for entity in entities:
                entity["text"] = canonical_source_span(item["text"], entity["text"]) or entity["text"]
            entities = list({(e["text"], e["type"]): e for e in entities}.values())
            _, tags = entity_tags(item["text"], entities)
            return tags == expected
        except (ValueError, KeyError, TypeError):
            return False
    actual = prediction.get("answer")
    if isinstance(expected, list):
        if "error_indices" in prediction:
            indices = prediction["error_indices"]
            if (not isinstance(indices, list) or any(type(i) is not int or i < 0 or i >= len(expected)
                                                    for i in indices)):
                return False
            return ["b-err" if i in indices else "o" for i in range(len(expected))] == expected
        try:
            tags = json.loads(actual) if isinstance(actual, str) else actual
            return isinstance(tags, list) and all(isinstance(t, str) for t in tags) and [
                t.lower() for t in tags] == expected
        except (ValueError, TypeError):
            return False
    if not isinstance(actual, str):
        return False
    if item["task"] == "reading-comprehension":
        return reading_answer_match(actual, expected, item["target"])
    return actual.strip().casefold() == expected


def review_passes(review: dict) -> bool:
    return all(review.get(key) is True for key in REVIEW_FIELDS)
