"""One configurable OpenAI-compatible connection shared by all pipelines."""
from __future__ import annotations

import copy
import os
from urllib.parse import urlparse

import requests


def resolved_config(config: dict, model_id: str | None = None, endpoint: str | None = None) -> dict:
    result = copy.deepcopy(config)
    model = result["model"]
    model["id"] = model_id or os.getenv("S2S_MODEL") or model["id"]
    model["endpoint"] = endpoint or os.getenv("S2S_ENDPOINT") or model.get("endpoint")
    if model["provider"] == "local" and not model["endpoint"]:
        raise ValueError("The local provider requires model.endpoint or S2S_ENDPOINT")
    return result


def check_backend(config: dict) -> dict:
    model = resolved_config(config)["model"]
    endpoint = model.get("endpoint")
    if not endpoint:
        raise ValueError("model-check requires an OpenAI-compatible endpoint")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Model endpoint must be an HTTP(S) URL")
    headers = {"Authorization": "Bearer " + os.environ["S2S_API_KEY"]} if os.getenv("S2S_API_KEY") else {}
    try:
        response = requests.get(endpoint.rstrip("/") + "/models", headers=headers, timeout=15)
        response.raise_for_status()
        available = [item["id"] for item in response.json()["data"]]
    except (requests.RequestException, ValueError, KeyError) as error:
        raise ValueError(f"Cannot reach model server at {endpoint}. Start scripts/serve-local.sh first. ({type(error).__name__})") from error
    if model["id"] not in available:
        raise ValueError(f"Requested model {model['id']!r} is not served. Available: {available}. Set --model or the server --alias.")
    return {"endpoint": endpoint, "model": model["id"], "available_models": available,
            "loopback": parsed.hostname in {"localhost", "127.0.0.1", "::1"}}
