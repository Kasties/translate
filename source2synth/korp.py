from __future__ import annotations

import itertools
import json
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import contains, fingerprint, write_json


def entity_cqp(entity: str) -> str:
    if not entity.strip():
        raise ValueError("An entity cannot be empty")
    # Both regex and CQP string syntax need escaping. Never interpolate raw CQP.
    return " ".join('[word="' + re.escape(word).replace('"', '\\"') + '"]' for word in entity.split())


class KorpClient:
    def __init__(self, api: str, cache: Path, offline: bool = False):
        self.api, self.cache, self.offline = api.rstrip("/"), cache, offline
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "TDDE19-Source2Synth/0.1 (bounded research sample)"
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
        )))

    def get(self, command: str, params: dict) -> dict:
        request = {"url": f"{self.api}/{command}", "params": params}
        path = self.cache / f"{fingerprint(request)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.offline:
            raise FileNotFoundError(f"Missing cached Korp request: {path}")
        response = self.session.get(request["url"], params=params, timeout=(10, 90))
        response.raise_for_status()
        payload = response.json()
        if "ERROR" in payload or "error" in payload:
            raise ValueError(f"Korp rejected {command}: {payload.get('ERROR', payload.get('error'))}")
        envelope = {
            "request": request, "response_url": response.url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "response_sha256": fingerprint(payload), "response": payload,
        }
        write_json(path, envelope)
        time.sleep(0.2)
        return envelope


def document_split(document_id: str, seed: int) -> str:
    # Assign an ARTICLE, never a hit or a row, before constructing any pairs.
    bucket = int(fingerprint([seed, document_id])[:8], 16) / 2**32
    return "slice0" if bucket < 0.45 else "slice1" if bucket < 0.9 else "eval"


def passage_from_hit(hit: dict, entity: str, cfg: dict, envelope: dict, seed: int) -> dict | None:
    attrs = hit.get("structs", {})
    words = [token["word"] for token in hit.get("tokens", [])]
    # Keep the token-spaced original so evidence quotes can be checked exactly.
    text = " ".join(words)
    title, url = attrs.get("text_title"), attrs.get("text_url")
    if not title or not url or not cfg["min_words"] <= len(words) <= cfg["max_words"]:
        return None
    if not contains(text, entity) or words.count("•") > 2:
        return None
    doc_id = fingerprint([cfg["id"], url])[:24]
    match = hit["match"]
    return {
        "id": fingerprint([doc_id, text])[:24], "document_id": doc_id,
        "title": title, "url": url, "permalink": attrs.get("text_permalink", ""),
        "corpus_text_id": attrs.get("text__id", ""), "text": text,
        "corpus_position": match["position"],
        "context_start": match["position"] - match["start"],
        "corpus": cfg["id"], "entity": entity,
        "split": document_split(doc_id, seed),
        "retrieved_at": envelope["retrieved_at"], "query_url": envelope["response_url"],
        "response_sha256": envelope["response_sha256"],
        "license_as_listed": cfg["license_as_listed"], "resource": cfg["resource"], "doi": cfg["doi"],
    }


def prepare(config: dict, data_dir: Path, offline: bool = False) -> dict:
    cfg, seed = config["corpus"], config["random_seed"]
    client = KorpClient(cfg["api"], data_dir / "raw", offline)
    info = client.get("corpus_info", {"corpus": cfg["id"]})
    corpus = info["response"]["corpora"][cfg["id"]]
    required = {"text_title", "text_url", "paragraph"}
    if not required.issubset(corpus["attrs"]["s"]):
        raise ValueError(f"Corpus lacks required structures: {required}")
    passages, pairs = {}, defaultdict(list)
    for entity in cfg["entities"]:
        envelope = client.get("query", {
            "corpus": cfg["id"], "cqp": entity_cqp(entity),
            "start": 0, "end": cfg["hits_per_entity"] - 1,
            "default_context": "1 paragraph", "default_within": "sentence", "show": "word",
            "show_struct": "text_title,text_url,text_permalink,text__id",
            "sort": "random", "random_seed": seed,
        })
        groups = defaultdict(dict)
        for hit in envelope["response"].get("kwic", []):
            passage = passage_from_hit(hit, entity, cfg, envelope, seed)
            if passage:
                passages[passage["id"]] = passage
                # At most one passage per article per entity; no same-article hops.
                groups[passage["split"]].setdefault(passage["document_id"], passage)
        for split, documents in groups.items():
            candidates = list(itertools.combinations(sorted(documents), 2))
            random.Random(f"{seed}:{entity}:{split}").shuffle(candidates)
            for left, right in candidates[:cfg["pairs_per_entity_per_split"]]:
                d1, d2 = documents[left], documents[right]
                row = {"seed_id": fingerprint([entity, d1["id"], d2["id"]])[:24],
                       "bridge_entity": entity, "split": split}
                for index, doc in enumerate((d1, d2), 1):
                    row.update({f"d{index}_{key}": value for key, value in doc.items() if key not in {"entity", "split"}})
                pairs[split].append(row)
        print(f"{entity}: {sum(len(d) for d in groups.values())} usable articles", flush=True)
    # Equal generation budgets for the two training halves.
    half_size = min(len(pairs["slice0"]), len(pairs["slice1"]), cfg["max_pairs_per_training_slice"])
    if half_size < 1:
        raise ValueError("No disjoint training pairs. Increase hits_per_entity or add entities.")
    prepared = data_dir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for split in ("slice0", "slice1", "eval"):
        random.Random(f"{seed}:{split}").shuffle(pairs[split])
        pairs[split] = pairs[split][:cfg["max_eval_pairs"] if split == "eval" else half_size]
        if not pairs[split]:
            raise ValueError(f"No {split} pairs. Increase the source sample.")
        pd.DataFrame(pairs[split]).to_parquet(prepared / f"{split}.parquet", index=False)
    assert_disjoint(pairs)
    from .common import write_jsonl
    write_jsonl(prepared / "passages.jsonl", list(passages.values()))
    manifest = {
        "config": config, "corpus_info": corpus, "retrieved_at": info["retrieved_at"],
        "passages": len(passages), "pairs": {split: len(rows) for split, rows in pairs.items()},
        "seed_sha256": {split: fingerprint(rows) for split, rows in pairs.items()},
        "sampling": "Entity-conditioned random Korp hits; not a representative Wikipedia sample",
        "splitting": "45/45/10 article hash partition before pairing; equal-sized training seed slices",
    }
    write_json(prepared / "manifest.json", manifest)
    return manifest


def assert_disjoint(pairs: dict[str, list[dict]]) -> None:
    seen = {}
    for split, rows in pairs.items():
        for row in rows:
            if row["d1_document_id"] == row["d2_document_id"]:
                raise ValueError("A two-hop seed must use two distinct articles")
            for doc in (row["d1_document_id"], row["d2_document_id"]):
                if doc in seen and seen[doc] != split:
                    raise ValueError(f"Source document leaks between {seen[doc]} and {split}")
                seen[doc] = split
