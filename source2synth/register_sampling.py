"""Document embeddings, deterministic register/cluster sampling and honest cost metrics."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from .common import fingerprint
from .sources import topic_matchers


def unit_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a nonempty finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embedding has zero norm")
    return values / norms


def embed_documents(documents: list[str], settings: dict):
    """Embed bounded chunks, then mean-pool per document; never truncate a whole book."""
    if settings["model"] == "hash-test":
        # Explicit offline plumbing test, never represented as semantic embeddings.
        values = np.zeros((len(documents), 256), dtype=np.float32)
        for i, text in enumerate(documents):
            for word in re.findall(r"\w+", text.casefold()):
                values[i, int(fingerprint(word)[:8], 16) % 256] += 1
        return unit_vectors(values)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ValueError("Install register embeddings: uv sync --extra registers") from error
    model = SentenceTransformer(settings["model"], revision=settings.get("revision"),
                                device=settings.get("device", "cpu"))
    texts, owners = [], []
    for i, text in enumerate(documents):
        words = text.split()
        chunks = [" ".join(words[j:j + 160]) for j in range(0, len(words), 160)]
        limit = settings.get("max_chunks_per_document", 16)
        indices = np.linspace(0, len(chunks) - 1, min(len(chunks), limit), dtype=int)
        for index in indices:
            texts.append(settings.get("prefix", "passage: ") + chunks[index])
            owners.append(i)
    encoded = model.encode(texts, batch_size=settings.get("batch_size", 32),
                           normalize_embeddings=True, show_progress_bar=True)
    pooled = np.zeros((len(documents), encoded.shape[1]), dtype=np.float32)
    for owner, vector in zip(owners, encoded):
        pooled[owner] += vector
    return unit_vectors(pooled)


def cached_embeddings(texts: list[str], settings: dict, cache_dir: Path, embed=embed_documents) -> np.ndarray:
    """Vectors for texts in input order; embeds only texts this cache has not seen.

    One file per embedding settings, keyed by text, so every build that shares the
    cache directory reuses the others' vectors.
    """
    path = cache_dir / f"{fingerprint(settings)[:16]}.npz"
    keys = [fingerprint(text) for text in texts]
    stored_keys, stored = [], None
    if path.exists():
        with np.load(path) as saved:
            stored_keys, stored = saved["keys"].tolist(), saved["vectors"]
    index = {key: i for i, key in enumerate(stored_keys)}
    missing = {key: text for key, text in zip(keys, texts) if key not in index}
    if missing:
        fresh = np.asarray(embed(list(missing.values()), settings), dtype=np.float32)
        stored = fresh if stored is None else np.vstack([stored, fresh])
        index.update((key, len(stored_keys) + i) for i, key in enumerate(missing))
        stored_keys += list(missing)
        cache_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part.npz")
        np.savez(partial, keys=np.array(stored_keys), vectors=stored)
        partial.replace(path)
    return stored[[index[key] for key in keys]]


def cluster_documents(vectors, count: int, seed: int):
    """Small bounded spherical k-means, with seeded k-means++ initialization."""
    values = unit_vectors(vectors)
    if count < 1:
        raise ValueError("clusters must be positive")
    rng = np.random.default_rng(seed)
    centers = [values[int(rng.integers(len(values)))]]
    for _ in range(1, min(count, len(values))):
        distances = np.maximum(0, 1 - (values @ np.array(centers).T).max(axis=1)) ** 2
        if distances.sum() < 1e-8:
            break
        centers.append(values[rng.choice(len(values), p=distances / distances.sum())])
    centers = np.array(centers)
    labels = np.full(len(values), -1)
    for _ in range(30):
        updated = (values @ centers.T).argmax(axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for i in range(len(centers)):
            if (labels == i).any():
                center = values[labels == i].mean(axis=0)
                norm = np.linalg.norm(center)
                if norm:
                    centers[i] = center / norm
    return labels.tolist()


def round_robin(groups: dict) -> list[dict]:
    queues = [deque(rows) for _, rows in sorted(groups.items()) if rows]
    result = []
    while queues:
        next_queues = []
        for queue in queues:
            result.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return result


def select(rows: list[dict], strategy: str, count: int, seed: int, topics: dict) -> list[dict]:
    """One passage/document. Balance registers first, clusters within each register.

    quality is balanced with rows ordered by their quality_score instead of at random.
    """
    if count < 1:
        raise ValueError("sample size must be positive")
    ordered = sorted(rows, key=lambda row: fingerprint([seed, row["id"]]))
    if strategy == "quality":
        # A document is represented by its best-scoring passage, not a random one.
        ordered.sort(key=lambda row: -row["quality_score"])
    if strategy != "keyword":
        # Pick a representative first, then rank documents. Long documents must
        # not get more lottery tickets merely because they contain more passages.
        documents = {}
        for row in ordered:
            documents.setdefault(row["document_id"], row)
        ordered = sorted(documents.values(), key=lambda row: fingerprint([seed, row["document_id"]]))
    if strategy == "keyword":
        # Reproduce the old whole-word, Wikipedia-only topic-conditioned baseline.
        groups = defaultdict(list)
        for row in ordered:
            if row["register"] != "encyclopedia":
                continue
            for topic, _, pattern in topic_matchers(topics):
                if pattern.search(row["text"]):
                    groups[topic].append(row)
                    break
        ordered = round_robin(groups)
    elif strategy in {"balanced", "quality"}:
        registers = defaultdict(lambda: defaultdict(list))
        for row in ordered:
            registers[row["register"]][row["cluster"]].append(row)
        if strategy == "quality":
            # Same register and cluster balance, best-scoring document first in each cluster.
            for clusters in registers.values():
                for members in clusters.values():
                    members.sort(key=lambda row: -row["quality_score"])
        ordered = round_robin({register: round_robin(clusters)
                               for register, clusters in registers.items()})
    elif strategy != "random":
        raise ValueError("sampling strategy must be balanced, quality, random or keyword")
    result, seen = [], set()
    for row in ordered:
        if row["document_id"] not in seen:
            result.append(row)
            seen.add(row["document_id"])
        if len(result) == count:
            break
    return result


def entropy(values) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values()) if total else 0.0


def diversity(rows: list[dict], vectors_by_document: dict | None = None) -> dict:
    documents = {r["document_id"]: r for r in rows}
    distance = None
    if vectors_by_document is not None and len(documents) > 1:
        values = unit_vectors([vectors_by_document[d] for d in documents])
        n = len(values)
        # Exact mean pairwise cosine distance without a quadratic distance matrix.
        distance = float(1 - (np.square(values.sum(axis=0)).sum() - n) / (n * (n - 1)))
    words = [re.findall(r"\w+", r["text"].casefold()) for r in rows]
    bigrams = [tuple(ws[i:i + 2]) for ws in words for i in range(len(ws) - 1)]
    return {"rows": len(rows), "documents": len(documents),
            "by_register": dict(Counter(r["register"] for r in rows)),
            "clusters": len({r["cluster"] for r in rows}),
            "register_entropy_bits": entropy(r["register"] for r in rows),
            "cluster_entropy_bits": entropy(r["cluster"] for r in rows),
            "mean_document_cosine_distance": distance,
            "distinct_bigrams": len(set(bigrams)),
            "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0}


def efficiency(metrics: dict, calls: list[dict]) -> dict:
    """Count every attempted call, including rejected examples and semantic review."""
    known = [c.get("usage", {}).get("completion_tokens") for c in calls]
    complete = bool(calls) and all(type(n) is int and n >= 0 for n in known)
    total = sum(n for n in known if type(n) is int and n >= 0)
    return {"model_calls": len(calls), "usage_complete": complete,
            "known_completion_tokens": total,
            "completion_tokens": total if complete else None,
            "per_1000_generated_tokens": {
                k: metrics[k] * 1000 / total if complete and total else None
                for k in ("documents", "clusters", "distinct_bigrams")}}
