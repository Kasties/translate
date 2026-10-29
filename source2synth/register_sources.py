"""Register-first source ingestion. Source text is never retokenized or modernized."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests

from .common import ROOT, fingerprint, normalize, write_json, write_jsonl
from .korp import KorpClient, document_split
from .nlu_sources import assert_source_disjoint
from .register_sampling import cached_embeddings, cluster_documents, diversity, embed_documents, select
from .sources import iter_hf_records, iter_local_records


# Sparv stores the whitespace around each token escaped: \s space, \n newline, \t tab.
SPARV_WHITESPACE = {"s": " ", "n": "\n", "t": "\t", "r": "\r"}
ATTRIBUTE = re.compile(r"[a-z][a-z0-9_]*")
KORP_PAGE = 200
KORP_SCAN_FACTOR = 20  # units read per requested document before reporting a shortfall
HPLT_DATA = "https://data.hplt-project.org"
# Top-level Turku web registers. MT (machine translated) is filtered on, never a register.
HPLT_REGISTERS = {"NA": "narrative", "IN": "informational", "OP": "opinion", "IP": "persuasion",
                  "HI": "how-to", "ID": "discussion", "SP": "spoken", "LY": "lyrical"}
# Function words of Danish/Norwegian with no Swedish homograph, and their Swedish
# counterparts. Language ID labels some Norwegian pages swe_Latn.
DANISH_NORWEGIAN = {"ikke", "jeg", "og", "hva", "hvordan", "hvorfor", "hvis", "også", "være", "kanskje",
                    "mye", "meg", "ble", "fra", "noe", "hvor", "disse"}
SWEDISH_ONLY = {"inte", "jag", "och", "vad", "hur", "varför", "också", "vara", "kanske", "mycket", "mig",
                "blev", "från", "något", "var", "denna", "dessa"}
# Explicit sexual words. "porr" is left out: news and debate about pornography use it.
EXPLICIT = re.compile(r"\b(?:kuk(?:en|ar|arna)?|fitt(?:a|an|or|orna)|knull\w*|halvstånd\w*|buttplug\w*|"
                      r"dildo\w*|avsugning\w*|runka\w*|sperma\w*|analsex\w*|kåt|kåta|kåtare|sexfilm\w*|"
                      r"sexleksak\w*|orgasm\w*|onani\w*)\b", re.IGNORECASE)


def sparv_whitespace(value: str | None) -> str:
    return re.sub(r"\\([sntr])", lambda m: SPARV_WHITESPACE[m.group(1)], value or "")


def token_word(token: dict) -> str:
    # Some corpora (e.g. ARBETSFORMEDLINGEN-2024) keep runs of whitespace as escaped tokens.
    word = token["word"]
    return sparv_whitespace(word) if re.fullmatch(r"(?:\\[sntr])+", word) else word


def korp_text(tokens: list[dict]) -> tuple[str, int]:
    """Rebuild original text from Sparv _head/_tail; count word boundaries with no whitespace.

    Such gaps mean a separator (usually a stripped line break) was lost when the
    corpus was built, which would glue two words together in a training target.
    """
    words = [token_word(token) for token in tokens]
    parts, gaps = [], 0
    for i, token in enumerate(tokens):
        parts += [sparv_whitespace(token.get("_head")), words[i], sparv_whitespace(token.get("_tail"))]
        if (i + 1 < len(tokens) and not token.get("_tail") and not tokens[i + 1].get("_head")
                and words[i][-1:].isalnum() and words[i + 1][:1].isalnum()):
            gaps += 1
    return "".join(parts).strip(), gaps


def attribute_name(name: str, value: str) -> str:
    # Attribute names are interpolated into CQP and the query; never pass raw config text.
    if not isinstance(value, str) or not ATTRIBUTE.fullmatch(value):
        raise ValueError(f"{name}: invalid Korp attribute name {value!r}")
    return value


def korp_cqp(spec: dict) -> str:
    """Match the first token of every unit, optionally restricted by date and attribute values."""
    conditions = []
    for key, attribute, operator in (("date_from", "text_datefrom", ">="), ("date_to", "text_dateto", "<=")):
        if spec.get(key) is not None:
            day = str(spec[key]).replace("-", "")
            if not re.fullmatch(r"\d{8}", day):
                raise ValueError(f"{spec['name']}: {key} must be YYYY-MM-DD")
            conditions.append(f"int(_.{attribute}) {operator} {day}")
    for attribute, value in spec.get("where", {}).items():
        literal = re.escape(str(value)).replace('"', '\\"')
        conditions.append(f'_.{attribute_name(spec["name"], attribute)} = "{literal}"')
    unit = attribute_name(spec["name"], spec["unit"])
    return f"<{unit}> [" + " & ".join(f"({c})" for c in conditions) + "]"


def korp_quotas(hits: dict, total: int, allocation: str) -> dict:
    """Largest-remainder split of a document budget over corpora."""
    if allocation not in {"proportional", "equal"}:
        raise ValueError("Korp allocation must be proportional or equal")
    weights = {c: (n if allocation == "proportional" else 1) for c, n in hits.items() if n > 0}
    if not weights:
        return {}
    exact = {c: total * w / sum(weights.values()) for c, w in weights.items()}
    quotas = {c: int(v) for c, v in exact.items()}
    for corpus in sorted(exact, key=lambda c: (quotas[c] - exact[c], c))[:total - sum(quotas.values())]:
        quotas[corpus] += 1
    return quotas


def assert_korp_coherent(client: KorpClient, corpus: str, name: str) -> None:
    """Refuse sentence-scrambled corpora; Korp scrambles e.g. its forum and news corpora.

    Read in corpus order, coherent documents form contiguous runs of sentences. A
    scrambled corpus changes text at nearly every sentence and revisits texts it left.
    """
    envelope = client.get("query", {"corpus": corpus, "cqp": "<sentence> []", "start": 0, "end": 99,
                                     "default_context": "1 sentence", "show": "word",
                                     "show_struct": "text__id"})
    ids = [hit.get("structs", {}).get("text__id") for hit in envelope["response"].get("kwic", [])]
    runs = [text for i, text in enumerate(ids) if i == 0 or text != ids[i - 1]]
    if len(runs) != len(set(runs)) or (len(ids) > 1 and (len(runs) - 1) / (len(ids) - 1) > 0.8):
        raise ValueError(f"{name}: {corpus} is sentence-scrambled in Korp and cannot supply coherent passages")


def korp_records(spec: dict, client: KorpClient, seed: int, rejected: Counter):
    """Seeded random units (texts, paragraphs or pages) from coherent Korp corpora.

    Korp's random sort is seeded server side, so a sample replays against the same
    corpus build. Units shorter than min_tokens are skipped while paging, which keeps
    the sample uniform over eligible units without slow CQP length patterns.
    """
    corpora = [c.upper() for c in spec["corpora"]]
    unit, group_by = spec["unit"], spec.get("group_by")
    fields = {key: attribute_name(spec["name"], value) for key, value in spec.get("attributes", {}).items()}
    if set(fields) - {"title", "url", "author", "date", "license"}:
        raise ValueError(f"{spec['name']}: attributes may map title, url, author, date and license")
    shown = sorted({"text__id", *fields.values(), *([attribute_name(spec["name"], group_by)] if group_by else [])})
    needed = {*shown, *spec.get("where", {})} | ({"text_datefrom", "text_dateto"}
                                                 if spec.get("date_from") or spec.get("date_to") else set())
    info = client.get("corpus_info", {"corpus": ",".join(corpora)})["response"]["corpora"]
    for corpus in corpora:
        attrs = info[corpus]["attrs"]
        if not {"_head", "_tail"} <= set(attrs["p"]):
            raise ValueError(f"{spec['name']}: {corpus} lacks _head/_tail whitespace, so its text cannot be rebuilt")
        if missing := sorted((needed | {unit}) - set(attrs["s"])):
            raise ValueError(f"{spec['name']}: {corpus} lacks structural attributes {missing}")
        assert_korp_coherent(client, corpus, spec["name"])
    params = {"cqp": korp_cqp(spec), "default_context": f"1 {unit}", "default_within": unit,
              "show": "word,_head,_tail", "show_struct": ",".join(shown),
              "sort": "random", "random_seed": seed}
    hits = client.get("query", {**params, "corpus": ",".join(corpora), "start": 0, "end": 0})
    hits = hits["response"].get("corpus_hits", {})
    minimum = spec.get("min_tokens", 1)
    for corpus, quota in korp_quotas(hits, spec["max_documents"], spec.get("allocation", "proportional")).items():
        found = 0
        for start in range(0, min(hits[corpus], quota * KORP_SCAN_FACTOR), KORP_PAGE):
            envelope = client.get("query", {**params, "corpus": corpus, "start": start,
                                            "end": start + KORP_PAGE - 1})
            for hit in envelope["response"].get("kwic", []):
                if len(hit["tokens"]) < minimum:
                    continue
                text, gaps = korp_text(hit["tokens"])
                if gaps:
                    rejected["lost_whitespace"] += 1
                    continue
                structs, position = hit.get("structs", {}), hit["match"]["position"]
                # An empty group value must not pool every such unit into one split.
                group = structs.get(group_by) if group_by else None
                yield {**{key: structs.get(attribute) for key, attribute in fields.items()},
                       "id": f"{corpus}:{position}", "text": text,
                       "group_id": f"{corpus}:{group}" if group else None,
                       "corpus": corpus, "corpus_position": position,
                       "corpus_text_id": structs.get("text__id"),
                       "corpus_updated": info[corpus]["info"].get("Updated"),
                       "text_transform": "Korp tokens joined with Sparv _head/_tail whitespace",
                       "query_url": envelope["response_url"], "retrieved_at": envelope["retrieved_at"],
                       "response_sha256": envelope["response_sha256"]}
                found += 1
                if found == quota:
                    break
            if found == quota:
                break


def hplt_prefix(url: str, size: int, cache: Path, offline: bool) -> tuple[Path, str]:
    """Cache the first `size` bytes of a shard; return the file and its sha256.

    HPLT shards are single zstd frames, so a prefix is the only cheap read. Each shard
    is sorted by WDS, so a prefix holds that shard's best-formed documents.
    """
    path = cache / f"{fingerprint([url, size])[:24]}.jsonl.zst"
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Missing cached HPLT prefix of {url}: {path}")
        response = requests.get(url, headers={"Range": f"bytes=0-{size - 1}"}, stream=True, timeout=(10, 90))
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
        partial.rename(path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def hplt_documents(path: Path):
    """Complete JSON lines of a truncated zstd prefix; the cut-off last line is dropped."""
    import pyarrow as pa

    stream, buffer = pa.CompressedInputStream(pa.OSFile(str(path)), "zstd"), b""
    try:
        while chunk := stream.read(1 << 20):
            *lines, buffer = (buffer + chunk).split(b"\n")
            yield from (json.loads(line) for line in lines if line.strip())
    except OSError as error:
        if "Truncated" not in str(error):
            raise


def hplt_register(labels: dict) -> str:
    return HPLT_REGISTERS[max(HPLT_REGISTERS, key=lambda code: labels.get(code, 0.0))]


def hplt_site_reason(host: str, spec: dict) -> str | None:
    """Why a host is not a source of native Swedish pages, or None.

    Content farms sit on blocked domains. Swedish text on another country's TLD
    is spam or translated, except Blogspot's regional mirrors of Swedish blogs. A
    language subdomain (sv.example.com) off the Swedish-language TLDs is a
    translated copy of a foreign site.
    """
    labels, dotted = host.split("."), f".{host}"
    if dotted.endswith(tuple(f".{d}" for d in spec.get("blocked_domains", []))):
        return "hplt_blocked_domain"
    native = set(spec.get("native_tlds", []))
    if not native or labels[-1] in native:
        return None
    if len(labels[-1]) == 2 and not set(labels) & set(spec.get("foreign_tld_exempt", [])):
        return "hplt_foreign_tld"
    if (labels[0] in spec.get("language_subdomains", [])
            and not dotted.endswith(tuple(f".{d}" for d in spec.get("language_subdomain_exempt", [])))):
        return "hplt_language_subdomain"
    return None


def hplt_records(spec: dict, cache: Path, offline: bool, rejected: Counter):
    """Documents from the prefix of every shard of one HPLT 3.0 language, round robin.

    Pages the register classifier scores as machine translated are dropped, as are
    pages past a per-host cap. A host is the split group: one site never spans splits.
    """
    base = f"{HPLT_DATA}/{spec.get('release', 'three/sorted')}"
    shards = spec.get("shards") or sorted(requests.get(f"{base}/{spec['language']}.map", timeout=(10, 90))
                                          .text.split(), key=lambda url: url.rsplit("/", 1)[-1])
    shards = [s if s.startswith("https://") else f"{base}/{spec['language']}/{s}" for s in shards]
    streams = []
    for url in shards:
        path, digest = hplt_prefix(url, int(spec.get("prefix_mb", 32) * 2**20), cache, offline)
        streams.append(((url, digest), hplt_documents(path)))
    hosts = Counter()
    while streams:
        live = []
        for (url, digest), documents in streams:
            for doc in documents:
                labels, host = doc.get("web-register") or {}, urlsplit(doc["u"]).hostname or ""
                if not labels:
                    rejected["hplt_unlabelled"] += 1
                elif reason := hplt_site_reason(host, spec):
                    rejected[reason] += 1
                elif labels.get("MT", 1.0) >= spec.get("max_mt", 0.1):
                    rejected["hplt_machine_translated"] += 1
                elif hosts[host] >= spec.get("max_per_host", 5):
                    rejected["hplt_host_cap"] += 1
                else:
                    hosts[host] += 1
                    yield {"id": doc["id"], "text": doc["text"], "url": doc["u"], "date": doc["ts"],
                           "group_id": host, "register": hplt_register(labels),
                           "hplt_mt": labels["MT"], "hplt_wds": doc["doc_scores"][0],
                           "hplt_crawl": doc["crawl_id"], "warc_file": doc["f"], "warc_offset": doc["o"],
                           "query_url": url, "response_sha256": digest,
                           "text_transform": "none (HPLT Trafilatura extraction)"}
                    live.append(((url, digest), documents))
                    break
        streams = live


def records(spec: dict, client: KorpClient, seed: int, rejected: Counter):
    kind = spec["kind"]
    if kind == "korp":
        yield from korp_records(spec, client, seed, rejected)
    elif kind == "hplt":
        yield from hplt_records(spec, client.cache.parent / "hplt", client.offline, rejected)
    elif kind in {"local", "hf"}:
        iterator = (iter_local_records(spec["path"]) if kind == "local" else
                    iter_hf_records(spec["dataset"], spec.get("subset"), spec.get("split", "train"),
                                    spec.get("revision")))
        for row, origin, digest in islice(iterator, spec["max_documents"]):
            fields = spec.get("fields", {})
            mapped = {key: row.get(fields.get(key, key))
                      for key in ("id", "text", "title", "url", "group_id", "author", "date")}
            yield {**row, **mapped, "query_url": origin, "response_sha256": digest,
                   "url": mapped["url"] or origin, "text_transform": "none"}
    else:
        raise ValueError(f"Unknown register source kind {kind!r}; use korp, hplt, local or hf")


def native_passages(text: str, minimum: int, maximum: int):
    """Yield exact substrings and offsets; split long blocks only at sentence ends."""
    for match in re.finditer(r"[^\n]+", text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        block = text[start:end]
        if len(block.split()) <= maximum:
            if len(block.split()) >= minimum:
                yield block, start, end
            continue
        chunk_start = start
        last_end = start
        for sentence in re.finditer(r".*?[.!?](?=\s|$)|.+$", block):
            sentence_end = start + sentence.end()
            if len(text[chunk_start:sentence_end].split()) > maximum:
                if minimum <= len(text[chunk_start:last_end].split()) <= maximum:
                    yield text[chunk_start:last_end], chunk_start, last_end
                chunk_start = start + sentence.start()
                while chunk_start < sentence_end and text[chunk_start].isspace():
                    chunk_start += 1
            last_end = sentence_end
        if minimum <= len(text[chunk_start:last_end].split()) <= maximum:
            yield text[chunk_start:last_end], chunk_start, last_end


def joined_passages(text: str, minimum: int, maximum: int, min_block_words: int, end_on_colon: bool = True):
    """Join consecutive paragraphs into exact substrings of minimum..maximum words.

    A paragraph shorter than min_block_words (a menu item, caption or button) ends
    the run, so joined text never spans page furniture. So does any text between two
    paragraphs that native_passages dropped, such as one sentence longer than maximum.
    Without end_on_colon a run cannot end on a lead-in ("Så här gör du:"); it grows
    into the next paragraph or is dropped.
    """
    run_start = run_end = None
    words = 0
    for block, start, end in native_passages(text, 1, maximum):
        size = len(block.split())
        if run_start is not None and (size < min_block_words or words + size > maximum
                                      or text[run_end:start].strip()):
            run_start, words = None, 0
        if size < min_block_words:
            continue
        if run_start is None:
            run_start = start
        run_end, words = end, words + size
        if words >= minimum and (end_on_colon or not block.rstrip().endswith(":")):
            yield text[run_start:run_end], run_start, run_end
            run_start, words = None, 0


def quality_reason(text: str, settings: dict) -> str | None:
    words = re.findall(r"\w+", text.casefold())
    if not words or sum(ch.isalpha() for ch in text) / max(1, len(text)) < 0.55:
        return "non_prose"
    if "�" in text or re.search(r"<[^>]+>|https?://|\S+@\S+", text):
        return "markup_or_contact"
    if len(set(words)) / len(words) < settings.get("min_unique_word_ratio", 0.25):
        return "repetitive"
    if settings.get("swedish_markers", True):
        markers = {"och", "att", "det", "som", "en", "ett", "är", "på", "för", "med", "har",
                   "inte", "jag", "han", "hon", "den", "de", "af", "icke", "hvad", "äro"}
        if len(set(words) & markers) < 2:
            return "no_swedish_markers"
    limit = settings.get("max_explicit_terms")
    if limit is not None and len(EXPLICIT.findall(text)) > limit:
        return "explicit"
    if settings.get("reject_danish_norwegian"):
        theirs = sum(word in DANISH_NORWEGIAN for word in words)
        if theirs >= 3 and theirs > sum(word in SWEDISH_ONLY for word in words):
            return "danish_or_norwegian"
    limit = settings.get("max_truncated_lines")
    if limit is not None and sum(line.rstrip().endswith(("...", "…")) for line in text.splitlines()) >= limit:
        return "truncated_lines"
    return None


def prepare_registers(config: dict, data_dir: Path, cache: Path | None = None,
                      offline: bool = False, embedder=None) -> dict:
    cfg, seed = config["registers"], config["random_seed"]
    cache = cache or ROOT / "data/register-cache"
    client = KorpClient(cfg["korp_api"], cache / "korp", offline)
    rows, rejected, raw_docs = [], Counter(), []
    seen_documents, seen_texts = set(), {}
    for spec in cfg["sources"]:
        if not spec.get("enabled", True):
            continue
        # HPLT labels each document's register; other kinds take the source's.
        if (not spec.get("register") and spec.get("kind") != "hplt") or spec.get("max_documents", 0) < 1:
            raise ValueError("Every source needs register and positive max_documents")
        if spec.get("coherent") is not True:
            raise ValueError(f"{spec['name']}: require coherent: true; shuffled sentence corpora are unsuitable")
        if not spec.get("license") or not spec.get("license_url"):
            raise ValueError(f"{spec['name']}: license and license_url are required")
        count = 0
        for record in islice(records(spec, client, seed, rejected), spec["max_documents"]):
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                rejected["missing_text"] += 1
                continue
            # Erotica also has milder passages; a page this explicit gives none of them.
            page_limit = cfg.get("quality", {}).get("max_explicit_terms_per_page")
            if page_limit is not None and len(EXPLICIT.findall(text)) > page_limit:
                rejected["explicit_page"] += 1
                continue
            raw_id = str(record.get("id") or fingerprint(text))
            document_id = fingerprint([spec.get("namespace", spec["name"]), raw_id])[:24]
            if document_id in seen_documents:
                continue
            seen_documents.add(document_id)
            # Group books, threads and parliamentary protocols before selecting passages.
            group_id = fingerprint([spec.get("namespace", spec["name"]),
                                    str(record.get("group_id") or raw_id)])[:24]
            split = document_split(group_id, seed)
            doc = {**record, "id": document_id, "group_id": group_id, "split": split,
                   "source_name": spec["name"],
                   "register": record["register"] if spec["kind"] == "hplt" else spec["register"],
                   "era": spec.get("era", "contemporary"),
                   "license_as_listed": record.get("license") or spec.get("license"),
                   "license_url": record.get("license_url") or spec.get("license_url")}
            raw_docs.append(doc)
            passages = (joined_passages(text, cfg["min_words"], cfg["max_words"], cfg.get("min_block_words", 8),
                                        cfg.get("end_on_colon", True))
                        if cfg.get("join_paragraphs") else native_passages(text, cfg["min_words"], cfg["max_words"]))
            for passage, start, end in passages:
                reason = quality_reason(passage, cfg.get("quality", {}))
                if reason:
                    rejected[reason] += 1
                    continue
                digest = fingerprint(normalize(passage))
                if digest in seen_texts:
                    rejected["duplicate_passage"] += 1
                    continue
                seen_texts[digest] = split
                rows.append({**{k: v for k, v in doc.items() if k != "text"},
                             "id": fingerprint([document_id, start, end])[:24],
                             "document_id": document_id, "source_topic": "unassigned",
                             "text": passage, "char_start": start, "char_end": end,
                             "document_sha256": fingerprint(text), "native_text": True})
                count += 1
        print(f"Register source {spec['name']}: {count} eligible passages", flush=True)
    if not rows:
        raise ValueError("No register passages survived source/quality checks")
    docs = {r["document_id"] for r in rows}
    raw_docs = sorted((d for d in raw_docs if d["id"] in docs), key=lambda d: d["id"])
    vectors = cached_embeddings([d["text"] for d in raw_docs], cfg["embedding"], cache / "embeddings",
                                embedder or embed_documents)
    vectors_by_doc = dict(zip([d["id"] for d in raw_docs], vectors))
    # Fit separately per split: held-out text never influences training centroids.
    labels = {}
    for split in ("slice0", "slice1", "eval"):
        ids = [d["id"] for d in raw_docs if d["split"] == split]
        if ids:
            labels.update(zip(ids, cluster_documents([vectors_by_doc[d] for d in ids],
                                                     cfg["sampling"]["clusters"], seed)))
    for row in rows:
        row["cluster"] = f"{row['split']}:{labels[row['document_id']]}"
    settings = cfg["sampling"]
    selected, comparison = [], {}
    for split in ("slice0", "slice1", "eval"):
        pool = [r for r in rows if r["split"] == split]
        for strategy in ("balanced", "random", "keyword"):
            sample = select(pool, strategy, settings["per_split"], seed, config["nlu"]["topics"])
            comparison[f"{split}/{strategy}"] = diversity(sample, vectors_by_doc)
            comparison[f"{split}/{strategy}"]["shortfall"] = max(0, settings["per_split"] - len(sample))
            if strategy == settings["strategy"]:
                selected.extend(sample)
    if settings["strategy"] not in {"balanced", "random", "keyword"}:
        raise ValueError("Unknown register sampling strategy")
    assert_source_disjoint(selected)
    out = data_dir / "nlu"
    write_jsonl(out / "documents.jsonl", raw_docs)
    write_jsonl(out / "pool.jsonl", rows)
    write_jsonl(out / "passages.jsonl", selected)
    np.savez_compressed(out / "embeddings.npz", ids=np.array(list(vectors_by_doc)), vectors=vectors)
    embedding_sha = hashlib.sha256(np.asarray(vectors, dtype=np.float32).tobytes()).hexdigest()
    manifest = {"source": "registers", "passages": len(selected), "pool_passages": len(rows),
                "documents": len(raw_docs), "sha256": fingerprint(selected),
                "pool_sha256": fingerprint([rows, cfg["embedding"], embedding_sha]),
                "embeddings_sha256": embedding_sha, "config": cfg, "random_seed": seed,
                "korp_corpora": {d["corpus"]: d["corpus_updated"] for d in raw_docs if "corpus_updated" in d},
                "rejected": dict(rejected), "comparison": comparison,
                "empty_registers": sorted({s["register"] for s in cfg["sources"]
                                           if s.get("enabled", True) and s["kind"] != "hplt"}
                                          - {r["register"] for r in rows}),
                "embedding_kind": "test lexical hashing" if cfg["embedding"]["model"] == "hash-test"
                                  else "multilingual document embeddings",
                "notes": [("Comparisons use one quality-filtered pool; keyword is a Wikipedia-only proxy "
                           "for the legacy sampler, not a rerun of Korp queries."),
                          ("Metrics describe source coverage, not idiomatic quality. No token efficiency "
                           "claim is made until generation and review usage has been recorded."),
                          ("Korp samples use the server's seeded random sort: exact replay needs the same "
                           "corpus builds (korp_corpora lists their Updated dates) or the saved cache.")]}
    write_json(out / "manifest.json", manifest)
    return manifest
