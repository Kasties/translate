# Register-first Swedish instruction data

`register-native` adds document sampling by register and semantic cluster and
puts original human Swedish in the assistant response for most examples.
The existing keyword-conditioned NLU and MHQA experiments remain available.

```bash
uv sync --locked --extra registers
./pipeline run register-native --only sources
./pipeline run register-native --from generate
```

The second command downloads bounded source samples and a multilingual embedding
model; it does not start a generator. The third command generates, curates, trains
and evaluates using the configured model servers. Use `--only generate` to stop
after exporting data. For an existing server, set `generator.backend=external`
and `generator.endpoint=http://127.0.0.1:8080/v1`, with its served model ID.

The task defaults are **60% instruction backtranslation, 25% rewriting,
10% grounded responses, and 5% legacy NLU**. These are editable starting weights,
not experimentally established optimal proportions. A 1,000-row build requests
600/250/100/50 accepted examples. `require_complete: true` prevents training on an
unfinished mix. Source preparation reports actual availability; lower the target
or increase source collection when a baseline has too few eligible documents.

### Register shares

`instruction_data.register_weights` sets each source register's share of the
accepted rows: **40% debate, 40% encyclopedia, 20% job ads** by default. The shares
apply within every task, so a 1,000-row build has 240/240/120 backtranslation rows
and 400/400/200 rows overall. Weights are relative and normalized, so set all of
them to get exact percentages. Every register in the source pool needs an entry (0
excludes it); enabling a new source, such as Litteraturbanken's `fiction`, stops the
build until it has one. `null` gives equal shares.

The shares are applied at generation time, not when sources are sampled. A changed
mix is a new experiment name (or the same override under `datagen.overrides` in an
experiment file):

```bash
./pipeline run register-native --name native-jobs10 \
  --set 'datagen.overrides.instruction_data.register_weights={debate: 0.45, encyclopedia: 0.45, job-ad: 0.10}'
```

On the 24 GB RTX 4090 add `--profile rtx4090-pilot` (see [the pipeline guide](pipeline.md)).
Its sources stage replays the shared Korp cache (`data/register-cache/korp`, no new
downloads) and recomputes the embeddings, which gives the same pool for the same
seed and source settings. The build stops before generating if a register's
backtranslation or rewrite quota exceeds the distinct passages it has in the split.
`report.json` lists `register_quotas` and the accepted `by_register` counts.

## Sources and exact targets

Settings live under `registers` and `instruction_data` in
[`configs/source2synth.yaml`](../configs/source2synth.yaml). Override them under
`datagen.overrides` in an experiment. YAML source lists replace the full list.

All built-in sources are sampled from Språkbanken's
[Korp API](https://ws.spraakbanken.gu.se/ws/korp/v8) (`kind: korp`); nothing is
fetched from the original websites.

| Source | Status | Register and handling |
| --- | --- | --- |
| Riksdag debates (`RD-PROT`) | Enabled | Debate. Seeded random paragraphs of chamber protocols from 2010 to June 2022 (older protocols are OCR scans), at least 50 tokens. Paragraphs of one protocol stay in one split. |
| Swedish Wikipedia (`WIKIPEDIA-SV`) | Enabled | Encyclopedia. Seeded random whole articles of at least 50 tokens, without keywords. Article URLs are retained. |
| Job ads (`ARBETSFORMEDLINGEN-2024`) | Enabled | Job ad. Seeded random Platsbanken ads (July 2023–2024) of at least 50 tokens; headline, employer and publication date are kept. One employer's ads stay in one split, since ads are often templated. Språkbanken has no resource page for this corpus, so the license recorded is that of [Arbetsförmedlingen's open job-ad data](https://data.arbetsformedlingen.se/dataset/job-ads/) (CC0). Passages with e-mail addresses or URLs are dropped; contact names are not. |
| Litteraturbanken (`LB-OPEN`) | Configured, opt in | Historical fiction. Random pages of transcribed e-texts (`text_mediatype: etext`, not OCR facsimiles), each with its work's own `text_license`. Pages keep printed line breaks, so few lines reach 40 words until lines are joined. |
| Forums (Flashback, Familjeliv) | Local coherent-export importer, opt in | Forum. Korp's forum corpora are sentence-scrambled, so they are refused; supply a coherent export with its license and thread `group_id`. |
| Opinion writing / everyday advice | Local or Hugging Face importer | Add separate source entries and registers with their actual rights metadata. |
| HPLT 3.0 web (`swe_Latn`) | `register-hplt` experiment | Web pages of every register. Each page's register comes from HPLT's own Turku register scores; pages scored as machine translated are dropped. See [HPLT web text](#hplt-web-text). |

A Korp source names one or more `corpora`, the structural `unit` that becomes one
document (`text`, `paragraph`, `page`, …), and optionally `min_tokens`,
`date_from`/`date_to` (quoted `"YYYY-MM-DD"`, matched against `text_datefrom` and
`text_dateto`), exact `where` values, a `group_by` attribute that keeps related
units in one split, `attributes` mapping `title`, `url`, `author`, `date` and
`license` to Korp structural attributes, and `allocation: proportional | equal`
for splitting `max_documents` over several corpora. Only the mapped attributes are
requested, so forum usernames are never downloaded.

Sampling uses Korp's server-side `sort=random` with the experiment seed. Units
below `min_tokens` are skipped while paging; this samples uniformly from eligible
units and avoids slow CQP length patterns on large corpora. Before sampling, each
corpus is checked for:

- **Whitespace**: Sparv's `_head`/`_tail` attributes must exist. Older corpora
  without them (e.g. `GP*`, `BLOGGMIX*`, `ROM*`) would need re-spacing and are refused.
- **Coherence**: in corpus order, coherent documents form contiguous runs of
  sentences. Korp scrambles its forum and news corpora (Flashback, Familjeliv,
  `SVT-*`, `ATTASIDOR`): there consecutive sentences come from different texts,
  and such corpora are refused.

Documents in which a word boundary has no whitespace (a separator lost when the
corpus was built, which would glue two words together) are skipped and counted as
`lost_whitespace`; this is about 10% of job ads and rare elsewhere. Whitespace that
a corpus stores as its own escaped token (`\n\n`, `\s`) is decoded as whitespace.

Språkbanken lists [RD-PROT](https://sprakbanken.se/resurser/rd-prot),
[WIKIPEDIA-SV](https://sprakbanken.se/resurser/wikipedia-sv) and
[Litteraturbanken: fria verk](https://sprakbanken.se/resurser/lb-open) under
CC-BY-4.0; the Wikipedia article text itself is CC BY-SA. Keep attribution and
applicable share-alike terms when distributing the resulting data. These catalog
details were checked on 2026-10-03.

### HPLT web text

`kind: hplt` reads [HPLT 3.0](https://huggingface.co/datasets/HPLT/HPLT3.0), a
cleaned Common Crawl and Internet Archive collection with about 98 million Swedish
pages. Every page carries scores from a multilingual web register classifier. One of
the classes is MT, machine translated, which flags the translated spam (pharmacy,
casino and listicle pages on `.se` domains) that URL filters miss.

```bash
./pipeline run register-hplt --only sources
```

- **Machine translation:** pages with `MT >= max_mt` (default 0.1) are dropped and
  counted as `hplt_machine_translated`. In a 2,000-page sample, pages under 0.1
  read as native Swedish and pages over 0.7 were spam.
- **Register:** a kept page gets its highest-scoring top-level register:
  `informational`, `narrative`, `opinion`, `persuasion`, `discussion`, `how-to`,
  `spoken` or `lyrical`. `instruction_data.register_weights` needs an entry for each.
- **Reading:** each shard is one zstd frame sorted by document quality (WDS), so it
  can only be read from the start. The source caches the first `prefix_mb` of every
  shard under `data/register-cache/hplt/` and takes pages round robin over shards,
  so every quality bin (5 to 10) contributes. Pages inside a bin are not a random
  sample of that bin.
- **Sites:** at most `max_per_host` pages per host, and a host is a split group, so
  one site's pages never appear in both training and evaluation.
- **Content farms:** `blocked_domains` drops hosts by suffix: cheap TLDs (`xyz`,
  `fun`, `pw`, …) and `netlify.app` / `web.app`, whose random subdomains get past
  the host cap. In the first full run these pages copied Swedish news or were
  machine translated with MT scores under 0.1. Counted as `hplt_blocked_domain`.
- **Foreign sites:** with `native_tlds` set, a page on another country's TLD is
  dropped (`hplt_foreign_tld`) unless a host label is in `foreign_tld_exempt`
  (`blogspot`, whose regional mirrors such as `name.blogspot.no` serve Swedish
  blogs), and a language subdomain such as `sv.example.com` off those TLDs is
  dropped as a translated copy of a foreign site (`hplt_language_subdomain`),
  except hosts under `language_subdomain_exempt` (Wikipedia, Wikisource).
- **Norwegian and Danish:** language ID labels some Norwegian pages Swedish.
  `registers.quality.reject_danish_norwegian` drops a passage with at least three
  Danish/Norwegian-only function words (`ikke`, `jeg`, `og`, …) and more of them
  than Swedish ones (`danish_or_norwegian`). Off by default.
- **Adult content:** `registers.quality.max_explicit_terms` drops a passage with
  more than that many explicit sexual words (`explicit`), and
  `max_explicit_terms_per_page` drops a whole page (`explicit_page`), which also
  catches erotica's milder passages. "porr" is not on the list, since news and
  debate use it. Both are off by default, so Korp sources are unchanged.
- **Passage length:** `register-hplt` sets `join_paragraphs: true` with 150 to 400
  words, so a passage is several consecutive paragraphs of one page rather than
  one paragraph (median 62 words before, 176 after). A paragraph under
  `min_block_words` (menus, captions, buttons) ends a run. Off by default.
- **Why the MT cutoff stays at 0.1:** about 90% of kept pages score between 0.05
  and 0.1, and read samples from the bottom and top of that band were about as
  often translated (2 and 3 of 15). A cutoff of 0.05 would keep 10% of pages and
  almost no opinion or discussion pages.
- **Rights:** HPLT licenses its packaging under CC0; page text keeps its owners'
  rights under [HPLT's terms and takedown policy](https://huggingface.co/datasets/HPLT/HPLT3.0#terms-of-use-and-takedown).

To import an authorized local collection:

```yaml
datagen:
  overrides:
    registers:
      sources:
        - name: my-forum-export
          namespace: my-forum
          kind: local
          register: forum
          era: contemporary
          path: data/register-input/forums.jsonl
          max_documents: 2000
          coherent: true
          license: CC-BY-4.0  # use the actual resource's terms
          license_url: https://example.org/resource-license
```

Each JSONL record has `id`, `text`, `title`, `url`, and optionally `author`,
`date`, and `group_id` (the thread/work ID). `fields` maps these names when needed.
Plain `.txt`/`.md`, JSON, CSV and Parquet use the existing local loader. For Hub
datasets use `kind: hf`, `dataset`, `subset`, `split`, and `revision` instead of
`path`; license/register/coherence settings still apply.

`documents.jsonl` saves each document's full text, and every passage records its
character offsets, document hash, source response hash, license and URL. Korp
documents also record the corpus, corpus position and text ID. Local plain text is
preserved. Korp text is rebuilt from the tokens and Sparv's recorded whitespace,
recorded as `text_transform`, so no spaces are inserted around punctuation.
Subsequent passage selection and training export preserve punctuation, diacritics,
spelling and internal spaces.

## Sampling and comparisons

1. Apply bounded ingestion, paragraph/sentence length limits, simple Swedish/prose
   heuristics, contact/markup filtering, and exact normalized passage deduplication.
2. Assign 45/45/10 group-hash splits. Paragraphs from a document and documents with
   the same `group_id` cannot cross splits. Near-duplicate editions still need a
   shared group ID; exact deduplication is not a near-duplicate detector.
3. Embed document chunks with `intfloat/multilingual-e5-small` on CPU. Mean-pool
   up to 16 evenly spaced 160-word chunks per document. Save the actual vectors.
4. Fit seeded spherical k-means separately in each split. Training centroids do
   not depend on held-out documents. Select one representative passage per
   document, round-robin across registers and then their occupied clusters.
5. Independently review generated examples for Swedish quality, completeness,
   source fidelity and task fulfillment before export.

The Swedish-marker/prose checks are heuristics, not language identification or an
idiom detector. Register balance is a sampling control, not evidence of improved
idiomaticity. Historical language is labeled, not silently modernized.

Each source manifest reports three selections from the **same filtered pool**:

- `balanced`: register balance first, then clusters within each register.
- `random`: seeded document sampling without replacement.
- `keyword`: Wikipedia-only, whole-word matches from the existing six topic
  keyword lists, balanced across topic strata. This is a controlled proxy for
  the legacy sampling rule; it does not recreate Korp's original hit distribution.

Every strategy uses the same per-split cap and one passage per document. Shortfalls
are reported, never filled with duplicates. Korp responses are cached in
`data/register-cache/korp`, so source requests are replayable offline. Korp's
random sort is seeded, but a rebuilt corpus can change the sample: the manifest's
`korp_corpora` lists each corpus's `Updated` date, so keep the cache and saved pool
to reproduce the exact experiment. Pin the
embedding revision before a published experiment, and retain `embeddings.npz`.

For a matched comparison, first inspect each source manifest and choose a target
that all strategies can supply. For example, with a target of 100:

```bash
./pipeline run register-native --only sources --name native-100 --set datagen.target=100
./pipeline run register-random --only sources --name random-100 --set datagen.target=100
./pipeline run register-keyword --only sources --name keyword-100 --set datagen.target=100
./pipeline run register-native --only generate --name native-100 --set datagen.target=100
./pipeline run register-random --only generate --name random-100 --set datagen.target=100
./pipeline run register-keyword --only generate --name keyword-100 --set datagen.target=100
.venv/bin/python -m source2synth compare-register-runs --runs \
  artifacts/source2synth/runs/native-100 \
  artifacts/source2synth/runs/random-100 \
  artifacts/source2synth/runs/keyword-100 \
  --save artifacts/register-comparison.json
```

The comparison command checks pool, model, split, target and generation settings.
It reports source document count, cluster coverage, register/cluster entropy,
mean pairwise document cosine distance, distinct bigrams and distinct-2. Coverage
and unique bigrams per 1,000 generated tokens use **all observed completion
tokens**, including rejected drafts, independent NLU solving and semantic review.
Missing usage produces `null` efficiency rather than a word-count estimate.
Failed or interrupted requests leave an unknown-cost journal entry. Prompt tokens
are recorded in the raw usage but are not the generated-token denominator.
Embedding compute and source download cost are not generation tokens.

Compare complete runs at matched accepted-example counts, and inspect incomplete
runs as failures/shortfalls. These source coverage metrics do not measure idiom
learning: use the existing idiom and EuroEval evaluation stages plus manual review.
Repeated random seeds are recommended for a substantive sampling-efficiency claim.

## Generation and training

| Format | Generator creates | Assistant target |
| --- | --- | --- |
| Backtranslation | A short standalone instruction | The unchanged native passage |
| Rewrite | A flat literal version of that passage | The unchanged native passage |
| Grounded summary/explanation/QA | An instruction, response and source evidence | The reviewed model response; source context is in the user message |
| Legacy NLU (small share) | Reading-comprehension and NLI candidates | Existing validated NLU outputs, independently solved and reviewed |

Backtranslation follows the construction idea in
[Li et al. (2023), *Self-Alignment with Instruction Backtranslation*](https://arxiv.org/abs/2308.06259).
This implementation does not claim to reproduce Humpback's iterative self-training
experiments. The grounded formats likewise use
[Source2Synth-style construction/curation](https://arxiv.org/abs/2409.08239v2),
not a reproduction of that paper's results.

The short instruction has a 128-token JSON completion cap by default, with a
20–50-token instruction requested in the prompt. This is not a promise that
total cost is 50 tokens: review completions and unsuccessful attempts also cost
tokens and appear in the report. Native passages are assigned to the response
by code, never copied back from a model-generated field.

The new builder uses the existing OpenAI-compatible backend directly to journal
per-call usage; legacy generation still uses DataDesigner. Calls are sequential
in this initial native-data builder. `batch_size` is total candidate attempts,
and `max_batches` bounds work per invocation. Rerunning resumes completed calls
and attempts without paying for them twice. Source/config changes require a new
run; `refresh_inputs` is intentionally unavailable for this path.

Exports are `train.jsonl`, `accepted.jsonl`, `rejected.jsonl`, `report.json`,
`state.json`, and per-attempt/per-call journals. The fine-tuner's existing strict
provenance loader accepts these exports. Its loss remains assistant response and
EOS only: **all system/user tokens, including generated instructions and literal
rewrites, have label `-100`**. The eval source split cannot be exported for training.

Use `hash-test` as the embedding model only for offline plumbing tests. It is
explicitly labeled lexical hashing in the manifest and is not semantic evidence.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_register_data.py' -v
```
