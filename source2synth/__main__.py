from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from .common import ROOT, exact_answer, fingerprint, read_config, read_jsonl, write_json, write_jsonl
from .curation import curation_reasons, deduplicate, flatten_generation, generation_reasons, training_record
from .korp import assert_disjoint, prepare
from .pipeline import curation_builder, designer, evaluation_builder, generation_builder
from .backend import check_backend, resolved_config
from .nlu_sources import TASKS, prepare_nlu, schedule
from .nlu_pipeline import generation_builder as nlu_generation_builder, curation_builder as nlu_curation_builder
from .nlu_tasks import build_example, source_excerpt, training_record as nlu_training_record
from .nlu_curation import CONTROLLED_TASKS, REVIEW_FIELDS, prediction_matches, review_passes


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Swedish Korp Source2Synth / DataDesigner")
    result.add_argument("--config", type=Path, default=ROOT / "configs/source2synth.yaml")
    result.add_argument("--data-dir", type=Path, default=ROOT / "data")
    result.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/source2synth")
    result.add_argument("--endpoint", help="Override model.endpoint (also S2S_ENDPOINT)")
    commands = result.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("prepare", help="Fetch a bounded Korp sample and build disjoint seeds")
    fetch.add_argument("--offline", action="store_true", help="Rebuild only from cached HTTP responses")
    commands.add_parser("validate", help="Compile DataDesigner generation and curation configs offline")
    nlu_prepare = commands.add_parser("prepare-nlu", help="Fetch broader, article-disjoint Korp NLU sources")
    nlu_prepare.add_argument("--offline", action="store_true")
    commands.add_parser("model-check", help="Check that the configured local model is served")
    commands.add_parser("validate-nlu", help="Compile all seven NLU DataDesigner configs offline")
    nlu_generate = commands.add_parser("nlu-generate", help="Generate diverse candidates for all seven NLU tasks")
    nlu_generate.add_argument("--split", choices=["slice0", "slice1", "eval"], default="slice1")
    nlu_generate.add_argument("--tasks", nargs="+", choices=[*TASKS, "all"], default=["all"])
    nlu_generate.add_argument("--records", type=int, default=12, help="Candidate rows PER task")
    nlu_generate.add_argument("--offset", type=int, default=0)
    nlu_generate.add_argument("--model", help="Override configured model id (also S2S_MODEL)")
    nlu_generate.add_argument("--run-name", required=True)
    nlu_curate = commands.add_parser("nlu-curate", help="Independently solve and review generated NLU rows")
    nlu_curate.add_argument("--source-run", type=Path, nargs="+",
                            help="One or more NLU generation runs; defaults to latest")
    nlu_curate.add_argument("--model", help="Override configured model id (also S2S_MODEL)")
    nlu_curate.add_argument("--tasks", nargs="+", choices=[*TASKS, "all"], default=["all"],
                            help="Curate only selected tasks from saved generation runs")
    nlu_curate.add_argument("--run-name", required=True)
    nlu_merge = commands.add_parser("nlu-merge", help="Merge curated NLU runs without new model calls")
    nlu_merge.add_argument("--source-run", type=Path, nargs="+", required=True)
    nlu_merge.add_argument("--run-name", required=True)
    nlu_build = commands.add_parser("nlu-build", help="Generate exactly N accepted NLU rows in resumable batches")
    nlu_build.add_argument("--target", type=int, required=True, help="Accepted, deduplicated examples to export")
    nlu_build.add_argument("--batch-size", type=int, default=12, help="Candidates PER task in each batch")
    nlu_build.add_argument("--max-batches", type=int, default=10, help="Maximum new batches this invocation")
    nlu_build.add_argument("--split", choices=["slice0", "slice1", "eval"], default="slice1")
    nlu_build.add_argument("--tasks", nargs="+", choices=[*TASKS, "all"], default=["all"])
    nlu_build.add_argument("--offset", type=int, default=0, help="Starting seed index per task")
    nlu_build.add_argument("--model", help="Override configured model id (also S2S_MODEL)")
    nlu_build.add_argument("--run-name", required=True)
    nlu_build.add_argument("--resume", action="store_true", help="Continue a stopped build with the same settings")
    nlu_build.add_argument("--refresh-inputs", action="store_true",
                           help="With --resume, record a new source/config revision and preserve completed batches")
    nlu_build.add_argument("--balance-tasks", action="store_true",
                           help="Require nearly equal accepted counts per task and top up deficient tasks")
    generate = commands.add_parser("generate", help="Generate one split, retain rejects, export training messages")
    generate.add_argument("--split", choices=["slice0", "slice1", "eval"], default="slice0")
    generate.add_argument("--records", type=int, default=5)
    generate.add_argument("--offset", type=int, default=0, help="Starting seed row for a new batch")
    generate.add_argument("--model", help="Override configured local model id")
    curate = commands.add_parser("curate", help="Impute and filter generated slice1 data")
    curate.add_argument("--mode", choices=["baseline", "paper"], required=True)
    curate.add_argument("--records", type=int, default=5)
    curate.add_argument("--curator-model", help="Served model ID of the model trained on slice0")
    curate.add_argument("--training-manifest", type=Path,
                        help="Paper mode: checkpoint provenance (see README)")
    evaluate = commands.add_parser("evaluate", help="Score a model against held-out synthetic QA")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--records", type=int, default=5)
    refilter = commands.add_parser("refilter", help="Reapply curation checks to saved outputs without model calls")
    refilter.add_argument("--source-run", required=True, type=Path)
    refilter.add_argument("--run-name", required=True)
    for command in (generate, curate, evaluate):
        command.add_argument("--provider", help="Override the configured provider name")
        command.add_argument("--run-name", required=True, help="New output directory; never silently overwrite a run")
    return result


def new_run(args) -> Path:
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("--run-name must be a single directory name")
    target = args.output_dir / "runs" / args.run_name
    target.mkdir(parents=True, exist_ok=False)
    return target


def require_credentials(config: dict, provider: str | None = None) -> None:
    if resolved_config(config)["model"].get("endpoint"):
        return
    provider = provider or config["model"]["provider"]
    env_name = {"nvidia": "NVIDIA_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(provider)
    if env_name and not os.getenv(env_name):
        # Respect credentials configured via the installed DataDesigner CLI as well.
        import yaml
        path = Path.home() / ".data-designer/model_providers.yaml"
        configured = yaml.safe_load(path.read_text()) if path.exists() else {}
        for entry in (configured or {}).get("providers", []):
            key = entry.get("api_key") or ""
            if entry.get("name") == provider and key and (os.getenv(key) or key != env_name):
                if not (key.upper() == key and key.replace("_", "").isalnum()):
                    return
                if os.getenv(key):
                    return
        raise ValueError(f"Set {env_name} in your shell, then rerun. No model calls were made.")


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    # Convert Arrow/numpy containers to regular JSON values.
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def validate(config: dict, args) -> dict:
    prepared = args.data_dir / "prepared"
    pairs = {split: dataframe_records(pd.read_parquet(prepared / f"{split}.parquet"))
             for split in ("slice0", "slice1", "eval")}
    assert_disjoint(pairs)
    target = args.output_dir / "configs"
    target.mkdir(parents=True, exist_ok=True)
    # Curation schema fixture is only for compilation. It is never exported as data.
    fixture = {**pairs["slice1"][0], "question": "SCHEMA ONLY", "q2": "SCHEMA ONLY", "answer": "SCHEMA ONLY"}
    schema_seed = target / "curation_schema_only.parquet"
    pd.DataFrame([fixture]).to_parquet(schema_seed, index=False)
    engine = designer(args.output_dir / "validation", config)
    for split in pairs:
        builder = generation_builder(prepared / f"{split}.parquet", config)
        engine.validate(builder)
        builder.write_config(target / f"generate_{split}.yaml")
    for stage, factory in (("curate", curation_builder), ("evaluate", evaluation_builder)):
        builder = factory(schema_seed, config)
        engine.validate(builder)
        builder.write_config(target / f"{stage}_schema_only.yaml")
    report = {
        "data_designer_version": importlib.metadata.version("data-designer"),
        "document_disjoint": True, "pairs": {s: len(r) for s, r in pairs.items()},
        "configs_compiled": 5, "llm_calls": 0,
        "model": config["model"]["id"],
    }
    write_json(args.output_dir / "validation.json", report)
    return report


def record_report(run: Path, config: dict, stage: str, accepted: list[dict], rejected: list[dict], **extra) -> dict:
    write_jsonl(run / "accepted.jsonl", accepted)
    write_jsonl(run / "rejected.jsonl", rejected)
    report = {
        "stage": stage, "config": config,
        "data_designer_version": importlib.metadata.version("data-designer"),
        "accepted": len(accepted), "rejected": len(rejected),
        "rejection_counts": dict(Counter(reason for row in rejected for reason in row["rejection_reasons"])),
        **extra,
    }
    write_json(run / "report.json", report)
    return report


def generate(config: dict, args) -> dict:
    prepared = args.data_dir / "prepared"
    seed = prepared / f"{args.split}.parquet"
    frame = pd.read_parquet(seed)
    count = len(frame)
    if args.offset < 0 or args.records < 1 or args.offset + args.records > count:
        raise ValueError(f"--offset + --records must select a nonempty range within {count} seeds")
    require_credentials(config, args.provider)
    run = new_run(args)
    seed = run / "generation_input.parquet"
    frame.iloc[args.offset:args.offset + args.records].to_parquet(seed, index=False)
    builder = generation_builder(seed, config, model_id=args.model, provider=args.provider)
    builder.write_config(run / "datadesigner.yaml")
    results = designer(run / "dd", config).create(builder, num_records=args.records, dataset_name="generation")
    rows = dataframe_records(results.load_dataset())
    accepted, rejected = [], []
    for row in rows:
        try:
            reasons = generation_reasons(row)
        except (ValidationError, KeyError, TypeError) as error:
            reasons = [f"invalid_schema:{type(error).__name__}"]
        if reasons:
            rejected.append({**row, "rejection_reasons": reasons})
        else:
            accepted.append(flatten_generation(row))
    accepted, duplicates = deduplicate(accepted)
    rejected.extend(duplicates)
    returned_ids = {row["seed_id"] for row in rows}
    rejected.extend({**row, "rejection_reasons": ["datadesigner_generation_failed"]}
                    for row in dataframe_records(pd.read_parquet(seed)) if row["seed_id"] not in returned_ids)
    if args.split != "eval":
        write_jsonl(run / "train.jsonl", [training_record(row) for row in accepted])
    report = record_report(run, config, "generation", accepted, rejected,
                         split=args.split, offset=args.offset, requested=args.records, returned=len(rows),
                         incomplete_records=max(0, args.records - len(rows)),
                         seed_sha256=fingerprint(dataframe_records(pd.read_parquet(seed))),
                         model=builder.model_configs[0].model_dump(mode="json"))
    write_json(args.output_dir / f"latest_{args.split}.json", {"run": str(run.resolve())})
    return report


def latest_run(output: Path, split: str) -> Path:
    pointer = output / f"latest_{split}.json"
    if not pointer.exists():
        raise ValueError(f"Generate {split} first; no {pointer} exists")
    return Path(json.loads(pointer.read_text())["run"])


def training_provenance(config: dict, args) -> dict | None:
    if args.mode == "baseline":
        if args.training_manifest:
            raise ValueError("A training manifest is only used with --mode paper")
        return None
    if not args.curator_model or not args.training_manifest:
        raise ValueError("Paper mode requires --curator-model and --training-manifest from slice0 training")
    manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    source = latest_run(args.output_dir, "slice0") / "train.jsonl"
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    required = {"model_id", "base_model", "slice0_train_sha256", "checkpoint", "training_command"}
    if not required.issubset(manifest) or any(not manifest[k] for k in required):
        raise ValueError(f"Training manifest requires nonempty fields: {sorted(required)}")
    if manifest["model_id"] != args.curator_model or manifest["slice0_train_sha256"] != expected:
        raise ValueError("Training manifest does not match the served curator or latest slice0 training data")
    if not read_jsonl(source):
        raise ValueError("Cannot curate in paper mode with an empty slice0 training dataset")
    return manifest


def curate(config: dict, args) -> dict:
    source = latest_run(args.output_dir, "slice1")
    rows = read_jsonl(source / "accepted.jsonl")
    if not 1 <= args.records <= len(rows):
        raise ValueError(f"--records must be 1..{len(rows)} accepted slice1 rows")
    provenance = training_provenance(config, args)
    if provenance:
        slice0 = read_jsonl(latest_run(args.output_dir, "slice0") / "accepted.jsonl")
        assert_disjoint({"slice0": slice0, "slice1": rows})
    require_credentials(config, args.provider)
    run = new_run(args)
    # Keep source rows and provenance, but do not seed the model with original Q1.
    # Prompts also omit the target answer in all solver and imputation columns.
    seed = run / "curation_input.parquet"
    inputs = [{k: v for k, v in row.items()
               if k not in {"hop1", "hop2", "merged", "q1", "evidence1"} and not k.endswith("_raw")}
              for row in rows[:args.records]]
    pd.DataFrame(inputs).to_parquet(seed, index=False)
    builder = curation_builder(seed, config, model_id=args.curator_model, provider=args.provider)
    builder.write_config(run / "datadesigner.yaml")
    results = designer(run / "dd", config).create(builder, num_records=args.records, dataset_name="curation")
    generated = dataframe_records(results.load_dataset())
    accepted, rejected = [], []
    for row in generated:
        try:
            reasons = curation_reasons(row, config["curation"]["attempts"])
        except (ValidationError, KeyError, TypeError) as error:
            reasons = [f"invalid_schema:{type(error).__name__}"]
        if reasons:
            rejected.append({**row, "rejection_reasons": reasons})
        else:
            accepted.append(row)
    accepted, duplicates = deduplicate(accepted, "recomposed")
    rejected.extend(duplicates)
    returned_ids = {row["seed_id"] for row in generated}
    rejected.extend({**row, "rejection_reasons": ["datadesigner_generation_failed"]}
                    for row in inputs if row["seed_id"] not in returned_ids)
    write_jsonl(run / "train.jsonl", [training_record(row, curated=True) for row in accepted])
    report = record_report(run, config, "curation", accepted, rejected, mode=args.mode,
                          source_run=str(source), training_provenance=provenance,
                          requested=args.records, returned=len(generated),
                          incomplete_records=max(0, args.records - len(generated)),
                          model=builder.model_configs[0].model_dump(mode="json"))
    return report


def evaluate(config: dict, args) -> dict:
    source = latest_run(args.output_dir, "eval")
    rows = read_jsonl(source / "accepted.jsonl")
    if not 1 <= args.records <= len(rows):
        raise ValueError(f"--records must be 1..{len(rows)} held-out rows")
    require_credentials(config, args.provider)
    run = new_run(args)
    seed = run / "evaluation_input.parquet"
    selected = rows[:args.records]
    # The target, bridge and decomposed questions are absent from evaluation seeds.
    pd.DataFrame([{key: row[key] for key in ("seed_id", "question", "d1_title", "d1_text", "d2_title", "d2_text")}
                  for row in selected]).to_parquet(seed, index=False)
    builder = evaluation_builder(seed, config, model_id=args.model, provider=args.provider)
    builder.write_config(run / "datadesigner.yaml")
    results = designer(run / "dd", config).create(builder, num_records=args.records, dataset_name="evaluation")
    predictions = {r["seed_id"]: r.get("prediction") for r in dataframe_records(results.load_dataset())}
    scored = []
    for row in selected:
        pred = predictions.get(row["seed_id"]) or {}
        score = pred.get("answerable") is True and exact_answer(pred.get("answer", ""), row["answer"])
        scored.append({"seed_id": row["seed_id"], "expected": row["answer"], "prediction": pred, "exact_match": score})
    write_jsonl(run / "scores.jsonl", scored)
    report = {"model": args.model, "records": len(scored),
              "normalized_exact_match": sum(r["exact_match"] for r in scored) / len(scored),
              "limitation": "Held-out synthetic reference answers require manual review; not HotpotQA results"}
    write_json(run / "report.json", report)
    return report


def refilter(config: dict, args) -> dict:
    source = args.source_run.resolve()
    original = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if original["stage"] != "curation":
        raise ValueError("refilter requires a saved curation run")
    if config["curation"]["attempts"] != original["config"]["curation"]["attempts"]:
        raise ValueError("Cannot change the attempt count when reusing saved model outputs")
    rows = read_jsonl(source / "accepted.jsonl") + read_jsonl(source / "rejected.jsonl")
    accepted, rejected = [], []
    for row in rows:
        row = {k: v for k, v in row.items() if k != "rejection_reasons"}
        try:
            reasons = curation_reasons(row, config["curation"]["attempts"])
        except (ValidationError, KeyError, TypeError):
            reasons = ["datadesigner_generation_failed"]
        if reasons:
            rejected.append({**row, "rejection_reasons": reasons})
        else:
            accepted.append(row)
    accepted, duplicates = deduplicate(accepted, "recomposed")
    rejected.extend(duplicates)
    run = new_run(args)
    write_jsonl(run / "train.jsonl", [training_record(row, curated=True) for row in accepted])
    return record_report(run, config, "curation", accepted, rejected,
                         mode=original["mode"], source_run=str(source), refiltered=True, llm_calls=0,
                         model=original["model"], training_provenance=original.get("training_provenance"))


def nlu_generate(config: dict, args) -> dict:
    tasks = list(TASKS) if "all" in args.tasks else list(dict.fromkeys(args.tasks))
    seeds = schedule(config, args.data_dir, args.split, tasks, args.records, args.offset)
    require_credentials(config)
    backend = check_backend(config)
    run = new_run(args)
    accepted, rejected = [], []
    for task in tasks:
        selected = [row for row in seeds if row["task"] == task]
        task_dir = run / task
        task_dir.mkdir()
        flat = []
        for row in selected:
            source = row["source"]
            flat.append({"seed_id": row["seed_id"], "task": task,
                         "target": row["target"], "style": row["style"],
                         "difficulty": row["difficulty"], "source_text": source["text"],
                         "source_title": source["title"],
                         "source_excerpt": source_excerpt(source["text"])})
        seed_path = task_dir / "seed.parquet"
        pd.DataFrame(flat).to_parquet(seed_path, index=False)
        builder = nlu_generation_builder(seed_path, config, task)
        builder.write_config(task_dir / "datadesigner.yaml")
        results = designer(task_dir / "dd", config).create(builder, num_records=len(selected), dataset_name=task)
        returned = dataframe_records(results.load_dataset())
        lookup = {row["seed_id"]: row for row in selected}
        seen = set()
        for row in returned:
            seed_id = row.get("seed_id")
            if seed_id not in lookup:
                rejected.append({"task": task, "seed_id": seed_id, "rejection_reasons": ["unknown_seed_id"]})
                continue
            seen.add(seed_id)
            try:
                item = build_example(lookup[seed_id], row["draft"], list(config["nlu"]["topics"]))
            except (ValueError, ValidationError, KeyError, TypeError) as error:
                rejected.append({"task": task, "seed_id": seed_id, "raw": row.get("draft_raw"),
                                 "draft": row.get("draft"), "rejection_reasons": [str(error)]})
            else:
                accepted.append(item)
        rejected.extend({"task": task, "seed_id": row["seed_id"],
                         "rejection_reasons": ["datadesigner_generation_failed"]}
                        for row in selected if row["seed_id"] not in seen)
    # A task-level dedup also catches repeated articles or questions within this run.
    unique, duplicates = [], set()
    for item in accepted:
        key = fingerprint([item["task"], item["input"]])
        if key in duplicates:
            rejected.append({"task": item["task"], "seed_id": item["id"],
                             "rejection_reasons": ["duplicate_task_input"]})
        else:
            duplicates.add(key)
            unique.append(item)
    if args.split == "slice0":
        write_jsonl(run / "uncurated_train.jsonl", [nlu_training_record(x) for x in unique])
    report = record_report(run, config, "nlu_generation", unique, rejected,
                           split=args.split, tasks=tasks, candidates=len(seeds),
                           by_task=dict(Counter(x["task"] for x in unique)), backend=backend,
                           model=config["model"]["id"],
                           note="Candidate labels are model-created; run nlu-curate before training")
    write_json(args.output_dir / "latest_nlu.json", {"run": str(run.resolve())})
    return report


def validate_nlu(config: dict, args) -> dict:
    rows = schedule(config, args.data_dir, "slice1", list(TASKS), 12, 0)
    target = args.output_dir / "nlu_configs"
    target.mkdir(parents=True, exist_ok=True)
    engine = designer(args.output_dir / "nlu_validation", config)
    for task in TASKS:
        sample = next(row for row in rows if row["task"] == task)
        path = target / f"{task}_schema_only.parquet"
        pd.DataFrame([{"seed_id": sample["seed_id"], "task": task,
                       "target": sample["target"], "style": sample["style"],
                       "difficulty": sample["difficulty"],
                       "source_text": sample["source"]["text"],
                       "source_title": sample["source"]["title"],
                       "source_excerpt": source_excerpt(sample["source"]["text"])}]).to_parquet(path, index=False)
        builder = nlu_generation_builder(path, config, task)
        engine.validate(builder)
        builder.write_config(target / f"{task}.yaml")
    path = target / "curation_schema_only.parquet"
    pd.DataFrame([{"seed_id": "schema-only", "task": "classification", "target": "kultur",
                   "input": "SCHEMA ONLY", "source_text": "SCHEMA ONLY",
                   "expected_json": '"kultur"', "draft_json": "{}"}]).to_parquet(path, index=False)
    builder = nlu_curation_builder(path, config)
    engine.validate(builder)
    builder.write_config(target / "curation.yaml")
    ner_path = target / "ner_curation_schema_only.parquet"
    pd.DataFrame([{"seed_id": "schema-only", "task": "named-entity-recognition", "target": "per",
                   "input": "SCHEMA ONLY", "ner_text": "SCHEMA ONLY", "source_text": "SCHEMA ONLY",
                   "expected_json": "[]", "draft_json": "{}"}]).to_parquet(ner_path, index=False)
    ner_builder = nlu_curation_builder(ner_path, config, ner=True)
    engine.validate(ner_builder)
    ner_builder.write_config(target / "ner_curation.yaml")
    ged_path = target / "ged_curation_schema_only.parquet"
    pd.DataFrame([{"seed_id": "schema-only", "task": "grammatical-error-detection", "target": "correct",
                   "input": "SCHEMA ONLY", "indexed_tokens": "0: SCHEMA 1: ONLY", "source_text": "SCHEMA ONLY",
                   "expected_json": "[]", "draft_json": "{}"}]).to_parquet(ged_path, index=False)
    ged_builder = nlu_curation_builder(ged_path, config, ged=True)
    engine.validate(ged_builder)
    ged_builder.write_config(target / "ged_curation.yaml")
    report = {"data_designer_version": importlib.metadata.version("data-designer"),
              "tasks": list(TASKS), "configs_compiled": 10,
              "source_article_disjoint": True, "llm_calls": 0,
              "model": config["model"]["id"]}
    write_json(args.output_dir / "nlu_validation.json", report)
    return report


def _nlu_curator_responses(rows: list[dict], run: Path, config: dict) -> list[dict]:
    """Use compact entity predictions for NER and retry only answer mismatches."""
    attempts = config["nlu"].get("curation_attempts", 1)
    if type(attempts) is not int or attempts < 1:
        raise ValueError("nlu.curation_attempts must be a positive integer")
    returned = []
    token_tasks = {"named-entity-recognition": "ner", "grammatical-error-detection": "ged"}
    for group in ("other", "ner", "ged"):
        ner, ged = group == "ner", group == "ged"
        selected = [x for x in rows if token_tasks.get(x["task"], "other") == group]
        if not selected:
            continue
        directory = run / group if ner or ged else run
        pending = selected
        responses = {}
        histories = {x["id"]: [] for x in selected}
        for attempt in range(attempts):
            stage = directory / f"retry-{attempt}" if attempt else directory
            stage.mkdir(parents=True, exist_ok=True)
            flat = [{"seed_id": x["id"], "task": x["task"], "target": x["target"],
                     "input": x["input"], "ner_text": x.get("text", ""),
                     "indexed_tokens": json.dumps(list(enumerate(x.get("tokens", []))), ensure_ascii=False),
                     "source_text": x["source"]["text"],
                     "expected_json": json.dumps(x["output"], ensure_ascii=False),
                     "draft_json": json.dumps(x["draft"], ensure_ascii=False)} for x in pending]
            seed_path = stage / "seed.parquet"
            pd.DataFrame(flat).to_parquet(seed_path, index=False)
            builder = nlu_curation_builder(seed_path, config, ner=ner, ged=ged,
                                          prediction_only=bool(attempt), attempt=attempt)
            builder.write_config(stage / "datadesigner.yaml")
            results = designer(stage / "dd", config).create(
                builder, num_records=len(pending), dataset_name="nlu_curation")
            for row in dataframe_records(results.load_dataset()):
                seed_id = row.get("seed_id")
                if seed_id not in histories:
                    returned.append(row)  # The caller records unknown IDs as rejections.
                    continue
                histories[seed_id].append(row.get("prediction") or {})
                previous = responses.get(seed_id, {})
                responses[seed_id] = {**previous, **row,
                                     "review": previous.get("review", {}) if attempt else row.get("review", {})}
            pending = [x for x in pending if x["id"] in responses
                       and x["task"] not in CONTROLLED_TASKS
                       and review_passes(responses[x["id"]].get("review") or {})
                       and not prediction_matches(x, responses[x["id"]].get("prediction") or {})]
            if not pending:
                break
        returned.extend({**row, "prediction_attempts": histories[seed_id]}
                        for seed_id, row in responses.items())
    return returned


def nlu_curate(config: dict, args) -> dict:
    sources = args.source_run or [latest_run(args.output_dir, "nlu")]
    rows = []
    for source in sources:
        if json.loads((source / "report.json").read_text())["stage"] != "nlu_generation":
            raise ValueError("nlu-curate requires NLU generation runs")
        rows.extend({**row, "generation_run": str(source.resolve())}
                    for row in read_jsonl(source / "accepted.jsonl"))
    selected_tasks = getattr(args, "tasks", ["all"])
    if "all" not in selected_tasks:
        rows = [row for row in rows if row["task"] in selected_tasks]
    if not rows:
        raise ValueError("No generated NLU candidates to curate")
    if len({row["split"] for row in rows}) != 1:
        raise ValueError("Cannot mix training and evaluation splits in one curation run")
    requested = len(rows)
    screened, pre_rejected, seen_inputs, seen_ids = [], [], set(), set()
    for item in rows:
        if item["id"] in seen_ids:
            pre_rejected.append({**item, "rejection_reasons": ["duplicate_seed_id_across_runs"]})
            continue
        seen_ids.add(item["id"])
        seed = {"seed_id": item["id"], "task": item["task"], "split": item["split"],
                "target": item["target"], "style": item["style"],
                "difficulty": item["difficulty"], "source": item["source"]}
        try:
            candidate_indices = ([item["sample_index"]] if "sample_index" in item else
                                 [0, 1, 2] if item["task"] == "linguistic-acceptability"
                                 and item["target"] == "incorrect" else [0])
            for index in candidate_indices:
                seed["sample_index"] = index
                try:
                    checked = build_example(seed, item["draft"], list(config["nlu"]["topics"]))
                except (ValueError, ValidationError, KeyError, TypeError):
                    continue
                if checked["input"] == item["input"] and checked["output"] == item["output"]:
                    break
            else:
                raise ValueError("saved_candidate_does_not_match_current_rules")
        except (ValueError, ValidationError, KeyError, TypeError) as error:
            pre_rejected.append({**item, "rejection_reasons": ["deterministic_recheck:" + str(error)]})
        else:
            dedup_key = fingerprint([item["task"], item["input"]])
            if dedup_key in seen_inputs:
                pre_rejected.append({**item, "rejection_reasons": ["duplicate_task_input_across_runs"]})
            else:
                seen_inputs.add(dedup_key)
                screened.append(item)
    rows = screened
    prefilter_count = len(pre_rejected)
    if not rows:
        run = new_run(args)
        if pre_rejected[0]["split"] != "eval":
            write_jsonl(run / "train.jsonl", [])
        return record_report(run, config, "nlu_curation", [], pre_rejected,
                             source_runs=[str(path.resolve()) for path in sources],
                             requested=requested, returned=0,
                             llm_calls=0, limitation="No candidate survived deterministic recheck")
    require_credentials(config)
    backend = check_backend(config)
    run = new_run(args)
    returned = _nlu_curator_responses(rows, run, config)
    lookup = {x["id"]: x for x in rows}
    accepted, rejected, seen = [], pre_rejected, set()
    for row in returned:
        seed_id = row.get("seed_id")
        if seed_id not in lookup:
            rejected.append({"seed_id": seed_id, "rejection_reasons": ["unknown_seed_id"]})
            continue
        seen.add(seed_id)
        item = lookup[seed_id]
        pred = row.get("prediction") or {}
        review = row.get("review") or {}
        match = prediction_matches(item, pred)
        # The labels for controlled grammar edits and verb permutations are
        # computed from the saved edit, not inferred by the generator. A weak
        # local solver can misread these token tasks; retain its disagreement
        # for audit, while requiring the independent semantic review to pass.
        mechanical_label = item["task"] in CONTROLLED_TASKS
        item = {**item, "prediction_attempts": row.get("prediction_attempts", [])}
        reasons = []
        if not match and not mechanical_label:
            reasons.append("independent_answer_mismatch")
        reasons.extend("review_failed_" + key for key in
                       REVIEW_FIELDS
                       if review.get(key) is not True)
        if reasons:
            rejected.append({**item, "prediction": pred, "review": review, "rejection_reasons": reasons})
        else:
            accepted.append({**item, "prediction": pred, "review": review,
                             "solver_disagreed_with_deterministic_label": bool(not match and mechanical_label)})
    rejected.extend({**lookup[x], "rejection_reasons": ["datadesigner_curation_failed"]}
                    for x in lookup if x not in seen)
    if rows[0]["split"] != "eval":
        write_jsonl(run / "train.jsonl", [nlu_training_record(x) for x in accepted])
    return record_report(run, config, "nlu_curation", accepted, rejected,
                         source_runs=[str(path.resolve()) for path in sources], split=rows[0]["split"],
                         by_task=dict(Counter(x["task"] for x in accepted)),
                         by_label=dict(Counter(f"{x['task']}/{x['target']}" for x in accepted)),
                         solver_disagreements_on_controlled_tasks=sum(
                             x["solver_disagreed_with_deterministic_label"] for x in accepted),
                         requested=requested, prefilter_rejected=prefilter_count,
                         max_solver_attempts=config["nlu"].get("curation_attempts", 1),
                         solver_attempts=sum(len(row.get("prediction_attempts", [])) for row in returned),
                         returned=len(returned), backend=backend,
                         model=config["model"]["id"],
                         limitation="Synthetic references and model review still need human spot checks")


def nlu_merge(config: dict, args) -> dict:
    source_runs = [path.resolve() for path in args.source_run]
    accepted, rejected, seen_ids, seen_inputs, splits, models = [], [], set(), set(), set(), set()
    for path in source_runs:
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        if report["stage"] != "nlu_curation":
            raise ValueError(f"Not a curated NLU run: {path}")
        splits.add(report["split"])
        models.add(report["model"])
        for row in read_jsonl(path / "accepted.jsonl"):
            row = {**row, "curation_run": str(path)}
            key = fingerprint([row["task"], row["input"]])
            if row["id"] in seen_ids or key in seen_inputs:
                rejected.append({**row, "rejection_reasons": ["duplicate_across_curated_runs"]})
                continue
            seen_ids.add(row["id"])
            seen_inputs.add(key)
            accepted.append(row)
    if len(splits) != 1:
        raise ValueError("Cannot merge curated runs from different splits")
    split = splits.pop()
    run = new_run(args)
    if split != "eval":
        write_jsonl(run / "train.jsonl", [nlu_training_record(row) for row in accepted])
    return record_report(run, config, "nlu_merge", accepted, rejected,
                         source_runs=[str(path) for path in source_runs], split=split,
                         by_task=dict(Counter(row["task"] for row in accepted)),
                         by_label=dict(Counter(f"{row['task']}/{row['target']}" for row in accepted)),
                         models=sorted(models), llm_calls=0,
                         limitation="Synthetic references and model review still need human spot checks")


def _complete_child_run(root: Path, prefix: str, stage: str) -> Path | None:
    """Reuse a finished child after interruption; ignore unfinished attempts."""
    for path in sorted(root.iterdir()):
        if not (path.is_dir() and (path.name == prefix or path.name.startswith(prefix + "-retry-"))):
            continue
        report_path = path / "report.json"
        if report_path.exists():
            try:
                if json.loads(report_path.read_text(encoding="utf-8"))["stage"] == stage:
                    return path
            except (ValueError, KeyError):
                pass
    return None


def _child_name(root: Path, prefix: str) -> str:
    if not (root / prefix).exists():
        return prefix
    attempt = 1
    while (root / f"{prefix}-retry-{attempt}").exists():
        attempt += 1
    return f"{prefix}-retry-{attempt}"


def _build_output(run: Path, state: dict, config: dict) -> dict:
    accepted, rejected, surplus = [], [], []
    seen_ids, seen_inputs = set(), set()
    tasks = state["tasks"]
    quotas = {task: state["target"] // len(tasks) + (i < state["target"] % len(tasks))
              for i, task in enumerate(tasks)} if state.get("balance_tasks") else {}
    task_counts = Counter()
    for batch in state["batches"]:
        if not batch["curation_run"]:
            continue
        rows = read_jsonl(Path(batch["curation_run"]) / "accepted.jsonl")
        if any(row["task"] not in tasks for row in rows):
            raise ValueError("A curated child contains a task outside this build")
        # Avoid letting a large task at the start of the last batch fill every
        # remaining slot before other task families are considered.
        grouped = {task: [row for row in rows if row["task"] == task] for task in tasks}
        ordered = []
        while any(grouped.values()):
            for task in tasks:
                if grouped[task]:
                    ordered.append(grouped[task].pop(0))
        for item in ordered:
            item = {**item, "curation_run": batch["curation_run"]}
            key = fingerprint([item["task"], item["input"]])
            if item["id"] in seen_ids or key in seen_inputs:
                rejected.append({**item, "rejection_reasons": ["duplicate_across_batches"]})
                continue
            seen_ids.add(item["id"])
            seen_inputs.add(key)
            if len(accepted) < state["target"] and (not quotas or task_counts[item["task"]] < quotas[item["task"]]):
                accepted.append(item)
                task_counts[item["task"]] += 1
            else:
                surplus.append(item)
    write_jsonl(run / "surplus.jsonl", surplus)
    if state["split"] != "eval":
        write_jsonl(run / "train.jsonl", [nlu_training_record(row) for row in accepted])
    return record_report(run, config, "nlu_build", accepted, rejected,
                         target=state["target"], complete=len(accepted) == state["target"],
                         split=state["split"], tasks=tasks, batch_size=state["batch_size"],
                         batches=state["batches"], next_offset=state["offset"] +
                         len(state["batches"]) * state["batch_size"],
                         curated_before_dedup=sum(batch["curated"] for batch in state["batches"]),
                         surplus=len(surplus),
                         task_quotas=quotas,
                         input_revisions=state.get("input_revisions", []),
                         unique_source_articles=len({row["source"]["document_id"] for row in accepted}),
                         by_task=dict(Counter(row["task"] for row in accepted)),
                         by_label=dict(Counter(f"{row['task']}/{row['target']}" for row in accepted)),
                         model=config["model"]["id"],
                         limitation="Accepted synthetic labels still need human spot checks")


def nlu_build(config: dict, args) -> dict:
    if args.target < 1 or args.batch_size < 1 or args.max_batches < 1 or args.offset < 0:
        raise ValueError("--target, --batch-size and --max-batches must be positive; --offset cannot be negative")
    tasks = list(TASKS) if "all" in args.tasks else list(dict.fromkeys(args.tasks))
    if args.refresh_inputs and not args.resume:
        raise ValueError("--refresh-inputs requires --resume")
    passages = args.data_dir / "nlu/passages.jsonl"
    if not passages.exists():
        raise ValueError("No NLU passages found; run prepare-nlu first")
    contract = {"target": args.target, "batch_size": args.batch_size,
                "split": args.split, "tasks": tasks, "offset": args.offset,
                "balance_tasks": args.balance_tasks,
                "implementation_sha256": fingerprint({name: (ROOT / "source2synth" / name).read_text(encoding="utf-8")
                    for name in ["__main__.py", "nlu_tasks.py", "nlu_sources.py", "nlu_pipeline.py", "nlu_curation.py"]}),
                "config_sha256": fingerprint(config),
                "passages_sha256": hashlib.sha256(passages.read_bytes()).hexdigest()}
    run = args.output_dir / "runs" / args.run_name
    if args.resume:
        state_path = run / "state.json"
        if not state_path.exists():
            raise ValueError(f"No resumable build at {run}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.setdefault("balance_tasks", False)
        changed = [key for key, value in contract.items() if state.get(key) != value]
        refreshable = {"config_sha256", "passages_sha256", "balance_tasks", "implementation_sha256"}
        if changed and (not args.refresh_inputs or set(changed) - refreshable):
            raise ValueError("Resume settings, model configuration or source passages differ from the saved build")
        if changed:
            # A new revision must never reuse an orphaned child from old inputs.
            check_backend(config)
            schedule(config, args.data_dir, args.split, tasks, 1,
                     args.offset + len(state["batches"]) * args.batch_size)
            state.setdefault("input_revisions", []).append({
                "after_batches": len(state["batches"]),
                "previous": {key: state.get(key) for key in refreshable},
                "current": {key: contract[key] for key in refreshable},
                "config": config,
            })
            state.update(contract)
            temp_state = run / "state.json.tmp"
            write_json(temp_state, state)
            temp_state.replace(state_path)
    else:
        state = {**contract, "batches": []}
        # Catch missing strata and an unavailable model before claiming a run name.
        schedule(config, args.data_dir, args.split, tasks, 1, args.offset)
        check_backend(config)
        run = new_run(args)
        write_json(run / "state.json", state)
    report = _build_output(run, state, config)
    if report["complete"]:
        return report
    check_backend(config)
    root = args.output_dir / "runs"
    for _ in range(args.max_batches):
        index = len(state["batches"])
        offset = args.offset + index * args.batch_size
        pending_tasks = [task for task in tasks if not report["task_quotas"]
                         or report["by_task"].get(task, 0) < report["task_quotas"][task]]
        revision = f"-v{len(state['input_revisions'])}" if state.get("input_revisions") else ""
        prefix = f"{args.run_name}{revision}-gen-{index:04d}"
        generated = _complete_child_run(root, prefix, "nlu_generation")
        if generated is None:
            name = _child_name(root, prefix)
            gen_args = argparse.Namespace(**{**vars(args), "tasks": pending_tasks,
                                             "records": args.batch_size,
                                             "offset": offset, "run_name": name})
            nlu_generate(config, gen_args)
            generated = root / name
        generated_report = json.loads((generated / "report.json").read_text(encoding="utf-8"))
        curated = None
        curated_count = 0
        if generated_report["accepted"]:
            prefix = f"{args.run_name}{revision}-cur-{index:04d}"
            curated = _complete_child_run(root, prefix, "nlu_curation")
            if curated is None:
                name = _child_name(root, prefix)
                cur_args = argparse.Namespace(**{**vars(args), "source_run": [generated],
                                               "run_name": name})
                nlu_curate(config, cur_args)
                curated = root / name
            curated_count = len(read_jsonl(curated / "accepted.jsonl"))
        state["batches"].append({"offset": offset,
                                 "generation_run": str(generated.resolve()),
                                 "generated": generated_report["accepted"],
                                 "curation_run": str(curated.resolve()) if curated else None,
                                 "curated": curated_count})
        temp_state = run / "state.json.tmp"
        write_json(temp_state, state)
        temp_state.replace(run / "state.json")
        report = _build_output(run, state, config)
        print(f"Batch {index + 1}: {report['accepted']}/{args.target} accepted; "
              f"next offset {report['next_offset']}", flush=True)
        if report["complete"]:
            break
    return report


def main() -> None:
    args = parser().parse_args()
    config = resolved_config(read_config(args.config), getattr(args, "model", None), args.endpoint)
    try:
        if args.command == "prepare":
            report = prepare(config, args.data_dir, args.offline)
        elif args.command == "prepare-nlu":
            report = prepare_nlu(config, args.data_dir, args.offline)
        elif args.command == "validate":
            report = validate(config, args)
        elif args.command == "model-check":
            report = check_backend(config)
        elif args.command == "validate-nlu":
            report = validate_nlu(config, args)
        else:
            report = {"generate": generate, "curate": curate, "evaluate": evaluate,
                      "refilter": refilter, "nlu-generate": nlu_generate,
                      "nlu-curate": nlu_curate, "nlu-merge": nlu_merge,
                      "nlu-build": nlu_build}[args.command](config, args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
