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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Swedish Korp Source2Synth / DataDesigner")
    result.add_argument("--config", type=Path, default=ROOT / "configs/source2synth.yaml")
    result.add_argument("--data-dir", type=Path, default=ROOT / "data")
    result.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/source2synth")
    commands = result.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("prepare", help="Fetch a bounded Korp sample and build disjoint seeds")
    fetch.add_argument("--offline", action="store_true", help="Rebuild only from cached HTTP responses")
    commands.add_parser("validate", help="Compile DataDesigner generation and curation configs offline")
    generate = commands.add_parser("generate", help="Generate one split, retain rejects, export training messages")
    generate.add_argument("--split", choices=["slice0", "slice1", "eval"], default="slice0")
    generate.add_argument("--records", type=int, default=5)
    generate.add_argument("--offset", type=int, default=0, help="Starting seed row for a new batch")
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
    if os.getenv("S2S_ENDPOINT"):
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
    engine = designer(args.output_dir / "validation")
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
        "nvidia_key_in_environment": bool(os.getenv("NVIDIA_API_KEY")),
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
    builder = generation_builder(seed, config, provider=args.provider)
    builder.write_config(run / "datadesigner.yaml")
    results = designer(run / "dd").create(builder, num_records=args.records, dataset_name="generation")
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
    results = designer(run / "dd").create(builder, num_records=args.records, dataset_name="curation")
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
    results = designer(run / "dd").create(builder, num_records=args.records, dataset_name="evaluation")
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


def main() -> None:
    args = parser().parse_args()
    config = read_config(args.config)
    try:
        if args.command == "prepare":
            report = prepare(config, args.data_dir, args.offline)
        elif args.command == "validate":
            report = validate(config, args)
        else:
            report = {"generate": generate, "curate": curate, "evaluate": evaluate,
                      "refilter": refilter}[args.command](config, args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
