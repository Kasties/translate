"""Collect experiment records from artifacts/ for the experiment tracker page.

Writes one JSON file per experiment to artifacts/experiments/<id>.json plus an
index.json. Each record is a self-contained document: data builds, curation and
generation runs, fine-tuning runs, EuroEval comparisons and throughput benchmarks.

    uv run python scripts/export-experiments.py
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CHILD_RUN = re.compile(r"-(gen|cur)-\d{4}$")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:190]


def model_name(report: dict) -> str | None:
    model = report.get("model")
    if isinstance(model, dict):
        return model.get("id")
    if isinstance(model, str):
        return model
    return (report.get("config") or {}).get("model", {}).get("id")


# --- Data generation, curation and builds -----------------------------------

DATA_STAGES = {
    "nlu_build": "build",
    "nlu_generation": "generation",
    "nlu_curation": "curation",
    "nlu_merge": "merge",
    "generation": "generation",
    "curation": "curation",
}


def data_record(run: Path) -> dict | None:
    report_path = run / "report.json"
    if not report_path.exists():
        return {
            "id": f"data-{slug(run.name)}",
            "kind": "data",
            "stage": "unknown",
            "family": "mhqa" if run.name.startswith("nvidia") else "nlu",
            "title": run.name,
            "status": "incomplete",
            "started": mtime(run),
            "path": rel(run),
            "summary": "No report.json; the run stopped before writing its report.",
        }
    report = load(report_path)
    if "human_target_rows" in report:
        return instruction_record(run, report, report_path)
    stage = DATA_STAGES.get(report.get("stage"), report.get("stage", "unknown"))
    family = "nlu" if str(report.get("stage", "")).startswith("nlu") else "mhqa"
    accepted = report.get("accepted", 0)
    rejected = report.get("rejected", 0)
    requested = report.get("target") or report.get("candidates") or report.get("requested")
    record = {
        "id": f"data-{slug(run.name)}",
        "kind": "data",
        "stage": stage,
        "family": family,
        "title": run.name,
        "status": "complete" if report.get("complete", True) else "incomplete",
        "started": mtime(report_path),
        "path": rel(run),
        "model": model_name(report),
        "split": report.get("split"),
        "tasks": report.get("tasks"),
        "metrics": {
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": round(accepted / (accepted + rejected), 4)
            if accepted + rejected
            else None,
        },
        "by_task": report.get("by_task") or {},
        "rejection_counts": report.get("rejection_counts") or {},
    }
    if requested:
        record["metrics"]["requested"] = requested
    sources = report.get("source_runs") or ([report["source_run"]] if report.get("source_run") else [])
    if sources:
        record["sources"] = [f"data-{slug(Path(s).name)}" for s in sources]
    if stage == "build":
        record["metrics"].update(
            target=report.get("target"),
            curated_before_dedup=report.get("curated_before_dedup"),
            surplus=report.get("surplus"),
            unique_source_articles=report.get("unique_source_articles"),
        )
        record["task_quotas"] = report.get("task_quotas") or {}
        record["batches"] = [
            {
                "offset": b.get("offset"),
                "generated": b.get("generated"),
                "curated": b.get("curated"),
                "tasks": b.get("tasks"),
            }
            for b in report.get("batches", [])
        ]
        record["summary"] = (
            f"{accepted} of {report.get('target')} target rows accepted over "
            f"{len(record['batches'])} batches"
            + ("" if report.get("complete") else "; build not complete")
        )
    else:
        record["summary"] = f"{accepted} accepted, {rejected} rejected"
    for key in ("solver_disagreements_on_controlled_tasks", "solver_attempts", "prefilter_rejected"):
        if key in report:
            record["metrics"][key] = report[key]
    note = report.get("limitation") or report.get("note")
    if note:
        record["note"] = note
    record["metrics"] = {k: v for k, v in record["metrics"].items() if v is not None}
    return record


def instruction_record(run: Path, report: dict, report_path: Path) -> dict:
    """Register-first native-target build (instruction-build): every attempt is journaled."""
    attempts, accepted = report.get("attempts", 0), report.get("accepted", 0)
    reasons: dict[str, int] = {}
    if (run / "rejected.jsonl").exists():
        for line in (run / "rejected.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                # Pydantic messages span lines; the first names the failing schema.
                reason = json.loads(line).get("reason", "unknown").splitlines()[0]
                reasons[reason] = reasons.get(reason, 0) + 1
    state = load(run / "state.json") if (run / "state.json").exists() else {}
    model = None
    for call in sorted((run / "calls").glob("*.json"))[:1]:
        model = (load(call).get("response") or {}).get("model")
    efficiency = report.get("efficiency") or {}
    complete = bool(report.get("complete"))
    record = {
        "id": f"data-{slug(run.name)}",
        "kind": "data",
        "stage": "instructions",
        "family": "registers",
        "title": run.name,
        "status": "complete" if complete else "incomplete",
        "started": mtime(report_path),
        "path": rel(run),
        "model": model,
        "split": state.get("split"),
        "metrics": {
            "accepted": accepted,
            "rejected": attempts - accepted,
            "acceptance_rate": round(accepted / attempts, 4) if attempts else None,
            "requested": report.get("target"),
            "attempts": attempts,
            "human_target_rows": report.get("human_target_rows"),
            "completion_tokens": efficiency.get("completion_tokens"),
        },
        "by_task": report.get("by_task") or {},
        "by_register": report.get("by_register") or {},
        "task_quotas": report.get("task_quotas") or {},
        "register_quotas": report.get("register_quotas") or {},
        "rejection_counts": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "summary": (f"{accepted} of {report.get('target')} rows accepted from {attempts} attempts"
                    + (f" ({accepted / attempts:.0%})" if attempts else "")
                    + f"; {report.get('human_target_rows')} with human-written answers"
                    + ("" if complete else "; build not complete")),
    }
    record["metrics"] = {k: v for k, v in record["metrics"].items() if v is not None}
    return record


# --- Fine-tuning ------------------------------------------------------------


def finetune_record(run: Path) -> dict | None:
    manifest_path = run / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = load(manifest_path)
    config = manifest.get("config", {})
    data = manifest.get("data", {})
    record = {
        "id": f"finetune-{slug(run.name)}",
        "kind": "finetune",
        "title": run.name,
        "status": manifest.get("status", "unknown"),
        "started": mtime(manifest_path),
        "path": rel(run),
        "model": config.get("model"),
        "params": {
            k: config.get(k)
            for k in (
                "epochs",
                "learning_rate",
                "batch_size",
                "gradient_accumulation_steps",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "max_length",
                "seed",
            )
        },
        "metrics": {
            k: v
            for k, v in (manifest.get("metrics") or {}).items()
            if k in ("train_loss", "train_runtime", "train_samples_per_second", "epoch")
        },
        "train_rows": data.get("rows"),
        "by_task": data.get("by_task") or {},
        "sources": [f"data-{slug(Path(data['train_file']).parent.name)}"] if data.get("train_file") else [],
        "tokens": manifest.get("tokens"),
        "gpu": manifest.get("gpu"),
    }
    if manifest.get("error"):
        record["error"] = manifest["error"]
        record["summary"] = manifest["error"]
    trainer_state = run / "trainer_state.json"
    if trainer_state.exists():
        record["loss"] = [
            {
                "step": entry["step"],
                "loss": round(entry["loss"], 5),
                "acc": round(entry.get("mean_token_accuracy", 0), 4),
                "lr": entry.get("learning_rate"),
            }
            for entry in load(trainer_state).get("log_history", [])
            if "loss" in entry
        ]
    if "summary" not in record:
        loss = record["metrics"].get("train_loss")
        runtime = record["metrics"].get("train_runtime")
        record["summary"] = (
            f"LoRA r={config.get('lora_rank')} on {data.get('rows')} rows"
            + (f", mean train loss {loss:.3f}" if loss is not None else "")
            + (f", {runtime / 60:.0f} min" if runtime else "")
        )
    return record


# --- EuroEval ---------------------------------------------------------------


def euroeval_scores(path: Path) -> dict:
    scores = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        details = row["eval_library"]["additional_details"]
        dataset = details["dataset"]
        entry = {"task": details.get("task"), "timestamp": row.get("evaluation_timestamp")}
        raw = json.loads(details.get("raw_results") or "[]")
        entry["failed"] = sum(len(r.get("failed_instances", [])) for r in raw)
        # Per-iteration scores, used for paired base/tuned differences, then dropped.
        entry["per_iteration"] = {key: [r[key] * 100 for r in raw if key in r] for key in ("mcc", "macro_f1")}
        for result in row["evaluation_results"]:
            name = result["evaluation_name"].removeprefix("test_")
            score = result["score_details"]
            entry[name] = round(score["score"], 3)
            interval = (score.get("uncertainty") or {}).get("confidence_interval")
            if interval:
                entry[f"{name}_lo"] = round(interval["lower"], 3)
                entry[f"{name}_hi"] = round(interval["upper"], 3)
        scores[dataset] = entry
    return scores


# Two-sided 95% Student t quantiles by degrees of freedom (normal beyond 30).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
       18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def paired_difference(base: list | None, tuned: list | None) -> dict | None:
    """Tuned minus base per iteration. EuroEval seeds each iteration's sample and few-shot
    examples by its index, so both models see the same data in the same iteration."""
    if not base or not tuned or len(base) != len(tuned) or len(base) < 2:
        return None
    diffs = [t - b for b, t in zip(base, tuned)]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
    df = n - 1
    t95 = T95.get(df) or next((v for k, v in sorted(T95.items()) if k >= df), 1.96)
    half = t95 * sd / n ** 0.5
    return {"diff": round(mean, 3), "lo": round(mean - half, 3), "hi": round(mean + half, 3),
            "t": round(mean / (sd / n ** 0.5), 2) if sd else None, "n": n,
            "tuned_better": sum(d > 0 for d in diffs)}


def euroeval_record(run: Path) -> dict | None:
    manifest_path = run / "manifest.json"
    summary_path = run / "summary.json"
    if not manifest_path.exists() and not summary_path.exists():
        return None
    manifest = load(manifest_path) if manifest_path.exists() else {}
    results = {}
    started = None
    for model_dir in sorted(p for p in run.iterdir() if p.is_dir() and p.name != "cache"):
        results_file = model_dir / "euroeval_benchmark_results.jsonl"
        if results_file.exists():
            for dataset, entry in euroeval_scores(results_file).items():
                results.setdefault(dataset, {})[model_dir.name] = entry
                started = min(filter(None, [started, entry["timestamp"]]), default=None)
    record = {
        "id": f"eval-{slug(run.name)}",
        "kind": "eval",
        "title": run.name,
        "status": manifest.get("status", "complete"),
        "started": started or mtime(manifest_path if manifest_path.exists() else summary_path),
        "path": rel(run),
        "diagnostic": not manifest_path.exists(),
        "params": {
            "iterations": manifest.get("iterations"),
            "split": manifest.get("split"),
            "few_shot": manifest.get("few_shot"),
            "generative_type": manifest.get("generative_type"),
            "euroeval_version": manifest.get("euroeval_version"),
            "prompt_stripping": bool(manifest.get("prompt_stripping")),
        },
        "results": results,
    }
    record["params"] = {k: v for k, v in record["params"].items() if v is not None}
    training = manifest.get("training_manifest")
    if training:
        record["model"] = training.get("config", {}).get("model")
        record["sources"] = [f"finetune-{slug(Path(training['config']['output_dir']).name)}"]
    elif manifest.get("tokenizer"):
        record["model"] = manifest["tokenizer"]["model"]
    if summary_path.exists():
        summary = load(summary_path)
        record["params"]["prompt_stripping"] = bool(summary.get("prompts_stripped"))
        record["parsing"] = {
            name: {
                "samples": s.get("samples"),
                "label_found": s.get("label_found"),
                "accuracy_all": s.get("accuracy_all"),
                "accuracy_when_found": s.get("accuracy_when_found"),
            }
            for name, s in summary.get("runs", {}).items()
        }
    if manifest.get("note"):
        record["note"] = manifest["note"]
        # Older manifests wrote the one-iteration caveat regardless of the iteration count.
        if (manifest.get("iterations") or 1) > 1 and "one iteration" in manifest["note"]:
            record["note"] = (f"Validation split, {manifest['iterations']} iterations through the local API "
                              "server; compare against this project's own base runs rather than leaderboard rows.")
    # Pipeline runs record their served names; the pilot used these two.
    pair = manifest.get("models") or ["gemma4-base", "gemma4-nlu"]
    paired = {}
    for dataset, by_model in results.items():
        base, tuned = by_model.get(pair[0], {}), by_model.get(pair[-1], {})
        for metric in ("mcc", "macro_f1"):
            diff = paired_difference((base.get("per_iteration") or {}).get(metric),
                                     (tuned.get("per_iteration") or {}).get(metric))
            if diff:
                paired.setdefault(dataset, {})[metric] = diff
    for by_model in results.values():
        for entry in by_model.values():
            entry.pop("per_iteration", None)
    if paired:
        record["paired"] = paired
    parts = []
    for dataset, by_model in results.items():
        base, tuned = by_model.get(pair[0], {}), by_model.get(pair[-1], {})
        if "mcc" in base and "mcc" in tuned:
            parts.append(f"{dataset} MCC {base['mcc']:.1f} → {tuned['mcc']:.1f}")
    if parts:
        record["summary"] = "; ".join(parts)
    elif record.get("parsing"):
        found = [
            f"{name}: {p['label_found']}/{p['samples']} labels parsed" for name, p in record["parsing"].items()
        ]
        record["summary"] = "; ".join(found)
    return record


def idioms_record(run: Path) -> dict | None:
    manifest_path = run / "manifest.json"
    summary_path = run / "summary.json"
    if not manifest_path.exists():
        return None
    manifest = load(manifest_path)
    summary = load(summary_path) if summary_path.exists() else {}
    results = {}
    for model, subsets in summary.get("models", {}).items():
        for subset, score in subsets.items():
            dataset = "swedish-idioms" if subset == "all" else f"swedish-idioms-{subset}"
            results.setdefault(dataset, {})[model] = {
                "task": "idiom minimal pairs",
                "accuracy": round(100 * score["accuracy"], 2),
                "accuracy_lo": round(100 * score["ci95"][0], 2),
                "accuracy_hi": round(100 * score["ci95"][1], 2),
                "correct": score["correct"],
                "total": score["total"],
            }
    record = {
        "id": f"eval-idioms-{slug(run.name)}",
        "kind": "eval",
        "title": f"idioms/{run.name}",
        "status": manifest.get("status", "unknown"),
        "started": mtime(manifest_path),
        "path": rel(run),
        "diagnostic": False,
        "params": {
            "benchmark": manifest.get("benchmark"),
            "dataset_revision": manifest.get("dataset_revision"),
            "items": manifest.get("items"),
            "metric": manifest.get("metric"),
        },
        "results": results,
        "paired": summary.get("paired"),
        "note": manifest.get("note"),
    }
    training = manifest.get("training_manifest")
    if training:
        record["model"] = training.get("config", {}).get("model")
        record["sources"] = [f"finetune-{slug(Path(training['config']['output_dir']).name)}"]
    elif manifest.get("tokenizer"):
        record["model"] = manifest["tokenizer"]["model"]
    # Pipeline runs record their served names; the pilot used these two.
    pair = manifest.get("models") or ["gemma4-base", "gemma4-nlu"]
    parts = []
    for dataset, by_model in results.items():
        base, tuned = by_model.get(pair[0]), by_model.get(pair[-1])
        if base and tuned:
            parts.append(f"{dataset} acc {base['accuracy']:.1f} → {tuned['accuracy']:.1f}")
    if parts:
        record["summary"] = "; ".join(parts)
    return record


# --- Throughput benchmarks --------------------------------------------------


def benchmark_record(path: Path) -> dict:
    bench = load(path)
    name = path.stem.removeprefix("benchmark-")
    spec = next((s for s in ("mtp", "ngram", "none") if s in name.split("-")), "unknown")
    results = bench.get("results", [])
    valid = sum(1 for r in results if r.get("valid_json"))
    tokens = sum(r.get("completion_tokens", 0) for r in results)
    return {
        "id": f"bench-{slug(name)}",
        "kind": "benchmark",
        "title": path.stem,
        "status": "complete",
        "started": mtime(path),
        "path": rel(path),
        "model": bench.get("model"),
        "params": {"speculation": spec, "parallel": bench.get("parallel"), "requests": bench.get("requests")},
        "metrics": {
            "output_tokens_per_second": round(bench.get("output_tokens_per_second", 0), 2),
            "seconds": round(bench.get("seconds", 0), 3),
            "completion_tokens": tokens,
            "valid_json_rate": round(valid / len(results), 3) if results else None,
        },
        "summary": f"{bench.get('output_tokens_per_second', 0):.0f} tok/s, speculation={spec}, "
        f"parallel={bench.get('parallel')}",
        "note": bench.get("note"),
    }


def collect() -> list[dict]:
    records = []
    runs = ARTIFACTS / "source2synth" / "runs"
    for run in sorted(p for p in runs.iterdir() if p.is_dir() and not CHILD_RUN.search(p.name)):
        records.append(data_record(run))
    for run in sorted(p for p in (ARTIFACTS / "finetuning").iterdir() if p.is_dir()):
        records.append(finetune_record(run))
    for run in sorted(p for p in (ARTIFACTS / "euroeval").iterdir() if p.is_dir()):
        records.append(euroeval_record(run))
    idioms = ARTIFACTS / "idioms"
    for run in sorted(p for p in idioms.iterdir() if p.is_dir()) if idioms.exists() else []:
        records.append(idioms_record(run))
    for path in sorted(ARTIFACTS.glob("benchmark-*.json")):
        records.append(benchmark_record(path))
    for record in filter(None, records):
        record["source"] = "exporter"
    return sorted(filter(None, records), key=lambda r: r["started"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "experiments")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = collect()
    for record in records:
        (args.output / f"{record['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    index = [{k: r.get(k) for k in ("id", "kind", "title", "status", "started", "summary")} for r in records]
    (args.output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} experiment records to {rel(args.output)}")


if __name__ == "__main__":
    main()
