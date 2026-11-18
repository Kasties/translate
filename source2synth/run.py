"""One-command experiment pipeline: sources -> generate -> sft -> eval.

    ./pipeline new                                   # interactive: pick data + models
    ./pipeline run configs/pipeline/my-exp.yaml      # all stages, resumes if rerun
    ./pipeline run my-exp --from sft --set sft.model=Qwen/Qwen3-8B --name qwen3-sft
    ./pipeline status my-exp
    ./pipeline doctor

Each stage runs in its own environment (.venv data generation, .venv-train
training, .venv-eval vLLM + EuroEval) and model servers are started and stopped
automatically. Stages that already finished are skipped; a finished stage whose
settings changed is refused so results are never silently mixed. See
docs/pipeline.md.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .common import ROOT, write_json

STAGES = ["sources", "generate", "sft", "eval"]
CONFIG_DIR = ROOT / "configs/pipeline"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
# Values the user may fill with arbitrary keys.
FREE_FORM = {("datagen", "overrides"), ("generator", "llamacpp_env"), ("reviewer", "extra_body")}
VENVS = {"data": ROOT / ".venv", "train": ROOT / ".venv-train", "eval": ROOT / ".venv-eval"}
LORA_RANKS = [8, 16, 32, 64, 128, 256, 320, 512]


class PipelineError(Exception):
    pass


# --- Configuration -----------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def check_keys(defaults: dict, override: dict, path: tuple = ()) -> None:
    for key, value in (override or {}).items():
        where = ".".join((*path, key))
        if key not in defaults:
            raise PipelineError(f"Unknown setting {where!r}; see configs/pipeline/default.yaml")
        if isinstance(value, dict) and isinstance(defaults[key], dict) and (*path, key) not in FREE_FORM:
            check_keys(defaults[key], value, (*path, key))


def config_path(spec: str) -> Path:
    for folder in (Path(), CONFIG_DIR, CONFIG_DIR / "examples"):
        for candidate in (folder / spec, folder / f"{spec}.yaml"):
            if candidate.is_file():
                return candidate.resolve()
    raise PipelineError(f"No experiment config {spec!r} (looked in . and {CONFIG_DIR.relative_to(ROOT)})")


def read_layers(path: Path, seen: tuple = ()) -> dict:
    if path in seen:
        raise PipelineError(f"Circular extends: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        base = read_layers((path.parent / parent).resolve(), (*seen, path))
        base.pop("name", None)  # A child is its own experiment.
        data = deep_merge(base, data)
    return data


def parse_set(item: str) -> tuple[list[str], object]:
    key, sep, value = item.partition("=")
    if not sep or not key:
        raise PipelineError(f"--set expects key=value, got {item!r}")
    return key.split("."), yaml.safe_load(value) if value else None


def load_experiment(spec: str | None, sets: list[str] = (), name: str | None = None,
                    profile: str | None = None) -> dict:
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    path = config_path(spec) if spec else None
    layers = read_layers(path) if path and path != DEFAULT_CONFIG else {}
    check_keys(defaults, layers)
    config = deep_merge(defaults, layers)
    # A hardware profile (e.g. rtx4090-pilot) is layered over any experiment, before --set.
    profile = profile or os.environ.get("S2S_PIPELINE_PROFILE") or None
    if profile == "none":
        profile = None
    if profile:
        hardware = read_layers(config_path(profile))
        hardware.pop("name", None)
        check_keys(defaults, hardware)
        config = deep_merge(config, hardware)
    for item in sets:
        keys, value = parse_set(item)
        check_keys(defaults, _nest(keys, value))
        config = deep_merge(config, _nest(keys, value))
    config["name"] = name or config.get("name") or (path.stem if path and path != DEFAULT_CONFIG else None)
    if not config["name"]:
        raise PipelineError("Give the experiment a name: --name <name> or name: in the config")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config["name"]):
        raise PipelineError("Experiment names may only use letters, digits, '.', '_' and '-'")
    config["_file"] = str(path) if path else None
    config["_profile"] = profile
    config["_sets"] = list(sets)
    return config


def _nest(keys: list[str], value) -> dict:
    for key in reversed(keys):
        value = {key: value}
    return value


# --- Paths and references ------------------------------------------------------


class Paths:
    def __init__(self, name: str):
        self.name = name
        self.root = ROOT / "artifacts/pipeline" / name
        self.data_dir = self.root / "data"
        self.passages = self.data_dir / "nlu/passages.jsonl"
        self.datagen_config = self.root / "source2synth.yaml"
        self.finetune_config = self.root / "finetune.yaml"
        self.logs = self.root / "logs"
        self.state = self.root / "pipeline.json"
        self.datagen_output = ROOT / "artifacts/source2synth"
        self.build = self.datagen_output / "runs" / name
        self.sft = ROOT / "artifacts/finetuning" / name
        self.euroeval = ROOT / "artifacts/euroeval" / name
        self.idioms = ROOT / "artifacts/idioms" / name


def resolve_ref(value: str, kind: str, name: str) -> str:
    """Expand experiment:<name> to that experiment's output of the given kind."""
    if not isinstance(value, str) or not value.startswith("experiment:"):
        return value
    other = Paths(value.split(":", 1)[1])
    path = {"passages": other.passages, "train": other.build / "train.jsonl",
            "adapter": other.sft / "adapter"}[kind]
    if other.name == name:
        raise PipelineError(f"{value} refers to this experiment itself")
    return str(path)


def display(path: Path | str) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)


def train_files(config: dict, paths: Paths) -> list[str]:
    value = config["sft"]["train_file"]
    if value is None:
        return [display(paths.build / "train.jsonl")]
    values = value if isinstance(value, list) else [value]
    return [v if v.startswith("hf:") else display(resolve_ref(v, "train", paths.name)) for v in values]


def eval_target(config: dict, paths: Paths) -> dict:
    evaluation, sft = config["eval"], config["sft"]
    model = evaluation["model"] or sft["model"]
    revision = evaluation["revision"] if evaluation["model"] else (evaluation["revision"] or sft["revision"])
    adapter = evaluation["adapter"]
    if adapter is None:
        adapter = str(paths.sft / "adapter")
    elif adapter == "none":
        adapter = None
    else:
        adapter = resolve_ref(adapter, "adapter", paths.name)
        adapter = str(adapter if Path(adapter).is_absolute() else ROOT / adapter)
    return {"model": model, "revision": revision, "adapter": adapter}


# --- State -----------------------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(paths: Paths) -> dict:
    if paths.state.exists():
        return json.loads(paths.state.read_text(encoding="utf-8"))
    return {"name": paths.name, "stages": {}}


def save_state(paths: Paths, state: dict) -> None:
    state["updated"] = now()
    write_json(paths.state, state)


def stage_settings(stage: str, config: dict, paths: Paths) -> dict:
    """The settings that define a stage's result; changing them needs a new experiment."""
    if stage == "sources":
        data = config["data"]
        source = data["source"]
        return {"source": source, "options": data.get(source),
                "fields": data["fields"] if source in {"hf", "local"} else None,
                "datagen_config": config["datagen"]["config"], "overrides": config["datagen"]["overrides"]}
    if stage == "generate":
        datagen = {k: v for k, v in config["datagen"].items()
                   if k not in {"max_batches", "require_complete", "refresh_inputs", "upload_repo"}}
        settings = {"model": config["generator"]["model"], "revision": config["generator"]["revision"],
                    "datagen": datagen}
        reviewer = config["reviewer"]
        if reviewer["model"]:  # Only when set, so experiments without a reviewer keep their settings.
            settings["reviewer"] = {k: reviewer[k] for k in ("model", "revision", "extra_body")}
            settings["generator_vllm_args"] = config["generator"]["vllm_args"]
        return settings
    if stage == "sft":
        # Where the adapter is uploaded afterwards does not change it.
        settings = {k: v for k, v in config["sft"].items() if k != "upload_repo"}
        return {**settings, "train_file": train_files(config, paths)}
    # Serving knobs and which benchmarks are enabled do not change a finished result.
    operational = {"port", "gpu_memory_utilization", "max_num_seqs"}
    settings = {k: v for k, v in config["eval"].items() if k not in operational}
    settings["euroeval"] = {k: v for k, v in settings["euroeval"].items() if k != "enabled"}
    settings["idioms"] = {k: v for k, v in settings["idioms"].items() if k != "enabled"}
    return {**settings, **eval_target(config, paths)}


def stage_status(stage: str, config: dict, paths: Paths) -> tuple[str, str]:
    """Return (done|partial|missing, detail) from the artifacts on disk."""
    if stage == "sources":
        manifest = paths.passages.parent / "manifest.json"
        if paths.passages.exists() and manifest.exists():
            count = json.loads(manifest.read_text(encoding="utf-8"))["passages"]
            return "done", f"{count} passages in {display(paths.passages)}"
        return "missing", ""
    if stage == "generate":
        report = paths.build / "report.json"
        if report.exists():
            data = json.loads(report.read_text(encoding="utf-8"))
            state = "done" if data.get("complete") else "partial"
            return state, f"{data['accepted']}/{data['target']} accepted rows in {display(paths.build)}"
        return ("partial", "build started") if paths.build.exists() else ("missing", "")
    if stage == "sft":
        manifest = paths.sft / "manifest.json"
        if manifest.exists():
            status = json.loads(manifest.read_text(encoding="utf-8")).get("status")
            if status == "complete":
                return "done", f"adapter in {display(paths.sft / 'adapter')}"
            return "partial", f"status {status!r} in {display(paths.sft)}"
        if paths.sft.exists() and any(paths.sft.iterdir()):
            return "partial", f"files in {display(paths.sft)}"
        return "missing", ""
    parts, states = [], []
    for enabled, directory in ((config["eval"]["euroeval"]["enabled"], paths.euroeval),
                               (config["eval"]["idioms"]["enabled"], paths.idioms)):
        if not enabled:
            continue
        state = output_status(directory)
        states.append(state)
        parts.append(f"{display(directory)}: {state}")
    if not states:
        return "done", "no benchmarks enabled"
    overall = "done" if all(s == "done" for s in states) else (
        "missing" if all(s == "missing" for s in states) else "partial")
    return overall, "; ".join(parts)


def output_status(directory: Path) -> str:
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete":
        return "done"
    return "partial" if directory.exists() and any(directory.iterdir()) else "missing"


def guard_changed(stage: str, config: dict, paths: Paths, state: dict) -> None:
    recorded = state["stages"].get(stage, {}).get("settings")
    current = stage_settings(stage, config, paths)
    if recorded is None or recorded == current:
        return
    changed = sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
    raise PipelineError(
        f"Stage {stage!r} of {paths.name!r} already ran with different settings ({', '.join(changed)}).\n"
        f"Use a new experiment name, e.g. --name {paths.name}-2, and reuse earlier outputs with "
        f"experiment:{paths.name} (see docs/pipeline.md)."
    )


# --- Processes and servers -------------------------------------------------------


def environment() -> dict:
    """The same caches as scripts/experiment-env.sh, plus the repository on PYTHONPATH."""
    env = dict(os.environ)
    cache = ROOT / "artifacts/cache"
    env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
    # A machine with a shared model cache sets HF_HUB_CACHE and keeps HF_HOME, which holds the token.
    if "HF_HUB_CACHE" not in env:
        env["HF_HOME"] = env.get("TDDE19_HF_HOME", str(cache / "huggingface"))
        env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
    env.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    env.setdefault("VLLM_CACHE_ROOT", str(cache / "vllm"))
    env.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # FlashInfer JIT-compiles its sampler with the system nvcc, which must match the
    # wheels' CUDA 13; vLLM's PyTorch sampler needs no compiler and gives the same results.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    token = Path.home() / ".cache/huggingface/token"
    if "HF_TOKEN" not in env and "HF_TOKEN_PATH" not in env and token.is_file():
        env["HF_TOKEN_PATH"] = str(token)  # `hf auth login` token, despite the moved HF_HOME.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env["PYTHONUNBUFFERED"] = "1"
    return env


def python(venv: str) -> str:
    return str(VENVS[venv] / "bin/python")


def run_step(command: list[str], log: Path, dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    executable = Path(command[0])
    if executable.is_absolute() and not executable.exists():
        raise PipelineError(f"{display(executable)} is missing; run scripts/setup.sh")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] $ {shlex.join(command)}\n")
        process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        for line in process.stdout:
            # Flush so progress shows up when the pipeline runs detached (nohup, log file).
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise PipelineError(f"Command failed with exit code {code}; full log: {display(log)}")


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def served_models(port: int) -> set[str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
        return {m["id"] for m in json.load(response)["data"]}


class Server:
    """Start a model server in its own process group and always stop it again."""

    def __init__(self, label: str, command: list[str], port: int, models: list[str], log: Path,
                 env: dict | None = None, timeout: int = 3600):
        self.label, self.command, self.port, self.models = label, command, port, models
        self.log, self.env, self.timeout, self.process = log, env or {}, timeout, None

    def __enter__(self):
        if port_in_use(self.port):
            raise PipelineError(
                f"Port {self.port} is already in use. Stop the old server (e.g. from an interrupted "
                f"run), or set the port / backend: external in the config.")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        print(f"Starting {self.label} server; log: {display(self.log)}", flush=True)
        handle = self.log.open("a", encoding="utf-8")
        handle.write(f"\n[{now()}] $ {shlex.join(self.command)}\n")
        handle.flush()
        self.log_start = handle.tell()  # earlier runs' output stays in the log; inspect only this run
        self.process = subprocess.Popen(self.command, cwd=ROOT, env={**environment(), **self.env},
                                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return self._wait_until_ready()
        except BaseException:
            # __exit__ does not run when __enter__ fails; never leave a server holding the GPU.
            self.__exit__()
            raise

    def _wait_until_ready(self):
        started, last_note = time.monotonic(), 0.0
        while True:
            if self.process.poll() is not None:
                with self.log.open("rb") as log:
                    log.seek(self.log_start)
                    output = log.read().decode("utf-8", errors="replace")
                hint = ("\nThe GPU ran out of memory. On a 24 GB card use the 4-bit/llama.cpp profile: "
                        "--profile rtx4090-pilot (or export S2S_PIPELINE_PROFILE=rtx4090-pilot)."
                        if re.search(r"out of memory|OutOfMemoryError|failed with OOM", output, re.I) else "")
                raise PipelineError(f"{self.label} server exited with code {self.process.returncode}:\n"
                                    f"{tail(self.log)}\nFull log: {display(self.log)}{hint}")
            try:
                if set(self.models) <= served_models(self.port):
                    print(f"{self.label} server ready: {', '.join(self.models)}", flush=True)
                    return self
            except OSError:
                pass
            elapsed = time.monotonic() - started
            if elapsed > self.timeout:
                raise PipelineError(f"{self.label} server not ready after {self.timeout}s; see {display(self.log)}")
            if elapsed - last_note >= 60:
                last_note = elapsed
                lines = tail(self.log, 1).strip()
                print(f"  waiting for {self.label} server ({elapsed:.0f}s; downloads can take a while)"
                      + (f": {lines[-160:]}" if lines else ""), flush=True)
            time.sleep(5)

    def __exit__(self, *exc):
        if self.process and self.process.poll() is None:
            print(f"Stopping {self.label} server", flush=True)
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        return False


class DryServer:
    def __init__(self, label: str, command: list[str], *args, **kwargs):
        self.label, self.command = label, command

    def __enter__(self):
        print(f"[{self.label} server] $ {shlex.join(self.command)}", flush=True)
        return self

    def __exit__(self, *exc):
        return False


def tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def is_multimodal(model: str, revision: str | None) -> bool:
    try:
        local = Path(model)
        if (local / "config.json").is_file():
            data = json.loads((local / "config.json").read_text(encoding="utf-8"))
        else:
            os.environ.update({k: v for k, v in environment().items() if k.startswith("HF_")})
            from huggingface_hub import hf_hub_download

            data = json.loads(Path(hf_hub_download(model, "config.json", revision=revision)).read_text())
    except Exception as error:  # Offline or gated: let vLLM report the real problem.
        print(f"Could not read {model} config.json ({type(error).__name__}); "
              "assuming a text-only model", flush=True)
        return False
    return any(key in data for key in ("vision_config", "audio_config"))


def vllm_command(model: str, revision: str | None, served: str, port: int, settings: dict) -> list[str]:
    command = [str(VENVS["eval"] / "bin/vllm"), "serve", model, "--served-model-name", served,
               "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
               "--max-model-len", str(settings["max_model_len"]),
               "--gpu-memory-utilization", str(settings["gpu_memory_utilization"])]
    if revision:
        command += ["--revision", revision]
    only = settings["language_model_only"]
    if only is True or (only == "auto" and is_multimodal(model, revision)):
        command.append("--language-model-only")
    return command


# --- Stages ------------------------------------------------------------------------


def endpoint(config: dict) -> str:
    generator = config["generator"]
    if generator["backend"] == "external":
        if not generator["endpoint"]:
            raise PipelineError("generator.backend external needs generator.endpoint")
        return generator["endpoint"]
    return f"http://127.0.0.1:{generator['port']}/v1"


def reviewer_endpoint(config: dict) -> str:
    reviewer = config["reviewer"]
    if reviewer["backend"] == "external":
        if not reviewer["endpoint"]:
            raise PipelineError("reviewer.backend external needs reviewer.endpoint")
        return reviewer["endpoint"]
    return f"http://127.0.0.1:{reviewer['port']}/v1"


def reviewer_server(config: dict, paths: Paths, dry_run: bool):
    reviewer = config["reviewer"]
    if not reviewer["model"] or config["datagen"]["mode"] != "instructions" or reviewer["backend"] == "external":
        return None
    if reviewer["backend"] != "vllm":
        raise PipelineError("reviewer.backend must be vllm or external")
    command = vllm_command(reviewer["model"], reviewer["revision"], reviewer["model"], reviewer["port"], reviewer)
    command += ["--max-num-seqs", str(reviewer["max_parallel_requests"]), *reviewer["vllm_args"]]
    factory = DryServer if dry_run else Server
    return factory("reviewer", command, reviewer["port"], [reviewer["model"]], paths.logs / "reviewer-server.log")


def write_datagen_config(config: dict, paths: Paths) -> None:
    source = ROOT / config["datagen"]["config"]
    base = yaml.safe_load(source.read_text(encoding="utf-8"))
    merged = deep_merge(base, config["datagen"]["overrides"])
    merged["model"].update(id=config["generator"]["model"],
                           max_parallel_requests=config["generator"]["max_parallel_requests"])
    if config["generator"]["backend"] != "external" or config["generator"]["endpoint"]:
        merged["model"]["endpoint"] = endpoint(config)
    reviewer = config["reviewer"]
    if reviewer["model"]:
        merged.setdefault("instruction_data", {})["reviewer"] = {
            "id": reviewer["model"], "endpoint": reviewer_endpoint(config),
            "extra_body": reviewer["extra_body"] or {}}
    paths.root.mkdir(parents=True, exist_ok=True)
    text = f"# Generated by ./pipeline from {config['datagen']['config']}; edit the experiment config instead.\n"
    paths.datagen_config.write_text(text + yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
                                    encoding="utf-8")


def datagen_cli(paths: Paths) -> list[str]:
    return [python("data"), "-m", "source2synth", "--config", display(paths.datagen_config),
            "--data-dir", display(paths.data_dir)]


def run_sources(config: dict, paths: Paths, dry_run: bool) -> None:
    data, fields = config["data"], config["data"]["fields"]
    source = data["source"]
    write_datagen_config(config, paths)
    if source.startswith("experiment:"):
        command = datagen_cli(paths) + ["prepare-sources", "--source", "local",
                                        "--path", display(resolve_ref(source, "passages", paths.name))]
    elif source == "registers":
        command = datagen_cli(paths) + ["prepare-registers", "--cache-dir", data["registers"]["cache_dir"]]
        if data["registers"]["offline"]:
            command.append("--offline")
    elif source == "korp":
        command = datagen_cli(paths) + ["prepare-nlu", "--cache-dir", "data/raw"]
        if data["korp"]["offline"]:
            command.append("--offline")
    elif source in {"hf", "local"}:
        options = data[source]
        command = datagen_cli(paths) + ["prepare-sources", "--source", source]
        if source == "hf":
            if not options["dataset"]:
                raise PipelineError("data.hf.dataset is required for data.source: hf")
            command += ["--dataset", options["dataset"], "--split", options["split"]]
            command += ["--subset", options["subset"]] if options["subset"] else []
            command += ["--revision", options["revision"]] if options["revision"] else []
        else:
            if not options["path"]:
                raise PipelineError("data.local.path is required for data.source: local")
            command += ["--path", options["path"]]
        for flag, key in (("--text-field", "text"), ("--title-field", "title"),
                          ("--id-field", "id"), ("--url-field", "url")):
            command += [flag, fields[key] or ""]
        command += ["--topic-field", fields["topic"]] if fields["topic"] else []
        command += ["--max-documents", str(options["max_documents"])] if options["max_documents"] else []
        command += ["--license", options["license"]] if options["license"] else []
    else:
        raise PipelineError(f"data.source must be registers, korp, hf, local or experiment:<name>, not {source!r}")
    run_step(command, paths.logs / "sources.log", dry_run)


def generator_server(config: dict, paths: Paths, dry_run: bool):
    generator = config["generator"]
    log = paths.logs / "generator-server.log"
    factory = DryServer if dry_run else Server
    if generator["backend"] == "vllm":
        command = vllm_command(generator["model"], generator["revision"], generator["model"],
                               generator["port"], generator)
        command += ["--max-num-seqs", str(generator["max_parallel_requests"]), *generator["vllm_args"]]
        return factory("generator", command, generator["port"], [generator["model"]], log)
    if generator["backend"] == "llamacpp":
        env = {"S2S_MODEL": generator["model"], "S2S_PORT": str(generator["port"]),
               "S2S_PARALLEL": str(generator["max_parallel_requests"]),
               **{k: str(v) for k, v in generator["llamacpp_env"].items()}}
        return factory("generator", ["bash", "scripts/serve-local.sh"], generator["port"],
                       [generator["model"]], log, env=env)
    if generator["backend"] == "external":
        return None
    raise PipelineError("generator.backend must be vllm, llamacpp or external")


def run_generate(config: dict, paths: Paths, dry_run: bool) -> None:
    datagen = config["datagen"]
    mode = datagen["mode"]
    if mode not in {"nlu", "instructions"}:
        raise PipelineError("datagen.mode must be nlu or instructions")
    if mode == "instructions" and config["data"]["source"] != "registers":
        raise PipelineError("datagen.mode: instructions requires data.source: registers")
    if not dry_run and not paths.passages.exists():
        raise PipelineError(f"No source passages at {display(paths.passages)}; run the sources stage first")
    write_datagen_config(config, paths)
    command = datagen_cli(paths) + [
        "--output-dir", display(paths.datagen_output), "--endpoint", endpoint(config),
        "instruction-build" if mode == "instructions" else "nlu-build",
        "--target", str(datagen["target"]), "--batch-size", str(datagen["batch_size"]),
        "--max-batches", str(datagen["max_batches"]), "--split", datagen["split"],
        *(["--tasks", *datagen["tasks"]] if mode == "nlu" else []),
        "--model", config["generator"]["model"], "--run-name", paths.name]
    if mode == "nlu" and datagen["balance_tasks"]:
        command.append("--balance-tasks")
    if (paths.build / "state.json").exists():
        command.append("--resume")
        if datagen["refresh_inputs"]:
            if mode == "instructions":
                raise PipelineError("Instruction builds require unchanged inputs; choose a new experiment name")
            command.append("--refresh-inputs")
    elif paths.build.exists() and not dry_run:
        raise PipelineError(f"{display(paths.build)} exists but is not an nlu-build; choose another name")
    # The reviewer starts after the generator, which then has already taken its share of the GPU.
    with contextlib.ExitStack() as servers:
        for server in (generator_server(config, paths, dry_run), reviewer_server(config, paths, dry_run)):
            if server is not None:
                servers.enter_context(server)
        run_step(command, paths.logs / "generate.log", dry_run)
    if dry_run:
        return
    report = json.loads((paths.build / "report.json").read_text(encoding="utf-8"))
    if not report["complete"]:
        message = (f"Generation reached {report['accepted']}/{report['target']} rows after this run's "
                   f"batch limit ({report['by_task']}). Rerun the stage to add batches.")
        if datagen["require_complete"]:
            raise PipelineError(message)
        print(f"Warning: {message} Continuing with the rows accepted so far.", flush=True)


def run_sft(config: dict, paths: Paths, dry_run: bool) -> None:
    sft = config["sft"]
    files = train_files(config, paths)
    if not dry_run:
        for spec in files:
            if not spec.startswith("hf:") and not (ROOT / spec).exists():
                raise PipelineError(f"Training data {spec} does not exist"
                                    + (" (run the generate stage first)" if sft["train_file"] is None else ""))
        if paths.sft.exists() and any(paths.sft.iterdir()):
            raise PipelineError(f"{display(paths.sft)} holds an unfinished training run. Inspect "
                                f"{display(paths.logs / 'sft.log')}, then delete the directory to retry.")
    settings = {k: v for k, v in sft.items() if k not in {"max_steps", "upload_repo"}}
    settings.update(train_file=files if len(files) > 1 else files[0], output_dir=display(paths.sft))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.finetune_config.write_text(
        "# Generated by ./pipeline; edit the experiment config instead.\n"
        + yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
    command = [python("train"), "-m", "source2synth.finetune", "--config", display(paths.finetune_config),
               "--max-steps", str(sft["max_steps"])]
    run_step(command, paths.logs / "sft.log", dry_run)


def hub_api():
    """An HfApi with the login token the subprocesses use, even when HF_HOME was moved."""
    from huggingface_hub import HfApi

    env = environment()
    token = env.get("HF_TOKEN") or (Path(env["HF_TOKEN_PATH"]).read_text(encoding="utf-8").strip()
                                    if env.get("HF_TOKEN_PATH") else None)
    return HfApi(token=token)


def upload_build(config: dict, paths: Paths, state: dict, dry_run: bool) -> None:
    """Upload a finished build's exports to datagen.upload_repo under <experiment>/data/.

    Runs whenever generation is done, also when it was skipped, so a failed upload is retried;
    a build extended with more batches is uploaded again because its report changes.
    """
    repo = config["datagen"]["upload_repo"]
    if not repo or stage_status("generate", config, paths)[0] != "done":
        return
    files = [paths.build / name for name in ("train.jsonl", "accepted.jsonl", "rejected.jsonl",
                                             "report.json", "state.json") if (paths.build / name).exists()]
    digest = fingerprint_file(paths.build / "report.json")
    marker = paths.build / "uploaded.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("repo") == repo and saved.get("report_sha256") == digest:
            print(f"Data already uploaded to {repo}/{paths.name}/data", flush=True)
            return
    print(f"Uploading {len(files)} data files to {repo}/{paths.name}/data", flush=True)
    if dry_run:
        return
    from huggingface_hub import CommitOperationAdd

    api = hub_api()
    api.create_repo(repo, private=True, exist_ok=True)
    commit = api.create_commit(repo, commit_message=f"Add generated data {paths.name}", operations=[
        CommitOperationAdd(path_in_repo=f"{paths.name}/data/{path.name}", path_or_fileobj=str(path))
        for path in files])
    record = {"repo": repo, "path": f"{paths.name}/data", "commit": commit.oid, "url": commit.commit_url,
              "report_sha256": digest, "uploaded": now()}
    write_json(marker, record)
    state["stages"].setdefault("generate", {})["upload"] = record
    save_state(paths, state)
    print(f"Uploaded: {commit.commit_url}", flush=True)


def fingerprint_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def upload_sft(config: dict, paths: Paths, state: dict, dry_run: bool) -> None:
    """Upload a finished adapter to sft.upload_repo under <experiment>/, once, then delete checkpoints.

    Runs whenever the sft stage is done, also when it was skipped, so a failed upload is retried
    on the next run and earlier experiments can be uploaded with --only sft.
    """
    repo = config["sft"]["upload_repo"]
    if not repo or stage_status("sft", config, paths)[0] != "done":
        return
    marker = paths.sft / "uploaded.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("repo") == repo:
        print(f"Adapter already uploaded to {repo}/{paths.name}", flush=True)
        return
    files = [path for path in sorted(paths.sft.rglob("*")) if path.is_file()
             and not path.relative_to(paths.sft).parts[0].startswith("checkpoint-")]
    if paths.finetune_config.exists():
        files.append(paths.finetune_config)
    print(f"Uploading {len(files)} files (adapter, manifest, finetune.yaml) to {repo}/{paths.name}", flush=True)
    if dry_run:
        return
    from huggingface_hub import CommitOperationAdd

    api = hub_api()
    api.create_repo(repo, private=True, exist_ok=True)
    operations = [CommitOperationAdd(
        path_in_repo=f"{paths.name}/{path.relative_to(paths.sft if path.is_relative_to(paths.sft) else paths.root)}",
        path_or_fileobj=str(path)) for path in files]
    commit = api.create_commit(repo, operations=operations, commit_message=f"Add adapter {paths.name}")
    record = {"repo": repo, "path": paths.name, "commit": commit.oid, "url": commit.commit_url, "uploaded": now()}
    write_json(marker, record)
    for checkpoint in sorted(paths.sft.glob("checkpoint-*")):
        shutil.rmtree(checkpoint)
    state["stages"].setdefault("sft", {})["upload"] = record
    save_state(paths, state)
    print(f"Uploaded: {commit.commit_url}; deleted the trainer checkpoints", flush=True)


def run_eval(config: dict, paths: Paths, dry_run: bool) -> None:
    evaluation = config["eval"]
    target = eval_target(config, paths)
    adapter = target["adapter"]
    names = [evaluation["names"]["base"]] + ([evaluation["names"]["tuned"]] if adapter else [])
    if not dry_run and adapter and not (Path(adapter) / "adapter_config.json").is_file():
        raise PipelineError(f"No adapter at {display(adapter)}; run the sft stage first, "
                            "or set eval.adapter: none to evaluate only the base model")
    todo = []
    for enabled, directory in ((evaluation["euroeval"]["enabled"], paths.euroeval),
                               (evaluation["idioms"]["enabled"], paths.idioms)):
        state = output_status(directory) if enabled else "done"
        if state == "partial" and not dry_run:
            raise PipelineError(f"{display(directory)} holds an unfinished evaluation; "
                                "delete it to rerun that benchmark")
        todo.append(state != "done")
    run_euroeval, run_idioms = todo
    training_run = str(Path(adapter).parent) if adapter and (Path(adapter).parent / "manifest.json").exists() else "none"
    api = f"http://127.0.0.1:{evaluation['port']}/v1"
    tokenizer = ["--tokenizer", target["model"]] + (
        ["--tokenizer-revision", target["revision"]] if target["revision"] else [])
    commands = []
    if run_euroeval:
        commands.append([python("eval"), "-m", "source2synth.euroeval", "--api-base", api,
                         "--datasets", *evaluation["euroeval"]["datasets"],
                         "--iterations", str(evaluation["euroeval"]["iterations"]),
                         "--generative-type", evaluation["euroeval"]["generative_type"],
                         "--max-context-length", str(evaluation["max_model_len"]),
                         "--output-dir", display(paths.euroeval), "--training-run", display(training_run),
                         "--models", *names, *tokenizer])
    if run_idioms:
        commands.append([python("eval"), "-m", "source2synth.idioms", "--api-base", api,
                         "--output-dir", display(paths.idioms), "--training-run", display(training_run),
                         "--models", *names, *tokenizer]
                        + (["--limit", str(evaluation["idioms"]["limit"])] if evaluation["idioms"]["limit"] else []))
    if not commands:
        return
    command = vllm_command(target["model"], target["revision"], names[0], evaluation["port"], evaluation)
    command += ["--max-num-seqs", str(evaluation["max_num_seqs"])]
    if evaluation["quantization"] == "bitsandbytes":
        command += ["--quantization", "bitsandbytes"]
    elif evaluation["quantization"] != "none":
        raise PipelineError("eval.quantization must be none or bitsandbytes")
    if adapter:
        rank = 16
        if (Path(adapter) / "adapter_config.json").is_file():
            rank = json.loads((Path(adapter) / "adapter_config.json").read_text())["r"]
        command += ["--enable-lora", "--max-lora-rank", str(next(r for r in LORA_RANKS if r >= rank)),
                    "--lora-modules", f"{names[1]}={adapter}"]
    command += evaluation["vllm_args"]
    factory = DryServer if dry_run else Server
    with factory("eval", command, evaluation["port"], names, paths.logs / "eval-server.log"):
        for step in commands:
            run_step(step, paths.logs / "eval.log", dry_run)


RUNNERS = {"sources": run_sources, "generate": run_generate, "sft": run_sft, "eval": run_eval}


def run(config: dict, first: str, last: str, dry_run: bool = False) -> None:
    paths = Paths(config["name"])
    selected = STAGES[STAGES.index(first): STAGES.index(last) + 1]
    if not selected:
        raise PipelineError(f"--from {first} comes after --to {last}")
    state = load_state(paths)
    print(f"Experiment {paths.name}: stages {' -> '.join(selected)}"
          + (f" (profile {config['_profile']})" if config.get("_profile") else "")
          + (" (dry run: only prints commands and writes the resolved configs)" if dry_run else ""),
          flush=True)
    if not dry_run:
        state["config"] = {k: v for k, v in config.items() if not k.startswith("_")}
        state["config_file"] = config["_file"]
        save_state(paths, state)
    for stage in selected:
        status, detail = stage_status(stage, config, paths)
        print(f"\n=== {stage} ({status}{': ' + detail if detail else ''})", flush=True)
        recorded = state["stages"].get(stage, {}).get("status")
        # A build that stopped at its batch limit is usable; only add batches when asked.
        if stage == "generate" and status == "partial" and recorded == "partial" and stage != first:
            status = "done"
            print("Stopped at its batch limit earlier; using the rows it has. "
                  "Run with --from generate to add batches.", flush=True)
        if status == "done":
            if dry_run:
                print("Already complete; would skip.", flush=True)
                continue
            guard_changed(stage, config, paths, state)
            state["stages"].setdefault(stage, {}).setdefault("settings", stage_settings(stage, config, paths))
            save_state(paths, state)
            print("Already complete; skipping.", flush=True)
            if stage == "sft":
                upload_sft(config, paths, state, dry_run)
            if stage == "generate":
                upload_build(config, paths, state, dry_run)
            continue
        if not dry_run:
            guard_changed(stage, config, paths, state)
            state["stages"][stage] = {"status": "running", "started": now(),
                                      "settings": stage_settings(stage, config, paths)}
            save_state(paths, state)
        try:
            RUNNERS[stage](config, paths, dry_run)
        except BaseException as error:
            if not dry_run:
                state["stages"][stage].update(status="failed", finished=now(), error=str(error)[:2000])
                save_state(paths, state)
            raise
        if not dry_run:
            status, detail = stage_status(stage, config, paths)
            state["stages"][stage].update(status="complete" if status == "done" else status,
                                          finished=now(), result=detail)
            save_state(paths, state)
        if stage == "sft":
            upload_sft(config, paths, state, dry_run)
        if stage == "generate":
            upload_build(config, paths, state, dry_run)
    if not dry_run:
        print(f"\nDone. Record: {display(paths.state)}", flush=True)


# --- Commands ----------------------------------------------------------------------


def print_status(config: dict) -> None:
    paths = Paths(config["name"])
    print(f"Experiment {paths.name}" + (f" ({display(config['_file'])})" if config["_file"] else ""))
    next_stage = None
    for stage in STAGES:
        status, detail = stage_status(stage, config, paths)
        mark = {"done": "✓", "partial": "…", "missing": "·"}[status]
        print(f"  {mark} {stage:<9} {status:<8} {detail}")
        if status != "done" and next_stage is None:
            next_stage = stage
    if next_stage:
        # Repeat everything that shaped this config, or the hint can run different settings.
        file = config["_file"]
        command = ["./pipeline", "run", *([display(file)] if file else []),
                   *(["--name", paths.name] if not file or Path(file).stem != paths.name else []),
                   *(["--profile", config["_profile"]] if config.get("_profile") else []),
                   *(arg for item in config.get("_sets", []) for arg in ("--set", item)),
                   "--from", next_stage]
        print(f"\nContinue with: {shlex.join(command)}")


def ask(prompt: str, default: str | None = None, choices: list[str] | None = None) -> str:
    """Prompt until valid; default None means an answer is required, "" allows none."""
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip() or (default or "")
        if not answer and default == "":
            return ""
        if answer and (not choices or answer in choices or any(answer.startswith(c) for c in choices
                                                                 if c.endswith(":"))):
            return answer
        print(f"  Choose one of: {', '.join(choices)}" if choices else "  A value is required.")


def new_experiment() -> None:
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    print("New pipeline experiment. Press Enter to keep the [default].\n")
    name = ask("Experiment name")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise PipelineError("Names may only use letters, digits, '.', '_' and '-'")
    target = CONFIG_DIR / f"{name}.yaml"
    if target.exists():
        raise PipelineError(f"{display(target)} already exists")
    config: dict = {}
    print("\nTraining data:\n  generate  build synthetic NLU data with the pipeline (needs source passages)\n"
          "  existing  train on data you already have (skips sources + generate)")
    mode = ask("Training data", "generate", ["generate", "existing"])
    start = "sources"
    if mode == "generate":
        source = ask("Source passages: korp | hf | local | experiment:<name>", "korp",
                     ["korp", "hf", "local", "experiment:"])
        data: dict = {"source": source}
        if source == "hf":
            data["hf"] = {"dataset": ask("  Hugging Face dataset", "wikimedia/wikipedia")}
            subset = ask("  Subset/config (Enter for none)", "20231101.sv" if
                         data["hf"]["dataset"] == "wikimedia/wikipedia" else "")
            if subset:
                data["hf"]["subset"] = subset
            text = ask("  Text column", "text")
            if text != "text":
                data["fields"] = {"text": text}
        elif source == "local":
            data["local"] = {"path": ask("  File, directory or glob")}
        config["data"] = data
        model = ask("Generator model (Hugging Face id)", defaults["generator"]["model"])
        if model != defaults["generator"]["model"]:
            config["generator"] = {"model": model}
        target_rows = int(ask("Accepted rows to generate", str(defaults["datagen"]["target"])))
        if target_rows != defaults["datagen"]["target"]:
            config["datagen"] = {"target": target_rows}
    else:
        start = "sft"
        spec = ask("Training data: path, hf:org/name[:split] or experiment:<name>")
        config["sft"] = {"train_file": spec}
    model = ask("Model to fine-tune (Hugging Face id)", defaults["sft"]["model"])
    if model != defaults["sft"]["model"]:
        config.setdefault("sft", {})["model"] = model
    quantization = ask("Fine-tune in bf16 (none) or 4-bit QLoRA (nf4)", defaults["sft"]["quantization"],
                       ["none", "nf4"])
    if quantization != defaults["sft"]["quantization"]:
        config.setdefault("sft", {})["quantization"] = quantization
    target.write_text("# Created by ./pipeline new; defaults in configs/pipeline/default.yaml\n"
                      + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\nWrote {display(target)}:\n{target.read_text(encoding='utf-8')}")
    print(f"Check it:  ./pipeline run {name} --from {start} --dry-run")
    print(f"Run it:    ./pipeline run {name} --from {start}")


def doctor(config: dict | None) -> None:
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  {'✓' if passed else '✗'} {label}" + (f": {detail}" if detail else ""))

    print("Environments")
    for key, packages in (("data", "data_designer, pandas"), ("train", "trl, peft, torch"),
                          ("eval", "vllm, euroeval")):
        exe = Path(python(key))
        if not exe.exists():
            check(f"{display(VENVS[key])}", False, "missing; run scripts/setup.sh")
            continue
        result = subprocess.run([str(exe), "-c", f"import {packages}"], capture_output=True, text=True, check=False,
                                env=environment())
        check(f"{display(VENVS[key])} ({packages})", result.returncode == 0,
              result.stderr.strip().splitlines()[-1] if result.returncode else "")
    print("GPU")
    if shutil.which("nvidia-smi"):
        info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
        check("nvidia-smi", bool(info), info)
    else:
        check("nvidia-smi", False, "not found")
    if Path(python("train")).exists():
        # Kernels for sm_XY run on any sm_XZ with Z >= Y (e.g. sm_86 on an sm_89 RTX 4090);
        # compute_XY (PTX) is compiled for any newer GPU.
        probe = ("import re, torch; ok = torch.cuda.is_available(); cap = torch.cuda.get_device_capability() "
                 "if ok else None; arch = f'sm_{cap[0]}{cap[1]}' if cap else '?'; "
                 "built = [(k, int(v[:-1]), int(v[-1])) for k, v in "
                 "(re.match(r'(sm|compute)_(\\d+)', a).groups() for a in torch.cuda.get_arch_list())]; "
                 "fits = bool(cap) and any((k == 'sm' and major == cap[0] and minor <= cap[1]) or "
                 "(k == 'compute' and (major, minor) <= tuple(cap)) for k, major, minor in built); "
                 "print(ok, torch.__version__, torch.version.cuda, arch, fits)")
        result = subprocess.run([python("train"), "-c", probe], capture_output=True, text=True, check=False)
        fields = result.stdout.split()
        supported = len(fields) == 5 and fields[0] == "True" and fields[4] == "True"
        check("PyTorch sees the GPU with kernels for its architecture", supported,
              f"torch {fields[1]} CUDA {fields[2]} {fields[3]}" if len(fields) == 5 else result.stderr[-300:])
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)
    if nvcc:
        output = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False).stdout
        release = re.search(r"release (\d+\.\d+)", output)
        version = release.group(1) if release else "?"
        # Informational: only JIT-compiled kernels use it; the pipeline avoids them by default.
        print(f"  {'✓' if version.startswith('13') else '!'} system CUDA toolkit {version} ({nvcc})"
              + ("" if version.startswith("13") else "; wheels use CUDA 13, so keep "
                 "VLLM_USE_FLASHINFER_SAMPLER=0 (the pipeline default)"))
    print("Hugging Face")
    token = os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").is_file()
    check("login token for gated models (Gemma)", bool(token),
          "" if token else "run: .venv/bin/hf auth login (or export HF_TOKEN)")
    print("Disk")
    free = shutil.disk_usage(ROOT / "artifacts" if (ROOT / "artifacts").exists() else ROOT).free / 2**30
    check("free space for models and caches", free > 100, f"{free:.0f} GiB free")
    if config:
        print(f"Experiment {config['name']}")
        for label, port in (("generator", config["generator"]["port"]), ("eval", config["eval"]["port"])):
            check(f"{label} port {port} free", not port_in_use(port))
        if config["generator"]["backend"] == "llamacpp":
            server = os.environ.get("LLAMA_SERVER", f"{os.environ.get('S2S_MODELS_DIR', '/mnt/ssd/models')}"
                                    "/llama.cpp/build/bin/llama-server")
            check("llama-server", Path(server).exists(), server)
    print("\nAll checks passed." if ok else "\nSome checks failed; see above.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="./pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def experiment_args(sub):
        sub.add_argument("config", nargs="?", help="Experiment YAML (path or name in configs/pipeline)")
        sub.add_argument("--name", help="Override the experiment name (outputs are named after it)")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="Override a setting, e.g. --set sft.model=Qwen/Qwen3-8B (repeatable)")
        sub.add_argument("--profile", help="Hardware profile layered over the experiment, e.g. rtx4090-pilot "
                                           "for a 24 GB card (default: $S2S_PIPELINE_PROFILE; 'none' disables)")

    run_parser = commands.add_parser("run", help="Run pipeline stages (finished stages are skipped)")
    experiment_args(run_parser)
    run_parser.add_argument("--from", dest="first", choices=STAGES, default=STAGES[0])
    run_parser.add_argument("--to", dest="last", choices=STAGES, default=STAGES[-1])
    run_parser.add_argument("--only", choices=STAGES, help="Run a single stage")
    run_parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    for name, text in (("status", "Show which stages are done"), ("show", "Print the resolved config")):
        experiment_args(commands.add_parser(name, help=text))
    commands.add_parser("new", help="Create an experiment config interactively")
    experiment_args(commands.add_parser("doctor", help="Check environments, GPU, login and ports"))
    args = parser.parse_args(argv)
    try:
        if args.command == "new":
            return new_experiment()
        if args.command == "doctor":
            return doctor(load_experiment(args.config, args.set, args.name, args.profile)
                          if args.config or args.name else None)
        config = load_experiment(args.config, args.set, args.name, args.profile)
        if args.command == "show":
            print(yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")},
                                 allow_unicode=True, sort_keys=False))
        elif args.command == "status":
            print_status(config)
        else:
            first, last = (args.only, args.only) if args.only else (args.first, args.last)
            run(config, first, last, args.dry_run)
    except PipelineError as error:
        raise SystemExit(f"Error: {error}") from None
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; rerun the same command to continue.") from None


if __name__ == "__main__":
    main()
