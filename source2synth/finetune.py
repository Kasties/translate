"""Reproducible TRL LoRA/QLoRA fine-tuning on chat-format training data.

Run with the separate training environment; data preparation needs only PyYAML.
`train_file` may be one entry or a list of: a pipeline export (train.jsonl next to
accepted.jsonl, checked against its provenance), any other .jsonl/.json/.parquet
file, or `hf:org/name[:split]` / `hf:org/name@subset[:split]` for a Hub dataset.
Rows may use messages/conversations, prompt+completion, instruction/input/output
or question/answer columns; only the final assistant turn is supervised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from .common import ROOT, read_config, read_jsonl, write_json

# The base model has no instruction template. This is deliberately explicit and
# saved with the adapter. Unlike the generation-only llama.cpp template, it also
# closes training answers with EOS and adds a header only when asked to generate.
BASE_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ message['role'] | upper }}:\n{{ message['content'] }}"
    "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}\n\n"
    "{% endfor %}{% if add_generation_prompt %}ASSISTANT:\n{% endif %}"
)


# Keys added after the Gemma pilot; defaults reproduce the pilot's behaviour.
DEFAULTS = {
    "revision": None,
    "model_class": "auto",  # auto = the checkpoint's declared architecture
    "quantization": "nf4",  # nf4 (QLoRA) or none (bf16 LoRA)
    "chat_template": "auto",  # auto = tokenizer's own if any, else base; or base/model/<file>
    "attn_implementation": "sdpa",
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
    "warmup_ratio": 0.05,
    "require_provenance": False,
}
ROLE_ALIASES = {"human": "user", "gpt": "assistant", "model": "assistant", "bot": "assistant"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_training_rows(path: Path) -> tuple[list[dict], dict]:
    """Validate the actual export against its provenance, without resampling it."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty training export: {path}")
    prompts = set()
    for index, row in enumerate(rows, 1):
        messages = row.get("messages", [])
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Row {index}: expected system/user/assistant messages")
        if any(
            not isinstance(m.get("content"), str) or not m["content"].strip()
            for m in messages
        ):
            raise ValueError(f"Row {index}: empty or non-text message")
        prompt = messages[1]["content"]
        if prompt in prompts:
            raise ValueError(f"Row {index}: duplicate training prompt")
        prompts.add(prompt)

    accepted_path = path.with_name("accepted.jsonl")
    if not accepted_path.is_file():
        raise ValueError(f"Missing curated provenance: {accepted_path}")
    accepted = read_jsonl(accepted_path)
    # Match by content, not row position: provenance can include surplus records.
    lookup = {}
    for item in accepted:
        output = item["output"]
        answer = (
            output
            if isinstance(output, str)
            else json.dumps(output, ensure_ascii=False)
        )
        key = (item["input"], answer)
        if item.get("split") == "eval" or item.get("source", {}).get("split") == "eval":
            raise ValueError("Held-out eval sources cannot be used for training")
        lookup[key] = item
    selected = []
    for row in rows:
        key = (row["messages"][1]["content"], row["messages"][2]["content"])
        if key not in lookup:
            raise ValueError("Training row has no matching curated provenance")
        selected.append(lookup[key])
    report_path = path.with_name("report.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return rows, {
        "train_file": str(path.resolve()),
        "train_sha256": sha256(path),
        "accepted_file": str(accepted_path.resolve()),
        "accepted_sha256": sha256(accepted_path),
        "rows": len(rows),
        "by_task": dict(Counter(item["task"] for item in selected)),
        "source_splits": sorted({item["split"] for item in selected}),
        "source_build_complete": report.get("complete"),
        "source_build_target": report.get("target"),
    }


def normalize_row(row: dict, index: int) -> dict:
    """Map common SFT layouts to {"messages": [...]} ending in an assistant turn."""
    if isinstance(row.get("messages") or row.get("conversations"), list):
        messages = []
        for m in row.get("messages") or row["conversations"]:
            role = m.get("role", m.get("from"))
            messages.append({"role": ROLE_ALIASES.get(role, role),
                             "content": m.get("content", m.get("value"))})
    elif "prompt" in row and "completion" in row:
        messages = [{"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]}]
    elif "instruction" in row and "output" in row:
        prompt = row["instruction"] + (f"\n\n{row['input']}" if row.get("input") else "")
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": row["output"]}]
    elif "question" in row and "answer" in row:
        messages = [{"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]}]
    else:
        raise ValueError(f"Row {index}: no messages, prompt/completion, instruction/output "
                         f"or question/answer columns (found {sorted(row)})")
    if row.get("system") and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": row["system"]})
    roles = [m["role"] for m in messages]
    if set(roles) - {"system", "user", "assistant"} or roles[-1] != "assistant" or "user" not in roles:
        raise ValueError(f"Row {index}: roles {roles} must include user and end with assistant")
    if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
        raise ValueError(f"Row {index}: empty or non-text message")
    return {"messages": messages}


def read_rows(spec: str) -> tuple[list[dict], dict]:
    if spec.startswith("hf:"):
        from datasets import load_dataset

        name, _, split = spec[3:].partition(":")
        name, _, subset = name.partition("@")
        dataset = load_dataset(name, subset or None, split=split or "train")
        info = {"hf_dataset": name, "subset": subset or None, "split": split or "train",
                "fingerprint": dataset._fingerprint}
        return [dict(row) for row in dataset], info
    path = project_path(spec)
    if path.suffix == ".parquet":
        import pandas as pd

        rows = json.loads(pd.read_parquet(path).to_json(orient="records", force_ascii=False))
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [data]
    else:
        rows = read_jsonl(path)
    return rows, {"train_file": str(path.resolve()), "train_sha256": sha256(path)}


def load_training_data(config: dict) -> tuple[list[dict], dict]:
    """Load every train_file entry; pipeline exports keep their strict provenance check."""
    specs = config["train_file"]
    specs = [specs] if isinstance(specs, str) else list(specs)
    if not specs:
        raise ValueError("train_file is empty")
    all_rows, infos = [], []
    for spec in specs:
        path = None if spec.startswith("hf:") else project_path(spec)
        if path is not None and path.with_name("accepted.jsonl").is_file() and path.suffix == ".jsonl":
            rows, info = load_training_rows(path)
            info["provenance"] = "curated"
        elif config["require_provenance"]:
            raise ValueError(f"{spec} has no accepted.jsonl provenance and require_provenance is set")
        else:
            raw, info = read_rows(spec)
            rows, seen, duplicates = [], set(), 0
            for index, item in enumerate(raw, 1):
                row = normalize_row(item, index)
                key = json.dumps(row["messages"][:-1], ensure_ascii=False)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                rows.append(row)
            if not rows:
                raise ValueError(f"Empty training data: {spec}")
            info.update(rows=len(rows), duplicates_dropped=duplicates, provenance="none")
        infos.append(info)
        all_rows.extend(rows)
    if len(infos) == 1:
        return all_rows, infos[0]
    by_task = Counter()
    for info in infos:
        by_task.update(info.get("by_task", {}))
    return all_rows, {
        # train_file keeps the first source for tools that read one path.
        "train_file": infos[0].get("train_file", specs[0]),
        "rows": len(all_rows),
        "by_task": dict(by_task),
        "sources": infos,
    }


def resolve_model_class(config: dict):
    import transformers

    if config["model_class"] != "auto":
        return getattr(transformers, config["model_class"])
    architectures = transformers.AutoConfig.from_pretrained(
        config["model"], revision=config["revision"]
    ).architectures or []
    for name in architectures:
        if hasattr(transformers, name):
            return getattr(transformers, name)
    return transformers.AutoModelForCausalLM


def select_chat_template(tokenizer, choice: str) -> str:
    if choice == "base" or (choice == "auto" and not tokenizer.chat_template):
        return BASE_CHAT_TEMPLATE
    if choice in {"auto", "model"}:
        if not tokenizer.chat_template:
            raise ValueError("chat_template=model, but the tokenizer has no chat template")
        return tokenizer.chat_template
    return project_path(choice).read_text(encoding="utf-8")


def tokenize_rows(
    rows: list[dict], tokenizer, max_length: int
) -> tuple[list[dict], dict]:
    """Mask prompts and refuse truncation that could silently erase supervision."""
    result = []
    lengths = []
    answer_lengths = []
    for index, row in enumerate(rows, 1):
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Row {index}: chat template does not preserve the tokenized prompt prefix"
            )
        if len(ids) > max_length:
            raise ValueError(
                f"Row {index}: {len(ids)} tokens exceeds max_length={max_length}; increase it"
            )
        if len(ids) <= len(prompt_ids):
            raise ValueError(f"Row {index}: no answer tokens")
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids) :]
        result.append(
            {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
        )
        lengths.append(len(ids))
        answer_lengths.append(len(ids) - len(prompt_ids))
    return result, {
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "supervised_tokens": sum(answer_lengths),
        "truncated_rows": 0,
    }


def train(config: dict, *, max_steps: int = -1, dry_run: bool = False) -> dict:
    config = {**DEFAULTS, **config}
    if config["quantization"] not in {"nf4", "none"}:
        raise ValueError("quantization must be nf4 or none")
    rows, data_info = load_training_data(config)
    if dry_run:
        return {"config": config, "data": data_info}
    output = project_path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output}. Choose a new output_dir."
        )
    # A machine with a shared model cache sets HF_HUB_CACHE and keeps HF_HOME, which holds the token.
    if "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HOME"] = os.environ.get(
            "TDDE19_HF_HOME", str(ROOT / "artifacts/cache/huggingface")
        )
        os.environ["HF_HUB_CACHE"] = str(Path(os.environ["HF_HOME"]) / "hub")
    # Keep `hf auth login` credentials usable for gated models despite the moved HF_HOME.
    default_token = Path.home() / ".cache/huggingface/token"
    if "HF_TOKEN_PATH" not in os.environ and default_token.is_file():
        os.environ["HF_TOKEN_PATH"] = str(default_token)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("This QLoRA configuration requires a CUDA GPU")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"]
    )
    tokenizer.chat_template = select_chat_template(tokenizer, config["chat_template"])
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized, token_info = tokenize_rows(rows, tokenizer, config["max_length"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "config": config,
        "max_steps": max_steps,
        "data": data_info,
        "tokens": token_info,
        "packages": {
            name: version(name)
            for name in [
                "trl",
                "transformers",
                "peft",
                "torch",
                "bitsandbytes",
                "datasets",
            ]
        },
        "gpu": torch.cuda.get_device_name(),
        "command": sys.argv,
        "loss": "assistant answer and EOS only; prompt labels masked with -100",
        "evaluation": "EuroEval validation is separate; all curated export rows are used for training",
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    quantized = config["quantization"] == "nf4"
    model = resolve_model_class(config).from_pretrained(
        config["model"],
        revision=config["revision"],
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        attn_implementation=config["attn_implementation"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if quantized
        else None,
    )
    if quantized:
        model = prepare_model_for_kbit_training(
            model, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.enable_input_require_grads()
    for model_config in (model.config, getattr(model.config, "text_config", None)):
        if model_config is not None:
            model_config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            target_modules=config["lora_target_modules"],
        ),
    )
    if config["revision"]:
        model.peft_config["default"].revision = config["revision"]
    model.print_trainable_parameters()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(tokenized),
        args=SFTConfig(
            output_dir=str(output),
            num_train_epochs=config["epochs"],
            max_steps=max_steps,
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_steps=math.ceil(
                config["warmup_ratio"]
                * (
                    max_steps
                    if max_steps > 0
                    else math.ceil(
                        len(rows)
                        / (config["batch_size"] * config["gradient_accumulation_steps"])
                    )
                    * config["epochs"]
                )
            ),
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=config["max_length"],
            packing=False,
            loss_type="chunked_nll",
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            seed=config["seed"],
            data_seed=config["seed"],
            optim="adamw_torch",
            push_to_hub=False,
        ),
    )
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    manifest.update(
        status="complete", metrics=result.metrics, adapter=str(output / "adapter")
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/finetune.yaml")
    parser.add_argument(
        "--output-dir", help="Override the output directory (never overwritten)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Override epochs for a smoke test"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/provenance without loading a model",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    print(
        json.dumps(
            train(config, max_steps=args.max_steps, dry_run=args.dry_run),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
