"""Train a DPO LoRA on a merged SFT checkpoint with the pairs from build_pairs.py, then merge it.

    .venv-train/bin/python scripts/dpo/train_dpo.py --base <merged SFT folder> \
        --pairs artifacts/dpo/<name>/pairs.jsonl --output artifacts/dpo/<name> --merged <folder> [--dry-run]

The reference model is the base with the adapter switched off, so the weights are loaded once.
Prompts are rendered with the base's own chat template and the system message of the
instruction-mode SFT data; TRL appends EOS to both answers and tokenizes without adding a
second BOS. The LoRA matches the SFT stage: rank 16, alpha 32, attention and MLP projections.
--dry-run prints the first tokenized pair and stops before training.
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source2synth.finetune import resolve_model_class  # noqa: E402

SYSTEM = "Följ instruktionen och skriv på svenska i begärt format."  # instruction_data.training_record

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--base", required=True, help="merged SFT checkpoint folder")
parser.add_argument("--pairs", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--merged", type=Path, help="also save base + DPO adapter as a full checkpoint here")
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--learning-rate", type=float, default=2e-5)
parser.add_argument("--epochs", type=float, default=2)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base)
rows = [json.loads(line) for line in args.pairs.open(encoding="utf-8")]


def to_example(row):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}],
        tokenize=False, add_generation_prompt=True)
    # The template writes BOS and TRL's tokenizer adds one too; keep only the tokenizer's.
    prompt = prompt.removeprefix(tokenizer.bos_token)
    return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}


train = Dataset.from_list([to_example(r) for r in rows if r["split"] == "train"])
evaluation = Dataset.from_list([to_example(r) for r in rows if r["split"] == "eval"])
print(f"{len(train)} training pairs, {len(evaluation)} held-out pairs", flush=True)

model_class = resolve_model_class({"model": args.base, "revision": None, "model_class": "auto"})
model = model_class.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
                                    attn_implementation="sdpa")
model.enable_input_require_grads()
for model_config in (model.config, getattr(model.config, "text_config", None)):
    if model_config is not None:
        model_config.use_cache = False
lora = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
# This transformers version has no warmup_ratio; warm up for 10% of the steps, as finetune.py does.
steps = math.ceil(len(train) / (args.batch_size * args.gradient_accumulation_steps)) * args.epochs
training_args = DPOConfig(
    output_dir=str(args.output / "trainer"), beta=args.beta, learning_rate=args.learning_rate,
    num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_length=args.max_length, warmup_steps=math.ceil(0.1 * steps), lr_scheduler_type="cosine", bf16=True,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=5, eval_strategy="steps", eval_steps=10, save_strategy="no", report_to="none", seed=args.seed)
trainer = DPOTrainer(model=model, ref_model=None, args=training_args, train_dataset=train,
                     eval_dataset=evaluation, processing_class=tokenizer, peft_config=lora)

first = trainer.train_dataset[0]
for name in ("prompt_ids", "chosen_ids", "rejected_ids"):
    ids = first[name]
    print(f"{name}: {len(ids)} tokens | start {tokenizer.decode(ids[:12])!r} | end {tokenizer.decode(ids[-6:])!r}",
          flush=True)
if args.dry_run:
    sys.exit(0)

before = trainer.evaluate()
result = trainer.train()
after = trainer.evaluate()
trainer.model.save_pretrained(args.output / "adapter")
tokenizer.save_pretrained(args.output / "adapter")
manifest = {
    "base": args.base, "pairs": str(args.pairs),
    "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
    "train_pairs": len(train), "eval_pairs": len(evaluation), "system": SYSTEM,
    "settings": {k: v for k, v in vars(args).items() if k not in {"base", "pairs", "output", "merged", "dry_run"}},
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": sorted(lora.target_modules)},
    "train": result.metrics, "eval_before": before, "eval_after": after,
    "log_history": trainer.state.log_history, "gpu": torch.cuda.get_device_name(),
}
(args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"eval_before": before, "eval_after": after}, indent=2), flush=True)

if args.merged:
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.merged, safe_serialization=True)
    tokenizer.save_pretrained(args.merged)
    processor = Path(args.base) / "processor_config.json"
    if processor.exists():
        shutil.copy(processor, args.merged / "processor_config.json")
    print(f"Saved merged checkpoint to {args.merged}", flush=True)
