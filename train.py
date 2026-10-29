import shutil
import argparse
import glob
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

TEMPLATES = {
    "gemma3": ("<start_of_turn>", "<end_of_turn>"),
    "gemma4": ("<|turn>", "<turn|>"),
}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("base"))
        base = load_config(base_path)
        base.update(cfg)
        cfg = base
    return cfg


def expand(path):
    return os.path.expandvars(path)


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_config(args.config)

family = "gemma4" if "gemma-4" in cfg["model"].lower() else "gemma3"
turn_start, turn_end = TEMPLATES[family]


def to_prompt_completion(example):
    messages = example["messages"]
    prompt = ""
    for m in messages[:-1]:
        role = "model" if m["role"] == "assistant" else "user"
        prompt += f"{turn_start}{role}\n{m['content']}{turn_end}\n"
    prompt += f"{turn_start}model\n"
    completion = messages[-1]["content"] + turn_end
    return {"prompt": prompt, "completion": completion}


train_files = []
for pattern in cfg["train_files"]:
    train_files += sorted(glob.glob(expand(pattern)))
if not train_files:
    raise FileNotFoundError("No training files matched the config")

batch_size = cfg["batch_size"]
if cfg["effective_batch_size"] % batch_size != 0:
    raise ValueError("effective_batch_size must be divisible by batch_size")
grad_accum = cfg["effective_batch_size"] // batch_size

work_dir = expand(cfg["work_dir"])
run_name = cfg["run_name"]

train_ds = load_dataset("json", data_files=train_files, split="train")
if "validation_size" in cfg:
    split = train_ds.train_test_split(test_size=cfg["validation_size"], seed=cfg.get("seed", 42))
    train_ds, val_ds = split["train"], split["test"]
else:
    val_ds = load_dataset("json", data_files=expand(cfg["validation_file"]), split="train")
train_ds = train_ds.map(to_prompt_completion, remove_columns=train_ds.column_names)
val_ds = val_ds.map(to_prompt_completion, remove_columns=val_ds.column_names)

print("Run", run_name, "Model", cfg["model"])
print("Train files", len(train_files), "Train examples", len(train_ds), "Validation examples", len(val_ds))
print("Batch size", batch_size, "Grad accum", grad_accum)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
model = AutoModelForCausalLM.from_pretrained(
    cfg["model"],
    quantization_config=bnb_config,
    device_map={"": 0},
)

modules = cfg["lora"]["modules"]
if family == "gemma4":
    target_modules = r".*language_model.*\.(" + "|".join(modules) + ")"
else:
    target_modules = modules

lora_config = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

sft_config = SFTConfig(
    output_dir=f"{work_dir}/checkpoints/{run_name}",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=cfg["epochs"],
    learning_rate=cfg["learning_rate"],
    lr_scheduler_type=cfg["lr_scheduler"],
    warmup_steps=cfg["warmup_steps"],
    logging_steps=cfg["logging_steps"],
    eval_strategy="steps",
    eval_steps=cfg["eval_steps"],
    save_strategy="steps",
    save_steps=cfg["eval_steps"],
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_length=cfg["max_length"],
    bf16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=sft_config,
    peft_config=lora_config,
    processing_class=tokenizer,
)

trainer.model.print_trainable_parameters()
trainer.train()

adapter_dir = f"{work_dir}/outputs/{run_name}"
trainer.save_model(adapter_dir)
with open(f"{adapter_dir}/run_config.yaml", "w") as f:
    yaml.safe_dump(cfg, f)

safe_dir = os.path.expanduser(f"~/adapters/{run_name}")
shutil.copytree(adapter_dir, safe_dir, dirs_exist_ok=True)
print("Adapter copy saved to", safe_dir)

print("Training done, adapter saved to", adapter_dir)
print("Peak VRAM GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))