import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "google/gemma-3-1b-pt"
DATA_DIR = "/scratch/muhsh/swedish-ft/muri_data"
WORK_DIR = "/scratch/muhsh/swedish-ft"


def parse_chunks(text):
    if "-" in text:
        start, end = map(int, text.split("-"))
        return list(range(start, end + 1))
    return [int(x) for x in text.split(",")]


def to_prompt_completion(example):
    messages = example["messages"]
    prompt = ""
    for m in messages[:-1]:
        role = "model" if m["role"] == "assistant" else "user"
        prompt += f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    completion = messages[-1]["content"] + "<end_of_turn>"
    return {"prompt": prompt, "completion": completion}


parser = argparse.ArgumentParser()
parser.add_argument("--chunks", default="1")
parser.add_argument("--epochs", type=float, default=1)
parser.add_argument("--max_length", type=int, default=1024)
args = parser.parse_args()

chunks = parse_chunks(args.chunks)
run_name = "muri_chunks_" + "_".join(str(c) for c in chunks)
train_files = [f"{DATA_DIR}/chunks/train_chunk_{c:02d}_of_08.jsonl" for c in chunks]

train_ds = load_dataset("json", data_files=train_files, split="train")
val_ds = load_dataset("json", data_files=f"{DATA_DIR}/validation.jsonl", split="train")
train_ds = train_ds.map(to_prompt_completion, remove_columns=train_ds.column_names)
val_ds = val_ds.map(to_prompt_completion, remove_columns=val_ds.column_names)
print("Train examples", len(train_ds), "Validation examples", len(val_ds))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

sft_config = SFTConfig(
    output_dir=f"{WORK_DIR}/checkpoints/{run_name}",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=args.epochs,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_length=args.max_length,
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

trainer.train()
adapter_dir = f"{WORK_DIR}/outputs/{run_name}"
trainer.save_model(adapter_dir)
print("Training done, adapter saved to", adapter_dir)
print("Peak VRAM GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))