"""Merge a LoRA adapter into its base model and save a full bf16 checkpoint.

    .venv-train/bin/python scripts/dpo/merge_adapter.py <adapter dir> <output dir> [--base <model or folder>]

The base defaults to the adapter's base_model_name_or_path. The output gets the adapter's
tokenizer and chat template, plus the base's processor_config.json when it has one (vLLM
needs it for Gemma 4).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source2synth.finetune import resolve_model_class  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("adapter", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--base")
args = parser.parse_args()

base = args.base or json.loads((args.adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
model_class = resolve_model_class({"model": base, "revision": None, "model_class": "auto"})
print(f"Loading {base} as {model_class.__name__}", flush=True)
model = model_class.from_pretrained(base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
model.save_pretrained(args.output, safe_serialization=True)
AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)
if Path(base).is_dir():
    processor = Path(base) / "processor_config.json"
    processor = processor if processor.exists() else None
else:
    try:
        processor = Path(hf_hub_download(base, "processor_config.json"))
    except Exception:
        processor = None
if processor:
    shutil.copy(processor, args.output / "processor_config.json")
print(f"Saved merged checkpoint to {args.output}", flush=True)
