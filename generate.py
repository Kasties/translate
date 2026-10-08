import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-3-1b-pt"
ADAPTER_DIR = "/scratch/muhsh/swedish-ft/outputs/test-adapter"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
base.eval()

prompt = "<start_of_turn>user\nVad är huvudstaden i Sverige?<end_of_turn>\n<start_of_turn>model\n"
inputs = tok(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = base.generate(**inputs, max_new_tokens=40, do_sample=False)
print("BASE", tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print("LORA", tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))