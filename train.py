import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "google/gemma-3-1b-pt"
OUTPUT_DIR = "/scratch/muhsh/swedish-ft/checkpoints/test-run"
ADAPTER_DIR = "/scratch/muhsh/swedish-ft/outputs/test-adapter"

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
print("Model loaded successfully")

dummy_rows = [
    {"instruction": "Vad är huvudstaden i Sverige?", "response": "Huvudstaden i Sverige är Stockholm."},
    {"instruction": "Skriv en kort hälsning på svenska.", "response": "Hej och välkommen! Så trevligt att ses."},
    {"instruction": "Vad heter den största sjön i Sverige?", "response": "Den största sjön i Sverige är Vänern."},
    {"instruction": "Ge ett svenskt ordspråk.", "response": "Övning ger färdighet."},
    {"instruction": "Vilken är den högsta bergstoppen i Sverige?", "response": "Den högsta bergstoppen i Sverige är Kebnekaise."},
    {"instruction": "Vad är valutan i Sverige?", "response": "Valutan i Sverige är svensk krona."},
    {"instruction": "Nämn tre stora städer i Sverige.", "response": "Tre stora städer i Sverige är Stockholm, Göteborg och Malmö."},
    {"instruction": "Vad heter Sveriges nationaldag?", "response": "Sveriges nationaldag firas den sjätte juni."},
    {"instruction": "Skriv en mening om hösten.", "response": "På hösten faller löven och vädret blir kallare."},
    {"instruction": "Vad är motsatsen till varm?", "response": "Motsatsen till varm är kall."},
    {"instruction": "Översätt ordet hund till engelska.", "response": "Ordet hund betyder dog på engelska."},
    {"instruction": "Vad heter havet öster om Sverige?", "response": "Havet öster om Sverige heter Östersjön."},
    {"instruction": "Ge ett tips för att lära sig svenska.", "response": "Ett bra tips är att läsa och lyssna på svenska varje dag."},
    {"instruction": "Vad är en älg?", "response": "En älg är ett stort däggdjur som lever i skogen."},
    {"instruction": "Skriv en kort mening om kaffe.", "response": "Många svenskar dricker gärna kaffe på fikapausen."},
    {"instruction": "Vad betyder ordet tack?", "response": "Ordet tack använder man för att visa uppskattning."},
]

def format_row(row):
    text = (
        "<start_of_turn>user\n"
        + row["instruction"]
        + "<end_of_turn>\n<start_of_turn>model\n"
        + row["response"]
        + "<end_of_turn>"
    )
    return {"text": text}

dataset = Dataset.from_list(dummy_rows).map(format_row)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=30,
    learning_rate=2e-4,
    logging_steps=1,
    max_length=512,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=lora_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(ADAPTER_DIR)
print("Training done, adapter saved to", ADAPTER_DIR)