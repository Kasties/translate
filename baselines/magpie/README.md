# Magpie baseline: Swedish fine-tuning data

These steps build the Swedish Magpie dataset ([matsten/Magpie-SWE-Llama-3.3-70B-Instruct-V3-0324-responses](https://huggingface.co/datasets/matsten/Magpie-SWE-Llama-3.3-70B-Instruct-V3-0324-responses)) in chat format, split into 2,000-example chunks for staged fine-tuning. The data is not stored in git, so you generate it locally. It takes under a minute.

The dataset was published with *Synthetic Instruction Generation for Low-Resource Nordic Languages: Viability and Limitations in LLM Instruction-Tuning* ([LREC 2026](https://aclanthology.org/2026.lrec-1.838/)). Llama-3.3-70B-Instruct generated the instructions directly in Swedish, and they were quality-filtered. DeepSeek-V3-0324 generated the responses.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Internet access. You don't need a Hugging Face account, because the dataset is not gated.

## Steps

Run these from the repository root:

```bash
# 1. Install dependencies
uv sync

# 2. Download the dataset (about 24 MB) to baselines/magpie/data/raw/
uv run python -m baselines.magpie.download

# 3. Convert to chat-format JSONL
uv run python -m baselines.magpie.format_for_finetuning

# 4. Split into 2,000-example chunks
uv run python -m baselines.magpie.chunk
```

## Output

All output goes to `baselines/magpie/data/processed/`:

| File | Examples | Use |
|---|---|---|
| `chunks/train_chunk_01_of_06.jsonl` … `train_chunk_05_of_06.jsonl` | 2,000 each | Training, one chunk per stage |
| `chunks/train_chunk_06_of_06.jsonl` | 1,635 | Training, last stage |
| `train.jsonl` | 11,635 | The full training set. It contains the same rows as all the chunks together. |

Each line is one example, in the same format as the MURI baseline:

```json
{"id": "magpie-train-0", "source": "magpie-train", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Notes

- **There is no validation or test split.** The dataset is published as training data only. For checking progress during training and for the final evaluation, use the same held-out set for every model, such as MURI's `validation.jsonl` and `test.jsonl`, so that results are comparable.
- 7 of the 11,642 original rows have empty responses and are dropped. IDs keep the original row number, so they skip those 7.
- Every row in the source has the same Swedish system prompt. It is left out, so that the data matches MURI, which has none.
- The chunks keep the original order. Task types are already evenly mixed (about 71% information-seeking in every chunk), so each chunk is representative.
- Responses are at most 2,091 Gemma 3 tokens, with a median of 711, so no length cap is needed for this dataset.
- **Licence:** the dataset card states no licence. The instructions are Llama 3.3 output, so the Llama 3.3 Community License likely applies to them. You may fine-tune non-Llama models such as Gemma on it, but a published model trained on it must have a name starting with "Llama" and display "Built with Llama".
