# MURI baseline: Swedish fine-tuning data

This is the Swedish MURI-IT dataset ([akoksal/muri-it-language-split](https://huggingface.co/datasets/akoksal/muri-it-language-split), Apache-2.0) in chat format, split into 2,000-example chunks for staged fine-tuning.

**The ready-to-use files are committed to the repository.** After pulling, you will find them in `baselines/muri/data/processed/`: the `chunks/` folder, `validation.jsonl` and `test.jsonl`. You only need the steps below to regenerate them, which takes about a minute.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Internet access. You don't need a Hugging Face account, because MURI is not gated.

## Steps

Run these from the repository root:

```bash
# 1. Install dependencies
uv sync

# 2. Download the Swedish split (about 20 MB) to baselines/muri/data/raw/
uv run python -m baselines.muri.download

# 3. Keep only Swedish rows (MURI tags Swedish as "swe")
uv run python -m baselines.muri.filter_swedish

# 4. Convert to chat-format JSONL (train / validation / test)
uv run python -m baselines.muri.format_for_finetuning

# 5. Split the training set into 2,000-example chunks
uv run python -m baselines.muri.chunk
```

## Output

All output goes to `baselines/muri/data/processed/`:

| File | Examples | Use |
|---|---|---|
| `chunks/train_chunk_01_of_08.jsonl` … `train_chunk_07_of_08.jsonl` | 2,000 each | Training, one chunk per stage |
| `chunks/train_chunk_08_of_08.jsonl` | 1,120 | Training, last stage |
| `validation.jsonl` | 840 | Validation |
| `test.jsonl` | 840 | Final evaluation |
| `train.jsonl` | 15,120 | The full training set. It contains the same rows as all the chunks together. |

Each line is one example:

```json
{"id": "muri-train-0", "source": "muri-train", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Most fine-tuning libraries can load the `messages` field directly, including TRL's `SFTTrainer` and Hugging Face chat templates.

## Notes

- The chunks keep the original row order. The data sources (Wikipedia, mC4, OSCAR) are already mixed throughout the file, so every chunk is representative. The `id` field links each example back to `train.jsonl`.
- To use a different chunk size, run `uv run python -m baselines.muri.chunk --size 1000`. Re-running the script replaces the old chunks.
- Some responses are raw web text, and a few are very long. The median is about 258 Gemma 3 tokens and the maximum about 37,500. Set a maximum sequence length in the trainer to truncate or drop the longest examples.
