# Evaluation Results

Experiment: `first`  
Base model: `google/gemma-4-12B`  
Tuned model: base model with the trained adapter  
Evaluation run: 2026-11-09

## EuroEval

EuroEval ran on the validation split with few-shot prompting and one iteration.
Scores are reported on a 0–100 scale.

| Dataset | Task | Model | MCC | Macro F1 |
|---|---|---|---:|---:|
| swerec | Sentiment classification | Base | 77.83 | 79.06 |
| swerec | Sentiment classification | Tuned | 80.08 | 81.82 |
| scala-sv | Linguistic acceptability | Base | 78.02 | 89.01 |
| scala-sv | Linguistic acceptability | Tuned | 68.69 | 84.16 |

These are pilot validation scores from one iteration, not leaderboard-comparable
results.

## Swedish idiom pairs

The idiom evaluation scored 478 pairs. A pair is counted as correct when the
idiomatic option has lower mean-token perplexity than its distractor; ties count
as incorrect.

| Model | All pairs | Challenge pairs |
|---|---:|---:|
| Base | 436/478 (91.21%; 95% CI 88.34–93.43%) | 87/100 (87.00%; 95% CI 79.02–92.24%) |
| Tuned | 422/478 (88.28%; 95% CI 85.09–90.87%) | 83/100 (83.00%; 95% CI 74.45–89.11%) |

| Paired comparison | Only base correct | Only tuned correct | Exact McNemar p-value |
|---|---:|---:|---:|
| All pairs | 16 | 2 | 0.00131 |
| Challenge pairs | 6 | 2 | 0.28906 |

## Run notes

The pipeline completed source preparation, SFT, and evaluation. Data generation
was partial: 776 of the 1,000 requested rows were accepted and used for SFT.

Source files:

- EuroEval: `artifacts/euroeval/first/{base,tuned}/euroeval_benchmark_results.jsonl`
- Idiom summary: `artifacts/idioms/first/summary.json`
- Pipeline record: `artifacts/pipeline/first/pipeline.json`
