# Fine-tuning results

## Swedish idioms evaluation

The evaluation scored 478 idiom/non-idiom sentence pairs. For each pair, the
scorer checks whether the model assigns lower perplexity to the idiomatic
sentence. The 100-pair challenge set is a subset of those 478 pairs.

| Set | Base(accuracy) | Fine-tuned(accuracy) | Difference |
| --- | ---: | ---: | ---: |
| All 478 pairs | 436/478 (91.2%) | 423/478 (88.5%) | -2.7 percentage points |
| Challenge 100 | 87/100 (87.0%) | 82/100 (82.0%) | -5.0 percentage points |

On all 478 pairs, the base model was correct on 436 and the fine-tuned model on
423. Their 95% confidence intervals were 88.3–93.4% and 85.3–91.1%,
respectively. On the challenge subset, the intervals were 79.0–92.2% for base
and 73.3–88.3% for fine-tuned.

Since both models were evaluated on the same pairs, the paired comparison is
informative: on all pairs, 15 disagreements favored base and 2 favored the
fine-tuned model (exact McNemar p = 0.00235). This is evidence that the tuned
model performed worse on this benchmark. On the challenge subset, 6
disagreements favored base and 1 favored fine-tuned (p = 0.125); this is not
enough evidence to conclude that the difference on that subset is statistically
significant.

These results indicate a regression on this idiom benchmark, not necessarily
on Swedish language ability overall. They come from one run and one evaluation
set; they do not establish how the models compare on other tasks or across
independent training runs.

## Was the instruction-synthesis LLM judge used?

The pipeline record for this run shows `datagen.mode: nlu`, so this was an NLU
data-generation run, not an instruction-synthesis run. The idiom benchmark
scores are not LLM-as-a-judge ratings; they come from comparing model
perplexities for paired sentences.

The project contains an instruction judge path in
[`source2synth/instruction_data.py`](source2synth/instruction_data.py), but this
run's pipeline record does not show that it was invoked. To establish that the
judge worked during instruction synthesis, inspect logs or results from a run
using instruction mode; the current run's idiom scores cannot confirm it.
