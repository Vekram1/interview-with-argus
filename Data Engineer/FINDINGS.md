# Baseline Evaluation Findings

Last Evaluator used: eval-v5  
Rubric used: rubric-v2  
Model chosen: gemini-3.8-flash as LLM judge  
Run stored: `runs/evidence-gated-prompt-v3-eval`

The complete result is saved under:
`runs/baseline-eval-gemini-all-v5-semantic/`. The compact case results are in
`cases.jsonl`; detailed model output, claim judgments, citation observations,
and judge metadata are under `audit/`.

## What I measured and why?

The first thing I wanted to measure was pipeline task completion and correction. I did this because I wanted to be sure that the agent was working as trivial as that sounds.

Task completeness was easy to measure through viewing what the status of each case’s findings had a complete status. A case counted as complete when the pipeline produced a valid output without any obvious errors.

Now, correctness was a lot harder to define. While I did treat the reference data as the gold set, because I didn’t thoroughly examine it and cross check with the SEC files, I couldn’t definitively say that they were 100% correct. That may be a weakness in my evaluation.

Nonetheless, it was my chosen benchmark.

## Correctness Measurements

### Coverage Rate

This was measuring how many findings the pipeline agent correctly identified from the reference set.

### Claim Support

Many findings made by the agent were not simple findings in the sense that they did not only contain one fact. Oftentimes, they included many facts. This would be an example:

> Philip Guido's salary increased from $725,000 to $750,000 and Forrest Norrod's from $710,000 to $750,000.

There are two claims here, and I wanted to measure if each claim could semantically be supported by the prior and/or current documents. The reason for this is that a finding could be partially correct. I could easily have just marked a finding as partially correct if one claim was incorrect and the other was correct but I thought this would be more accurate.

### Semantic Citation Support

The baseline produced 140 citations. In the cached baseline run, 128 of the 140 citations had enough evidence to support the claims. The remaining 12 were unrelated or incomplete citations. This involved actually checking the chunk. Claim support largely suffices as a groundedness metric, but this was a further verification in the event that a claim had multiple citations.

### Classification Accuracy

The agent classified findings as new, changed, repeated, or uncertain. I needed to check whether that matched the reference or not. This also checks for unsupported claims about events being new.

### Extra Findings

I needed to account for findings that were not part of the reference set. I reviewed these separately instead of counting them as false positives.

### Context Sufficiency

This was a more unique thing I measured with regards to the fixed size chunking that is done. I was curious to see if the chunks themselves had the necessary information to support the findings from the reference data. I thought this was important in the event some data got cut off. I classified the context as sufficient, partial, or insufficient.

### Metrics Measured

Token usage, Cost, agent/judge failures. These measurements were important because some large cases initially failed as the model produced truncated structured output.

### Grading/Composite Score

Based on the judges assessment of these metrics: coverage, claim support, citation support, and classification accuracy, as score from 0-100 was given to the run. This wasn’t a subjective score. It would be based on the run’s statistics. It was primarily there just to check if things were improving or not.

## Baseline Cache Results

| Case | Score | Coverage /30 | Claim support /30 | Citation support /25 | Classification /15 | Context |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AMD 0000002488-24-000109 | 92.5 | 30.0 | 30.0 | 25.0 | 7.5 | sufficient |
| AMD 0000002488-24-000115 | 100.0 | 30.0 | 30.0 | 25.0 | 15.0 | sufficient |
| CRWD 0001104659-24-081571 | 100.0 | 30.0 | 30.0 | 25.0 | 15.0 | partial |
| AMD 0000002488-24-000121 | 100.0 | 30.0 | 30.0 | 25.0 | 15.0 | insufficient |
| AMD 0001193125-24-202457 | 85.0 | 15.0 | 30.0 | 25.0 | 15.0 | insufficient |
| CRWD 0001535527-24-000017 | 80.9 | 20.0 | 30.0 | 15.9 | 15.0 | sufficient |
| AMD 0000002488-24-000161 | 87.5 | 22.5 | 30.0 | 25.0 | 10.0 | insufficient |
| AMD 0000002488-24-000172 | 100.0 | 30.0 | 30.0 | 25.0 | 15.0 | partial |
| CRWD 0001535527-24-000024 | 87.5 | 22.5 | 30.0 | 25.0 | 10.0 | partial |

The cached results looked great. Coverage was great, but there was one missing finding which was for case CRWD 0001535527-24-000024, finding 4. Context Sufficiency measurements were a little tough and I will need to change what I measure there. I was requiring that the reference passage appear in the input context, which is simply not reasonable nor expected. A change that I plan on making is making sure this works with semantic search. The goal was to check whether the context was good enough for the agent to reach the golden set.

An important thing was that I was looking at extras. The baseline cache had 74 findings vs 23 reference ones, and 45 of them were checked as being supported extras, which was very well possible. It felt unlikely. In a discussion with my agent, we found 6 extras that had partial issues (as in all their claims were not fully supported).

I tried to change the prompt to tighten it up around comparison and interpretation, but the results were less than spectacular.

| Metric | Cached baseline | Revised prompt |
| --- | ---: | ---: |
| Completed cases | 9/9 | 9/9 |
| Expected-finding coverage | 22/23 (95.7%) | 19/23 (82.6%) |
| Supported claims | 211/220 | 68/68 |
| Unsupported/unknown claims | 9 | 0 |
| Semantic citation support | 91.4% | 91.5% |
| Classification accuracy | 96.6% | 84.2% |
| Findings emitted | 74 | 22 |
| Mean composite score | 94.8 | 92.6 |

I didn’t test this run out on the holdout data. I did try some other experiments on the holdout data with trying out different methods of chunking to see if we could get better results. I will need to measure them. I was running out of time.
