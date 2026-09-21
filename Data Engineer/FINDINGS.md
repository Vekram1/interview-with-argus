# Evaluation findings

## What I measured and why

The product must identify material developments and support them with the source
text sent to the model. Most production outputs will not receive human review.
The evaluation therefore measures both missed information and unsupported
information. It also separates answer quality from retrieval and system failures.

I measured the following dimensions:

1. **Expected-finding coverage.** This is the share of reference findings that
   the output captured. It measures whether the agent omitted a material
   development that an investor should see. Coverage is measurable on the
   labeled development set, but not on an unlabeled production batch.

2. **Claim support.** The evaluator splits each finding into factual claims and
   checks whether the supplied source context supports each claim. This catches
   incorrect numbers, unsupported comparisons, causal statements, and added
   qualifiers. I report supported, unsupported or contradicted, and unknown
   claims separately because an unknown claim is not evidence of correctness.

3. **Semantic citation support.** This checks whether the exact cited input
   chunks support the associated claim. The cited text can support a paraphrase.
   It does not need to contain the model's wording verbatim. This metric matters
   because a citation to the correct filing is not enough if the cited passage
   does not show the evidence.

4. **Classification accuracy.** This checks whether the agent correctly labels
   a finding as new, changed, repeated, or uncertain. The check is separate from
   factual support. A finding can state a correct number but make an unsupported
   comparison with an earlier period.

5. **Supported extra findings.** The reference set is not exhaustive, so I did
   not treat every unmatched output as a false positive. The evaluator checks
   those findings at the claim level. An extra is fully supported only when all
   its claims are supported. Partial extras retain a supported core but include
   at least one unsupported, contradicted, or unknown claim.

6. **Context sufficiency.** This checks whether the exact model input contained
   the current and historical evidence required for each reference finding. It
   helps separate a retrieval failure from an agent failure. The present check
   uses the canonical reference passages, so it can miss equivalent evidence in
   another passage. I report it separately from output quality.

7. **Operational reliability.** I measured completion rate, model and judge
   failures, latency, token use, and cost when the provider returned it. A
   correct system is not reliable if large cases truncate, fail validation, or
   cost too much to monitor.

I also calculated a weighted case score from coverage, groundedness, citation
support, and classification. The score is useful for comparison, but it is not
the primary result. Its components show the actual trade-offs and prevent one
high number from hiding missed findings or unsupported claims.

All rates include counts and denominators. This is important because the test
set has only nine cases and 23 reference findings. Small changes can produce
large percentage changes, so the case-level evidence remains part of the report.

## Baseline evaluation

This report records the first structured evaluation of the cached baseline
output. It evaluates the nine development cases in `runs/baseline-replay` against
reference version `1.0.1` using evaluator `eval-v5`, rubric `rubric-v2`, and
`google/gemini-3.8-flash` as the LLM judge.

The complete result is saved under
`runs/baseline-eval-gemini-all-v5-semantic/`. The compact case results are in
`cases.jsonl`; detailed model output, claim judgments, citation observations,
and judge metadata are under `audit/`.

## Baseline result

| Measure | Result |
| --- | ---: |
| Cases completed | 9/9 |
| Expected reference findings covered | 22/23 (95.7%) |
| Model findings | 74 |
| Fully supported extra findings | 39/45 (86.7%) |
| Partially supported extra findings | 6/45 (13.3%) |
| Supported claims | 211/220 |
| Unsupported or contradicted claims | 6/220 |
| Unknown claims | 3/220 |
| Correct decisive classifications | 28/29 (96.6%) |
| Semantic citation support | 128/140 (91.4%) |
| Mean case score | 94.8/100 |

The per-case scores were:

| Case | Score | Context status |
| --- | ---: | --- |
| AMD 0000002488-24-000109 | 98.2 | sufficient |
| AMD 0000002488-24-000115 | 100.0 | sufficient |
| CRWD 0001104659-24-081571 | 92.5 | partial |
| AMD 0000002488-24-000121 | 99.3 | insufficient |
| AMD 0001193125-24-202457 | 100.0 | insufficient |
| CRWD 0001535527-24-000017 | 94.1 | sufficient |
| AMD 0000002488-24-000161 | 94.7 | insufficient |
| AMD 0000002488-24-000172 | 84.8 | partial |
| CRWD 0001535527-24-000024 | 89.4 | partial |

## Main findings

### 1. Coverage was strong, but not complete

The baseline covered 22 of 23 expected reference findings. The missing finding
was CRWD 0001535527-24-000024:f04, which expected the distinction between a Q3
GAAP net loss and positive non-GAAP diluted EPS.

The model often split one reference finding into several output findings. The
evaluator treats this as acceptable when the split findings collectively cover
the expected facts.

The reference set is non-exhaustive, so unmatched findings were reviewed as
potential supported extras rather than treated as automatic false positives.
The claim-aware evaluator summary counts an extra as fully supported only when
it has at least one claim and every claim is supported. Six extras had a supported
core but also contained an unsupported, contradicted, or unknown claim.

### 2. Claim-level factual support was generally strong

The judge marked 211 of 220 atomic claims as supported. Six claims were
unsupported or contradicted, and three were unknown.

This rerun used the revised semantic-support rubric, so the claim counts should
not be interpreted as a model improvement over the earlier run. The evaluator
now grades the actual cited current/prior chunks and does not penalize a
paraphrased evidence string merely because it is not a literal source substring.

### 3. Citation support is now measured semantically

The semantic judge found that 128 of 140 citations were supported by the actual
cited chunks. The evaluator no longer requires the model's evidence string to be
a verbatim substring of the source. This matches the product requirement: the
finding may paraphrase, but the cited current and prior source text must entail
the claim.

Document identity, source eligibility, and chunk resolution remain deterministic
pointer checks. Literal quote presence is retained in the audit record as a
diagnostic only; it does not affect the score.

The baseline also emitted shortened chunk IDs in some findings. The evaluator
can uniquely recover those IDs from the cited document and input context, but
records them as `recovered`, not as correctly formatted citations.

### 4. Retrieval/context was insufficient for much of the reference set

Context sufficiency is measured independently from model correctness by checking
whether the source-checked reference evidence was present in the exact chunks
sent to the model:

| Context status | Reference findings |
| --- | ---: |
| Sufficient | 8/23 |
| Partial | 6/23 |
| Insufficient | 9/23 |

This means only 8 reference findings had all canonical required evidence intact
in the model input. The high mean output score must therefore not be read as a
pure model-quality score. The current score is an output score; context quality
is reported separately and is not yet used to produce a conditional model score.

The context check is deterministic and conservative. It checks for canonical
reference passages and detects missing or fragmented evidence. It does not yet
prove that an alternative equivalent passage was semantically sufficient.

### 5. Operationally, the run completed successfully

All nine agent runs completed without errors. The baseline agent used 145,888
tokens and recorded approximately $0.7743 in agent-model cost. The judge made
83 requests and used 528,430 tokens; OpenRouter did not expose a judge cost in
the saved response metadata.

The evaluator now runs independent finding judgments concurrently, then performs
one case-level reconciliation. This keeps the finding-level audit trail while
reducing evaluation wall-clock time.

## Scoring

Each case score is deterministic arithmetic over the LLM’s structured judgments:

- Coverage: 30 points.
- Groundedness: 30 points.
- Citation support: 25 points.
- Classification: 15 points.

The LLM does not directly choose the numerical score. It supplies claim,
citation, finding-match, and classification judgments. The evaluator converts
those judgments into the weighted score. The judgments remain subject to normal
LLM-judge variability and require calibration.

## Limitations

The reference file is a development reference set, not unquestionable ground
truth. Its quotes and source eligibility were checked by one reviewer, but the
interpretations have not received independent human adjudication. The references
are also explicitly non-exhaustive.

The baseline therefore establishes a reproducible starting point, not a final
quality claim. Before using the score for release decisions, we should:

1. independently review and calibrate the reference findings;
2. add a semantic context-sufficiency judgment for cases where canonical quote
   matching is inconclusive;
3. add a conditional model score that excludes retrieval-limited findings; and
4. repeat the same inputs across multiple model runs to measure consistency.

## Reproduction

From `Data Engineer/`:

```sh
uv run --env-file .env.openrouter python -m net_new.evaluate \
  --dataset data/pilot \
  --run runs/baseline-replay \
  --references references/development.jsonl \
  --model google/gemini-3.8-flash \
  --output runs/baseline-eval-gemini-all-v5-semantic
```

The output directory must be new or removed through an intentional, recoverable
workflow before rerunning. Do not use the holdout references for development
calibration.
