# Manual audit of the semantic baseline

This is an AI-assisted manual review of
`runs/baseline-eval-gemini-all-v5-semantic`. It is intended as a compact review
packet for human confirmation, not as independent human adjudication.

The audit covers:

- the one missing reference finding;
- all six unsupported or contradicted claims;
- all three unknown claims; and
- a stratified sample of ten findings labeled `supported_extra`, spanning cases,
  usefulness levels, and clean and disputed claim judgments.

## Executive result

The baseline is strong at extracting facts that appear in current filings. The
largest evaluation problem is that `supported_extra` is too permissive and does
not mean that every material claim in the finding is supported.

Six of the 45 `supported_extra` findings already contain an unsupported,
contradicted, or unknown claim. The manual sample also found one additional
factual error that the judge missed because it restated the model's incorrect
wording as a correct claim.

In the stratified ten-finding sample:

| Manual outcome | Findings |
| --- | ---: |
| Clean pass | 4 |
| Partial: useful core, but overclaim or unverified novelty | 3 |
| Fail: factual error or not useful as a current net-new finding | 3 |

This sample is deliberately stratified around important edge cases and should
not be extrapolated as an unbiased error rate for all 45 extras.

## Missing reference finding

### CRWD:0001535527-24-000024:f04

Expected: Q3 GAAP net loss attributable to CrowdStrike was $16.8 million while
non-GAAP diluted EPS remained positive at $0.93, compared with Q2 GAAP net income
of $47.0 million.

Manual verdict: **genuine output miss, partly retrieval-limited**.

The model input contained the current GAAP net loss in chunk `9d60ee3e5462` and
the non-GAAP diluted EPS in adjacent chunk `fb79bafd7f64`. The evidence was split
across a chunk boundary. The selected prior chunks did not contain the Q2 GAAP
net income of $47.0 million. The agent could have reported the current
GAAP/non-GAAP distinction, but it did not have the full prior comparison required
by the reference.

The agent emitted related findings about GAAP operating loss and non-GAAP
operating income, but not the expected net-income-versus-adjusted-EPS distinction.

## Unsupported and contradicted claims

| Output finding | Manual verdict | Reason |
| --- | --- | --- |
| AMD 109 out-05, executive award mix | Citation failure, fact supported elsewhere | The award mix appears in another supplied current chunk, but not in the cited chunk. Claim correctness and citation grounding should be scored separately. |
| CRWD 017 out-04, guidance reduction linked to commitment package | Unsupported | The cited guidance evidence does not link the reduction to the incident commitment package. |
| CRWD 017 out-06, module rates declined | Unsupported | No prior rate is supplied, and the finding body itself says the direction cannot be confirmed. |
| AMD 161 out-08, ZT announcement and unchanged timeline | Unsupported citation | The cited prior chunk is a cover page and contains no acquisition or timeline evidence. |
| AMD 172 out-04, Carter RSUs vest quarterly | Contradicted | The source says one-quarter vests on each annual anniversary, not quarterly. |
| CRWD 024 out-03, filing quantifies incident impact | Unsupported overclaim | The filing reports an operating loss and qualitative incident headwinds but does not isolate a dollar impact attributable to the incident. |

Five of the six are genuine output or citation problems. The AMD award-mix claim
shows a missing distinction in the evaluator: a claim can be factually supported
by the full supplied context while being improperly grounded by its chosen
citation.

## Unknown claims

| Output finding | Manual verdict | Reason |
| --- | --- | --- |
| AMD 121 out-10, annual roadmap and Azure availability are new | Appropriately unknown | Current facts are supported, but no prior evidence establishes novelty. |
| CRWD 017 out-09, CDW milestone absent from Q1 | Appropriately unknown | No prior evidence is supplied for the absence claim. |
| AMD 161 out-06, MI325X is newly announced and prior disclosure focused on MI300X ramp | Appropriately unknown | Current product facts are supported, but the prior-state comparison is not. |

All three unknown judgments are reasonable. They are novelty claims made without
enough prior evidence.

## Ten supported-extra findings

| Finding | Manual outcome | Materiality | Audit note |
| --- | --- | --- | --- |
| AMD 109 out-06, stock-option and RSU vesting | Partial | Secondary | Core dates and terms are supported, but the model says the RSUs vest on a "quarterly schedule" when the dates are annual. The judge silently corrected the claim and missed the model error. |
| AMD 115 out-04, Victor Peng transition agreement | Pass | Secondary | The filing supports that an agreement was being prepared. The absence of a named successor is accurate within the filing. |
| CRWD outage out-03, customer remediation | Pass | Secondary | The current filing directly supports mobilization, portal remediation information, and blog updates. |
| AMD 121 out-05, gross-margin expansion | Pass | Material | Current margin, year-over-year and sequential comparisons, prior guidance, and the stated driver are all supported. |
| AMD 121 out-10, accelerator roadmap | Partial | Secondary | Current roadmap and Azure facts are supported; the `new` classification is not established by prior evidence. |
| AMD acquisition out-04, outside date and termination fee | Pass | Material | The current agreement directly supports the deadlines, extensions, and $300 million regulatory termination fee. |
| AMD acquisition out-08, Victor Peng retirement | Fail | Low value | The retirement fact is supported, but it is unrelated prior history rather than a useful development in the acquisition filing. The asserted non-relationship is also an inference. |
| CRWD 017 out-06, module adoption decline | Fail | Secondary | The title asserts a decline that the supplied evidence cannot establish. |
| AMD 172 out-04, Carter equity awards | Fail | Secondary | The RSU vesting frequency is factually wrong, even though the PRSU terms are supported. |
| CRWD 024 out-03, operating loss due to incident costs | Partial | Material | The loss and qualitative headwinds are supported; the title and final sentence overstate causal and quantitative attribution. |

## What the audit reveals

### 1. `supported_extra` is not a clean precision metric

The reconciler can label a finding `supported_extra` even when one of its claims
is unsupported, contradicted, or unknown. At minimum, the summary should split:

- fully supported extras: every material claim is `supported`;
- partially supported extras: supported core with an unsupported or unknown
  qualifier, comparison, or interpretation; and
- unsupported extras: no reliable supported core or a material contradiction.

Using the existing judge results alone, only 39 of 45 extras have all claims
marked supported. The manual audit found at least one additional missed error, so
even 39/45 is an upper bound rather than a calibrated precision estimate.

### 2. Claim correctness and citation grounding are different questions

The AMD award-mix claim is present in the model's full input but absent from the
selected citation chunk. It should receive:

- factual support: supported by supplied context; and
- citation grounding: unsupported by the chosen citation.

The current single claim verdict conflates those outcomes.

### 3. The judge must preserve the model's exact meaning

For the AMD vesting finding, the model wrote "quarterly" while listing annual
dates. The judge decomposed this into a corrected annual-vesting claim and marked
it supported. Claim decomposition must quote or faithfully preserve the original
assertion so the judge cannot repair it silently.

### 4. Novelty requires the complete supplied prior context

Several extras have well-supported current facts but unverified `new`
classifications. The evidence judge currently focuses on cited chunks. To assess
`new`, `changed`, or `repeated`, it should receive all prior chunks supplied to
the agent, not only prior chunks the agent chose to cite.

### 5. Supported facts can still be poor product findings

The Victor Peng retirement item is true but irrelevant to the current acquisition
filing. Materiality, redundancy, and current-filing relevance need to be reported
alongside factual support. Otherwise, a verbose fact extractor can score highly
without producing a concise net-new investor report.

## Recommended evaluator changes

1. Replace `supported_extra_rate` with counts for fully supported, partial, and
   unsupported extras. **Implemented in evaluator `eval-v5`.**
2. Evaluate factual support against the complete context supplied to the agent.
3. Evaluate citation grounding separately against the cited chunks.
4. Require atomic claims to preserve the model's exact asserted meaning.
5. Assess novelty against all supplied prior chunks and return `unknown` when the
   supplied context cannot establish it.
6. Report low-value and redundant findings instead of treating them as successful
   extras.
