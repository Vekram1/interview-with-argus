# Data Engineer take-home: What's net new?

At Argus, data engineers build and operate data pipelines, with an initial focus
on evaluating AI outputs, detecting quality regressions, and making changes
measurable. This exercise focuses on that quality and operations work within a
supplied pipeline.

An investor selects a company and a period on its price chart. The supplied
application surfaces nearby 8-K disclosures and explains what was announced,
what changed from earlier filings, and the evidence supporting that comparison.
We want to know how reliably it does this, whether changes improve it, and how
we would notice degradation when most outputs receive no human review.

> [!NOTE]
> **Investing context: what does "net new" mean?**
>
> Investors read company announcements to understand what has changed in the
> business and whether that changes their expectations for its future. An
> announcement can repeat familiar facts alongside a small but important update.
> "Net new" means identifying that update relative to what was already known.
>
> The documents here are public reports filed with the U.S. Securities and
> Exchange Commission (SEC). A **10-K** is an annual report, a **10-Q** is a
> quarterly report, and an **8-K** reports significant developments, such as a
> leadership change or an acquisition. An 8-K may include an attached press
> release with the useful detail. See the SEC's
> [guide to these reports](https://www.investor.gov/introduction-investing/getting-started/researching-investments/using-edgar-research-investments).
>
> For a fictional example, suppose a company says in January that it expects
> $100 million in revenue for the year. In May, it raises that expectation to
> $120 million for the same year. The revised forecast, often called **guidance**,
> is the change. Repeating the $120 million forecast in June adds no new forecast
> information. These are expectations, not revenue the company has already earned.
>
> In this app, selecting a period on the share-price chart helps an investor
> find potentially relevant announcements and inspect what changed. Timing alone
> does not establish that an announcement caused the price move. For this
> exercise, the supplied earlier filings define what was previously known.

You receive the complete local pipeline, source documents, development reference
examples, and a baseline run. Model inference is available through a funded
LiteLLM endpoint. Everything else runs locally and can be inspected or modified.

## Your task

Build a repeatable evaluation and monitoring layer around this workflow.

1. **Establish baseline quality.** Compare the supplied outputs with the labeled
   development cases. Choose and justify metrics suited to investor usefulness.
   Account for unsupported claims, missed developments, novelty judgments and
   evidence. Trace conclusions to concrete cases, including failures.
2. **Make changes measurable.** Identify consequential design choices in the
   implementation and explain which you would investigate and why. Your harness
   should support comparing a fixed baseline and another run. If you experiment,
   state your hypothesis, measure the change on the same cases, and describe
   regressions and operational trade-offs. A credible negative result is useful.
3. **Demonstrate monitoring.** Treat the historical replay as arriving batches
   with no fresh human labels. Implement checks, justify alert thresholds, and
   explain what remains invisible. If no batch crosses a justified threshold,
   say so; do not manufacture an alert.
4. **Assess transfer to another company.** Freeze your selected configuration,
   then run it on the supplied holdout company. Report observable behavior,
   failures, costs and limits of your conclusions without claiming labeled
   accuracy you have not measured. We retain its labels for assessment. If you
   change your approach after inspecting holdout results, disclose that and call
   the subsequent run exploratory.

Prioritize a working evaluation and monitoring core, then choose how far to take
the remaining work. A measured improvement is encouraged, not required for a
strong submission. Creating a large annotation set, building an ingestion system,
or redesigning the investor app is not the assignment.

## How to use the references

The development set contains nine AMD/CrowdStrike disclosures and expected
findings with current and earlier source evidence. It is a source-checked,
non-exhaustive reference set, not a string-matching answer key. Correct wording,
finding count, and grouping may vary. Use the facts, periods, novelty distinctions,
and evidence to define agreement. A supported extra finding can be valid even
when absent from the references; explain how you handle it. Flag reference errors
or ambiguity with source evidence rather than silently changing labels to fit a run.
We assess the supporting evidence, including justified challenges to a reference;
agreement with an incorrect reference is not the goal.

"Previously known" means available in the supplied earlier documents at the
filing's acceptance timestamp. Annual and quarterly reports and earlier 8-Ks
provide history. The corpus is not all public news. Multiple relevant filings or
no supported match are valid; a nearby disclosure does not establish what caused
a stock-price move. Missing and failed outputs are not the same as successful
outputs containing no findings.

Reference wording is not shown as pipeline output. View development examples
separately with `reference_app.py`. The frozen partition is in
`references/split.json`; private holdout labels are not included in your materials.
The small company sample cannot establish broad production accuracy.

## Deliverables

- Working evaluation and monitoring code with reproducible commands.
- Case-level evidence and result artifacts needed to check your claims, with
  commands to reproduce them, and any proposed reference corrections or additional
  labels. Link to large artifacts; there is no need to resend the supplied datasets.
- A short `FINDINGS.md`: what you measured and why; baseline results and important
  failures; any experiments and trade-offs; what you would alert on and at what
  thresholds; transfer results if attempted and their uncertainty; and next steps.
- Identify the reference version, baseline/run paths, selected configuration,
  and dependencies.

You may add dependencies; explain why. If you leave something unfinished, tell us
what you would do next and what evidence you would need.

We assess evaluation judgment, evidence, reproducibility, and operational
usefulness. We do not prescribe a metric, judge model, or evaluation framework.
A fluent summary, valid JSON, or a successful run does not prove correctness.

**Time window: one weekend.** Your invitation will state the submission deadline.
We expect most submissions to leave parts unfinished. A well-scoped, working core
with convincing evidence counts for more than covering everything. Explain what
you prioritized, what you left out, and why. Extra hours and optional features do
not earn credit by themselves. Report approximate time spent for context, not as
a score to maximize or minimize.

AI assistance is welcome; briefly describe how you used it in `FINDINGS.md`.
Write `FINDINGS.md` in your own words, with your evidence and reasoning.
In the follow-up walkthrough, we will use your code, evidence, and results to
discuss your decisions and their limits.
Prioritize a defensible, working core.
Optional extensions include measured pipeline improvements, a monitoring
visualization, or comparisons with historical market reactions to similar news.

## Questions and submission

If you get stuck, something is unclear, or the supplied data or inference access
isn't working, email [aman@getargus.tech](mailto:aman@getargus.tech) or reply to
your invitation. Include what you tried and any relevant command or error, with
credentials removed. You do not need investing experience; questions about the
domain are welcome too.

**Default: submit a pull request in your own fork.** Put `FINDINGS.md` next to
this README, commit your work on a branch, open the PR in your fork, and email
its link to [aman@getargus.tech](mailto:aman@getargus.tech).
**Do not open a pull request on the Argus repository.**

If GitHub is a blocker, email one ZIP containing your code, `FINDINGS.md`, and
supporting results instead. Keep credentials out of either submission.

## Getting the materials

The code and instructions are in this repository. Your hiring email states your
submission deadline and includes two attachments:

- **`candidate-data.zip`:** development and holdout company filings and exhibits,
  historical prices, source provenance, and the saved development model run.
  Holdout answers are not included.
- **`.env`:** your personal, funded inference access. The email states when that
  access expires. Keep this file private and out of Git.

The data ZIP is the only archive you need. The data and saved baseline are not in
Git, so cloning the repository alone is not sufficient. The attached data does
not expire. If an attachment is missing or access is not working, reply to your
invitation for help.

## Setup

Fork this repository, clone your fork, and create a working branch. If your
invitation specifies a starting commit, use that commit. Extract
`candidate-data.zip` inside
`Data Engineer/`, then place your supplied `.env` in that directory. You should
have `Data Engineer/data/pilot/manifest.json` and
`Data Engineer/data/holdout/manifest.json`. Run the commands below from
`Data Engineer/`.

If GitHub is a blocker, reply to your invitation so we can provide the code
another way. You can use the email-ZIP submission fallback described above.

```sh
uv sync --frozen
uv run streamlit run streamlit_app.py
# Separate reference viewer; run in another terminal.
uv run streamlit run reference_app.py --server.port 8502
uv run pytest -q
```

Your private candidate `.env` contains `ARGUS_INFERENCE_URL`,
`ARGUS_INFERENCE_KEY`, and `ARGUS_MODEL`. Use `.env.candidate.example` as a template
and provide your own `SEC_USER_AGENT` name/contact for downloads. Do not commit
credentials. Candidates receive a LiteLLM virtual key, never the provider key.
The template uses the deployed candidate endpoint and `anthropic-fast-v1`;
`inference-status` lists all six available aliases.

```sh
# Inspect the source corpus.
uv run python -m net_new.cli inspect --dataset data/pilot

# Reproduce the supplied baseline without any model access.
uv run python -m net_new.cli replay --dataset data/pilot \
  --provider cache --cache data/pilot/baseline --output runs/baseline-replay

# Generate another run using your candidate inference access.
uv run --env-file .env python -m net_new.cli replay --dataset data/pilot \
  --provider litellm --output runs/experiment-01

# Join cases and outputs. This supplies no matching algorithm or accuracy score.
uv run python -m net_new.compare --dataset data/pilot \
  --references references/development.jsonl --run runs/experiment-01 \
  --output runs/experiment-01/aligned.jsonl

# Grade the model output with the structured LLM-as-a-judge evaluator.
# The judge checks citation identity, semantic support from the cited chunks,
# coverage, comparison, classification, and supported extra findings.
uv run --env-file .env.openrouter python -m net_new.evaluate \
  --dataset data/pilot --run runs/baseline-replay \
  --references references/development.jsonl --output runs/baseline-eval

# Reproduce the evidence-gated prompt experiment and evaluate it.
uv run --env-file .env.openrouter python -m net_new.cli replay \
  --dataset data/pilot --provider litellm \
  --output runs/evidence-gated-prompt-v3
uv run --env-file .env.openrouter python -m net_new.evaluate \
  --dataset data/pilot --run runs/evidence-gated-prompt-v3 \
  --references references/development.jsonl \
  --output runs/evidence-gated-prompt-v3-eval

# Evaluate one case first; repeat --case-id to select several cases.
uv run --env-file .env.openrouter python -m net_new.evaluate \
  --dataset data/pilot --run runs/baseline-replay \
  --references references/development.jsonl --case-id AMD:0000002488-24-000109 \
  --output runs/baseline-eval-amd-109

# Final transfer run after freezing your selected configuration.
uv run --env-file .env python -m net_new.cli replay --dataset data/holdout \
  --provider litellm --output runs/holdout-final
```

For offline baseline replay, use the unmodified starter checkout; cache validation
rejects changed pipeline code/configuration or data. Keep that baseline run when
experimenting. `--activate` can display a complete model run in the investor app;
its output directory must be inside the dataset. Use `ARGUS_DATASET=data/holdout`
to open holdout outputs in the app. The UI never executes a model automatically.

The downloader is provided for additional exploration. For example, to reproduce
the holdout source corpus (not its labels):

```sh
uv run --env-file .env python -m net_new.cli download --ticker NVDA \
  --start 2024-08-28 --end 2024-11-21 --output data/nvda-extra
```

The public price snapshot has gaps, including MSFT and AAPL in the tested 2024
period. Missing prices produce an error, not invented values; downloaded source
filings remain usable. Source coverage and price provenance accompany each corpus.

The evaluator uses small finding-level evidence judgments followed by a compact
case-level reconciliation for coverage, split/merged findings, classification,
and supported extras. The primary `cases.jsonl` contains one concise result per
case: its 0–100 score, score breakdown, context-sufficiency summary, and compact
finding verdicts. Detailed model output, atomic claims, citation observations,
and judge metadata are retained separately under `audit/`.

For development references, context sufficiency checks whether the source-checked
current and prior quotes were present in the exact chunks supplied to the model.
It distinguishes intact, fragmented, and missing evidence so retrieval failures
are not confused with model failures. A digest-only chunk ID can be uniquely
recovered for evaluation, but remains marked as an invalid citation format.
`summary.json` reports coverage, claim support, semantic citation support,
claim-aware fully/partially supported extra counts, classification accuracy,
context sufficiency, scores, latency, tokens, cost, and failures. Literal quote
matching is retained only as an audit diagnostic. The
judge endpoint must be OpenAI-compatible and configured through
`EVAL_INFERENCE_URL`, `EVAL_INFERENCE_KEY`, and `EVAL_MODEL` (the `ARGUS_*` names
are accepted too).

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for input/output formats and
[references/README.md](references/README.md) for reference and alignment conventions.
`data/demo` is a fictional unit-test fixture only; it is never an investor-data
fallback or part of the scored development/holdout set.

## Inspect inference access and choose a model

```sh
uv run --env-file .env python -m net_new.cli inference-status
uv run --env-file .env python -m net_new.cli inference-status --json
uv run --env-file .env python -m net_new.cli replay --dataset data/pilot --provider litellm --model gemini-fast-v1 --output runs/gemini-fast-01
```

Choose an alias returned by `inference-status`. `--model` overrides `ARGUS_MODEL`
for that run and is recorded in its configuration. Cache replay cannot change
models. Request IDs, catalog version, reported model and available cost metadata
are saved locally; a returned alias alone does not verify the provider model.
The allowance is a soft cap: recorded spend can lag and in-flight calls can
overshoot. Exhaustion requires an operator top-up.
