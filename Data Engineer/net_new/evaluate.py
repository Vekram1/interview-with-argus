"""Evaluate disclosure predictions with deterministic citation checks and an LLM judge.

The evaluator keeps three questions separate:

* Does the cited document and chunk exist and belong to the right time window?
* Does the cited current/prior source text actually support the claim?
* Is the claim useful and correctly compared with earlier disclosures?

The first question is deterministic. The second and third are delegated to a
versioned structured-output judge, while the original run record remains the
source of truth for the model output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from .dataset import Dataset, Document, local_file
from .pipeline import load_records
from .references import ReferenceCase, load_references

EVALUATOR_VERSION = "eval-v6"
RUBRIC_VERSION = "rubric-v2"


ClaimType = Literal["fact", "comparison", "classification", "interpretation", "limitation"]
ClaimVerdict = Literal["supported", "unsupported", "contradicted", "unknown"]
EvidenceQuality = Literal["complete", "partial", "insufficient", "missing", "unknown"]
Disposition = Literal[
    "expected_match", "supported_extra", "unsupported_extra", "duplicate", "unresolved"
]
Usefulness = Literal["material", "secondary", "low_value", "unknown"]
ClassificationVerdict = Literal["correct", "incorrect", "unknown"]
EvidenceAvailability = Literal["available", "fragmented", "missing", "not_required"]
ContextStatus = Literal["sufficient", "partial", "insufficient", "unknown"]


class CitationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_finding_id: str
    citation_id: str
    side: Literal["current", "prior"]
    document_id: str
    chunk_id: str
    quote: str
    document_resolves: bool
    source_eligible: bool
    chunk_resolves: bool
    chunk_id_format_valid: bool
    resolved_chunk_id: str | None
    quote_present_in_document: bool
    quote_present_in_chunk: bool
    entailment: Literal["pending", "supports", "does_not_support", "unknown"]
    entails_claim: bool | None
    quality: Literal["pending", "sufficient", "partial", "insufficient", "invalid"]


class ClaimJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_type: ClaimType
    text: str = Field(min_length=1)
    citation_ids: list[str]
    verdict: ClaimVerdict
    evidence_quality: EvidenceQuality
    rationale: str = Field(min_length=1)


class FindingEvidenceJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_finding_id: str
    claims: list[ClaimJudgeResult]
    usefulness: Usefulness
    rationale: str = Field(min_length=1)


class ClassificationJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted: Literal["new", "changed", "repeated", "uncertain"]
    acceptable_reference_values: list[Literal["new", "changed", "repeated", "uncertain"]]
    verdict: ClassificationVerdict
    rationale: str = Field(min_length=1)


class FindingJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_finding_id: str
    disposition: Disposition
    matched_reference_ids: list[str]
    claims: list[ClaimJudgeResult]
    classification: ClassificationJudgeResult
    usefulness: Usefulness
    rationale: str = Field(min_length=1)


class FindingReconciliationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_finding_id: str
    disposition: Disposition
    matched_reference_ids: list[str]
    classification: ClassificationJudgeResult
    usefulness: Usefulness
    rationale: str = Field(min_length=1)


class CaseReconciliationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    missing_expected_finding_ids: list[str]
    findings: list[FindingReconciliationResult]
    case_notes: str = Field(min_length=1)


class CaseJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    missing_expected_finding_ids: list[str]
    findings: list[FindingJudgeResult]
    case_notes: str = Field(min_length=1)


class EvaluatedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_type: ClaimType
    text: str
    citation_ids: list[str]
    verdict: ClaimVerdict
    evidence_quality: EvidenceQuality
    rationale: str


class EvaluatedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_finding_id: str
    output_index: int
    model_output: dict
    disposition: Disposition
    matched_reference_ids: list[str]
    claims: list[EvaluatedClaim]
    citations: list[CitationObservation]
    classification: ClassificationJudgeResult
    usefulness: Usefulness
    rationale: str


class CaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    ticker: str
    filing_id: str
    run_status: Literal["completed", "error", "missing"]
    expected_finding_ids: list[str]
    missing_expected_finding_ids: list[str]
    model_output: dict | None
    agent_run_metadata: dict
    findings: list[EvaluatedFinding]
    judge_result: CaseJudgeResult | None
    judge_error: str | None
    case_notes: str


class ReferenceContextEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_finding_id: str
    status: ContextStatus
    current_evidence: EvidenceAvailability
    prior_evidence: EvidenceAvailability
    chunk_integrity: Literal["intact", "fragmented", "missing", "not_applicable"]
    missing_requirements: list[str]
    reason: str


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _document_text(dataset: Dataset, document: Document) -> str:
    soup = BeautifulSoup(local_file(dataset.root, document.path).read_bytes(), "html.parser")
    for node in soup(["script", "style", "ix:hidden", "noscript"]):
        node.decompose()
    return _normalized(soup.get_text(" ", strip=True))


def _document_index(dataset: Dataset) -> dict[str, Document]:
    return {document.id: document for document in dataset.documents}


def _chunk_index(record: dict) -> dict[str, dict[str, dict]]:
    context = record.get("input") or {}
    return {
        "current": {chunk["id"]: chunk for chunk in context.get("current", [])},
        "prior": {chunk["id"]: chunk for chunk in context.get("prior", [])},
    }


def _resolve_input_chunk(
    chunks: dict[str, dict[str, dict]], side: str, document_id: str, chunk_id: str
) -> tuple[dict | None, str | None, bool]:
    """Resolve exact chunk IDs and uniquely recover a digest-only model citation."""

    exact = chunks[side].get(chunk_id)
    if exact is not None and exact.get("document_id") == document_id:
        return exact, chunk_id, True
    candidates = [
        chunk
        for candidate_id, chunk in chunks[side].items()
        if chunk.get("document_id") == document_id and candidate_id.rsplit(":", 1)[-1] == chunk_id
    ]
    if len(candidates) != 1:
        return None, None, False
    return candidates[0], candidates[0]["id"], False


def _output_finding_id(case_id: str, index: int) -> str:
    return f"{case_id}:out-{index + 1:02d}"


def inspect_output_citations(
    dataset: Dataset, filing: Document, record: dict
) -> list[CitationObservation]:
    """Perform citation pointer checks without judging semantic entailment.

    The model's evidence text may be a paraphrase or may be omitted. Semantic
    support is judged later against the actual cited input chunks. Literal quote
    matches are retained as diagnostics, not as a prerequisite for support.
    """

    prediction = record.get("prediction") or {}
    document_by_id = _document_index(dataset)
    document_texts: dict[str, str] = {}
    eligible = {
        "current": {document.id for document in dataset.current(filing)},
        "prior": {document.id for document in dataset.history(filing)},
    }
    chunks = _chunk_index(record)
    observations: list[CitationObservation] = []
    for index, finding in enumerate(prediction.get("findings", [])):
        output_id = _output_finding_id(f"{filing.ticker}:{filing.filing_id}", index)
        for side in ("current", "prior"):
            for citation_index, citation in enumerate(finding.get(f"{side}_evidence", [])):
                document_id = citation.get("document_id", "")
                chunk_id = citation.get("chunk_id", "")
                quote = citation.get("quote", "")
                document = document_by_id.get(document_id)
                resolves = document is not None
                if resolves and document_id not in document_texts:
                    document_texts[document_id] = _document_text(dataset, document)
                normalized_quote = _normalized(quote)
                quote_in_document = bool(
                    resolves
                    and normalized_quote
                    and normalized_quote in document_texts[document_id]
                )
                chunk, resolved_chunk_id, format_valid = _resolve_input_chunk(
                    chunks, side, document_id, chunk_id
                )
                quote_in_chunk = bool(
                    chunk and normalized_quote and normalized_quote in _normalized(chunk["text"])
                )
                quality = "pending"
                if not resolves or document_id not in eligible[side]:
                    quality = "invalid"
                observations.append(
                    CitationObservation(
                        output_finding_id=output_id,
                        citation_id=f"{side}-{citation_index + 1:02d}",
                        side=side,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        quote=quote,
                        document_resolves=resolves,
                        source_eligible=document_id in eligible[side],
                        chunk_resolves=chunk is not None,
                        chunk_id_format_valid=format_valid,
                        resolved_chunk_id=resolved_chunk_id,
                        quote_present_in_document=quote_in_document,
                        quote_present_in_chunk=quote_in_chunk,
                        entailment="pending",
                        entails_claim=None,
                        quality=quality,
                    )
                )
    return observations


def _quote_availability(quote: Any, chunks: list[dict]) -> EvidenceAvailability:
    relevant = [chunk for chunk in chunks if chunk.get("document_id") == quote.document_id]
    wanted = _normalized(quote.quote)
    if any(wanted in _normalized(chunk.get("text", "")) for chunk in relevant):
        return "available"

    ordered = sorted(relevant, key=lambda chunk: (chunk.get("start", 0), chunk.get("end", 0)))
    contiguous_runs: list[list[dict]] = []
    for chunk in ordered:
        if not contiguous_runs or chunk.get("start", 0) > contiguous_runs[-1][-1].get("end", 0) + 1:
            contiguous_runs.append([chunk])
        else:
            contiguous_runs[-1].append(chunk)
    if any(
        wanted in _normalized(" ".join(chunk.get("text", "") for chunk in run))
        for run in contiguous_runs
        if len(run) > 1
    ):
        return "fragmented"
    return "missing"


def _evidence_availability(quotes: list[Any], chunks: list[dict]) -> EvidenceAvailability:
    if not quotes:
        return "not_required"
    states = [_quote_availability(quote, chunks) for quote in quotes]
    if "missing" in states:
        return "missing"
    if "fragmented" in states:
        return "fragmented"
    return "available"


def assess_context_sufficiency(
    reference: ReferenceCase | None, record: dict
) -> list[ReferenceContextEvaluation]:
    """Grade whether each reference finding was answerable from the supplied context."""

    if reference is None:
        return []
    context = record.get("input") or {}
    current_chunks = context.get("current", [])
    prior_chunks = context.get("prior", [])
    results = []
    for finding in reference.findings:
        current = _evidence_availability(finding.current_evidence, current_chunks)
        prior = _evidence_availability(finding.prior_evidence, prior_chunks)
        required = [state for state in (current, prior) if state != "not_required"]
        missing_requirements = []
        if current == "missing":
            missing_requirements.append("current evidence")
        if prior == "missing":
            missing_requirements.append("prior evidence")

        if not required:
            status: ContextStatus = "unknown"
            integrity = "not_applicable"
            reason = "The reference finding contains no evidence requirements."
        elif all(state == "available" for state in required):
            status = "sufficient"
            integrity = "intact"
            reason = "All reference evidence was present intact in the model input."
        elif any(state in {"available", "fragmented"} for state in required):
            status = "partial"
            integrity = "missing" if "missing" in required else "fragmented"
            reason = (
                "Some required reference evidence was absent from the model input."
                if "missing" in required
                else "Required evidence was split across input chunks."
            )
        else:
            status = "insufficient"
            integrity = "missing"
            reason = "Required reference evidence was absent from the model input."
        results.append(
            ReferenceContextEvaluation(
                reference_finding_id=finding.finding_id,
                status=status,
                current_evidence=current,
                prior_evidence=prior,
                chunk_integrity=integrity,
                missing_requirements=missing_requirements,
                reason=reason,
            )
        )
    return results


def finalize_context_sufficiency(
    results: list[ReferenceContextEvaluation], supported_reference_ids: set[str]
) -> list[ReferenceContextEvaluation]:
    """Finalize user-facing context status after output evidence is judged."""

    finalized = []
    for result in results:
        if result.reference_finding_id in supported_reference_ids:
            finalized.append(
                result.model_copy(
                    update={
                        "status": "sufficient",
                        "reason": (
                            "A grounded output matched this reference finding, proving that "
                            "the supplied context was semantically sufficient for the result."
                        ),
                    }
                )
            )
        elif result.status != "sufficient":
            finalized.append(
                result.model_copy(
                    update={
                        "status": "unknown",
                        "reason": (
                            "Canonical reference evidence was missing or fragmented; this "
                            "does not prove the supplied context was semantically insufficient."
                        ),
                    }
                )
            )
        else:
            finalized.append(result)
    return finalized


def _supported_reference_ids(evaluation: CaseEvaluation) -> set[str]:
    supported = set()
    for finding in evaluation.findings:
        if finding.claims and all(claim.verdict == "supported" for claim in finding.claims):
            supported.update(finding.matched_reference_ids)
    return supported


def _compact_reference(
    reference: ReferenceCase | None, include_evidence: bool = True
) -> dict | None:
    if reference is None:
        return None
    findings = []
    for finding in reference.findings:
        compact_finding = {
            "finding_id": finding.finding_id,
            "title": finding.title,
            "expected_facts": finding.expected_facts,
            "comparison": finding.comparison,
            "acceptable_classifications": finding.acceptable_classifications,
            "acceptable_variations": finding.acceptable_variations,
            "adjudication_notes": finding.adjudication_notes,
        }
        if include_evidence:
            compact_finding["current_evidence"] = [
                quote.model_dump() for quote in finding.current_evidence
            ]
            compact_finding["prior_evidence"] = [
                quote.model_dump() for quote in finding.prior_evidence
            ]
        findings.append(compact_finding)
    return {
        "case_id": reference.case_id,
        "title": reference.title,
        "overview": reference.overview,
        "coverage": reference.coverage,
        "findings": findings,
        "other_supported_topics": reference.other_supported_topics,
        "limitations": reference.limitations,
    }


def _relevant_chunks(
    record: dict, observations: list[CitationObservation], side: str
) -> list[dict]:
    chunks = (record.get("input") or {}).get(side, [])
    by_id = {chunk["id"]: (index, chunk) for index, chunk in enumerate(chunks)}
    cited_ids = {
        observation.resolved_chunk_id
        for observation in observations
        if observation.side == side and observation.resolved_chunk_id in by_id
    }
    selected_indices = set()
    for chunk_id in cited_ids:
        index, chunk = by_id[chunk_id]
        selected_indices.add(index)
        for neighbor in (index - 1, index + 1):
            if (
                0 <= neighbor < len(chunks)
                and chunks[neighbor]["document_id"] == chunk["document_id"]
            ):
                selected_indices.add(neighbor)
    return [chunks[index] for index in sorted(selected_indices)]


def build_finding_judge_payload(
    dataset: Dataset,
    filing: Document,
    reference: ReferenceCase | None,
    record: dict,
    output_index: int,
    observations: list[CitationObservation],
) -> dict:
    """Build a small evidence-judgment payload for one output finding."""

    prediction = record.get("prediction") or {}
    case_id = f"{filing.ticker}:{filing.filing_id}"
    output_id = _output_finding_id(case_id, output_index)
    finding = prediction.get("findings", [])[output_index]
    finding_observations = [
        observation for observation in observations if observation.output_finding_id == output_id
    ]
    return {
        "case": {
            "case_id": case_id,
            "ticker": filing.ticker,
            "filing_id": filing.filing_id,
            "available_at": filing.available_at.isoformat(),
            "reference": _compact_reference(reference, include_evidence=False),
        },
        "model_output": {"finding_id": output_id, "findings": [finding]},
        "citation_observations": [observation.model_dump() for observation in finding_observations],
        "source_context": {
            "current_chunks": _relevant_chunks(record, finding_observations, "current"),
            "prior_chunks": _relevant_chunks(record, finding_observations, "prior"),
        },
    }


def build_reconciliation_payload(
    dataset: Dataset,
    filing: Document,
    reference: ReferenceCase | None,
    record: dict,
    observations: list[CitationObservation],
    finding_judgments: list[FindingEvidenceJudgeResult],
) -> dict:
    """Build a compact case-level payload for coverage and classification."""

    prediction = record.get("prediction") or {}
    case_id = f"{filing.ticker}:{filing.filing_id}"
    output_findings = [
        {
            "output_finding_id": _output_finding_id(case_id, index),
            "title": finding.get("title", ""),
            "announced": finding.get("announced", ""),
            "change": finding.get("change", ""),
            "classification": finding.get("classification", "uncertain"),
            "current_evidence_count": len(finding.get("current_evidence", [])),
            "prior_evidence_count": len(finding.get("prior_evidence", [])),
        }
        for index, finding in enumerate(prediction.get("findings", []))
    ]
    citation_summary = [
        {
            key: getattr(observation, key)
            for key in (
                "output_finding_id",
                "citation_id",
                "side",
                "document_id",
                "chunk_id",
                "document_resolves",
                "source_eligible",
                "chunk_resolves",
                "chunk_id_format_valid",
                "resolved_chunk_id",
                "quote_present_in_document",
                "quote_present_in_chunk",
                "entailment",
                "quality",
            )
        }
        for observation in observations
    ]
    return {
        "case": {
            "case_id": case_id,
            "ticker": filing.ticker,
            "filing_id": filing.filing_id,
            "available_at": filing.available_at.isoformat(),
            "reference": _compact_reference(reference, include_evidence=False),
        },
        "model_findings": output_findings,
        "citation_summary": citation_summary,
        "finding_judgments": [judgment.model_dump() for judgment in finding_judgments],
    }


def build_judge_payload(
    dataset: Dataset,
    filing: Document,
    reference: ReferenceCase | None,
    record: dict,
    observations: list[CitationObservation],
) -> dict:
    """Build the stable, auditable input sent to the LLM judge."""

    prediction = record.get("prediction") or {}
    input_context = record.get("input") or {}
    current_chunks = input_context.get("current", [])
    prior_chunks = input_context.get("prior", [])
    grouped = {}
    for observation in observations:
        grouped.setdefault(observation.output_finding_id, []).append(observation.model_dump())
    return {
        "case": {
            "case_id": f"{filing.ticker}:{filing.filing_id}",
            "ticker": filing.ticker,
            "filing_id": filing.filing_id,
            "available_at": filing.available_at.isoformat(),
            "reference": _compact_reference(reference),
        },
        "model_output": prediction,
        "citation_observations": grouped,
        "source_context": {
            "current_chunk_count": len(current_chunks),
            "prior_chunk_count": len(prior_chunks),
            "current_chunks": current_chunks,
            "prior_chunks": prior_chunks,
            "current_document_ids": [d.id for d in dataset.current(filing)],
            "prior_document_ids": [d.id for d in dataset.history(filing)],
        },
    }


JUDGE_INSTRUCTIONS = """You are the semantic evaluator for an investor-facing disclosure pipeline.

Evaluate the model output as an auditable artifact. Decompose each finding into
atomic claims: factual claims, comparisons with prior disclosures, classification
claims, and interpretation or limitation claims.

For every material claim, inspect the actual current and prior source chunks
associated with its citations. The model may paraphrase the source. A supplied
quote is an optional locator or summary; it does not need to be a verbatim
substring of the source. A document ID, accession number, filing date, company
name, or other metadata is not evidence. A citation passes only when the cited
source chunk text entails the claim. If the cited source text is absent, unrelated,
or too weak, mark the claim unsupported or unknown and mark the evidence quality
insufficient. Do not repair a missing citation by assuming that an uncited document
might contain the claim.

Use `contradicted` when the cited evidence conflicts with the claim. Use `unknown`
when the supplied evidence cannot establish whether the claim is true. A claim
that says an item is new merely because it is absent from retrieved excerpts is
not established as new.

When a development reference is provided, match findings by facts rather than
wording. Merged and split findings are acceptable. The reference set is
non-exhaustive, so an unmatched finding can be a supported extra. Mark it
unsupported only when its material claims lack evidence. Mark it unresolved when
the supplied evidence is insufficient to decide.

For classification, respect the reference's acceptable classifications. In
monitoring mode, where no reference is provided, return `unknown`.

Do not grade the retrieval path. Grade the investor-facing outcome and the
evidence attached to it. Return one structured result for the case.
"""

FINDING_JUDGE_INSTRUCTIONS = """You are the evidence judge for one finding from an investor-facing disclosure pipeline.

Evaluate only the supplied finding. Decompose it into atomic factual, comparison,
classification, interpretation, and limitation claims. For every material claim,
inspect the actual current and prior source chunks associated with its citations.
A model finding may paraphrase the source. A supplied quote is an optional locator
or summary; it does not need to be a verbatim substring of the source. A document
ID, accession number, filing date, company name, or metadata is not evidence. A
citation passes only when the cited source chunk text entails the claim.

If the cited chunk text is absent, unrelated, or too weak, mark the claim
unsupported or unknown and mark evidence quality insufficient. Do not repair a
bad citation by assuming an uncited document might contain the claim. Use
`contradicted` when the supplied evidence conflicts with the claim and `unknown`
when the supplied material cannot establish the claim.

Return one structured evidence judgment for this finding. Do not decide whether
the finding matches a reference finding; a separate case-level reconciler does
that after all findings have been inspected.
"""

RECONCILIATION_INSTRUCTIONS = """You are the case-level reconciler for an investor-facing disclosure pipeline.

Use the model findings and their evidence judgments to decide coverage and
classification for the whole case. Match findings to reference findings by
facts, not wording. Merged and split findings are acceptable. The reference set
is non-exhaustive, so an unmatched finding can be a supported extra. Mark it
unsupported only when its material claims lack evidence. Mark it unresolved when
the evidence judgments do not permit a reliable decision.

For each output finding, return exactly one reconciliation record. Decide whether
its classification is acceptable for the matched reference finding. If there is
no reference, return `unknown` classification. List every expected reference
finding that is not covered. Do not grade retrieval paths or invent support not
present in the finding judgments.

Return only the requested structured result. Keep rationales concise and do not
include repeated source text or extra fields.
"""


def _judge_connection(model: str | None = None) -> dict:
    url = (os.getenv("EVAL_INFERENCE_URL") or os.getenv("ARGUS_INFERENCE_URL", "")).rstrip("/")
    key = os.getenv("EVAL_INFERENCE_KEY") or os.getenv("ARGUS_INFERENCE_KEY")
    chosen_model = model or os.getenv("EVAL_MODEL") or os.getenv("ARGUS_MODEL")
    parsed = urlparse(url)
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.scheme != "https"
        or not parsed.path.rstrip("/").endswith("/v1")
    ):
        raise ValueError("Set EVAL_INFERENCE_URL or ARGUS_INFERENCE_URL to an HTTPS /v1 endpoint")
    if not key:
        raise ValueError("Set EVAL_INFERENCE_KEY or ARGUS_INFERENCE_KEY for the judge")
    if not chosen_model:
        raise ValueError("Set EVAL_MODEL or ARGUS_MODEL for the judge")
    return {"base_url": url, "api_key": key, "model": chosen_model}


def _call_structured(
    payload: dict,
    instructions: str,
    result_type: type[BaseModel],
    max_output_tokens: int,
    model: str | None,
    kind: str,
) -> tuple[BaseModel, dict]:
    connection = _judge_connection(model)
    client = OpenAI(
        base_url=connection["base_url"],
        api_key=connection["api_key"],
        timeout=120,
        max_retries=0,
    )
    raw_response = client.responses.with_raw_response.create(
        model=connection["model"],
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": result_type.__name__,
                "strict": True,
                "schema": result_type.model_json_schema(),
            }
        },
        reasoning={"effort": "low"},
        max_output_tokens=max_output_tokens,
        store=False,
    )
    response = raw_response.parse()
    if response.status != "completed":
        details = response.incomplete_details
        reason = getattr(details, "reason", None) if details else None
        raise ValueError(
            "Judge returned no structured result "
            f"(status={response.status}, reason={reason or 'unknown'})"
        )
    parsed = result_type.model_validate_json(response.output_text)
    return parsed, {
        "kind": kind,
        "response_id": response.id,
        "requested_model": connection["model"],
        "reported_model": response.model,
        "request_id": raw_response.headers.get("x-request-id"),
        "cost_usd": raw_response.headers.get("x-openrouter-cost")
        or raw_response.headers.get("x-litellm-response-cost"),
        "usage": response.usage.model_dump() if response.usage else None,
    }


def judge_finding(
    payload: dict, model: str | None = None
) -> tuple[FindingEvidenceJudgeResult, dict]:
    """Judge claim and citation support for one output finding."""

    result, metadata = _call_structured(
        payload,
        FINDING_JUDGE_INSTRUCTIONS,
        FindingEvidenceJudgeResult,
        max_output_tokens=4000,
        model=model,
        kind="finding",
    )
    return FindingEvidenceJudgeResult.model_validate(result), metadata


def reconcile_case(
    payload: dict, model: str | None = None
) -> tuple[CaseReconciliationResult, dict]:
    """Reconcile compact finding judgments at case level."""

    result, metadata = _call_structured(
        payload,
        RECONCILIATION_INSTRUCTIONS,
        CaseReconciliationResult,
        max_output_tokens=8000,
        model=model,
        kind="reconciliation",
    )
    return CaseReconciliationResult.model_validate(result), metadata


def judge_case(payload: dict, model: str | None = None) -> tuple[CaseJudgeResult, dict]:
    """Call the legacy single-request case judge."""

    result, metadata = _call_structured(
        payload,
        JUDGE_INSTRUCTIONS,
        CaseJudgeResult,
        max_output_tokens=8000,
        model=model,
        kind="legacy_case",
    )
    return CaseJudgeResult.model_validate(result), metadata


def judge_case_hybrid(
    dataset: Dataset,
    filing: Document,
    reference: ReferenceCase | None,
    record: dict,
    observations: list[CitationObservation],
    model: str | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[CaseJudgeResult, dict]:
    """Judge findings locally, then reconcile their compact results by case."""

    finding_judgments: list[FindingEvidenceJudgeResult] = []
    request_metadata: list[dict] = []
    errors: list[str] = []
    output_findings = (record.get("prediction") or {}).get("findings", [])
    case_id = f"{filing.ticker}:{filing.filing_id}"

    def judge_one(output_index: int) -> tuple[FindingEvidenceJudgeResult, dict, str | None]:
        payload = build_finding_judge_payload(
            dataset, filing, reference, record, output_index, observations
        )
        output_id = _output_finding_id(case_id, output_index)
        try:
            finding_judgment, metadata = judge_finding(payload, model=model)
        except Exception as exc:  # noqa: BLE001 - preserve case-level progress
            finding_judgment = FindingEvidenceJudgeResult(
                output_finding_id=output_id,
                claims=[],
                usefulness="unknown",
                rationale=f"Finding evidence judge failed: {type(exc).__name__}: {exc}",
            )
            metadata = {"kind": "finding", "output_finding_id": output_id, "error": str(exc)}
            return finding_judgment, metadata, f"{output_id}: {type(exc).__name__}: {exc}"
        return finding_judgment, metadata, None

    if output_findings:
        for output_index in range(len(output_findings)):
            print(
                f"{case_id}: judging finding {output_index + 1}/{len(output_findings)}", flush=True
            )
        with ThreadPoolExecutor(max_workers=min(4, len(output_findings))) as executor:
            jobs = [
                executor.submit(judge_one, output_index)
                for output_index in range(len(output_findings))
            ]
            results = [job.result() for job in jobs]
        for output_index, (finding_judgment, metadata, error) in enumerate(results):
            output_id = _output_finding_id(case_id, output_index)
            finding_judgments.append(finding_judgment)
            request_metadata.append(metadata)
            if error:
                errors.append(error)
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                (checkpoint_dir / f"{output_id}.json").write_text(
                    json.dumps(
                        {"judgment": finding_judgment.model_dump(), "metadata": metadata}, indent=2
                    )
                )

    reconciliation_payload = build_reconciliation_payload(
        dataset, filing, reference, record, observations, finding_judgments
    )
    print(f"{case_id}: reconciling {len(finding_judgments)} findings", flush=True)
    reconciliation, metadata = reconcile_case(reconciliation_payload, model=model)
    request_metadata.append(metadata)
    evidence_by_id = {judgment.output_finding_id: judgment for judgment in finding_judgments}
    findings = []
    for reconciled in reconciliation.findings:
        evidence = evidence_by_id.get(reconciled.output_finding_id)
        findings.append(
            FindingJudgeResult(
                output_finding_id=reconciled.output_finding_id,
                disposition=reconciled.disposition,
                matched_reference_ids=reconciled.matched_reference_ids,
                claims=evidence.claims if evidence else [],
                classification=reconciled.classification,
                usefulness=reconciled.usefulness,
                rationale=reconciled.rationale,
            )
        )
    return (
        CaseJudgeResult(
            case_id=reconciliation.case_id,
            missing_expected_finding_ids=reconciliation.missing_expected_finding_ids,
            findings=findings,
            case_notes=reconciliation.case_notes,
        ),
        {"requests": request_metadata, "errors": errors},
    )


def _apply_judge_to_citations(
    observations: list[CitationObservation], claims: list[ClaimJudgeResult]
) -> list[CitationObservation]:
    by_id = {observation.citation_id: observation for observation in observations}
    claim_verdicts: dict[str, list[ClaimJudgeResult]] = {}
    for claim in claims:
        for citation_id in claim.citation_ids:
            claim_verdicts.setdefault(citation_id, []).append(claim)
    for citation_id, citation_claims in claim_verdicts.items():
        observation = by_id.get(citation_id)
        if observation is None:
            continue
        verdicts = {claim.verdict for claim in citation_claims}
        if "supported" in verdicts and not verdicts - {"supported"}:
            observation.entailment = "supports"
            observation.entails_claim = True
            observation.quality = (
                "sufficient" if observation.quality == "pending" else observation.quality
            )
        elif "unsupported" in verdicts or "contradicted" in verdicts:
            observation.entailment = "does_not_support"
            observation.entails_claim = False
            observation.quality = "insufficient"
        else:
            observation.entailment = "unknown"
            observation.entails_claim = None
            observation.quality = "insufficient"
    return list(by_id.values())


def evaluate_case(
    dataset: Dataset,
    filing: Document,
    reference: ReferenceCase | None,
    record: dict,
    judge: CaseJudgeResult,
) -> CaseEvaluation:
    """Combine deterministic citation observations with one judge result."""

    case_id = f"{filing.ticker}:{filing.filing_id}"
    observations = inspect_output_citations(dataset, filing, record)
    by_finding: dict[str, list[CitationObservation]] = {}
    for observation in observations:
        by_finding.setdefault(observation.output_finding_id, []).append(observation)
    output_findings = (record.get("prediction") or {}).get("findings", [])
    expected_ids = [finding.finding_id for finding in reference.findings] if reference else []
    matched = {
        reference_id for finding in judge.findings for reference_id in finding.matched_reference_ids
    }
    missing = [finding_id for finding_id in expected_ids if finding_id not in matched]
    missing.extend(
        finding_id
        for finding_id in judge.missing_expected_finding_ids
        if finding_id in expected_ids and finding_id not in missing
    )
    evaluated: list[EvaluatedFinding] = []
    judge_by_output_id = {finding.output_finding_id: finding for finding in judge.findings}

    def unknown_classification(output: dict) -> ClassificationJudgeResult:
        predicted = output.get("classification", "uncertain")
        if predicted not in {"new", "changed", "repeated", "uncertain"}:
            predicted = "uncertain"
        return ClassificationJudgeResult(
            predicted=predicted,
            acceptable_reference_values=[],
            verdict="unknown",
            rationale="The semantic judge did not return a classification decision for this finding.",
        )

    def add_evaluated_finding(
        output_finding_id: str, output_index: int, judge_finding: FindingJudgeResult | None
    ) -> None:
        output = output_findings[output_index] if 0 <= output_index < len(output_findings) else {}
        citations = _apply_judge_to_citations(
            by_finding.get(output_finding_id, []), judge_finding.claims if judge_finding else []
        )
        if judge_finding is None:
            evaluated.append(
                EvaluatedFinding(
                    output_finding_id=output_finding_id,
                    output_index=output_index,
                    model_output=output,
                    disposition="unresolved",
                    matched_reference_ids=[],
                    claims=[],
                    citations=citations,
                    classification=unknown_classification(output),
                    usefulness="unknown",
                    rationale="The semantic judge did not return a decision for this output finding.",
                )
            )
            return
        evaluated.append(
            EvaluatedFinding(
                output_finding_id=judge_finding.output_finding_id,
                output_index=output_index,
                model_output=output,
                disposition=judge_finding.disposition,
                matched_reference_ids=judge_finding.matched_reference_ids,
                claims=[
                    EvaluatedClaim.model_validate(claim.model_dump())
                    for claim in judge_finding.claims
                ],
                citations=citations,
                classification=judge_finding.classification,
                usefulness=judge_finding.usefulness,
                rationale=judge_finding.rationale,
            )
        )

    actual_output_ids = set()
    for output_index, output in enumerate(output_findings):
        output_id = _output_finding_id(case_id, output_index)
        actual_output_ids.add(output_id)
        add_evaluated_finding(output_id, output_index, judge_by_output_id.get(output_id))

    # Preserve judge anomalies for auditability without letting them inflate model counts.
    for judge_finding in judge.findings:
        if judge_finding.output_finding_id in actual_output_ids:
            continue
        match = re.search(r":out-(\d+)$", judge_finding.output_finding_id)
        output_index = int(match.group(1)) - 1 if match else -1
        add_evaluated_finding(judge_finding.output_finding_id, output_index, judge_finding)
    return CaseEvaluation(
        case_id=case_id,
        ticker=filing.ticker,
        filing_id=filing.filing_id,
        run_status="completed" if record.get("status") == "completed" else "error",
        expected_finding_ids=expected_ids,
        missing_expected_finding_ids=missing,
        model_output=record.get("prediction"),
        agent_run_metadata={
            key: record[key]
            for key in (
                "run_id",
                "cache_hit",
                "generation_latency_ms",
                "model_metadata",
                "original_input_hash",
                "original_run_id",
            )
            if key in record
        },
        findings=evaluated,
        judge_result=judge,
        judge_error=None,
        case_notes=judge.case_notes,
    )


def empty_case_evaluation(
    filing: Document,
    reference: ReferenceCase | None,
    status: Literal["error", "missing"],
    record: dict | None = None,
    judge_error: str | None = None,
) -> CaseEvaluation:
    expected_ids = [finding.finding_id for finding in reference.findings] if reference else []
    return CaseEvaluation(
        case_id=f"{filing.ticker}:{filing.filing_id}",
        ticker=filing.ticker,
        filing_id=filing.filing_id,
        run_status=status,
        expected_finding_ids=expected_ids,
        missing_expected_finding_ids=expected_ids,
        model_output=(record or {}).get("prediction"),
        agent_run_metadata={
            key: (record or {})[key]
            for key in (
                "run_id",
                "cache_hit",
                "generation_latency_ms",
                "model_metadata",
                "original_input_hash",
                "original_run_id",
            )
            if key in (record or {})
        },
        findings=[],
        judge_result=None,
        judge_error=judge_error,
        case_notes=(
            judge_error
            if judge_error
            else "The agent did not produce a completed prediction to judge."
        ),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _finding_context_status(
    matched_reference_ids: list[str], contexts: dict[str, ReferenceContextEvaluation]
) -> ContextStatus:
    statuses = [
        contexts[reference_id].status
        for reference_id in matched_reference_ids
        if reference_id in contexts
    ]
    if not statuses:
        return "unknown"
    for status in ("insufficient", "partial", "unknown", "sufficient"):
        if status in statuses:
            return status
    return "unknown"


def _claim_support_status(claims: list[EvaluatedClaim]) -> str:
    verdicts = {claim.verdict for claim in claims}
    if verdicts == {"supported"}:
        return "supported"
    if verdicts & {"unsupported", "contradicted"} and "supported" in verdicts:
        return "mixed"
    if verdicts and verdicts <= {"unsupported", "contradicted"}:
        return "unsupported"
    return "unknown"


def _extra_support_status(finding: EvaluatedFinding) -> str | None:
    if finding.disposition not in {"supported_extra", "unsupported_extra", "unresolved"}:
        return None
    if finding.disposition == "unsupported_extra":
        return "unsupported"
    if finding.disposition == "unresolved" or not finding.claims:
        return "unresolved"

    verdicts = [claim.verdict for claim in finding.claims]
    if all(verdict == "supported" for verdict in verdicts):
        return "fully_supported"
    if any(verdict == "supported" for verdict in verdicts):
        return "partially_supported"
    if any(verdict in {"unsupported", "contradicted"} for verdict in verdicts):
        return "unsupported"
    return "unresolved"


def _citation_support_status(citations: list[CitationObservation]) -> str:
    if not citations:
        return "missing"
    supported = sum(citation.entailment == "supports" for citation in citations)
    if supported == len(citations):
        return "supported"
    if supported:
        return "mixed"
    if any(citation.entailment == "does_not_support" for citation in citations):
        return "unsupported"
    return "unknown"


def _citation_format_status(citations: list[CitationObservation]) -> str:
    if not citations:
        return "missing"
    if all(citation.chunk_id_format_valid for citation in citations):
        return "valid"
    if all(citation.chunk_resolves for citation in citations):
        return "recovered"
    return "invalid"


def compact_case_evaluation(
    case: CaseEvaluation, context_results: list[ReferenceContextEvaluation]
) -> dict:
    """Create the small, user-facing result while the CaseEvaluation remains the audit record."""

    contexts = {result.reference_finding_id: result for result in context_results}
    context_counts = {
        status: sum(result.status == status for result in context_results)
        for status in ("sufficient", "partial", "insufficient", "unknown")
    }
    if not context_results:
        case_context: ContextStatus = "unknown"
    elif context_counts["insufficient"]:
        case_context = "insufficient"
    elif context_counts["partial"]:
        case_context = "partial"
    elif context_counts["unknown"]:
        case_context = "unknown"
    else:
        case_context = "sufficient"

    claims = [claim for finding in case.findings for claim in finding.claims]
    citations = [citation for finding in case.findings for citation in finding.citations]
    expected = len(case.expected_finding_ids)
    matched_ids = set(case.expected_finding_ids) - set(case.missing_expected_finding_ids)
    correct_ids = {
        reference_id
        for finding in case.findings
        if finding.classification.verdict == "correct"
        for reference_id in finding.matched_reference_ids
        if reference_id in matched_ids
    }
    score_breakdown = {
        "coverage": round(30 * len(matched_ids) / expected, 1) if expected else 0.0,
        "groundedness": (
            round(30 * sum(claim.verdict == "supported" for claim in claims) / len(claims), 1)
            if claims
            else 0.0
        ),
        "citation_support": (
            round(
                25
                * sum(citation.entailment == "supports" for citation in citations)
                / len(citations),
                1,
            )
            if citations
            else 0.0
        ),
        "classification": (
            round(15 * len(correct_ids) / len(matched_ids), 1) if matched_ids else 0.0
        ),
    }
    score = round(sum(score_breakdown.values()), 1) if expected else None
    compact_findings = []
    for finding in case.findings:
        output = finding.model_output
        compact_findings.append(
            {
                "finding_id": finding.output_finding_id,
                "title": output.get("title", ""),
                "disposition": finding.disposition,
                "matched_reference_ids": finding.matched_reference_ids,
                "claim_support": _claim_support_status(finding.claims),
                "citation_support": _citation_support_status(finding.citations),
                "citation_format": _citation_format_status(finding.citations),
                "classification": finding.classification.verdict,
                "context": _finding_context_status(finding.matched_reference_ids, contexts),
                "notes": finding.rationale,
            }
        )
    return {
        "case_id": case.case_id,
        "run_status": case.run_status,
        "score": score,
        "score_breakdown": score_breakdown if score is not None else None,
        "context": {
            "status": case_context,
            "sufficient": context_counts["sufficient"],
            "partial": context_counts["partial"],
            "insufficient": context_counts["insufficient"],
            "unknown": context_counts["unknown"],
            "total": len(context_results),
            "rate": _rate(context_counts["sufficient"], len(context_results)),
            "retrieval_limited_finding_ids": [
                result.reference_finding_id
                for result in context_results
                if result.status in {"partial", "insufficient"}
            ],
        },
        "findings": compact_findings,
        "missing_expected_finding_ids": case.missing_expected_finding_ids,
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": round(mean(values), 2),
        "max": max(values),
    }


def _usage_summary(metadata: list[dict]) -> dict:
    usage = [item.get("usage") or {} for item in metadata]

    def total(field: str) -> int:
        return sum(int(item[field]) for item in usage if item.get(field) is not None)

    costs = []
    for item in metadata:
        try:
            if item.get("cost_usd") is not None:
                costs.append(float(item["cost_usd"]))
        except (TypeError, ValueError):
            continue
    return {
        "requests": len(metadata),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "total_tokens": total("total_tokens"),
        "cost_usd": round(sum(costs), 8) if costs else None,
    }


def _flatten_judge_metadata(metadata: list[dict]) -> list[dict]:
    requests = []
    for item in metadata:
        requests.extend(item.get("requests", [item]))
    return requests


def _operations_summary(records: list[dict], judge_metadata: list[dict]) -> dict:
    latencies = [
        float(record["generation_latency_ms"])
        for record in records
        if record.get("generation_latency_ms") is not None
    ]
    agent_metadata = [record.get("model_metadata") or {} for record in records]
    judge_requests = _flatten_judge_metadata(judge_metadata)
    return {
        "agent": {
            "latency_ms": _stats(latencies),
            "usage": _usage_summary(agent_metadata),
            "errors": sum(record.get("status") != "completed" for record in records),
        },
        "judge": {"usage": _usage_summary(judge_requests)},
    }


def summarize_cases(
    cases: list[CaseEvaluation],
    records: list[dict] | None = None,
    judge_metadata: list[dict] | None = None,
    compact_cases: list[dict] | None = None,
) -> dict:
    completed = sum(case.run_status == "completed" for case in cases)
    expected = sum(len(case.expected_finding_ids) for case in cases)
    missing = sum(len(case.missing_expected_finding_ids) for case in cases)
    findings = [finding for case in cases for finding in case.findings]
    claims = [claim for finding in findings for claim in finding.claims]
    citations = [citation for finding in findings for citation in finding.citations]
    decisive_claims = [claim for claim in claims if claim.verdict != "unknown"]
    decisive_classifications = [
        finding.classification
        for finding in findings
        if finding.classification.verdict != "unknown"
    ]
    extra_statuses = [
        status for finding in findings if (status := _extra_support_status(finding)) is not None
    ]
    extra_counts = {
        status: extra_statuses.count(status)
        for status in ("fully_supported", "partially_supported", "unsupported", "unresolved")
    }
    supported_claims = sum(claim.verdict == "supported" for claim in decisive_claims)
    bad_claims = sum(claim.verdict in {"unsupported", "contradicted"} for claim in decisive_claims)
    supported_citations = sum(c.entailment == "supports" for c in citations)
    invalid_citation_pointers = sum(
        not (
            citation.document_resolves
            and citation.source_eligible
            and citation.chunk_resolves
        )
        for citation in citations
    )
    correct_classifications = sum(x.verdict == "correct" for x in decisive_classifications)
    result = {
        "cases": {
            "total": len(cases),
            "completed": completed,
            "completion_rate": _rate(completed, len(cases)),
        },
        "findings": {
            "model_total": len(findings),
            "expected_total": expected,
            "matched_expected": expected - missing,
            "missing_expected": missing,
            "coverage_rate": _rate(expected - missing, expected),
            "fully_supported_extras": extra_counts["fully_supported"],
            "partially_supported_extras": extra_counts["partially_supported"],
            "unsupported_extras": extra_counts["unsupported"],
            "unresolved_extras": extra_counts["unresolved"],
            "extras_reviewed": len(extra_statuses),
            "fully_supported_extra_rate": _rate(
                extra_counts["fully_supported"], len(extra_statuses)
            ),
        },
        "claims": {
            "total": len(claims),
            "supported": supported_claims,
            "unsupported_or_contradicted": bad_claims,
            "unknown": sum(claim.verdict == "unknown" for claim in claims),
            "support_rate_among_decisive": _rate(supported_claims, len(decisive_claims)),
            "unsupported_rate_among_decisive": _rate(bad_claims, len(decisive_claims)),
        },
        "citations": {
            "total": len(citations),
            "semantically_supported": supported_citations,
            "semantic_support_rate": _rate(supported_citations, len(citations)),
            "invalid_or_unresolved_pointers": invalid_citation_pointers,
        },
        "classification": {
            "decisive": len(decisive_classifications),
            "correct": correct_classifications,
            "accuracy": _rate(correct_classifications, len(decisive_classifications)),
        },
        "operations": _operations_summary(records or [], judge_metadata or []),
    }
    if compact_cases is not None:
        scores = [case["score"] for case in compact_cases if case["score"] is not None]
        context_totals = {
            status: sum(case["context"][status] for case in compact_cases)
            for status in ("sufficient", "partial", "insufficient", "unknown")
        }
        context_count = sum(context_totals.values())
        result["scores"] = {
            "case_count": len(scores),
            "mean": round(mean(scores), 1) if scores else None,
        }
        result["context"] = {
            **context_totals,
            "total": context_count,
            "sufficiency_rate": _rate(context_totals["sufficient"], context_count),
        }
    return result


def run_evaluation(
    dataset: Dataset,
    run: Path,
    references: list[ReferenceCase] | None,
    output: Path,
    model: str | None = None,
    case_ids: list[str] | None = None,
) -> dict:
    if output.exists():
        raise ValueError("Evaluation output directory already exists; choose a new path")
    output.mkdir(parents=True)
    records = load_records(run)
    records_by_case: dict[str, dict] = {}
    for record in records:
        case_id = f"{record['ticker']}:{record['filing_id']}"
        if case_id in records_by_case:
            raise ValueError(f"Duplicate output case: {case_id}")
        records_by_case[case_id] = record
    reference_by_case = {reference.case_id: reference for reference in references or []}
    filings = {f"{filing.ticker}:{filing.filing_id}": filing for filing in dataset.replay_filings()}
    unknown_cases = sorted(set(records_by_case) - set(filings))
    if unknown_cases:
        raise ValueError(f"Run record does not match replay filings: {unknown_cases[0]}")
    selected_case_ids = set(case_ids) if case_ids else set(filings)
    unknown_selected = sorted(selected_case_ids - set(filings))
    if unknown_selected:
        raise ValueError(f"Requested case does not match replay filings: {unknown_selected[0]}")
    evaluations: list[CaseEvaluation] = []
    compact_evaluations: list[dict] = []
    judge_metadata: list[dict] = []
    audit_dir = output / "audit"
    audit_dir.mkdir()
    cases_path = output / "cases.jsonl"
    audit_cases_path = audit_dir / "cases.jsonl"
    with cases_path.open("w") as cases_file, audit_cases_path.open("w") as audit_cases_file:
        for case_id, filing in filings.items():
            if case_id not in selected_case_ids:
                continue
            record = records_by_case.get(case_id)
            reference = reference_by_case.get(case_id)
            if record is None:
                evaluation = empty_case_evaluation(filing, reference, "missing")
                context_results = []
            elif record.get("status") != "completed" or not record.get("prediction"):
                evaluation = empty_case_evaluation(filing, reference, "error", record=record)
                context_results = []
            else:
                observations = inspect_output_citations(dataset, filing, record)
                context_results = assess_context_sufficiency(reference, record)
                try:
                    judge, metadata = judge_case_hybrid(
                        dataset,
                        filing,
                        reference,
                        record,
                        observations,
                        model=model,
                        checkpoint_dir=audit_dir / "finding-judgments",
                    )
                except Exception as exc:  # noqa: BLE001 - isolate judge failures per case
                    evaluation = empty_case_evaluation(
                        filing,
                        reference,
                        "error",
                        record=record,
                        judge_error=f"Judge failed: {type(exc).__name__}: {exc}",
                    )
                else:
                    evaluation = evaluate_case(dataset, filing, reference, record, judge)
                    judge_metadata.append(metadata)
                context_results = finalize_context_sufficiency(
                    context_results, _supported_reference_ids(evaluation)
                )
            evaluations.append(evaluation)
            compact = compact_case_evaluation(evaluation, context_results)
            compact_evaluations.append(compact)
            cases_file.write(json.dumps(compact, ensure_ascii=False) + "\n")
            cases_file.flush()
            audit_cases_file.write(
                json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False) + "\n"
            )
            audit_cases_file.flush()
    selected_records = [
        records_by_case[case_id] for case_id in selected_case_ids if case_id in records_by_case
    ]
    summary = summarize_cases(
        evaluations,
        records=selected_records,
        judge_metadata=judge_metadata,
        compact_cases=compact_evaluations,
    )
    summary["operations"]["judge"]["failures"] = sum(
        len(item.get("errors", [])) for item in judge_metadata
    ) + sum(case.judge_error is not None for case in evaluations)
    metadata = {
        "schema_version": 2,
        "evaluator_version": EVALUATOR_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": dataset.manifest.dataset_id,
        "dataset_fingerprint": dataset.fingerprint,
        "input_run": str(run),
        "reference_version": references[0].reference_version if references else None,
        "judge_model": model or os.getenv("EVAL_MODEL") or os.getenv("ARGUS_MODEL"),
        "audit_path": str(audit_dir.resolve()),
        "summary": summary,
    }
    audit_metadata = {**metadata, "judge_responses": judge_metadata}
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    (audit_dir / "run.json").write_text(json.dumps(audit_metadata, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model runs with a structured LLM judge")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    args = parser.parse_args()
    dataset = Dataset(args.dataset)
    references = load_references(args.references, dataset) if args.references else None
    metadata = run_evaluation(
        dataset, args.run, references, args.output, model=args.model, case_ids=args.case_ids
    )
    print(
        json.dumps({"output": str(args.output.resolve()), "summary": metadata["summary"]}, indent=2)
    )


if __name__ == "__main__":
    main()
