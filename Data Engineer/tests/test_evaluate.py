import json
from threading import Barrier

from net_new.dataset import Dataset
from net_new.evaluate import (
    CaseJudgeResult,
    CaseReconciliationResult,
    ClaimJudgeResult,
    ClassificationJudgeResult,
    FindingEvidenceJudgeResult,
    FindingJudgeResult,
    FindingReconciliationResult,
    assess_context_sufficiency,
    build_finding_judge_payload,
    build_judge_payload,
    build_reconciliation_payload,
    compact_case_evaluation,
    evaluate_case,
    finalize_context_sufficiency,
    inspect_output_citations,
    judge_case_hybrid,
    run_evaluation,
    summarize_cases,
)
from net_new.pipeline import build_input, config
from net_new.references import Quote, ReferenceCase, ReferenceFinding, load_references

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def demo_case_and_record():
    dataset = Dataset(ROOT / "data/demo")
    filing = dataset.replay_filings()[0]
    settings = config("litellm")
    context = build_input(dataset, filing, settings)
    chunk = context["current"][0]
    record = {
        "filing_id": filing.filing_id,
        "ticker": filing.ticker,
        "status": "completed",
        "input": context,
        "prediction": {
            "findings": [
                {
                    "title": "Revenue guidance",
                    "announced": "2024-09-03",
                    "change": "Aster Systems expects annual revenue between $105 million and $115 million.",
                    "classification": "changed",
                    "current_evidence": [
                        {
                            "document_id": chunk["document_id"],
                            "chunk_id": chunk["id"],
                            "quote": chunk["text"],
                        }
                    ],
                    "prior_evidence": [],
                }
            ]
        },
    }
    return dataset, filing, record


def demo_reference(dataset, filing, record, *, include_prior=True):
    current = record["input"]["current"][0]
    prior = record["input"]["prior"][0]
    return ReferenceCase(
        reference_version="test-v1",
        case_id=f"{filing.ticker}:{filing.filing_id}",
        split="development",
        dataset_fingerprint=dataset.fingerprint,
        filing_id=filing.filing_id,
        ticker=filing.ticker,
        available_at=filing.available_at.isoformat(),
        current_document_ids=[document.id for document in dataset.current(filing)],
        history_document_ids=[document.id for document in dataset.history(filing)],
        title="Guidance changed",
        overview="Aster raised guidance.",
        findings=[
            ReferenceFinding(
                finding_id=f"{filing.ticker}:{filing.filing_id}:f01",
                title="Revenue guidance increased",
                acceptable_classifications=["changed"],
                expected_facts=["Annual revenue guidance increased."],
                comparison="The current range is above the prior range.",
                current_evidence=[
                    Quote(
                        document_id=current["document_id"],
                        quote="Aster Systems now expects annual revenue between $105 million and $115 million",
                    )
                ],
                prior_evidence=(
                    [
                        Quote(
                            document_id=prior["document_id"],
                            quote="Aster Systems expects annual revenue between $90 million and $100 million",
                        )
                    ]
                    if include_prior
                    else []
                ),
                acceptable_variations=[],
                adjudication_notes="Compare the two ranges.",
            )
        ],
        coverage="non_exhaustive",
        other_supported_topics=[],
        limitations=[],
        review_status="source_checked_single_reviewer",
    )


def test_inspect_output_citations_distinguishes_document_and_chunk_resolution():
    dataset, filing, record = demo_case_and_record()
    citation = record["prediction"]["findings"][0]["current_evidence"][0]
    citation["quote"] = "Aster Systems Inc."

    observations = inspect_output_citations(dataset, filing, record)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.document_resolves is True
    assert observation.source_eligible is True
    assert observation.quote_present_in_document is True
    assert observation.quote_present_in_chunk is True
    assert observation.entailment == "pending"
    assert observation.entails_claim is None


def test_inspect_output_citations_recovers_unique_abbreviated_chunk_id():
    dataset, filing, record = demo_case_and_record()
    citation = record["prediction"]["findings"][0]["current_evidence"][0]
    full_chunk_id = citation["chunk_id"]
    citation["chunk_id"] = full_chunk_id.rsplit(":", 1)[-1]
    citation["quote"] = "Aster Systems Inc."

    observation = inspect_output_citations(dataset, filing, record)[0]

    assert observation.chunk_resolves is True
    assert observation.chunk_id_format_valid is False
    assert observation.resolved_chunk_id == full_chunk_id
    assert observation.quote_present_in_chunk is True
    payload = build_finding_judge_payload(dataset, filing, None, record, 0, [observation])
    assert payload["source_context"]["current_chunks"][0]["id"] == full_chunk_id


def test_context_sufficiency_uses_reference_evidence_in_actual_model_input():
    dataset, filing, record = demo_case_and_record()
    reference = demo_reference(dataset, filing, record)

    context = assess_context_sufficiency(reference, record)[0]

    assert context.status == "sufficient"
    assert context.current_evidence == "available"
    assert context.prior_evidence == "available"
    assert context.chunk_integrity == "intact"
    assert context.missing_requirements == []


def test_context_sufficiency_attributes_omitted_prior_evidence_to_retrieval():
    dataset, filing, record = demo_case_and_record()
    reference = demo_reference(dataset, filing, record)
    record["input"]["prior"] = []

    context = assess_context_sufficiency(reference, record)[0]

    assert context.status == "partial"
    assert context.current_evidence == "available"
    assert context.prior_evidence == "missing"
    assert context.missing_requirements == ["prior evidence"]


def test_context_sufficiency_marks_unneeded_prior_evidence_explicitly():
    dataset, filing, record = demo_case_and_record()
    reference = demo_reference(dataset, filing, record, include_prior=False)

    context = assess_context_sufficiency(reference, record)[0]

    assert context.status == "sufficient"
    assert context.prior_evidence == "not_required"


def test_context_sufficiency_detects_evidence_split_across_adjacent_chunks():
    dataset, filing, record = demo_case_and_record()
    reference = demo_reference(dataset, filing, record, include_prior=False)
    document_id = record["input"]["current"][0]["document_id"]
    record["input"]["current"] = [
        {
            "id": f"{document_id}:left",
            "document_id": document_id,
            "text": "Aster Systems now expects annual revenue between $105 million",
            "start": 0,
            "end": 63,
        },
        {
            "id": f"{document_id}:right",
            "document_id": document_id,
            "text": "and $115 million",
            "start": 64,
            "end": 80,
        },
    ]

    context = assess_context_sufficiency(reference, record)[0]

    assert context.status == "partial"
    assert context.current_evidence == "fragmented"
    assert context.chunk_integrity == "fragmented"


def test_context_status_does_not_treat_missing_canonical_quote_as_proof_of_insufficiency():
    dataset, filing, record = demo_case_and_record()
    reference = demo_reference(dataset, filing, record, include_prior=False)
    reference.findings[0].current_evidence[0].quote = (
        "A canonical wording that is not present in the supplied chunk."
    )

    availability = assess_context_sufficiency(reference, record)
    context = finalize_context_sufficiency(availability, supported_reference_ids=set())[0]

    assert context.status == "unknown"
    assert "does not prove" in context.reason


def test_grounded_reference_match_proves_context_was_semantically_sufficient():
    dataset, filing, record = demo_case_and_record()
    reference = demo_reference(dataset, filing, record, include_prior=False)
    reference.findings[0].current_evidence[0].quote = (
        "A canonical wording that is not present in the supplied chunk."
    )

    availability = assess_context_sufficiency(reference, record)
    finding_id = reference.findings[0].finding_id
    context = finalize_context_sufficiency(availability, {finding_id})[0]

    assert context.status == "sufficient"
    assert context.current_evidence == "missing"
    assert "grounded output" in context.reason


def test_judge_payload_includes_source_chunks_not_only_citation_metadata():
    dataset, filing, record = demo_case_and_record()
    observations = inspect_output_citations(dataset, filing, record)

    payload = build_judge_payload(dataset, filing, None, record, observations)

    assert payload["source_context"]["current_chunks"] == record["input"]["current"]
    assert payload["source_context"]["prior_chunks"] == record["input"]["prior"]


def test_finding_judge_payload_is_scoped_to_one_finding_and_relevant_chunks():
    dataset, filing, record = demo_case_and_record()
    observations = inspect_output_citations(dataset, filing, record)

    payload = build_finding_judge_payload(dataset, filing, None, record, 0, observations)

    assert len(payload["model_output"]["findings"]) == 1
    assert payload["model_output"]["findings"][0] == record["prediction"]["findings"][0]
    assert payload["source_context"]["current_chunks"] == [record["input"]["current"][0]]
    assert payload["case"]["reference"] is None


def test_reconciliation_payload_contains_judgments_without_raw_source_context():
    dataset, filing, record = demo_case_and_record()
    observations = inspect_output_citations(dataset, filing, record)
    evidence = FindingEvidenceJudgeResult(
        output_finding_id=f"{filing.ticker}:{filing.filing_id}:out-01",
        claims=[],
        usefulness="unknown",
        rationale="Test evidence judgment.",
    )

    payload = build_reconciliation_payload(dataset, filing, None, record, observations, [evidence])

    assert "source_context" not in payload
    assert payload["finding_judgments"][0] == evidence.model_dump()
    assert "current_evidence" not in payload["model_findings"][0]
    assert "quote" not in payload["model_findings"][0]


def test_hybrid_judge_scopes_evidence_then_reconciles_case(monkeypatch):
    dataset, filing, record = demo_case_and_record()
    observations = inspect_output_citations(dataset, filing, record)
    calls = []

    def fake_finding(payload, model=None):
        calls.append(("finding", payload))
        output_id = payload["model_output"]["finding_id"]
        return (
            FindingEvidenceJudgeResult(
                output_finding_id=output_id,
                claims=[],
                usefulness="material",
                rationale="Evidence is sufficient.",
            ),
            {"kind": "finding", "usage": None},
        )

    def fake_reconcile(payload, model=None):
        calls.append(("reconciliation", payload))
        output_id = payload["model_findings"][0]["output_finding_id"]
        return (
            CaseReconciliationResult(
                case_id=f"{filing.ticker}:{filing.filing_id}",
                missing_expected_finding_ids=[],
                findings=[
                    FindingReconciliationResult(
                        output_finding_id=output_id,
                        disposition="supported_extra",
                        matched_reference_ids=[],
                        classification=ClassificationJudgeResult(
                            predicted="changed",
                            acceptable_reference_values=[],
                            verdict="unknown",
                            rationale="No reference.",
                        ),
                        usefulness="material",
                        rationale="Supported extra.",
                    )
                ],
                case_notes="Reconciled.",
            ),
            {"kind": "reconciliation", "usage": None},
        )

    monkeypatch.setattr("net_new.evaluate.judge_finding", fake_finding)
    monkeypatch.setattr("net_new.evaluate.reconcile_case", fake_reconcile)

    result, metadata = judge_case_hybrid(dataset, filing, None, record, observations)

    assert [kind for kind, _ in calls] == ["finding", "reconciliation"]
    assert "source_context" not in calls[-1][1]
    assert result.findings[0].disposition == "supported_extra"
    assert len(metadata["requests"]) == 2


def test_hybrid_judge_runs_independent_finding_judgments_concurrently(monkeypatch):
    dataset, filing, record = demo_case_and_record()
    record["prediction"]["findings"].append(record["prediction"]["findings"][0].copy())
    observations = inspect_output_citations(dataset, filing, record)
    barrier = Barrier(2)

    def fake_finding(payload, model=None):
        barrier.wait(timeout=1)
        output_id = payload["model_output"]["finding_id"]
        return (
            FindingEvidenceJudgeResult(
                output_finding_id=output_id,
                claims=[],
                usefulness="material",
                rationale="Evidence judged.",
            ),
            {"kind": "finding", "output_finding_id": output_id, "usage": None},
        )

    def fake_reconcile(payload, model=None):
        output_ids = [item["output_finding_id"] for item in payload["finding_judgments"]]
        expected_ids = [
            f"{filing.ticker}:{filing.filing_id}:out-01",
            f"{filing.ticker}:{filing.filing_id}:out-02",
        ]
        assert output_ids == expected_ids
        return (
            CaseReconciliationResult(
                case_id=f"{filing.ticker}:{filing.filing_id}",
                missing_expected_finding_ids=[],
                findings=[
                    FindingReconciliationResult(
                        output_finding_id=output_id,
                        disposition="supported_extra",
                        matched_reference_ids=[],
                        classification=ClassificationJudgeResult(
                            predicted="changed",
                            acceptable_reference_values=[],
                            verdict="unknown",
                            rationale="No reference.",
                        ),
                        usefulness="material",
                        rationale="Supported extra.",
                    )
                    for output_id in output_ids
                ],
                case_notes="Reconciled.",
            ),
            {"kind": "reconciliation", "usage": None},
        )

    monkeypatch.setattr("net_new.evaluate.judge_finding", fake_finding)
    monkeypatch.setattr("net_new.evaluate.reconcile_case", fake_reconcile)

    _, metadata = judge_case_hybrid(dataset, filing, None, record, observations)

    assert metadata["errors"] == []
    assert len(metadata["requests"]) == 3


def test_openrouter_api_v1_endpoint_is_accepted(monkeypatch):
    monkeypatch.setenv("EVAL_INFERENCE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("EVAL_INFERENCE_KEY", "test-key")
    monkeypatch.setenv("EVAL_MODEL", "test-model")

    from net_new.evaluate import _judge_connection

    assert _judge_connection() == {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "model": "test-model",
    }


def test_evaluate_case_preserves_failed_claim_evidence_even_when_citation_resolves():
    dataset, filing, record = demo_case_and_record()
    reference = None
    judge = CaseJudgeResult(
        case_id=f"{filing.ticker}:{filing.filing_id}",
        missing_expected_finding_ids=[],
        findings=[
            FindingJudgeResult(
                output_finding_id=f"{filing.ticker}:{filing.filing_id}:out-01",
                disposition="supported_extra",
                matched_reference_ids=[],
                claims=[
                    ClaimJudgeResult(
                        claim_id="c01",
                        claim_type="fact",
                        text="Aster Systems expects annual revenue between $105 million and $115 million.",
                        citation_ids=["current-01"],
                        verdict="unsupported",
                        evidence_quality="insufficient",
                        rationale="The quoted company header does not state the revenue guidance.",
                    )
                ],
                classification=ClassificationJudgeResult(
                    predicted="changed",
                    acceptable_reference_values=[],
                    verdict="unknown",
                    rationale="No reference is available in monitoring mode.",
                ),
                usefulness="unknown",
                rationale="The claim lacks auditable evidence.",
            )
        ],
        case_notes="Monitoring case.",
    )

    result = evaluate_case(dataset, filing, reference, record, judge)

    citation = result.findings[0].citations[0]
    assert citation.document_resolves is True
    assert citation.quote_present_in_document is True
    assert citation.entails_claim is False
    assert result.findings[0].claims[0].verdict == "unsupported"


def test_evaluate_case_accepts_semantic_support_for_paraphrased_evidence():
    dataset, filing, record = demo_case_and_record()
    citation = record["prediction"]["findings"][0]["current_evidence"][0]
    citation["quote"] = "Paraphrase: management raised its annual outlook."
    output_id = f"{filing.ticker}:{filing.filing_id}:out-01"
    judge = CaseJudgeResult(
        case_id=f"{filing.ticker}:{filing.filing_id}",
        missing_expected_finding_ids=[],
        findings=[
            FindingJudgeResult(
                output_finding_id=output_id,
                disposition="supported_extra",
                matched_reference_ids=[],
                claims=[
                    ClaimJudgeResult(
                        claim_id="c01",
                        claim_type="fact",
                        text="Annual revenue guidance increased.",
                        citation_ids=["current-01"],
                        verdict="supported",
                        evidence_quality="complete",
                        rationale="The cited input chunk supports the paraphrased claim.",
                    )
                ],
                classification=ClassificationJudgeResult(
                    predicted="changed",
                    acceptable_reference_values=[],
                    verdict="unknown",
                    rationale="No reference is available in monitoring mode.",
                ),
                usefulness="material",
                rationale="The cited chunk supports the finding.",
            )
        ],
        case_notes="Monitoring case.",
    )

    result = evaluate_case(dataset, filing, None, record, judge)

    citation_observation = result.findings[0].citations[0]
    assert citation_observation.quote_present_in_document is False
    assert citation_observation.quote_present_in_chunk is False
    assert citation_observation.entailment == "supports"
    assert citation_observation.quality == "sufficient"


def test_evaluate_case_keeps_output_findings_the_judge_omits():
    dataset, filing, record = demo_case_and_record()
    record["prediction"]["findings"].append(record["prediction"]["findings"][0].copy())
    judge = CaseJudgeResult(
        case_id=f"{filing.ticker}:{filing.filing_id}",
        missing_expected_finding_ids=[],
        findings=[],
        case_notes="The judge returned no per-finding decisions.",
    )

    result = evaluate_case(dataset, filing, None, record, judge)

    assert len(result.findings) == 2
    assert {finding.disposition for finding in result.findings} == {"unresolved"}


def test_summary_counts_only_all_supported_claim_extras_as_fully_supported():
    dataset, filing, record = demo_case_and_record()
    original = record["prediction"]["findings"][0]
    record["prediction"]["findings"] = [original, original.copy(), original.copy()]
    case_id = f"{filing.ticker}:{filing.filing_id}"

    def classification():
        return ClassificationJudgeResult(
            predicted="changed",
            acceptable_reference_values=[],
            verdict="unknown",
            rationale="No reference is available in monitoring mode.",
        )

    def claim(claim_id, verdict):
        return ClaimJudgeResult(
            claim_id=claim_id,
            claim_type="fact",
            text="Annual revenue guidance increased.",
            citation_ids=["current-01"],
            verdict=verdict,
            evidence_quality="complete" if verdict == "supported" else "insufficient",
            rationale="Test claim judgment.",
        )

    judge = CaseJudgeResult(
        case_id=case_id,
        missing_expected_finding_ids=[],
        findings=[
            FindingJudgeResult(
                output_finding_id=f"{case_id}:out-01",
                disposition="supported_extra",
                matched_reference_ids=[],
                claims=[claim("c01", "supported")],
                classification=classification(),
                usefulness="material",
                rationale="Every claim is supported.",
            ),
            FindingJudgeResult(
                output_finding_id=f"{case_id}:out-02",
                disposition="supported_extra",
                matched_reference_ids=[],
                claims=[claim("c01", "supported"), claim("c02", "unknown")],
                classification=classification(),
                usefulness="secondary",
                rationale="The core is supported but one claim is unknown.",
            ),
            FindingJudgeResult(
                output_finding_id=f"{case_id}:out-03",
                disposition="unsupported_extra",
                matched_reference_ids=[],
                claims=[claim("c01", "unsupported")],
                classification=classification(),
                usefulness="low_value",
                rationale="No claim is supported.",
            ),
        ],
        case_notes="Extra-support summary test.",
    )
    evaluation = evaluate_case(dataset, filing, None, record, judge)

    findings = summarize_cases([evaluation])["findings"]

    assert findings["extras_reviewed"] == 3
    assert findings["fully_supported_extras"] == 1
    assert findings["partially_supported_extras"] == 1
    assert findings["unsupported_extras"] == 1
    assert findings["unresolved_extras"] == 0
    assert findings["fully_supported_extra_rate"] == 0.3333
    assert "supported_extras" not in findings
    assert "supported_extra_rate" not in findings


def test_compact_case_result_hides_audit_payload_and_explains_score():
    dataset, filing, record = demo_case_and_record()
    reference = demo_reference(dataset, filing, record)
    reference_id = reference.findings[0].finding_id
    output_id = f"{filing.ticker}:{filing.filing_id}:out-01"
    judge = CaseJudgeResult(
        case_id=f"{filing.ticker}:{filing.filing_id}",
        missing_expected_finding_ids=[],
        findings=[
            FindingJudgeResult(
                output_finding_id=output_id,
                disposition="expected_match",
                matched_reference_ids=[reference_id],
                claims=[
                    ClaimJudgeResult(
                        claim_id="c01",
                        claim_type="fact",
                        text="Annual revenue guidance increased.",
                        citation_ids=["current-01"],
                        verdict="supported",
                        evidence_quality="complete",
                        rationale="The quote gives the new range.",
                    )
                ],
                classification=ClassificationJudgeResult(
                    predicted="changed",
                    acceptable_reference_values=["changed"],
                    verdict="correct",
                    rationale="The guidance range changed.",
                ),
                usefulness="material",
                rationale="Matches the expected development.",
            )
        ],
        case_notes="Complete.",
    )
    evaluation = evaluate_case(dataset, filing, reference, record, judge)
    contexts = assess_context_sufficiency(reference, record)

    compact = compact_case_evaluation(evaluation, contexts)

    assert compact["score"] == 100.0
    assert compact["score_breakdown"] == {
        "coverage": 30.0,
        "groundedness": 30.0,
        "citation_support": 25.0,
        "classification": 15.0,
    }
    assert compact["context"]["status"] == "sufficient"
    assert compact["findings"][0]["context"] == "sufficient"
    assert "model_output" not in compact
    assert "judge_result" not in compact
    assert "claims" not in compact["findings"][0]
    assert "citations" not in compact["findings"][0]


def test_run_evaluation_writes_compact_results_and_separate_audit(tmp_path, monkeypatch):
    dataset, filing, record = demo_case_and_record()
    reference = demo_reference(dataset, filing, record)
    output_id = f"{filing.ticker}:{filing.filing_id}:out-01"

    def fake_hybrid(*args, **kwargs):
        return (
            CaseJudgeResult(
                case_id=reference.case_id,
                missing_expected_finding_ids=[],
                findings=[
                    FindingJudgeResult(
                        output_finding_id=output_id,
                        disposition="expected_match",
                        matched_reference_ids=[reference.findings[0].finding_id],
                        claims=[],
                        classification=ClassificationJudgeResult(
                            predicted="changed",
                            acceptable_reference_values=["changed"],
                            verdict="correct",
                            rationale="Correct.",
                        ),
                        usefulness="material",
                        rationale="Matched.",
                    )
                ],
                case_notes="Complete.",
            ),
            {"requests": [{"kind": "test", "usage": None}], "errors": []},
        )

    monkeypatch.setattr("net_new.evaluate.judge_case_hybrid", fake_hybrid)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "records.jsonl").write_text(json.dumps(record) + "\n")
    output_dir = tmp_path / "evaluation"

    run_evaluation(
        dataset,
        run_dir,
        [reference],
        output_dir,
        case_ids=[reference.case_id],
    )

    compact = json.loads((output_dir / "cases.jsonl").read_text())
    audit = json.loads((output_dir / "audit" / "cases.jsonl").read_text())
    run_metadata = json.loads((output_dir / "run.json").read_text())
    audit_metadata = json.loads((output_dir / "audit" / "run.json").read_text())
    assert "model_output" not in compact
    assert audit["model_output"] == record["prediction"]
    assert "judge_responses" not in run_metadata
    assert audit_metadata["judge_responses"][0]["requests"][0]["kind"] == "test"


def test_development_reference_ids_are_preserved():
    dataset_path = ROOT / "data/pilot"
    if not (dataset_path / "manifest.json").exists():
        return
    dataset = Dataset(dataset_path)
    references = load_references(ROOT / "references/development.jsonl", dataset)
    assert len(references) == 9
    assert sum(len(case.findings) for case in references) == 23
