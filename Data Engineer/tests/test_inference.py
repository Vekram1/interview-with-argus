import json

import httpx
import pytest
from openai import OpenAI
from test_workflow import DEMO

from net_new.dataset import Dataset
from net_new.pipeline import (
    PROMPT,
    build_input,
    config,
    fresh_prediction,
    inference_connection,
    run_replay,
)


def test_agent_prompt_requires_evidence_for_comparison_novelty_and_causation():
    prompt = " ".join(PROMPT.split())
    assert "classify it as uncertain" in prompt
    assert "Do not infer novelty from absence" in prompt
    assert "Do not claim causation or quantified impact" in prompt
    assert "title and change text" in prompt


def test_gateway_receives_only_local_model_request(monkeypatch):
    monkeypatch.setenv("ARGUS_INFERENCE_URL", "https://inference.example/v1")
    monkeypatch.setenv("ARGUS_INFERENCE_KEY", "candidate-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = []

    def respond(request):
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "argus-eval",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "headline": "Local pipeline result",
                                        "overview": "Test response",
                                        "findings": [],
                                        "limitations": "",
                                    }
                                ),
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    def make_client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr("openai.OpenAI", make_client)
    dataset = Dataset(DEMO)
    context = build_input(dataset, dataset.replay_filings()[0], config("litellm"))
    prediction, _ = fresh_prediction(context, config("litellm"))
    assert prediction.headline == "Local pipeline result"
    request = captured[0]
    assert str(request.url) == "https://inference.example/v1/responses"
    assert request.headers["authorization"] == "Bearer candidate-key"
    body = json.loads(request.content)
    assert json.loads(body["input"]) == {"CURRENT": context["current"], "PRIOR": context["prior"]}
    assert "documents" not in body and "content_base64" not in request.content.decode()
    assert body["instructions"] and body["text"]["format"]["type"] == "json_schema"
    assert body["max_output_tokens"] == 16000


def test_gateway_mode_never_falls_back_to_provider_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("OPENAI_MODEL", "provider-model")
    monkeypatch.setenv("ARGUS_INFERENCE_URL", "https://inference.example/v1")
    monkeypatch.delenv("ARGUS_INFERENCE_KEY", raising=False)
    with pytest.raises(ValueError, match="ARGUS_INFERENCE_KEY"):
        inference_connection("litellm")
    monkeypatch.setenv("ARGUS_INFERENCE_KEY", "candidate-key")
    monkeypatch.setenv("ARGUS_INFERENCE_URL", "http://public.example/v1")
    with pytest.raises(ValueError, match="HTTPS"):
        inference_connection("litellm")


def test_gateway_outputs_replay_offline_without_credentials(tmp_path, monkeypatch):
    from net_new.pipeline import InvestorPrediction, load_records

    monkeypatch.setenv("ARGUS_INFERENCE_URL", "https://inference.example/v1")
    monkeypatch.setenv("ARGUS_INFERENCE_KEY", "private-candidate-key")
    monkeypatch.setattr(
        "net_new.pipeline.fresh_prediction",
        lambda *a: (
            InvestorPrediction(headline="Test", overview="Test", findings=[], limitations=""),
            {},
        ),
    )
    dataset = Dataset(DEMO)
    run_replay(dataset, "litellm", tmp_path / "first")
    monkeypatch.delenv("ARGUS_INFERENCE_KEY")
    monkeypatch.delenv("ARGUS_INFERENCE_URL")

    def forbidden(*args):
        raise AssertionError("Cache must never call inference")

    monkeypatch.setattr("net_new.pipeline.fresh_prediction", forbidden)
    run_replay(dataset, "cache", tmp_path / "cached", tmp_path / "first")
    assert [r["prediction"] for r in load_records(tmp_path / "first")] == [
        r["prediction"] for r in load_records(tmp_path / "cached")
    ]
    assert "private-candidate-key" not in "".join(
        p.read_text() for p in (tmp_path / "first").iterdir()
    )


def test_explicit_model_is_frozen_in_replay_config(tmp_path, monkeypatch):
    from net_new.pipeline import InvestorPrediction

    monkeypatch.setenv("ARGUS_INFERENCE_URL", "https://inference.example/v1")
    monkeypatch.setenv("ARGUS_INFERENCE_KEY", "candidate-key")
    monkeypatch.setenv("ARGUS_MODEL", "default-model")
    seen = []

    def predict(context, settings):
        seen.append(settings["model"])
        return InvestorPrediction(headline="Test", overview="Test", findings=[], limitations=""), {}

    monkeypatch.setattr("net_new.pipeline.fresh_prediction", predict)
    meta = run_replay(Dataset(DEMO), "litellm", tmp_path / "chosen", model="explicit-model")
    assert meta["config"]["model"] == "explicit-model"
    assert seen and set(seen) == {"explicit-model"}
    with pytest.raises(ValueError, match="Requested model differs"):
        run_replay(Dataset(DEMO), "cache", tmp_path / "replay", tmp_path / "chosen", model="other")


def test_account_status_sends_only_candidate_auth(monkeypatch):
    from net_new.inference_status import account_status

    monkeypatch.setenv("ARGUS_INFERENCE_URL", "https://inference.example/v1")
    monkeypatch.setenv("ARGUS_INFERENCE_KEY", "candidate-key")
    real_client = httpx.Client

    def transport(request):
        assert str(request.url) == "https://inference.example/v1/account"
        assert request.headers["Authorization"] == "Bearer candidate-key"
        assert not request.content
        return httpx.Response(
            200,
            json={
                "models": ["chosen"],
                "budget_usd": 25,
                "spent_usd": 26,
                "available_usd": 0,
                "expires_at": "2099-01-01T00:00:00Z",
            },
        )

    monkeypatch.setattr(
        "net_new.inference_status.httpx.Client",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(transport)),
    )
    assert account_status()["available_usd"] == 0
