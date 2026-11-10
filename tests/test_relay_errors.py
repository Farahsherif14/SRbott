"""K4 guard for groq_relay.py: a Groq error never sends or prints the user's text.

Groq's error text can contain the model's output (e.g. "failed_generation" when the JSON is
invalid), and that output repeats what the resident wrote. The relay must answer with the
error TYPE and status code only. No network, no key: the Groq client is a fake.
"""

import importlib
from types import ModuleType, SimpleNamespace

import dotenv
import groq
import httpx
import pytest
from fastapi.testclient import TestClient

FAKE_USER_TEXT = "Fake Resident, 0100000000, the kitchen tap leaks in unit A-101"


@pytest.fixture
def relay(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The relay module, imported without reading the real .env (it calls load_dotenv())."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    return importlib.import_module("groq_relay")


def groq_bad_request() -> groq.BadRequestError:
    """A real Groq error class, as the SDK raises it for an HTTP 400 answer."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"code": "json_validate_failed", "failed_generation": FAKE_USER_TEXT}}
    return groq.BadRequestError(
        f"Error code: 400 - {body}", response=httpx.Response(400, request=request), body=body
    )


def fake_client(error: Exception) -> SimpleNamespace:
    """Looks like groq.Groq() to the relay: client.chat.completions.create(...) raises."""

    def create(**kwargs: object) -> None:
        raise error

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (groq_bad_request(), {"error": "BadRequestError", "status_code": 400}),
        (RuntimeError(FAKE_USER_TEXT), {"error": "RuntimeError", "status_code": None}),
    ],
)
def test_provider_error_returns_only_type_and_status(
    relay: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    expected: dict[str, object],
) -> None:
    monkeypatch.setattr(relay, "_groq", lambda: fake_client(error))

    response = TestClient(relay.app).post(
        "/extract", json={"model": "fake-model", "prompt": FAKE_USER_TEXT}
    )

    assert response.status_code == 502
    assert response.json() == expected
    printed = capsys.readouterr()
    assert FAKE_USER_TEXT not in response.text + printed.out + printed.err


def test_health_shows_only_whether_a_key_is_set(
    relay: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")

    response = TestClient(relay.app).get("/health")

    assert response.json() == {"status": "ok", "has_key": True}
    assert "fake-groq-key" not in response.text
