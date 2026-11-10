"""Runtime adapter for the trained multilingual request classifier."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from urllib.error import HTTPError, URLError

from app.config import settings


class ClassifierUnavailable(RuntimeError):
    """Raised when the trained model or its local embedding service is unavailable."""


@lru_cache(maxsize=1)
def _load_classifier():
    model_path = Path(settings.classifier_path).expanduser()
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[2] / model_path
    if not model_path.is_file():
        raise ClassifierUnavailable(f"Trained request classifier not found: {model_path}.")
    os.environ["OLLAMA_HOST"] = settings.ollama_host
    from ml.category_model import CategoryClassifier

    try:
        return CategoryClassifier.load(model_path)
    except Exception as exc:
        raise ClassifierUnavailable(
            f"Could not load the trained request classifier ({type(exc).__name__})."
        ) from exc


def classify(text: str) -> dict:
    """Classify an already-confirmed request using the trained local ML model."""
    classifier = _load_classifier()
    try:
        result = classifier.predict(text)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if "not found" in detail.lower() and "qwen3-embedding" in detail.lower():
            raise ClassifierUnavailable(
                "Ollama is missing qwen3-embedding:4b. Install it with "
                "`ollama pull qwen3-embedding:4b`."
            ) from exc
        raise ClassifierUnavailable(
            f"Ollama embedding request failed with HTTP {exc.code}."
        ) from exc
    except (URLError, ConnectionError, TimeoutError) as exc:
        raise ClassifierUnavailable(
            "Ollama is not reachable. Start Ollama and ensure qwen3-embedding:4b is installed."
        ) from exc
    except Exception as exc:
        detail = str(exc)
        if "model" in detail.lower() and "not found" in detail.lower():
            raise ClassifierUnavailable(
                "Ollama is missing qwen3-embedding:4b. Install it with "
                "`ollama pull qwen3-embedding:4b`."
            ) from exc
        raise

    category = str(result["category"]).strip().lower().replace("_", " ").replace("-", " ")
    if category == "ac":
        category = "AC"
    urgency = str(result.get("urgency", "medium")).lower()
    if urgency not in {"low", "medium", "high", "emergency"}:
        urgency = "medium"
    return {
        "category": category,
        "confidence": float(result["confidence"]),
        "urgency": urgency,
        "urgency_confidence": float(result.get("urgency_confidence", 0.0)),
        "needs_review": bool(result.get("needs_review", False)),
        "language": str(result.get("language", "unknown")),
    }
