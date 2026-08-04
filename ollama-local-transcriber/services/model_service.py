"""Model selection helpers for Whisper and Ollama."""

from __future__ import annotations

from typing import Iterable


PREFERRED_OLLAMA_MODELS = [
    "qwen3:4b",
    "qwen3:latest",
    "phi3:latest",
    "gemma4:latest",
    "deepseek-r1:8b",
]

EXCLUDED_MODEL_MARKERS = (
    "embed",
    "embedding",
    "nomic-embed-text",
    "code",
    "coder",
    "sqlcoder",
)


def safe_compute_type(device: str, requested: str | None = None) -> str:
    """Choose a safe Faster-Whisper compute type for the selected device."""
    normalized = device.lower()
    if requested and requested != "Auto":
        if normalized == "cpu" and requested in {"float16", "int8_float16"}:
            return "int8"
        return requested
    return "float16" if normalized == "cuda" else "int8"


def filter_transcript_models(models: Iterable[str]) -> list[str]:
    """Prefer general chat models over embedding or coding-only models."""
    names = list(dict.fromkeys(models))
    preferred = [name for name in PREFERRED_OLLAMA_MODELS if name in names]
    general = [
        name
        for name in names
        if not any(marker in name.lower() for marker in EXCLUDED_MODEL_MARKERS)
    ]
    ordered = list(dict.fromkeys(preferred + general))
    return ordered or names


def choose_default_ollama_model(models: Iterable[str], fallback: str = "qwen3:4b") -> str:
    """Choose the best default Ollama model from installed models."""
    filtered = filter_transcript_models(models)
    for preferred in PREFERRED_OLLAMA_MODELS:
        if preferred in filtered:
            return preferred
    return filtered[0] if filtered else fallback
