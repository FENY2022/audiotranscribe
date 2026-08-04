"""Local Ollama transcript processing service."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.model_service import filter_transcript_models
from utils.file_utils import read_prompt
from utils.text_utils import split_text_into_chunks


ProgressCallback = Callable[[float, str], None]


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is unavailable."""


class OllamaProcessingError(RuntimeError):
    """Raised when Ollama generation fails."""


def _client(host: str):
    try:
        import ollama
    except ImportError as exc:
        raise OllamaUnavailableError("The ollama Python package is not installed.") from exc
    return ollama.Client(host=host)


def check_connection(host: str = "http://localhost:11434") -> tuple[bool, str]:
    """Check local Ollama service availability."""
    try:
        _client(host).list()
        return True, "Ollama is available."
    except Exception:
        return False, "Ollama is unavailable. Raw transcription is still available."


def list_models(host: str = "http://localhost:11434") -> list[str]:
    """List installed local Ollama models."""
    try:
        response = _client(host).list()
        models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        names: list[str] = []
        for model in models:
            if isinstance(model, dict):
                name = model.get("name") or model.get("model")
            else:
                name = getattr(model, "name", None) or getattr(model, "model", None)
            if name:
                names.append(str(name))
        return filter_transcript_models(names)
    except Exception:
        return []


def _generate(
    prompt: str,
    model: str,
    host: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        response = _client(host).generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": max_tokens},
            stream=False,
        )
    except Exception as exc:
        raise OllamaProcessingError("Ollama generation failed. Check that Ollama is running and the model is installed.") from exc

    if isinstance(response, dict):
        return str(response.get("response", "")).strip()
    return str(getattr(response, "response", "")).strip()


def process_custom_prompt(
    transcript: str,
    prompt_template: str,
    model: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    progress_callback: ProgressCallback | None = None,
    hierarchical: bool = False,
) -> str:
    """Process transcript through Ollama with chunking for long inputs."""
    chunks = split_text_into_chunks(transcript)
    if not chunks:
        return ""

    if len(chunks) == 1:
        if progress_callback:
            progress_callback(0.2, "Processing transcript")
        result = _generate(
            f"{prompt_template}\n\nTRANSCRIPT:\n{chunks[0]}",
            model,
            host,
            temperature,
            max_tokens,
        )
        if progress_callback:
            progress_callback(1.0, "Complete")
        return result

    partial_results: list[str] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(index / (total + 1), f"Processing chunk {index} of {total}")
        chunk_prompt = f"{prompt_template}\n\nProcess this transcript chunk. Do not invent missing context.\n\nTRANSCRIPT CHUNK:\n{chunk}"
        partial_results.append(_generate(chunk_prompt, model, host, temperature, max_tokens))

    combined = "\n\n".join(partial_results)
    if hierarchical:
        final_prompt = (
            f"{prompt_template}\n\nThe following are summaries of transcript chunks. "
            "Combine them into one final coherent result without inventing information.\n\n"
            f"CHUNK SUMMARIES:\n{combined}"
        )
    else:
        final_prompt = (
            "Merge the following processed transcript chunks into one coherent result. "
            "Remove duplicate headings where appropriate and do not invent information.\n\n"
            f"{combined}"
        )
    if progress_callback:
        progress_callback(0.95, "Combining chunk results")
    return _generate(final_prompt, model, host, temperature, max_tokens)


def clean_transcript(transcript: str, model: str, host: str, temperature: float, max_tokens: int, progress_callback: ProgressCallback | None = None) -> str:
    """Clean and format a transcript with Ollama."""
    return process_custom_prompt(transcript, read_prompt("clean_transcript.txt"), model, host, temperature, max_tokens, progress_callback)


def generate_summary(transcript: str, model: str, host: str, temperature: float, max_tokens: int, progress_callback: ProgressCallback | None = None) -> str:
    """Generate a hierarchical summary with Ollama."""
    return process_custom_prompt(transcript, read_prompt("summary.txt"), model, host, temperature, max_tokens, progress_callback, hierarchical=True)


def generate_meeting_minutes(transcript: str, model: str, host: str, temperature: float, max_tokens: int, progress_callback: ProgressCallback | None = None) -> str:
    """Generate meeting minutes with Ollama."""
    return process_custom_prompt(transcript, read_prompt("meeting_minutes.txt"), model, host, temperature, max_tokens, progress_callback, hierarchical=True)


def generate_action_items(transcript: str, model: str, host: str, temperature: float, max_tokens: int, progress_callback: ProgressCallback | None = None) -> str:
    """Extract action items with Ollama."""
    return process_custom_prompt(transcript, read_prompt("action_items.txt"), model, host, temperature, max_tokens, progress_callback, hierarchical=True)
