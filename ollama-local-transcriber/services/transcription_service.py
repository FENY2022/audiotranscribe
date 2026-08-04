"""Local Faster-Whisper transcription service."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from services.model_service import safe_compute_type


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


class TranscriptionError(RuntimeError):
    """Raised when local transcription fails."""


def cuda_available() -> bool:
    """Return True when PyTorch reports CUDA availability."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(selected_device: str) -> str:
    """Resolve Auto/CPU/CUDA to a Faster-Whisper device string."""
    if selected_device.lower() == "cuda":
        if not cuda_available():
            raise TranscriptionError("CUDA was selected but is not available. Use CPU or install CUDA support.")
        return "cuda"
    if selected_device.lower() == "cpu":
        return "cpu"
    return "cuda" if cuda_available() else "cpu"


def get_whisper_model(model_name: str, device: str, compute_type: str):
    """Load and cache a Faster-Whisper model."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError("faster-whisper is not installed. Run install_windows.bat first.") from exc

    key = (model_name, device, compute_type)
    if key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
        except MemoryError as exc:
            raise TranscriptionError("Insufficient RAM or VRAM to load the selected Whisper model.") from exc
        except Exception as exc:
            raise TranscriptionError(
                "Unable to load the Whisper model. Check your internet connection for first-time download, "
                "available disk space, and selected compute type."
            ) from exc
    return _MODEL_CACHE[key]


def transcribe_audio(
    audio_path: Path,
    model_name: str = "small",
    selected_device: str = "Auto",
    compute_type: str | None = None,
    language: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    translate: bool = False,
    condition_on_previous_text: bool = False,
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
) -> dict[str, Any]:
    """Transcribe an audio file locally and return metadata plus segments.

    progress_callback(fraction, text) is called for every segment.
    cancel_check() is called between segments; returning True stops early.
    """
    start_time = time.perf_counter()
    device = resolve_device(selected_device)
    selected_compute = safe_compute_type(device, compute_type)
    model = get_whisper_model(model_name, device, selected_compute)

    cancelled = False
    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            task="translate" if translate else "transcribe",
            condition_on_previous_text=condition_on_previous_text,
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        segments: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for index, segment in enumerate(segments_iter, start=1):
            if cancel_check and cancel_check():
                cancelled = True
                break
            item: dict[str, Any] = {
                "id": index,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
            if word_timestamps and getattr(segment, "words", None):
                item["words"] = [
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "word": word.word.strip(),
                        "probability": float(getattr(word, "probability", 0.0) or 0.0),
                    }
                    for word in segment.words
                ]
            segments.append(item)
            if item["text"]:
                text_parts.append(item["text"])
            if progress_callback:
                fraction = (float(item["end"]) / duration) if duration else 0.0
                progress_callback(min(0.99, fraction), item["text"])
    except MemoryError as exc:
        raise TranscriptionError("Insufficient RAM or VRAM during transcription.") from exc
    except Exception as exc:
        raise TranscriptionError("Transcription failed. The media may be corrupted or unsupported.") from exc

    processing_time = time.perf_counter() - start_time
    return {
        "language": getattr(info, "language", None),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "processing_time": processing_time,
        "text": " ".join(text_parts).strip(),
        "segments": segments,
        "device": device,
        "compute_type": selected_compute,
        "model": model_name,
        "cancelled": cancelled,
    }
