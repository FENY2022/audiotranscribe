"""Text formatting helpers for transcripts, subtitles, and chunking."""

from __future__ import annotations

import re
from typing import Iterable


def seconds_to_hhmmss(seconds: float | int) -> str:
    """Convert seconds to HH:MM:SS."""
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def seconds_to_srt_time(seconds: float | int) -> str:
    """Convert seconds to an SRT timestamp."""
    value = max(0.0, float(seconds))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = int(value % 60)
    millis = int(round((value - int(value)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_segments_with_timestamps(segments: Iterable[dict]) -> str:
    """Return readable transcript text with timestamp headers."""
    blocks: list[str] = []
    for segment in segments:
        start = seconds_to_hhmmss(segment.get("start", 0))
        end = seconds_to_hhmmss(segment.get("end", 0))
        text = str(segment.get("text", "")).strip()
        if text:
            blocks.append(f"[{start} - {end}]\n{text}")
    return "\n\n".join(blocks)


def build_srt(segments: Iterable[dict]) -> str:
    """Create valid SRT content from timestamped segments."""
    entries: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = seconds_to_srt_time(segment.get("start", 0))
        end = seconds_to_srt_time(segment.get("end", 0))
        entries.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(entries) + ("\n" if entries else "")


def normalize_text(text: str) -> str:
    """Normalize line endings and trim excessive whitespace."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_text_into_chunks(text: str, max_chars: int = 9000) -> list[str]:
    """Split long text by paragraphs and sentence boundaries without blind truncation."""
    clean = normalize_text(text)
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            sentence_group: list[str] = []
            sentence_len = 0
            for sentence in sentences:
                if sentence_len + len(sentence) + 1 > max_chars and sentence_group:
                    chunks.append(" ".join(sentence_group).strip())
                    sentence_group = []
                    sentence_len = 0
                sentence_group.append(sentence)
                sentence_len += len(sentence) + 1
            if sentence_group:
                chunks.append(" ".join(sentence_group).strip())
            continue

        addition = len(paragraph) + 2
        if current_len + addition > max_chars:
            flush()
        current.append(paragraph)
        current_len += addition
    flush()
    return chunks
