"""Validation helpers for uploaded media."""

from __future__ import annotations

from pathlib import Path


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def get_extension(filename: str) -> str:
    """Return lower-case file extension."""
    return Path(filename).suffix.lower()


def is_supported_file(filename: str) -> bool:
    """Check if the file extension is supported."""
    return get_extension(filename) in SUPPORTED_EXTENSIONS


def is_video_file(filename: str) -> bool:
    """Check if the file extension is a supported video type."""
    return get_extension(filename) in VIDEO_EXTENSIONS


def is_audio_file(filename: str) -> bool:
    """Check if the file extension is a supported audio type."""
    return get_extension(filename) in AUDIO_EXTENSIONS


def validate_upload(filename: str, size: int, max_size: int = DEFAULT_MAX_UPLOAD_BYTES) -> None:
    """Validate uploaded media metadata."""
    if not is_supported_file(filename):
        raise ValueError("Unsupported file type. Please upload a supported audio or video file.")
    if size <= 0:
        raise ValueError("The uploaded file is empty.")
    if size > max_size:
        raise ValueError("The uploaded file exceeds the configured maximum upload size.")
