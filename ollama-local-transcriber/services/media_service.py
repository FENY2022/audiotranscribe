"""Media preparation with local FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from utils.file_utils import safe_unlink, unique_temp_path
from utils.validators import is_video_file


class FFmpegMissingError(RuntimeError):
    """Raised when FFmpeg is not available."""


class MediaProcessingError(RuntimeError):
    """Raised when media conversion fails."""


def ffmpeg_available() -> bool:
    """Return True when ffmpeg is available on PATH."""
    return _ffmpeg_executable() is not None


def _ffmpeg_executable() -> str | None:
    """Find FFmpeg from PATH or common Windows package locations."""
    path_match = shutil.which("ffmpeg")
    if path_match:
        return path_match

    winget_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        matches = sorted(winget_root.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"), reverse=True)
        if matches:
            return str(matches[0])
    return None


def prepare_audio_for_transcription(source_path: Path, original_name: str) -> tuple[Path, list[Path]]:
    """Prepare a mono 16 kHz WAV file for Faster-Whisper.

    The original uploaded file is never modified. A converted temporary WAV is
    returned and should be deleted by the caller after processing.
    """
    ffmpeg_path = _ffmpeg_executable()
    if not ffmpeg_path:
        raise FFmpegMissingError("FFmpeg is missing. Install FFmpeg and add it to your PATH.")

    output_path = unique_temp_path(original_name, ".wav")
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=None,
            shell=False,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FFmpegMissingError("FFmpeg is missing. Install FFmpeg and add it to your PATH.") from exc
    except OSError as exc:
        raise MediaProcessingError("Unable to start FFmpeg for media processing.") from exc

    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        safe_unlink(output_path)
        detail = (result.stderr or result.stdout or "Unknown FFmpeg error").strip()
        raise MediaProcessingError(f"Media conversion failed. {detail[:500]}")

    return output_path, [output_path]


def media_kind(filename: str) -> str:
    """Return audio or video for a supported filename."""
    return "video" if is_video_file(filename) else "audio"
