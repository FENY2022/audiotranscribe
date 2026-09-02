"""Media preparation with local FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from utils.file_utils import safe_unlink, unique_temp_path
from utils.validators import is_video_file


MAX_CUT_SECONDS = 150 * 60


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


def _ffprobe_executable() -> str | None:
    """Find FFprobe from PATH or beside the detected FFmpeg executable."""
    path_match = shutil.which("ffprobe")
    if path_match:
        return path_match

    ffmpeg_path = _ffmpeg_executable()
    if ffmpeg_path:
        sibling = Path(ffmpeg_path).with_name("ffprobe.exe")
        if sibling.exists():
            return str(sibling)
    return None


def media_duration_seconds(source_path: Path) -> float:
    """Return media duration in seconds using FFprobe."""
    ffprobe_path = _ffprobe_executable()
    if not ffprobe_path:
        raise FFmpegMissingError("FFprobe is missing. Install FFmpeg and add it to your PATH.")

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=False, check=False)
    except FileNotFoundError as exc:
        raise FFmpegMissingError("FFprobe is missing. Install FFmpeg and add it to your PATH.") from exc
    except OSError as exc:
        raise MediaProcessingError("Unable to start FFprobe for media inspection.") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Unknown FFprobe error").strip()
        raise MediaProcessingError(f"Unable to read media duration. {detail[:500]}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise MediaProcessingError("Unable to read media duration from FFprobe output.") from exc
    if duration <= 0:
        raise MediaProcessingError("Media duration is empty or invalid.")
    return duration


def cut_media(
    source_path: Path,
    original_name: str,
    start_seconds: float,
    end_seconds: float,
    progress_callback: Any | None = None,
) -> Path:
    """Cut a media segment, capped at 150 minutes, and return the output path."""
    ffmpeg_path = _ffmpeg_executable()
    if not ffmpeg_path:
        raise FFmpegMissingError("FFmpeg is missing. Install FFmpeg and add it to your PATH.")

    if start_seconds < 0:
        raise MediaProcessingError("Start time cannot be negative.")
    if end_seconds <= start_seconds:
        raise MediaProcessingError("End time must be after start time.")
    cut_seconds = end_seconds - start_seconds
    if cut_seconds > MAX_CUT_SECONDS:
        raise MediaProcessingError("Cut length cannot exceed 150 minutes.")

    output_path = unique_temp_path(original_name)
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{cut_seconds:.3f}",
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_path),
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
        output_lines: list[str] = []
        if process.stdout:
            for line in process.stdout:
                text = line.strip()
                if text:
                    output_lines.append(text)
                if text.startswith("out_time_ms=") and progress_callback:
                    try:
                        current_seconds = float(text.split("=", 1)[1]) / 1_000_000
                    except ValueError:
                        continue
                    progress_callback(min(0.99, current_seconds / cut_seconds), "Cutting media")
        return_code = process.wait()
    except FileNotFoundError as exc:
        raise FFmpegMissingError("FFmpeg is missing. Install FFmpeg and add it to your PATH.") from exc
    except OSError as exc:
        raise MediaProcessingError("Unable to start FFmpeg for media cutting.") from exc

    if return_code != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        safe_unlink(output_path)
        detail = "\n".join(output_lines).strip() or "Unknown FFmpeg error"
        raise MediaProcessingError(f"Media cutting failed. {detail[:500]}")
    if progress_callback:
        progress_callback(1.0, "Cut complete")
    return output_path


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
