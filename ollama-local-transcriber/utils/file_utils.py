"""File and path helpers."""

from __future__ import annotations

import re
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def ensure_directories() -> None:
    """Create application runtime directories."""
    for directory in (TEMP_DIR, OUTPUTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """Return a Windows-safe filename while preserving the extension."""
    name = Path(filename).name.strip()
    if not name:
        return "upload"
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9._ -]", "_", stem).strip(". ")
    safe_stem = re.sub(r"\s+", "_", safe_stem)[:120] or "upload"
    return f"{safe_stem}{suffix}"


def unique_temp_path(original_name: str, suffix: str | None = None) -> Path:
    """Build a UUID-based temporary path."""
    safe = sanitize_filename(original_name)
    extension = suffix if suffix is not None else Path(safe).suffix
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    return TEMP_DIR / f"{uuid.uuid4().hex}{extension}"


def human_file_size(size: int) -> str:
    """Return human-readable file size."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def read_prompt(name: str) -> str:
    """Read a prompt file from the prompts directory."""
    path = PROMPTS_DIR / name
    with path.open("r", encoding="utf-8") as file:
        return file.read().strip()


def safe_unlink(path: Path | None) -> None:
    """Delete a file if it exists, ignoring cleanup failures."""
    if not path:
        return
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        pass
