"""Streamlit application for OLLAMA LOCAL TRANSCRIBER."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from services import export_service, media_service, ollama_service, transcription_service
from services.model_service import choose_default_ollama_model, safe_compute_type
from utils.file_utils import ensure_directories, human_file_size, safe_unlink, sanitize_filename, unique_temp_path
from utils.text_utils import build_srt, format_segments_with_timestamps
from utils.validators import DEFAULT_MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS, validate_upload


APP_NAME = "Ollama Local Transcriber"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
LANGUAGE_OPTIONS = {
    "Auto Detect": None,
    "English": "en",
    "Filipino": "tl",
    "Cebuano": "ceb",
    "Spanish": "es",
}


def setup_logging() -> None:
    """Configure file logging."""
    ensure_directories()
    logging.basicConfig(
        filename=Path("logs") / "app.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def init_state() -> None:
    """Initialize stable Streamlit session keys."""
    defaults: dict[str, Any] = {
        "uploaded_file_path": None,
        "uploaded_filename": "",
        "uploaded_file_size": 0,
        "raw_transcript": "",
        "cleaned_transcript": "",
        "summary": "",
        "minutes": "",
        "action_items": "",
        "metadata": {},
        "segments": [],
        "processing_log": [],
        "whisper_settings": {},
        "ollama_model": "qwen3:4b",
        "cancel_requested": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def log_message(message: str) -> None:
    """Append a timestamped UI processing log entry."""
    stamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_log.append(f"[{stamp}] {message}")
    logging.info(message)


def show_error(message: str, exc: Exception | None = None) -> None:
    """Show a user-safe error and log technical details."""
    st.error(message)
    if exc:
        logging.exception("%s: %s", message, exc)
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {str(exc)[:1000]}")


def clear_session() -> None:
    """Clear generated outputs and uploaded temp file."""
    path = st.session_state.get("uploaded_file_path")
    if path:
        safe_unlink(Path(path))
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def save_uploaded_file(uploaded_file) -> Path:
    """Persist Streamlit upload to a UUID temp file."""
    safe_name = sanitize_filename(uploaded_file.name)
    output_path = unique_temp_path(safe_name)
    with output_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    return output_path


def collect_export_data() -> dict[str, Any]:
    """Collect current session output for document export."""
    return {
        "source_filename": st.session_state.uploaded_filename,
        "date_processed": st.session_state.metadata.get("date_processed"),
        "whisper_settings": st.session_state.whisper_settings,
        "ollama_model": st.session_state.ollama_model,
        "detected_language": st.session_state.metadata.get("language"),
        "raw_transcript": st.session_state.raw_transcript,
        "cleaned_transcript": st.session_state.cleaned_transcript,
        "summary": st.session_state.summary,
        "minutes": st.session_state.minutes,
        "action_items": st.session_state.action_items,
        "segments": st.session_state.segments,
        "metadata": st.session_state.metadata,
    }


def render_header(ollama_ok: bool) -> None:
    """Render application header."""
    status = "Available" if ollama_ok else "Unavailable"
    status_color = "#166534" if ollama_ok else "#991b1b"
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <h1>{APP_NAME}</h1>
                <p>Private Offline Audio and Video Transcription</p>
            </div>
            <div class="badges">
                <span class="badge private">Offline and private</span>
                <span class="badge" style="border-color:{status_color};color:{status_color};">Ollama: {status}</span>
                <span class="badge">Whisper: Local</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_styles() -> None:
    """Apply a clean government-office-friendly style."""
    st.markdown(
        """
        <style>
        .app-header {display:flex;justify-content:space-between;align-items:center;padding:1rem 1.2rem;border:1px solid #d7dee8;border-radius:12px;background:#f8fafc;margin-bottom:1rem;}
        .app-header h1 {margin:0;color:#0f2a44;font-size:2rem;}
        .app-header p {margin:.2rem 0 0;color:#475569;}
        .badges {display:flex;gap:.5rem;flex-wrap:wrap;justify-content:flex-end;}
        .badge {border:1px solid #64748b;border-radius:999px;padding:.35rem .65rem;background:#fff;font-size:.85rem;color:#334155;}
        .private {border-color:#0f766e;color:#0f766e;background:#ecfdf5;}
        .metric-card {border:1px solid #d7dee8;border-radius:10px;padding:.8rem;background:#fff;}
        .small-note {color:#64748b;font-size:.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict[str, Any]:
    """Render sidebar settings and return selected values."""
    st.sidebar.header("Whisper Settings")
    whisper_model = st.sidebar.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large-v3", "turbo"], index=2)
    device = st.sidebar.selectbox("Device", ["Auto", "CPU", "CUDA"], index=0)
    resolved_for_compute = "cuda" if device == "CUDA" else "cpu"
    default_compute = safe_compute_type(resolved_for_compute)
    compute_options = ["int8", "int8_float16", "float16", "float32"]
    compute_type = st.sidebar.selectbox("Compute type", compute_options, index=compute_options.index(default_compute))
    language_label = st.sidebar.selectbox("Language", list(LANGUAGE_OPTIONS.keys()), index=0)
    beam_size = st.sidebar.slider("Beam size", 1, 10, 5)
    vad_filter = st.sidebar.checkbox("Enable VAD filter", value=True)
    include_timestamps = st.sidebar.checkbox("Include timestamps", value=True)
    word_timestamps = st.sidebar.checkbox("Word-level timestamps", value=False)
    translate = st.sidebar.checkbox("Translate to English", value=False)

    st.sidebar.divider()
    st.sidebar.header("Ollama Settings")
    host = st.sidebar.text_input("Ollama host", value=DEFAULT_OLLAMA_HOST)
    with st.sidebar.status("Checking Ollama connection", expanded=False) as status:
        ollama_ok, ollama_message = ollama_service.check_connection(host)
        status.update(label=ollama_message, state="complete" if ollama_ok else "error")
    if st.sidebar.button("Test Ollama Connection"):
        ok, message = ollama_service.check_connection(host)
        st.sidebar.success(message) if ok else st.sidebar.warning(message)

    models = ollama_service.list_models(host) if ollama_ok else []
    default_model = choose_default_ollama_model(models, "qwen3:4b")
    model_options = models or [default_model]
    model_index = model_options.index(default_model) if default_model in model_options else 0
    ollama_model = st.sidebar.selectbox("Installed Ollama model", model_options, index=model_index, disabled=not ollama_ok)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.sidebar.number_input("Maximum output tokens", min_value=512, max_value=32768, value=4096, step=512)

    return {
        "whisper_model": whisper_model,
        "device": device,
        "compute_type": compute_type,
        "language": LANGUAGE_OPTIONS[language_label],
        "language_label": language_label,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "include_timestamps": include_timestamps,
        "word_timestamps": word_timestamps,
        "translate": translate,
        "ollama_host": host,
        "ollama_ok": ollama_ok,
        "ollama_model": ollama_model,
        "temperature": temperature,
        "max_tokens": int(max_tokens),
    }


def render_file_info() -> None:
    """Render uploaded file and transcription metadata cards."""
    metadata = st.session_state.metadata
    cols = st.columns(5)
    values = [
        ("Filename", st.session_state.uploaded_filename or "No file"),
        ("File size", human_file_size(st.session_state.uploaded_file_size) if st.session_state.uploaded_file_size else "Not available"),
        ("Duration", f"{metadata.get('duration', 0):.1f} sec" if metadata else "Not available"),
        ("Detected language", metadata.get("language", "Not available") if metadata else "Not available"),
        ("Segments", str(len(st.session_state.segments)) if st.session_state.segments else "0"),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label, value)
    if metadata:
        st.caption(f"Processing time: {metadata.get('processing_time', 0):.1f} seconds")


def run_transcription(settings: dict[str, Any]) -> None:
    """Prepare media and run Faster-Whisper transcription."""
    source = Path(st.session_state.uploaded_file_path)
    temp_files: list[Path] = []
    progress = st.progress(0, text="Preparing media")
    try:
        log_message("Preparing media")
        audio_path, generated = media_service.prepare_audio_for_transcription(source, st.session_state.uploaded_filename)
        temp_files.extend(generated)
        progress.progress(25, text="Loading Whisper model")
        log_message("Loading Whisper model")
        if settings["whisper_model"] in {"medium", "large-v3", "turbo"}:
            st.warning("Large Whisper models may download on first use and require more RAM, VRAM, disk space, and time.")
        progress.progress(45, text="Transcribing audio")
        log_message("Transcribing audio")
        result = transcription_service.transcribe_audio(
            audio_path=audio_path,
            model_name=settings["whisper_model"],
            selected_device=settings["device"],
            compute_type=settings["compute_type"],
            language=settings["language"],
            beam_size=settings["beam_size"],
            vad_filter=settings["vad_filter"],
            word_timestamps=settings["word_timestamps"],
            translate=settings["translate"],
        )
        progress.progress(80, text="Detecting language")
        text = format_segments_with_timestamps(result["segments"]) if settings["include_timestamps"] else result["text"]
        st.session_state.raw_transcript = text
        st.session_state.cleaned_transcript = ""
        st.session_state.summary = ""
        st.session_state.minutes = ""
        st.session_state.action_items = ""
        st.session_state.segments = result["segments"]
        st.session_state.metadata = {
            "language": result.get("language"),
            "language_probability": result.get("language_probability"),
            "duration": result.get("duration"),
            "processing_time": result.get("processing_time"),
            "device": result.get("device"),
            "compute_type": result.get("compute_type"),
            "whisper_model": result.get("model"),
            "date_processed": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.whisper_settings = {k: settings[k] for k in (
            "whisper_model", "device", "compute_type", "language_label", "beam_size", "vad_filter", "include_timestamps", "word_timestamps", "translate"
        )}
        progress.progress(100, text="Complete")
        log_message("Transcription complete")
        st.success("Transcription complete.")
    except Exception as exc:
        show_error("Transcription failed. Check the uploaded media, FFmpeg, Whisper model, device, and compute settings.", exc)
    finally:
        for temp_file in temp_files:
            safe_unlink(temp_file)


def ai_progress_callback(progress_bar):
    """Build progress callback for Ollama processing."""
    def callback(value: float, message: str) -> None:
        progress_bar.progress(min(100, max(0, int(value * 100))), text=message)
        log_message(message)

    return callback


def run_ai_task(task_name: str, settings: dict[str, Any]) -> None:
    """Run one Ollama transcript-processing task."""
    transcript = st.session_state.cleaned_transcript or st.session_state.raw_transcript
    if not transcript.strip():
        st.warning("No transcript is available yet.")
        return
    if not settings["ollama_ok"]:
        st.warning("Ollama is unavailable. Raw transcription is still available.")
        return

    progress = st.progress(0, text="Processing transcript")
    try:
        kwargs = {
            "model": settings["ollama_model"],
            "host": settings["ollama_host"],
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
            "progress_callback": ai_progress_callback(progress),
        }
        st.session_state.ollama_model = settings["ollama_model"]
        if task_name == "clean":
            st.session_state.cleaned_transcript = ollama_service.clean_transcript(st.session_state.raw_transcript, **kwargs)
        elif task_name == "summary":
            st.session_state.summary = ollama_service.generate_summary(transcript, **kwargs)
        elif task_name == "minutes":
            st.session_state.minutes = ollama_service.generate_meeting_minutes(transcript, **kwargs)
        elif task_name == "actions":
            st.session_state.action_items = ollama_service.generate_action_items(transcript, **kwargs)
        log_message(f"AI task complete: {task_name}")
        st.success("AI processing complete.")
    except Exception as exc:
        show_error("Ollama processing failed. Check that Ollama is running and the selected model is installed.", exc)


def render_tabs() -> None:
    """Render editable result tabs."""
    tabs = st.tabs([
        "Raw Transcript",
        "Cleaned Transcript",
        "Summary",
        "Meeting Minutes",
        "Action Items",
        "Subtitles",
        "Processing Log",
    ])
    with tabs[0]:
        st.warning("Editing the raw transcript will affect later AI-generated outputs.")
        st.session_state.raw_transcript = st.text_area("Raw transcript editor", st.session_state.raw_transcript, height=420)
    with tabs[1]:
        st.session_state.cleaned_transcript = st.text_area("Cleaned transcript editor", st.session_state.cleaned_transcript, height=420)
    with tabs[2]:
        st.session_state.summary = st.text_area("Summary editor", st.session_state.summary, height=420)
    with tabs[3]:
        st.session_state.minutes = st.text_area("Meeting minutes editor", st.session_state.minutes, height=420)
    with tabs[4]:
        st.session_state.action_items = st.text_area("Action items editor", st.session_state.action_items, height=420)
    with tabs[5]:
        st.text_area("SRT preview", build_srt(st.session_state.segments), height=420)
        st.caption("Automatic speaker diarization is not included in this version. You may add speaker names manually in transcript text if needed.")
        st.text_input("Optional manual speaker name", placeholder="Example: Chairperson, Secretary, Participant 1")
    with tabs[6]:
        st.text_area("Processing log", "\n".join(st.session_state.processing_log), height=420)


def render_exports() -> None:
    """Render download buttons."""
    data = collect_export_data()
    st.subheader("Export Options")
    cols = st.columns(5)
    try:
        cols[0].download_button("Download TXT", export_service.export_txt(data), file_name="transcription_result.txt", mime="text/plain")
        cols[1].download_button("Download Word", export_service.export_docx(data), file_name="transcription_result.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        cols[2].download_button("Download PDF", export_service.export_pdf(data), file_name="transcription_result.pdf", mime="application/pdf")
        cols[3].download_button("Download JSON", export_service.export_json(data), file_name="transcription_result.json", mime="application/json")
        cols[4].download_button("Download SRT", export_service.export_srt(st.session_state.segments), file_name="transcription_result.srt", mime="application/x-subrip")
    except Exception as exc:
        show_error("Export failed. Check generated text and installed export packages.", exc)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title=APP_NAME, page_icon="", layout="wide")
    setup_logging()
    init_state()
    render_styles()
    settings = render_sidebar()
    render_header(settings["ollama_ok"])

    if not settings["ollama_ok"]:
        st.warning("Ollama is unavailable. Raw transcription is still available.")

    st.subheader("Upload Audio or Video")
    uploaded = st.file_uploader(
        "Choose a supported audio or video file",
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
    )
    if uploaded is not None:
        try:
            validate_upload(uploaded.name, uploaded.size, DEFAULT_MAX_UPLOAD_BYTES)
            if uploaded.name != st.session_state.uploaded_filename or uploaded.size != st.session_state.uploaded_file_size:
                old_path = st.session_state.get("uploaded_file_path")
                if old_path:
                    safe_unlink(Path(old_path))
                path = save_uploaded_file(uploaded)
                st.session_state.uploaded_file_path = str(path)
                st.session_state.uploaded_filename = sanitize_filename(uploaded.name)
                st.session_state.uploaded_file_size = uploaded.size
                log_message(f"Uploaded file saved: {st.session_state.uploaded_filename}")
        except Exception as exc:
            show_error("Upload validation failed.", exc)

    render_file_info()

    st.subheader("Transcription Settings")
    st.caption("Faster-Whisper downloads the selected model during first use. Large models may require substantial RAM, VRAM, disk space, and time.")
    col_a, col_b = st.columns([1, 1])
    start_clicked = col_a.button("Start Transcription", type="primary", disabled=not st.session_state.uploaded_file_path)
    if col_b.button("Stop or Cancel"):
        st.session_state.cancel_requested = True
        st.warning("Cancel requested. If a model is currently transcribing, it may stop after the current operation completes.")
    if start_clicked:
        st.session_state.cancel_requested = False
        run_transcription(settings)

    st.subheader("Ollama Processing")
    ai_disabled = not bool(st.session_state.raw_transcript.strip()) or not settings["ollama_ok"]
    ai_cols = st.columns(5)
    if ai_cols[0].button("Clean Transcript with Ollama", disabled=ai_disabled):
        run_ai_task("clean", settings)
    if ai_cols[1].button("Generate Summary", disabled=ai_disabled):
        run_ai_task("summary", settings)
    if ai_cols[2].button("Generate Meeting Minutes", disabled=ai_disabled):
        run_ai_task("minutes", settings)
    if ai_cols[3].button("Extract Action Items", disabled=ai_disabled):
        run_ai_task("actions", settings)
    if ai_cols[4].button("Run All AI Processing", disabled=ai_disabled):
        run_ai_task("clean", settings)
        run_ai_task("summary", settings)
        run_ai_task("minutes", settings)
        run_ai_task("actions", settings)

    render_tabs()
    render_exports()

    if st.button("Clear Session"):
        clear_session()
        st.rerun()


if __name__ == "__main__":
    main()
