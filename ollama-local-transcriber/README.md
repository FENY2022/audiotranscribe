# Ollama Local Transcriber

Private Offline Audio and Video Transcription for Windows desktop use.

This project is a local Streamlit web application that uses Faster-Whisper for speech-to-text transcription and a locally installed Ollama model for transcript cleanup, summaries, meeting minutes, and action items. It does not use OpenAI, Google, Gemini, or any paid online transcription API.

## Features

- Local audio and video transcription with Faster-Whisper.
- Transcription language choices limited to Auto Detect, English, Tagalog, and Cebuano.
- Local AI cleanup and summarization through Ollama.
- Supported audio: MP3, WAV, M4A, AAC, FLAC, OGG.
- Supported video: MP4, MKV, MOV, AVI, WEBM.
- FFmpeg conversion to mono 16 kHz WAV for reliable transcription.
- Raw transcript, cleaned transcript, summary, meeting minutes, and action item editors.
- TXT, DOCX, PDF, JSON, and SRT downloads.
- Timestamped transcript segments and SRT subtitle export.
- Long transcript chunking for Ollama processing.
- No cloud upload of audio, video, transcript, or user data.

## Requirements

- Windows 10 or Windows 11.
- Python 3.11 or newer.
- FFmpeg installed separately and available on PATH.
- Ollama installed locally for AI cleanup and summaries.
- At least one Ollama text model. Recommended default: `qwen3:4b`.

FFmpeg is an external dependency. It is not installed by `pip` and must be installed separately.

## Install Python

1. Download Python from `https://www.python.org/downloads/`.
2. During installation, enable `Add python.exe to PATH`.
3. Verify in Command Prompt:

```bat
python --version
```

## Install FFmpeg

1. Download a Windows FFmpeg build from `https://ffmpeg.org/download.html`.
2. Extract it to a folder such as `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your Windows PATH.
4. Verify:

```bat
ffmpeg -version
```

If FFmpeg is missing, video audio extraction and media conversion will fail.

## Install Ollama

1. Download Ollama for Windows from `https://ollama.com/download/windows`.
2. Install and start Ollama.
3. Verify:

```bat
ollama list
```

Install the recommended model if needed:

```bat
ollama pull qwen3:4b
```

The target computer already has `qwen3:4b`, so it is used as the default model.

## Install The Application

Open Command Prompt in the project folder and run:

```bat
install_windows.bat
```

The installer checks Python, creates `venv`, installs Python dependencies, checks Ollama, checks FFmpeg, and creates runtime folders.

Manual setup alternative:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run The Application

From the project folder:

```bat
run_app.bat
```

Or manually:

```bat
venv\Scripts\activate
streamlit run app.py
```

Streamlit will open the local app in your browser.

## How To Use

1. Upload a supported audio or video file.
2. Choose Whisper settings in the sidebar. Auto Detect is intended for English, Tagalog, and Cebuano only; if another language is detected, the app shows a warning but keeps the transcript.
3. Click `Start Transcription`.
4. Review or edit the raw transcript.
5. If Ollama is available, run cleanup, summary, meeting minutes, action items, or all AI processing.
6. Edit generated outputs if needed.
7. Download TXT, Word, PDF, JSON, or SRT.

## Whisper Model Choices

Faster-Whisper downloads the selected Whisper model during first use. The app warns before larger models are used. Exact download sizes can change, so they are not hardcoded.

- `tiny`: fastest, lower accuracy.
- `base`: fast.
- `small`: recommended for normal computers.
- `medium`: improved accuracy, requires more resources.
- `large-v3`: highest accuracy, heavy.
- `turbo`: faster large-model option when supported.

## Recommended CPU Settings

- Whisper model: `small`
- Device: `CPU`
- Compute type: `int8`
- Transcription speed: `Fastest` (beam size 1, greedy decoding)
- VAD: Enabled
- Recommended Ollama model: `qwen3:4b`

Fastest mode is the default on CPU and is roughly 3-5x faster than beam size 5. For the quickest results use the `tiny` or `base` model; for better accuracy use `small` or `medium` at a slower speed mode.

## Recommended NVIDIA GPU Settings

- Whisper model: `medium` or `large-v3`
- Device: `CUDA`
- Compute type: `float16`
- Transcription speed: `Accurate` (beam size 5)
- VAD: Enabled

Do not assume CUDA is installed. If CUDA is unavailable, use CPU.

## Privacy

The application is designed for local processing only. Uploaded files are saved temporarily in the local `temp` folder, processed locally, and converted locally using FFmpeg. Faster-Whisper runs locally. Ollama runs locally through `http://localhost:11434` by default.

No audio, video, transcript, or user data is uploaded to an online service by this application.

After required models and dependencies have been downloaded, transcription and AI processing can operate locally.

## Speaker Identification Limitation

This first version does not include automatic speaker diarization. It uses timestamped transcript segments only. You may manually add speaker names in the transcript editors.

## Folder Structure

```text
ollama-local-transcriber/
├── app.py
├── requirements.txt
├── README.md
├── install_windows.bat
├── run_app.bat
├── services/
├── utils/
├── prompts/
├── temp/
├── outputs/
└── logs/
```

## Common Errors

### Python was not found

Install Python 3.11 or newer and enable `Add python.exe to PATH`.

### FFmpeg is missing

Install FFmpeg and add its `bin` folder to PATH. Verify with:

```bat
ffmpeg -version
```

### Ollama is unavailable

Start Ollama and verify:

```bat
ollama list
```

Raw transcription still works when Ollama is unavailable. Only AI cleanup and summaries are disabled.

### Selected Ollama model unavailable

Install the model:

```bat
ollama pull qwen3:4b
```

### CUDA unavailable

Use CPU settings, or install compatible NVIDIA drivers and CUDA-enabled PyTorch support.

### Whisper model download failure

Check internet access for first-time model download, disk space, and permissions. Try a smaller model such as `small` or `base`.

### Insufficient RAM or VRAM

Use a smaller Whisper model, CPU `int8`, lower beam size, or shorter media files.

### Corrupted media or unsupported codec

Try converting the file manually with FFmpeg or export a standard MP3/WAV/MP4 from the source application.

## Exact Windows Commands

From `C:\xampp\htdocs\audiotranscribe\ollama-local-transcriber`:

```bat
install_windows.bat
run_app.bat
```
