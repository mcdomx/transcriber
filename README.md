# Transcriber

A tool for transcribing audio files to text using OpenAI Whisper, with optional speaker diarization via pyannote.audio. Includes both a command-line interface and a browser-based web UI.

## Supported Formats

- mp3, mp4, mpeg, mpga, m4a, wav, webm

## Installation

```bash
pip install pipenv
pipenv install
```

## Web UI

### Starting the server

**Option 1 — Double-click** `start_server.command` in Finder. A Terminal window will open and the server will start.

**Option 2 — Terminal:**
```bash
pipenv run python app.py
```

Then open **http://127.0.0.1:8000** in your browser.

### Features

- Drag-and-drop (or click-to-browse) audio file upload
- Model quality selector and speaker diarization toggle
- Configurable output directory per job
- Real-time progress bar during transcription
- Transcript displayed in the browser when complete, with saved file paths shown
- Settings panel (gear icon) to set your HuggingFace token and default output directory

### Settings

Click the **Settings** button in the top-right corner to:
- Set or update your HuggingFace API token (required for speaker diarization)
- Set a default output directory for all transcription jobs

Settings are saved to the local `.env` file.

## Command-Line Usage

```bash
pipenv run python transcriber.py <audio_file> [-o OUTPUT_DIR] [-q QUALITY] [-d] [--hf-token TOKEN]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `file_path` | Path to the audio file (required) |
| `-o, --output_dir` | Output directory for transcription (default: same as input) |
| `-q, --convert_quality` | Model quality 1-5 (1=tiny, 2=base, 3=small, 4=medium, 5=large) |
| `-d, --diarize` | Enable speaker diarization (requires HuggingFace token) |
| `--hf-token` | HuggingFace API token (overrides `HF_TOKEN` env var) |

### Examples

```bash
# Basic transcription
pipenv run python transcriber.py recording.m4a

# Specify output directory
pipenv run python transcriber.py recording.m4a -o ./transcripts

# Use higher quality model
pipenv run python transcriber.py recording.m4a -q 4

# Transcribe with speaker diarization
pipenv run python transcriber.py recording.m4a -d

# Diarization with explicit token
pipenv run python transcriber.py recording.m4a -d --hf-token hf_xxxxxxxxxxxx
```

## Output

Without diarization, transcriptions are saved as `<filename>-transcription.txt`.

With diarization (`-d`), two files are written:

- `<filename>-transcription.txt` — readable transcript with speaker labels and timestamps:
  ```
  [SPEAKER_00 | 00:00:03] Well I tried to make it Sunday...
  [SPEAKER_01 | 00:00:08] That's interesting, yeah.
  ```
- `<filename>-transcription.json` — structured data with per-segment speaker, start/end times, and text.

## Speaker Diarization Setup

Diarization uses the `pyannote/speaker-diarization-3.1` model, which requires accepting terms of use on HuggingFace before use.

1. Accept terms at: https://hf.co/pyannote/speaker-diarization-3.1
2. Accept terms at: https://hf.co/pyannote/segmentation-3.0
3. Create a HuggingFace access token at: https://hf.co/settings/tokens

### Providing the token

**Option 1 — Web UI Settings panel** (recommended): click the Settings button after starting the server.

**Option 2 — `.env` file:**
```
HF_TOKEN=hf_xxxxxxxxxxxx
```

**Option 3 — environment variable:**
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
pipenv run python transcriber.py recording.m4a -d
```

**Option 4 — CLI flag:**
```bash
pipenv run python transcriber.py recording.m4a -d --hf-token hf_xxxxxxxxxxxx
```

## Requirements

- Python 3.9+
- OpenAI Whisper
- pyannote.audio 3.1+ (for diarization)
- ffmpeg (required for diarization of non-WAV files — `brew install ffmpeg`)
