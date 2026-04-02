# Transcriber

A command-line tool for transcribing audio files to text using OpenAI Whisper, with optional speaker diarization via pyannote.audio.

## Supported Formats

- mp3, mp4, mpeg, mpga, m4a, wav, webm

## Installation

```bash
pip install pipenv
pipenv install
```

## Usage

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

**Option 1 — `.env` file** (recommended for local use):
```
HF_TOKEN=hf_xxxxxxxxxxxx
```

**Option 2 — environment variable:**
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
pipenv run python transcriber.py recording.m4a -d
```

**Option 3 — CLI flag:**
```bash
pipenv run python transcriber.py recording.m4a -d --hf-token hf_xxxxxxxxxxxx
```

## Requirements

- Python 3.9+
- OpenAI Whisper
- pyannote.audio 3.1+ (for diarization)
