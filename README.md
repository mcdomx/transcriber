# Transcriber

A command-line tool for transcribing audio files to text using OpenAI Whisper.

## Supported Formats

- mp3, mp4, mpeg, mpga, m4a, wav, webm

## Installation

```bash
pip install pipenv
pipenv install
```

## Usage

```bash
pipenv run python transcriber.py <audio_file> [-o OUTPUT_DIR] [-q QUALITY]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `file_path` | Path to the audio file (required) |
| `-o, --output_dir` | Output directory for transcription (default: same as input) |
| `-q, --convert_quality` | Model quality 1-5 (1=tiny, 2=base, 3=small, 4=medium, 5=large) |

### Examples

```bash
# Basic transcription
pipenv run python transcriber.py recording.m4a

# Specify output directory
pipenv run python transcriber.py recording.m4a -o ./transcripts

# Use higher quality model
pipenv run python transcriber.py recording.m4a -q 4
```

## Output

Transcriptions are saved as `<original_filename>-transcription.txt` in the output directory.

## Requirements

- Python 3.9+
- OpenAI Whisper
