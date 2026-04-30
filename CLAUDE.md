# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install pipenv
pipenv install

# Run transcription
pipenv run python transcriber.py <audio_file> [-o OUTPUT_DIR] [-q QUALITY] [-d] [--hf-token TOKEN]

# Run with diarization (requires HuggingFace token in .env as HF_TOKEN)
pipenv run python transcriber.py recording.m4a -d

# Quality levels: 1=tiny, 2=base, 3=small, 4=medium, 5=large (default: base)
pipenv run python transcriber.py recording.m4a -q 4
```

No test suite exists. Test manually with files in `test_data/`.

## Architecture

Single-file project (`transcriber.py`) with one public function: `transcribe_mp3()`, which is also the CLI entry point via `main()`.

### Execution flow

1. **Transcription** — Whisper loads model and transcribes audio to segments with `start`/`end`/`text`.
2. **Diarization** (optional, `-d` flag) — `run_diarization()` uses pyannote.audio to identify speaker turns. Non-wav files are converted to mono 16kHz wav via ffmpeg before passing to pyannote (pyannote cannot read m4a/mp4 directly).
3. **Alignment** — `align_speakers()` assigns each Whisper segment a speaker label by maximum time overlap with diarization turns.
4. **Output** — `format_diarized_output()` collapses consecutive same-speaker segments and writes both `.txt` (human-readable) and `.json` (structured).

### Key implementation notes

- **PyTorch 2.6 compatibility**: `run_diarization()` patches `torch.load` to force `weights_only=False` during pyannote pipeline initialization, then restores it. This is required because PyTorch 2.6 changed the `weights_only` default to `True`, which breaks pyannote model loading.
- **HuggingFace token resolution**: `load_hf_token()` checks CLI arg → env var → `.env` file, in that order. Required only for diarization.
- **Diarization model**: `pyannote/speaker-diarization-3.1` — users must accept terms of use on HuggingFace before first use (also `pyannote/segmentation-3.0`).
- **Output naming**: `<stem>-transcription.txt` and (with diarization) `<stem>-transcription.json`, written to the same directory as the input file unless `-o` is specified.
