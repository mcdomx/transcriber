import os
import json
from pathlib import Path
import argparse
import whisper
from dotenv import load_dotenv


def load_hf_token(cli_token=None):
    """
    Resolve a HuggingFace API token from (in order):
      1. cli_token argument
      2. HF_TOKEN environment variable
      3. HF_TOKEN in a .env file in the working directory

    Raises:
        EnvironmentError: If no token is found
    """
    if cli_token:
        return cli_token

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    raise EnvironmentError(
        "HuggingFace token required for diarization.\n"
        "  1. Accept model terms at: https://hf.co/pyannote/speaker-diarization-3.1\n"
        "  2. Accept model terms at: https://hf.co/pyannote/segmentation-3.0\n"
        "  3. Set HF_TOKEN as an environment variable, in a .env file, or pass --hf-token"
    )


def run_diarization(audio_path, hf_token):
    """
    Run speaker diarization on an audio file using pyannote.audio.

    Args:
        audio_path (str): Path to the audio file
        hf_token (str): HuggingFace API token

    Returns:
        list[dict]: Speaker turns as [{"start": float, "end": float, "speaker": str}, ...]
    """
    import torch
    from pyannote.audio import Pipeline
    from huggingface_hub import login

    # PyTorch 2.6 changed weights_only default to True, which breaks pyannote model loading.
    # Patch torch.load to force weights_only=False for the duration of pipeline initialization.
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    try:
        login(token=hf_token, add_to_git_credential=False)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    finally:
        torch.load = _original_torch_load

    # pyannote cannot read m4a/mp4 directly — convert to wav via ffmpeg first
    import tempfile
    import subprocess
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() not in ('.wav',):
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(audio_path), '-ar', '16000', '-ac', '1', tmp.name],
            check=True, capture_output=True
        )
        diarize_path = tmp.name
    else:
        diarize_path = str(audio_path)
        tmp = None

    try:
        diarization = pipeline(diarize_path)
    finally:
        if tmp is not None:
            os.unlink(tmp.name)

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    return sorted(turns, key=lambda t: t["start"])


def align_speakers(whisper_segments, diarization_turns):
    """
    Assign a speaker label to each Whisper segment by maximum time overlap
    with pyannote diarization turns.

    Args:
        whisper_segments (list[dict]): Whisper result segments with "start", "end", "text"
        diarization_turns (list[dict]): Diarization turns with "start", "end", "speaker"

    Returns:
        list[dict]: Segments with "start", "end", "text", "speaker" fields
    """
    aligned = []
    for seg in whisper_segments:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn in diarization_turns:
            overlap = max(0.0, min(seg["end"], turn["end"]) - max(seg["start"], turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        aligned.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": best_speaker,
            "text": seg["text"].strip(),
        })
    return aligned


def format_diarized_output(aligned_segments, audio_filename, whisper_model_name):
    """
    Format aligned segments into a readable text transcript and a structured JSON string.
    Consecutive segments from the same speaker are collapsed into a single block.

    Args:
        aligned_segments (list[dict]): Output of align_speakers()
        audio_filename (str): Original audio filename for JSON metadata
        whisper_model_name (str): Whisper model name for JSON metadata

    Returns:
        tuple[str, str]: (formatted_text, json_string)
    """
    # Collapse consecutive same-speaker segments
    collapsed = []
    for seg in aligned_segments:
        if collapsed and collapsed[-1]["speaker"] == seg["speaker"]:
            collapsed[-1]["end"] = seg["end"]
            collapsed[-1]["text"] += " " + seg["text"]
        else:
            collapsed.append(dict(seg))

    # Build formatted text
    lines = []
    for block in collapsed:
        seconds = int(block["start"])
        timestamp = f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
        lines.append(f"[{block['speaker']} | {timestamp}] {block['text']}")
    formatted_text = "\n".join(lines)

    # Build JSON
    json_data = {
        "audio_file": audio_filename,
        "whisper_model": whisper_model_name,
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "segments": collapsed,
    }
    json_string = json.dumps(json_data, indent=2)

    return formatted_text, json_string


def transcribe_mp3(file_path, output_dir=None, convert_quality=None, diarize=False, hf_token=None, progress_callback=None, save_txt=True, save_json=True, original_filename=None):
    """
    Transcribe an audio file to text using OpenAI Whisper, with optional speaker diarization.
    Supported file types: mp3, mp4, mpeg, mpga, m4a, wav, and webm

    Args:
        file_path (str): Path to the audio file to transcribe
        output_dir (str, optional): Directory to save the transcription.
                                    If None, saves in same directory as source audio file.
        convert_quality (int, optional): 1=tiny, 2=base, 3=small, 4=medium, 5=large
        diarize (bool): If True, run speaker diarization and include speaker labels in output.
        hf_token (str, optional): HuggingFace API token for diarization. Falls back to
                                  HF_TOKEN env var or .env file.
        save_txt (bool): If True, write the .txt transcript file.
        save_json (bool): If True, write the .json output file.
        original_filename (str, optional): Name to derive the output filename from, used when
                                           file_path points to a temp file (e.g. an upload) whose
                                           name shouldn't appear in the output filename.

    Returns:
        tuple[str, str | None, str | None]: (text_content, txt_path, json_path)
            txt_path and json_path are None when the respective file was not saved.

    Raises:
        FileNotFoundError: If the audio file doesn't exist
        Exception: If transcription fails
    """

    _path = Path(file_path)

    if not _path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    _supported_types = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
    if _path.suffix.lower() not in _supported_types:
        raise ValueError(
            f"File must be a supported audio file (mp3, mp4, mpeg, mpga, m4a, wav or webm). "
            f"Got: {_path.suffix}"
        )

    if output_dir is None:
        output_directory = _path.parent
    else:
        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

    base_name = Path(original_filename).stem if original_filename else _path.stem
    output_path = output_directory / f"{base_name}-transcription.txt"

    def _emit(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)
        else:
            print(msg, flush=True)

    try:
        # Resolve whisper model name
        quality_map = {1: "tiny", 2: "base", 3: "small", 4: "medium", 5: "large"}
        model_name = quality_map.get(convert_quality, "base")

        _emit(f"Loading Whisper model '{model_name}'...", 5)
        model = whisper.load_model(model_name)
        _emit(f"Whisper model '{model_name}' loaded.", 20)

        _emit(f"Transcribing {_path.name}...", 25)
        result = model.transcribe(str(_path), fp16=False)
        _emit("Transcription complete.", 70)

        if diarize:
            token = load_hf_token(hf_token)

            _emit("Running speaker diarization...", 72)
            turns = run_diarization(str(_path), token)
            _emit("Speaker diarization complete.", 90)

            aligned = align_speakers(result["segments"], turns)
            formatted_text, json_string = format_diarized_output(aligned, _path.name, model_name)

            saved_txt_path = None
            saved_json_path = None

            if save_txt:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_text)
                saved_txt_path = str(output_path)
                _emit(f"Saved: {output_path}", 95)

            if save_json:
                json_path = output_directory / f"{base_name}-transcription.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    f.write(json_string)
                saved_json_path = str(json_path)
                _emit(f"Saved JSON: {json_path}", 96 if save_txt else 95)

            _emit("Done.", 100)
            return formatted_text, saved_txt_path, saved_json_path

        else:
            plain_text = result["text"].strip()
            saved_txt_path = None
            saved_json_path = None

            if save_txt:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(plain_text)
                saved_txt_path = str(output_path)

            if save_json:
                json_data = {
                    "audio_file": _path.name,
                    "whisper_model": model_name,
                    "text": plain_text,
                }
                json_path = output_directory / f"{base_name}-transcription.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(json_data, indent=2))
                saved_json_path = str(json_path)

            _emit("Done.", 100)
            return plain_text, saved_txt_path, saved_json_path

    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}") from e


def main():
    parser = argparse.ArgumentParser(description='Parse transcription arguments')
    parser.add_argument('file_path', help='Input file full path')
    parser.add_argument('-o', '--output_dir', help='Output directory. Defaults to input path.')
    parser.add_argument('-q', '--convert_quality', type=int, help='Conversion quality (1-5)')
    parser.add_argument('-d', '--diarize', action='store_true',
                        help='Enable speaker diarization (requires HuggingFace token)')
    parser.add_argument('--hf-token', dest='hf_token',
                        help='HuggingFace API token (overrides HF_TOKEN env var)')

    args = parser.parse_args()

    try:
        _, txt_path, _ = transcribe_mp3(**vars(args))
        if txt_path:
            print(f"Transcription saved to: {txt_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
