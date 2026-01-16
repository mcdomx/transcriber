import os
from pathlib import Path
import argparse
import whisper

def transcribe_mp3(file_path, output_dir=None, convert_quality:int=None):
    """
    Transcribe an audio file file to text using OpenAI Whisper.
    Supported file types: mp3, mp4, mpeg, mpga, m4a, wav, and webm
    
    Args:
        file_path (str): Path to the audio file to transcribe
        output_dir (str, optional): Directory to save the transcription. 
                                  If None, saves in same directory as source audio file.
        convert_quality (int, optional): 1=tiny, 2=base, 3=small, 4=medium, 5=large 
    
    Returns:
        str: Path to the created transcription file
    
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        Exception: If transcription fails
    """
    
    # Convert to Path object for easier manipulation
    _path = Path(file_path)
    
    # Check if MP3 file exists
    if not _path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check if it's actually an MP3/M4A file
    _supported_types = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
    if _path.suffix.lower() not in _supported_types:
        raise ValueError(f"File must be a supported audio file (mp3, mp4, mpeg, mpga, m4a, wav or webm). Got: {_path.suffix}")
    
    # Determine output directory
    if output_dir is None:
        output_directory = _path.parent
    else:
        output_directory = Path(output_dir)
        # Create output directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)
    
    # Create output filename: original_name-transcription.txt
    base_name = _path.stem  # filename without extension
    output_filename = f"{base_name}-transcription.txt"
    output_path = output_directory / output_filename
    
    try:
        print(f"Loading Whisper model with quality {convert_quality}...", end="")
        # Load the Whisper model (you can change model size: tiny, base, small, medium, large)
        if convert_quality==1:
            model = whisper.load_model("tiny")
        if convert_quality==None or convert_quality==2:
            model = whisper.load_model("base")
        if convert_quality==3:
            model = whisper.load_model("small")
        if convert_quality==4:
            model = whisper.load_model("medium")
        if convert_quality==5:
            model = whisper.load_model("large")
        else:
            model = whisper.load_model("base")
        print("DONE")
        
        print(f"Transcribing {_path.name}...", end="")
        # Transcribe the audio file
        result = model.transcribe(str(_path), fp16=False)
        
        # Extract the transcribed text
        transcription_text = result["text"]
        
        # Save transcription to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text.strip())

        print("DONE")

        print(f"Transcription completed successfully! Quality: {str(convert_quality)}")
        
        return str(output_path)
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Parse transcription arguments')
    parser.add_argument('file_path', help='Input file full path') # positional and required
    parser.add_argument('-o', '--output_dir', help='Output directory. If none, Defaults to input path.') # optional
    parser.add_argument('-q', '--convert_quality', help='Conversion quality (1-5)') # optional
    
    args = parser.parse_args()
    
    try:
        transcription_file = transcribe_mp3(**vars(args))
        print(f"Transcription saved to: {transcription_file}")
    except Exception as e:
        print(f"Error: {e}")


    
if __name__ == "__main__":
    main()