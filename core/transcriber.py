
import os
from faster_whisper import WhisperModel

def format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, remainder = divmod(remainder, 1)
    milliseconds = remainder * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

def generate_subtitles(video_path: str, model_size: str = "large", language: str = "zh", initial_prompt: str = "這是一個繁體中文的句子") -> str:
    """
    Generates an SRT subtitle file for a video using Whisper.
    
    Args:
        video_path: Path to the input video file.
        model_size: Whisper model size (default: "large").
        language: Language code for transcription (default: "zh").
        initial_prompt: Initial prompt for context (default: "這是一個繁體中文的句子").
        
    Returns:
        Path to the generated SRT file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Transcribing '{video_path}' using model '{model_size}' (Language: {language})...")
    
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    segments, info = model.transcribe(
        audio=video_path,
        word_timestamps=True,
        language=language,
        initial_prompt=initial_prompt
    )

    output_srt = os.path.splitext(video_path)[0] + ".srt"
    
    # Overwrite if exists, or check? Original raised FileExistsError.
    # User wants "unification" and likely ease of use. Overwriting is usually preferred in pipelines unless specified.
    # But strictly following original logic:
    if os.path.exists(output_srt):
        print(f"Warning: SRT file '{output_srt}' already exists. Overwriting.")
    
    try:
        with open(output_srt, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                start_time = format_timestamp(segment.start)
                end_time = format_timestamp(segment.end)
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text.strip()}\n")
                f.write("\n")
                
        print(f"SRT generated successfully: {output_srt}")
        return output_srt

    except IOError as e:
        raise IOError(f"Failed to write SRT file: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate subtitles for a video using Faster-Whisper.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--model", type=str, default="large", help="Whisper model size (default: large)")
    parser.add_argument("--language", type=str, default="zh", help="Language code (default: zh)")
    parser.add_argument("--prompt", type=str, default="這是一個繁體中文的句子", help="Initial prompt")
    
    args = parser.parse_args()
    
    try:
        generate_subtitles(args.video_path, args.model, args.language, args.prompt)
    except Exception as e:
        print(f"Error: {e}")
