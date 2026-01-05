
import os
import subprocess
from typing import List, Tuple
from faster_whisper import WhisperModel

def remove_video_silence(input_file: str, gap_threshold: float = 1.0) -> str:
    """
    Removes silent segments from a video file using ffmpeg and Whisper for silence detection.
    
    Args:
        input_file: Path to the input video file.
        gap_threshold: Minimum duration of silence (in seconds) to be removed.
        
    Returns:
        Path to the processed video file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Video file not found: {input_file}")

    print(f"Detecting silence in '{input_file}' using faster-whisper...")
    
    # Use 'base' model for fast silence detection as per original implementation
    model = WhisperModel("base", device="cuda", compute_type="float16")
    
    segments, info = model.transcribe(
        audio=input_file, 
        word_timestamps=True, 
        language="zh" # Default to Chinese as per context
    )

    video_duration = info.duration
    print(f"Video duration: {video_duration:.2f}s")

    # Identify segments to DELETE (silence)
    segments_to_delete: List[Tuple[float, float]] = []
    previous_end = 0.0

    # Check for silence at the beginning
    first_segment = next(segments, None)
    if not first_segment:
        print("No speech detected.")
        return input_file

    if first_segment.start > gap_threshold:
        segments_to_delete.append((0.0, first_segment.start))
    
    previous_end = first_segment.end

    # Check for silence between segments
    for segment in segments:
        if segment.start - previous_end > gap_threshold:
            segments_to_delete.append((previous_end, segment.start))
        previous_end = segment.end

    # Check for silence at the end
    if video_duration - previous_end > gap_threshold:
        segments_to_delete.append((previous_end, video_duration))

    if not segments_to_delete:
        print(f"No silence gaps longer than {gap_threshold}s found.")
        return input_file

    # Calculate segments to KEEP
    segments_to_keep: List[Tuple[float, float]] = []
    current_keep_start = 0.0
    half_gap = gap_threshold / 2

    # Logic: Preserve 'half_gap' of silence around speech to avoid abrupt cuts
    # The original logic iterates through deletion zones to define keep zones.
    
    # Re-evaluating the keep logic based on original 'delete_blank.py':
    # previous_gap_end = silence_end - half_gap
    # keep_start = previous_gap_end (from last iteration)
    # keep_end = current_silence_start + half_gap
    
    previous_keep_end_inc_buffer = 0.0 # Effectively 0.0 start for the first block

    for silence_start, silence_end in segments_to_delete:
        # End of the speech block is start of silence + buffer
        keep_end = silence_start + half_gap
        
        # Start of this speech block was determined in previous iteration
        keep_start = previous_keep_end_inc_buffer

        if keep_end > keep_start:
            segments_to_keep.append((keep_start, keep_end))
        
        # Determine start of NEXT speech block (end of this silence - buffer)
        previous_keep_end_inc_buffer = silence_end - half_gap

    # Add the final segment if applicable
    if previous_keep_end_inc_buffer < video_duration:
        segments_to_keep.append((previous_keep_end_inc_buffer, video_duration))

    if not segments_to_keep:
        print("Warning: No video segments remain after trimming.")
        return input_file

    # Build ffmpeg command
    print("Building ffmpeg command...")
    filter_parts = []
    concat_inputs = []
    
    for i, (start, end) in enumerate(segments_to_keep):
        filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
        concat_inputs.append(f"[v{i}][a{i}]")

    filter_complex = ";".join(filter_parts) + ";" + \
                     "".join(concat_inputs) + \
                     f"concat=n={len(segments_to_keep)}:v=1:a=1[outv][outa]"

    output_file = os.path.splitext(input_file)[0] + "_trimmed.mp4"

    command = [
        'ffmpeg',
        '-i', input_file,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',   # Switched to stable software encoding
        '-preset', 'medium',
        '-c:a', 'aac',
        '-y',
        output_file
    ]

    print(f"Executing ffmpeg processing -> {output_file}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e.stderr}")
        raise RuntimeError("FFmpeg processing failed") from e

    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove silence from video files.")
    parser.add_argument("input_file", type=str, help="Path to the input video file")
    parser.add_argument("--gap", type=float, default=1.0, help="Minimum silence duration to remove (seconds)")
    
    args = parser.parse_args()
    
    try:
        remove_video_silence(args.input_file, args.gap)
    except Exception as e:
        print(f"Error: {e}")
