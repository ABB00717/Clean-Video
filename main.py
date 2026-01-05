
import os
import sys
import argparse
from multiprocessing import Pool
from core.cleaner import remove_video_silence
from core.transcriber import generate_subtitles
from core.editor import process_subtitles

def process_single_video(video_path: str, gap_threshold: float = 1.0, language: str = "zh", initial_prompt: str = "這是一個繁體中文的句子"):
    """
    Runs the full pipeline on a single video file.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}\n")
    
    try:
        # Step 1: Remove Silence
        print(">>> Step 1: Silence Removal")
        trimmed_video = remove_video_silence(video_path, gap_threshold)
        print(f"✓ Video trimmed: {trimmed_video}\n")
        
        # Step 2: Generate Subtitles
        print(">>> Step 2: Transcription")
        srt_file = generate_subtitles(trimmed_video, language=language, initial_prompt=initial_prompt)
        print(f"✓ SRT generated: {srt_file}\n")
        
        # Step 3: Editor (AI Refinement + Deduplication)
        print(">>> Step 3: Subtitle Refinement")
        final_srt = process_subtitles(srt_file)
        print(f"✓ SRT refined: {final_srt}\n")
        
        # Step 4: Finalize & Rename
        print(">>> Step 4: Finalizing Files")
        
        # Original Video -> _orig.mp4
        orig_backup = os.path.splitext(video_path)[0] + "_orig.mp4"
        if not os.path.exists(orig_backup):
            os.rename(video_path, orig_backup)
            print(f"  Renamed original video to: {os.path.basename(orig_backup)}")
            
        # Trimmed Video -> Original Name
        os.rename(trimmed_video, video_path)
        print(f"  Renamed trimmed video to: {os.path.basename(video_path)}")
        
        # Refined SRT -> Original Name .srt
        target_srt_name = os.path.splitext(video_path)[0] + ".srt"
        if os.path.exists(target_srt_name):
            os.remove(target_srt_name) # Remove existing if present to avoid conflict
        os.rename(final_srt, target_srt_name)
        print(f"  Renamed final SRT to: {os.path.basename(target_srt_name)}")
        
        # Cleanup intermediate SRT (the unrefined one)
        unrefined_srt = os.path.splitext(trimmed_video)[0] + ".srt"
        if os.path.exists(unrefined_srt) and unrefined_srt != final_srt:
             os.remove(unrefined_srt)
             print(f"  Removed intermediate SRT: {os.path.basename(unrefined_srt)}")

        print(f"\n✓ Processing Complete for: {os.path.basename(video_path)}")
        
    except Exception as e:
        print(f"\n✗ Error processing {video_path}: {e}")
        # raise e # Optional: Raise if we want to stop batch processing

def main():
    parser = argparse.ArgumentParser(description="Clean Video Tool: Silence Removal, Transcription, and Refinement.")
    
    parser.add_argument("input", type=str, help="Path to a video file OR a directory containing .mp4 files")
    parser.add_argument("--gap", type=float, default=1.0, help="Minimum silence duration to remove (seconds)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for directory processing")
    parser.add_argument("--language", type=str, default="zh", help="Language code for transcription (default: zh)")
    parser.add_argument("--initial_prompt", type=str, default="這是一個繁體中文的句子", help="Initial prompt for context")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        process_single_video(args.input, args.gap, args.language, args.initial_prompt)
    elif os.path.isdir(args.input):
        video_files = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith('.mp4') and not f.endswith('_orig.mp4')
        ]
        
        if not video_files:
            print(f"No valid .mp4 files found in {args.input}")
            return
            
        print(f"Found {len(video_files)} videos to process.")
        
        if args.workers > 1:
            with Pool(processes=args.workers) as pool:
                pool.starmap(process_single_video, [(v, args.gap, args.language, args.initial_prompt) for v in video_files])
        else:
            for v in video_files:
                process_single_video(v, args.gap, args.language, args.initial_prompt)
    else:
        print(f"Error: Invalid input path: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()
