
import os
import textwrap
import srt
import time
import json
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

class ModifiedSentence(BaseModel):
    output: str
    should_merge_next: bool = False

class GlobalSummary(BaseModel):
    summary: str
    style_guide: str

def load_math_symbols() -> str:
    """Loads math symbols from text file."""
    try:
        with open(os.path.join(os.path.dirname(__file__), "math_symbols.txt"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

MATH_SYMBOLS = load_math_symbols()

SYSTEM_INSTRUCTION_TEMPLATE = """
You are a strict subtitle editor for a mathematics video using Traditional Chinese (繁體中文).

**CRITICAL RULE: SUBTRACTION & MERGING**
1. **Subtraction Only**: Remove filler words, stutters, and redundant phrases.
2. **Merging**: Check if the current line flows grammatically into the NEXT line. 
   - If merging them creates a better sentence AND the combined length is ≤ 30 characters, set `should_merge_next` to true.
   - Example Input: "That I would say" (Line 1), "in this case." (Line 2).
   - If Line 1 + Line 2 ≤ 30 chars, Line 1 Output: "That I would say" (with `should_merge_next=True`).
   - Do NOT manually merge the text in `output`. Just return the refined text of the CURRENT line.
3. **No Rewriting**: Do not change sentence structure or add words unless using Math Notation.

**MATHEMATICAL NOTATION**
Replace spoken math terms with symbols:
{math_symbols}

**FORMATTING**
1. Output valid JSON.
2. Add spaces between English/Numbers vs Chinese.
3. Traditional Chinese only.

**GLOBAL SUMMARY**
{global_summary}
"""

def get_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")
    return genai.Client(api_key=str(api_key))

def upload_to_gemini(path: str, client):
    """Uploads a file to Gemini and waits for it to be processed."""
    print(f"Uploading file: {path}...")
    file = client.files.upload(file=path)
    print(f"  Uploaded: {file.name}")
    
    # Wait for processing if it's a video or other large file
    while file.state.name == "PROCESSING":
        print("  Waiting for file processing...", end="\r")
        time.sleep(2)
        file = client.files.get(name=file.name)
    
    if file.state.name != "ACTIVE":
        raise ValueError(f"File {file.name} failed to process. State: {file.state.name}")
        
    print(f"  File ready: {file.uri}")
    return file

def prepare_gemini_context(srt_path: str, client):
    """
    Uploads video/aux files and generates global summary.
    Returns: (global_summary_string, list_of_context_files)
    """
    subtitles = list(srt.parse(open(srt_path, encoding='utf-8').read()))
    full_text = " ".join([sub.content for sub in subtitles])
    base_name = os.path.splitext(srt_path)[0]
    
    context_files = []
    
    # Video
    video_path = base_name + ".mp4"
    if os.path.exists(video_path):
        try:
            print(f"Uploading Video: {video_path}")
            context_files.append(upload_to_gemini(video_path, client))
        except Exception as e:
            print(f"Video upload failed: {e}")
            
    # Aux Files
    for ext in [".pptx", ".pdf", ".txt", ".md"]:
        aux_path = base_name + ext
        if os.path.exists(aux_path) and aux_path != srt_path:
            try:
                print(f"Uploading Aux: {aux_path}")
                context_files.append(upload_to_gemini(aux_path, client))
            except Exception as e:
                print(f"Aux upload failed: {e}")
    
    # Generate Summary
    print("Generating Global Summary...")
    summary_prompt = """
    Analyze the uploaded video (if available), auxiliary documents (slides/notes), and the following transcript.
    Task:
    1. Provide a comprehensive summary of the mathematical topic.
    2. Extract a "Style Guide" of specific notations and terminology used in the video (visuals on blackboard/slides).
    
    Transcript:
    """
    
    contents = context_files + [summary_prompt, full_text]
    
    try:
        response = client.models.generate_content(
            model="gemini-3.0-pro-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GlobalSummary,
            ),
        )
        data = GlobalSummary.model_validate_json(response.text)
        summary_text = f"Summary: {data.summary}\nKey Terms/Style: {data.style_guide}"
        print(f"Global Summary Generated.")
        return summary_text, context_files
    except Exception as e:
        print(f"Global Summary failed: {e}")
        return "Topic: Mathematics.", context_files

def ai_review_chunks(subtitles: List[srt.Subtitle], context_files: List[object], client) -> List[srt.Subtitle]:
    """
    Pass 2: Review subtitles in chunks of 100 using Gemini Pro + Video Context.
    """
    CHUNK_SIZE = 100
    print(f"Starting Chunk Review (Gemini Pro) - {len(subtitles)} lines...")
    
    class ChunkCorrection(BaseModel):
        index: int
        content: str
        
    class ChunkReviewResponse(BaseModel):
        corrections: List[ChunkCorrection]

    for i in range(0, len(subtitles), CHUNK_SIZE):
        chunk = subtitles[i : i + CHUNK_SIZE]
        chunk_text = "\n".join([f"{sub.index}: {sub.content}" for sub in chunk])
        
        prompt = f"""
        Review the following subtitle chunk (Lines {i} to {i+len(chunk)}).
        You have access to the VIDEO and AUXILIARY FILES.
        
        Task:
        1. Check for any mismatches between the text and better visual context (blackboard, slides).
        2. Ensure strict mathematical terminology consistency.
        3. OUTPUT: A JSON list of ONLY the lines that need correction. If a line is correct, do not include it.
        
        Subtitles:
        {chunk_text}
        """
        
        contents = context_files + [prompt]
        
        print(f"Reviewing chunk {i}-{i+len(chunk)}...", end="\r")
        try:
            response = client.models.generate_content(
                model="gemini-3.0-pro-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ChunkReviewResponse,
                ),
            )
            data = ChunkReviewResponse.model_validate_json(response.text)
            
            updates_count = 0
            for correction in data.corrections:
                # Find the subtitle with this index (srt index is 1-based usually, but here we iterate list)
                # Wait, srt.index is the file's index. We should map carefully.
                # 'chunk' is a slice of objects.
                # Let's verify index matching. 
                # The prompt received "sub.index: sub.content".
                # We should look up by srt index.
                target = next((s for s in chunk if s.index == correction.index), None)
                if target:
                    if target.content != correction.content:
                        target.content = correction.content
                        updates_count += 1
            
            if updates_count > 0:
                print(f"  Chunk {i}: {updates_count} corrections applied.")
                
        except Exception as e:
            print(f"  Chunk {i} review failed: {e}")
            
    print("\nChunk Review Complete.")
    return subtitles

def ai_refine_subtitles(srt_path: str, global_summary: str, context_files: list, client) -> str:
    """
    Pass 1: Line-by-line refinement (Flash).
    """
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found")

    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    if not subtitles:
        return srt_path
    
    print(f"Pass 1: Line-by-Line Refinement (Flash) - 20 threads...")
    
    results = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_index = {
            executor.submit(process_single_subtitle, i, subtitles, global_summary, client): i 
            for i in range(len(subtitles))
        }
        
        for i, future in enumerate(as_completed(future_to_index)):
            idx = future_to_index[future]
            try:
                data = future.result()
                results[idx] = data if data else ModifiedSentence(output=subtitles[idx].content)
            except Exception:
                results[idx] = ModifiedSentence(output=subtitles[idx].content)
            
            if i % 100 == 0:
                print(f"  Processed {i}/{len(subtitles)}...", end="\r")

    # Reassemble & Merge
    final_subtitles = []
    skip_indices = set()
    sorted_indices = sorted(results.keys())
    
    for i in sorted_indices:
        if i in skip_indices: continue
        
        current_data = results[i]
        current_sub = subtitles[i]
        current_sub.content = current_data.output
        
        if current_data.should_merge_next and (i + 1) in results:
            next_data = results[i+1]
            combined = current_data.output + next_data.output
            if len(combined) <= 30:
                current_sub.content = combined
                current_sub.end = subtitles[i+1].end
                skip_indices.add(i+1)
        

    # Save Intermediate
    inter_filename = os.path.splitext(srt_path)[0] + "_flash.srt"
    with open(inter_filename, "w", encoding="utf-8") as f:
        f.write(srt.compose(final_subtitles))
        
    return inter_filename, final_subtitles

class OffTopicSegment(BaseModel):
    start_time: str
    end_time: str
    description: str

class OffTopicReport(BaseModel):
    segments: List[OffTopicSegment]

def detect_off_topic_segments(srt_path: str, global_summary: str, context_files: List[object], client):
    """
    Analyzes the refined transcript to identify segments unrelated to the main topic.
    """
    if not os.path.exists(srt_path):
        return

    print("Detecting off-topic segments...")
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))
        
    transcript_text = "\n".join([f"[{sub.start} --> {sub.end}] {sub.content}" for sub in subtitles])
    
    prompt = f"""
    Analyze the following lecture transcript.
    Goal: Identify segments that are **Off-Topic** or unrelated to the main educational content.
    Examples:
    - Logistics (assignments, exam dates, "don't come tomorrow").
    - Jokes, personal stories, idle chatter.
    - Political commentary or unrelated current events.
    - Classroom management ("don't be afraid to raise hands").
    
    Video Topic Summary:
    {global_summary}
    
    Transcript:
    {transcript_text}
    """
    
    contents = context_files + [prompt]
    
    try:
        response = client.models.generate_content(
            model="gemini-3.0-pro-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=OffTopicReport,
            ),
        )
        data = OffTopicReport.model_validate_json(response.text)
        
        # Save Report
        report_path = os.path.splitext(srt_path)[0] + "_off_topic.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Off-Topic Segments Report\n")
            f.write(f"Video: {os.path.basename(srt_path)}\n")
            f.write(f"{'='*50}\n\n")
            
            if not data.segments:
                f.write("No significant off-topic segments detected.\n")
            else:
                for seg in data.segments:
                    f.write(f"Time: {seg.start_time} - {seg.end_time}\n")
                    f.write(f"Content: {seg.description}\n")
                    f.write(f"{'-'*30}\n")
                    
        print(f"Off-Topic Report saved: {report_path}")
        
    except Exception as e:
        print(f"Off-topic detection failed: {e}")


def process_subtitles(srt_path: str) -> str:
    client = get_client()
    
    # 0. Prepare Context
    global_summary, context_files = prepare_gemini_context(srt_path, client)
    
    # 1. Flash Refinement
    flash_path, refined_subtitles = ai_refine_subtitles(srt_path, global_summary, context_files, client)
    
    # 2. Pro Chunk Review (using result from Pass 1)
    reviewed_subtitles = ai_review_chunks(refined_subtitles, context_files, client)
    
    # Save Final
    final_path = os.path.splitext(srt_path)[0] + "_refined.srt"
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(reviewed_subtitles))
        
    # 3. Off-Topic Detection (New)
    detect_off_topic_segments(final_path, global_summary, context_files, client)
        
    # 4. Deduplicate (Optional heuristic final polish)
    # 4. Deduplicate (Optional heuristic final polish)
    # final_path_dedup = deduplicate_subtitles(final_path)
    # WARNING: deduplicate_subtitles is missing, returning final_path for now
    
    return final_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Refine subtitles using Gemini API.")
    parser.add_argument("srt_path", type=str, help="Path to the input SRT file")
    
    args = parser.parse_args()
    
    try:
        process_subtitles(args.srt_path)
    except Exception as e:
        print(f"Error: {e}")
