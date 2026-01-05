# Clean Video

An automated video processing pipeline designed for educational content. It streamlines the creation of high-quality modifications by removing silence, generating accurate subtitles, and leveraging AI for context-aware refinement.

## Features

- **Smart Silence Removal**: Uses `faster-whisper` (base model) to accurately detect speech segments and remove silence gaps (`core/cleaner.py`).
- **High-Fidelity Transcription**: Generates subtitles using `faster-whisper` (large model) with customizable prompts and language support (`core/transcriber.py`).
- **AI-Powered Refinement**: A sophisticated multi-pass editor powered by Google Gemini (`core/editor.py`):
    - **Global Context**: Generates a topic summary and style guide from the video and auxiliary files (slides, notes).
    - **Flash Refinement**: Fast, line-by-line grammar correction and flow improvement.
    - **Mathematical Notation**: Standardizes math symbols based on a predefined dictionary.
    - **Smart Merging**: Intelligently merges short, fragmented sentences for better readability.
    - **Pro Chunk Review**: A second pass with `Gemini 3.0 Pro` to cross-check text against video visuals (blackboard/slides) for maximum accuracy.
    - **Off-Topic Detection**: Identifies non-educational segments (logistics, chit-chat) for potential removal.

## Requirements

- **Python 3.10+**
- **FFmpeg**: Must be installed and accessible in your system path.
- **NVIDIA GPU**: Highly recommended for `faster-whisper` and overall performance.
- **Google Gemini API Key**: Required for the AI refinement steps.

## Installation

This project uses `uv` for modern, fast Python package management.

### 1. Clone the repository
```bash
git clone <repository-url>
cd clean-video
```

### 2. Install dependencies
Using `uv` (Recommended):
```bash
uv sync
```

Or using standard pip:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root and add your Gemini API key:
```bash
GEMINI_API_KEY=your_api_key_here
```

## Usage

The primary entry point is `main.py`, which orchestrates the entire pipeline.

### Processing a Single Video
```bash
uv run main.py /path/to/video.mp4
```

### Batch Processing a Directory
Process all `.mp4` files in a folder, utilizing parallel workers for speed.
```bash
uv run main.py /path/to/videos/ --workers 3
```

### CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input` | Path | Required | Path to a single video file OR a directory containing videos. |
| `--gap` | Float | `1.0` | Minimum silence duration (in seconds) to initiate a cut. |
| `--workers` | Int | `1` | Number of parallel processes to use (Directory mode only). |
| `--language` | String | `"zh"` | Code for the spoken language (e.g., "en", "zh", "ja"). |
| `--initial_prompt` | String | *"這是一個繁體中文的句子"* | Context prompt passed to Whisper to guide style/terminology. |

## Modular Usage

The core components can also be run independently for specific tasks:

- **Silence Removal Only**:
  ```bash
  uv run core/cleaner.py input.mp4 --gap 0.5
  ```
- **Transcription Only**:
  ```bash
  uv run core/transcriber.py input.mp4 --model large-v3 --language en
  ```
- **AI Refinement Only** (Requires an existing SRT):
  ```bash
  uv run core/editor.py input.srt
  ```

## Project Structure

```
$ tree clean-video/
clean-video/
├── core/
│   ├── cleaner.py       # Silence detection and ffmpeg trimming
│   ├── transcriber.py   # Whisper-based subtitle generation
│   ├── editor.py        # Gemini-powered subtitle refinement & merging
│   └── math_symbols.txt # Symbol replacement rules for math content
├── main.py              # Main orchestrator script
├── download_videos.py   # Download videos from urls.txt
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```
