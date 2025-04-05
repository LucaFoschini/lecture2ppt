# lecture2ppt

A Python tool for processing video presentations, extracting slides, transcribing audio, and generating comprehensive transcripts with slide descriptions.

## Project Structure

```
lecture2ppt/
├── sample/           # Sample video files
├── output/           # Generated output files (PDFs, transcripts, etc.)
├── temp/            # Temporary files (images, etc.)
└── src/             # Source code
    ├── main.py      # Main processing script
    ├── merge_transcripts.py  # Transcript merging script
    └── create_ppt.py         # PowerPoint generation script
```

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (install using `pip install -r requirements.txt`):
  - moviepy
  - openai
  - python-dotenv
  - opencv-python
  - numpy
  - Pillow
  - PyMuPDF
  - python-pptx

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lecture2ppt.git
cd lecture2ppt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### 1. Process Video and Extract Slides

The main script processes a video file, extracts slides, and generates transcripts:

```bash
python src/main.py sample/test_video.mp4
```

Optional flags:
- `--skip-audio-extraction`: Skip audio extraction
- `--skip-audio-transcription`: Skip audio transcription
- `--skip-text-extraction`: Skip text extraction from slides

### 2. Merge Transcripts

Combine slide descriptions with audio transcript:

```bash
python src/merge_transcripts.py --slide-transcripts output/slide_transcripts.txt --transcript output/transcript.txt --output output/merged_transcript.txt
```

### 3. Create PowerPoint Presentation

Generate a PowerPoint presentation with slides and speaker notes:

```bash
python src/create_ppt.py --slides-pdf output/slides.pdf --transcript output/transcript.txt --output output/presentation.pptx
```

## Output Files

All output files are saved in the `output/` directory:
- `slides.pdf`: Extracted slides from the video
- `transcript.txt`: Audio transcription
- `slide_transcripts.txt`: Slide descriptions
- `merged_transcript.txt`: Combined slide descriptions and audio transcript
- `presentation.pptx`: PowerPoint presentation with slides and speaker notes

## Example: Tim Urban's TED Talk

To process Tim Urban's TED Talk about procrastination:

1. Download the video:
```bash
curl -L -o sample/test_video.mp4 "https://download.ted.com/talks/TimUrban_2016-480p.mp4"
```

2. Process the video:
```bash
python src/main.py sample/test_video.mp4
```

3. Merge transcripts:
```bash
python src/merge_transcripts.py --slide-transcripts output/slide_transcripts.txt --transcript output/transcript.txt --output output/merged_transcript.txt
```

4. Create PowerPoint:
```bash
python src/create_ppt.py --slides-pdf output/slides.pdf --transcript output/transcript.txt --output output/presentation.pptx
```

The script will:
- Extract slides from the video
- Transcribe the audio
- Generate slide descriptions
- Create a merged transcript
- Generate a PowerPoint presentation with slides and corresponding speaker notes

## License

Apache License 2.0
