import re
from datetime import datetime, timedelta
import os

def parse_timestamp(timestamp_str):
    """Convert timestamp string to timedelta"""
    hours, minutes, seconds = map(float, timestamp_str.split(':'))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def parse_transcript_line(line):
    """Parse a line from the transcript file"""
    match = re.match(r'\[(.*?) --> (.*?)\] (.*)', line)
    if match:
        start_time = parse_timestamp(match.group(1))
        end_time = parse_timestamp(match.group(2))
        text = match.group(3)
        return start_time, end_time, text
    return None, None, None

def get_transcript_for_slide(slide_timestamp, next_slide_timestamp, transcript_lines):
    """Get all transcript lines that fall between this slide's timestamp and the next slide's timestamp"""
    slide_time = parse_timestamp(slide_timestamp)
    next_time = parse_timestamp(next_slide_timestamp) if next_slide_timestamp else None
    transcript_text = []
    
    for line in transcript_lines:
        start_time, end_time, text = parse_transcript_line(line)
        if start_time and end_time:
            # Include lines that start during this slide's time
            if slide_time <= start_time and (next_time is None or start_time < next_time):
                transcript_text.append(text)
    
    return ' '.join(transcript_text)

def merge_transcripts(slide_transcripts_file, transcript_file, output_file):
    # Read transcript
    with open(transcript_file, 'r') as f:
        transcript_lines = f.readlines()
    
    # Read slide descriptions
    with open(slide_transcripts_file, 'r') as f:
        slide_descriptions = f.read().split('\n\n')
    
    # Extract timestamps and descriptions
    slides = []
    for slide_desc in slide_descriptions:
        if not slide_desc.strip():
            continue
        timestamp_match = re.match(r'\[(.*?)\]', slide_desc)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            description = slide_desc[timestamp_match.end():].strip()
            slides.append((timestamp, description))
    
    # Create merged transcript
    merged_text = []
    
    # Process each slide
    for i, (timestamp, description) in enumerate(slides):
        # Get next slide's timestamp
        next_timestamp = slides[i + 1][0] if i + 1 < len(slides) else None
        
        # Get transcript text for this slide
        transcript_text = get_transcript_for_slide(timestamp, next_timestamp, transcript_lines)
        
        # Add slide description
        merged_text.append(f"[Slide at {timestamp}]")
        merged_text.append(description)
        
        # Add transcript text if any
        if transcript_text:
            merged_text.append("\n[Narrator]")
            merged_text.append(transcript_text)
        
        # Add separator between slides
        merged_text.append("\n" + "="*80 + "\n")
    
    # Write merged transcript to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(merged_text))
    
    print(f"Merged transcript saved to {output_file}")

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Define file paths
    slide_transcripts_file = os.path.join("output", "slide_transcripts.txt")
    transcript_file = os.path.join("output", "transcript.txt")
    output_file = os.path.join("output", "merged_transcript.txt")
    
    merge_transcripts(slide_transcripts_file, transcript_file, output_file) 