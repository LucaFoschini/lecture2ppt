import os
from moviepy.editor import VideoFileClip
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import timedelta
import tempfile
import base64
import fitz
import pytesseract
import json

# Load environment variables
load_dotenv()

class SmartTranscript:
    def __init__(self, video_path, output_dir="output", skip_audio=False, skip_audio_extraction=False, skip_text_extraction=False, skip_slide_extraction=False):
        self.video_path = video_path
        self.output_dir = output_dir
        self.audio_path = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '.mp3'))
        self.transcript = None
        self.skip_audio = skip_audio
        self.skip_audio_extraction = skip_audio_extraction
        self.skip_text_extraction = skip_text_extraction
        self.skip_slide_extraction = skip_slide_extraction
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.slides_dir = output_dir
        self.temp_dir = "temp"
        
        # Create necessary directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_audio(self):
        """Extract audio from video file using moviepy"""
        if self.skip_audio_extraction:
            print("Skipping audio extraction...")
            return
            
        print("Extracting audio...")
        video = VideoFileClip(self.video_path)
        video.audio.write_audiofile(self.audio_path, codec='mp3', bitrate='64k')
        video.close()
        print(f"Audio extracted to {self.audio_path}")

    def format_timestamp(self, seconds):
        """Format seconds into HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = td.total_seconds() % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def transcribe_audio(self):
        """Transcribe audio using OpenAI's Whisper API"""
        if self.skip_audio:
            print("Using existing transcript.txt...")
            with open(os.path.join(self.output_dir, "transcript.txt"), "r") as f:
                self.transcript = f.read()
            return
            
        print("Transcribing audio...")
        
        # Get video duration
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps
        cap.release()
        
        with open(self.audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
            
            # Calculate the actual start time by comparing audio duration with video duration
            if transcript.segments:
                audio_duration = transcript.segments[-1].end - transcript.segments[0].start
                time_offset = video_duration - audio_duration
                print(f"Video duration: {self.format_timestamp(video_duration)}")
                print(f"Audio duration: {self.format_timestamp(audio_duration)}")
                print(f"Time offset: {self.format_timestamp(time_offset)}")
            else:
                time_offset = 0
                print("No speech detected in audio")
            
            # Format transcript with timestamps
            lines = []
            for segment in transcript.segments:
                # Adjust timestamps to be relative to video start
                start = self.format_timestamp(segment.start + time_offset)
                end = self.format_timestamp(segment.end + time_offset)
                text = segment.text.strip()
                lines.append(f"[{start} --> {end}] {text}")
            
            self.transcript = "\n".join(lines)
            
            # Save transcript to file in output directory
            transcript_path = os.path.join(self.output_dir, "transcript.txt")
            with open(transcript_path, "w") as f:
                f.write(self.transcript)
            print("Transcript saved to", transcript_path)

    def is_slide(self, frame):
        """Determine if a frame is likely a presentation slide"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate image statistics
        mean, std = cv2.meanStdDev(gray)
        
        # Check for uniform background (low standard deviation)
        if std[0][0] > 100:  # Much more lenient threshold
            print(f"Rejected: High standard deviation ({std[0][0]:.2f})")
            return False
            
        # Check for text-like features using edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Slides typically have some text (edges) but not too much
        if edge_density < 0.001 or edge_density > 0.5:  # Much more lenient thresholds
            print(f"Rejected: Edge density out of range ({edge_density:.4f})")
            return False
            
        # Check for dominant background color (white or black)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Check for white or black background
        white_black_ratio = (hist[0] + hist[-1]) / hist.sum()
        if white_black_ratio < 0.1:  # Much more lenient threshold
            print(f"Rejected: Low white/black ratio ({white_black_ratio:.2f})")
            return False
            
        print(f"Accepted slide: std={std[0][0]:.2f}, edges={edge_density:.4f}, wb_ratio={white_black_ratio:.2f}")
        return True

    def is_similar_to_existing(self, frame, existing_slides, threshold=0.98):
        """Check if a frame is similar to any existing slide"""
        if not existing_slides:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for existing_frame in existing_slides:
            # Convert existing frame to grayscale
            existing_gray = cv2.cvtColor(existing_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate structural similarity
            similarity = cv2.matchTemplate(gray, existing_gray, cv2.TM_CCOEFF_NORMED)[0][0]
            if similarity > threshold:
                return True
                
        return False

    def has_solid_background(self, frame, threshold=0.7, max_monochrome=0.95):
        """Check if the frame has a dominant solid color background"""
        # Convert to RGB for color analysis
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Reshape to a list of pixels
        pixels = rgb.reshape(-1, 3)
        
        # Count unique colors
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        unique_colors = unique_colors[sorted_indices]
        counts = counts[sorted_indices]
        
        # Check if the most common color is dominant
        most_common_ratio = counts[0] / len(pixels)
        most_common_color = unique_colors[0]
        
        # For the most common color, check if it's a solid color (all channels equal)
        is_solid = np.all(most_common_color == most_common_color[0])
        
        # Check if the image is too monochromatic (almost all black or white)
        if most_common_ratio > max_monochrome:
            return False
            
        # Special check for black backgrounds - be more strict
        if np.all(most_common_color == [0, 0, 0]) and most_common_ratio > 0.5:  # If more than 50% is black
            return False
            
        # Special check for white backgrounds - be more lenient
        if np.all(most_common_color >= [250, 250, 250]):  # If the background is very close to white
            return True
            
        return most_common_ratio > threshold and is_solid

    def extract_slides(self):
        """Extract slides from video using scene detection and solid background filtering"""
        print("Extracting slides...")
        
        # Open the video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize variables
        prev_frame = None
        slides = []
        existing_slides = []  # Store actual frames for comparison
        min_scene_duration = int(fps * 0.3)  # Minimum 0.3 seconds for a slide
        frame_count = 0
        stable_frame_count = 0
        current_slide = None
        current_slide_start = 0
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Check if we have a pending slide at the end of the video
                if current_slide is not None:
                    if self.has_solid_background(current_slide):
                        timestamp = self.format_timestamp(current_slide_start)
                        slide_path = os.path.join(self.temp_dir, f"slide_{len(slides):03d}.jpg")  # Changed to temp directory
                        cv2.imwrite(slide_path, current_slide)
                        slides.append((slide_path, timestamp))
                        print(f"Saved final slide at {timestamp}")
                break
                
            frame_count += 1
            current_time = frame_count / fps
            
            # Convert frame to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate difference between frames
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = np.mean(diff)
                
                # If the difference is small, we're on a stable scene
                if mean_diff < 5:  # Threshold for scene change
                    stable_frame_count += 1
                    if stable_frame_count == min_scene_duration:
                        # We've found a stable scene, check if it's a slide
                        if (current_slide is not None and 
                            self.has_solid_background(current_slide) and 
                            not self.is_similar_to_existing(current_slide, existing_slides)):
                            
                            timestamp = self.format_timestamp(current_time - min_scene_duration/fps)
                            slide_path = os.path.join(self.temp_dir, f"slide_{len(slides):03d}.jpg")  # Changed to temp directory
                            
                            # Save the slide
                            cv2.imwrite(slide_path, current_slide)
                            slides.append((slide_path, timestamp))
                            existing_slides.append(current_slide)
                            print(f"Saved slide at {timestamp}")
                            
                            # Debug output around 9 minutes
                            if 8.5 * 60 <= current_time <= 9.5 * 60:
                                print(f"Debug at {timestamp}: mean_diff={mean_diff:.2f}, stable_frames={stable_frame_count}")
                else:
                    stable_frame_count = 0
                    current_slide = frame
                    current_slide_start = current_time
            
            prev_frame = gray
            
            # Show progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%", end='\r')
        
        cap.release()
        print("\nExtracted", len(slides), "slides")
        return slides

    def create_slides_pdf(self, slides):
        """Create a PDF with extracted slides and timestamps (in document metadata)"""
        print("Creating slides PDF...")
        
        if not slides:
            print("No slides found!")
            return
            
        # Create PDF
        pdf_path = os.path.join(self.output_dir, "slides.pdf")
        pdf_doc = fitz.open()
        
        # Store all timestamps in document metadata as JSON
        # Use the "subject" field to store our timestamps
        timestamps = [timestamp for _, timestamp in slides]
        pdf_doc.set_metadata({
            "subject": json.dumps(timestamps),
            "creator": "SmartTranscript",
            "producer": "SmartTranscript Slide Extractor"
        })
        
        # Process each slide
        for slide_path, timestamp in slides:
            # Open the image
            img = Image.open(slide_path)
            
            # Convert to RGB (required for PDF)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Add timestamp to the image (smaller font, top right)
            draw = ImageDraw.Draw(img)
            # Use a default font if custom font is not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)  # Smaller font size
            except:
                font = ImageFont.load_default()
            
            # Draw timestamp at the top right
            text = f"Time: {timestamp}"
            text_width = draw.textlength(text, font=font)
            draw.text((img.width - text_width - 10, 10),  # Position at top right
                     text, fill='black', font=font)
            
            # Save image as temporary PNG (better quality than JPEG)
            temp_png = os.path.join(self.temp_dir, "temp_slide.png")
            img.save(temp_png)
            
            # Create new PDF page with image dimensions
            page = pdf_doc.new_page(width=img.width, height=img.height)
            
            # Insert image into page
            page.insert_image(page.rect, filename=temp_png)
            
            # Clean up temporary PNG
            os.remove(temp_png)
        
        # Save the PDF
        pdf_doc.save(pdf_path)
        pdf_doc.close()
        print(f"Slides saved to {pdf_path}")
        
        # Clean up individual slide files from temp directory
        for slide_path, _ in slides:
            if os.path.exists(slide_path):
                os.remove(slide_path)

    def call_vision_api(self, base64_image, prompt, max_tokens=500):
        """Call the GPT-4 Vision API to analyze an image."""
        timestamp = prompt.split("[")[1].split("]")[0]
        try:
            print(f"Calling GPT-4 Vision for slide at {timestamp}")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing slide at {timestamp}: {str(e)}")
            raise

    def read_slides_from_pdf(self):
        """Read slides from existing PDF and extract their timestamps and text content using GPT-4 Vision"""
        print("Reading slides from PDF...")
        slides = []
        
        try:
            # Open the PDF
            pdf_path = os.path.join(self.output_dir, "slides.pdf")
            if not os.path.exists(pdf_path):
                print("Error: slides.pdf not found")
                return []
                
            pdf = fitz.open(pdf_path)
            print(f"PDF opened successfully with {len(pdf)} pages")
            
            # Get timestamps from document metadata
            metadata = pdf.metadata
            if not metadata or "subject" not in metadata or not metadata["subject"]:
                print("Error: No timestamp metadata found in PDF")
                return []
                
            try:
                timestamps = json.loads(metadata["subject"])
                if len(timestamps) != len(pdf):
                    print(f"Error: Number of timestamps ({len(timestamps)}) doesn't match number of pages ({len(pdf)})")
                    return []
            except json.JSONDecodeError:
                print("Error: Invalid timestamp metadata in PDF")
                return []
            
            # Extract each page as an image
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                timestamp = timestamps[page_num]
                print(f"\nProcessing page {page_num + 1}...")
                print(f"Found timestamp: {timestamp}")
                
                # Convert page to image
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Save the image with a descriptive name in the temp directory
                slide_path = os.path.join(self.temp_dir, f"slide_{page_num + 1}_{timestamp.replace(':', '-')}.jpg")
                img.save(slide_path)
                
                # Extract text using GPT-4 Vision if not skipped
                text = ""
                if not self.skip_text_extraction:
                    with open(slide_path, "rb") as image_file:
                        image_data = image_file.read()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        
                        try:
                            print("Calling GPT-4 Vision API to extract text...")
                            text = self.call_vision_api(
                                base64_image,
                                f"[{timestamp}] Please extract all text content from this presentation slide. Ignore any timestamps or time-related information. Format the response as a clean text block with proper line breaks."
                            )
                        except Exception as e:
                            print(f"Error extracting text: {str(e)}")
                
                slides.append((slide_path, timestamp, text))
                # Note: We don't delete the temporary image here anymore
        
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return []
            
        finally:
            if 'pdf' in locals():
                pdf.close()
                
        print(f"Successfully read {len(slides)} slides from PDF")
        return slides

    def describe_slide(self, image_path, timestamp, extracted_text=""):
        """Generate a natural language description of a slide using GPT-4 Vision"""
        print(f"Analyzing slide at {timestamp}...")
        
        try:
            # Check if the image file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image file {image_path} not found. Skipping description generation.")
                return f"[{timestamp}]\nNo description available (image file not found)\n\n"
            
            # Read the image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # Prepare the image for GPT-4 Vision
                prompt = f"[{timestamp}] Please describe this presentation slide in detail, ignoring any timestamps or time-related information. "
                if extracted_text:
                    prompt += f"Here is the text extracted from the slide:\n{extracted_text}\n\n"
                prompt += "Include any text, diagrams, or references. Format the response as a concise paragraph (3-5 lines). Focus on the content and meaning rather than visual style or timing information."
                
                try:
                    print("Calling GPT-4 Vision API to generate description...")
                    description = self.call_vision_api(base64_image, prompt, max_tokens=300)
                    return f"[{timestamp}]\n{description}\n\n"
                except Exception as e:
                    print(f"Error generating description: {str(e)}")
                    return f"[{timestamp}]\nError generating description: {str(e)}\n\n"
                    
        except Exception as e:
            print(f"Error processing slide at {timestamp}: {str(e)}")
            return f"[{timestamp}]\nError generating description: {str(e)}\n\n"

    def process_video(self):
        """Process the video and create transcript and slides"""
        if not self.skip_audio_extraction:
            self.extract_audio()
        if not self.skip_audio:
            self.transcribe_audio()
            
        # First extract slides and create PDF
        if not self.skip_slide_extraction:
            print("Extracting slides from video...")
            slides = self.extract_slides()
            if slides:
                self.create_slides_pdf(slides)
            else:
                print("No slides were extracted from the video.")
                return self.transcript
        
        # Now read the slides from the created PDF
        print("Reading slides from PDF...")
        slides = self.read_slides_from_pdf()
        
        # Generate descriptions for slides
        print("Generating slide descriptions...")
        with open(os.path.join(self.output_dir, "slide_transcripts.txt"), "w") as f:
            for slide_path, timestamp, text in slides:
                description = self.describe_slide(slide_path, timestamp, text)
                f.write(description)
                # Clean up the temporary image after both text extraction and description generation
                if os.path.exists(slide_path):
                    os.remove(slide_path)
        print("Slide descriptions saved to slide_transcripts.txt")
            
        return self.transcript

def main():
    parser = argparse.ArgumentParser(description='Extract audio, create transcript, and extract slides from video.')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output-dir', default='output', help='Directory to store output files')
    parser.add_argument('--skip-audio-transcription', action='store_true', help='Skip audio transcription and use existing transcript.txt')
    parser.add_argument('--skip-audio-extraction', action='store_true', help='Skip audio extraction from video')
    parser.add_argument('--skip-text-extraction', action='store_true', help='Skip text extraction from slides using GPT-4 Vision')
    parser.add_argument('--skip-slide-extraction', action='store_true', help='Skip slide extraction from video and use existing slides.pdf')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    processor = SmartTranscript(
        args.video_path,
        output_dir=args.output_dir,
        skip_audio=args.skip_audio_transcription,
        skip_audio_extraction=args.skip_audio_extraction,
        skip_text_extraction=args.skip_text_extraction,
        skip_slide_extraction=args.skip_slide_extraction
    )
    processor.process_video()

if __name__ == "__main__":
    main() 