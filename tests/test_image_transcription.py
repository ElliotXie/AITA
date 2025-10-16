#!/usr/bin/env python3
"""
Test script for transcribing a raw image using the LLM API.
This script takes an image (local file or URL) and transcribes its content to text.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aita.services.llm.openrouter import create_openrouter_client
from aita.services.storage import upload_image_to_gcs, get_public_url
from aita.utils.prompts import get_transcription_prompt
from aita.config import get_config


def setup_test_directories():
    """Create test output directories"""
    test_dir = Path(__file__).parent
    results_dir = test_dir / "transcription_results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def upload_local_image_to_gcs(image_path: str) -> str:
    """Upload a local image to GCS and return the public URL"""
    try:
        config = get_config()

        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(image_path).stem
        blob_name = f"test_transcription/{timestamp}_{original_name}.jpg"

        print(f"📤 Uploading {image_path} to GCS...")
        upload_image_to_gcs(image_path, blob_name)

        url = get_public_url(blob_name)
        print(f"✅ Image uploaded: {url}")
        return url

    except Exception as e:
        print(f"❌ Failed to upload image: {e}")
        raise


def transcribe_image(image_url: str, question_context: Optional[str] = None) -> dict:
    """Transcribe an image using the LLM API"""
    try:
        # Create LLM client
        print("🤖 Creating OpenRouter client...")
        client = create_openrouter_client()

        # Prepare prompt
        if question_context:
            prompt = get_transcription_prompt(question_context)
        else:
            prompt = get_transcription_prompt("Complete exam page or handwritten content")

        print(f"🔍 Transcribing image: {image_url}")
        print(f"📝 Using model: {client.model}")

        # Make the transcription request
        response = client.complete_with_image(prompt, image_url)

        # Try to parse as JSON, fallback to plain text
        try:
            result = json.loads(response)
            result["raw_response"] = response
        except json.JSONDecodeError:
            result = {
                "transcribed_text": response,
                "confidence": None,
                "notes": "Response was not in JSON format",
                "raw_response": response
            }

        return result

    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        raise


def save_results(results: dict, image_url: str, results_dir: Path):
    """Save transcription results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_file = results_dir / f"transcription_{timestamp}.json"
    full_results = {
        "timestamp": timestamp,
        "image_url": image_url,
        "transcription_results": results
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    # Save text-only version
    text_file = results_dir / f"transcription_{timestamp}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(f"Image URL: {image_url}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write("TRANSCRIBED TEXT:\n")
        f.write("=" * 50 + "\n")
        f.write(results.get('transcribed_text', 'No text found') + "\n")

        if results.get('notes'):
            f.write("\n" + "=" * 50 + "\n")
            f.write("NOTES:\n")
            f.write("=" * 50 + "\n")
            f.write(results['notes'] + "\n")

        if results.get('confidence'):
            f.write(f"\nConfidence: {results['confidence']}\n")

    print(f"💾 Results saved:")
    print(f"   JSON: {json_file}")
    print(f"   Text: {text_file}")

    return json_file, text_file


def main():
    """Main test function"""
    print("🚀 AITA Image Transcription Test")
    print("=" * 50)

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        print("Set your API key with: export OPENROUTER_API_KEY='your_key_here'")
        return False

    # Setup directories
    results_dir = setup_test_directories()

    # Get image input from user or use default
    image_input = input("Enter image path (local file) or URL (press Enter for demo URL): ").strip()

    if not image_input:
        # Use demo URL from the working example
        image_url = "https://storage.googleapis.com/exam1_uwmadison/courses/test_assignment/2025-10-14/test_student/page_1.jpg"
        print(f"🖼️  Using demo image: {image_url}")
    elif image_input.startswith(('http://', 'https://')):
        # Direct URL
        image_url = image_input
        print(f"🖼️  Using provided URL: {image_url}")
    else:
        # Local file - upload to GCS
        if not Path(image_input).exists():
            print(f"❌ File not found: {image_input}")
            return False

        try:
            image_url = upload_local_image_to_gcs(image_input)
        except Exception as e:
            print(f"❌ Failed to upload image: {e}")
            return False

    # Optional question context
    question_context = input("Enter question context (optional, press Enter to skip): ").strip()
    if not question_context:
        question_context = None

    try:
        # Perform transcription
        print("\n🔄 Starting transcription...")
        results = transcribe_image(image_url, question_context)

        # Save results
        json_file, text_file = save_results(results, image_url, results_dir)

        # Display results
        print("\n📋 TRANSCRIPTION RESULTS:")
        print("=" * 50)
        print(f"Text: {results.get('transcribed_text', 'No text found')}")

        if results.get('confidence'):
            print(f"Confidence: {results['confidence']}")

        if results.get('notes'):
            print(f"Notes: {results['notes']}")

        print("\n✅ Transcription completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)