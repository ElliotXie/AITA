#!/usr/bin/env python3
"""
Quick integration test: Load image from GCS URL and analyze with LLM API
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aita.services.llm.openrouter import create_openrouter_client


def test_gcs_image_with_llm():
    """Test using a GCS image URL with the LLM API"""

    print("🔗 Testing GCS Image URL with LLM API")
    print("=" * 50)

    # Check if API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set")
        print("Set your API key with: export OPENROUTER_API_KEY='your_key_here'")
        return False

    # Your existing GCS image URL (from your working example)
    image_url = "https://storage.googleapis.com/exam1_uwmadison/courses/test_assignment/2025-10-14/test_student/page_1.jpg"

    print(f"🖼️  Using image URL: {image_url}")
    print()

    try:
        # Create LLM client
        print("📡 Creating LLM client...")
        client = create_openrouter_client()
        print(f"✅ Client created with model: {client.model}")

        # Test 1: Basic image analysis
        print("\n🤖 Test 1: Basic Image Analysis")
        print("Asking: 'What is in this image?'")

        response = client.complete_with_image(
            prompt="What is in this image?",
            image_url=image_url
        )

        print("📝 LLM Response:")
        print("-" * 30)
        print(response)
        print("-" * 30)

        # Test 2: More specific analysis
        print("\n🤖 Test 2: Exam-Specific Analysis")
        print("Asking: 'Is this an exam paper? What can you see written on it?'")

        response2 = client.complete_with_image(
            prompt="Is this an exam paper? What can you see written on it? Describe any text, questions, or answers you can identify.",
            image_url=image_url
        )

        print("📝 LLM Response:")
        print("-" * 30)
        print(response2)
        print("-" * 30)

        # Test 3: Using AITA prompts
        print("\n🤖 Test 3: AITA Name Extraction")
        from aita.utils.prompts import get_name_extraction_prompt

        name_prompt = get_name_extraction_prompt()
        print("Using AITA name extraction prompt...")

        name_response = client.complete_with_image(
            prompt=name_prompt,
            image_url=image_url
        )

        print("📝 Extracted Name:")
        print("-" * 30)
        print(name_response)
        print("-" * 30)

        print("\n✅ All tests completed successfully!")
        print("🎉 GCS + LLM integration is working perfectly!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False


def test_storage_service_with_upload():
    """Test uploading a local image and then analyzing it"""

    print("\n📤 Testing Storage Service + LLM Pipeline")
    print("=" * 50)

    # Check if we have any local images to test with
    test_dirs = [
        Path("data/raw_images"),
        Path("examples"),
        Path(".")
    ]

    test_image = None
    for test_dir in test_dirs:
        if test_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images = list(test_dir.glob(ext))
                if images:
                    test_image = images[0]
                    break
        if test_image:
            break

    if not test_image:
        print("⚠️  No local test images found")
        print("💡 Place a .jpg or .png file in data/raw_images/ to test upload")
        return False

    print(f"🖼️  Found test image: {test_image}")

    try:
        # Test the storage service
        from aita.services.storage import create_storage_service

        print("📡 Creating storage service...")
        storage = create_storage_service()

        print("📤 Uploading test image...")
        uploaded_url = storage.upload_image(
            local_path=str(test_image),
            student_name="test_integration",
            assignment_name="api_test"
        )

        print(f"✅ Upload successful!")
        print(f"🔗 Uploaded URL: {uploaded_url}")

        # Now test with LLM
        print("\n🤖 Analyzing uploaded image with LLM...")
        client = create_openrouter_client()

        response = client.complete_with_image(
            prompt="Describe what you see in this image. Is it text, a drawing, a photo, or something else?",
            image_url=uploaded_url
        )

        print("📝 LLM Analysis of Uploaded Image:")
        print("-" * 30)
        print(response)
        print("-" * 30)

        print("✅ Complete pipeline test successful!")
        print("📊 Local Image → GCS Upload → LLM Analysis → Results")

        return True

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 AITA Integration Test")
    print("Testing the complete GCS + LLM pipeline")
    print()

    # Test 1: Existing GCS image with LLM
    success1 = test_gcs_image_with_llm()

    # Test 2: Upload + LLM pipeline (if possible)
    success2 = test_storage_service_with_upload()

    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   GCS Image + LLM: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Upload + LLM Pipeline: {'✅ PASS' if success2 else '❌ FAIL'}")

    if success1:
        print("\n🎯 Key Achievement: Your GCS images work perfectly with LLM API!")
        print("   - Image URLs are accessible")
        print("   - LLM can analyze the content")
        print("   - AITA prompts are working")
        print("   - Ready for full pipeline integration")

    if not success1:
        print("\n🔧 Setup needed:")
        print("   - Ensure OPENROUTER_API_KEY is set")
        print("   - Check internet connectivity")
        print("   - Verify GCS image URL is accessible")