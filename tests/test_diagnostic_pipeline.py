#!/usr/bin/env python3
"""
Diagnostic test for AITA smart grouping pipeline.
Shows exactly what's happening: cropped images, OCR results, and detection logic.
All outputs saved in tests/ folder for inspection.
"""

import logging
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

from aita.services.first_page_detector import create_first_page_detector
from aita.services.ocr import create_ocr_service
from aita.services.crop import create_crop_service
from aita.utils.files import find_images, get_modification_time
from aita.utils.logging import setup_logging


def main():
    """Run comprehensive diagnostic test."""
    # Set up logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("AITA DIAGNOSTIC TEST - DETAILED ANALYSIS")
    print("=" * 80)
    print("🔍 This test shows you exactly what the pipeline sees and does")
    print("📁 All results will be saved in tests/diagnostic_output/")
    print()

    # Setup paths
    data_dir = Path("data")
    raw_images_dir = data_dir / "raw_images"
    test_output_dir = Path("tests") / "diagnostic_output"

    # Create output directories
    test_output_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir = test_output_dir / "cropped_images"
    cropped_dir.mkdir(exist_ok=True)
    header_crop_dir = test_output_dir / "header_regions"
    header_crop_dir.mkdir(exist_ok=True)
    name_crop_dir = test_output_dir / "name_regions"
    name_crop_dir.mkdir(exist_ok=True)

    print(f"📁 Test output directory: {test_output_dir.absolute()}")
    print()

    if not raw_images_dir.exists():
        print(f"❌ ERROR: Raw images directory not found: {raw_images_dir}")
        return False

    try:
        # Initialize services
        print("🔧 Initializing services...")
        first_page_detector = create_first_page_detector(
            ocr_confidence=0.5,
            crop_top_percent=30  # Header analysis region
        )

        ocr_service = create_ocr_service(confidence_threshold=0.5)
        header_crop_service = create_crop_service(crop_top_percent=30)  # For first page detection
        name_crop_service = create_crop_service(crop_top_percent=20)    # For name extraction

        # Discover images
        print("🖼️  Discovering images...")
        image_paths = list(find_images(raw_images_dir))
        image_paths.sort(key=get_modification_time)

        print(f"Found {len(image_paths)} images:")
        for i, img_path in enumerate(image_paths):
            print(f"  {i+1:2d}. {img_path.name}")
        print()

        # Process each image with detailed diagnostics
        diagnostic_results = []

        for i, image_path in enumerate(image_paths):
            print(f"🔍 ANALYZING IMAGE {i+1}/{len(image_paths)}: {image_path.name}")
            print("-" * 60)

            result = {
                "image_index": i + 1,
                "image_name": image_path.name,
                "timestamp": get_modification_time(image_path),
                "analysis": {}
            }

            try:
                # Step 1: Crop header region for first page detection
                print("📋 Step 1: Cropping header region (top 30%) for first page detection...")
                header_region = header_crop_service.crop_name_region(image_path)
                header_crop_path = header_crop_dir / f"{i+1:02d}_header_{image_path.stem}.png"
                header_region.save(header_crop_path)
                print(f"   ✅ Saved header crop: {header_crop_path.name}")

                # Step 2: OCR on header region
                print("🔤 Step 2: Running OCR on header region...")
                header_ocr_results = ocr_service.extract_text_from_pil(header_region, detail=1)

                # Extract all text for analysis
                header_text_parts = [r["text"] for r in header_ocr_results if r["text"].strip()]
                header_full_text = " ".join(header_text_parts).lower()

                print(f"   📝 OCR extracted {len(header_ocr_results)} text elements:")
                for j, ocr_result in enumerate(header_ocr_results[:5]):  # Show first 5
                    text = ocr_result["text"][:50] + "..." if len(ocr_result["text"]) > 50 else ocr_result["text"]
                    print(f"      {j+1}. '{text}' (conf: {ocr_result['confidence']:.2f})")
                if len(header_ocr_results) > 5:
                    print(f"      ... and {len(header_ocr_results) - 5} more")

                result["analysis"]["header_ocr"] = {
                    "elements_found": len(header_ocr_results),
                    "full_text": header_full_text,
                    "detailed_results": header_ocr_results
                }

                # Step 3: First page detection
                print("🎯 Step 3: Running first page detection...")
                is_first_page, first_page_metadata = first_page_detector.is_first_page(image_path)

                patterns_found = first_page_metadata.get("patterns_found", [])
                confidence = first_page_metadata.get("confidence_score", 0)

                print(f"   🎯 First page detected: {'YES' if is_first_page else 'NO'}")
                print(f"   📊 Confidence score: {confidence:.1f}")
                print(f"   🔍 Patterns found: {patterns_found}")

                result["analysis"]["first_page_detection"] = {
                    "is_first_page": is_first_page,
                    "confidence": confidence,
                    "patterns_found": patterns_found,
                    "metadata": first_page_metadata
                }

                # Step 4: Name extraction (if first page)
                if is_first_page:
                    print("👤 Step 4: Attempting name extraction (first page detected)...")

                    # Crop name region (top 20%)
                    name_region = name_crop_service.crop_name_region(image_path)
                    name_crop_path = name_crop_dir / f"{i+1:02d}_name_{image_path.stem}.png"
                    name_region.save(name_crop_path)
                    print(f"   ✅ Saved name crop: {name_crop_path.name}")

                    # Extract name
                    extracted_name, name_confidence = ocr_service.extract_student_name(name_region)
                    print(f"   👤 Extracted name: '{extracted_name}' (conf: {name_confidence:.2f})")

                    result["analysis"]["name_extraction"] = {
                        "extracted_name": extracted_name,
                        "confidence": name_confidence,
                        "name_crop_saved": str(name_crop_path)
                    }
                else:
                    print("⏭️  Step 4: Skipping name extraction (not a first page)")
                    result["analysis"]["name_extraction"] = {
                        "skipped": "Not a first page"
                    }

                # Step 5: Show image dimensions and crop info
                with Image.open(image_path) as img:
                    width, height = img.size
                    header_height = int(height * 0.30)
                    name_height = int(height * 0.20)

                print(f"📐 Step 5: Image dimensions and crop info:")
                print(f"   📏 Original size: {width} x {height} pixels")
                print(f"   📋 Header region: {width} x {header_height} pixels (top 30%)")
                print(f"   👤 Name region: {width} x {name_height} pixels (top 20%)")

                result["analysis"]["image_info"] = {
                    "original_size": [width, height],
                    "header_crop_size": [width, header_height],
                    "name_crop_size": [width, name_height],
                    "header_crop_saved": str(header_crop_path)
                }

            except Exception as e:
                print(f"   ❌ ERROR processing {image_path.name}: {e}")
                result["analysis"]["error"] = str(e)

            diagnostic_results.append(result)
            print()

        # Generate summary report
        print("📊 DIAGNOSTIC SUMMARY:")
        print("=" * 60)

        first_pages = [r for r in diagnostic_results if r["analysis"].get("first_page_detection", {}).get("is_first_page", False)]
        all_patterns = []
        for r in first_pages:
            patterns = r["analysis"]["first_page_detection"].get("patterns_found", [])
            all_patterns.extend(patterns)

        print(f"📈 Total images analyzed: {len(diagnostic_results)}")
        print(f"📈 First pages detected: {len(first_pages)}")
        print(f"📈 Detection rate: {len(first_pages)/len(diagnostic_results)*100:.1f}%")

        if all_patterns:
            pattern_counts = {}
            for pattern in all_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            print(f"📈 Most common patterns found:")
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   - '{pattern}': {count} times")

        # Save detailed diagnostic report
        report_file = test_output_dir / "diagnostic_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                "test_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_images": len(diagnostic_results),
                    "first_pages_detected": len(first_pages),
                    "detection_rate": len(first_pages)/len(diagnostic_results)*100 if diagnostic_results else 0,
                    "pattern_counts": pattern_counts if all_patterns else {}
                },
                "detailed_results": diagnostic_results
            }, f, indent=2, default=str)

        print(f"\n💾 Detailed diagnostic report saved: {report_file}")

        # Show what files were created
        print(f"\n📁 FILES CREATED FOR INSPECTION:")
        print(f"   📋 Header crops (30%): {header_crop_dir}")
        for file in sorted(header_crop_dir.glob("*.png")):
            print(f"      - {file.name}")

        if any(r["analysis"].get("first_page_detection", {}).get("is_first_page", False) for r in diagnostic_results):
            print(f"   👤 Name crops (20%): {name_crop_dir}")
            for file in sorted(name_crop_dir.glob("*.png")):
                print(f"      - {file.name}")

        print(f"   📊 JSON report: {report_file.name}")

        print(f"\n🎉 Diagnostic test completed!")
        print(f"📁 Check {test_output_dir} to see exactly what the pipeline sees")

        return True

    except Exception as e:
        logger.error(f"Diagnostic test failed: {e}")
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)