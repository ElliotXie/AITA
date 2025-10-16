#!/usr/bin/env python3
"""
Test script for the improved AITA smart grouping ingest pipeline.
This version uses first page detection instead of relying solely on name extraction.
"""

import logging
from pathlib import Path
from aita.pipelines.ingest import create_ingest_pipeline
from aita.utils.logging import setup_logging


def main():
    """Run the improved smart grouping pipeline test."""
    # Set up logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("AITA IMPROVED SMART GROUPING PIPELINE - TEST RUN")
    print("=" * 80)
    print("🎯 NEW APPROACH: First page detection + sequential grouping")
    print("📋 Strategy: Detect exam headers → Group next N pages → Try name matching")
    print()

    # Configure paths
    data_dir = Path("data")

    # Verify required files exist
    raw_images_dir = data_dir / "raw_images"

    if not raw_images_dir.exists():
        print(f"❌ ERROR: Raw images directory not found: {raw_images_dir}")
        return False

    print(f"📁 Data directory: {data_dir.absolute()}")
    print(f"🖼️  Raw images: {raw_images_dir}")
    print()

    try:
        # Create and configure the improved ingest pipeline
        print("🔧 Initializing improved ingest pipeline...")
        pipeline = create_ingest_pipeline(
            data_dir=data_dir,
            pages_per_student=5,        # Assume 5 pages per student
            header_crop_percent=30,     # Look at top 30% for exam headers/patterns
            name_crop_percent=20,       # Crop top 20% for name extraction
            ocr_confidence=0.5,         # 50% OCR confidence threshold
            try_name_extraction=True    # Try to extract names for better folder naming
        )
        print("✅ Pipeline initialized successfully")

        # Run the smart grouping process
        print("\n🚀 Starting improved smart grouping process...")
        print("-" * 40)

        results = pipeline.run_smart_grouping()

        print("\n✅ Smart grouping completed!")
        print("-" * 40)

        # Display results summary
        print("\n📊 RESULTS SUMMARY:")
        print(f"   Total images processed: {results.total_images}")
        print(f"   Successfully processed: {results.successfully_processed}")
        print(f"   Student groups created: {results.students_found}")
        print(f"   Success rate: {results.statistics.get('success_rate', 0):.1f}%")

        # Show grouped students
        print(f"\n👥 STUDENT GROUPS CREATED ({len(results.grouped_images)}):")
        for student_name, images in results.grouped_images.items():
            print(f"   📄 {student_name}: {len(images)} pages")

        # Show unmatched images if any
        if results.unmatched_images:
            print(f"\n❓ UNMATCHED IMAGES ({len(results.unmatched_images)}):")
            for img_path in results.unmatched_images:
                print(f"   🖼️  {img_path.name}")

        # Display first page detection statistics
        print(f"\n🔍 FIRST PAGE DETECTION ANALYSIS:")
        first_pages_detected = 0
        total_confidence = 0
        patterns_found = []

        for detail in results.processing_details:
            if detail.is_first_page:
                first_pages_detected += 1
                total_confidence += detail.first_page_confidence

                # Extract patterns found
                fp_metadata = detail.processing_metadata.get("first_page_detection", {})
                if "patterns_found" in fp_metadata:
                    patterns_found.extend(fp_metadata["patterns_found"])

        if first_pages_detected > 0:
            avg_confidence = total_confidence / first_pages_detected
            print(f"   📈 First pages detected: {first_pages_detected}")
            print(f"   📈 Average confidence: {avg_confidence:.1f}")

            # Show most common patterns
            if patterns_found:
                pattern_counts = {}
                for pattern in patterns_found:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

                top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"   📈 Top patterns found:")
                for pattern, count in top_patterns:
                    print(f"      - '{pattern}': {count} times")
        else:
            print(f"   ❌ No first pages detected")

        # Display name extraction results
        if pipeline.try_name_extraction:
            print(f"\n👤 NAME EXTRACTION ANALYSIS:")
            names_extracted = sum(1 for detail in results.processing_details
                                if detail.detected_name is not None)
            print(f"   📝 Names successfully extracted/matched: {names_extracted}")

            for detail in results.processing_details:
                if detail.detected_name:
                    name_data = detail.processing_metadata.get("name_extraction", {})
                    extracted = name_data.get("extracted_name", "N/A")
                    print(f"      - {detail.image_path.name}: '{extracted}' → '{detail.detected_name}'")

        # Generate and display detailed report
        print("\n📄 DETAILED REPORT:")
        print("-" * 40)
        report = pipeline.generate_summary_report(results)
        print(report)

        # Save results to file
        grouped_dir = data_dir / "grouped"
        print(f"\n💾 Results saved to: {grouped_dir.absolute()}")
        print("   Each student folder contains their exam pages and processing metadata.")

        print(f"\n🎉 Improved smart grouping pipeline test completed successfully!")

        # Show next steps
        print(f"\n🔄 NEXT STEPS:")
        print(f"   1. Review the grouped folders to verify accuracy")
        print(f"   2. Adjust header_crop_percent if first page detection needs improvement")
        print(f"   3. Adjust pages_per_student based on your actual exam length")
        print(f"   4. The pipeline is ready for production use!")

        return True

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)