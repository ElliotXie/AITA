#!/usr/bin/env python3
"""
Test script for the AITA smart grouping ingest pipeline.
This script demonstrates the complete workflow from raw images to organized student folders.
"""

import logging
from pathlib import Path
from aita.pipelines.ingest import create_ingest_pipeline
from aita.utils.logging import setup_logging


def main():
    """Run the smart grouping pipeline test."""
    # Set up logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("AITA SMART GROUPING PIPELINE - TEST RUN")
    print("=" * 80)

    # Configure paths
    data_dir = Path("data")

    # Verify required files exist
    raw_images_dir = data_dir / "raw_images"
    roster_file = data_dir / "roster.csv"

    if not raw_images_dir.exists():
        print(f"❌ ERROR: Raw images directory not found: {raw_images_dir}")
        return False

    if not roster_file.exists():
        print(f"❌ ERROR: Roster file not found: {roster_file}")
        return False

    print(f"📁 Data directory: {data_dir.absolute()}")
    print(f"🖼️  Raw images: {raw_images_dir}")
    print(f"📋 Student roster: {roster_file}")
    print()

    try:
        # Create and configure the ingest pipeline
        print("🔧 Initializing ingest pipeline...")
        pipeline = create_ingest_pipeline(
            data_dir=data_dir,
            crop_top_percent=20,      # Crop top 20% for name extraction
            pages_per_student=5,      # Assume 5 pages per student
            similarity_threshold=80,  # 80% fuzzy match threshold
            ocr_confidence=0.5        # 50% OCR confidence threshold
        )
        print("✅ Pipeline initialized successfully")

        # Run the smart grouping process
        print("\n🚀 Starting smart grouping process...")
        print("-" * 40)

        results = pipeline.run_smart_grouping()

        print("\n✅ Smart grouping completed!")
        print("-" * 40)

        # Display results summary
        print("\n📊 RESULTS SUMMARY:")
        print(f"   Total images processed: {results.total_images}")
        print(f"   Successfully matched: {results.successfully_processed}")
        print(f"   Success rate: {results.statistics['success_rate']:.1f}%")
        print(f"   Students found: {results.students_found}")
        print(f"   Unmatched images: {len(results.unmatched_images)}")
        print(f"   Average confidence: {results.statistics['average_confidence']:.1f}")

        # Show confidence distribution
        conf_dist = results.statistics['confidence_distribution']
        print(f"\n📈 CONFIDENCE DISTRIBUTION:")
        print(f"   High (90-100): {conf_dist['high (90-100)']} images")
        print(f"   Medium (70-89): {conf_dist['medium (70-89)']} images")
        print(f"   Low (0-69): {conf_dist['low (0-69)']} images")

        # Show grouped students
        print(f"\n👥 STUDENTS FOUND ({len(results.grouped_images)}):")
        for student_name, images in results.grouped_images.items():
            print(f"   📄 {student_name}: {len(images)} pages")

        # Show unmatched images if any
        if results.unmatched_images:
            print(f"\n❓ UNMATCHED IMAGES ({len(results.unmatched_images)}):")
            for img_path in results.unmatched_images:
                print(f"   🖼️  {img_path.name}")

        # Generate and display detailed report
        print("\n📄 DETAILED REPORT:")
        print("-" * 40)
        report = pipeline.generate_summary_report(results)
        print(report)

        # Save results to file
        grouped_dir = data_dir / "grouped"
        print(f"\n💾 Results saved to: {grouped_dir.absolute()}")
        print("   Each student folder contains their exam pages and processing metadata.")

        print(f"\n🎉 Smart grouping pipeline test completed successfully!")
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