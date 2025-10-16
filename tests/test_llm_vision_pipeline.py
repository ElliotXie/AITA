#!/usr/bin/env python3
"""
Test script for AITA smart grouping pipeline with LLM vision name extraction.
This version uses:
1. EasyOCR for first page detection (find "Name:", "Exam" patterns)
2. Gemini 2.5 Flash vision API for precise name extraction from first pages
"""

import logging
from pathlib import Path
from aita.pipelines.ingest import create_ingest_pipeline
from aita.utils.logging import setup_logging


def main():
    """Run the enhanced smart grouping pipeline test with LLM vision."""
    # Set up logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("AITA ENHANCED SMART GROUPING - LLM VISION NAME EXTRACTION")
    print("=" * 80)
    print("🎯 HYBRID APPROACH:")
    print("   📋 EasyOCR → Detect first pages (find 'Name:', 'Exam' patterns)")
    print("   🤖 Gemini 2.5 Flash → Extract actual student names from first pages")
    print("   📁 Group sequential pages by student")
    print("📁 All results saved to tests/ folder structure")
    print()

    # Configure paths
    data_dir = Path("data")
    test_output_dir = Path("tests") / "llm_vision_output"

    # Create test output directory
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Verify required files exist
    raw_images_dir = data_dir / "raw_images"

    if not raw_images_dir.exists():
        print(f"❌ ERROR: Raw images directory not found: {raw_images_dir}")
        return False

    print(f"📁 Data directory: {data_dir.absolute()}")
    print(f"🖼️  Raw images: {raw_images_dir}")
    print(f"🧪 Test output: {test_output_dir.absolute()}")
    print()

    try:
        # Create and configure the enhanced ingest pipeline
        print("🔧 Initializing enhanced ingest pipeline with LLM vision...")
        pipeline = create_ingest_pipeline(
            data_dir=data_dir,
            pages_per_student=5,        # Assume 5 pages per student
            header_crop_percent=30,     # EasyOCR: Look at top 30% for first page patterns
            name_crop_percent=20,       # LLM: Analyze top 20% for name extraction
            ocr_confidence=0.5,         # 50% OCR confidence threshold
            try_name_extraction=True,   # Enable name extraction
            use_llm_for_names=True      # 🚀 Use LLM vision for name extraction
        )
        print("✅ Pipeline initialized successfully")
        print("   🔍 First page detection: EasyOCR (patterns: 'Name:', 'Exam')")
        print("   🤖 Name extraction: Gemini 2.5 Flash vision API")
        print("   ☁️  Image processing: Google Cloud Storage + OpenRouter")

        # Run the enhanced smart grouping process
        print("\n🚀 Starting enhanced smart grouping process...")
        print("-" * 50)

        results = pipeline.run_smart_grouping()

        print("\n✅ Enhanced smart grouping completed!")
        print("-" * 50)

        # Display results summary
        print("\n📊 RESULTS SUMMARY:")
        print(f"   Total images processed: {results.total_images}")
        print(f"   Successfully processed: {results.successfully_processed}")
        print(f"   Student groups created: {results.students_found}")
        print(f"   Success rate: {results.statistics.get('success_rate', 0):.1f}%")

        # Show grouped students with name extraction details
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

        for detail in results.processing_details:
            if detail.is_first_page:
                first_pages_detected += 1
                total_confidence += detail.first_page_confidence

        if first_pages_detected > 0:
            avg_confidence = total_confidence / first_pages_detected
            print(f"   📈 First pages detected: {first_pages_detected}")
            print(f"   📈 Average confidence: {avg_confidence:.1f}")
        else:
            print(f"   ❌ No first pages detected")

        # Display LLM name extraction results
        print(f"\n🤖 LLM VISION NAME EXTRACTION ANALYSIS:")
        llm_extractions = 0
        successful_llm_extractions = 0

        for detail in results.processing_details:
            if detail.is_first_page:
                name_data = detail.processing_metadata.get("name_extraction", {})
                extraction_metadata = name_data.get("extraction_metadata", {})

                if extraction_metadata.get("method") == "llm_vision":
                    llm_extractions += 1
                    extracted_name = name_data.get("extracted_name", "UNKNOWN")

                    if extracted_name != "UNKNOWN":
                        successful_llm_extractions += 1
                        print(f"   ✅ {detail.image_path.name}: '{extracted_name}'")

                        # Show LLM response details
                        raw_response = extraction_metadata.get("raw_response", "")
                        if raw_response:
                            response_preview = raw_response[:100] + "..." if len(raw_response) > 100 else raw_response
                            print(f"      🤖 LLM response: {response_preview}")
                    else:
                        print(f"   ❌ {detail.image_path.name}: No name extracted")

        if llm_extractions > 0:
            success_rate = successful_llm_extractions / llm_extractions * 100
            print(f"   📊 LLM extraction success: {successful_llm_extractions}/{llm_extractions} ({success_rate:.1f}%)")
        else:
            print(f"   ℹ️  No LLM extractions attempted (no first pages or LLM disabled)")

        # Generate and display detailed report
        print("\n📄 DETAILED REPORT:")
        print("-" * 50)
        report = pipeline.generate_summary_report(results)
        print(report)

        # Save results to test output directory
        grouped_dir = data_dir / "grouped"
        print(f"\n💾 RESULTS SAVED:")
        print(f"   📁 Grouped student folders: {grouped_dir.absolute()}")
        print(f"   🧪 Test output directory: {test_output_dir.absolute()}")
        print("   📊 Each student folder contains pages + processing metadata with LLM details")

        # Copy some diagnostic info to test output
        if grouped_dir.exists():
            import shutil
            test_grouped_dir = test_output_dir / "grouped_results"
            if test_grouped_dir.exists():
                shutil.rmtree(test_grouped_dir)
            shutil.copytree(grouped_dir, test_grouped_dir)
            print(f"   📋 Copied results to: {test_grouped_dir}")

        print(f"\n🎉 Enhanced pipeline test completed successfully!")

        # Show next steps
        print(f"\n🔄 NEXT STEPS:")
        print(f"   1. Check {grouped_dir} for organized student folders")
        print(f"   2. Review processing_metadata.json in each folder for LLM extraction details")
        print(f"   3. Verify LLM successfully extracted actual student names")
        print(f"   4. The hybrid pipeline is ready for production use!")

        return True

    except Exception as e:
        logger.error(f"Enhanced pipeline test failed: {e}")
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)