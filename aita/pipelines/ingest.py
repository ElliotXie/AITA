from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import os
from datetime import datetime

from ..services.ocr import OCRService, create_ocr_service
from ..services.crop import ImageCropService, create_crop_service
from ..services.fuzzy import FuzzyMatchingService, create_fuzzy_matching_service
from ..services.first_page_detector import FirstPageDetector, create_first_page_detector
from ..services.storage import GoogleCloudStorageService, create_storage_service
from ..services.llm.openrouter import OpenRouterClient, create_openrouter_client
from ..utils.files import (
    find_images, read_student_roster, get_modification_time,
    create_student_directory, safe_filename, copy_file
)
from ..domain.models import Student

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessingResult:
    """Result of processing a single image."""
    image_path: Path
    is_first_page: bool
    detected_name: Optional[str]  # Optional name extraction for folder naming
    first_page_confidence: float
    processing_metadata: Dict[str, Any]
    timestamp: float


@dataclass
class IngestResults:
    """Results of the complete ingest pipeline."""
    total_images: int
    successfully_processed: int
    students_found: int
    grouped_images: Dict[str, List[Path]]
    unmatched_images: List[Path]
    processing_details: List[ImageProcessingResult]
    statistics: Dict[str, Any]


class IngestPipeline:
    def __init__(
        self,
        data_dir: Path,
        pages_per_student: int = 5,
        header_crop_percent: int = 30,
        name_crop_percent: int = 20,
        ocr_confidence: float = 0.5,
        try_name_extraction: bool = True,
        use_llm_for_names: bool = True
    ):
        """
        Initialize the ingest pipeline for smart image grouping based on first page detection.

        Args:
            data_dir: Base data directory containing raw_images, etc.
            pages_per_student: Expected number of pages per student
            header_crop_percent: Percentage of page top to analyze for first page patterns
            name_crop_percent: Percentage of page top to crop for name extraction
            ocr_confidence: OCR confidence threshold (0.0-1.0)
            try_name_extraction: Whether to try extracting names for better folder naming
            use_llm_for_names: Use LLM vision API for name extraction instead of OCR
        """
        self.data_dir = Path(data_dir)
        self.raw_images_dir = self.data_dir / "raw_images"
        self.grouped_dir = self.data_dir / "grouped"
        self.roster_file = self.data_dir / "roster.csv"

        self.pages_per_student = pages_per_student
        self.header_crop_percent = header_crop_percent
        self.name_crop_percent = name_crop_percent
        self.ocr_confidence = ocr_confidence
        self.try_name_extraction = try_name_extraction
        self.use_llm_for_names = use_llm_for_names

        # Initialize services
        self.first_page_detector = create_first_page_detector(
            ocr_confidence=ocr_confidence,
            crop_top_percent=header_crop_percent
        )

        # Services for name extraction
        if try_name_extraction:
            self.crop_service = create_crop_service(crop_top_percent=name_crop_percent)
            self.fuzzy_service = create_fuzzy_matching_service(similarity_threshold=80)

            if use_llm_for_names:
                # LLM vision-based name extraction
                self.storage_service = create_storage_service()
                self.llm_client = create_openrouter_client()
                logger.info("Using LLM vision API for name extraction")
            else:
                # Fallback to OCR-based name extraction
                self.ocr_service = create_ocr_service(confidence_threshold=ocr_confidence)
                logger.info("Using OCR for name extraction")

        # Ensure directories exist
        self._ensure_directories()

        logger.info(f"Initialized ingest pipeline:")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Pages per student: {pages_per_student}")
        logger.info(f"  Header analysis: top {header_crop_percent}%")
        logger.info(f"  Name extraction: {'LLM vision' if try_name_extraction and use_llm_for_names else 'OCR' if try_name_extraction else 'disabled'}")
        logger.info(f"  OCR confidence: {ocr_confidence}")

    def run_smart_grouping(self) -> IngestResults:
        """
        Execute the complete smart grouping pipeline.

        Returns:
            IngestResults containing all processing results and statistics
        """
        logger.info("Starting smart grouping pipeline...")

        try:
            # Step 1: Load student roster (optional for name matching)
            student_names = []
            if self.try_name_extraction:
                student_names = self._load_student_roster()

            # Step 2: Discover raw images
            image_paths = self._discover_images()
            if not image_paths:
                raise ValueError("No images found in raw_images directory")

            # Step 3: Process each image (detect first pages + optional name extraction)
            processing_results = self._process_images(image_paths)

            # Step 4: Group images based on first page detection + sequential ordering
            grouped_results = self._group_images_by_first_pages(processing_results)

            # Step 5: Organize into student folders
            organization_results = self._organize_student_folders(grouped_results)

            # Step 6: Generate statistics and results
            results = self._compile_results(processing_results, organization_results)

            logger.info(f"Smart grouping completed successfully:")
            logger.info(f"  Processed: {results.successfully_processed}/{results.total_images} images")
            logger.info(f"  Students found: {results.students_found}")
            logger.info(f"  Unmatched images: {len(results.unmatched_images)}")

            return results

        except Exception as e:
            logger.error(f"Smart grouping pipeline failed: {e}")
            raise

    def _load_student_roster(self) -> List[str]:
        """Load student roster from CSV file (optional for name matching)."""
        logger.info(f"Loading student roster from {self.roster_file}")

        if not self.roster_file.exists():
            logger.warning(f"Roster file not found: {self.roster_file} (proceeding without name matching)")
            return []

        # Read roster and extract names (handle CSV format)
        try:
            import csv
            student_names = []

            with open(self.roster_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try different possible column names for student names
                    name = None
                    for col in ['Student Name', 'Name', 'student_name', 'name']:
                        if col in row and row[col].strip():
                            name = row[col].strip()
                            break

                    if name:
                        student_names.append(name)

            if not student_names:
                # Fallback to simple line-by-line reading
                student_names = read_student_roster(self.roster_file)

            if student_names and hasattr(self, 'fuzzy_service'):
                self.fuzzy_service.load_student_roster(student_names)
                logger.info(f"Loaded {len(student_names)} students from roster")
            else:
                logger.warning("No students found in roster or fuzzy service not initialized")

            return student_names

        except Exception as e:
            logger.error(f"Failed to load student roster: {e} (proceeding without name matching)")
            return []

    def _discover_images(self) -> List[Path]:
        """Discover all image files in the raw images directory."""
        logger.info(f"Discovering images in {self.raw_images_dir}")

        if not self.raw_images_dir.exists():
            raise FileNotFoundError(f"Raw images directory not found: {self.raw_images_dir}")

        image_paths = list(find_images(self.raw_images_dir))

        # Sort by modification time for consistent ordering
        image_paths.sort(key=get_modification_time)

        logger.info(f"Found {len(image_paths)} images")
        if image_paths:
            logger.debug(f"First image: {image_paths[0]}")
            logger.debug(f"Last image: {image_paths[-1]}")

        return image_paths

    def _extract_name_with_llm(self, name_region_image, image_name: str) -> tuple[str, float, dict]:
        """
        Extract student name using LLM vision API.

        Args:
            name_region_image: PIL Image of the name region
            image_name: Name of the original image for logging

        Returns:
            Tuple of (extracted_name, confidence, metadata)
        """
        try:
            logger.debug(f"Extracting name using LLM vision for {image_name}")

            # Save name region to temporary file
            import tempfile
            temp_name = f"name_extraction_{image_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                name_region_image.save(temp_path, "PNG")

            # Upload the name region image to GCS
            public_url = self.storage_service.upload_image(
                local_path=temp_path,
                destination_path=f"temp_name_extraction/{temp_name}",
                assignment_name="name_extraction"
            )

            # Prepare prompt for name extraction
            extraction_prompt = """You are analyzing a cropped section from the top of an exam paper that contains a name field.

Your task is to extract the student's name from this image.

Instructions:
1. Look for the student's name that was handwritten in the "Name:" field
2. Ignore any printed text like "Name:", "Student ID:", "Exam", etc.
3. Focus only on the handwritten name
4. If you see multiple names, extract the primary student name
5. If no clear name is visible, respond with "UNKNOWN"

Please respond with just the student's name in this format:
NAME: [student name]

For example:
NAME: John Smith
or
NAME: UNKNOWN

What is the student's name in this image?"""

            # Call LLM vision API
            response = self.llm_client.complete_with_image(
                prompt=extraction_prompt,
                image_url=public_url
            )

            # Parse the response
            extracted_name = "UNKNOWN"
            confidence = 0.0

            if response and "NAME:" in response:
                name_part = response.split("NAME:")[-1].strip()
                if name_part and name_part != "UNKNOWN":
                    extracted_name = name_part
                    confidence = 0.95  # High confidence for LLM extraction

            # Clean up temporary files
            try:
                os.unlink(temp_path)  # Remove local temp file
                self.storage_service.delete_object(f"temp_name_extraction/{temp_name}")  # Remove from GCS
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")

            metadata = {
                "method": "llm_vision",
                "model": "google/gemini-2.0-flash-exp",
                "public_url": public_url,
                "raw_response": response,
                "temp_file": temp_name,
                "temp_path": temp_path
            }

            logger.debug(f"LLM extracted name: '{extracted_name}' (confidence: {confidence:.2f})")
            return extracted_name, confidence, metadata

        except Exception as e:
            logger.error(f"LLM name extraction failed for {image_name}: {e}")
            return "UNKNOWN", 0.0, {"error": str(e), "method": "llm_vision"}

    def _extract_name_with_ocr(self, name_region_image) -> tuple[str, float, dict]:
        """
        Extract student name using OCR (fallback method).

        Args:
            name_region_image: PIL Image of the name region

        Returns:
            Tuple of (extracted_name, confidence, metadata)
        """
        try:
            extracted_name, ocr_confidence = self.ocr_service.extract_student_name(name_region_image)
            metadata = {
                "method": "ocr",
                "ocr_confidence": ocr_confidence
            }
            return extracted_name, ocr_confidence, metadata
        except Exception as e:
            logger.error(f"OCR name extraction failed: {e}")
            return "UNKNOWN", 0.0, {"error": str(e), "method": "ocr"}

    def _process_images(self, image_paths: List[Path]) -> List[ImageProcessingResult]:
        """Process each image to detect first pages and optionally extract names."""
        logger.info(f"Processing {len(image_paths)} images for first page detection...")

        results = []
        first_pages_found = 0

        for i, image_path in enumerate(image_paths):
            try:
                logger.debug(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")

                # Step 1: Detect if this is a first page
                is_first_page, first_page_metadata = self.first_page_detector.is_first_page(image_path)

                # Step 2: Optional name extraction for better folder naming
                detected_name = None
                name_metadata = {}

                if is_first_page and self.try_name_extraction:
                    try:
                        # Crop name region
                        name_region = self.crop_service.crop_name_region(image_path)

                        # Extract name using LLM vision or OCR
                        if self.use_llm_for_names:
                            extracted_name, extraction_confidence, extraction_metadata = self._extract_name_with_llm(
                                name_region, image_path.name
                            )
                        else:
                            extracted_name, extraction_confidence, extraction_metadata = self._extract_name_with_ocr(
                                name_region
                            )

                        # Try fuzzy match to roster if available
                        if hasattr(self, 'fuzzy_service') and self.fuzzy_service.student_roster and extracted_name != "UNKNOWN":
                            matched_student, match_confidence, match_metadata = self.fuzzy_service.find_best_match(
                                extracted_name,
                                return_all_candidates=False
                            )
                            if matched_student:
                                detected_name = matched_student
                                logger.debug(f"  -> Name matched: {extracted_name} -> {matched_student}")
                            else:
                                # If LLM extracted a name but fuzzy match failed, still use the LLM name
                                if self.use_llm_for_names and extracted_name != "UNKNOWN":
                                    detected_name = extracted_name
                                    logger.debug(f"  -> Using LLM name (no fuzzy match): {extracted_name}")
                                else:
                                    logger.debug(f"  -> Name not matched: {extracted_name} (confidence: {match_confidence})")

                            name_metadata = {
                                "extracted_name": extracted_name,
                                "extraction_confidence": extraction_confidence,
                                "extraction_metadata": extraction_metadata,
                                "match_confidence": match_confidence,
                                "match_metadata": match_metadata
                            }
                        else:
                            # No roster available or no name extracted, use extracted name if reasonable
                            if extracted_name and extracted_name != "UNKNOWN" and len(extracted_name) > 2:
                                detected_name = extracted_name
                                logger.debug(f"  -> Using extracted name: {extracted_name}")

                            name_metadata = {
                                "extracted_name": extracted_name,
                                "extraction_confidence": extraction_confidence,
                                "extraction_metadata": extraction_metadata,
                                "note": "No roster matching attempted or no name extracted"
                            }

                    except Exception as e:
                        logger.warning(f"Name extraction failed for {image_path.name}: {e}")
                        name_metadata = {"name_extraction_error": str(e)}

                # Get timestamp
                timestamp = get_modification_time(image_path)

                # Compile processing metadata
                processing_metadata = {
                    "first_page_detection": first_page_metadata,
                    "name_extraction": name_metadata,
                    "header_crop_percent": self.header_crop_percent,
                    "name_crop_percent": self.name_crop_percent
                }

                result = ImageProcessingResult(
                    image_path=image_path,
                    is_first_page=is_first_page,
                    detected_name=detected_name,
                    first_page_confidence=first_page_metadata.get("confidence_score", 0.0),
                    processing_metadata=processing_metadata,
                    timestamp=timestamp
                )

                results.append(result)

                if is_first_page:
                    first_pages_found += 1
                    name_info = f" (name: {detected_name})" if detected_name else ""
                    logger.debug(f"  -> FIRST PAGE detected{name_info}")
                else:
                    logger.debug(f"  -> Regular page")

            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                # Create a failed result
                result = ImageProcessingResult(
                    image_path=image_path,
                    is_first_page=False,
                    detected_name=None,
                    first_page_confidence=0.0,
                    processing_metadata={"error": str(e)},
                    timestamp=get_modification_time(image_path)
                )
                results.append(result)

        logger.info(f"First page detection: {first_pages_found} first pages found out of {len(image_paths)} images")
        return results

    def _group_images_by_first_pages(self, processing_results: List[ImageProcessingResult]) -> Dict[str, List[ImageProcessingResult]]:
        """Group images based on first page detection and sequential ordering."""
        logger.info("Grouping images by first page detection...")

        # Sort by timestamp first to get chronological order
        sorted_results = sorted(processing_results, key=lambda x: x.timestamp)

        # Find all first pages
        first_page_indices = []
        for i, result in enumerate(sorted_results):
            if result.is_first_page:
                first_page_indices.append(i)

        logger.info(f"Found {len(first_page_indices)} first pages at indices: {first_page_indices}")

        # Group images based on first pages + sequential pages
        grouped = {}
        student_counter = 1

        for i, first_page_index in enumerate(first_page_indices):
            # Determine the range of pages for this student
            start_index = first_page_index

            # End index is either the start of the next student or the end of images
            if i + 1 < len(first_page_indices):
                end_index = first_page_indices[i + 1]
            else:
                # For the last student, take remaining images up to pages_per_student
                end_index = min(start_index + self.pages_per_student, len(sorted_results))

            # Extract this student's pages
            student_pages = sorted_results[start_index:end_index]

            # Determine student folder name
            first_page_result = student_pages[0]
            if first_page_result.detected_name:
                student_name = first_page_result.detected_name
            else:
                student_name = f"Student_{student_counter:03d}"

            # Handle duplicate names by adding a counter
            original_name = student_name
            counter = 1
            while student_name in grouped:
                counter += 1
                student_name = f"{original_name}_{counter}"

            grouped[student_name] = student_pages

            logger.info(f"Student {student_counter}: '{student_name}' - {len(student_pages)} pages")
            logger.debug(f"  Pages: {[p.image_path.name for p in student_pages]}")

            student_counter += 1

        # Handle any remaining images that weren't grouped
        if first_page_indices:
            last_grouped_index = first_page_indices[-1] + min(self.pages_per_student,
                                                              len(sorted_results) - first_page_indices[-1])
            if last_grouped_index < len(sorted_results):
                remaining_images = sorted_results[last_grouped_index:]
                if remaining_images:
                    grouped[f"Remaining_Images"] = remaining_images
                    logger.warning(f"Found {len(remaining_images)} ungrouped images at the end")
        else:
            # No first pages found - group all images as unknown students
            logger.warning("No first pages detected! Grouping all images by pages_per_student")
            for i in range(0, len(sorted_results), self.pages_per_student):
                student_pages = sorted_results[i:i + self.pages_per_student]
                student_name = f"Unknown_Student_{(i // self.pages_per_student) + 1:03d}"
                grouped[student_name] = student_pages
                logger.info(f"Unknown group: '{student_name}' - {len(student_pages)} pages")

        logger.info(f"Grouped {len(sorted_results)} images into {len(grouped)} student folders")
        return grouped

    def _organize_student_folders(self, grouped_results: Dict[str, List[ImageProcessingResult]]) -> Dict[str, List[Path]]:
        """Organize images into student folders."""
        logger.info("Organizing images into student folders...")

        # Clear existing grouped directory
        if self.grouped_dir.exists():
            import shutil
            shutil.rmtree(self.grouped_dir)

        self.grouped_dir.mkdir(parents=True, exist_ok=True)

        organized_paths = {}

        for student_name, results in grouped_results.items():
            try:
                # Create student directory
                student_dir = create_student_directory(self.grouped_dir, student_name)

                # Copy images to student folder
                student_image_paths = []
                for i, result in enumerate(results):
                    # Generate new filename with page number
                    page_num = i + 1
                    original_name = result.image_path.name
                    extension = result.image_path.suffix

                    new_filename = f"page_{page_num:03d}_{safe_filename(original_name)}"
                    new_path = student_dir / new_filename

                    # Copy the image
                    copied_path = copy_file(result.image_path, new_path)
                    student_image_paths.append(copied_path)

                    logger.debug(f"Copied: {original_name} -> {student_name}/{new_filename}")

                organized_paths[student_name] = student_image_paths

                # Save processing metadata for this student
                metadata_file = student_dir / "processing_metadata.json"
                student_metadata = {
                    "student_name": student_name,
                    "image_count": len(results),
                    "processing_timestamp": datetime.now().isoformat(),
                    "images": [
                        {
                            "original_path": str(result.image_path),
                            "new_path": str(new_path),
                            "is_first_page": result.is_first_page,
                            "detected_name": result.detected_name,
                            "first_page_confidence": result.first_page_confidence,
                            "timestamp": result.timestamp,
                            "metadata": result.processing_metadata
                        }
                        for result, new_path in zip(results, student_image_paths)
                    ]
                }

                with open(metadata_file, 'w') as f:
                    json.dump(student_metadata, f, indent=2, default=str)

                logger.info(f"Organized {len(results)} images for student: {student_name}")

            except Exception as e:
                logger.error(f"Failed to organize images for student {student_name}: {e}")
                continue

        return organized_paths

    def _compile_results(self, processing_results: List[ImageProcessingResult], organized_paths: Dict[str, List[Path]]) -> IngestResults:
        """Compile final results and statistics."""
        total_images = len(processing_results)
        successfully_processed = sum(1 for r in processing_results if r.is_first_page or r.detected_name is not None)
        students_found = len([k for k in organized_paths.keys() if not k.startswith("Unknown_") and k != "Remaining_Images"])

        unmatched_images = []
        grouped_images = {}

        for student, paths in organized_paths.items():
            if student.startswith("Unknown_") or student == "Remaining_Images":
                unmatched_images.extend(paths)
            else:
                grouped_images[student] = paths

        # Generate detailed statistics
        confidences = [r.first_page_confidence for r in processing_results if r.first_page_confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        statistics = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_images": total_images,
            "successfully_matched": successfully_processed,
            "success_rate": (successfully_processed / total_images * 100) if total_images > 0 else 0,
            "students_found": students_found,
            "unmatched_count": len(unmatched_images),
            "average_confidence": avg_confidence,
            "pages_per_student": self.pages_per_student,
            "header_crop_percentage": self.header_crop_percent,
            "name_crop_percentage": self.name_crop_percent,
            "confidence_distribution": {
                "high (90-100)": sum(1 for c in confidences if c >= 90),
                "medium (70-89)": sum(1 for c in confidences if 70 <= c < 90),
                "low (0-69)": sum(1 for c in confidences if c < 70)
            }
        }

        return IngestResults(
            total_images=total_images,
            successfully_processed=successfully_processed,
            students_found=students_found,
            grouped_images=grouped_images,
            unmatched_images=unmatched_images,
            processing_details=processing_results,
            statistics=statistics
        )

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        required_dirs = [
            self.data_dir,
            self.raw_images_dir,
            self.grouped_dir
        ]

        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def generate_summary_report(self, results: IngestResults) -> str:
        """Generate a human-readable summary report."""
        report_lines = [
            "=" * 60,
            "AITA SMART GROUPING PIPELINE - SUMMARY REPORT",
            "=" * 60,
            "",
            f"Processing Date: {results.statistics['processing_timestamp']}",
            f"Data Directory: {self.data_dir}",
            "",
            "OVERVIEW:",
            f"  Total Images: {results.total_images}",
            f"  Successfully Matched: {results.successfully_processed}",
            f"  Success Rate: {results.statistics['success_rate']:.1f}%",
            f"  Students Found: {results.students_found}",
            f"  Unmatched Images: {len(results.unmatched_images)}",
            "",
            "CONFIGURATION:",
            f"  Pages per Student: {self.pages_per_student}",
            f"  Header Analysis Region: Top {self.header_crop_percent}%",
            f"  Name Crop Region: Top {self.name_crop_percent}%",
            f"  OCR Confidence: {self.ocr_confidence}",
            f"  Name Extraction: {'Enabled' if self.try_name_extraction else 'Disabled'}",
            "",
            "CONFIDENCE DISTRIBUTION:",
            f"  High (90-100): {results.statistics['confidence_distribution']['high (90-100)']} images",
            f"  Medium (70-89): {results.statistics['confidence_distribution']['medium (70-89)']} images",
            f"  Low (0-69): {results.statistics['confidence_distribution']['low (0-69)']} images",
            f"  Average: {results.statistics['average_confidence']:.1f}",
            "",
            "STUDENTS FOUND:",
        ]

        for student, images in results.grouped_images.items():
            report_lines.append(f"  {student}: {len(images)} images")

        if results.unmatched_images:
            report_lines.extend([
                "",
                "UNMATCHED IMAGES:",
                f"  {len(results.unmatched_images)} images could not be matched to students"
            ])

        report_lines.extend([
            "",
            "=" * 60
        ])

        return "\n".join(report_lines)


def create_ingest_pipeline(
    data_dir: Path,
    pages_per_student: int = 5,
    header_crop_percent: int = 30,
    name_crop_percent: int = 20,
    ocr_confidence: float = 0.5,
    try_name_extraction: bool = True,
    use_llm_for_names: bool = True
) -> IngestPipeline:
    """Factory function to create an IngestPipeline instance."""
    return IngestPipeline(
        data_dir=data_dir,
        pages_per_student=pages_per_student,
        header_crop_percent=header_crop_percent,
        name_crop_percent=name_crop_percent,
        ocr_confidence=ocr_confidence,
        try_name_extraction=try_name_extraction,
        use_llm_for_names=use_llm_for_names
    )