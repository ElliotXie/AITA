from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from PIL import Image

from .ocr import OCRService, create_ocr_service
from .crop import ImageCropService, create_crop_service

logger = logging.getLogger(__name__)


class FirstPageDetector:
    """
    Service to detect first pages of exams based on printed text patterns.

    Looks for common first page indicators like:
    - "Name:" field
    - "Student ID:" field
    - "Date:" field
    - Exam title/header text
    - Form field patterns
    """

    def __init__(
        self,
        ocr_confidence: float = 0.5,
        crop_top_percent: int = 30  # Look at more of the page for headers
    ):
        """
        Initialize first page detector.

        Args:
            ocr_confidence: OCR confidence threshold
            crop_top_percent: Percentage of page top to analyze for headers
        """
        self.ocr_service = create_ocr_service(confidence_threshold=ocr_confidence)
        self.crop_service = create_crop_service(crop_top_percent=crop_top_percent)
        self.crop_top_percent = crop_top_percent

        # Patterns that indicate a first page
        self.first_page_patterns = [
            # Name field patterns
            "name:",
            "name :",
            "student name:",
            "student name :",
            "your name:",

            # ID field patterns
            "student id:",
            "student id :",
            "id:",
            "id :",
            "student number:",

            # Date field patterns
            "date:",
            "date :",
            "exam date:",
            "today's date:",

            # Exam header patterns
            "exam",
            "test",
            "quiz",
            "midterm",
            "final",
            "assignment",

            # Course patterns
            "course:",
            "class:",
            "subject:",

            # Instruction patterns
            "instructions:",
            "directions:",
            "please write",
            "write your name",
            "fill in",

            # Form patterns
            "___________",  # Blank lines for filling in
            "[ ]",          # Checkboxes
            "( )",          # Radio buttons
        ]

        logger.info(f"Initialized first page detector with {len(self.first_page_patterns)} patterns")

    def is_first_page(self, image_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if an image is a first page of an exam.

        Args:
            image_path: Path to the image to analyze

        Returns:
            Tuple of (is_first_page, detection_metadata)
        """
        try:
            logger.debug(f"Analyzing first page patterns in: {image_path.name}")

            # Crop the header region (top portion of the page)
            header_region = self.crop_service.crop_name_region(image_path)

            # Extract all text from the header region
            text_results = self.ocr_service.extract_text_from_pil(header_region, detail=1)

            if not text_results:
                logger.debug(f"No text found in header region of {image_path.name}")
                return False, {
                    "reason": "No text found in header region",
                    "patterns_found": [],
                    "text_extracted": "",
                    "confidence_score": 0.0
                }

            # Combine all extracted text
            all_text = " ".join([result["text"] for result in text_results]).lower()

            # Count pattern matches
            patterns_found = []
            for pattern in self.first_page_patterns:
                if pattern in all_text:
                    patterns_found.append(pattern)

            # Calculate confidence score based on:
            # 1. Number of patterns found
            # 2. Quality of OCR results
            # 3. Length of text (first pages usually have more structured text)

            pattern_score = len(patterns_found) / len(self.first_page_patterns) * 100
            ocr_scores = [result["confidence"] for result in text_results]
            avg_ocr_confidence = sum(ocr_scores) / len(ocr_scores) if ocr_scores else 0
            text_length_score = min(len(all_text) / 200, 1.0) * 100  # Normalize to 100

            # Weighted confidence score
            confidence_score = (
                pattern_score * 0.6 +      # Pattern matching is most important
                avg_ocr_confidence * 0.3 + # OCR quality matters
                text_length_score * 0.1    # Text quantity helps
            )

            # Determine if this is a first page
            # High confidence: multiple patterns found
            # Medium confidence: at least one strong pattern
            is_first = len(patterns_found) >= 2 or (
                len(patterns_found) >= 1 and confidence_score > 50
            )

            metadata = {
                "patterns_found": patterns_found,
                "pattern_count": len(patterns_found),
                "text_extracted": all_text[:200] + "..." if len(all_text) > 200 else all_text,
                "confidence_score": confidence_score,
                "avg_ocr_confidence": avg_ocr_confidence,
                "text_length": len(all_text),
                "crop_region_percent": self.crop_top_percent,
                "reason": f"Found {len(patterns_found)} patterns" if is_first else "Insufficient patterns"
            }

            if is_first:
                logger.debug(f"✅ First page detected: {image_path.name} (patterns: {patterns_found})")
            else:
                logger.debug(f"❌ Not first page: {image_path.name} (patterns: {patterns_found})")

            return is_first, metadata

        except Exception as e:
            logger.error(f"Error detecting first page for {image_path}: {e}")
            return False, {
                "error": str(e),
                "reason": "Detection failed",
                "patterns_found": [],
                "confidence_score": 0.0
            }

    def batch_detect_first_pages(
        self,
        image_paths: List[Path]
    ) -> List[Tuple[Path, bool, Dict[str, Any]]]:
        """
        Detect first pages in a batch of images.

        Args:
            image_paths: List of image paths to analyze

        Returns:
            List of tuples: (image_path, is_first_page, metadata)
        """
        results = []
        first_pages_found = 0

        logger.info(f"Analyzing {len(image_paths)} images for first page patterns...")

        for i, image_path in enumerate(image_paths):
            try:
                is_first, metadata = self.is_first_page(image_path)
                results.append((image_path, is_first, metadata))

                if is_first:
                    first_pages_found += 1

                logger.debug(f"Processed {i+1}/{len(image_paths)}: {image_path.name} -> {'FIRST' if is_first else 'NOT_FIRST'}")

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append((image_path, False, {"error": str(e)}))

        logger.info(f"First page detection complete: {first_pages_found} first pages found out of {len(image_paths)} images")
        return results

    def get_detection_statistics(
        self,
        detection_results: List[Tuple[Path, bool, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Generate statistics from batch detection results.

        Args:
            detection_results: Results from batch_detect_first_pages

        Returns:
            Dictionary with detection statistics
        """
        if not detection_results:
            return {"total": 0, "first_pages": 0, "detection_rate": 0.0}

        total = len(detection_results)
        first_pages = sum(1 for _, is_first, _ in detection_results if is_first)

        # Collect confidence scores
        confidences = [
            metadata.get("confidence_score", 0)
            for _, _, metadata in detection_results
            if "confidence_score" in metadata
        ]

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Collect most common patterns
        all_patterns = []
        for _, is_first, metadata in detection_results:
            if is_first and "patterns_found" in metadata:
                all_patterns.extend(metadata["patterns_found"])

        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Sort patterns by frequency
        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_images": total,
            "first_pages_detected": first_pages,
            "detection_rate": (first_pages / total * 100) if total > 0 else 0.0,
            "average_confidence": avg_confidence,
            "top_patterns": top_patterns,
            "pattern_distribution": pattern_counts
        }

    def update_patterns(self, additional_patterns: List[str]) -> None:
        """Add additional patterns to look for."""
        original_count = len(self.first_page_patterns)
        self.first_page_patterns.extend([pattern.lower() for pattern in additional_patterns])
        # Remove duplicates while preserving order
        seen = set()
        self.first_page_patterns = [
            pattern for pattern in self.first_page_patterns
            if not (pattern in seen or seen.add(pattern))
        ]

        new_count = len(self.first_page_patterns)
        logger.info(f"Updated patterns: {original_count} -> {new_count} (+{new_count - original_count})")

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service configuration."""
        return {
            "crop_top_percent": self.crop_top_percent,
            "ocr_confidence": self.ocr_service.confidence_threshold,
            "pattern_count": len(self.first_page_patterns),
            "patterns": self.first_page_patterns[:10]  # Show first 10 patterns
        }


def create_first_page_detector(
    ocr_confidence: float = 0.5,
    crop_top_percent: int = 30
) -> FirstPageDetector:
    """Factory function to create a FirstPageDetector instance."""
    return FirstPageDetector(
        ocr_confidence=ocr_confidence,
        crop_top_percent=crop_top_percent
    )