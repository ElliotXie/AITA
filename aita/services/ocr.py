import easyocr
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(
        self,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        gpu: bool = False
    ):
        """
        Initialize OCR service with EasyOCR.

        Args:
            languages: List of language codes (default: ['en'])
            confidence_threshold: Minimum confidence score for text detection
            gpu: Whether to use GPU acceleration

        Raises:
            ValueError: If confidence_threshold is not between 0 and 1
            ImportError: If EasyOCR is not properly installed
            RuntimeError: If OCR initialization fails
        """
        if languages is None:
            languages = ['en']

        # Validate parameters
        if not isinstance(languages, list) or not languages:
            raise ValueError("Languages must be a non-empty list")

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

        if not isinstance(gpu, bool):
            raise ValueError("GPU flag must be a boolean")

        self.languages = languages
        self.confidence_threshold = confidence_threshold
        self.gpu = gpu
        self.reader = None

        try:
            import easyocr
            self.reader = easyocr.Reader(languages, gpu=gpu)
            logger.info(f"Initialized OCR service with languages: {languages}")
        except ImportError as e:
            logger.error(f"EasyOCR not properly installed: {e}")
            raise ImportError(f"EasyOCR not available: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize OCR service: {e}")
            raise RuntimeError(f"OCR initialization failed: {e}") from e

    def extract_text_from_image(
        self,
        image_path: Path,
        detail: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Extract text from an image using EasyOCR.

        Args:
            image_path: Path to the image file
            detail: Level of detail in output (0=text only, 1=coordinates+text)

        Returns:
            List of dictionaries containing text and metadata

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If detail level is invalid
            RuntimeError: If OCR is not initialized
        """
        if self.reader is None:
            raise RuntimeError("OCR service not properly initialized")

        if detail not in [0, 1]:
            raise ValueError("Detail level must be 0 or 1")

        # Validate image path
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not image_path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")

        # Check file size (warn if very large)
        file_size = image_path.stat().st_size
        if file_size > 50 * 1024 * 1024:  # 50MB
            logger.warning(f"Large image file ({file_size / 1024 / 1024:.1f}MB): {image_path}")

        try:
            logger.debug(f"Extracting text from {image_path}")

            # Validate and fix orientation of the image
            try:
                with Image.open(image_path) as img:
                    # Check image dimensions
                    if img.size[0] == 0 or img.size[1] == 0:
                        raise ValueError(f"Image has zero dimensions: {img.size}")
                    if img.size[0] > 20000 or img.size[1] > 20000:
                        logger.warning(f"Very large image: {img.size}")

                    # Fix orientation before processing
                    from ..utils.images import fix_image_orientation
                    img = fix_image_orientation(img)

                    # Convert to numpy array for EasyOCR
                    import numpy as np
                    image_array = np.array(img)

            except (UnidentifiedImageError, OSError) as e:
                raise ValueError(f"Invalid image file: {image_path}") from e

            # Read image and extract text (using the orientation-fixed array)
            results = self.reader.readtext(image_array, detail=detail)

            # Process results based on detail level
            if detail == 0:
                # Simple text extraction
                return [{"text": text, "confidence": 1.0} for text in results if text.strip()]
            else:
                # Detailed extraction with coordinates
                processed_results = []
                for result in results:
                    try:
                        bbox, text, confidence = result

                        # Skip empty text
                        if not text.strip():
                            continue

                        if confidence >= self.confidence_threshold:
                            # Validate bbox format
                            if not self._is_valid_bbox(bbox):
                                logger.warning(f"Invalid bbox format for text '{text}': {bbox}")
                                continue

                            processed_results.append({
                                "text": text,
                                "confidence": confidence,
                                "bbox": bbox,
                                "center": self._get_bbox_center(bbox)
                            })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping malformed OCR result: {result}, error: {e}")
                        continue

                return processed_results

        except (FileNotFoundError, ValueError, RuntimeError):
            raise
        except MemoryError as e:
            logger.error(f"Out of memory processing {image_path}: {e}")
            raise RuntimeError(f"Insufficient memory to process image: {image_path}") from e
        except Exception as e:
            logger.error(f"Failed to extract text from {image_path}: {e}")
            raise RuntimeError(f"OCR processing failed for {image_path}: {e}") from e

    def extract_text_from_pil(
        self,
        image: Image.Image,
        detail: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a PIL Image object.

        Args:
            image: PIL Image object
            detail: Level of detail in output

        Returns:
            List of dictionaries containing text and metadata

        Raises:
            ValueError: If image is invalid or detail level is wrong
            RuntimeError: If OCR is not initialized or processing fails
        """
        if self.reader is None:
            raise RuntimeError("OCR service not properly initialized")

        if detail not in [0, 1]:
            raise ValueError("Detail level must be 0 or 1")

        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image object")

        # Validate image dimensions
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("Image has zero width or height")

        # Check image size (warn if very large)
        pixel_count = image.size[0] * image.size[1]
        if pixel_count > 50_000_000:  # 50 megapixels
            logger.warning(f"Large image ({pixel_count / 1_000_000:.1f} megapixels)")

        try:
            # Convert PIL image to numpy array
            try:
                image_array = np.array(image)
            except Exception as e:
                raise ValueError(f"Failed to convert PIL image to numpy array: {e}") from e

            # Validate array
            if image_array.size == 0:
                raise ValueError("Image conversion resulted in empty array")

            # Extract text
            results = self.reader.readtext(image_array, detail=detail)

            if detail == 0:
                return [{"text": text, "confidence": 1.0} for text in results if text.strip()]
            else:
                processed_results = []
                for result in results:
                    try:
                        bbox, text, confidence = result

                        # Skip empty text
                        if not text.strip():
                            continue

                        if confidence >= self.confidence_threshold:
                            # Validate bbox format
                            if not self._is_valid_bbox(bbox):
                                logger.warning(f"Invalid bbox format for text '{text}': {bbox}")
                                continue

                            processed_results.append({
                                "text": text,
                                "confidence": confidence,
                                "bbox": bbox,
                                "center": self._get_bbox_center(bbox)
                            })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping malformed OCR result: {result}, error: {e}")
                        continue

                return processed_results

        except (ValueError, RuntimeError):
            raise
        except MemoryError as e:
            logger.error(f"Out of memory processing PIL image: {e}")
            raise RuntimeError("Insufficient memory to process image") from e
        except Exception as e:
            logger.error(f"Failed to extract text from PIL image: {e}")
            raise RuntimeError(f"OCR processing failed for PIL image: {e}") from e

    def extract_student_name(
        self,
        name_region_image: Image.Image
    ) -> Tuple[str, float]:
        """
        Extract student name from a cropped name region.

        Args:
            name_region_image: PIL Image of the name region

        Returns:
            Tuple of (extracted_name, confidence)
        """
        try:
            # Extract all text from the name region
            text_results = self.extract_text_from_pil(name_region_image, detail=1)

            if not text_results:
                logger.warning("No text found in name region")
                return "UNKNOWN", 0.0

            # Find the most likely name
            # Look for text that might be a name (longer text, higher confidence)
            best_candidate = None
            best_score = 0.0

            for result in text_results:
                text = result["text"].strip()
                confidence = result["confidence"]

                # Skip very short text (likely not names)
                if len(text) < 2:
                    continue

                # Skip obvious non-name text
                if any(keyword in text.lower() for keyword in ["name:", "student", "id:", "date", "class"]):
                    continue

                # Calculate a score based on length and confidence
                score = confidence * min(len(text) / 10, 1.0)  # Normalize length impact

                if score > best_score:
                    best_score = score
                    best_candidate = text

            if best_candidate:
                logger.debug(f"Extracted name: '{best_candidate}' (confidence: {best_score:.2f})")
                return self._clean_name(best_candidate), best_score
            else:
                logger.warning("No suitable name candidate found")
                return "UNKNOWN", 0.0

        except Exception as e:
            logger.error(f"Failed to extract student name: {e}")
            return "UNKNOWN", 0.0

    def extract_handwritten_text(
        self,
        image_path: Path,
        enhance_for_handwriting: bool = True
    ) -> str:
        """
        Extract handwritten text from an exam answer region.

        Args:
            image_path: Path to the image containing handwritten text
            enhance_for_handwriting: Whether to apply handwriting-specific enhancements

        Returns:
            Extracted text as a string

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If OCR processing fails
        """
        if self.reader is None:
            raise RuntimeError("OCR service not properly initialized")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            if enhance_for_handwriting:
                # Apply preprocessing for handwriting
                try:
                    from ..utils.images import preprocess_for_ocr
                    processed_image = preprocess_for_ocr(image_path)
                    text_results = self.extract_text_from_pil(processed_image, detail=0)
                except ImportError as e:
                    logger.warning(f"Image preprocessing not available: {e}")
                    text_results = self.extract_text_from_image(image_path, detail=0)
            else:
                text_results = self.extract_text_from_image(image_path, detail=0)

            # Combine all text results with proper spacing
            text_parts = [result["text"].strip() for result in text_results if result["text"].strip()]
            all_text = " ".join(text_parts)
            return all_text.strip()

        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            logger.error(f"Failed to extract handwritten text from {image_path}: {e}")
            raise RuntimeError(f"Handwriting extraction failed for {image_path}: {e}") from e

    def batch_extract_names(
        self,
        name_images: List[Image.Image]
    ) -> List[Tuple[str, float]]:
        """
        Extract names from multiple name region images.

        Args:
            name_images: List of PIL Images containing name regions

        Returns:
            List of (name, confidence) tuples

        Raises:
            ValueError: If input list is empty or contains invalid images
            RuntimeError: If OCR is not initialized
        """
        if self.reader is None:
            raise RuntimeError("OCR service not properly initialized")

        if not isinstance(name_images, list):
            raise ValueError("Input must be a list of PIL Images")

        if not name_images:
            logger.warning("Empty list provided to batch_extract_names")
            return []

        # Validate all images before processing
        for i, image in enumerate(name_images):
            if not isinstance(image, Image.Image):
                raise ValueError(f"Item {i} is not a PIL Image object")
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Image {i} has zero width or height")

        results = []
        successful_extractions = 0

        for i, image in enumerate(name_images):
            try:
                name, confidence = self.extract_student_name(image)
                results.append((name, confidence))
                if name != "UNKNOWN":
                    successful_extractions += 1
                logger.debug(f"Processed name image {i+1}/{len(name_images)}: {name}")
            except Exception as e:
                logger.error(f"Failed to process name image {i+1}: {e}")
                results.append(("UNKNOWN", 0.0))

        # Log batch processing statistics
        success_rate = successful_extractions / len(name_images) * 100
        logger.info(f"Batch name extraction: {successful_extractions}/{len(name_images)} successful ({success_rate:.1f}%)")

        return results

    def _get_bbox_center(self, bbox: List[List[int]]) -> Tuple[int, int]:
        """Calculate the center point of a bounding box.

        Args:
            bbox: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            Tuple of (center_x, center_y)

        Raises:
            ValueError: If bbox format is invalid
        """
        if not self._is_valid_bbox(bbox):
            raise ValueError(f"Invalid bbox format: {bbox}")

        try:
            # Convert numpy types to regular Python types
            x_coords = [float(point[0]) for point in bbox]
            y_coords = [float(point[1]) for point in bbox]

            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))

            return center_x, center_y
        except (IndexError, TypeError, ZeroDivisionError) as e:
            raise ValueError(f"Failed to calculate bbox center: {bbox}") from e

    def _is_valid_bbox(self, bbox) -> bool:
        """Validate bounding box format.

        Args:
            bbox: Bounding box to validate

        Returns:
            True if bbox is valid, False otherwise
        """
        try:
            # Check if it's a list/tuple of 4 points
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return False

            # Check if each point has 2 coordinates
            for point in bbox:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    return False
                # Check if coordinates are numeric (including numpy types)
                for coord in point:
                    if not isinstance(coord, (int, float, np.integer, np.floating)):
                        return False

            return True
        except Exception:
            return False

    def _clean_name(self, name: str) -> str:
        """Clean and normalize extracted name text."""
        # Remove common OCR artifacts
        name = name.strip()

        # Remove common prefixes/suffixes that might be picked up
        prefixes_to_remove = ["name:", "student:", "name", "student"]
        for prefix in prefixes_to_remove:
            if name.lower().startswith(prefix):
                name = name[len(prefix):].strip()

        # Remove special characters that are likely OCR errors
        # But keep common name characters (letters, spaces, hyphens, apostrophes)
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '-.")
        cleaned_name = "".join(char for char in name if char in allowed_chars)

        # Remove extra spaces
        cleaned_name = " ".join(cleaned_name.split())

        return cleaned_name

    def get_ocr_stats(self) -> Dict[str, Any]:
        """Get OCR service statistics and info."""
        return {
            "languages": self.languages,
            "confidence_threshold": self.confidence_threshold,
            "gpu_enabled": self.gpu,
            "reader_initialized": hasattr(self, 'reader')
        }


def create_ocr_service(
    languages: List[str] = None,
    confidence_threshold: float = 0.5,
    gpu: bool = False
) -> OCRService:
    """Factory function to create an OCR service instance."""
    return OCRService(
        languages=languages,
        confidence_threshold=confidence_threshold,
        gpu=gpu
    )