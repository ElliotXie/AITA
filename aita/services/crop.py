from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import logging

from ..utils.images import crop_top_region, enhance_image_for_ocr, save_processed_image

logger = logging.getLogger(__name__)


class ImageCropService:
    def __init__(self, crop_top_percent: int = 20):
        self.crop_top_percent = crop_top_percent

    def crop_name_region(
        self,
        image_path: Path,
        output_path: Optional[Path] = None
    ) -> Image.Image:
        """
        Crop the top region of an exam image where the student name is typically located.

        Args:
            image_path: Path to the input image
            output_path: Optional path to save the cropped image

        Returns:
            PIL Image object of the cropped region
        """
        try:
            logger.debug(f"Cropping name region from {image_path}")

            # Crop the top region
            cropped_image = crop_top_region(image_path, self.crop_top_percent)

            # Enhance for better OCR results
            enhanced_image = enhance_image_for_ocr(cropped_image)

            # Save if output path is provided
            if output_path:
                save_processed_image(enhanced_image, output_path)
                logger.debug(f"Saved cropped name region to {output_path}")

            return enhanced_image

        except Exception as e:
            logger.error(f"Failed to crop name region from {image_path}: {e}")
            raise

    def crop_question_region(
        self,
        image_path: Path,
        bounds: dict,
        output_path: Optional[Path] = None
    ) -> Image.Image:
        """
        Crop a specific question region from an exam image.

        Args:
            image_path: Path to the input image
            bounds: Dictionary with keys 'x', 'y', 'width', 'height'
            output_path: Optional path to save the cropped image

        Returns:
            PIL Image object of the cropped question region
        """
        try:
            with Image.open(image_path) as img:
                # Extract bounds
                x = bounds['x']
                y = bounds['y']
                width = bounds['width']
                height = bounds['height']

                # Crop the region
                cropped = img.crop((x, y, x + width, y + height))

                # Save if output path is provided
                if output_path:
                    save_processed_image(cropped, output_path)
                    logger.debug(f"Saved question region to {output_path}")

                return cropped.copy()

        except Exception as e:
            logger.error(f"Failed to crop question region from {image_path}: {e}")
            raise

    def batch_crop_names(
        self,
        image_paths: list[Path],
        output_dir: Path
    ) -> list[Path]:
        """
        Crop name regions from multiple images in batch.

        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save cropped images

        Returns:
            List of paths to the saved cropped images
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []

        for i, image_path in enumerate(image_paths):
            try:
                # Generate output filename
                output_filename = f"name_crop_{i:03d}_{image_path.stem}.png"
                output_path = output_dir / output_filename

                # Crop and save
                self.crop_name_region(image_path, output_path)
                output_paths.append(output_path)

                logger.debug(f"Processed {i+1}/{len(image_paths)}: {image_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue

        logger.info(f"Successfully cropped {len(output_paths)}/{len(image_paths)} images")
        return output_paths

    def get_crop_preview(
        self,
        image_path: Path,
        crop_percent: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Get a preview showing the original image and the cropped region.

        Args:
            image_path: Path to the input image
            crop_percent: Override the default crop percentage

        Returns:
            Tuple of (original_image, cropped_image)
        """
        with Image.open(image_path) as original:
            crop_pct = crop_percent or self.crop_top_percent
            cropped = crop_top_region(image_path, crop_pct)

            return original.copy(), cropped

    def validate_crop_region(
        self,
        image_path: Path,
        crop_percent: Optional[int] = None
    ) -> dict:
        """
        Validate that the crop region contains useful content.

        Args:
            image_path: Path to the input image
            crop_percent: Override the default crop percentage

        Returns:
            Dictionary with validation results
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                crop_pct = crop_percent or self.crop_top_percent
                crop_height = int(height * (crop_pct / 100))

                # Get crop region info
                cropped = crop_top_region(image_path, crop_pct)
                crop_width, crop_height_actual = cropped.size

                # Basic validation
                min_size = 100  # Minimum reasonable dimensions
                is_valid = crop_width >= min_size and crop_height_actual >= min_size

                return {
                    "is_valid": is_valid,
                    "original_size": (width, height),
                    "crop_size": (crop_width, crop_height_actual),
                    "crop_percentage": crop_pct,
                    "crop_area_ratio": (crop_width * crop_height_actual) / (width * height)
                }

        except Exception as e:
            logger.error(f"Failed to validate crop region for {image_path}: {e}")
            return {
                "is_valid": False,
                "error": str(e)
            }


def create_crop_service(crop_top_percent: int = 20) -> ImageCropService:
    """Factory function to create an ImageCropService instance."""
    return ImageCropService(crop_top_percent=crop_top_percent)