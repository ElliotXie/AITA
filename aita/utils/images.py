from PIL import Image, ImageEnhance, ExifTags
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def crop_top_region(
    image_path: Path,
    top_percent: int = 20
) -> Image.Image:
    with Image.open(image_path) as img:
        # Fix orientation before cropping
        img = fix_image_orientation(img)

        width, height = img.size
        crop_height = int(height * (top_percent / 100))

        # Crop the top region
        cropped = img.crop((0, 0, width, crop_height))
        return cropped.copy()


def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)

    return image


def resize_image(
    image: Image.Image,
    max_width: int = 1920,
    max_height: int = 1080
) -> Image.Image:
    width, height = image.size

    if width <= max_width and height <= max_height:
        return image

    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def save_processed_image(
    image: Image.Image,
    output_path: Path,
    quality: int = 95
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        image.save(output_path, 'JPEG', quality=quality, optimize=True)
    elif output_path.suffix.lower() == '.png':
        image.save(output_path, 'PNG', optimize=True)
    else:
        image.save(output_path)

    logger.debug(f"Saved processed image to {output_path}")


def get_image_info(image_path: Path) -> dict:
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "file_size": image_path.stat().st_size
            }
    except Exception as e:
        logger.error(f"Failed to get image info for {image_path}: {e}")
        return {}


def validate_image_format(image_path: Path) -> bool:
    supported_formats = ['JPEG', 'PNG', 'TIFF', 'BMP']

    try:
        with Image.open(image_path) as img:
            return img.format in supported_formats
    except Exception as e:
        logger.warning(f"Invalid image format for {image_path}: {e}")
        return False


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def crop_region(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int
) -> Image.Image:
    return image.crop((x, y, x + width, y + height))


def create_thumbnail(
    image_path: Path,
    thumbnail_path: Path,
    size: Tuple[int, int] = (200, 200)
) -> None:
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            save_processed_image(img, thumbnail_path)
            logger.debug(f"Created thumbnail: {thumbnail_path}")
    except Exception as e:
        logger.error(f"Failed to create thumbnail: {e}")


def fix_image_orientation(image: Image.Image) -> Image.Image:
    """Fix image orientation based on EXIF data."""
    try:
        # Check for EXIF orientation tag
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif is not None:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "Orientation":
                        # Apply rotation based on EXIF orientation
                        if value == 3:
                            image = image.rotate(180, expand=True)
                        elif value == 6:
                            image = image.rotate(270, expand=True)
                        elif value == 8:
                            image = image.rotate(90, expand=True)
                        logger.debug(f"Applied orientation correction: {value}")
                        break
    except (AttributeError, TypeError, KeyError) as e:
        logger.debug(f"Could not read EXIF orientation: {e}")

    return image


def detect_text_regions(image: Image.Image) -> list:
    # Placeholder for text region detection
    # In a real implementation, you might use:
    # - OpenCV for contour detection
    # - EAST text detector
    # - Or other computer vision techniques

    # For now, return the full image as a single region
    width, height = image.size
    return [{"x": 0, "y": 0, "width": width, "height": height}]


def preprocess_for_ocr(
    image_path: Path,
    output_path: Optional[Path] = None
) -> Image.Image:
    with Image.open(image_path) as img:
        # Fix orientation first
        img = fix_image_orientation(img)

        # Resize if too large
        img = resize_image(img)

        # Enhance for OCR
        img = enhance_image_for_ocr(img)

        if output_path:
            save_processed_image(img, output_path)

        return img