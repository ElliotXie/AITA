import os
import shutil
from pathlib import Path
from typing import List, Generator, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_data_tree(data_dir: Path) -> None:
    directories = [
        data_dir,
        data_dir / "raw_images",
        data_dir / "grouped",
        data_dir / "results",
        data_dir / "results" / "transcriptions",
        data_dir / "reports"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def find_images(
    directory: Path,
    extensions: List[str] = None
) -> Generator[Path, None, None]:
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

    extensions = [ext.lower() for ext in extensions]

    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            yield file_path


def copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    return shutil.copy2(src, dst)


def move_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    return shutil.move(str(src), str(dst))


def get_file_size(file_path: Path) -> int:
    return file_path.stat().st_size


def get_modification_time(file_path: Path) -> float:
    return file_path.stat().st_mtime


def safe_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.strip()


def create_student_directory(
    base_dir: Path,
    student_name: str
) -> Path:
    safe_name = safe_filename(student_name)
    student_dir = base_dir / safe_name
    student_dir.mkdir(parents=True, exist_ok=True)
    return student_dir


def organize_images_by_timestamp(
    image_paths: List[Path],
    pages_per_student: int
) -> List[List[Path]]:
    # Sort images by modification time
    sorted_images = sorted(image_paths, key=get_modification_time)

    # Group into chunks of pages_per_student
    groups = []
    for i in range(0, len(sorted_images), pages_per_student):
        group = sorted_images[i:i + pages_per_student]
        groups.append(group)

    return groups


def count_files_in_directory(directory: Path, pattern: str = "*") -> int:
    return len(list(directory.glob(pattern)))


def get_directory_size(directory: Path) -> int:
    total = 0
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            total += get_file_size(file_path)
    return total


def cleanup_empty_directories(directory: Path) -> None:
    for subdir in directory.rglob("*"):
        if subdir.is_dir():
            try:
                subdir.rmdir()
                logger.debug(f"Removed empty directory: {subdir}")
            except OSError:
                # Directory not empty
                pass


def read_student_roster(roster_file: Path) -> List[str]:
    if not roster_file.exists():
        logger.warning(f"Roster file not found: {roster_file}")
        return []

    try:
        with open(roster_file, 'r', encoding='utf-8') as f:
            # Skip header if present
            lines = [line.strip() for line in f.readlines()]
            if lines and lines[0].lower() == 'name':
                lines = lines[1:]

            # Filter out empty lines
            student_names = [name for name in lines if name]
            logger.info(f"Loaded {len(student_names)} students from roster")
            return student_names

    except Exception as e:
        logger.error(f"Failed to read roster file: {e}")
        return []


def validate_image_file(file_path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Invalid image file {file_path}: {e}")
        return False


def get_image_dimensions(file_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            return img.size
    except Exception as e:
        logger.error(f"Failed to get image dimensions for {file_path}: {e}")
        return (0, 0)