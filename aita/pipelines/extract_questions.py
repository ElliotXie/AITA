"""
Question Extraction Pipeline

Extracts question structure from exam images using LLM vision capabilities.
Takes one student's exam as a sample to build the complete ExamSpec.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from aita.domain.models import Question, ExamSpec, QuestionType
from aita.domain.exam_reconstruct import ExamReconstructor
from aita.services.llm.base import BaseLLMClient
from aita.services.storage import GoogleCloudStorageService
from aita.utils.prompts import get_question_extraction_prompt

logger = logging.getLogger(__name__)
console = Console()


class QuestionExtractionError(Exception):
    """Raised when question extraction fails."""
    pass


class QuestionExtractionPipeline:
    """
    Pipeline for extracting question structure from exam images.

    Uses LLM vision API to analyze exam pages and extract:
    - Question identifiers (1a, 1b, 2, etc.)
    - Question text/content
    - Point values
    - Page numbers
    - Question types
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        storage_service: GoogleCloudStorageService,
        data_dir: Path,
        max_parse_retries: int = 2
    ):
        """
        Initialize the question extraction pipeline.

        Args:
            llm_client: LLM client for vision analysis
            storage_service: GCS service for image uploads
            data_dir: Base data directory
            max_parse_retries: Maximum retries for JSON parsing failures
        """
        self.llm_client = llm_client
        self.storage_service = storage_service
        self.data_dir = Path(data_dir)
        self.max_parse_retries = max_parse_retries
        self.reconstructor = ExamReconstructor(data_dir)

        logger.info("QuestionExtractionPipeline initialized")

    def extract_from_student(
        self,
        student_folder: Path,
        assignment_name: str,
        exam_name: Optional[str] = None
    ) -> ExamSpec:
        """
        Extract question structure from a student's exam images.

        Args:
            student_folder: Path to student's folder containing exam images
            assignment_name: Name of assignment for GCS organization
            exam_name: Optional exam name (defaults to assignment_name)

        Returns:
            ExamSpec object with extracted questions

        Raises:
            QuestionExtractionError: If extraction fails
        """
        student_folder = Path(student_folder)
        if not student_folder.exists():
            raise QuestionExtractionError(f"Student folder not found: {student_folder}")

        console.print(f"\n📚 [bold cyan]Question Extraction Pipeline[/bold cyan]")
        console.print(f"   Student: {student_folder.name}")
        console.print(f"   Assignment: {assignment_name}\n")

        try:
            # Step 1: Find exam images
            image_paths = self._find_exam_images(student_folder)
            console.print(f"   Found {len(image_paths)} exam pages")

            # Step 2: Upload to GCS
            image_urls = self._upload_images_to_gcs(
                image_paths,
                student_folder.name,
                assignment_name
            )

            # Step 3: Extract with LLM
            exam_spec = self._extract_with_llm(
                image_urls,
                exam_name or assignment_name,
                len(image_paths)
            )

            # Step 4: Validate results
            self._validate_exam_spec(exam_spec)

            # Step 5: Save results
            saved_path = self._save_results(exam_spec)

            # Success summary
            console.print(f"\n✅ [bold green]Extraction Complete![/bold green]")
            console.print(f"   📋 Exam: {exam_spec.exam_name}")
            console.print(f"   📄 Pages: {exam_spec.total_pages}")
            console.print(f"   ❓ Questions: {len(exam_spec.questions)}")
            console.print(f"   💯 Total Points: {exam_spec.total_points}")
            console.print(f"   💾 Saved to: {saved_path}\n")

            return exam_spec

        except Exception as e:
            logger.error(f"Question extraction failed: {e}", exc_info=True)
            raise QuestionExtractionError(f"Extraction failed: {e}") from e

    def _find_exam_images(self, student_folder: Path) -> List[Path]:
        """
        Find all exam image files in student folder.

        Args:
            student_folder: Path to student folder

        Returns:
            List of image file paths, sorted by name
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        image_files = [
            f for f in student_folder.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        # Sort by filename to maintain page order
        image_files.sort()

        if not image_files:
            raise QuestionExtractionError(f"No image files found in {student_folder}")

        logger.info(f"Found {len(image_files)} exam images")
        return image_files

    def _upload_images_to_gcs(
        self,
        image_paths: List[Path],
        student_name: str,
        assignment_name: str
    ) -> List[str]:
        """
        Upload exam images to GCS and return public URLs.

        Args:
            image_paths: List of local image paths
            student_name: Student name for organization
            assignment_name: Assignment name for organization

        Returns:
            List of public GCS URLs
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"📤 Uploading {len(image_paths)} images to GCS...",
                total=None
            )

            try:
                # Convert Path objects to strings
                image_paths_str = [str(p) for p in image_paths]

                # Batch upload
                results = self.storage_service.upload_images_batch(
                    image_paths=image_paths_str,
                    student_name=student_name,
                    assignment_name=assignment_name
                )

                # Extract URLs and check for failures
                urls = []
                failed = []

                for result in results:
                    if result.get('public_url'):
                        urls.append(result['public_url'])
                    else:
                        failed.append(result['local_path'])
                        logger.warning(f"Failed to upload: {result['local_path']}")

                progress.update(task, completed=True)

                if failed:
                    console.print(f"   ⚠️  Failed to upload {len(failed)} images")

                if not urls:
                    raise QuestionExtractionError("All image uploads failed")

                console.print(f"   ✅ Uploaded {len(urls)} images to GCS")
                logger.info(f"Successfully uploaded {len(urls)} images")

                return urls

            except Exception as e:
                progress.update(task, completed=True)
                raise QuestionExtractionError(f"GCS upload failed: {e}") from e

    def _extract_with_llm(
        self,
        image_urls: List[str],
        exam_name: str,
        total_pages: int
    ) -> ExamSpec:
        """
        Extract question structure using LLM vision analysis.

        Args:
            image_urls: List of public image URLs
            exam_name: Name of the exam
            total_pages: Total number of pages

        Returns:
            ExamSpec object with extracted questions
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                "🔍 Analyzing exam structure with LLM...",
                total=None
            )

            try:
                # Get extraction prompt
                prompt = get_question_extraction_prompt()

                # Try parsing with retries
                for attempt in range(self.max_parse_retries):
                    try:
                        # Call LLM with all images
                        logger.info(f"Calling LLM with {len(image_urls)} images (attempt {attempt + 1})")
                        response_text = self.llm_client.complete_with_images(
                            prompt=prompt,
                            image_urls=image_urls,
                            temperature=0.1
                        )

                        logger.debug(f"LLM response: {response_text[:500]}...")

                        # Parse response
                        exam_spec = parse_question_extraction_response(
                            response_text,
                            exam_name,
                            total_pages
                        )

                        progress.update(task, completed=True)
                        console.print(f"   ✅ Extracted {len(exam_spec.questions)} questions")

                        return exam_spec

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {e}")

                        if attempt < self.max_parse_retries - 1:
                            # Retry with enhanced prompt
                            prompt = (
                                "Your previous response was not valid JSON. "
                                "Please respond ONLY with valid JSON, no markdown formatting "
                                "or additional text.\n\n" +
                                get_question_extraction_prompt()
                            )
                        else:
                            raise QuestionExtractionError(
                                f"Failed to parse LLM response after {self.max_parse_retries} attempts"
                            ) from e

            except Exception as e:
                progress.update(task, completed=True)
                raise QuestionExtractionError(f"LLM extraction failed: {e}") from e

    def _validate_exam_spec(self, exam_spec: ExamSpec) -> None:
        """
        Validate the extracted exam specification.

        Args:
            exam_spec: ExamSpec to validate

        Raises:
            QuestionExtractionError: If validation fails
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("✅ Validating results...", total=None)

            errors = validate_exam_spec(exam_spec)

            progress.update(task, completed=True)

            if errors:
                console.print(f"   ❌ Validation failed with {len(errors)} errors:")
                for error in errors:
                    console.print(f"      • {error}")

                raise QuestionExtractionError(
                    f"Exam spec validation failed: {', '.join(errors)}"
                )

            console.print("   ✅ Validation passed")
            logger.info("Exam spec validation successful")

    def _save_results(self, exam_spec: ExamSpec) -> Path:
        """
        Save exam specification to file.

        Args:
            exam_spec: ExamSpec to save

        Returns:
            Path where spec was saved
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("💾 Saving exam specification...", total=None)

            saved_path = self.reconstructor.save_exam_spec(exam_spec)

            progress.update(task, completed=True)
            console.print(f"   ✅ Saved to {saved_path}")

            return saved_path


def parse_question_extraction_response(
    response_text: str,
    exam_name: str,
    total_pages: int
) -> ExamSpec:
    """
    Parse LLM JSON response into ExamSpec object.

    Handles various response formats including markdown-wrapped JSON.

    Args:
        response_text: Raw LLM response text
        exam_name: Exam name to use if not in response
        total_pages: Total pages to use if not in response

    Returns:
        ExamSpec object

    Raises:
        json.JSONDecodeError: If JSON parsing fails
        ValueError: If required fields are missing
    """
    # Try to extract JSON from response (may have markdown wrapping)
    json_text = _extract_json_from_response(response_text)

    # Parse JSON
    data = json.loads(json_text)

    # Extract exam metadata
    exam_name_extracted = data.get('exam_name', exam_name)
    total_pages_extracted = data.get('total_pages', total_pages)

    # Extract questions
    questions_data = data.get('questions', [])
    if not questions_data:
        raise ValueError("No questions found in response")

    questions = []
    question_ids_seen = set()

    for q_data in questions_data:
        try:
            # Handle duplicate question IDs
            question_id = q_data.get('question_id', f"q{len(questions) + 1}")
            original_id = question_id
            counter = 2
            while question_id in question_ids_seen:
                question_id = f"{original_id}_v{counter}"
                counter += 1
            question_ids_seen.add(question_id)

            # Map question type
            question_type_str = q_data.get('question_type', 'short_answer')
            question_type = _map_question_type(question_type_str)

            # Create Question object
            question = Question(
                question_id=question_id,
                question_text=q_data.get('question_text', '').strip(),
                points=float(q_data.get('points', 0)),
                question_type=question_type,
                page_number=q_data.get('page_number'),
                image_bounds=q_data.get('image_bounds')
            )

            questions.append(question)

        except Exception as e:
            logger.warning(f"Failed to parse question: {q_data}. Error: {e}")
            # Continue with other questions

    if not questions:
        raise ValueError("Failed to parse any questions from response")

    # Create ExamSpec
    exam_spec = ExamSpec(
        exam_name=exam_name_extracted,
        total_pages=total_pages_extracted,
        questions=questions
    )

    logger.info(f"Parsed {len(questions)} questions from LLM response")
    return exam_spec


def _extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON from LLM response, handling markdown wrapping.

    Args:
        response_text: Raw response text

    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try to find raw JSON object
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # Return as-is and let JSON parser handle it
    return response_text.strip()


def _map_question_type(type_str: str) -> QuestionType:
    """
    Map question type string to QuestionType enum.

    Args:
        type_str: Question type string from LLM

    Returns:
        QuestionType enum value
    """
    type_mapping = {
        'multiple_choice': QuestionType.MULTIPLE_CHOICE,
        'short_answer': QuestionType.SHORT_ANSWER,
        'long_answer': QuestionType.LONG_ANSWER,
        'calculation': QuestionType.CALCULATION,
        'diagram': QuestionType.DIAGRAM,
    }

    # Normalize and lookup
    normalized = type_str.lower().replace(' ', '_').replace('-', '_')
    return type_mapping.get(normalized, QuestionType.SHORT_ANSWER)


def validate_exam_spec(exam_spec: ExamSpec) -> List[str]:
    """
    Validate exam specification and return list of errors.

    Args:
        exam_spec: ExamSpec to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check has questions
    if not exam_spec.questions:
        errors.append("No questions found in exam spec")
        return errors  # Can't validate further

    # Check unique question IDs
    question_ids = [q.question_id for q in exam_spec.questions]
    if len(question_ids) != len(set(question_ids)):
        duplicates = [qid for qid in question_ids if question_ids.count(qid) > 1]
        errors.append(f"Duplicate question IDs found: {set(duplicates)}")

    # Check each question
    for q in exam_spec.questions:
        # Check points
        if q.points <= 0:
            errors.append(f"Question {q.question_id} has invalid points: {q.points}")

        # Check question text
        if not q.question_text.strip():
            errors.append(f"Question {q.question_id} has empty question text")

        # Check page number (warning, not error)
        if q.page_number is not None:
            if q.page_number < 1 or q.page_number > exam_spec.total_pages:
                errors.append(
                    f"Question {q.question_id} page number {q.page_number} "
                    f"out of range (1-{exam_spec.total_pages})"
                )

    # Check total points reasonable
    if exam_spec.total_points and exam_spec.total_points <= 0:
        errors.append(f"Invalid total points: {exam_spec.total_points}")

    return errors


# Factory and convenience functions

def create_extraction_pipeline(
    assignment_name: str,
    data_dir: Optional[Path] = None
) -> QuestionExtractionPipeline:
    """
    Create question extraction pipeline with services from config.

    Args:
        assignment_name: Name of assignment
        data_dir: Optional data directory (uses config default if not provided)

    Returns:
        Configured QuestionExtractionPipeline
    """
    from aita.config import get_config
    from aita.services.llm.openrouter import create_openrouter_client
    from aita.services.storage import create_storage_service

    config = get_config()

    llm_client = create_openrouter_client()
    storage_service = create_storage_service()

    return QuestionExtractionPipeline(
        llm_client=llm_client,
        storage_service=storage_service,
        data_dir=data_dir or Path(config.data_dir)
    )


def extract_questions_from_student(
    student_folder: str,
    assignment_name: str = "exam1",
    exam_name: Optional[str] = None,
    data_dir: Optional[Path] = None
) -> ExamSpec:
    """
    High-level function to extract questions from a student's exam.

    Args:
        student_folder: Path to student folder with exam images
        assignment_name: Name of assignment for GCS organization
        exam_name: Optional custom exam name
        data_dir: Optional data directory

    Returns:
        ExamSpec with extracted questions

    Example:
        >>> exam_spec = extract_questions_from_student(
        ...     "data/grouped/Student_001",
        ...     assignment_name="BMI541_Midterm"
        ... )
        >>> print(f"Extracted {len(exam_spec.questions)} questions")
    """
    pipeline = create_extraction_pipeline(assignment_name, data_dir)

    return pipeline.extract_from_student(
        student_folder=Path(student_folder),
        assignment_name=assignment_name,
        exam_name=exam_name
    )
