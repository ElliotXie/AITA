"""
Transcription Pipeline

Transcribes all student exam images to text using LLM vision capabilities.
Processes grouped student folders and generates comprehensive transcription results.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from aita.domain.models import Student, StudentAnswer, ExamSpec
from aita.services.llm.base import BaseLLMClient
from aita.services.storage import GoogleCloudStorageService
from aita.utils.prompts import get_transcription_prompt
from aita.utils.files import find_images
from aita.utils.transcription_helpers import (
    load_exam_spec,
    create_question_mapped_answers,
    validate_question_mapping,
    get_question_mapping_stats
)
from aita.report.transcription_report import TranscriptionReportGenerator
from aita.utils.parallel_executor import ParallelLLMExecutor
from aita.utils.llm_task_implementations import TranscriptionTask

logger = logging.getLogger(__name__)
console = Console()


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


@dataclass
class PageTranscription:
    """Result of transcribing a single page."""
    page_number: int
    image_path: Path
    transcribed_text: str
    confidence: float
    notes: Optional[str] = None
    processing_time: Optional[float] = None
    public_url: Optional[str] = None
    raw_llm_response: Optional[str] = None


@dataclass
class StudentTranscriptionResult:
    """Result of transcribing all pages for one student."""
    student_name: str
    student: Student
    pages: List[PageTranscription] = field(default_factory=list)
    total_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return (self.successful_pages / self.total_pages) * 100

    @property
    def average_confidence(self) -> float:
        confidences = [p.confidence for p in self.pages if p.confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0


@dataclass
class TranscriptionResults:
    """Results of the complete transcription pipeline."""
    total_students: int
    successful_students: int
    total_pages: int
    successful_pages: int
    student_results: List[StudentTranscriptionResult] = field(default_factory=list)
    processing_time: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return (self.successful_pages / self.total_pages) * 100

    @property
    def average_confidence(self) -> float:
        all_confidences = []
        for student in self.student_results:
            all_confidences.extend([p.confidence for p in student.pages if p.confidence > 0])
        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0


class TranscriptionPipeline:
    """
    Pipeline for transcribing student exam images to text.

    Uses LLM vision API to analyze exam pages and extract:
    - Handwritten student responses
    - Mathematical notation and symbols
    - Diagrams and drawings (as descriptions)
    - Quality confidence scores
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        storage_service: GoogleCloudStorageService,
        data_dir: Path,
        assignment_name: str = "exam_transcription",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_question_mapping: bool = True,
        generate_html_reports: bool = True,
        parallel_executor: Optional[ParallelLLMExecutor] = None
    ):
        """
        Initialize the transcription pipeline.

        Args:
            llm_client: LLM client for vision analysis
            storage_service: GCS service for image uploads
            data_dir: Base data directory containing grouped student folders
            assignment_name: Assignment name for GCS organization
            max_retries: Maximum retries for failed transcriptions (deprecated, use executor)
            retry_delay: Delay between retry attempts (deprecated, use executor)
            use_question_mapping: Whether to create question-mapped transcriptions
            generate_html_reports: Whether to generate HTML reports
            parallel_executor: Optional parallel executor (will create one if not provided)
        """
        self.llm_client = llm_client
        self.storage_service = storage_service
        self.data_dir = Path(data_dir)
        self.assignment_name = assignment_name
        self.max_retries = max_retries  # Kept for backward compatibility
        self.retry_delay = retry_delay  # Kept for backward compatibility
        self.use_question_mapping = use_question_mapping
        self.generate_html_reports = generate_html_reports

        # Setup parallel executor
        if parallel_executor:
            self.executor = parallel_executor
        else:
            # Create executor from config
            from aita.config import get_config
            config = get_config()
            self.executor = ParallelLLMExecutor(
                llm_client=llm_client,
                max_workers=config.parallel_execution.max_workers,
                rate_limit=config.parallel_execution.rate_limit_rps,
                enable_checkpointing=config.parallel_execution.enable_checkpointing,
                checkpoint_interval=config.parallel_execution.checkpoint_interval,
                max_retries=max_retries,
                show_progress=config.parallel_execution.show_progress
            )

        # Directory setup
        self.grouped_dir = self.data_dir / "grouped"
        self.output_dir = self.data_dir.parent / "intermediateproduct" / "transcription_results"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load exam spec for question mapping
        self.exam_spec: Optional[ExamSpec] = None
        self.report_generator: Optional[TranscriptionReportGenerator] = None

        if self.use_question_mapping or self.generate_html_reports:
            self.exam_spec = self._load_exam_spec()
            if self.exam_spec:
                logger.info(f"  ExamSpec loaded: {len(self.exam_spec.questions)} questions")

                # Initialize report generator if HTML reports are enabled
                if self.generate_html_reports:
                    self.report_generator = TranscriptionReportGenerator(self.exam_spec)
                    logger.info("  HTML report generator initialized")
            else:
                logger.warning("  ExamSpec not found - question mapping and HTML reports will be skipped")
                self.use_question_mapping = False
                self.generate_html_reports = False

        logger.info("TranscriptionPipeline initialized")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Grouped dir: {self.grouped_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Question mapping: {self.use_question_mapping}")
        logger.info(f"  HTML reports: {self.generate_html_reports}")

    def transcribe_all_students(self) -> TranscriptionResults:
        """
        Transcribe all students in the grouped directory.

        Returns:
            TranscriptionResults containing all processing results

        Raises:
            TranscriptionError: If no students found or critical error occurs
        """
        console.print(f"\n📝 [bold cyan]Transcription Pipeline[/bold cyan]")
        console.print(f"   Assignment: {self.assignment_name}")
        console.print(f"   Data Directory: {self.data_dir}\n")

        start_time = datetime.now()

        try:
            # Step 1: Discover student folders
            student_folders = self._discover_student_folders()
            if not student_folders:
                raise TranscriptionError("No student folders found in grouped directory")

            console.print(f"   Found {len(student_folders)} student folders to process\n")

            # Step 2: Process each student
            student_results = []
            total_pages = 0
            successful_pages = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                main_task = progress.add_task(
                    f"Processing {len(student_folders)} students...",
                    total=len(student_folders)
                )

                for i, student_folder in enumerate(student_folders):
                    try:
                        console.print(f"\n🔄 Processing: {student_folder.name}")

                        student_result = self._transcribe_student(student_folder)
                        student_results.append(student_result)

                        total_pages += student_result.total_pages
                        successful_pages += student_result.successful_pages

                        # Save individual student results
                        self._save_student_results(student_result)

                        console.print(f"   ✅ Completed {student_folder.name}: {student_result.successful_pages}/{student_result.total_pages} pages")

                    except Exception as e:
                        logger.error(f"Failed to process student {student_folder.name}: {e}")
                        console.print(f"   ❌ Failed {student_folder.name}: {e}")
                        # Continue with other students

                    progress.update(main_task, advance=1)

            # Step 3: Generate final results and summary
            processing_time = (datetime.now() - start_time).total_seconds()

            results = TranscriptionResults(
                total_students=len(student_folders),
                successful_students=len([r for r in student_results if r.successful_pages > 0]),
                total_pages=total_pages,
                successful_pages=successful_pages,
                student_results=student_results,
                processing_time=processing_time
            )

            # Step 4: Generate statistics
            results.statistics = self._generate_statistics(results)

            # Step 5: Save summary results
            self._save_summary_results(results)

            # Step 6: Display final summary
            self._display_summary(results)

            return results

        except Exception as e:
            logger.error(f"Transcription pipeline failed: {e}", exc_info=True)
            raise TranscriptionError(f"Pipeline failed: {e}") from e

    def _load_exam_spec(self) -> Optional[ExamSpec]:
        """
        Load exam specification from question extraction results.

        Returns:
            ExamSpec if found, None otherwise
        """
        try:
            exam_spec = load_exam_spec(self.data_dir)
            if exam_spec:
                logger.info(f"Loaded ExamSpec: {exam_spec.exam_name} ({len(exam_spec.questions)} questions)")
            return exam_spec
        except Exception as e:
            logger.error(f"Failed to load ExamSpec: {e}")
            return None

    def _discover_student_folders(self) -> List[Path]:
        """
        Discover all student folders in the grouped directory.

        Returns:
            List of student folder paths
        """
        if not self.grouped_dir.exists():
            raise TranscriptionError(f"Grouped directory not found: {self.grouped_dir}")

        student_folders = [
            folder for folder in self.grouped_dir.iterdir()
            if folder.is_dir() and not folder.name.startswith('.')
        ]

        # Sort by folder name for consistent processing order
        student_folders.sort(key=lambda x: x.name)

        logger.info(f"Found {len(student_folders)} student folders")
        for folder in student_folders:
            logger.debug(f"  Student folder: {folder.name}")

        return student_folders

    def _transcribe_student(self, student_folder: Path) -> StudentTranscriptionResult:
        """
        Transcribe all pages for a single student using parallel processing.

        Args:
            student_folder: Path to student's folder

        Returns:
            StudentTranscriptionResult with all transcription data
        """
        start_time = datetime.now()
        student_name = student_folder.name

        # Create Student object
        student = Student(name=student_name)

        # Find all image files in student folder
        image_paths = self._find_student_images(student_folder)

        if not image_paths:
            logger.warning(f"No images found for student {student_name}")
            return StudentTranscriptionResult(
                student_name=student_name,
                student=student,
                total_pages=0,
                errors=[f"No images found in {student_folder}"]
            )

        logger.info(f"Processing {len(image_paths)} images for student {student_name}")

        # Step 1: Upload all images to GCS and create tasks
        tasks = []
        errors = []

        for i, image_path in enumerate(image_paths):
            page_number = i + 1

            try:
                # Upload image to GCS
                public_url = self.storage_service.upload_image(
                    local_path=str(image_path),
                    student_name=student_name,
                    assignment_name=self.assignment_name,
                    page_number=page_number
                )

                # Create transcription task
                task = TranscriptionTask(
                    image_url=public_url,
                    student_name=student_name,
                    page_number=page_number,
                    image_path=image_path,
                    question_context=f"Student response on page {page_number} of exam"
                )
                tasks.append(task)

            except Exception as e:
                error_msg = f"Failed to upload page {page_number} ({image_path.name}): {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Step 2: Execute transcription tasks in parallel
        page_results = []
        if tasks:
            result = self.executor.execute_batch(
                tasks=tasks,
                checkpoint_name=f"transcription_{student_name}_{len(tasks)}_pages",
                resume_from_checkpoint=False  # Don't resume for individual students
            )

            # Convert task results to PageTranscription objects
            for task_result in result.task_results:
                if task_result.success:
                    data = task_result.result
                    page_results.append(PageTranscription(
                        page_number=data['page_number'],
                        image_path=Path(data['image_path']) if data['image_path'] else None,
                        transcribed_text=data['transcribed_text'],
                        confidence=data['confidence'],
                        notes=data.get('notes'),
                        processing_time=task_result.execution_time,
                        public_url=data.get('image_url'),
                        raw_llm_response=task_result.raw_response
                    ))
                else:
                    # Create failed page result
                    page_num = task_result.task_id.split('_page_')[-1]
                    page_results.append(PageTranscription(
                        page_number=int(page_num) if page_num.isdigit() else 0,
                        image_path=None,
                        transcribed_text="",
                        confidence=0.0,
                        notes=f"Transcription failed: {task_result.error}",
                        processing_time=task_result.execution_time
                    ))
                    errors.append(f"Page {page_num}: {task_result.error}")

        # Sort results by page number
        page_results.sort(key=lambda p: p.page_number)

        successful_count = sum(1 for p in page_results if p.confidence > 0)
        processing_time = (datetime.now() - start_time).total_seconds()

        # Create student transcription result
        student_result = StudentTranscriptionResult(
            student_name=student_name,
            student=student,
            pages=page_results,
            total_pages=len(image_paths),
            successful_pages=successful_count,
            failed_pages=len(image_paths) - successful_count,
            processing_time=processing_time,
            errors=errors
        )

        # Generate question-mapped answers if enabled
        if self.use_question_mapping and self.exam_spec:
            try:
                logger.info(f"Creating question-mapped answers for {student_name}")

                # Prepare page transcription data for mapping
                page_transcriptions = [
                    {
                        'page_number': page.page_number,
                        'transcribed_text': page.transcribed_text,
                        'confidence': page.confidence,
                        'image_path': str(page.image_path)
                    }
                    for page in page_results
                ]

                # Create question-mapped answers
                question_answers = create_question_mapped_answers(
                    page_transcriptions=page_transcriptions,
                    exam_spec=self.exam_spec,
                    student=student
                )

                # Validate mapping
                validation_errors = validate_question_mapping(question_answers, self.exam_spec)
                if validation_errors:
                    logger.warning(f"Question mapping validation warnings for {student_name}:")
                    for error in validation_errors:
                        logger.warning(f"  - {error}")

                # Save question-mapped results
                self._save_question_mapped_results(student_result, question_answers)

                # Get mapping stats
                mapping_stats = get_question_mapping_stats(question_answers, self.exam_spec)
                logger.info(
                    f"Question mapping complete: {mapping_stats['mapped_questions']}/{mapping_stats['expected_questions']} "
                    f"questions ({mapping_stats['mapping_coverage']:.1f}% coverage)"
                )

                # Generate HTML report if enabled
                if self.generate_html_reports and self.report_generator:
                    try:
                        student_output_dir = self.output_dir / student_name
                        report_file = student_output_dir / "transcription_report.html"

                        self.report_generator.generate_student_report(
                            student_name=student_name,
                            question_answers=question_answers,
                            output_file=report_file
                        )

                        logger.info(f"HTML report generated: {report_file}")

                    except Exception as report_error:
                        logger.error(f"Failed to generate HTML report for {student_name}: {report_error}")
                        errors.append(f"HTML report generation failed: {report_error}")

            except Exception as e:
                logger.error(f"Failed to create question-mapped answers for {student_name}: {e}")
                errors.append(f"Question mapping failed: {e}")

        return student_result

    def _find_student_images(self, student_folder: Path) -> List[Path]:
        """
        Find all image files in a student folder.

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

        return image_files

    def _transcribe_page(
        self,
        image_path: Path,
        student_name: str,
        page_number: int
    ) -> PageTranscription:
        """
        Transcribe a single page using LLM vision API.

        Args:
            image_path: Path to the image file
            student_name: Name of the student
            page_number: Page number for organization

        Returns:
            PageTranscription result

        Raises:
            TranscriptionError: If transcription fails after retries
        """
        start_time = datetime.now()
        last_exception = None

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Transcription attempt {attempt + 1}/{self.max_retries} for {image_path.name}")

                # Step 1: Upload image to GCS
                public_url = self.storage_service.upload_image(
                    local_path=str(image_path),
                    student_name=student_name,
                    assignment_name=self.assignment_name,
                    page_number=page_number
                )

                # Step 2: Get transcription prompt
                transcription_prompt = get_transcription_prompt(
                    question_text=f"Student response on page {page_number} of exam"
                )

                # Step 3: Call LLM vision API
                response_text = self.llm_client.complete_with_image(
                    prompt=transcription_prompt,
                    image_url=public_url,
                    temperature=0.1
                )

                # Step 4: Parse response
                transcription_data = self._parse_transcription_response(response_text)

                processing_time = (datetime.now() - start_time).total_seconds()

                return PageTranscription(
                    page_number=page_number,
                    image_path=image_path,
                    transcribed_text=transcription_data.get('transcribed_text', ''),
                    confidence=float(transcription_data.get('confidence', 0.0)),
                    notes=transcription_data.get('notes'),
                    processing_time=processing_time,
                    public_url=public_url,
                    raw_llm_response=response_text
                )

            except Exception as e:
                last_exception = e
                logger.warning(f"Transcription attempt {attempt + 1} failed for {image_path.name}: {e}")

                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        # All retries failed
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"All {self.max_retries} transcription attempts failed for {image_path.name}")

        return PageTranscription(
            page_number=page_number,
            image_path=image_path,
            transcribed_text="",
            confidence=0.0,
            notes=f"Transcription failed after {self.max_retries} attempts: {last_exception}",
            processing_time=processing_time
        )

    def _parse_transcription_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM transcription response.

        Args:
            response_text: Raw LLM response

        Returns:
            Dict with transcribed_text, confidence, and notes

        Raises:
            ValueError: If response format is invalid
        """
        try:
            # Try to extract JSON from response
            json_text = self._extract_json_from_response(response_text)
            data = json.loads(json_text)

            # Validate required fields
            if 'transcribed_text' not in data:
                raise ValueError("Missing 'transcribed_text' field in response")

            # Ensure confidence is between 0 and 1
            confidence = float(data.get('confidence', 0.0))
            if not 0.0 <= confidence <= 1.0:
                logger.warning(f"Invalid confidence value: {confidence}, setting to 0.5")
                confidence = 0.5

            return {
                'transcribed_text': data['transcribed_text'],
                'confidence': confidence,
                'notes': data.get('notes', '')
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response, using raw text: {e}")

            # Fallback: use raw response text
            return {
                'transcribed_text': response_text.strip(),
                'confidence': 0.3,  # Lower confidence for raw text
                'notes': f"Raw LLM response (JSON parsing failed: {e})"
            }

    def _extract_json_from_response(self, response_text: str) -> str:
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

    def _save_student_results(self, student_result: StudentTranscriptionResult) -> None:
        """
        Save transcription results for a single student.

        Args:
            student_result: StudentTranscriptionResult to save
        """
        # Create student output directory
        student_output_dir = self.output_dir / student_result.student_name
        student_output_dir.mkdir(exist_ok=True)

        # Save individual page transcriptions
        for page in student_result.pages:
            page_file = student_output_dir / f"page_{page.page_number:03d}_transcription.json"

            page_data = {
                "page_number": page.page_number,
                "image_path": str(page.image_path),
                "transcribed_text": page.transcribed_text,
                "confidence": page.confidence,
                "notes": page.notes,
                "processing_time": page.processing_time,
                "public_url": page.public_url,
                "timestamp": datetime.now().isoformat()
            }

            with open(page_file, 'w', encoding='utf-8') as f:
                json.dump(page_data, f, indent=2, ensure_ascii=False)

        # Save student summary
        summary_file = student_output_dir / "transcriptions.json"

        # Convert pages to StudentAnswer objects for consistency with domain models
        student_answers = []
        for page in student_result.pages:
            if page.transcribed_text.strip():  # Only create answers for non-empty transcriptions
                answer = StudentAnswer(
                    student=student_result.student,
                    question_id=f"page_{page.page_number}",
                    raw_text=page.transcribed_text,
                    image_paths=[str(page.image_path)],
                    confidence=page.confidence,
                    transcription_notes=page.notes
                )
                student_answers.append(answer)

        summary_data = {
            "student_name": student_result.student_name,
            "student_id": student_result.student.student_id,
            "total_pages": student_result.total_pages,
            "successful_pages": student_result.successful_pages,
            "failed_pages": student_result.failed_pages,
            "success_rate": student_result.success_rate,
            "average_confidence": student_result.average_confidence,
            "processing_time": student_result.processing_time,
            "errors": student_result.errors,
            "student_answers": [answer.to_dict() for answer in student_answers],
            "timestamp": datetime.now().isoformat()
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved transcription results for {student_result.student_name}")

    def _save_question_mapped_results(
        self,
        student_result: StudentTranscriptionResult,
        question_answers: List[StudentAnswer]
    ) -> None:
        """
        Save question-mapped transcription results for a student.

        Args:
            student_result: StudentTranscriptionResult with metadata
            question_answers: List of StudentAnswer objects mapped to questions
        """
        # Create student output directory
        student_output_dir = self.output_dir / student_result.student_name
        student_output_dir.mkdir(exist_ok=True)

        # Save question-mapped transcriptions
        question_file = student_output_dir / "question_transcriptions.json"

        # Calculate statistics
        mapping_stats = get_question_mapping_stats(question_answers, self.exam_spec) if self.exam_spec else {}

        question_data = {
            "student_name": student_result.student_name,
            "student_id": student_result.student.student_id,
            "exam_name": self.exam_spec.exam_name if self.exam_spec else "Unknown",
            "total_questions": len(self.exam_spec.questions) if self.exam_spec else 0,
            "mapped_questions": len(question_answers),
            "mapping_stats": mapping_stats,
            "timestamp": datetime.now().isoformat(),
            "question_answers": [answer.to_dict() for answer in question_answers]
        }

        with open(question_file, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved question-mapped results for {student_result.student_name}: {len(question_answers)} questions")

    def _save_summary_results(self, results: TranscriptionResults) -> None:
        """
        Save overall pipeline summary results.

        Args:
            results: TranscriptionResults to save
        """
        summary_file = self.output_dir / "summary_report.json"

        summary_data = {
            "assignment_name": self.assignment_name,
            "total_students": results.total_students,
            "successful_students": results.successful_students,
            "total_pages": results.total_pages,
            "successful_pages": results.successful_pages,
            "success_rate": results.success_rate,
            "average_confidence": results.average_confidence,
            "processing_time": results.processing_time,
            "statistics": results.statistics,
            "timestamp": datetime.now().isoformat(),
            "student_results": [
                {
                    "student_name": sr.student_name,
                    "success_rate": sr.success_rate,
                    "average_confidence": sr.average_confidence,
                    "total_pages": sr.total_pages,
                    "successful_pages": sr.successful_pages
                }
                for sr in results.student_results
            ]
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved summary results to {summary_file}")

    def _generate_statistics(self, results: TranscriptionResults) -> Dict[str, Any]:
        """
        Generate detailed statistics from transcription results.

        Args:
            results: TranscriptionResults to analyze

        Returns:
            Dictionary of statistics
        """
        all_confidences = []
        processing_times = []
        page_counts = []

        for student in results.student_results:
            page_counts.append(student.total_pages)
            processing_times.append(student.processing_time)

            for page in student.pages:
                if page.confidence > 0:
                    all_confidences.append(page.confidence)
                if page.processing_time:
                    processing_times.append(page.processing_time)

        statistics = {
            "confidence_distribution": {
                "high (0.8-1.0)": sum(1 for c in all_confidences if c >= 0.8),
                "medium (0.5-0.79)": sum(1 for c in all_confidences if 0.5 <= c < 0.8),
                "low (0.0-0.49)": sum(1 for c in all_confidences if c < 0.5)
            },
            "processing_times": {
                "average_per_page": sum(processing_times) / len(processing_times) if processing_times else 0,
                "total_pipeline": results.processing_time
            },
            "page_statistics": {
                "average_pages_per_student": sum(page_counts) / len(page_counts) if page_counts else 0,
                "min_pages": min(page_counts) if page_counts else 0,
                "max_pages": max(page_counts) if page_counts else 0
            }
        }

        return statistics

    def _display_summary(self, results: TranscriptionResults) -> None:
        """
        Display final summary of transcription results.

        Args:
            results: TranscriptionResults to display
        """
        console.print(f"\n✅ [bold green]Transcription Pipeline Complete![/bold green]")
        console.print(f"   📊 Students: {results.successful_students}/{results.total_students}")
        console.print(f"   📄 Pages: {results.successful_pages}/{results.total_pages}")
        console.print(f"   🎯 Success Rate: {results.success_rate:.1f}%")
        console.print(f"   📈 Average Confidence: {results.average_confidence:.2f}")
        console.print(f"   ⏱️  Processing Time: {results.processing_time:.1f} seconds")
        console.print(f"   💾 Results Saved: {self.output_dir}\n")

        # Show confidence distribution
        stats = results.statistics
        conf_dist = stats.get("confidence_distribution", {})
        console.print("📊 Confidence Distribution:")
        console.print(f"   High (0.8-1.0): {conf_dist.get('high (0.8-1.0)', 0)} pages")
        console.print(f"   Medium (0.5-0.79): {conf_dist.get('medium (0.5-0.79)', 0)} pages")
        console.print(f"   Low (0.0-0.49): {conf_dist.get('low (0.0-0.49)', 0)} pages\n")


# Factory and convenience functions

def create_transcription_pipeline(
    assignment_name: str = "exam_transcription",
    data_dir: Optional[Path] = None,
    use_question_mapping: bool = True,
    generate_html_reports: bool = True
) -> TranscriptionPipeline:
    """
    Create transcription pipeline with services from config.

    Args:
        assignment_name: Name of assignment for GCS organization
        data_dir: Optional data directory (uses config default if not provided)
        use_question_mapping: Whether to create question-mapped transcriptions
        generate_html_reports: Whether to generate HTML reports

    Returns:
        Configured TranscriptionPipeline
    """
    from aita.config import get_config
    from aita.services.llm.openrouter import create_openrouter_client
    from aita.services.storage import create_storage_service

    config = get_config()

    llm_client = create_openrouter_client()
    storage_service = create_storage_service()

    return TranscriptionPipeline(
        llm_client=llm_client,
        storage_service=storage_service,
        data_dir=data_dir or Path(config.data_dir),
        assignment_name=assignment_name,
        use_question_mapping=use_question_mapping,
        generate_html_reports=generate_html_reports
    )


def transcribe_all_students(
    assignment_name: str = "exam_transcription",
    data_dir: Optional[Path] = None,
    use_question_mapping: bool = True,
    generate_html_reports: bool = True
) -> TranscriptionResults:
    """
    High-level function to transcribe all students' exams.

    Args:
        assignment_name: Name of assignment for GCS organization
        data_dir: Optional data directory
        use_question_mapping: Whether to create question-mapped transcriptions
        generate_html_reports: Whether to generate HTML reports

    Returns:
        TranscriptionResults with all transcription data

    Example:
        >>> results = transcribe_all_students("BMI541_Midterm")
        >>> print(f"Transcribed {results.successful_pages} pages")
    """
    pipeline = create_transcription_pipeline(
        assignment_name, data_dir, use_question_mapping, generate_html_reports
    )
    return pipeline.transcribe_all_students()


def transcribe_single_student(
    student_folder: str,
    assignment_name: str = "exam_transcription",
    data_dir: Optional[Path] = None,
    use_question_mapping: bool = True,
    generate_html_reports: bool = True
) -> StudentTranscriptionResult:
    """
    Transcribe a single student's exam.

    Args:
        student_folder: Path to student folder (relative to grouped directory)
        assignment_name: Name of assignment for GCS organization
        data_dir: Optional data directory
        use_question_mapping: Whether to create question-mapped transcriptions
        generate_html_reports: Whether to generate HTML reports

    Returns:
        StudentTranscriptionResult for the student

    Example:
        >>> result = transcribe_single_student("Mei, Elizabeth")
        >>> print(f"Transcribed {result.successful_pages} pages")
    """
    pipeline = create_transcription_pipeline(
        assignment_name, data_dir, use_question_mapping, generate_html_reports
    )

    # Find the student folder
    grouped_dir = (data_dir or Path(pipeline.data_dir)) / "grouped"
    student_path = grouped_dir / student_folder

    if not student_path.exists():
        raise TranscriptionError(f"Student folder not found: {student_path}")

    return pipeline._transcribe_student(student_path)