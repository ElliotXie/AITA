"""
Grading Pipeline

Grades student exam responses using LLM and generated rubrics.
Processes transcribed student answers and produces detailed grading results.
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

from aita.domain.models import (
    Student, Grade, ExamSpec, Question, AnswerKey,
    Rubric, StudentAnswer, GradeLevel
)
from aita.services.llm.base import BaseLLMClient
from aita.services.llm.openrouter import OpenRouterClient
from aita.services.llm.cost_tracker import get_global_tracker
from aita.utils.parallel_executor import ParallelLLMExecutor
from aita.utils.llm_task_implementations import (
    QuestionGradingTask,
    create_grading_tasks
)

logger = logging.getLogger(__name__)
console = Console()


def sort_question_id(question_id: str) -> Tuple[int, str]:
    """
    Sort key function for question IDs like '1a', '1b', '2a', '10b', etc.

    Args:
        question_id: Question identifier (e.g., '1a', '2b', '10c')

    Returns:
        Tuple of (numeric_part, letter_part) for sorting
    """
    match = re.match(r'(\d+)([a-z]*)', question_id.lower())
    if match:
        num_part = int(match.group(1))
        letter_part = match.group(2) or ''
        return (num_part, letter_part)
    return (0, question_id)


class GradingError(Exception):
    """Raised when grading fails."""
    pass


@dataclass
class StudentGradingResult:
    """Result of grading a single student."""
    student_name: str
    total_score: float
    total_possible: float
    percentage: float
    question_grades: List[Grade] = field(default_factory=list)
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def grade_letter(self) -> str:
        """Calculate letter grade based on percentage."""
        if self.percentage >= 97: return "A+"
        elif self.percentage >= 93: return "A"
        elif self.percentage >= 90: return "A-"
        elif self.percentage >= 87: return "B+"
        elif self.percentage >= 83: return "B"
        elif self.percentage >= 80: return "B-"
        elif self.percentage >= 77: return "C+"
        elif self.percentage >= 73: return "C"
        elif self.percentage >= 70: return "C-"
        elif self.percentage >= 67: return "D+"
        elif self.percentage >= 60: return "D"
        else: return "F"

    def to_dict(self) -> Dict[str, Any]:
        # Sort question grades by question ID
        sorted_grades = sorted(self.question_grades, key=lambda g: sort_question_id(g.question_id))

        return {
            "student_name": self.student_name,
            "total_score": self.total_score,
            "total_possible": self.total_possible,
            "percentage": self.percentage,
            "grade_letter": self.grade_letter,
            "question_grades": [grade.to_dict() for grade in sorted_grades],
            "processing_time": self.processing_time,
            "errors": self.errors,
            "graded_at": datetime.now().isoformat()
        }


@dataclass
class BatchGradingResult:
    """Result of batch grading multiple students."""
    assignment_name: str
    total_students: int
    successful_students: int
    total_questions: int
    average_score: float
    student_results: List[StudentGradingResult] = field(default_factory=list)
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_students == 0:
            return 0.0
        return (self.successful_students / self.total_students) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignment_name": self.assignment_name,
            "total_students": self.total_students,
            "successful_students": self.successful_students,
            "success_rate": self.success_rate,
            "total_questions": self.total_questions,
            "average_score": self.average_score,
            "student_results": [result.to_dict() for result in self.student_results],
            "processing_time": self.processing_time,
            "errors": self.errors,
            "timestamp": datetime.now().isoformat()
        }


class GradingPipeline:
    """
    Main grading pipeline for processing student exam responses.

    Integrates rubrics, answer keys, and transcribed student responses
    to produce comprehensive grading results using LLM evaluation.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        data_dir: Path,
        assignment_name: str = "exam_grading"
    ):
        """
        Initialize grading pipeline.

        Args:
            llm_client: LLM client for grading tasks
            data_dir: Data directory containing all exam data
            assignment_name: Name of the assignment for organization
        """
        self.llm_client = llm_client
        self.data_dir = data_dir
        self.assignment_name = assignment_name

        # Set up directory paths
        self.intermediate_dir = data_dir.parent / "intermediateproduct"
        self.rubrics_dir = self.intermediate_dir / "rubrics"
        self.transcription_dir = self.intermediate_dir / "transcription_results"
        self.grading_dir = self.intermediate_dir / "grading_results"

        # Create output directory if it doesn't exist
        self.grading_dir.mkdir(parents=True, exist_ok=True)
        (self.grading_dir / "students").mkdir(parents=True, exist_ok=True)

        # Initialize data containers
        self.exam_spec: Optional[ExamSpec] = None
        self.rubrics: Dict[str, Rubric] = {}
        self.answer_keys: Dict[str, AnswerKey] = {}
        self.questions: Dict[str, Question] = {}

        # Initialize executor for parallel processing
        from aita.config import get_config
        config = get_config()

        self.executor = ParallelLLMExecutor(
            llm_client=llm_client,
            max_workers=config.parallel_execution.max_workers,
            rate_limit=config.parallel_execution.rate_limit_rps,
            enable_checkpointing=config.parallel_execution.enable_checkpointing,
            checkpoint_interval=config.parallel_execution.checkpoint_interval,
            max_retries=3,
            show_progress=config.parallel_execution.show_progress
        )

    def load_grading_data(self) -> None:
        """
        Load all required data for grading: rubrics, answer keys, and exam spec.

        Raises:
            GradingError: If required files are missing or invalid
        """
        try:
            # Load exam specification
            exam_spec_file = self.data_dir / "results" / "exam_spec.json"
            if not exam_spec_file.exists():
                raise GradingError(f"Exam specification not found: {exam_spec_file}")

            self.exam_spec = ExamSpec.load_from_file(exam_spec_file)
            self.questions = {q.question_id: q for q in self.exam_spec.questions}

            # Load rubrics
            rubrics_file = self.rubrics_dir / "generated_rubrics.json"
            if not rubrics_file.exists():
                raise GradingError(f"Rubrics not found: {rubrics_file}")

            with open(rubrics_file, 'r') as f:
                rubrics_data = json.load(f)
                self.rubrics = {
                    r["question_id"]: Rubric.from_dict(r)
                    for r in rubrics_data["rubrics"]
                }

            # Load answer keys
            answer_keys_file = self.rubrics_dir / "generated_answer_keys.json"
            if not answer_keys_file.exists():
                raise GradingError(f"Answer keys not found: {answer_keys_file}")

            with open(answer_keys_file, 'r') as f:
                keys_data = json.load(f)
                self.answer_keys = {
                    k["question_id"]: AnswerKey.from_dict(k)
                    for k in keys_data["answer_keys"]
                }

            # Validate data consistency
            self._validate_grading_data()

            logger.info(f"Loaded grading data: {len(self.questions)} questions, "
                       f"{len(self.rubrics)} rubrics, {len(self.answer_keys)} answer keys")

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise GradingError(f"Failed to load grading data: {e}") from e

    def _validate_grading_data(self) -> None:
        """Validate that all required grading data is consistent."""
        missing_rubrics = []
        missing_answer_keys = []

        for question_id in self.questions.keys():
            if question_id not in self.rubrics:
                missing_rubrics.append(question_id)
            if question_id not in self.answer_keys:
                missing_answer_keys.append(question_id)

        if missing_rubrics:
            raise GradingError(f"Missing rubrics for questions: {missing_rubrics}")
        if missing_answer_keys:
            raise GradingError(f"Missing answer keys for questions: {missing_answer_keys}")

        # Validate point consistency
        for question_id, question in self.questions.items():
            rubric = self.rubrics[question_id]
            if abs(question.points - rubric.total_points) > 0.01:
                raise GradingError(
                    f"Point mismatch for {question_id}: "
                    f"question={question.points}, rubric={rubric.total_points}"
                )

    def load_student_transcriptions(self, student_name: str) -> Dict[str, Any]:
        """
        Load transcription data for a specific student.

        Args:
            student_name: Name of the student

        Returns:
            Dict containing student answers and metadata

        Raises:
            GradingError: If transcription data is not found or invalid
        """
        student_dir = self.transcription_dir / student_name
        transcription_file = student_dir / "question_transcriptions.json"

        if not transcription_file.exists():
            raise GradingError(f"Transcription data not found for {student_name}: {transcription_file}")

        try:
            with open(transcription_file, 'r') as f:
                data = json.load(f)

            # Extract answers and metadata
            student_answers = {}
            answer_confidences = {}
            transcription_notes = {}

            for qa in data["question_answers"]:
                question_id = qa["question_id"]
                student_answers[question_id] = qa["raw_text"]
                answer_confidences[question_id] = qa.get("confidence", 0.0)
                transcription_notes[question_id] = qa.get("transcription_notes", "")

            return {
                "student_answers": student_answers,
                "answer_confidences": answer_confidences,
                "transcription_notes": transcription_notes,
                "metadata": data.get("metadata", {})
            }

        except (json.JSONDecodeError, KeyError) as e:
            raise GradingError(f"Invalid transcription data for {student_name}: {e}") from e

    def grade_student(self, student_name: str) -> StudentGradingResult:
        """
        Grade all questions for a single student.

        Args:
            student_name: Name of the student to grade

        Returns:
            StudentGradingResult with grading results and statistics
        """
        start_time = datetime.now()
        errors = []

        try:
            # Load student transcriptions
            transcription_data = self.load_student_transcriptions(student_name)
            student_answers = transcription_data["student_answers"]
            answer_confidences = transcription_data["answer_confidences"]
            transcription_notes = transcription_data["transcription_notes"]

            # Create grading tasks
            tasks = create_grading_tasks(
                student_answers=student_answers,
                student_name=student_name,
                answer_keys=self.answer_keys,
                rubrics=self.rubrics,
                questions=self.questions,
                answer_confidences=answer_confidences,
                transcription_notes=transcription_notes
            )

            if not tasks:
                raise GradingError(f"No grading tasks created for {student_name}")

            logger.info(f"Grading {len(tasks)} questions for {student_name}")

            # Execute grading tasks in parallel
            results = self.executor.execute_batch(tasks)

            # Process results
            question_grades = []
            for task, result in zip(tasks, results.task_results):
                if result.success and result.result:
                    question_grades.append(result.result)
                else:
                    error_msg = f"Failed to grade {task.question.question_id}: {result.error}"
                    errors.append(error_msg)
                    logger.warning(error_msg)

            # Calculate totals
            total_score = sum(grade.points_earned for grade in question_grades)
            total_possible = sum(grade.points_possible for grade in question_grades)
            percentage = (total_score / total_possible * 100) if total_possible > 0 else 0.0

            processing_time = (datetime.now() - start_time).total_seconds()

            result = StudentGradingResult(
                student_name=student_name,
                total_score=total_score,
                total_possible=total_possible,
                percentage=percentage,
                question_grades=question_grades,
                processing_time=processing_time,
                errors=errors
            )

            # Save individual student results
            self._save_student_results(result)

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Failed to grade {student_name}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

            return StudentGradingResult(
                student_name=student_name,
                total_score=0.0,
                total_possible=0.0,
                percentage=0.0,
                processing_time=processing_time,
                errors=errors
            )

    def grade_all_students(self) -> BatchGradingResult:
        """
        Grade all students in the transcription results directory.

        Returns:
            BatchGradingResult with overall grading statistics
        """
        start_time = datetime.now()

        # Find all student directories
        if not self.transcription_dir.exists():
            raise GradingError(f"Transcription directory not found: {self.transcription_dir}")

        student_dirs = [d for d in self.transcription_dir.iterdir() if d.is_dir()]

        if not student_dirs:
            raise GradingError("No student transcription data found")

        logger.info(f"Starting batch grading for {len(student_dirs)} students")

        student_results = []
        successful_students = 0
        errors = []

        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:

            task_id = progress.add_task(
                f"Grading {len(student_dirs)} students...",
                total=len(student_dirs)
            )

            for student_dir in student_dirs:
                student_name = student_dir.name
                progress.update(task_id, description=f"Grading {student_name}...")

                try:
                    result = self.grade_student(student_name)
                    student_results.append(result)

                    if not result.errors:
                        successful_students += 1

                    progress.advance(task_id)

                except Exception as e:
                    error_msg = f"Failed to process {student_name}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    progress.advance(task_id)

        # Calculate overall statistics
        total_questions = len(self.questions) if self.questions else 0
        average_score = 0.0
        if student_results:
            valid_scores = [r.percentage for r in student_results if r.percentage > 0]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        processing_time = (datetime.now() - start_time).total_seconds()

        batch_result = BatchGradingResult(
            assignment_name=self.assignment_name,
            total_students=len(student_dirs),
            successful_students=successful_students,
            total_questions=total_questions,
            average_score=average_score,
            student_results=student_results,
            processing_time=processing_time,
            errors=errors
        )

        # Save batch results
        self._save_batch_results(batch_result)

        logger.info(f"Batch grading completed: {successful_students}/{len(student_dirs)} students")

        return batch_result

    def _save_student_results(self, result: StudentGradingResult) -> None:
        """Save individual student grading results."""
        student_dir = self.grading_dir / "students" / result.student_name
        student_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed grades
        detailed_file = student_dir / "detailed_grades.json"
        with open(detailed_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # Save summary with sorted question scores
        sorted_grades = sorted(result.question_grades, key=lambda g: sort_question_id(g.question_id))

        summary_data = {
            "student_name": result.student_name,
            "assignment": self.assignment_name,
            "total_score": result.total_score,
            "total_possible": result.total_possible,
            "percentage": result.percentage,
            "grade_letter": result.grade_letter,
            "graded_at": datetime.now().isoformat(),
            "question_scores": {
                grade.question_id: {
                    "earned": grade.points_earned,
                    "possible": grade.points_possible,
                    "percentage": grade.percentage
                }
                for grade in sorted_grades
            }
        }

        summary_file = student_dir / "grade_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

    def _save_batch_results(self, result: BatchGradingResult) -> None:
        """Save batch grading results and statistics."""
        # Save summary report
        summary_file = self.grading_dir / "summary_report.json"
        with open(summary_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # Save grade distribution analysis
        grade_distribution = self._calculate_grade_distribution(result.student_results)
        distribution_file = self.grading_dir / "grade_distribution.json"
        with open(distribution_file, 'w') as f:
            json.dump(grade_distribution, f, indent=2, ensure_ascii=False)

    def _calculate_grade_distribution(self, student_results: List[StudentGradingResult]) -> Dict[str, Any]:
        """Calculate grade distribution statistics."""
        if not student_results:
            return {}

        # Calculate letter grade distribution
        letter_grades = [result.grade_letter for result in student_results if result.percentage > 0]
        grade_counts = {}
        for grade in letter_grades:
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        # Calculate score statistics
        scores = [result.percentage for result in student_results if result.percentage > 0]

        if scores:
            scores.sort()
            n = len(scores)
            median = scores[n//2] if n % 2 == 1 else (scores[n//2-1] + scores[n//2]) / 2

            distribution = {
                "total_students": len(student_results),
                "graded_students": len(scores),
                "average_score": sum(scores) / len(scores),
                "median_score": median,
                "min_score": min(scores),
                "max_score": max(scores),
                "letter_grade_distribution": grade_counts,
                "score_ranges": {
                    "A_range": len([s for s in scores if s >= 90]),
                    "B_range": len([s for s in scores if 80 <= s < 90]),
                    "C_range": len([s for s in scores if 70 <= s < 80]),
                    "D_range": len([s for s in scores if 60 <= s < 70]),
                    "F_range": len([s for s in scores if s < 60])
                }
            }
        else:
            distribution = {
                "total_students": len(student_results),
                "graded_students": 0,
                "error": "No valid scores found"
            }

        return distribution


# Factory functions for easy pipeline creation

def create_grading_pipeline(
    data_dir: Path,
    assignment_name: str = "exam_grading"
) -> GradingPipeline:
    """
    Create a grading pipeline with default LLM client.

    Args:
        data_dir: Data directory containing exam data
        assignment_name: Assignment name for organization

    Returns:
        Configured GradingPipeline instance
    """
    llm_client = OpenRouterClient()
    return GradingPipeline(
        llm_client=llm_client,
        data_dir=data_dir,
        assignment_name=assignment_name
    )


def grade_all_students(
    assignment_name: str = "exam_grading",
    data_dir: Optional[Path] = None
) -> BatchGradingResult:
    """
    Convenience function to grade all students with default settings.

    Args:
        assignment_name: Assignment name for organization
        data_dir: Data directory (defaults to ./data)

    Returns:
        BatchGradingResult with grading statistics
    """
    if data_dir is None:
        data_dir = Path.cwd() / "data"

    pipeline = create_grading_pipeline(data_dir, assignment_name)
    pipeline.load_grading_data()

    return pipeline.grade_all_students()


def grade_single_student(
    student_name: str,
    assignment_name: str = "exam_grading",
    data_dir: Optional[Path] = None
) -> StudentGradingResult:
    """
    Convenience function to grade a single student.

    Args:
        student_name: Name of the student to grade
        assignment_name: Assignment name for organization
        data_dir: Data directory (defaults to ./data)

    Returns:
        StudentGradingResult for the specified student
    """
    if data_dir is None:
        data_dir = Path.cwd() / "data"

    pipeline = create_grading_pipeline(data_dir, assignment_name)
    pipeline.load_grading_data()

    return pipeline.grade_student(student_name)