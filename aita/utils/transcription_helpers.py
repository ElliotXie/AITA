"""
Transcription Helper Utilities

Helper functions for mapping transcriptions to questions, loading exam specs,
and validating transcription results.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import re

from aita.domain.models import ExamSpec, Question, StudentAnswer

logger = logging.getLogger(__name__)


def load_exam_spec(data_dir: Path) -> Optional[ExamSpec]:
    """
    Load ExamSpec from question extraction results.

    Args:
        data_dir: Base data directory

    Returns:
        ExamSpec if found, None otherwise
    """
    # Try multiple possible locations
    possible_paths = [
        data_dir / "results" / "exam_spec.json",
        data_dir / "intermediateproduct" / "question_extraction" / "exam_spec.json",
    ]

    for exam_spec_path in possible_paths:
        if exam_spec_path.exists():
            try:
                logger.info(f"Loading ExamSpec from {exam_spec_path}")
                exam_spec = ExamSpec.load_from_file(exam_spec_path)
                logger.info(f"Loaded ExamSpec with {len(exam_spec.questions)} questions")
                return exam_spec
            except Exception as e:
                logger.error(f"Failed to load ExamSpec from {exam_spec_path}: {e}")
                continue

    logger.warning("No ExamSpec found in any expected location")
    logger.debug(f"Searched paths: {[str(p) for p in possible_paths]}")
    return None


def map_page_to_questions(page_number: int, exam_spec: ExamSpec) -> List[Question]:
    """
    Find all questions that appear on a given page.

    Args:
        page_number: Page number (1-indexed)
        exam_spec: Exam specification with questions

    Returns:
        List of Question objects on that page
    """
    # Validate page_number
    if page_number < 1:
        logger.warning(f"Invalid page number: {page_number}. Must be >= 1.")
        return []

    questions_on_page = []

    for question in exam_spec.questions:
        # Skip questions without page numbers
        if question.page_number is None:
            logger.warning(
                f"Question {question.question_id} has no page_number, skipping in page mapping"
            )
            continue

        if question.page_number == page_number:
            questions_on_page.append(question)

    # Sort by question_id for consistent ordering
    questions_on_page.sort(key=lambda q: q.question_id)

    logger.debug(f"Found {len(questions_on_page)} questions on page {page_number}")

    return questions_on_page


def extract_question_text_from_page(
    full_page_text: str,
    question: Question,
    page_questions: List[Question]
) -> str:
    """
    Extract the transcribed answer for a specific question from full page text.

    This function attempts to intelligently split page text into question-specific
    portions when multiple questions appear on the same page.

    Args:
        full_page_text: Complete transcribed text from the page
        question: The specific question to extract answer for
        page_questions: All questions on this page (for context)

    Returns:
        Extracted text for this question (or full page text if only one question)
    """
    # If only one question on page, return full text
    if len(page_questions) == 1:
        return full_page_text.strip()

    # Multiple questions on page - try to split intelligently
    # Sort questions by question_id to get sequential order
    sorted_questions = sorted(page_questions, key=lambda q: q.question_id)

    # Find position of current question
    try:
        question_index = sorted_questions.index(question)
    except ValueError:
        logger.warning(f"Question {question.question_id} not found in page questions")
        return full_page_text.strip()

    # Look for question markers in the text (e.g., "1a.", "2.", "Question 3b:")
    question_markers = _find_question_markers(full_page_text, sorted_questions)

    if question_markers:
        # Extract text between this question's marker and the next
        current_marker = question_markers.get(question.question_id)

        if current_marker is not None:
            # Find next question marker
            next_marker_pos = None
            for i in range(question_index + 1, len(sorted_questions)):
                next_q_id = sorted_questions[i].question_id
                if next_q_id in question_markers:
                    next_marker_pos = question_markers[next_q_id]
                    break

            if next_marker_pos is not None:
                # Extract between markers
                extracted = full_page_text[current_marker:next_marker_pos].strip()
                logger.debug(f"Extracted {len(extracted)} chars for Q{question.question_id}")
                return extracted
            else:
                # Extract from marker to end of page
                extracted = full_page_text[current_marker:].strip()
                logger.debug(f"Extracted {len(extracted)} chars for Q{question.question_id} (to end)")
                return extracted

    # Fallback: Simple split by number of questions
    logger.debug(f"Using fallback split for Q{question.question_id}")
    lines = full_page_text.split('\n')
    lines_per_question = max(1, len(lines) // len(page_questions))

    start_line = question_index * lines_per_question
    end_line = start_line + lines_per_question if question_index < len(sorted_questions) - 1 else len(lines)

    return '\n'.join(lines[start_line:end_line]).strip()


def _find_question_markers(text: str, questions: List[Question]) -> Dict[str, int]:
    """
    Find positions of question markers in transcribed text.

    This function is resilient to malformed question IDs and regex errors.
    It tries multiple patterns to find question markers and gracefully handles failures.

    Args:
        text: Full page transcribed text
        questions: List of questions expected on this page

    Returns:
        Dict mapping question_id to character position in text.
        Returns empty dict if no markers found.
    """
    markers = {}

    for question in questions:
        # Validate question ID
        if not question.question_id:
            logger.warning("Skipping question with empty ID")
            continue

        try:
            # Sanitize and escape special regex characters
            safe_id = re.escape(str(question.question_id).strip())

            # Try various marker patterns in order of specificity
            patterns = [
                rf'\b{safe_id}[\.\)\:]',  # "1a.", "1a)", "1a:"
                rf'\bQuestion\s+{safe_id}\b',  # "Question 1a"
                rf'\bQ\s*{safe_id}\b',  # "Q1a", "Q 1a"
            ]

            for pattern in patterns:
                try:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        markers[question.question_id] = match.start()
                        logger.debug(
                            f"Found marker for Q{question.question_id} "
                            f"at position {match.start()} using pattern: {pattern}"
                        )
                        break
                except re.error as e:
                    logger.warning(
                        f"Regex error for pattern '{pattern}' "
                        f"with question {question.question_id}: {e}"
                    )
                    continue
            else:
                logger.debug(f"No marker found for Q{question.question_id}")

        except Exception as e:
            logger.error(
                f"Unexpected error processing question {question.question_id}: {e}",
                exc_info=True
            )
            continue

    return markers


def validate_question_mapping(
    student_answers: List[StudentAnswer],
    exam_spec: ExamSpec
) -> List[str]:
    """
    Validate that student answers are properly mapped to exam questions.

    Args:
        student_answers: List of student answer objects
        exam_spec: Exam specification

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Get all expected question IDs
    expected_question_ids = {q.question_id for q in exam_spec.questions}

    # Get all answer question IDs
    answer_question_ids = {ans.question_id for ans in student_answers}

    # Check for missing questions
    missing_questions = expected_question_ids - answer_question_ids
    if missing_questions:
        errors.append(f"Missing answers for questions: {sorted(missing_questions)}")

    # Check for unexpected questions
    unexpected_questions = answer_question_ids - expected_question_ids
    if unexpected_questions:
        # Filter out page-based IDs (e.g., "page_1")
        unexpected_non_page = [q for q in unexpected_questions if not q.startswith("page_")]
        if unexpected_non_page:
            errors.append(f"Unexpected question IDs: {sorted(unexpected_non_page)}")

    # Check for duplicate answers
    answer_ids_list = [ans.question_id for ans in student_answers]
    duplicates = [qid for qid in answer_ids_list if answer_ids_list.count(qid) > 1]
    if duplicates:
        errors.append(f"Duplicate answers for questions: {sorted(set(duplicates))}")

    # Check for empty answers
    empty_answers = [ans.question_id for ans in student_answers if not ans.raw_text.strip()]
    if empty_answers:
        errors.append(f"Empty transcriptions for questions: {sorted(empty_answers)}")

    return errors


def create_question_mapped_answers(
    page_transcriptions: List[Dict[str, Any]],
    exam_spec: ExamSpec,
    student
) -> List[StudentAnswer]:
    """
    Create question-mapped StudentAnswer objects from page transcriptions.

    Args:
        page_transcriptions: List of page transcription results
        exam_spec: Exam specification with questions
        student: Student object

    Returns:
        List of StudentAnswer objects mapped to specific questions
    """
    question_answers = []

    # Organize pages by page number
    pages_by_number = {}
    for page_data in page_transcriptions:
        page_num = page_data['page_number']
        pages_by_number[page_num] = page_data

    # Process each question in exam
    for question in exam_spec.questions:
        if question.page_number is None:
            logger.warning(f"Question {question.question_id} has no page number, skipping")
            continue

        page_num = question.page_number

        if page_num not in pages_by_number:
            logger.warning(f"Page {page_num} not found for question {question.question_id}")
            continue

        page_data = pages_by_number[page_num]
        full_page_text = page_data.get('transcribed_text', '')

        # Get all questions on this page
        page_questions = map_page_to_questions(page_num, exam_spec)

        # Extract question-specific text
        question_text = extract_question_text_from_page(
            full_page_text=full_page_text,
            question=question,
            page_questions=page_questions
        )

        # Create StudentAnswer
        answer = StudentAnswer(
            student=student,
            question_id=question.question_id,
            raw_text=question_text,
            image_paths=[page_data.get('image_path', '')],
            confidence=page_data.get('confidence', 0.0),
            transcription_notes=f"Extracted from page {page_num}"
        )

        question_answers.append(answer)
        logger.debug(f"Created answer for Q{question.question_id}: {len(question_text)} chars")

    logger.info(f"Created {len(question_answers)} question-mapped answers")

    return question_answers


def get_question_mapping_stats(
    student_answers: List[StudentAnswer],
    exam_spec: ExamSpec
) -> Dict[str, Any]:
    """
    Generate statistics about question mapping quality.

    Args:
        student_answers: List of student answers
        exam_spec: Exam specification

    Returns:
        Dictionary with mapping statistics
    """
    expected_count = len(exam_spec.questions)
    actual_count = len([a for a in student_answers if not a.question_id.startswith("page_")])

    validation_errors = validate_question_mapping(student_answers, exam_spec)

    # Calculate average answer length
    answer_lengths = [len(a.raw_text) for a in student_answers if a.raw_text]
    avg_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0

    # Calculate average confidence
    confidences = [a.confidence for a in student_answers if a.confidence is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    stats = {
        "expected_questions": expected_count,
        "mapped_questions": actual_count,
        "mapping_coverage": (actual_count / expected_count * 100) if expected_count > 0 else 0,
        "validation_errors": validation_errors,
        "average_answer_length": avg_length,
        "average_confidence": avg_confidence,
        "has_errors": len(validation_errors) > 0
    }

    return stats
