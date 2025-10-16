"""
Unit tests for transcription helper utilities.

Tests the question mapping, text extraction, and validation functions
used in the enhanced transcription pipeline.
"""

import pytest
from pathlib import Path
from typing import List

from aita.utils.transcription_helpers import (
    load_exam_spec,
    map_page_to_questions,
    extract_question_text_from_page,
    validate_question_mapping,
    create_question_mapped_answers,
    get_question_mapping_stats,
    _find_question_markers
)
from aita.domain.models import (
    ExamSpec,
    Question,
    QuestionType,
    Student,
    StudentAnswer
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_exam_spec():
    """Create a sample ExamSpec for testing."""
    return ExamSpec(
        exam_name="Test Exam",
        total_pages=3,
        questions=[
            Question(
                question_id="1a",
                question_text="What is 2+2?",
                points=5.0,
                question_type=QuestionType.SHORT_ANSWER,
                page_number=1
            ),
            Question(
                question_id="1b",
                question_text="What is 3+3?",
                points=5.0,
                question_type=QuestionType.SHORT_ANSWER,
                page_number=1
            ),
            Question(
                question_id="2",
                question_text="Explain calculus.",
                points=10.0,
                question_type=QuestionType.LONG_ANSWER,
                page_number=2
            ),
            Question(
                question_id="3a",
                question_text="Draw a diagram.",
                points=7.0,
                question_type=QuestionType.DIAGRAM,
                page_number=3
            ),
        ]
    )


@pytest.fixture
def sample_student():
    """Create a sample Student."""
    return Student(name="Test Student", student_id="12345")


@pytest.fixture
def sample_page_text_with_markers():
    """Sample page text with clear question markers."""
    return """
    1a. The answer to 2+2 is 4.
    This is a correct answer.

    1b. The answer to 3+3 is 6.
    Another correct answer here.
    """


@pytest.fixture
def sample_page_text_no_markers():
    """Sample page text without question markers."""
    return """
    The answer to the first question is 4.
    I think this is correct.

    For the second question, the answer is 6.
    I'm confident about this one.
    """


# ============================================================================
# TEST: map_page_to_questions
# ============================================================================

def test_map_page_to_questions_single_page(sample_exam_spec):
    """Test mapping questions to a page with multiple questions."""
    questions = map_page_to_questions(1, sample_exam_spec)

    assert len(questions) == 2
    assert questions[0].question_id == "1a"
    assert questions[1].question_id == "1b"


def test_map_page_to_questions_single_question_page(sample_exam_spec):
    """Test mapping questions to a page with one question."""
    questions = map_page_to_questions(2, sample_exam_spec)

    assert len(questions) == 1
    assert questions[0].question_id == "2"


def test_map_page_to_questions_invalid_page(sample_exam_spec):
    """Test mapping with invalid page number returns empty list."""
    questions = map_page_to_questions(0, sample_exam_spec)
    assert len(questions) == 0

    questions = map_page_to_questions(-1, sample_exam_spec)
    assert len(questions) == 0


def test_map_page_to_questions_nonexistent_page(sample_exam_spec):
    """Test mapping with page that has no questions."""
    questions = map_page_to_questions(99, sample_exam_spec)
    assert len(questions) == 0


def test_map_page_to_questions_with_none_page_numbers():
    """Test that questions with None page_number are skipped."""
    exam_spec = ExamSpec(
        exam_name="Test",
        total_pages=2,
        questions=[
            Question(
                question_id="1",
                question_text="Test",
                points=5.0,
                page_number=1
            ),
            Question(
                question_id="2",
                question_text="Test",
                points=5.0,
                page_number=None  # This should be skipped
            ),
            Question(
                question_id="3",
                question_text="Test",
                points=5.0,
                page_number=1
            ),
        ]
    )

    questions = map_page_to_questions(1, exam_spec)
    assert len(questions) == 2
    assert all(q.question_id in ["1", "3"] for q in questions)


# ============================================================================
# TEST: _find_question_markers
# ============================================================================

def test_find_question_markers_standard_format(sample_exam_spec):
    """Test finding markers with standard format (1a., 1b.)."""
    text = "1a. Answer one here.\n1b. Answer two here."

    page_questions = map_page_to_questions(1, sample_exam_spec)
    markers = _find_question_markers(text, page_questions)

    assert "1a" in markers
    assert "1b" in markers
    assert markers["1a"] < markers["1b"]


def test_find_question_markers_question_prefix(sample_exam_spec):
    """Test finding markers with 'Question' prefix."""
    text = "Question 1a: Answer here.\nQuestion 1b: Another answer."

    page_questions = map_page_to_questions(1, sample_exam_spec)
    markers = _find_question_markers(text, page_questions)

    assert "1a" in markers
    assert "1b" in markers


def test_find_question_markers_q_prefix(sample_exam_spec):
    """Test finding markers with 'Q' prefix."""
    text = "Q1a - Answer here.\nQ1b - Another answer."

    page_questions = map_page_to_questions(1, sample_exam_spec)
    markers = _find_question_markers(text, page_questions)

    assert "1a" in markers
    assert "1b" in markers


def test_find_question_markers_no_markers(sample_exam_spec):
    """Test when no markers are found."""
    text = "Just some random text without question markers."

    page_questions = map_page_to_questions(1, sample_exam_spec)
    markers = _find_question_markers(text, page_questions)

    assert len(markers) == 0


def test_find_question_markers_with_special_chars():
    """Test handling special characters in question IDs."""
    exam_spec = ExamSpec(
        exam_name="Test",
        total_pages=1,
        questions=[
            Question(
                question_id="2(i)",
                question_text="Test",
                points=5.0,
                page_number=1
            ),
        ]
    )

    text = "2(i). Answer here."
    markers = _find_question_markers(text, exam_spec.questions)

    # Should still find it due to re.escape
    assert "2(i)" in markers


def test_find_question_markers_empty_question_id():
    """Test handling questions with empty IDs."""
    questions = [
        Question(
            question_id="",
            question_text="Test",
            points=5.0,
            page_number=1
        )
    ]

    text = "Some text"
    markers = _find_question_markers(text, questions)

    # Should not crash, just skip
    assert len(markers) == 0


# ============================================================================
# TEST: extract_question_text_from_page
# ============================================================================

def test_extract_question_text_single_question(sample_exam_spec):
    """Test extracting text when only one question on page."""
    full_text = "This is the complete answer to the only question on this page."

    page_questions = map_page_to_questions(2, sample_exam_spec)  # Page 2 has only Q2
    question = page_questions[0]

    extracted = extract_question_text_from_page(full_text, question, page_questions)

    assert extracted == full_text.strip()


def test_extract_question_text_with_markers(sample_exam_spec, sample_page_text_with_markers):
    """Test extracting text when markers are present."""
    page_questions = map_page_to_questions(1, sample_exam_spec)

    # Extract first question
    q1a = page_questions[0]
    text_1a = extract_question_text_from_page(
        sample_page_text_with_markers, q1a, page_questions
    )

    assert "2+2 is 4" in text_1a
    assert "3+3 is 6" not in text_1a

    # Extract second question
    q1b = page_questions[1]
    text_1b = extract_question_text_from_page(
        sample_page_text_with_markers, q1b, page_questions
    )

    assert "3+3 is 6" in text_1b
    assert "2+2 is 4" not in text_1b


def test_extract_question_text_fallback_split(sample_exam_spec, sample_page_text_no_markers):
    """Test fallback to line-based splitting when no markers found."""
    page_questions = map_page_to_questions(1, sample_exam_spec)

    # Should use heuristic split
    q1a = page_questions[0]
    text_1a = extract_question_text_from_page(
        sample_page_text_no_markers, q1a, page_questions
    )

    # Should get approximately first half
    assert len(text_1a) > 0


def test_extract_question_text_question_not_on_page(sample_exam_spec):
    """Test extracting for a question not on this page."""
    full_text = "Some text"

    page_questions = map_page_to_questions(1, sample_exam_spec)
    wrong_question = Question(
        question_id="99",
        question_text="Wrong",
        points=1.0,
        page_number=99
    )

    # Should return full text as fallback
    extracted = extract_question_text_from_page(full_text, wrong_question, page_questions)
    assert extracted == full_text.strip()


# ============================================================================
# TEST: validate_question_mapping
# ============================================================================

def test_validate_question_mapping_complete(sample_exam_spec, sample_student):
    """Test validation with all questions answered."""
    student_answers = [
        StudentAnswer(
            student=sample_student,
            question_id="1a",
            raw_text="Answer 1a"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="1b",
            raw_text="Answer 1b"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="2",
            raw_text="Answer 2"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="3a",
            raw_text="Answer 3a"
        ),
    ]

    errors = validate_question_mapping(student_answers, sample_exam_spec)
    assert len(errors) == 0


def test_validate_question_mapping_missing_questions(sample_exam_spec, sample_student):
    """Test validation catches missing questions."""
    student_answers = [
        StudentAnswer(
            student=sample_student,
            question_id="1a",
            raw_text="Answer 1a"
        ),
        # Missing 1b, 2, 3a
    ]

    errors = validate_question_mapping(student_answers, sample_exam_spec)

    assert len(errors) > 0
    assert any("Missing answers" in err for err in errors)


def test_validate_question_mapping_duplicate_answers(sample_exam_spec, sample_student):
    """Test validation catches duplicate answers."""
    student_answers = [
        StudentAnswer(
            student=sample_student,
            question_id="1a",
            raw_text="Answer 1a first"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="1a",
            raw_text="Answer 1a duplicate"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="1b",
            raw_text="Answer 1b"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="2",
            raw_text="Answer 2"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="3a",
            raw_text="Answer 3a"
        ),
    ]

    errors = validate_question_mapping(student_answers, sample_exam_spec)

    assert len(errors) > 0
    assert any("Duplicate" in err for err in errors)


def test_validate_question_mapping_empty_answers(sample_exam_spec, sample_student):
    """Test validation catches empty answers."""
    student_answers = [
        StudentAnswer(
            student=sample_student,
            question_id="1a",
            raw_text=""  # Empty
        ),
        StudentAnswer(
            student=sample_student,
            question_id="1b",
            raw_text="   "  # Whitespace only
        ),
        StudentAnswer(
            student=sample_student,
            question_id="2",
            raw_text="Answer 2"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="3a",
            raw_text="Answer 3a"
        ),
    ]

    errors = validate_question_mapping(student_answers, sample_exam_spec)

    assert len(errors) > 0
    assert any("Empty transcriptions" in err for err in errors)


def test_validate_question_mapping_unexpected_questions(sample_exam_spec, sample_student):
    """Test validation catches unexpected question IDs."""
    student_answers = [
        StudentAnswer(
            student=sample_student,
            question_id="1a",
            raw_text="Answer 1a"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="1b",
            raw_text="Answer 1b"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="2",
            raw_text="Answer 2"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="3a",
            raw_text="Answer 3a"
        ),
        StudentAnswer(
            student=sample_student,
            question_id="99_unexpected",
            raw_text="Unexpected answer"
        ),
    ]

    errors = validate_question_mapping(student_answers, sample_exam_spec)

    assert len(errors) > 0
    assert any("Unexpected" in err for err in errors)


# ============================================================================
# TEST: create_question_mapped_answers
# ============================================================================

def test_create_question_mapped_answers_basic(sample_exam_spec, sample_student):
    """Test creating question-mapped answers from page transcriptions."""
    page_transcriptions = [
        {
            'page_number': 1,
            'transcribed_text': "1a. Answer to first question.\n1b. Answer to second question.",
            'confidence': 0.95,
            'image_path': '/path/to/page1.jpg'
        },
        {
            'page_number': 2,
            'transcribed_text': "Full answer to question 2.",
            'confidence': 0.92,
            'image_path': '/path/to/page2.jpg'
        },
        {
            'page_number': 3,
            'transcribed_text': "Answer to question 3a.",
            'confidence': 0.88,
            'image_path': '/path/to/page3.jpg'
        },
    ]

    question_answers = create_question_mapped_answers(
        page_transcriptions, sample_exam_spec, sample_student
    )

    assert len(question_answers) == 4
    assert all(isinstance(ans, StudentAnswer) for ans in question_answers)
    assert all(ans.student == sample_student for ans in question_answers)

    # Check question IDs
    question_ids = [ans.question_id for ans in question_answers]
    assert "1a" in question_ids
    assert "1b" in question_ids
    assert "2" in question_ids
    assert "3a" in question_ids


def test_create_question_mapped_answers_missing_page(sample_exam_spec, sample_student):
    """Test handling when a page is missing."""
    page_transcriptions = [
        {
            'page_number': 1,
            'transcribed_text': "Answers for page 1",
            'confidence': 0.95,
            'image_path': '/path/to/page1.jpg'
        },
        # Page 2 missing
        {
            'page_number': 3,
            'transcribed_text': "Answers for page 3",
            'confidence': 0.88,
            'image_path': '/path/to/page3.jpg'
        },
    ]

    question_answers = create_question_mapped_answers(
        page_transcriptions, sample_exam_spec, sample_student
    )

    # Should have answers for pages 1 and 3, but not page 2
    question_ids = [ans.question_id for ans in question_answers]
    assert "1a" in question_ids
    assert "1b" in question_ids
    assert "2" not in question_ids  # Page 2 was missing
    assert "3a" in question_ids


# ============================================================================
# TEST: get_question_mapping_stats
# ============================================================================

def test_get_question_mapping_stats_complete(sample_exam_spec, sample_student):
    """Test statistics for complete mapping."""
    student_answers = [
        StudentAnswer(
            student=sample_student,
            question_id="1a",
            raw_text="Answer 1a",
            confidence=0.95
        ),
        StudentAnswer(
            student=sample_student,
            question_id="1b",
            raw_text="Answer 1b",
            confidence=0.90
        ),
        StudentAnswer(
            student=sample_student,
            question_id="2",
            raw_text="Answer 2",
            confidence=0.85
        ),
        StudentAnswer(
            student=sample_student,
            question_id="3a",
            raw_text="Answer 3a",
            confidence=0.92
        ),
    ]

    stats = get_question_mapping_stats(student_answers, sample_exam_spec)

    assert stats['expected_questions'] == 4
    assert stats['mapped_questions'] == 4
    assert stats['mapping_coverage'] == 100.0
    assert not stats['has_errors']
    assert stats['average_confidence'] > 0.8


def test_get_question_mapping_stats_incomplete(sample_exam_spec, sample_student):
    """Test statistics for incomplete mapping."""
    student_answers = [
        StudentAnswer(
            student=sample_student,
            question_id="1a",
            raw_text="Answer 1a",
            confidence=0.95
        ),
        StudentAnswer(
            student=sample_student,
            question_id="2",
            raw_text="Answer 2",
            confidence=0.85
        ),
        # Missing 1b and 3a
    ]

    stats = get_question_mapping_stats(student_answers, sample_exam_spec)

    assert stats['expected_questions'] == 4
    assert stats['mapped_questions'] == 2
    assert stats['mapping_coverage'] == 50.0
    assert stats['has_errors']  # Should have validation errors


# ============================================================================
# TEST: load_exam_spec (integration test)
# ============================================================================

def test_load_exam_spec_not_found(tmp_path):
    """Test loading ExamSpec when file doesn't exist."""
    result = load_exam_spec(tmp_path)
    assert result is None


def test_load_exam_spec_success(tmp_path):
    """Test successfully loading an ExamSpec."""
    # Create a results directory with exam_spec.json
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    exam_spec = ExamSpec(
        exam_name="Test Exam",
        total_pages=2,
        questions=[
            Question(
                question_id="1",
                question_text="Test",
                points=10.0,
                page_number=1
            )
        ]
    )

    exam_spec.save_to_file(results_dir / "exam_spec.json")

    loaded_spec = load_exam_spec(tmp_path)

    assert loaded_spec is not None
    assert loaded_spec.exam_name == "Test Exam"
    assert len(loaded_spec.questions) == 1


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

def test_map_page_to_questions_sorts_consistently(sample_exam_spec):
    """Test that questions are always returned in consistent order."""
    questions1 = map_page_to_questions(1, sample_exam_spec)
    questions2 = map_page_to_questions(1, sample_exam_spec)

    ids1 = [q.question_id for q in questions1]
    ids2 = [q.question_id for q in questions2]

    assert ids1 == ids2


def test_validate_question_mapping_with_page_based_ids(sample_exam_spec, sample_student):
    """Test that page-based IDs (like 'page_1') don't trigger unexpected warnings."""
    student_answers = [
        StudentAnswer(student=sample_student, question_id="1a", raw_text="A1"),
        StudentAnswer(student=sample_student, question_id="1b", raw_text="A2"),
        StudentAnswer(student=sample_student, question_id="2", raw_text="A3"),
        StudentAnswer(student=sample_student, question_id="3a", raw_text="A4"),
        StudentAnswer(student=sample_student, question_id="page_1", raw_text="Page 1 full"),
    ]

    errors = validate_question_mapping(student_answers, sample_exam_spec)

    # Should not complain about page-based IDs
    assert not any("page_" in err for err in errors)
