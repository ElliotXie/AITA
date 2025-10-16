"""
Unit tests for question extraction pipeline.

Tests parser, validator, and error handling logic.
"""

import pytest
import json
from pathlib import Path

from aita.domain.models import Question, ExamSpec, QuestionType
from aita.pipelines.extract_questions import (
    parse_question_extraction_response,
    validate_exam_spec,
    _extract_json_from_response,
    _map_question_type,
    QuestionExtractionError
)


class TestJSONExtraction:
    """Test JSON extraction from various response formats."""

    def test_extract_json_with_markdown_wrapping(self):
        """Test extracting JSON from markdown code block."""
        response = """
        Here's the exam structure:
        ```json
        {"exam_name": "Test", "total_pages": 2, "questions": []}
        ```
        """
        result = _extract_json_from_response(response)
        assert '{"exam_name"' in result
        data = json.loads(result)
        assert data['exam_name'] == 'Test'

    def test_extract_json_without_markdown(self):
        """Test extracting raw JSON."""
        response = '{"exam_name": "Test", "total_pages": 2, "questions": []}'
        result = _extract_json_from_response(response)
        data = json.loads(result)
        assert data['exam_name'] == 'Test'

    def test_extract_json_with_surrounding_text(self):
        """Test extracting JSON with extra text around it."""
        response = """
        Let me analyze the exam:
        {"exam_name": "Test", "total_pages": 2, "questions": []}
        This is the structure I found.
        """
        result = _extract_json_from_response(response)
        data = json.loads(result)
        assert data['exam_name'] == 'Test'

    def test_extract_json_with_json_keyword(self):
        """Test extracting JSON with 'json' keyword in code block."""
        response = "```json\n{\"exam_name\": \"Test\", \"questions\": []}\n```"
        result = _extract_json_from_response(response)
        data = json.loads(result)
        assert data['exam_name'] == 'Test'


class TestQuestionTypeMapping:
    """Test question type string to enum mapping."""

    def test_map_multiple_choice(self):
        """Test mapping multiple choice variations."""
        assert _map_question_type('multiple_choice') == QuestionType.MULTIPLE_CHOICE
        assert _map_question_type('MULTIPLE_CHOICE') == QuestionType.MULTIPLE_CHOICE
        assert _map_question_type('multiple-choice') == QuestionType.MULTIPLE_CHOICE

    def test_map_short_answer(self):
        """Test mapping short answer."""
        assert _map_question_type('short_answer') == QuestionType.SHORT_ANSWER
        assert _map_question_type('short-answer') == QuestionType.SHORT_ANSWER

    def test_map_calculation(self):
        """Test mapping calculation type."""
        assert _map_question_type('calculation') == QuestionType.CALCULATION
        assert _map_question_type('CALCULATION') == QuestionType.CALCULATION

    def test_map_unknown_defaults_to_short_answer(self):
        """Test that unknown types default to short answer."""
        assert _map_question_type('unknown') == QuestionType.SHORT_ANSWER
        assert _map_question_type('essay') == QuestionType.SHORT_ANSWER


class TestResponseParsing:
    """Test parsing LLM responses into ExamSpec."""

    def test_parse_valid_response(self):
        """Test parsing a well-formed response."""
        response = json.dumps({
            "exam_name": "BMI541 Midterm",
            "total_pages": 3,
            "questions": [
                {
                    "question_id": "1a",
                    "question_text": "Calculate the probability",
                    "points": 5,
                    "page_number": 1,
                    "question_type": "calculation"
                },
                {
                    "question_id": "1b",
                    "question_text": "Explain your answer",
                    "points": 3,
                    "page_number": 1,
                    "question_type": "short_answer"
                }
            ]
        })

        exam_spec = parse_question_extraction_response(response, "Default", 3)

        assert exam_spec.exam_name == "BMI541 Midterm"
        assert exam_spec.total_pages == 3
        assert len(exam_spec.questions) == 2
        assert exam_spec.questions[0].question_id == "1a"
        assert exam_spec.questions[0].points == 5
        assert exam_spec.questions[0].question_type == QuestionType.CALCULATION

    def test_parse_response_with_missing_optional_fields(self):
        """Test parsing response with missing optional fields."""
        response = json.dumps({
            "questions": [
                {
                    "question_id": "1",
                    "question_text": "Question text",
                    "points": 10
                }
            ]
        })

        exam_spec = parse_question_extraction_response(response, "Default Exam", 1)

        assert exam_spec.exam_name == "Default Exam"
        assert exam_spec.total_pages == 1
        assert len(exam_spec.questions) == 1
        assert exam_spec.questions[0].page_number is None

    def test_parse_response_with_duplicate_question_ids(self):
        """Test that duplicate question IDs are handled."""
        response = json.dumps({
            "exam_name": "Test",
            "total_pages": 2,
            "questions": [
                {"question_id": "1", "question_text": "Q1", "points": 5},
                {"question_id": "1", "question_text": "Q2", "points": 5},
                {"question_id": "1", "question_text": "Q3", "points": 5}
            ]
        })

        exam_spec = parse_question_extraction_response(response, "Test", 2)

        question_ids = [q.question_id for q in exam_spec.questions]
        assert len(question_ids) == 3
        assert len(set(question_ids)) == 3  # All unique
        assert "1" in question_ids
        assert "1_v2" in question_ids
        assert "1_v3" in question_ids

    def test_parse_response_with_no_questions_raises_error(self):
        """Test that response with no questions raises error."""
        response = json.dumps({
            "exam_name": "Test",
            "total_pages": 1,
            "questions": []
        })

        with pytest.raises(ValueError, match="No questions found"):
            parse_question_extraction_response(response, "Test", 1)

    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        response = "This is not JSON at all!"

        with pytest.raises(json.JSONDecodeError):
            parse_question_extraction_response(response, "Test", 1)

    def test_parse_response_with_markdown_wrapping(self):
        """Test parsing JSON wrapped in markdown."""
        response = """
        ```json
        {
            "exam_name": "Test",
            "total_pages": 1,
            "questions": [
                {"question_id": "1", "question_text": "Q", "points": 10}
            ]
        }
        ```
        """

        exam_spec = parse_question_extraction_response(response, "Default", 1)

        assert exam_spec.exam_name == "Test"
        assert len(exam_spec.questions) == 1


class TestExamSpecValidation:
    """Test exam specification validation."""

    def test_validate_valid_exam_spec(self):
        """Test that valid exam spec has no errors."""
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=2,
            questions=[
                Question("1", "Question 1", 10.0, page_number=1),
                Question("2", "Question 2", 15.0, page_number=2)
            ]
        )

        errors = validate_exam_spec(exam_spec)
        assert errors == []

    def test_validate_empty_questions_list(self):
        """Test that empty questions list is caught."""
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=1,
            questions=[]
        )

        errors = validate_exam_spec(exam_spec)
        assert len(errors) == 1
        assert "No questions found" in errors[0]

    def test_validate_duplicate_question_ids(self):
        """Test that duplicate question IDs are caught."""
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=1,
            questions=[
                Question("1", "Question 1", 10.0),
                Question("1", "Question 1 duplicate", 5.0),
                Question("2", "Question 2", 8.0)
            ]
        )

        errors = validate_exam_spec(exam_spec)
        assert any("Duplicate question IDs" in e for e in errors)
        assert any("1" in e for e in errors)

    def test_validate_negative_points(self):
        """Test that negative points are caught."""
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=1,
            questions=[
                Question("1", "Question 1", -5.0)
            ]
        )

        errors = validate_exam_spec(exam_spec)
        assert any("invalid points" in e.lower() for e in errors)

    def test_validate_zero_points(self):
        """Test that zero points are caught."""
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=1,
            questions=[
                Question("1", "Question 1", 0.0)
            ]
        )

        errors = validate_exam_spec(exam_spec)
        assert any("invalid points" in e.lower() for e in errors)

    def test_validate_empty_question_text(self):
        """Test that empty question text is caught."""
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=1,
            questions=[
                Question("1", "", 10.0),
                Question("2", "   ", 10.0)  # Only whitespace
            ]
        )

        errors = validate_exam_spec(exam_spec)
        assert len([e for e in errors if "empty question text" in e.lower()]) == 2

    def test_validate_page_number_out_of_range(self):
        """Test that invalid page numbers are caught."""
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=3,
            questions=[
                Question("1", "Q1", 10.0, page_number=0),  # Too low
                Question("2", "Q2", 10.0, page_number=5),  # Too high
                Question("3", "Q3", 10.0, page_number=2)   # Valid
            ]
        )

        errors = validate_exam_spec(exam_spec)
        page_errors = [e for e in errors if "out of range" in e.lower()]
        assert len(page_errors) == 2

    def test_validate_multiple_errors_reported(self):
        """Test that multiple errors are all reported."""
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=2,
            questions=[
                Question("1", "", -5.0, page_number=10),  # Empty text, negative points, bad page
                Question("1", "Q2", 0.0)  # Duplicate ID, zero points
            ]
        )

        errors = validate_exam_spec(exam_spec)
        # Should have multiple errors
        assert len(errors) >= 4


class TestExamSpecCreation:
    """Test ExamSpec creation and properties."""

    def test_exam_spec_calculates_total_points(self):
        """Test that ExamSpec automatically calculates total points."""
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=1,
            questions=[
                Question("1", "Q1", 10.0),
                Question("2", "Q2", 15.0),
                Question("3", "Q3", 25.0)
            ]
        )

        assert exam_spec.total_points == 50.0

    def test_exam_spec_get_question_by_id(self):
        """Test retrieving questions by ID."""
        q1 = Question("1a", "Question 1a", 5.0)
        q2 = Question("1b", "Question 1b", 5.0)

        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=1,
            questions=[q1, q2]
        )

        retrieved = exam_spec.get_question("1a")
        assert retrieved is not None
        assert retrieved.question_id == "1a"
        assert retrieved.points == 5.0

        not_found = exam_spec.get_question("2")
        assert not_found is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_response_with_special_characters(self):
        """Test parsing questions with special characters."""
        response = json.dumps({
            "exam_name": "Test Exam #1 (Fall 2024)",
            "total_pages": 1,
            "questions": [
                {
                    "question_id": "1",
                    "question_text": "Calculate: ∫x²dx from 0 to π",
                    "points": 10
                }
            ]
        })

        exam_spec = parse_question_extraction_response(response, "Test", 1)
        assert "∫" in exam_spec.questions[0].question_text
        assert "π" in exam_spec.questions[0].question_text

    def test_parse_response_with_fractional_points(self):
        """Test parsing questions with fractional point values."""
        response = json.dumps({
            "exam_name": "Test",
            "total_pages": 1,
            "questions": [
                {"question_id": "1", "question_text": "Q1", "points": 2.5},
                {"question_id": "2", "question_text": "Q2", "points": 7.5}
            ]
        })

        exam_spec = parse_question_extraction_response(response, "Test", 1)
        assert exam_spec.questions[0].points == 2.5
        assert exam_spec.questions[1].points == 7.5
        assert exam_spec.total_points == 10.0

    def test_parse_response_continues_on_malformed_question(self):
        """Test that parser continues if one question is malformed."""
        # Note: Current implementation may skip malformed questions
        # This test documents that behavior
        response = json.dumps({
            "exam_name": "Test",
            "total_pages": 1,
            "questions": [
                {"question_id": "1", "question_text": "Q1", "points": 10},
                {"question_id": "2", "question_text": "Q2"},  # Missing points
                {"question_id": "3", "question_text": "Q3", "points": 15}
            ]
        })

        # Should parse what it can
        exam_spec = parse_question_extraction_response(response, "Test", 1)
        # At least the valid questions should be there
        assert len(exam_spec.questions) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
