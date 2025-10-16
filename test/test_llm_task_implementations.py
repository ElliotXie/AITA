"""
Tests for Concrete LLM Task Implementations
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock

from aita.utils.llm_task_implementations import (
    AnswerKeyGenerationTask,
    RubricGenerationTask,
    TranscriptionTask,
    create_answer_key_tasks,
    create_rubric_tasks,
    create_transcription_tasks
)
from aita.domain.models import Question, QuestionType, AnswerKey, Rubric, RubricCriterion


class TestAnswerKeyGenerationTask:
    """Tests for AnswerKeyGenerationTask."""

    @pytest.fixture
    def sample_question(self):
        """Create sample question."""
        return Question(
            question_id="1a",
            question_text="Calculate the derivative of f(x) = x² + 3x",
            points=10.0,
            question_type=QuestionType.CALCULATION,
            page_number=1
        )

    def test_task_creation(self, sample_question):
        """Test task creation."""
        task = AnswerKeyGenerationTask(
            question=sample_question,
            general_instructions="Be thorough",
            question_instructions="Show all work"
        )

        assert task.task_id == "answer_key_1a"
        assert task.question == sample_question

    def test_build_messages(self, sample_question):
        """Test message building."""
        task = AnswerKeyGenerationTask(question=sample_question)
        messages = task.build_messages()

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "1a" in messages[0].content
        assert "derivative" in messages[0].content.lower()

    def test_parse_response_success(self, sample_question):
        """Test parsing valid JSON response."""
        task = AnswerKeyGenerationTask(question=sample_question)

        response = json.dumps({
            "correct_answer": "f'(x) = 2x + 3",
            "solution_steps": ["Apply power rule", "Differentiate each term"],
            "alternative_answers": ["dy/dx = 2x + 3"],
            "explanation": "Using basic differentiation rules",
            "grading_notes": "Accept equivalent forms"
        })

        answer_key = task.parse_response(response)

        assert isinstance(answer_key, AnswerKey)
        assert answer_key.question_id == "1a"
        assert answer_key.correct_answer == "f'(x) = 2x + 3"
        assert len(answer_key.solution_steps) == 2

    def test_parse_response_with_markdown(self, sample_question):
        """Test parsing JSON wrapped in markdown."""
        task = AnswerKeyGenerationTask(question=sample_question)

        response = '''
        Here is the answer key:
        ```json
        {
            "correct_answer": "f'(x) = 2x + 3",
            "solution_steps": ["Step 1", "Step 2"],
            "alternative_answers": [],
            "explanation": "Test explanation",
            "grading_notes": "Test notes"
        }
        ```
        '''

        answer_key = task.parse_response(response)
        assert answer_key.correct_answer == "f'(x) = 2x + 3"

    def test_parse_response_invalid(self, sample_question):
        """Test parsing invalid response."""
        task = AnswerKeyGenerationTask(question=sample_question)

        with pytest.raises(ValueError):
            task.parse_response("This is not JSON")

    def test_get_checkpoint_data(self, sample_question):
        """Test checkpoint data generation."""
        task = AnswerKeyGenerationTask(question=sample_question)
        data = task.get_checkpoint_data()

        assert data["task_id"] == "answer_key_1a"
        assert data["task_type"] == "AnswerKeyGeneration"
        assert data["question_id"] == "1a"


class TestRubricGenerationTask:
    """Tests for RubricGenerationTask."""

    @pytest.fixture
    def sample_question(self):
        """Create sample question."""
        return Question(
            question_id="2a",
            question_text="Explain the concept of limits",
            points=15.0,
            question_type=QuestionType.SHORT_ANSWER,
            page_number=2
        )

    @pytest.fixture
    def sample_answer_key(self):
        """Create sample answer key."""
        return AnswerKey(
            question_id="2a",
            correct_answer="A limit describes the value a function approaches...",
            solution_steps=["Define limit", "Provide examples"],
            explanation="Core concept in calculus"
        )

    def test_task_creation(self, sample_question, sample_answer_key):
        """Test task creation."""
        task = RubricGenerationTask(
            question=sample_question,
            answer_key=sample_answer_key,
            general_instructions="Be fair",
            question_instructions="Focus on understanding"
        )

        assert task.task_id == "rubric_2a"
        assert task.question == sample_question
        assert task.answer_key == sample_answer_key

    def test_parse_response_success(self, sample_question):
        """Test parsing valid rubric response."""
        task = RubricGenerationTask(question=sample_question)

        response = json.dumps({
            "total_points": 15.0,
            "criteria": [
                {
                    "points": 8.0,
                    "description": "Correct definition of limit",
                    "examples": ["Defines limit properly"]
                },
                {
                    "points": 5.0,
                    "description": "Provides clear examples",
                    "examples": ["Uses concrete examples"]
                },
                {
                    "points": 2.0,
                    "description": "Notation is correct",
                    "examples": ["Uses proper notation"]
                }
            ]
        })

        rubric = task.parse_response(response)

        assert isinstance(rubric, Rubric)
        assert rubric.question_id == "2a"
        assert rubric.total_points == 15.0
        assert len(rubric.criteria) == 3

    def test_validate_rubric_points_mismatch(self, sample_question):
        """Test rubric validation catches point mismatches."""
        task = RubricGenerationTask(question=sample_question)

        # Total points don't match criteria sum
        response = json.dumps({
            "total_points": 15.0,
            "criteria": [
                {"points": 5.0, "description": "Test 1"},
                {"points": 5.0, "description": "Test 2"}
            ]
        })

        with pytest.raises(ValueError, match="doesn't match total points"):
            task.parse_response(response)

    def test_validate_rubric_question_mismatch(self):
        """Test rubric validation catches question point mismatch."""
        question = Question(
            question_id="test",
            question_text="Test",
            points=10.0,
            question_type=QuestionType.SHORT_ANSWER
        )

        task = RubricGenerationTask(question=question)

        # Rubric total doesn't match question points
        response = json.dumps({
            "total_points": 15.0,
            "criteria": [
                {"points": 15.0, "description": "Test"}
            ]
        })

        with pytest.raises(ValueError, match="doesn't match question points"):
            task.parse_response(response)


class TestTranscriptionTask:
    """Tests for TranscriptionTask."""

    def test_task_creation(self):
        """Test transcription task creation."""
        task = TranscriptionTask(
            image_url="https://storage.example.com/image.jpg",
            student_name="John Doe",
            page_number=3,
            image_path=Path("/path/to/image.jpg")
        )

        assert task.task_id == "transcription_John Doe_page_3"
        assert task.image_url == "https://storage.example.com/image.jpg"
        assert task.page_number == 3

    def test_get_image_urls(self):
        """Test getting image URLs."""
        task = TranscriptionTask(
            image_url="https://example.com/test.jpg",
            student_name="Test Student",
            page_number=1
        )

        urls = task.get_image_urls()
        assert urls == ["https://example.com/test.jpg"]

    def test_get_prompt_text(self):
        """Test prompt text generation."""
        task = TranscriptionTask(
            image_url="https://example.com/test.jpg",
            student_name="Test",
            page_number=2
        )

        prompt = task.get_prompt_text()
        assert "page 2" in prompt.lower()

    def test_parse_response_success(self):
        """Test parsing valid transcription response."""
        task = TranscriptionTask(
            image_url="https://example.com/test.jpg",
            student_name="Test Student",
            page_number=1,
            image_path=Path("/test/path.jpg")
        )

        response = json.dumps({
            "transcribed_text": "This is the transcribed text from the exam page.",
            "confidence": 0.95,
            "notes": "Handwriting is clear"
        })

        result = task.parse_response(response)

        assert result['transcribed_text'] == "This is the transcribed text from the exam page."
        assert result['confidence'] == 0.95
        assert result['page_number'] == 1

    def test_parse_response_fallback(self):
        """Test fallback when JSON parsing fails."""
        task = TranscriptionTask(
            image_url="https://example.com/test.jpg",
            student_name="Test",
            page_number=1
        )

        # Invalid JSON - should fall back to raw text
        response = "This is just plain text, not JSON"
        result = task.parse_response(response)

        assert result['transcribed_text'] == "This is just plain text, not JSON"
        assert result['confidence'] == 0.3  # Lower confidence for fallback
        assert "JSON parsing failed" in result['notes']

    def test_parse_response_invalid_confidence(self):
        """Test handling of invalid confidence values."""
        task = TranscriptionTask(
            image_url="https://example.com/test.jpg",
            student_name="Test",
            page_number=1
        )

        response = json.dumps({
            "transcribed_text": "Test text",
            "confidence": 1.5,  # Invalid: > 1.0
            "notes": ""
        })

        result = task.parse_response(response)
        assert result['confidence'] == 0.5  # Should be corrected


class TestTaskCreationHelpers:
    """Tests for task creation helper functions."""

    def test_create_answer_key_tasks(self):
        """Test creating multiple answer key tasks."""
        questions = [
            Question("1a", "Question 1a", 10.0, QuestionType.SHORT_ANSWER),
            Question("1b", "Question 1b", 15.0, QuestionType.LONG_ANSWER),
            Question("2", "Question 2", 20.0, QuestionType.CALCULATION)
        ]

        tasks = create_answer_key_tasks(
            questions=questions,
            general_instructions="Be thorough",
            question_instructions={"1a": "Show work"}
        )

        assert len(tasks) == 3
        assert all(isinstance(t, AnswerKeyGenerationTask) for t in tasks)
        assert tasks[0].task_id == "answer_key_1a"
        assert tasks[0].question_instructions == "Show work"
        assert tasks[1].question_instructions == ""

    def test_create_rubric_tasks(self):
        """Test creating multiple rubric tasks."""
        questions = [
            Question("1a", "Q1a", 10.0, QuestionType.SHORT_ANSWER),
            Question("1b", "Q1b", 15.0, QuestionType.LONG_ANSWER)
        ]

        answer_keys = {
            "1a": AnswerKey("1a", "Answer 1a"),
            "1b": AnswerKey("1b", "Answer 1b")
        }

        tasks = create_rubric_tasks(
            questions=questions,
            answer_keys=answer_keys,
            general_instructions="Be fair"
        )

        assert len(tasks) == 2
        assert all(isinstance(t, RubricGenerationTask) for t in tasks)
        assert tasks[0].answer_key.question_id == "1a"
        assert tasks[1].answer_key.question_id == "1b"

    def test_create_transcription_tasks(self):
        """Test creating transcription tasks."""
        image_data = [
            {
                "image_url": "https://example.com/img1.jpg",
                "page_number": 1,
                "image_path": Path("/path/img1.jpg")
            },
            {
                "image_url": "https://example.com/img2.jpg",
                "page_number": 2,
                "image_path": Path("/path/img2.jpg")
            }
        ]

        tasks = create_transcription_tasks(
            image_data=image_data,
            student_name="Jane Smith"
        )

        assert len(tasks) == 2
        assert all(isinstance(t, TranscriptionTask) for t in tasks)
        assert tasks[0].student_name == "Jane Smith"
        assert tasks[0].page_number == 1
        assert tasks[1].page_number == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
