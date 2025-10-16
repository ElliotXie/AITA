"""
Tests for the Rubric Generation Pipeline
"""

import logging
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from dataclasses import dataclass

from aita.pipelines.generate_rubric import (
    RubricGenerationPipeline,
    RubricGenerationError,
    create_rubric_generation_pipeline,
    generate_rubrics_for_assignment,
    format_grading_notes
)
from aita.domain.models import (
    ExamSpec, Question, AnswerKey, Rubric, RubricCriterion, QuestionType
)


@pytest.fixture
def sample_exam_spec():
    """Create a sample exam specification for testing."""
    questions = [
        Question(
            question_id="1a",
            question_text="Calculate the derivative of f(x) = x² + 3x",
            points=10.0,
            question_type=QuestionType.CALCULATION,
            page_number=1
        ),
        Question(
            question_id="1b",
            question_text="Explain the concept of limits in calculus",
            points=15.0,
            question_type=QuestionType.SHORT_ANSWER,
            page_number=1
        ),
        Question(
            question_id="2",
            question_text="Prove that the derivative of sin(x) is cos(x)",
            points=20.0,
            question_type=QuestionType.LONG_ANSWER,
            page_number=2
        )
    ]

    return ExamSpec(
        exam_name="Test Calculus Exam",
        total_pages=2,
        questions=questions
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    return client


@pytest.fixture
def temp_directories(tmp_path):
    """Create temporary directories for testing."""
    data_dir = tmp_path / "data"
    intermediate_dir = tmp_path / "intermediate"

    # Create directory structure
    data_dir.mkdir()
    intermediate_dir.mkdir()
    (data_dir / "results").mkdir()
    (intermediate_dir / "rubrics").mkdir(parents=True)

    return data_dir, intermediate_dir


class TestRubricGenerationPipeline:
    """Test cases for RubricGenerationPipeline."""

    def test_init(self, mock_llm_client, temp_directories):
        """Test pipeline initialization."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        assert pipeline.llm_client == mock_llm_client
        assert pipeline.data_dir == data_dir
        assert pipeline.intermediate_dir == intermediate_dir
        assert pipeline.max_parse_retries == 2
        assert (intermediate_dir / "rubrics").exists()

    def test_check_existing_files_false(self, mock_llm_client, temp_directories):
        """Test that check_existing_files returns False when files don't exist."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        assert not pipeline._check_existing_files()

    def test_check_existing_files_true(self, mock_llm_client, temp_directories):
        """Test that check_existing_files returns True when files exist."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        # Create the expected files
        (intermediate_dir / "rubrics" / "generated_answer_keys.json").touch()
        (intermediate_dir / "rubrics" / "generated_rubrics.json").touch()

        assert pipeline._check_existing_files()

    def test_load_user_inputs_empty(self, mock_llm_client, temp_directories):
        """Test loading user inputs when no files exist."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        user_inputs = pipeline._load_user_inputs(None, None)

        assert user_inputs == {
            'rubrics': {},
            'instructions': "",
            'question_instructions': {}
        }

    def test_load_user_inputs_with_files(self, mock_llm_client, temp_directories):
        """Test loading user inputs with actual files."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        # Create test files
        rubrics_file = intermediate_dir / "test_rubrics.json"
        instructions_file = intermediate_dir / "test_instructions.txt"

        rubrics_data = {"1a": {"total_points": 10, "criteria": []}}
        with open(rubrics_file, 'w') as f:
            json.dump(rubrics_data, f)

        with open(instructions_file, 'w') as f:
            f.write("Test grading instructions")

        user_inputs = pipeline._load_user_inputs(rubrics_file, instructions_file)

        assert user_inputs['rubrics'] == rubrics_data
        assert user_inputs['instructions'] == "Test grading instructions"
        assert user_inputs['question_instructions'] == {}

    def test_parse_answer_key_response(self, mock_llm_client, temp_directories):
        """Test parsing answer key response from LLM."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        # Test with JSON response
        response_text = '''
        {
            "correct_answer": "f'(x) = 2x + 3",
            "solution_steps": ["Apply power rule", "Differentiate each term"],
            "alternative_answers": [],
            "explanation": "Using basic differentiation rules",
            "grading_notes": "Accept equivalent forms"
        }
        '''

        answer_key = pipeline._parse_answer_key_response(response_text, "1a")

        assert answer_key.question_id == "1a"
        assert answer_key.correct_answer == "f'(x) = 2x + 3"
        assert answer_key.solution_steps == ["Apply power rule", "Differentiate each term"]
        assert answer_key.explanation == "Using basic differentiation rules"

    def test_parse_answer_key_response_with_markdown(self, mock_llm_client, temp_directories):
        """Test parsing answer key response with markdown wrapping."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        # Test with markdown-wrapped JSON
        response_text = '''
        ```json
        {
            "correct_answer": "f'(x) = 2x + 3",
            "solution_steps": ["Step 1", "Step 2"],
            "alternative_answers": [],
            "explanation": "Explanation here",
            "grading_notes": "Notes here"
        }
        ```
        '''

        answer_key = pipeline._parse_answer_key_response(response_text, "1a")

        assert answer_key.question_id == "1a"
        assert answer_key.correct_answer == "f'(x) = 2x + 3"

    def test_parse_rubric_response(self, mock_llm_client, temp_directories, sample_exam_spec):
        """Test parsing rubric response from LLM."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        response_text = '''
        {
            "total_points": 10,
            "criteria": [
                {
                    "points": 5,
                    "description": "Correct formula application",
                    "examples": ["Correctly applies power rule"]
                },
                {
                    "points": 3,
                    "description": "Clear work shown",
                    "examples": ["Shows all steps"]
                },
                {
                    "points": 2,
                    "description": "Final answer correct",
                    "examples": ["f'(x) = 2x + 3"]
                }
            ]
        }
        '''

        question = sample_exam_spec.questions[0]  # 1a
        rubric = pipeline._parse_rubric_response(response_text, question)

        assert rubric.question_id == "1a"
        assert rubric.total_points == 10
        assert len(rubric.criteria) == 3
        assert rubric.criteria[0].points == 5
        assert rubric.criteria[0].description == "Correct formula application"

    def test_create_rubric_from_user_input(self, mock_llm_client, temp_directories, sample_exam_spec):
        """Test creating rubric from user-provided input."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        user_rubric = {
            "total_points": 10,
            "criteria": [
                {
                    "points": 6,
                    "description": "Correct calculation",
                    "examples": ["Shows proper derivative"]
                },
                {
                    "points": 4,
                    "description": "Clear explanation",
                    "examples": ["Explains each step"]
                }
            ]
        }

        question = sample_exam_spec.questions[0]  # 1a
        rubric = pipeline._create_rubric_from_user_input(question, user_rubric)

        assert rubric.question_id == "1a"
        assert rubric.total_points == 10
        assert len(rubric.criteria) == 2
        assert rubric.criteria[0].points == 6

    @patch('aita.pipelines.generate_rubric.format_grading_notes')
    @patch('aita.pipelines.generate_rubric.console')
    def test_generate_answer_keys_success(self, mock_console, mock_format_notes, mock_llm_client, temp_directories, sample_exam_spec):
        """Test successful answer key generation."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        mock_format_notes.side_effect = lambda *args, **kwargs: kwargs.get("grading_notes")

        # Mock LLM responses
        mock_llm_client.complete.side_effect = [
            '''{"correct_answer": "f'(x) = 2x + 3", "solution_steps": ["Apply power rule"], "alternative_answers": [], "explanation": "Basic differentiation", "grading_notes": "Accept equivalent forms"}''',
            '''{"correct_answer": "A limit is...", "solution_steps": ["Define limit"], "alternative_answers": [], "explanation": "Concept explanation", "grading_notes": "Look for understanding"}''',
            '''{"correct_answer": "Proof complete", "solution_steps": ["Start with definition"], "alternative_answers": [], "explanation": "Rigorous proof", "grading_notes": "Check logic"}'''
        ]

        user_inputs = {'instructions': '', 'question_instructions': {}}
        raw_answer_keys, answer_keys = pipeline._generate_answer_keys(sample_exam_spec, user_inputs)

        assert len(raw_answer_keys) == 3
        assert len(answer_keys) == 3
        assert raw_answer_keys[0].question_id == "1a"
        assert answer_keys[0].question_id == "1a"
        assert answer_keys[0].correct_answer == "f'(x) = 2x + 3"
        assert answer_keys[1].question_id == "1b"
        assert answer_keys[2].question_id == "2"
        assert raw_answer_keys[0] is not answer_keys[0]
        assert raw_answer_keys[0].grading_notes == answer_keys[0].grading_notes

    @patch('aita.pipelines.generate_rubric.format_grading_notes')
    @patch('aita.pipelines.generate_rubric.console')
    def test_generate_answer_keys_with_failure(self, mock_console, mock_format_notes, mock_llm_client, temp_directories, sample_exam_spec):
        """Test answer key generation with some failures."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        mock_format_notes.side_effect = lambda *args, **kwargs: kwargs.get("grading_notes")

        # Mock LLM responses with one failure
        mock_llm_client.complete.side_effect = [
            '''{"correct_answer": "f'(x) = 2x + 3", "solution_steps": ["Apply power rule"], "alternative_answers": [], "explanation": "Basic differentiation", "grading_notes": "Accept equivalent forms"}''',
            Exception("LLM API error"),  # Failure for second question
            '''{"correct_answer": "Proof complete", "solution_steps": ["Start with definition"], "alternative_answers": [], "explanation": "Rigorous proof", "grading_notes": "Check logic"}'''
        ]

        user_inputs = {'instructions': '', 'question_instructions': {}}
        raw_answer_keys, answer_keys = pipeline._generate_answer_keys(sample_exam_spec, user_inputs)

        assert len(raw_answer_keys) == 3
        assert len(answer_keys) == 3
        assert answer_keys[0].question_id == "1a"
        assert answer_keys[0].correct_answer == "f'(x) = 2x + 3"

        # Check that the failed question has a fallback answer key
        assert answer_keys[1].question_id == "1b"
        assert "[Answer key generation failed" in answer_keys[1].correct_answer

        assert answer_keys[2].question_id == "2"
        assert answer_keys[2].correct_answer == "Proof complete"

    def test_format_grading_notes_partial_sum(self, mock_llm_client, sample_exam_spec, caplog):
        """Ensure formatted grading notes include partial point summaries."""
        question = sample_exam_spec.questions[0]  # 10 points
        xml_response = """
<gradingNotes>
  <partialCreditRules>
    <item>
      <description>First step</description>
      <points>2</points>
    </item>
    <item>
      <description>Second step</description>
      <points>3</points>
    </item>
  </partialCreditRules>
  <deductions>
    <item>
      <description>Major error</description>
      <pointsLost>5</pointsLost>
    </item>
  </deductions>
</gradingNotes>
""".strip()
        mock_llm_client.complete_text.return_value = xml_response

        with caplog.at_level(logging.WARNING):
            result = format_grading_notes(
                llm_client=mock_llm_client,
                question=question,
                grading_notes="Original notes",
                max_retries=1
            )

        structured = json.loads(result)
        assert structured["partial_credit_points_sum"] == 5
        assert structured["question_total_points"] == question.points
        assert structured["partial_credit_points_match_total"] is False
        assert structured["partial_credit_rules"][0]["points"] == 2
        assert structured["deductions"][0]["points_lost"] == 5
        assert any("Partial credit points sum" in record.getMessage() for record in caplog.records)

    def test_extract_json_from_response(self, mock_llm_client, temp_directories):
        """Test JSON extraction from various response formats."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        # Test plain JSON
        json_text = '{"key": "value"}'
        result = pipeline._extract_json_from_response(json_text)
        assert result == '{"key": "value"}'

        # Test markdown-wrapped JSON
        markdown_text = '```json\n{"key": "value"}\n```'
        result = pipeline._extract_json_from_response(markdown_text)
        assert result == '{"key": "value"}'

        # Test with extra text
        mixed_text = 'Here is the JSON:\n{"key": "value"}\nThat was the JSON.'
        result = pipeline._extract_json_from_response(mixed_text)
        assert result == '{"key": "value"}'

    @patch('aita.pipelines.generate_rubric.console')
    def test_save_results(self, mock_console, mock_llm_client, temp_directories):
        """Test saving results to files."""
        data_dir, intermediate_dir = temp_directories

        # Mock the ExamReconstructor
        with patch('aita.pipelines.generate_rubric.ExamReconstructor') as mock_reconstructor_class:
            mock_reconstructor = Mock()
            mock_reconstructor_class.return_value = mock_reconstructor

            pipeline = RubricGenerationPipeline(
                llm_client=mock_llm_client,
                data_dir=data_dir,
                intermediate_dir=intermediate_dir
            )

            # Create test data
            raw_answer_keys = [
                AnswerKey(
                    question_id="1a",
                    correct_answer="f'(x) = 2x + 3",
                    solution_steps=["Apply power rule"],
                    grading_notes="Initial notes"
                )
            ]

            answer_keys = [
                AnswerKey(
                    question_id="1a",
                    correct_answer="f'(x) = 2x + 3",
                    solution_steps=["Apply power rule"],
                    grading_notes="Formatted notes"
                )
            ]

            rubrics = [
                Rubric(
                    question_id="1a",
                    total_points=10,
                    criteria=[
                        RubricCriterion(points=5, description="Correct formula")
                    ]
                )
            ]

            results_dir = data_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            mock_reconstructor.results_dir = results_dir

            pipeline._save_results(raw_answer_keys, answer_keys, rubrics)

            # Check that files were created in intermediate directory
            raw_answer_keys_file = intermediate_dir / "rubrics" / "generated_answer_keys_raw.json"
            answer_keys_file = intermediate_dir / "rubrics" / "generated_answer_keys.json"
            rubrics_file = intermediate_dir / "rubrics" / "generated_rubrics.json"

            assert raw_answer_keys_file.exists()
            assert answer_keys_file.exists()
            assert rubrics_file.exists()

            # Check raw answer keys in results directory
            raw_results_file = results_dir / "answer_key_raw.json"
            assert raw_results_file.exists()

            # Check that ExamReconstructor methods were called
            mock_reconstructor.save_answer_keys.assert_called_once_with(answer_keys)
            mock_reconstructor.save_rubrics.assert_called_once_with(rubrics)


class TestFactoryFunctions:
    """Test factory and convenience functions."""

    @patch('aita.pipelines.generate_rubric.get_config')
    @patch('aita.pipelines.generate_rubric.create_openrouter_client')
    def test_create_rubric_generation_pipeline(self, mock_create_client, mock_get_config):
        """Test pipeline factory function."""
        # Mock config
        mock_config = Mock()
        mock_config.data_dir = "/test/data"
        mock_config.intermediate_dir = "/test/intermediate"
        mock_get_config.return_value = mock_config

        # Mock LLM client
        mock_llm_client = Mock()
        mock_create_client.return_value = mock_llm_client

        pipeline = create_rubric_generation_pipeline("test_assignment")

        assert pipeline.llm_client == mock_llm_client
        assert str(pipeline.data_dir) == "/test/data"
        assert str(pipeline.intermediate_dir) == "/test/intermediate"

    @patch('aita.pipelines.generate_rubric.create_rubric_generation_pipeline')
    def test_generate_rubrics_for_assignment(self, mock_create_pipeline):
        """Test high-level rubric generation function."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline

        # Mock exam spec
        mock_exam_spec = Mock()
        mock_pipeline.reconstructor.load_exam_spec.return_value = mock_exam_spec

        # Mock return values
        mock_answer_keys = [Mock()]
        mock_rubrics = [Mock()]
        mock_pipeline.generate_from_exam_spec.return_value = (mock_answer_keys, mock_rubrics)

        result = generate_rubrics_for_assignment(
            assignment_name="test_exam",
            user_rubrics_file="rubrics.json",
            instructions_file="instructions.txt",
            force_regenerate=True
        )

        assert result == (mock_answer_keys, mock_rubrics)

        # Verify pipeline was called correctly
        mock_pipeline.generate_from_exam_spec.assert_called_once()
        call_args = mock_pipeline.generate_from_exam_spec.call_args
        assert call_args[1]['assignment_name'] == "test_exam"
        assert call_args[1]['force_regenerate'] == True

    @patch('aita.pipelines.generate_rubric.create_rubric_generation_pipeline')
    def test_generate_rubrics_for_assignment_no_exam_spec(self, mock_create_pipeline):
        """Test error when exam spec doesn't exist."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline

        # Mock missing exam spec
        mock_pipeline.reconstructor.load_exam_spec.return_value = None

        with pytest.raises(RubricGenerationError, match="No exam specification found"):
            generate_rubrics_for_assignment("test_exam")


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_rubric_generation_error(self):
        """Test custom exception creation."""
        error = RubricGenerationError("Test error message")
        assert str(error) == "Test error message"

    @patch('aita.pipelines.generate_rubric.console')
    def test_json_parse_error_retry(self, mock_console, mock_llm_client, temp_directories, sample_exam_spec):
        """Test retry logic for JSON parsing errors."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir,
            max_parse_retries=2
        )

        # First call returns invalid JSON, second call returns valid JSON
        mock_llm_client.complete.side_effect = [
            "Invalid JSON response",
            '''{"correct_answer": "f'(x) = 2x + 3", "solution_steps": ["Apply power rule"], "alternative_answers": [], "explanation": "Basic differentiation", "grading_notes": "Accept equivalent forms"}'''
        ]

        user_inputs = {'instructions': '', 'question_instructions': {}}
        question = sample_exam_spec.questions[0]

        # Should succeed on second try
        answer_key = pipeline._generate_single_answer_key(question, user_inputs)

        assert answer_key.question_id == "1a"
        assert answer_key.correct_answer == "f'(x) = 2x + 3"

        # Verify that LLM was called twice
        assert mock_llm_client.complete.call_count == 2

    @patch('aita.pipelines.generate_rubric.console')
    def test_json_parse_error_exhausted_retries(self, mock_console, mock_llm_client, temp_directories, sample_exam_spec):
        """Test behavior when all retry attempts fail."""
        data_dir, intermediate_dir = temp_directories

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir,
            max_parse_retries=2
        )

        # All calls return invalid JSON
        mock_llm_client.complete.return_value = "Invalid JSON response"

        user_inputs = {'instructions': '', 'question_instructions': {}}
        question = sample_exam_spec.questions[0]

        # Should raise RubricGenerationError after exhausting retries
        with pytest.raises(RubricGenerationError, match="Failed to parse answer key"):
            pipeline._generate_single_answer_key(question, user_inputs)

        # Verify that LLM was called the maximum number of times
        assert mock_llm_client.complete.call_count == 2
