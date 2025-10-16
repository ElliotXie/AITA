"""
Integration Tests for Parallel Execution in Pipelines

Tests the end-to-end integration of parallel execution with actual pipelines.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from aita.pipelines.generate_rubric import RubricGenerationPipeline
from aita.domain.models import ExamSpec, Question, QuestionType
from aita.services.llm.base import LLMResponse
from aita.utils.parallel_executor import ParallelLLMExecutor


class TestRubricPipelineIntegration:
    """Integration tests for rubric generation with parallel executor."""

    @pytest.fixture
    def sample_exam_spec(self):
        """Create sample exam specification."""
        questions = [
            Question(
                question_id="1a",
                question_text="Calculate f'(x) for f(x) = x² + 3x",
                points=10.0,
                question_type=QuestionType.CALCULATION,
                page_number=1
            ),
            Question(
                question_id="1b",
                question_text="Explain the concept of derivatives",
                points=15.0,
                question_type=QuestionType.SHORT_ANSWER,
                page_number=1
            ),
            Question(
                question_id="2",
                question_text="Prove the chain rule",
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
    def mock_llm_client(self):
        """Create mock LLM client with realistic responses."""
        client = Mock()

        # Answer key responses
        answer_key_response = '''
        {
            "correct_answer": "f'(x) = 2x + 3",
            "solution_steps": ["Apply power rule to x²", "Apply power rule to 3x", "Combine terms"],
            "alternative_answers": ["dy/dx = 2x + 3"],
            "explanation": "Using basic differentiation rules for polynomials",
            "grading_notes": "Accept any equivalent notation"
        }
        '''

        # Rubric responses
        rubric_response = '''
        {
            "total_points": 10,
            "criteria": [
                {
                    "points": 5,
                    "description": "Correct application of differentiation rules",
                    "examples": ["Uses power rule correctly"]
                },
                {
                    "points": 3,
                    "description": "Shows clear work",
                    "examples": ["Lists all steps"]
                },
                {
                    "points": 2,
                    "description": "Final answer is correct",
                    "examples": ["f'(x) = 2x + 3"]
                }
            ]
        }
        '''

        client.complete = Mock(return_value=LLMResponse(
            content=answer_key_response,
            usage={"input_tokens": 50, "output_tokens": 100},
            model="test-model"
        ))

        return client

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        data_dir = tmp_path / "data"
        intermediate_dir = tmp_path / "intermediate"

        (data_dir / "results").mkdir(parents=True)
        (intermediate_dir / "rubrics").mkdir(parents=True)

        return data_dir, intermediate_dir

    def test_parallel_rubric_generation_end_to_end(
        self,
        mock_llm_client,
        sample_exam_spec,
        temp_dirs
    ):
        """Test complete rubric generation pipeline with parallel execution."""
        data_dir, intermediate_dir = temp_dirs

        # Create pipeline with parallel executor
        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            max_workers=3,
            enable_checkpointing=False,
            show_progress=False
        )

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir,
            parallel_executor=executor
        )

        # Generate rubrics
        answer_keys, rubrics = pipeline.generate_from_exam_spec(
            exam_spec=sample_exam_spec,
            assignment_name="test_exam",
            force_regenerate=True
        )

        # Verify results
        assert len(answer_keys) == 3
        assert len(rubrics) == 3

        # Verify answer keys
        assert all(ak.question_id in ["1a", "1b", "2"] for ak in answer_keys)
        assert all(ak.correct_answer for ak in answer_keys)

        # Verify rubrics
        assert all(r.question_id in ["1a", "1b", "2"] for r in rubrics)
        assert all(len(r.criteria) > 0 for r in rubrics)

        # Verify files were saved
        answer_key_file = intermediate_dir / "rubrics" / "generated_answer_keys.json"
        rubric_file = intermediate_dir / "rubrics" / "generated_rubrics.json"
        assert answer_key_file.exists()
        assert rubric_file.exists()

    def test_parallel_execution_with_user_rubrics(
        self,
        mock_llm_client,
        sample_exam_spec,
        temp_dirs
    ):
        """Test that user-provided rubrics are used and don't trigger LLM calls."""
        data_dir, intermediate_dir = temp_dirs

        # Create user rubrics file
        user_rubrics_file = intermediate_dir / "test_rubrics.json"
        user_rubrics = {
            "1a": {
                "total_points": 10,
                "criteria": [
                    {
                        "points": 10,
                        "description": "User-provided criterion",
                        "examples": ["User example"]
                    }
                ]
            }
        }

        import json
        with open(user_rubrics_file, 'w') as f:
            json.dump(user_rubrics, f)

        # Create pipeline
        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir
        )

        # Generate rubrics
        answer_keys, rubrics = pipeline.generate_from_exam_spec(
            exam_spec=sample_exam_spec,
            assignment_name="test_exam",
            user_rubrics_file=user_rubrics_file,
            force_regenerate=True
        )

        # Verify that user rubric was used
        rubric_1a = next(r for r in rubrics if r.question_id == "1a")
        assert len(rubric_1a.criteria) == 1
        assert rubric_1a.criteria[0].description == "User-provided criterion"

        # Verify other rubrics were generated
        assert len(rubrics) == 3

    def test_parallel_execution_performance_improvement(
        self,
        mock_llm_client,
        sample_exam_spec,
        temp_dirs
    ):
        """Test that parallel execution is faster than sequential (mock timing)."""
        import time

        data_dir, intermediate_dir = temp_dirs

        # Mock LLM client to have realistic delay
        def slow_complete(*args, **kwargs):
            time.sleep(0.1)  # Simulate 100ms API call
            return LLMResponse(
                content='{"correct_answer": "test", "solution_steps": [], "alternative_answers": [], "explanation": "test", "grading_notes": "test"}',
                usage={"input_tokens": 10, "output_tokens": 20}
            )

        mock_llm_client.complete = Mock(side_effect=slow_complete)

        # Parallel execution with 3 workers
        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            max_workers=3,
            enable_checkpointing=False,
            show_progress=False,
            rate_limit=None  # No rate limit for this test
        )

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir,
            parallel_executor=executor
        )

        start = time.time()
        answer_keys, rubrics = pipeline.generate_from_exam_spec(
            exam_spec=sample_exam_spec,
            assignment_name="test",
            force_regenerate=True
        )
        parallel_time = time.time() - start

        # With 3 questions and 3 workers, parallel should complete in ~0.2s
        # (2 batches: answer keys and rubrics, each taking ~100ms with 3 workers)
        # Sequential would take ~0.6s (6 calls * 100ms)
        assert parallel_time < 0.4  # Should be significantly faster than sequential

    def test_error_handling_in_parallel_execution(
        self,
        mock_llm_client,
        sample_exam_spec,
        temp_dirs
    ):
        """Test that errors in parallel execution are handled gracefully."""
        data_dir, intermediate_dir = temp_dirs

        # Make some LLM calls fail
        call_count = [0]

        def failing_complete(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail the second call
                raise Exception("Simulated API error")
            return LLMResponse(
                content='{"correct_answer": "test", "solution_steps": [], "alternative_answers": [], "explanation": "test", "grading_notes": "test"}',
                usage={"input_tokens": 10, "output_tokens": 20}
            )

        mock_llm_client.complete = Mock(side_effect=failing_complete)

        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            max_workers=2,
            max_retries=0,  # Don't retry for faster test
            enable_checkpointing=False,
            show_progress=False
        )

        pipeline = RubricGenerationPipeline(
            llm_client=mock_llm_client,
            data_dir=data_dir,
            intermediate_dir=intermediate_dir,
            parallel_executor=executor
        )

        # Should not raise exception - errors should be handled
        answer_keys, rubrics = pipeline.generate_from_exam_spec(
            exam_spec=sample_exam_spec,
            assignment_name="test",
            force_regenerate=True
        )

        # Should still get results (some will be fallback)
        assert len(answer_keys) == 3
        assert len(rubrics) == 3

        # At least one should have failed
        failed_answers = [ak for ak in answer_keys if "failed" in ak.correct_answer.lower()]
        assert len(failed_answers) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
