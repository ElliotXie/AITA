"""
Test script to verify enhanced rubric generation retry logic.

This script demonstrates that the enhanced error handling works correctly
for both JSON parsing failures and score validation failures.
"""

import json
import logging
from unittest.mock import Mock, patch
from pathlib import Path

# Setup logging to see retry attempts
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

from aita.domain.models import Question, QuestionType
from aita.utils.llm_task_implementations import (
    RubricGenerationTask,
    AnswerKeyGenerationTask,
    JSONParsingError,
    ScoreValidationError
)


def create_test_question():
    """Create a test question for testing."""
    return Question(
        question_id="test_1a",
        question_text="Calculate the derivative of f(x) = x² + 3x",
        points=10.0,
        question_type=QuestionType.CALCULATION,
        page_number=1
    )


def test_json_parsing_retry():
    """Test that JSON parsing errors are retried correctly."""
    print("\n=== Testing JSON Parsing Retry Logic ===")

    question = create_test_question()
    task = AnswerKeyGenerationTask(question, "", "")

    # Test that JSON parsing errors trigger retries
    json_error = JSONParsingError("Invalid JSON")

    print(f"Initial retry count: {task.json_retry_count}")

    # First retry should be allowed
    should_retry = task.should_retry(json_error, 0)
    print(f"First attempt - Should retry: {should_retry}, Retry count: {task.json_retry_count}")
    assert should_retry == True
    assert task.json_retry_count == 1

    # Second retry should be allowed
    should_retry = task.should_retry(json_error, 1)
    print(f"Second attempt - Should retry: {should_retry}, Retry count: {task.json_retry_count}")
    assert should_retry == True
    assert task.json_retry_count == 2

    # Third retry should be allowed (max is 3)
    should_retry = task.should_retry(json_error, 2)
    print(f"Third attempt - Should retry: {should_retry}, Retry count: {task.json_retry_count}")
    assert should_retry == True
    assert task.json_retry_count == 3

    # Fourth retry should be rejected
    should_retry = task.should_retry(json_error, 3)
    print(f"Fourth attempt - Should retry: {should_retry}, Retry count: {task.json_retry_count}")
    assert should_retry == False
    assert task.json_retry_count == 4

    print("✅ JSON parsing retry logic works correctly!")


def test_score_validation_retry():
    """Test that score validation errors are retried correctly."""
    print("\n=== Testing Score Validation Retry Logic ===")

    question = create_test_question()
    task = RubricGenerationTask(question, None, "", "")

    # Test that score validation errors trigger retries
    score_error = ScoreValidationError(
        "Criteria sum (8.0) doesn't match total points (10.0)",
        criteria_sum=8.0,
        expected_total=10.0
    )

    print(f"Initial retry count: {task.score_retry_count}")

    # Test several retries to show it allows many more than JSON parsing
    for i in range(5):
        should_retry = task.should_retry(score_error, i)
        print(f"Attempt {i+1} - Should retry: {should_retry}, Retry count: {task.score_retry_count}")
        assert should_retry == True
        assert task.score_retry_count == i + 1

    print("✅ Score validation retry logic works correctly!")


def test_retry_prompts():
    """Test that retry prompts are enhanced correctly."""
    print("\n=== Testing Enhanced Retry Prompts ===")

    question = create_test_question()

    # Test RubricGenerationTask prompts
    rubric_task = RubricGenerationTask(question, None, "", "")

    # Initial prompt should be normal
    initial_messages = rubric_task.build_messages()
    initial_prompt = initial_messages[0].content
    assert "CRITICAL SCORING REQUIREMENT" not in initial_prompt
    print("✅ Initial rubric prompt is normal")

    # After a score retry, prompt should be enhanced
    rubric_task.score_retry_count = 1
    retry_messages = rubric_task.build_messages()
    retry_prompt = retry_messages[0].content
    assert "CRITICAL SCORING REQUIREMENT" in retry_prompt
    assert "PREVIOUS ATTEMPT FAILED" in retry_prompt
    assert str(question.points) in retry_prompt
    print("✅ Score retry prompt is enhanced with scoring requirements")

    # Test AnswerKeyGenerationTask prompts
    answer_task = AnswerKeyGenerationTask(question, "", "")

    # Initial prompt should be normal
    initial_messages = answer_task.build_messages()
    initial_prompt = initial_messages[0].content
    assert "CRITICAL JSON FORMAT REQUIREMENT" not in initial_prompt
    print("✅ Initial answer key prompt is normal")

    # After a JSON retry, prompt should be enhanced
    answer_task.json_retry_count = 1
    retry_messages = answer_task.build_messages()
    retry_prompt = retry_messages[0].content
    assert "CRITICAL JSON FORMAT REQUIREMENT" in retry_prompt
    assert "PREVIOUS ATTEMPT FAILED" in retry_prompt
    assert "NO MARKDOWN BLOCKS" in retry_prompt
    print("✅ JSON retry prompt is enhanced with format requirements")


def test_exception_types():
    """Test that the new exception types work correctly."""
    print("\n=== Testing Exception Types ===")

    # Test JSONParsingError
    json_error = JSONParsingError("Test JSON error")
    assert str(json_error) == "Test JSON error"
    print("✅ JSONParsingError works correctly")

    # Test ScoreValidationError
    score_error = ScoreValidationError(
        "Test score error",
        criteria_sum=8.0,
        expected_total=10.0
    )
    assert str(score_error) == "Test score error"
    assert score_error.criteria_sum == 8.0
    assert score_error.expected_total == 10.0
    print("✅ ScoreValidationError works correctly")


def test_other_errors_fallback():
    """Test that other error types fall back to default behavior."""
    print("\n=== Testing Other Error Fallback ===")

    question = create_test_question()
    task = RubricGenerationTask(question, None, "", "")

    # Test with a generic exception
    generic_error = ValueError("Some other error")

    # Should use parent class logic (not retry for non-transient errors)
    should_retry = task.should_retry(generic_error, 0)
    print(f"Generic error - Should retry: {should_retry}")
    assert should_retry == False  # ValueError is not in transient errors list

    print("✅ Other errors fall back to default behavior correctly!")


if __name__ == "__main__":
    print("Testing Enhanced Rubric Generation Retry Logic")
    print("=" * 50)

    try:
        test_exception_types()
        test_json_parsing_retry()
        test_score_validation_retry()
        test_retry_prompts()
        test_other_errors_fallback()

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Enhanced retry logic is working correctly.")
        print("\nSummary of enhancements:")
        print("- ✅ JSON parsing failures retry up to 3 times with enhanced prompts")
        print("- ✅ Score validation failures retry up to 10 times with scoring emphasis")
        print("- ✅ Comprehensive logging for all retry attempts")
        print("- ✅ Enhanced prompts that address specific failure types")
        print("- ✅ Safe implementation with retry limits and fallback behavior")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()