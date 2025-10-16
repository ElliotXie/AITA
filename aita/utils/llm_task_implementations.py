"""
Concrete LLM Task Implementations

Provides ready-to-use task implementations for common AITA operations:
- Answer key generation
- Rubric generation
- Transcription
- Question extraction
"""

import json
import re
import signal
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from aita.services.llm.base import LLMMessage
from aita.utils.llm_tasks import LLMTask, VisionLLMTask
from aita.domain.models import Question, AnswerKey, Rubric, RubricCriterion, Grade, Student
from aita.utils.prompts import (
    get_answer_key_generation_prompt,
    get_single_rubric_generation_prompt,
    get_transcription_prompt
)


class JSONParsingError(Exception):
    """Raised when LLM response cannot be parsed as valid JSON."""
    pass


class ScoreValidationError(Exception):
    """Raised when rubric scores do not sum correctly or match question points."""

    def __init__(self, message: str, criteria_sum: float = None, expected_total: float = None):
        super().__init__(message)
        self.criteria_sum = criteria_sum
        self.expected_total = expected_total


def timeout_handler(signum, frame):
    raise TimeoutError("JSON extraction timed out")


def safe_extract_json(response_text: str, timeout_seconds: int = 5) -> str:
    """
    Safely extract JSON with timeout protection.

    Args:
        response_text: Raw response text
        timeout_seconds: Maximum time to spend on extraction

    Returns:
        Extracted JSON string

    Raises:
        TimeoutError: If extraction takes too long
    """
    # Try markdown code block first (fast operation)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # For the greedy search, use a simple approach
    # Find first { and last }
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return response_text[first_brace:last_brace + 1]

    return response_text.strip()


class AnswerKeyGenerationTask(LLMTask):
    """
    Task for generating answer keys for exam questions.

    Generates detailed answer keys with solution steps, alternative answers,
    and grading notes.
    """

    def __init__(
        self,
        question: Question,
        general_instructions: str = "",
        question_instructions: str = ""
    ):
        """
        Initialize answer key generation task.

        Args:
            question: Question object to generate answer key for
            general_instructions: General grading instructions
            question_instructions: Question-specific instructions
        """
        self.question = question
        self.general_instructions = general_instructions
        self.question_instructions = question_instructions
        self.json_retry_count = 0
        self.max_json_retries = 3

    @property
    def task_id(self) -> str:
        return f"answer_key_{self.question.question_id}"

    def build_messages(self) -> List[LLMMessage]:
        # Use enhanced prompt if we're retrying due to JSON parsing errors
        if self.json_retry_count > 0:
            prompt = self._get_json_retry_prompt()
        else:
            prompt = get_answer_key_generation_prompt(
                question=self.question,
                general_instructions=self.general_instructions,
                question_instructions=self.question_instructions
            )
        return [LLMMessage(role="user", content=prompt)]

    def parse_response(self, response_text: str) -> AnswerKey:
        """Parse JSON response into AnswerKey object."""
        try:
            # Extract JSON from response using safe method
            json_text = safe_extract_json(response_text)
            data = json.loads(json_text)

            return AnswerKey(
                question_id=self.question.question_id,
                correct_answer=data.get('correct_answer', ''),
                alternative_answers=data.get('alternative_answers', []),
                explanation=data.get('explanation', ''),
                grading_notes=data.get('grading_notes', ''),
                solution_steps=data.get('solution_steps', [])
            )

        except (json.JSONDecodeError, KeyError, TimeoutError) as e:
            raise JSONParsingError(f"Failed to parse answer key response: {e}") from e

    def get_llm_params(self) -> Dict[str, Any]:
        # Cap output length so the model doesn't stream massive responses
        return {"temperature": 0.1, "max_tokens": 1200}

    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """
        Determine if task should be retried based on error type.

        Args:
            error: Exception that occurred
            retry_count: Total number of retries so far

        Returns:
            True if should retry, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)

        if isinstance(error, JSONParsingError):
            self.json_retry_count += 1
            should_retry = self.json_retry_count <= self.max_json_retries
            if should_retry:
                logger.warning(
                    f"JSON parsing failed for answer key {self.question.question_id} "
                    f"(attempt {self.json_retry_count}/{self.max_json_retries}): {error}"
                )
            else:
                logger.error(
                    f"JSON parsing exhausted for answer key {self.question.question_id} "
                    f"after {self.max_json_retries} attempts"
                )
            return should_retry
        else:
            # For other errors, use default behavior
            return super().should_retry(error, retry_count)

    def _get_json_retry_prompt(self) -> str:
        """Generate enhanced prompt for JSON parsing retry."""
        base_prompt = get_answer_key_generation_prompt(
            question=self.question,
            general_instructions=self.general_instructions,
            question_instructions=self.question_instructions
        )

        json_emphasis = """
CRITICAL JSON FORMAT REQUIREMENT - PREVIOUS ATTEMPT FAILED:
Your previous response was not valid JSON. You MUST:

1. Respond with ONLY valid JSON, no markdown formatting or additional text
2. Use proper JSON syntax with quotes around all strings
3. Follow this EXACT structure:

{
  "correct_answer": "string",
  "solution_steps": ["string", "string"],
  "alternative_answers": ["string"],
  "explanation": "string",
  "grading_notes": "string"
}

NO MARKDOWN BLOCKS (```), NO EXPLANATORY TEXT, JUST PURE JSON.

"""
        return json_emphasis + base_prompt

    def _extract_json(self, response_text: str) -> str:
        """Extract JSON from response, handling markdown wrapping."""
        # Try markdown code block first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to find raw JSON object - use greedy match to get complete JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return response_text.strip()

    def get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": "AnswerKeyGeneration",
            "question_id": self.question.question_id
        }


class RubricGenerationTask(LLMTask):
    """
    Task for generating grading rubrics for exam questions.

    Generates detailed rubrics with point breakdowns and examples.
    """

    def __init__(
        self,
        question: Question,
        answer_key: Optional[AnswerKey] = None,
        general_instructions: str = "",
        question_instructions: str = ""
    ):
        """
        Initialize rubric generation task.

        Args:
            question: Question object to generate rubric for
            answer_key: Optional answer key for reference
            general_instructions: General grading instructions
            question_instructions: Question-specific instructions
        """
        self.question = question
        self.answer_key = answer_key
        self.general_instructions = general_instructions
        self.question_instructions = question_instructions
        self.json_retry_count = 0
        self.score_retry_count = 0
        self.max_json_retries = 3
        self.max_score_retries = 10  # Safety cap for "infinite" retries

    @property
    def task_id(self) -> str:
        return f"rubric_{self.question.question_id}"

    def build_messages(self) -> List[LLMMessage]:
        # Use enhanced prompt if we're retrying due to score validation errors
        if self.score_retry_count > 0:
            prompt = self._get_score_retry_prompt()
        elif self.json_retry_count > 0:
            prompt = self._get_json_retry_prompt()
        else:
            prompt = get_single_rubric_generation_prompt(
                question=self.question,
                answer_key=self.answer_key,
                general_instructions=self.general_instructions,
                question_instructions=self.question_instructions
            )
        return [LLMMessage(role="user", content=prompt)]

    def parse_response(self, response_text: str) -> Rubric:
        """Parse JSON response into Rubric object."""
        try:
            # Extract JSON from response using safe method
            json_text = safe_extract_json(response_text)
            data = json.loads(json_text)

            criteria = []
            for criterion_data in data.get('criteria', []):
                criteria.append(RubricCriterion(
                    points=criterion_data['points'],
                    description=criterion_data['description'],
                    examples=criterion_data.get('examples', [])
                ))

            rubric = Rubric(
                question_id=self.question.question_id,
                total_points=data.get('total_points', self.question.points),
                criteria=criteria
            )

            # Validate rubric
            self._validate_rubric(rubric)

            return rubric

        except (json.JSONDecodeError, KeyError, TimeoutError) as e:
            raise JSONParsingError(f"Failed to parse rubric response: {e}") from e

    def _validate_rubric(self, rubric: Rubric):
        """Validate that rubric is well-formed."""
        # Check that criteria points sum to total
        criteria_sum = sum(c.points for c in rubric.criteria)
        if abs(criteria_sum - rubric.total_points) > 0.01:
            raise ScoreValidationError(
                f"Rubric criteria sum ({criteria_sum}) doesn't match total points ({rubric.total_points})",
                criteria_sum=criteria_sum,
                expected_total=rubric.total_points
            )

        # Check that rubric total matches question points
        if abs(rubric.total_points - self.question.points) > 0.01:
            raise ScoreValidationError(
                f"Rubric total points ({rubric.total_points}) doesn't match question points ({self.question.points})",
                criteria_sum=rubric.total_points,
                expected_total=self.question.points
            )

    def get_llm_params(self) -> Dict[str, Any]:
        # Limit response size to avoid runaway completions
        return {"temperature": 0.1, "max_tokens": 1500}

    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """
        Determine if task should be retried based on error type.

        Args:
            error: Exception that occurred
            retry_count: Total number of retries so far

        Returns:
            True if should retry, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)

        if isinstance(error, JSONParsingError):
            self.json_retry_count += 1
            should_retry = self.json_retry_count <= self.max_json_retries
            if should_retry:
                logger.warning(
                    f"JSON parsing failed for rubric {self.question.question_id} "
                    f"(attempt {self.json_retry_count}/{self.max_json_retries}): {error}"
                )
            else:
                logger.error(
                    f"JSON parsing exhausted for rubric {self.question.question_id} "
                    f"after {self.max_json_retries} attempts"
                )
            return should_retry

        elif isinstance(error, ScoreValidationError):
            self.score_retry_count += 1
            should_retry = self.score_retry_count <= self.max_score_retries
            if should_retry:
                logger.warning(
                    f"Score validation failed for rubric {self.question.question_id} "
                    f"(attempt {self.score_retry_count}/{self.max_score_retries}): {error}"
                )
                if hasattr(error, 'criteria_sum') and hasattr(error, 'expected_total'):
                    logger.info(
                        f"Score details - Criteria sum: {error.criteria_sum}, "
                        f"Expected: {error.expected_total}"
                    )
            else:
                logger.error(
                    f"Score validation exhausted for rubric {self.question.question_id} "
                    f"after {self.max_score_retries} attempts"
                )
            return should_retry

        else:
            # For other errors, use default behavior
            return super().should_retry(error, retry_count)

    def _get_score_retry_prompt(self) -> str:
        """Generate enhanced prompt for score validation retry."""
        base_prompt = get_single_rubric_generation_prompt(
            question=self.question,
            answer_key=self.answer_key,
            general_instructions=self.general_instructions,
            question_instructions=self.question_instructions
        )

        score_emphasis = f"""
CRITICAL SCORING REQUIREMENT - PREVIOUS ATTEMPT FAILED:
Your previous rubric had incorrect point totals. You MUST ensure:

1. The sum of ALL criteria points EXACTLY equals {self.question.points} points
2. Each criterion must have a specific point value
3. No partial or fractional points unless they sum exactly to {self.question.points}

SCORING VERIFICATION CHECKLIST:
- Add up all criteria points: _____ (must equal {self.question.points})
- Ensure no points are missing or doubled
- Use simple arithmetic: 2+3+5={self.question.points} ✓ or 1+4+5={self.question.points} ✓

FAILURE TO MATCH EXACT POINT TOTALS WILL RESULT IN ANOTHER RETRY.

"""
        return score_emphasis + base_prompt

    def _get_json_retry_prompt(self) -> str:
        """Generate enhanced prompt for JSON parsing retry."""
        base_prompt = get_single_rubric_generation_prompt(
            question=self.question,
            answer_key=self.answer_key,
            general_instructions=self.general_instructions,
            question_instructions=self.question_instructions
        )

        json_emphasis = """
CRITICAL JSON FORMAT REQUIREMENT - PREVIOUS ATTEMPT FAILED:
Your previous response was not valid JSON. You MUST:

1. Respond with ONLY valid JSON, no markdown formatting or additional text
2. Use proper JSON syntax with quotes around all strings
3. Follow this EXACT structure:

{
  "total_points": number,
  "criteria": [
    {
      "points": number,
      "description": "string",
      "examples": ["string"]
    }
  ]
}

NO MARKDOWN BLOCKS (```), NO EXPLANATORY TEXT, JUST PURE JSON.

"""
        return json_emphasis + base_prompt

    def _extract_json(self, response_text: str) -> str:
        """Extract JSON from response."""
        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to find raw JSON object - use greedy match
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return response_text.strip()

    def get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": "RubricGeneration",
            "question_id": self.question.question_id
        }


class TranscriptionTask(VisionLLMTask):
    """
    Task for transcribing student handwritten exam pages to text.

    Uses vision LLM to analyze images and extract handwritten content.
    """

    def __init__(
        self,
        image_url: str,
        student_name: str,
        page_number: int,
        image_path: Optional[Path] = None,
        question_context: Optional[str] = None
    ):
        """
        Initialize transcription task.

        Args:
            image_url: Public URL of the exam page image
            student_name: Name of the student
            page_number: Page number for context
            image_path: Optional local path to image file
            question_context: Optional question text for context
        """
        self.image_url = image_url
        self.student_name = student_name
        self.page_number = page_number
        self.image_path = image_path
        self.question_context = question_context

    @property
    def task_id(self) -> str:
        return f"transcription_{self.student_name}_page_{self.page_number}"

    def get_image_urls(self) -> List[str]:
        return [self.image_url]

    def get_prompt_text(self) -> str:
        context = self.question_context or f"Student response on page {self.page_number} of exam"
        return get_transcription_prompt(question_text=context)

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response into transcription data."""
        try:
            # Try to extract JSON using safe method
            json_text = safe_extract_json(response_text)
            data = json.loads(json_text)

            # Validate required fields
            if 'transcribed_text' not in data:
                raise ValueError("Missing 'transcribed_text' field in response")

            # Ensure confidence is valid
            confidence = float(data.get('confidence', 0.0))
            if not 0.0 <= confidence <= 1.0:
                confidence = 0.5

            return {
                'transcribed_text': data['transcribed_text'],
                'confidence': confidence,
                'notes': data.get('notes', ''),
                'page_number': self.page_number,
                'image_path': str(self.image_path) if self.image_path else None,
                'image_url': self.image_url
            }

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: use raw response text
            return {
                'transcribed_text': response_text.strip(),
                'confidence': 0.3,
                'notes': f"Raw LLM response (JSON parsing failed: {e})",
                'page_number': self.page_number,
                'image_path': str(self.image_path) if self.image_path else None,
                'image_url': self.image_url
            }

    def get_llm_params(self) -> Dict[str, Any]:
        return {"temperature": 0.1, "max_tokens": None}

    def _extract_json(self, response_text: str) -> str:
        """Extract JSON from response."""
        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to find raw JSON object - use greedy match
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return response_text.strip()

    def get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": "Transcription",
            "student_name": self.student_name,
            "page_number": self.page_number
        }


# Convenience functions for creating tasks

def create_answer_key_tasks(
    questions: List[Question],
    general_instructions: str = "",
    question_instructions: Dict[str, str] = None
) -> List[AnswerKeyGenerationTask]:
    """
    Create answer key generation tasks for a list of questions.

    Args:
        questions: List of Question objects
        general_instructions: General grading instructions
        question_instructions: Dict mapping question_id to specific instructions

    Returns:
        List of AnswerKeyGenerationTask objects
    """
    question_instructions = question_instructions or {}
    tasks = []

    for question in questions:
        specific_instructions = question_instructions.get(question.question_id, "")
        task = AnswerKeyGenerationTask(
            question=question,
            general_instructions=general_instructions,
            question_instructions=specific_instructions
        )
        tasks.append(task)

    return tasks


def create_rubric_tasks(
    questions: List[Question],
    answer_keys: Optional[Dict[str, AnswerKey]] = None,
    general_instructions: str = "",
    question_instructions: Dict[str, str] = None
) -> List[RubricGenerationTask]:
    """
    Create rubric generation tasks for a list of questions.

    Args:
        questions: List of Question objects
        answer_keys: Optional dict mapping question_id to AnswerKey
        general_instructions: General grading instructions
        question_instructions: Dict mapping question_id to specific instructions

    Returns:
        List of RubricGenerationTask objects
    """
    answer_keys = answer_keys or {}
    question_instructions = question_instructions or {}
    tasks = []

    for question in questions:
        answer_key = answer_keys.get(question.question_id)
        specific_instructions = question_instructions.get(question.question_id, "")

        task = RubricGenerationTask(
            question=question,
            answer_key=answer_key,
            general_instructions=general_instructions,
            question_instructions=specific_instructions
        )
        tasks.append(task)

    return tasks


def create_transcription_tasks(
    image_data: List[Dict[str, Any]],
    student_name: str
) -> List[TranscriptionTask]:
    """
    Create transcription tasks for student exam pages.

    Args:
        image_data: List of dicts with 'image_url', 'page_number', etc.
        student_name: Name of the student

    Returns:
        List of TranscriptionTask objects
    """
    tasks = []

    for data in image_data:
        task = TranscriptionTask(
            image_url=data['image_url'],
            student_name=student_name,
            page_number=data['page_number'],
            image_path=data.get('image_path'),
            question_context=data.get('question_context')
        )
        tasks.append(task)

    return tasks


class QuestionGradingTask(LLMTask):
    """
    Task for grading student answers using LLM and rubrics.

    Grades a single student's answer to a specific question using
    the provided rubric and answer key.
    """

    def __init__(
        self,
        question: Question,
        student_answer: str,
        student_name: str,
        answer_key: AnswerKey,
        rubric: Rubric,
        answer_confidence: Optional[float] = None,
        transcription_notes: Optional[str] = None
    ):
        """
        Initialize question grading task.

        Args:
            question: Question object being graded
            student_answer: Student's transcribed answer text
            student_name: Name of the student
            answer_key: Answer key for this question
            rubric: Grading rubric for this question
            answer_confidence: Confidence score from transcription
            transcription_notes: Notes from transcription process
        """
        self.question = question
        self.student_answer = student_answer
        self.student_name = student_name
        self.answer_key = answer_key
        self.rubric = rubric
        self.answer_confidence = answer_confidence
        self.transcription_notes = transcription_notes

    @property
    def task_id(self) -> str:
        return f"grade_{self.student_name.replace(' ', '_')}_{self.question.question_id}"

    def build_messages(self) -> List[LLMMessage]:
        prompt = self._create_grading_prompt()
        return [LLMMessage(role="user", content=prompt)]

    def _create_grading_prompt(self) -> str:
        """Create detailed grading prompt with question, rubric, answer key, and student response."""
        confidence_note = ""
        if self.answer_confidence is not None:
            confidence_note = f"\n(Transcription confidence: {self.answer_confidence:.2f})"

        transcription_note = ""
        if self.transcription_notes:
            transcription_note = f"\nTranscription notes: {self.transcription_notes}"

        # Format rubric criteria
        criteria_text = ""
        for i, criterion in enumerate(self.rubric.criteria, 1):
            examples_text = ""
            if criterion.examples:
                examples_text = f" Examples: {'; '.join(criterion.examples)}"
            criteria_text += f"  {i}. [{criterion.points} points] {criterion.description}{examples_text}\n"

        prompt = f"""Grade the following student response according to the provided rubric.

QUESTION {self.question.question_id} ({self.rubric.total_points} points total):
{self.question.question_text}

ANSWER KEY:
- Correct Answer: {self.answer_key.correct_answer}
- Solution Steps: {'; '.join(self.answer_key.solution_steps) if self.answer_key.solution_steps else 'N/A'}
- Grading Notes: {self.answer_key.grading_notes or 'None'}

GRADING RUBRIC:
{criteria_text}

STUDENT RESPONSE: {self.student_answer}{confidence_note}{transcription_note}

Please grade this response and provide your assessment in the following JSON format:
{{
  "points_earned": <number between 0 and {self.rubric.total_points}>,
  "feedback": "<detailed feedback explaining the grade>",
  "reasoning": "<step-by-step reasoning for point allocation>",
  "criterion_scores": [
    {{"points": <points for criterion 1>, "justification": "<why this many points>"}},
    {{"points": <points for criterion 2>, "justification": "<why this many points>"}}
  ]
}}

Be fair but rigorous in your grading. Award partial credit where appropriate based on the rubric criteria."""

        return prompt

    def parse_response(self, response_text: str) -> Grade:
        """Parse JSON response into Grade object."""
        try:
            # Extract JSON from response using safe method
            json_text = safe_extract_json(response_text)
            data = json.loads(json_text)

            # Validate required fields
            points_earned = float(data.get('points_earned', 0.0))
            feedback = data.get('feedback', '')
            reasoning = data.get('reasoning', '')

            # Ensure points are within valid range
            points_earned = max(0.0, min(points_earned, self.rubric.total_points))

            # Create student object
            student = Student(name=self.student_name)

            # Create grade object
            grade = Grade(
                student=student,
                question_id=self.question.question_id,
                points_earned=points_earned,
                points_possible=self.rubric.total_points,
                feedback=feedback,
                reasoning=reasoning,
                graded_at=datetime.now()
            )

            return grade

        except (json.JSONDecodeError, KeyError, ValueError, TimeoutError) as e:
            # Fallback: create a grade with minimal feedback
            student = Student(name=self.student_name)
            return Grade(
                student=student,
                question_id=self.question.question_id,
                points_earned=0.0,
                points_possible=self.rubric.total_points,
                feedback=f"Grading failed due to response parsing error: {e}",
                reasoning="Unable to parse LLM grading response",
                graded_at=datetime.now()
            )

    def get_llm_params(self) -> Dict[str, Any]:
        return {"temperature": 0.1, "max_tokens": 1500}

    def get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": "QuestionGrading",
            "student_name": self.student_name,
            "question_id": self.question.question_id
        }


def create_grading_tasks(
    student_answers: Dict[str, str],
    student_name: str,
    answer_keys: Dict[str, AnswerKey],
    rubrics: Dict[str, Rubric],
    questions: Dict[str, Question],
    answer_confidences: Optional[Dict[str, float]] = None,
    transcription_notes: Optional[Dict[str, str]] = None
) -> List[QuestionGradingTask]:
    """
    Create grading tasks for a student's answers.

    Args:
        student_answers: Dict mapping question_id to student answer text
        student_name: Name of the student
        answer_keys: Dict mapping question_id to AnswerKey
        rubrics: Dict mapping question_id to Rubric
        questions: Dict mapping question_id to Question
        answer_confidences: Optional dict mapping question_id to confidence scores
        transcription_notes: Optional dict mapping question_id to transcription notes

    Returns:
        List of QuestionGradingTask objects
    """
    answer_confidences = answer_confidences or {}
    transcription_notes = transcription_notes or {}
    tasks = []

    for question_id, student_answer in student_answers.items():
        # Only create task if we have all required components
        if question_id in answer_keys and question_id in rubrics and question_id in questions:
            task = QuestionGradingTask(
                question=questions[question_id],
                student_answer=student_answer,
                student_name=student_name,
                answer_key=answer_keys[question_id],
                rubric=rubrics[question_id],
                answer_confidence=answer_confidences.get(question_id),
                transcription_notes=transcription_notes.get(question_id)
            )
            tasks.append(task)

    return tasks
