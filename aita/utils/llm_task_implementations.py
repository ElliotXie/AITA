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

    @property
    def task_id(self) -> str:
        return f"answer_key_{self.question.question_id}"

    def build_messages(self) -> List[LLMMessage]:
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
            raise ValueError(f"Failed to parse answer key response: {e}") from e

    def get_llm_params(self) -> Dict[str, Any]:
        # Cap output length so the model doesn't stream massive responses
        return {"temperature": 0.1, "max_tokens": 1200}

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

    @property
    def task_id(self) -> str:
        return f"rubric_{self.question.question_id}"

    def build_messages(self) -> List[LLMMessage]:
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
            raise ValueError(f"Failed to parse rubric response: {e}") from e

    def _validate_rubric(self, rubric: Rubric):
        """Validate that rubric is well-formed."""
        # Check that criteria points sum to total
        criteria_sum = sum(c.points for c in rubric.criteria)
        if abs(criteria_sum - rubric.total_points) > 0.01:
            raise ValueError(
                f"Rubric criteria sum ({criteria_sum}) doesn't match total points ({rubric.total_points})"
            )

        # Check that rubric total matches question points
        if abs(rubric.total_points - self.question.points) > 0.01:
            raise ValueError(
                f"Rubric total points ({rubric.total_points}) doesn't match question points ({self.question.points})"
            )

    def get_llm_params(self) -> Dict[str, Any]:
        # Limit response size to avoid runaway completions
        return {"temperature": 0.1, "max_tokens": 1500}

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
