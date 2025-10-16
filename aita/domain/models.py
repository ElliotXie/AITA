from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json


class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    CALCULATION = "calculation"
    DIAGRAM = "diagram"


class GradeLevel(Enum):
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    C_MINUS = "C-"
    D_PLUS = "D+"
    D = "D"
    F = "F"


@dataclass
class Student:
    name: str
    student_id: Optional[str] = None
    email: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Question:
    question_id: str  # e.g., "1a", "2b", "3"
    question_text: str
    points: float
    question_type: QuestionType = QuestionType.SHORT_ANSWER
    page_number: Optional[int] = None
    image_bounds: Optional[Dict[str, int]] = None  # x, y, width, height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "points": self.points,
            "question_type": self.question_type.value,
            "page_number": self.page_number,
            "image_bounds": self.image_bounds
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        return cls(
            question_id=data["question_id"],
            question_text=data["question_text"],
            points=data["points"],
            question_type=QuestionType(data.get("question_type", "short_answer")),
            page_number=data.get("page_number"),
            image_bounds=data.get("image_bounds")
        )


@dataclass
class ExamSpec:
    exam_name: str
    total_pages: int
    questions: List[Question] = field(default_factory=list)
    total_points: Optional[float] = None

    def __post_init__(self):
        if self.total_points is None:
            self.total_points = sum(q.points for q in self.questions)

    def get_question(self, question_id: str) -> Optional[Question]:
        for question in self.questions:
            if question.question_id == question_id:
                return question
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exam_name": self.exam_name,
            "total_pages": self.total_pages,
            "total_points": self.total_points,
            "questions": [q.to_dict() for q in self.questions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExamSpec":
        questions = [Question.from_dict(q) for q in data.get("questions", [])]
        return cls(
            exam_name=data["exam_name"],
            total_pages=data["total_pages"],
            questions=questions,
            total_points=data.get("total_points")
        )

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "ExamSpec":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class AnswerKey:
    question_id: str
    correct_answer: str
    alternative_answers: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    grading_notes: Optional[str] = None
    solution_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "correct_answer": self.correct_answer,
            "alternative_answers": self.alternative_answers,
            "explanation": self.explanation,
            "grading_notes": self.grading_notes,
            "solution_steps": self.solution_steps
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerKey":
        return cls(
            question_id=data["question_id"],
            correct_answer=data["correct_answer"],
            alternative_answers=data.get("alternative_answers", []),
            explanation=data.get("explanation"),
            grading_notes=data.get("grading_notes"),
            solution_steps=data.get("solution_steps", [])
        )


@dataclass
class RubricCriterion:
    points: float
    description: str
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": self.points,
            "description": self.description,
            "examples": self.examples
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RubricCriterion":
        return cls(
            points=data["points"],
            description=data["description"],
            examples=data.get("examples", [])
        )


@dataclass
class Rubric:
    question_id: str
    total_points: float
    criteria: List[RubricCriterion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "total_points": self.total_points,
            "criteria": [c.to_dict() for c in self.criteria]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rubric":
        criteria = [RubricCriterion.from_dict(c) for c in data.get("criteria", [])]
        return cls(
            question_id=data["question_id"],
            total_points=data["total_points"],
            criteria=criteria
        )


@dataclass
class StudentAnswer:
    student: Student
    question_id: str
    raw_text: str  # OCR/transcribed text
    image_paths: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    transcription_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_name": self.student.name,
            "student_id": self.student.student_id,
            "question_id": self.question_id,
            "raw_text": self.raw_text,
            "image_paths": self.image_paths,
            "confidence": self.confidence,
            "transcription_notes": self.transcription_notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudentAnswer":
        student = Student(
            name=data["student_name"],
            student_id=data.get("student_id")
        )
        return cls(
            student=student,
            question_id=data["question_id"],
            raw_text=data["raw_text"],
            image_paths=data.get("image_paths", []),
            confidence=data.get("confidence"),
            transcription_notes=data.get("transcription_notes")
        )


@dataclass
class Grade:
    student: Student
    question_id: str
    points_earned: float
    points_possible: float
    feedback: str
    reasoning: Optional[str] = None
    grade_level: Optional[GradeLevel] = None
    graded_at: datetime = field(default_factory=datetime.now)

    @property
    def percentage(self) -> float:
        if self.points_possible == 0:
            return 0.0
        return (self.points_earned / self.points_possible) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_name": self.student.name,
            "student_id": self.student.student_id,
            "question_id": self.question_id,
            "points_earned": self.points_earned,
            "points_possible": self.points_possible,
            "feedback": self.feedback,
            "reasoning": self.reasoning,
            "grade_level": self.grade_level.value if self.grade_level else None,
            "percentage": self.percentage,
            "graded_at": self.graded_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Grade":
        student = Student(
            name=data["student_name"],
            student_id=data.get("student_id")
        )

        graded_at = datetime.fromisoformat(data["graded_at"]) if "graded_at" in data else datetime.now()
        grade_level = GradeLevel(data["grade_level"]) if data.get("grade_level") else None

        return cls(
            student=student,
            question_id=data["question_id"],
            points_earned=data["points_earned"],
            points_possible=data["points_possible"],
            feedback=data["feedback"],
            reasoning=data.get("reasoning"),
            grade_level=grade_level,
            graded_at=graded_at
        )


@dataclass
class StudentExam:
    student: Student
    exam_spec: ExamSpec
    answers: List[StudentAnswer] = field(default_factory=list)
    grades: List[Grade] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)

    @property
    def total_points_earned(self) -> float:
        return sum(grade.points_earned for grade in self.grades)

    @property
    def total_points_possible(self) -> float:
        return self.exam_spec.total_points or 0

    @property
    def percentage_score(self) -> float:
        if self.total_points_possible == 0:
            return 0.0
        return (self.total_points_earned / self.total_points_possible) * 100

    def get_answer(self, question_id: str) -> Optional[StudentAnswer]:
        for answer in self.answers:
            if answer.question_id == question_id:
                return answer
        return None

    def get_grade(self, question_id: str) -> Optional[Grade]:
        for grade in self.grades:
            if grade.question_id == question_id:
                return grade
        return None

    def add_answer(self, answer: StudentAnswer) -> None:
        existing = self.get_answer(answer.question_id)
        if existing:
            self.answers.remove(existing)
        self.answers.append(answer)

    def add_grade(self, grade: Grade) -> None:
        existing = self.get_grade(grade.question_id)
        if existing:
            self.grades.remove(existing)
        self.grades.append(grade)