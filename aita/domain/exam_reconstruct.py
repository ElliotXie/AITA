from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging

from .models import ExamSpec, Question, AnswerKey, Rubric, Student, StudentExam

logger = logging.getLogger(__name__)


class ExamReconstructor:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

    def save_exam_spec(self, exam_spec: ExamSpec) -> Path:
        file_path = self.results_dir / "exam_spec.json"
        exam_spec.save_to_file(file_path)
        logger.info(f"Saved exam specification to {file_path}")
        return file_path

    def load_exam_spec(self) -> Optional[ExamSpec]:
        file_path = self.results_dir / "exam_spec.json"
        if not file_path.exists():
            logger.warning(f"Exam specification not found at {file_path}")
            return None

        try:
            return ExamSpec.load_from_file(file_path)
        except Exception as e:
            logger.error(f"Failed to load exam specification: {e}")
            return None

    def save_answer_keys(self, answer_keys: List[AnswerKey]) -> Path:
        file_path = self.results_dir / "answer_key.json"
        data = {
            "answer_keys": [key.to_dict() for key in answer_keys]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved answer keys to {file_path}")
        return file_path

    def load_answer_keys(self) -> List[AnswerKey]:
        file_path = self.results_dir / "answer_key.json"
        if not file_path.exists():
            logger.warning(f"Answer keys not found at {file_path}")
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return [AnswerKey.from_dict(key) for key in data.get("answer_keys", [])]
        except Exception as e:
            logger.error(f"Failed to load answer keys: {e}")
            return []

    def save_rubrics(self, rubrics: List[Rubric]) -> Path:
        file_path = self.results_dir / "rubric.json"
        data = {
            "rubrics": [rubric.to_dict() for rubric in rubrics]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved rubrics to {file_path}")
        return file_path

    def load_rubrics(self) -> List[Rubric]:
        file_path = self.results_dir / "rubric.json"
        if not file_path.exists():
            logger.warning(f"Rubrics not found at {file_path}")
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return [Rubric.from_dict(rubric) for rubric in data.get("rubrics", [])]
        except Exception as e:
            logger.error(f"Failed to load rubrics: {e}")
            return []

    def save_grades(self, student_exams: List[StudentExam]) -> Path:
        file_path = self.results_dir / "grades.json"
        data = {
            "students": []
        }

        for student_exam in student_exams:
            student_data = {
                "student_name": student_exam.student.name,
                "student_id": student_exam.student.student_id,
                "total_points_earned": student_exam.total_points_earned,
                "total_points_possible": student_exam.total_points_possible,
                "percentage_score": student_exam.percentage_score,
                "grades": [grade.to_dict() for grade in student_exam.grades]
            }
            data["students"].append(student_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved grades to {file_path}")
        return file_path

    def load_grades(self) -> Dict[str, StudentExam]:
        file_path = self.results_dir / "grades.json"
        if not file_path.exists():
            logger.warning(f"Grades not found at {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            exam_spec = self.load_exam_spec()
            if not exam_spec:
                logger.error("Cannot load grades without exam specification")
                return {}

            student_exams = {}
            for student_data in data.get("students", []):
                student = Student(
                    name=student_data["student_name"],
                    student_id=student_data.get("student_id")
                )

                student_exam = StudentExam(
                    student=student,
                    exam_spec=exam_spec
                )

                # Load grades
                for grade_data in student_data.get("grades", []):
                    from .models import Grade
                    grade = Grade.from_dict(grade_data)
                    student_exam.add_grade(grade)

                student_exams[student.name] = student_exam

            return student_exams

        except Exception as e:
            logger.error(f"Failed to load grades: {e}")
            return {}

    def get_exam_summary(self) -> Dict[str, Any]:
        exam_spec = self.load_exam_spec()
        answer_keys = self.load_answer_keys()
        rubrics = self.load_rubrics()
        student_exams = self.load_grades()

        summary = {
            "exam_info": {
                "name": exam_spec.exam_name if exam_spec else "Unknown",
                "total_pages": exam_spec.total_pages if exam_spec else 0,
                "total_questions": len(exam_spec.questions) if exam_spec else 0,
                "total_points": exam_spec.total_points if exam_spec else 0
            },
            "grading_info": {
                "answer_keys_count": len(answer_keys),
                "rubrics_count": len(rubrics),
                "students_graded": len(student_exams)
            },
            "statistics": self._calculate_statistics(student_exams)
        }

        return summary

    def _calculate_statistics(self, student_exams: Dict[str, StudentExam]) -> Dict[str, Any]:
        if not student_exams:
            return {}

        scores = [exam.percentage_score for exam in student_exams.values()]

        return {
            "mean_score": sum(scores) / len(scores),
            "median_score": sorted(scores)[len(scores) // 2],
            "min_score": min(scores),
            "max_score": max(scores),
            "passing_students": len([s for s in scores if s >= 60]),
            "total_students": len(scores)
        }