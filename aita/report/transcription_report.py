"""
Transcription Report Generator

Generates HTML reports for student transcriptions with side-by-side
question/answer layout.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from jinja2 import Template, Environment, FileSystemLoader, select_autoescape

from aita.domain.models import ExamSpec, StudentAnswer, Question

logger = logging.getLogger(__name__)


class TranscriptionReportGenerator:
    """
    Generates HTML reports for student transcriptions.

    Creates side-by-side layout with:
    - Left: Original question text from ExamSpec
    - Right: Student's transcribed answer
    """

    def __init__(self, exam_spec: ExamSpec, template_dir: Optional[Path] = None):
        """
        Initialize the report generator.

        Args:
            exam_spec: Exam specification with questions
            template_dir: Optional custom template directory
        """
        self.exam_spec = exam_spec

        # Setup Jinja2 environment
        if template_dir and template_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(['html', 'xml'])
            )
        else:
            # Use default template from package
            default_template_dir = Path(__file__).parent.parent / "templates"
            self.env = Environment(
                loader=FileSystemLoader(str(default_template_dir)),
                autoescape=select_autoescape(['html', 'xml'])
            )

        logger.info(f"TranscriptionReportGenerator initialized for exam: {exam_spec.exam_name}")

    def generate_student_report(
        self,
        student_name: str,
        question_answers: List[StudentAnswer],
        output_file: Path
    ) -> Path:
        """
        Generate HTML report for a single student.

        Args:
            student_name: Student name
            question_answers: List of StudentAnswer objects
            output_file: Path to save HTML report

        Returns:
            Path to generated HTML file
        """
        logger.info(f"Generating report for {student_name}")

        # Create question/answer pairs
        qa_pairs = self._create_qa_pairs(question_answers)

        # Calculate statistics
        stats = self._calculate_stats(question_answers)

        # Prepare template data
        template_data = {
            'student_name': student_name,
            'exam_name': self.exam_spec.exam_name,
            'total_questions': len(self.exam_spec.questions),
            'answered_questions': len([qa for qa in qa_pairs if qa['answer_text'].strip()]),
            'total_points': self.exam_spec.total_points,
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'qa_pairs': qa_pairs,
            'stats': stats
        }

        # Render template
        try:
            template = self.env.get_template('transcription_report.html')
            html_content = template.render(**template_data)
        except Exception as e:
            logger.error(f"Failed to load template, using inline template: {e}")
            html_content = self._render_inline_template(template_data)

        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Report saved to {output_file}")
        return output_file

    def _create_qa_pairs(self, question_answers: List[StudentAnswer]) -> List[Dict[str, Any]]:
        """
        Create question/answer pairs for template.

        Args:
            question_answers: List of StudentAnswer objects

        Returns:
            List of dictionaries with question and answer data
        """
        qa_pairs = []

        # Create a mapping of question_id to StudentAnswer
        answer_map = {ans.question_id: ans for ans in question_answers}

        # Iterate through exam questions to maintain order
        for question in self.exam_spec.questions:
            answer = answer_map.get(question.question_id)

            qa_pair = {
                'question_id': question.question_id,
                'question_text': question.question_text,
                'points': question.points,
                'question_type': question.question_type.value,
                'page_number': question.page_number,
                'answer_text': answer.raw_text if answer else "[No transcription available]",
                'confidence': answer.confidence if answer else 0.0,
                'has_answer': answer is not None and answer.raw_text.strip(),
                'transcription_notes': answer.transcription_notes if answer else None,
                'image_paths': answer.image_paths if answer else []
            }

            qa_pairs.append(qa_pair)

        return qa_pairs

    def _calculate_stats(self, question_answers: List[StudentAnswer]) -> Dict[str, Any]:
        """
        Calculate statistics for report.

        Args:
            question_answers: List of StudentAnswer objects

        Returns:
            Dictionary with statistics
        """
        # Filter out empty answers
        valid_answers = [ans for ans in question_answers if ans.raw_text.strip()]

        # Calculate confidence distribution
        if valid_answers:
            confidences = [ans.confidence for ans in valid_answers if ans.confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            high_conf = len([c for c in confidences if c >= 0.8])
            med_conf = len([c for c in confidences if 0.5 <= c < 0.8])
            low_conf = len([c for c in confidences if c < 0.5])
        else:
            avg_confidence = 0.0
            high_conf = med_conf = low_conf = 0

        # Calculate answer length stats
        answer_lengths = [len(ans.raw_text) for ans in valid_answers]
        avg_answer_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0

        stats = {
            'total_questions': len(self.exam_spec.questions),
            'answered_questions': len(valid_answers),
            'coverage_percent': (len(valid_answers) / len(self.exam_spec.questions) * 100) if self.exam_spec.questions else 0,
            'average_confidence': avg_confidence,
            'high_confidence_count': high_conf,
            'medium_confidence_count': med_conf,
            'low_confidence_count': low_conf,
            'average_answer_length': avg_answer_length,
            'total_characters': sum(answer_lengths)
        }

        return stats

    def _render_inline_template(self, data: Dict[str, Any]) -> str:
        """
        Render report using inline template (fallback).

        Args:
            data: Template data

        Returns:
            HTML string
        """
        # Inline template as fallback
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcription Report - {data['student_name']}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .meta {{ font-size: 1.1em; opacity: 0.9; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-card .label {{ font-size: 0.9em; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .qa-pair {{ background: white; border-radius: 10px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .qa-pair h2 {{ color: #667eea; margin-bottom: 20px; font-size: 1.5em; }}
        .qa-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        .question-panel, .answer-panel {{ padding: 20px; border-radius: 8px; }}
        .question-panel {{ background: #f8f9ff; border-left: 4px solid #667eea; }}
        .answer-panel {{ background: #fff8f0; border-left: 4px solid #f59e0b; }}
        .panel-header {{ font-weight: bold; font-size: 1.1em; margin-bottom: 15px; color: #333; display: flex; justify-content: space-between; align-items: center; }}
        .points-badge {{ background: #667eea; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9em; }}
        .question-text {{ font-size: 1.05em; line-height: 1.8; color: #444; white-space: pre-wrap; }}
        .answer-text {{ font-size: 1.05em; line-height: 1.8; color: #444; white-space: pre-wrap; font-family: 'Courier New', monospace; background: white; padding: 15px; border-radius: 5px; }}
        .confidence-bar {{ margin-top: 15px; }}
        .confidence-bar .label {{ font-size: 0.85em; color: #666; margin-bottom: 5px; }}
        .confidence-bar .bar {{ height: 8px; background: #e5e7eb; border-radius: 10px; overflow: hidden; }}
        .confidence-bar .fill {{ height: 100%; transition: width 0.3s ease; }}
        .confidence-high {{ background: #10b981; }}
        .confidence-medium {{ background: #f59e0b; }}
        .confidence-low {{ background: #ef4444; }}
        .no-answer {{ color: #999; font-style: italic; }}
        @media (max-width: 968px) {{ .qa-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{data['student_name']}</h1>
            <div class="meta">
                <div>{data['exam_name']}</div>
                <div>Generated: {data['generation_time']}</div>
            </div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="label">Questions</div>
                <div class="value">{data['answered_questions']}/{data['total_questions']}</div>
            </div>
            <div class="stat-card">
                <div class="label">Coverage</div>
                <div class="value">{data['stats']['coverage_percent']:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="label">Avg Confidence</div>
                <div class="value">{data['stats']['average_confidence']:.0%}</div>
            </div>
            <div class="stat-card">
                <div class="label">Total Points</div>
                <div class="value">{data['total_points']}</div>
            </div>
        </div>
"""

        # Add Q&A pairs
        for qa in data['qa_pairs']:
            confidence_class = 'confidence-high' if qa['confidence'] >= 0.8 else 'confidence-medium' if qa['confidence'] >= 0.5 else 'confidence-low'
            confidence_width = qa['confidence'] * 100

            html += f"""
        <div class="qa-pair">
            <h2>Question {qa['question_id']}</h2>
            <div class="qa-grid">
                <div class="question-panel">
                    <div class="panel-header">
                        <span>Question</span>
                        <span class="points-badge">{qa['points']} points</span>
                    </div>
                    <div class="question-text">{qa['question_text']}</div>
                </div>
                <div class="answer-panel">
                    <div class="panel-header">
                        <span>Student Answer</span>
                    </div>
                    <div class="answer-text {'no-answer' if not qa['has_answer'] else ''}">{qa['answer_text']}</div>
                    {f'''<div class="confidence-bar">
                        <div class="label">Transcription Confidence: {qa['confidence']:.0%}</div>
                        <div class="bar">
                            <div class="fill {confidence_class}" style="width: {confidence_width}%"></div>
                        </div>
                    </div>''' if qa['has_answer'] else ''}
                </div>
            </div>
        </div>
"""

        html += """
    </div>
</body>
</html>"""

        return html


def generate_transcription_reports(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    student_names: Optional[List[str]] = None
) -> List[Path]:
    """
    Generate transcription reports for students.

    Args:
        data_dir: Base data directory
        output_dir: Optional custom output directory (defaults to student folders)
        student_names: Optional list of specific students to generate reports for

    Returns:
        List of paths to generated HTML reports
    """
    from aita.utils.transcription_helpers import load_exam_spec

    # Load exam spec
    exam_spec = load_exam_spec(data_dir)
    if not exam_spec:
        raise ValueError("ExamSpec not found. Run question extraction first.")

    # Initialize generator
    generator = TranscriptionReportGenerator(exam_spec)

    # Find transcription results
    transcription_dir = data_dir / "intermediateproduct" / "transcription_results"
    if not transcription_dir.exists():
        raise ValueError(f"Transcription results not found: {transcription_dir}")

    # Find student folders
    student_folders = [
        d for d in transcription_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]

    # Filter by specific students if requested
    if student_names:
        student_folders = [
            d for d in student_folders
            if d.name in student_names
        ]

    generated_reports = []

    for student_folder in student_folders:
        try:
            student_name = student_folder.name

            # Load question transcriptions
            question_file = student_folder / "question_transcriptions.json"
            if not question_file.exists():
                logger.warning(f"Question transcriptions not found for {student_name}, skipping")
                continue

            with open(question_file, 'r', encoding='utf-8') as f:
                question_data = json.load(f)

            # Convert to StudentAnswer objects
            question_answers = [
                StudentAnswer.from_dict(ans_data)
                for ans_data in question_data.get('question_answers', [])
            ]

            # Determine output file
            if output_dir:
                output_file = output_dir / student_name / "transcription_report.html"
            else:
                output_file = student_folder / "transcription_report.html"

            # Generate report
            report_path = generator.generate_student_report(
                student_name=student_name,
                question_answers=question_answers,
                output_file=output_file
            )

            generated_reports.append(report_path)

        except Exception as e:
            logger.error(f"Failed to generate report for {student_folder.name}: {e}")

    logger.info(f"Generated {len(generated_reports)} transcription reports")
    return generated_reports
