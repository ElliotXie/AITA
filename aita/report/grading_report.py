"""
Grading Report Generator

Generates comprehensive HTML reports for student grading with:
- Question text and rubric criteria
- Student answers and grading feedback
- Score breakdowns and visual indicators
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from aita.domain.models import ExamSpec, Question, Rubric, AnswerKey, Grade

logger = logging.getLogger(__name__)


def sort_question_id(question_id: str) -> Tuple[int, str]:
    """
    Sort key function for question IDs like '1a', '1b', '2a', '10b', etc.

    Args:
        question_id: Question identifier (e.g., '1a', '2b', '10c')

    Returns:
        Tuple of (numeric_part, letter_part) for sorting
    """
    match = re.match(r'(\d+)([a-z]*)', question_id.lower())
    if match:
        num_part = int(match.group(1))
        letter_part = match.group(2) or ''
        return (num_part, letter_part)
    return (0, question_id)


class GradingReportGenerator:
    """
    Generates comprehensive HTML grading reports for students.

    Creates organized layout with:
    - Overall score summary and statistics
    - Per-question breakdown with rubrics
    - Student answers and grading feedback
    - Visual score indicators
    """

    def __init__(self, exam_spec: ExamSpec):
        """
        Initialize the grading report generator.

        Args:
            exam_spec: Exam specification with questions
        """
        self.exam_spec = exam_spec
        logger.info(f"GradingReportGenerator initialized for exam: {exam_spec.exam_name}")

    def generate_student_report(
        self,
        student_name: str,
        grading_data: Dict[str, Any],
        rubrics: Dict[str, Rubric],
        student_answers: Dict[str, str],
        output_file: Path
    ) -> Path:
        """
        Generate HTML grading report for a single student.

        Args:
            student_name: Student name
            grading_data: Grading results data with question_grades
            rubrics: Dictionary of Rubric objects keyed by question_id
            student_answers: Dictionary of student answers keyed by question_id
            output_file: Path to save HTML report

        Returns:
            Path to generated HTML file
        """
        logger.info(f"Generating grading report for {student_name}")

        # Create question details with all information
        question_details = self._create_question_details(
            grading_data["question_grades"],
            rubrics,
            student_answers
        )

        # Calculate summary statistics
        stats = self._calculate_stats(grading_data, question_details)

        # Prepare template data
        template_data = {
            'student_name': student_name,
            'exam_name': self.exam_spec.exam_name,
            'total_score': grading_data['total_score'],
            'total_possible': grading_data['total_possible'],
            'percentage': grading_data['percentage'],
            'grade_letter': grading_data['grade_letter'],
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question_details': question_details,
            'stats': stats
        }

        # Render HTML
        html_content = self._render_html_template(template_data)

        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Grading report saved to {output_file}")
        return output_file

    def _create_question_details(
        self,
        question_grades: List[Dict[str, Any]],
        rubrics: Dict[str, Rubric],
        student_answers: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Create detailed information for each question.

        Args:
            question_grades: List of grading results per question
            rubrics: Dictionary of rubrics
            student_answers: Dictionary of student answers

        Returns:
            List of dictionaries with comprehensive question data
        """
        details = []

        for grade_data in question_grades:
            question_id = grade_data['question_id']
            question = self.exam_spec.get_question(question_id)
            rubric = rubrics.get(question_id)

            if not question:
                logger.warning(f"Question {question_id} not found in exam spec")
                continue

            # Calculate percentage for color coding
            percentage = grade_data['percentage']
            if percentage >= 90:
                score_class = 'excellent'
            elif percentage >= 80:
                score_class = 'good'
            elif percentage >= 70:
                score_class = 'passing'
            elif percentage >= 50:
                score_class = 'struggling'
            else:
                score_class = 'failing'

            detail = {
                'question_id': question_id,
                'question_text': question.question_text,
                'points_possible': grade_data['points_possible'],
                'points_earned': grade_data['points_earned'],
                'percentage': percentage,
                'score_class': score_class,
                'feedback': grade_data['feedback'],
                'reasoning': grade_data.get('reasoning', ''),
                'student_answer': student_answers.get(question_id, '[No answer transcribed]'),
                'rubric_criteria': []
            }

            # Add rubric criteria if available
            if rubric:
                for criterion in rubric.criteria:
                    detail['rubric_criteria'].append({
                        'points': criterion.points,
                        'description': criterion.description,
                        'examples': criterion.examples
                    })

            details.append(detail)

        # Sort by question ID
        details.sort(key=lambda x: sort_question_id(x['question_id']))

        return details

    def _calculate_stats(
        self,
        grading_data: Dict[str, Any],
        question_details: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics for the report.

        Args:
            grading_data: Overall grading data
            question_details: Detailed question information

        Returns:
            Dictionary with statistics
        """
        if not question_details:
            return {
                'total_questions': 0,
                'average_score': 0.0,
                'best_question': None,
                'worst_question': None,
                'processing_time': grading_data.get('processing_time', 0.0)
            }

        # Find best and worst questions
        sorted_by_score = sorted(question_details, key=lambda x: x['percentage'])
        best_question = sorted_by_score[-1] if sorted_by_score else None
        worst_question = sorted_by_score[0] if sorted_by_score else None

        stats = {
            'total_questions': len(question_details),
            'average_score': grading_data['percentage'],
            'best_question': {
                'id': best_question['question_id'],
                'score': f"{best_question['points_earned']}/{best_question['points_possible']}",
                'percentage': best_question['percentage']
            } if best_question else None,
            'worst_question': {
                'id': worst_question['question_id'],
                'score': f"{worst_question['points_earned']}/{worst_question['points_possible']}",
                'percentage': worst_question['percentage']
            } if worst_question else None,
            'processing_time': grading_data.get('processing_time', 0.0)
        }

        return stats

    def _render_html_template(self, data: Dict[str, Any]) -> str:
        """
        Render grading report using inline HTML template.

        Args:
            data: Template data

        Returns:
            HTML string
        """
        # Determine overall score color
        if data['percentage'] >= 90:
            overall_color = '#10b981'
        elif data['percentage'] >= 80:
            overall_color = '#84cc16'
        elif data['percentage'] >= 70:
            overall_color = '#eab308'
        elif data['percentage'] >= 60:
            overall_color = '#f59e0b'
        else:
            overall_color = '#ef4444'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grading Report - {data['student_name']}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding-bottom: 50px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

        /* Header Section */
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .meta {{ font-size: 1.1em; opacity: 0.9; margin-bottom: 20px; }}
        .overall-score {{
            background: rgba(255,255,255,0.2);
            padding: 20px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .score-text {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .grade-letter {{
            font-size: 3em;
            font-weight: bold;
            background: {overall_color};
            padding: 10px 30px;
            border-radius: 10px;
        }}

        /* Progress Bar */
        .progress-bar-container {{
            margin-top: 15px;
            background: rgba(255,255,255,0.3);
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-bar-fill {{
            height: 100%;
            background: {overall_color};
            transition: width 0.5s ease;
        }}

        /* Stats Cards */
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card .label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        .stat-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-card .subtext {{
            font-size: 0.9em;
            color: #888;
            margin-top: 5px;
        }}

        /* Question Cards */
        .question-card {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .question-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .question-id {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
            background: #f0f0ff;
            padding: 8px 20px;
            border-radius: 8px;
        }}
        .question-score {{
            text-align: right;
        }}
        .score-earned {{
            font-size: 2em;
            font-weight: bold;
        }}
        .score-percentage {{
            font-size: 1.1em;
            margin-top: 5px;
        }}

        /* Score color classes */
        .excellent {{ color: #10b981; }}
        .good {{ color: #84cc16; }}
        .passing {{ color: #eab308; }}
        .struggling {{ color: #f59e0b; }}
        .failing {{ color: #ef4444; }}

        /* Question Content */
        .question-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 20px;
        }}
        .question-panel, .answer-panel {{
            padding: 20px;
            border-radius: 8px;
        }}
        .question-panel {{
            background: #f8f9ff;
            border-left: 4px solid #667eea;
        }}
        .answer-panel {{
            background: #fffbf0;
            border-left: 4px solid #f59e0b;
        }}
        .panel-header {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #333;
            display: flex;
            justify-content: space-between;
        }}
        .points-badge {{
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        .panel-content {{
            font-size: 1.05em;
            line-height: 1.8;
            color: #444;
            white-space: pre-wrap;
        }}
        .answer-text {{
            font-family: 'Courier New', monospace;
            background: white;
            padding: 15px;
            border-radius: 5px;
        }}

        /* Rubric Section */
        .rubric-section {{
            margin-top: 20px;
        }}
        .rubric-title {{
            font-weight: bold;
            font-size: 1.1em;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .criterion {{
            background: white;
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}
        .criterion-header {{
            display: flex;
            justify-content: space-between;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .criterion-points {{
            color: #667eea;
        }}
        .criterion-desc {{
            font-size: 0.95em;
            color: #555;
        }}
        .criterion-examples {{
            font-size: 0.9em;
            color: #777;
            margin-top: 8px;
            padding-left: 15px;
            border-left: 2px solid #e0e0e0;
        }}

        /* Feedback Section */
        .feedback-section {{
            margin-top: 20px;
            background: #f0fff4;
            border-left: 4px solid #10b981;
            padding: 20px;
            border-radius: 8px;
        }}
        .feedback-title {{
            font-weight: bold;
            font-size: 1.1em;
            color: #10b981;
            margin-bottom: 10px;
        }}
        .feedback-content {{
            line-height: 1.8;
            color: #333;
        }}

        /* Reasoning Section */
        .reasoning-section {{
            margin-top: 15px;
            background: #fef3f2;
            border-left: 4px solid #f59e0b;
            padding: 20px;
            border-radius: 8px;
        }}
        .reasoning-title {{
            font-weight: bold;
            font-size: 1.1em;
            color: #f59e0b;
            margin-bottom: 10px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .reasoning-content {{
            line-height: 1.8;
            color: #333;
            margin-top: 10px;
        }}
        .toggle-icon {{
            font-size: 0.8em;
        }}

        /* Responsive Design */
        @media (max-width: 968px) {{
            .question-content {{ grid-template-columns: 1fr; }}
            .overall-score {{ flex-direction: column; gap: 20px; }}
        }}

        /* Print Styles */
        @media print {{
            body {{ background: white; }}
            .question-card {{ page-break-inside: avoid; }}
            .header {{ background: #667eea !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{data['student_name']}</h1>
            <div class="meta">
                <div>{data['exam_name']}</div>
                <div>Generated: {data['generation_time']}</div>
            </div>
            <div class="overall-score">
                <div>
                    <div class="score-text">{data['total_score']:.1f} / {data['total_possible']:.1f}</div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {data['percentage']:.1f}%"></div>
                    </div>
                </div>
                <div class="grade-letter">{data['grade_letter']}</div>
            </div>
        </div>

        <!-- Stats Cards -->
        <div class="stats">
            <div class="stat-card">
                <div class="label">Overall Score</div>
                <div class="value">{data['percentage']:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="label">Questions</div>
                <div class="value">{data['stats']['total_questions']}</div>
            </div>
"""

        # Add best/worst question stats
        if data['stats']['best_question']:
            html += f"""
            <div class="stat-card">
                <div class="label">Best Question</div>
                <div class="value">{data['stats']['best_question']['id']}</div>
                <div class="subtext">{data['stats']['best_question']['score']} ({data['stats']['best_question']['percentage']:.0f}%)</div>
            </div>
"""

        if data['stats']['worst_question']:
            html += f"""
            <div class="stat-card">
                <div class="label">Needs Improvement</div>
                <div class="value">{data['stats']['worst_question']['id']}</div>
                <div class="subtext">{data['stats']['worst_question']['score']} ({data['stats']['worst_question']['percentage']:.0f}%)</div>
            </div>
"""

        html += """
        </div>
"""

        # Add question cards
        for q in data['question_details']:
            html += f"""
        <div class="question-card">
            <div class="question-header">
                <div class="question-id">Question {q['question_id']}</div>
                <div class="question-score">
                    <div class="score-earned {q['score_class']}">{q['points_earned']:.1f} / {q['points_possible']:.1f}</div>
                    <div class="score-percentage {q['score_class']}">{q['percentage']:.1f}%</div>
                </div>
            </div>

            <div class="question-content">
                <!-- Question Panel -->
                <div class="question-panel">
                    <div class="panel-header">
                        <span>Question</span>
                        <span class="points-badge">{q['points_possible']:.0f} points</span>
                    </div>
                    <div class="panel-content">{q['question_text']}</div>

                    <!-- Rubric -->
"""
            if q['rubric_criteria']:
                html += """
                    <div class="rubric-section">
                        <div class="rubric-title">Grading Rubric</div>
"""
                for crit in q['rubric_criteria']:
                    html += f"""
                        <div class="criterion">
                            <div class="criterion-header">
                                <span>{crit['description'][:80]}{'...' if len(crit['description']) > 80 else ''}</span>
                                <span class="criterion-points">{crit['points']:.1f} pts</span>
                            </div>
                            <div class="criterion-desc">{crit['description']}</div>
"""
                    if crit['examples']:
                        html += """
                            <div class="criterion-examples">
                                <strong>Examples:</strong><br>
"""
                        for ex in crit['examples'][:2]:  # Show first 2 examples
                            html += f"                                • {ex}<br>\n"
                        html += """
                            </div>
"""
                    html += """
                        </div>
"""
                html += """
                    </div>
"""
            html += """
                </div>

                <!-- Answer Panel -->
                <div class="answer-panel">
                    <div class="panel-header">
                        <span>Student Answer</span>
                    </div>
                    <div class="answer-text panel-content">{answer_text}</div>
                </div>
            </div>

            <!-- Feedback -->
            <div class="feedback-section">
                <div class="feedback-title">Grading Feedback</div>
                <div class="feedback-content">{feedback}</div>
            </div>
""".format(
                answer_text=q['student_answer'],
                feedback=q['feedback']
            )

            # Add reasoning if available
            if q['reasoning']:
                html += f"""
            <!-- Detailed Reasoning -->
            <div class="reasoning-section">
                <div class="reasoning-title" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
                    <span>Detailed Grading Reasoning</span>
                    <span class="toggle-icon">▼</span>
                </div>
                <div class="reasoning-content" style="display: none;">{q['reasoning']}</div>
            </div>
"""

            html += """
        </div>
"""

        html += """
    </div>
</body>
</html>"""

        return html


def generate_index_page(
    grading_results_dir: Path,
    exam_name: str,
    student_summaries: List[Dict[str, Any]]
) -> Path:
    """
    Generate an index/router HTML page with links to all student reports.

    Args:
        grading_results_dir: Grading results directory
        exam_name: Name of the exam
        student_summaries: List of student summary data

    Returns:
        Path to generated index HTML file
    """
    # Sort students by name
    student_summaries.sort(key=lambda x: x['student_name'])

    # Calculate class statistics
    if student_summaries:
        scores = [s['percentage'] for s in student_summaries]
        class_avg = sum(scores) / len(scores)
        class_median = sorted(scores)[len(scores) // 2]
        class_max = max(scores)
        class_min = min(scores)

        # Grade distribution
        grade_dist = {}
        for s in student_summaries:
            letter = s['grade_letter']
            grade_dist[letter] = grade_dist.get(letter, 0) + 1
    else:
        class_avg = class_median = class_max = class_min = 0.0
        grade_dist = {}

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grading Reports - {exam_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding-bottom: 50px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .header h1 {{ font-size: 3em; margin-bottom: 10px; }}
        .header .subtitle {{ font-size: 1.3em; opacity: 0.9; }}

        /* Stats Section */
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card .label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-card .subtext {{
            font-size: 0.95em;
            color: #888;
            margin-top: 8px;
        }}

        /* Grade Distribution */
        .grade-dist {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .grade-dist h2 {{
            color: #667eea;
            margin-bottom: 20px;
        }}
        .grade-bars {{
            display: flex;
            gap: 15px;
            align-items: flex-end;
            height: 200px;
        }}
        .grade-bar {{
            flex: 1;
            background: #667eea;
            border-radius: 5px 5px 0 0;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            align-items: center;
            color: white;
            font-weight: bold;
            padding: 10px;
            min-height: 40px;
            transition: all 0.3s ease;
        }}
        .grade-bar:hover {{ transform: translateY(-5px); opacity: 0.9; }}
        .grade-bar .count {{ font-size: 1.3em; margin-bottom: 5px; }}
        .grade-bar .label {{ font-size: 0.9em; }}

        /* Student List */
        .student-list {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .student-list h2 {{
            color: #667eea;
            margin-bottom: 20px;
        }}
        .student-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .student-table th {{
            background: #f8f9ff;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: #667eea;
            border-bottom: 2px solid #667eea;
        }}
        .student-table td {{
            padding: 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .student-table tr:hover {{
            background: #f8f9ff;
        }}
        .student-name {{
            font-weight: 600;
            color: #333;
        }}
        .student-link {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 8px 20px;
            border-radius: 5px;
            text-decoration: none;
            transition: all 0.3s ease;
        }}
        .student-link:hover {{
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }}
        .score {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        .grade-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .grade-A {{ background: #10b981; }}
        .grade-B {{ background: #84cc16; }}
        .grade-C {{ background: #eab308; }}
        .grade-D {{ background: #f59e0b; }}
        .grade-F {{ background: #ef4444; }}

        /* Responsive */
        @media (max-width: 768px) {{
            .student-table {{ font-size: 0.9em; }}
            .student-table th, .student-table td {{ padding: 10px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Grading Dashboard</h1>
            <div class="subtitle">{exam_name}</div>
            <div class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>

        <!-- Class Statistics -->
        <div class="stats">
            <div class="stat-card">
                <div class="label">Total Students</div>
                <div class="value">{len(student_summaries)}</div>
            </div>
            <div class="stat-card">
                <div class="label">Class Average</div>
                <div class="value">{class_avg:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="label">Median Score</div>
                <div class="value">{class_median:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="label">Highest Score</div>
                <div class="value">{class_max:.1f}%</div>
                <div class="subtext">Low: {class_min:.1f}%</div>
            </div>
        </div>

        <!-- Grade Distribution -->
        <div class="grade-dist">
            <h2>Grade Distribution</h2>
            <div class="grade-bars">
"""

    # Add grade distribution bars
    grade_letters = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F']
    max_count = max(grade_dist.values()) if grade_dist else 1

    for grade in grade_letters:
        count = grade_dist.get(grade, 0)
        height_pct = (count / max_count * 100) if max_count > 0 else 0
        html += f"""
                <div class="grade-bar" style="height: {height_pct}%;">
                    <div class="count">{count}</div>
                    <div class="label">{grade}</div>
                </div>
"""

    html += """
            </div>
        </div>

        <!-- Student List -->
        <div class="student-list">
            <h2>Student Reports</h2>
            <table class="student-table">
                <thead>
                    <tr>
                        <th>Student Name</th>
                        <th>Score</th>
                        <th>Percentage</th>
                        <th>Grade</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add student rows
    for student in student_summaries:
        # Determine grade class for badge color
        grade_letter = student['grade_letter']
        if grade_letter.startswith('A'):
            grade_class = 'grade-A'
        elif grade_letter.startswith('B'):
            grade_class = 'grade-B'
        elif grade_letter.startswith('C'):
            grade_class = 'grade-C'
        elif grade_letter.startswith('D'):
            grade_class = 'grade-D'
        else:
            grade_class = 'grade-F'

        html += f"""
                    <tr>
                        <td class="student-name">{student['student_name']}</td>
                        <td class="score">{student['total_score']:.1f} / {student['total_possible']:.1f}</td>
                        <td class="score">{student['percentage']:.1f}%</td>
                        <td><span class="grade-badge {grade_class}">{grade_letter}</span></td>
                        <td><a href="{student['report_link']}" class="student-link">View Report →</a></td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

    # Save index page
    index_file = grading_results_dir / "index.html"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Generated index page at {index_file}")
    return index_file


def generate_grading_reports(
    data_dir: Path,
    grading_results_dir: Optional[Path] = None
) -> List[Path]:
    """
    Generate grading reports for all graded students and an index page.

    Args:
        data_dir: Base data directory
        grading_results_dir: Optional custom grading results directory

    Returns:
        List of paths to generated HTML reports (including index page)
    """
    # Determine paths
    if grading_results_dir is None:
        grading_results_dir = data_dir.parent / "intermediateproduct" / "grading_results"

    if not grading_results_dir.exists():
        raise ValueError(f"Grading results directory not found: {grading_results_dir}")

    # Load exam spec
    exam_spec_file = data_dir / "results" / "exam_spec.json"
    if not exam_spec_file.exists():
        raise ValueError(f"Exam specification not found: {exam_spec_file}")

    exam_spec = ExamSpec.load_from_file(exam_spec_file)

    # Load rubrics
    rubrics_file = data_dir.parent / "intermediateproduct" / "rubrics" / "generated_rubrics.json"
    if not rubrics_file.exists():
        raise ValueError(f"Rubrics not found: {rubrics_file}")

    with open(rubrics_file, 'r', encoding='utf-8') as f:
        rubrics_data = json.load(f)
        rubrics = {r["question_id"]: Rubric.from_dict(r) for r in rubrics_data["rubrics"]}

    # Initialize generator
    generator = GradingReportGenerator(exam_spec)

    # Find student folders
    students_dir = grading_results_dir / "students"
    student_folders = [
        d for d in students_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]

    generated_reports = []
    student_summaries = []

    for student_folder in student_folders:
        try:
            student_name = student_folder.name

            # Load grading data
            detailed_grades_file = student_folder / "detailed_grades.json"
            if not detailed_grades_file.exists():
                logger.warning(f"Detailed grades not found for {student_name}, skipping")
                continue

            with open(detailed_grades_file, 'r', encoding='utf-8') as f:
                grading_data = json.load(f)

            # Load transcriptions for student answers
            transcription_file = data_dir.parent / "intermediateproduct" / "transcription_results" / student_name / "question_transcriptions.json"
            student_answers = {}
            if transcription_file.exists():
                with open(transcription_file, 'r', encoding='utf-8') as f:
                    trans_data = json.load(f)
                    student_answers = {
                        qa["question_id"]: qa["raw_text"]
                        for qa in trans_data.get("question_answers", [])
                    }

            # Generate report
            output_file = student_folder / "grading_report.html"
            report_path = generator.generate_student_report(
                student_name=student_name,
                grading_data=grading_data,
                rubrics=rubrics,
                student_answers=student_answers,
                output_file=output_file
            )

            generated_reports.append(report_path)
            logger.info(f"Generated grading report for {student_name}")

            # Collect summary for index page
            student_summaries.append({
                'student_name': student_name,
                'total_score': grading_data['total_score'],
                'total_possible': grading_data['total_possible'],
                'percentage': grading_data['percentage'],
                'grade_letter': grading_data['grade_letter'],
                'report_link': f"students/{student_name}/grading_report.html"
            })

        except Exception as e:
            logger.error(f"Failed to generate grading report for {student_folder.name}: {e}")

    # Generate index page
    if student_summaries:
        index_path = generate_index_page(
            grading_results_dir=grading_results_dir,
            exam_name=exam_spec.exam_name,
            student_summaries=student_summaries
        )
        generated_reports.insert(0, index_path)

    logger.info(f"Generated {len(generated_reports)} grading reports (including index)")
    return generated_reports
