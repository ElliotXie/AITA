"""Cost analysis utilities for LLM API usage."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import statistics

from .cost_tracker import CostEntry, SessionSummary


@dataclass
class CostAnalysis:
    """Comprehensive cost analysis across multiple sessions."""
    total_sessions: int
    total_calls: int
    total_cost: float
    average_cost_per_session: float
    average_cost_per_call: float
    total_input_tokens: int
    total_output_tokens: int
    total_images: int
    cost_by_operation: Dict[str, float]
    cost_by_model: Dict[str, float]
    cost_by_date: Dict[str, float]
    most_expensive_session: Optional[SessionSummary]
    most_expensive_operation: Optional[Tuple[str, float]]
    time_period_start: Optional[str]
    time_period_end: Optional[str]


class CostAnalyzer:
    """Analyze cost tracking data across multiple sessions."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize cost analyzer.

        Args:
            data_dir: Directory containing cost tracking files
        """
        self.data_dir = data_dir or Path("C:/Users/ellio/OneDrive - UW-Madison/AITA/intermediateproduct/cost_tracking")
        self.data_dir = Path(self.data_dir)

    def load_session(self, session_file: Path) -> Optional[SessionSummary]:
        """
        Load a single session from file.

        Args:
            session_file: Path to session JSON file

        Returns:
            SessionSummary or None if loading fails
        """
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct entries
            entries = []
            for entry_data in data.get("entries", []):
                entries.append(CostEntry(**entry_data))

            return SessionSummary(
                session_id=data.get("session_id", "unknown"),
                start_time=data.get("start_time", ""),
                end_time=data.get("end_time"),
                total_calls=data.get("total_calls", 0),
                total_input_tokens=data.get("total_input_tokens", 0),
                total_output_tokens=data.get("total_output_tokens", 0),
                total_images=data.get("total_images", 0),
                total_cost=data.get("total_cost", 0.0),
                cost_by_operation=data.get("cost_by_operation", {}),
                cost_by_model=data.get("cost_by_model", {}),
                entries=entries
            )

        except Exception as e:
            print(f"Error loading session {session_file}: {e}")
            return None

    def analyze_all_sessions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        operation_filter: Optional[str] = None,
        model_filter: Optional[str] = None
    ) -> CostAnalysis:
        """
        Analyze all sessions in the data directory.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            operation_filter: Optional operation type filter
            model_filter: Optional model filter

        Returns:
            CostAnalysis object with comprehensive statistics
        """
        session_files = list(self.data_dir.glob("session_*.json"))

        if not session_files:
            return CostAnalysis(
                total_sessions=0,
                total_calls=0,
                total_cost=0.0,
                average_cost_per_session=0.0,
                average_cost_per_call=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_images=0,
                cost_by_operation={},
                cost_by_model={},
                cost_by_date={},
                most_expensive_session=None,
                most_expensive_operation=None,
                time_period_start=None,
                time_period_end=None
            )

        # Load all sessions
        sessions: List[SessionSummary] = []
        for file in session_files:
            session = self.load_session(file)
            if session:
                # Apply date filters
                if start_date or end_date:
                    session_date = datetime.fromisoformat(session.start_time) if session.start_time else None
                    if session_date:
                        if start_date and session_date < start_date:
                            continue
                        if end_date and session_date > end_date:
                            continue
                sessions.append(session)

        # Calculate aggregates
        total_cost = 0.0
        total_calls = 0
        total_input = 0
        total_output = 0
        total_images = 0
        cost_by_operation = {}
        cost_by_model = {}
        cost_by_date = {}
        most_expensive_session = None
        max_session_cost = 0.0

        for session in sessions:
            # Apply filters to individual entries if needed
            session_cost = 0.0
            session_calls = 0

            for entry in session.entries:
                # Apply operation filter
                if operation_filter and entry.operation_type != operation_filter:
                    continue

                # Apply model filter
                if model_filter and entry.model != model_filter:
                    continue

                # Count this entry
                session_cost += entry.total_cost
                session_calls += 1
                total_input += entry.input_tokens
                total_output += entry.output_tokens
                total_images += entry.image_count

                # Aggregate by operation
                if entry.operation_type not in cost_by_operation:
                    cost_by_operation[entry.operation_type] = 0.0
                cost_by_operation[entry.operation_type] += entry.total_cost

                # Aggregate by model
                if entry.model not in cost_by_model:
                    cost_by_model[entry.model] = 0.0
                cost_by_model[entry.model] += entry.total_cost

                # Aggregate by date
                if entry.timestamp:
                    date_key = entry.timestamp.split("T")[0]
                    if date_key not in cost_by_date:
                        cost_by_date[date_key] = 0.0
                    cost_by_date[date_key] += entry.total_cost

            total_cost += session_cost
            total_calls += session_calls

            # Track most expensive session
            if session_cost > max_session_cost:
                max_session_cost = session_cost
                most_expensive_session = session

        # Find most expensive operation
        most_expensive_operation = None
        if cost_by_operation:
            most_expensive_operation = max(cost_by_operation.items(), key=lambda x: x[1])

        # Determine time period
        time_period_start = None
        time_period_end = None
        if sessions:
            start_times = [s.start_time for s in sessions if s.start_time]
            end_times = [s.end_time for s in sessions if s.end_time]
            if start_times:
                time_period_start = min(start_times)
            if end_times:
                time_period_end = max(end_times)

        # Calculate averages
        avg_cost_per_session = total_cost / len(sessions) if sessions else 0.0
        avg_cost_per_call = total_cost / total_calls if total_calls > 0 else 0.0

        return CostAnalysis(
            total_sessions=len(sessions),
            total_calls=total_calls,
            total_cost=round(total_cost, 6),
            average_cost_per_session=round(avg_cost_per_session, 6),
            average_cost_per_call=round(avg_cost_per_call, 6),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_images=total_images,
            cost_by_operation={k: round(v, 6) for k, v in cost_by_operation.items()},
            cost_by_model={k: round(v, 6) for k, v in cost_by_model.items()},
            cost_by_date={k: round(v, 6) for k, v in cost_by_date.items()},
            most_expensive_session=most_expensive_session,
            most_expensive_operation=most_expensive_operation,
            time_period_start=time_period_start,
            time_period_end=time_period_end
        )

    def get_cost_projections(
        self,
        students_per_exam: int,
        pages_per_student: int = 5
    ) -> Dict[str, float]:
        """
        Project costs for a full exam based on historical data.

        Args:
            students_per_exam: Number of students in the exam
            pages_per_student: Average pages per student

        Returns:
            Dictionary with cost projections
        """
        analysis = self.analyze_all_sessions()

        if analysis.total_calls == 0:
            return {
                "error": "No historical data available for projections",
                "students": students_per_exam,
                "pages_per_student": pages_per_student
            }

        # Estimate costs per operation
        cost_per_name = analysis.cost_by_operation.get("name_extraction", 0) / max(analysis.total_sessions, 1)
        cost_per_transcription = analysis.cost_by_operation.get("transcription", 0) / max(analysis.total_sessions, 1)
        cost_per_question = analysis.cost_by_operation.get("question_extraction", 0) / max(analysis.total_sessions, 1)
        cost_per_grading = analysis.cost_by_operation.get("grading", 0) / max(analysis.total_sessions, 1)

        # Project costs
        projected_name_cost = cost_per_name * students_per_exam
        projected_transcription_cost = cost_per_transcription * students_per_exam * pages_per_student
        projected_question_cost = cost_per_question * pages_per_student  # Once per exam
        projected_grading_cost = cost_per_grading * students_per_exam * pages_per_student

        total_projected = (
            projected_name_cost +
            projected_transcription_cost +
            projected_question_cost +
            projected_grading_cost
        )

        return {
            "students": students_per_exam,
            "pages_per_student": pages_per_student,
            "total_pages": students_per_exam * pages_per_student,
            "projected_name_extraction": round(projected_name_cost, 4),
            "projected_transcription": round(projected_transcription_cost, 4),
            "projected_question_extraction": round(projected_question_cost, 4),
            "projected_grading": round(projected_grading_cost, 4),
            "total_projected_cost": round(total_projected, 4),
            "cost_per_student": round(total_projected / students_per_exam, 4),
            "currency": "USD"
        }

    def export_to_csv(self, output_file: Path) -> None:
        """
        Export all cost entries to CSV file.

        Args:
            output_file: Path to output CSV file
        """
        import csv

        session_files = list(self.data_dir.glob("session_*.json"))

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'timestamp', 'session_id', 'model', 'operation_type',
                'input_tokens', 'output_tokens', 'image_count',
                'input_cost', 'output_cost', 'image_cost', 'total_cost',
                'currency', 'error'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for file in session_files:
                session = self.load_session(file)
                if session:
                    for entry in session.entries:
                        row = {
                            'timestamp': entry.timestamp,
                            'session_id': entry.session_id,
                            'model': entry.model,
                            'operation_type': entry.operation_type,
                            'input_tokens': entry.input_tokens,
                            'output_tokens': entry.output_tokens,
                            'image_count': entry.image_count,
                            'input_cost': entry.input_cost,
                            'output_cost': entry.output_cost,
                            'image_cost': entry.image_cost,
                            'total_cost': entry.total_cost,
                            'currency': entry.currency,
                            'error': entry.error
                        }
                        writer.writerow(row)

        print(f"Cost data exported to: {output_file}")

    def print_analysis(self, analysis: CostAnalysis) -> None:
        """
        Print formatted analysis to console.

        Args:
            analysis: CostAnalysis object to display
        """
        print("\n" + "=" * 60)
        print("📊 LLM Cost Analysis Report")
        print("=" * 60)

        if analysis.time_period_start and analysis.time_period_end:
            print(f"📅 Period: {analysis.time_period_start} to {analysis.time_period_end}")

        print(f"\n📈 Summary Statistics:")
        print(f"  Total Sessions: {analysis.total_sessions}")
        print(f"  Total API Calls: {analysis.total_calls:,}")
        print(f"  Total Tokens: {analysis.total_input_tokens:,} in / {analysis.total_output_tokens:,} out")
        print(f"  Total Images: {analysis.total_images:,}")
        print(f"  Total Cost: ${analysis.total_cost:.6f} USD")
        print(f"  Avg Cost/Session: ${analysis.average_cost_per_session:.6f}")
        print(f"  Avg Cost/Call: ${analysis.average_cost_per_call:.6f}")

        if analysis.cost_by_operation:
            print(f"\n📋 Cost by Operation Type:")
            for op_type, cost in sorted(analysis.cost_by_operation.items(), key=lambda x: x[1], reverse=True):
                percentage = (cost / analysis.total_cost * 100) if analysis.total_cost > 0 else 0
                print(f"  • {op_type}: ${cost:.6f} ({percentage:.1f}%)")

        if analysis.cost_by_model:
            print(f"\n🤖 Cost by Model:")
            for model, cost in sorted(analysis.cost_by_model.items(), key=lambda x: x[1], reverse=True):
                percentage = (cost / analysis.total_cost * 100) if analysis.total_cost > 0 else 0
                print(f"  • {model}: ${cost:.6f} ({percentage:.1f}%)")

        if analysis.most_expensive_operation:
            op_type, cost = analysis.most_expensive_operation
            print(f"\n💸 Most Expensive Operation Type: {op_type} (${cost:.6f})")

        if analysis.most_expensive_session:
            print(f"\n💰 Most Expensive Session:")
            print(f"  ID: {analysis.most_expensive_session.session_id}")
            print(f"  Cost: ${analysis.most_expensive_session.total_cost:.6f}")
            print(f"  Calls: {analysis.most_expensive_session.total_calls}")

        print("=" * 60 + "\n")