"""
AITA CLI - Automatic Image Test Analysis

Command-line interface for the AITA exam grading system.
"""

import typer
from pathlib import Path
from typing import Optional
import sys
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich import print as rprint

from .pipelines.transcribe import (
    transcribe_all_students,
    transcribe_single_student,
    TranscriptionError
)
from .pipelines.generate_rubric import (
    generate_rubrics_for_assignment,
    RubricGenerationError
)
from .pipelines.grade import (
    grade_all_students,
    grade_single_student,
    GradingError
)
from .services.llm.cost_tracker import (
    init_global_tracker,
    get_global_tracker
)

# Initialize Typer app
app = typer.Typer(
    name="aita",
    help="Automatic Image Test Analysis - AI-powered exam grading system",
    add_completion=False,
    rich_markup_mode="rich"
)

# Initialize console for rich output
console = Console()


def _get_cost_tracking_dir(data_dir: Optional[Path] = None) -> Path:
    """
    Get the cost tracking directory path.

    Args:
        data_dir: Optional data directory. If not provided, uses current directory.

    Returns:
        Path to cost tracking directory
    """
    if data_dir is None:
        data_dir = Path.cwd() / "data"

    return data_dir.parent / "intermediateproduct" / "cost_tracking"


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure rich logging
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.command()
def transcribe(
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory containing grouped student folders",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    assignment_name: str = typer.Option(
        "exam_transcription",
        "--assignment",
        "-a",
        help="Assignment name for GCS organization"
    ),
    student: Optional[str] = typer.Option(
        None,
        "--student",
        "-s",
        help="Transcribe only a specific student (folder name)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Custom output directory for transcription results"
    ),
    no_question_mapping: bool = typer.Option(
        False,
        "--no-question-mapping",
        help="Disable question-based mapping of transcriptions"
    ),
    no_html: bool = typer.Option(
        False,
        "--no-html",
        help="Disable HTML report generation"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be transcribed without actually doing it"
    )
):
    """
    Transcribe handwritten exam answers to text using LLM vision.

    Processes grouped student folders and converts all exam images to text
    using advanced AI vision models. Results are saved in organized JSON format
    with question-mapped transcriptions and HTML reports.

    Examples:

        # Transcribe all students with all features
        aita transcribe

        # Transcribe specific assignment
        aita transcribe --assignment "BMI541_Midterm"

        # Transcribe single student
        aita transcribe --student "Smith, John"

        # Disable HTML report generation
        aita transcribe --no-html

        # Only generate page-based transcriptions (no question mapping)
        aita transcribe --no-question-mapping

        # Use custom data directory
        aita transcribe --data-dir /path/to/exam/data
    """
    setup_logging(verbose)

    try:
        # Determine data directory first
        if data_dir is None:
            data_dir = Path.cwd() / "data"

        # Initialize cost tracking
        cost_tracker = init_global_tracker(
            data_dir=_get_cost_tracking_dir(data_dir)
        )

        # Display header
        console.print("\n[bold cyan]AITA Transcription Pipeline[/bold cyan]")
        console.print("Converting handwritten exam answers to text using AI vision\n")

        grouped_dir = data_dir / "grouped"

        # Validate directories
        if not data_dir.exists():
            console.print(f"[red]Error:[/red] Data directory not found: {data_dir}")
            raise typer.Exit(1)

        if not grouped_dir.exists():
            console.print(f"[red]Error:[/red] Grouped directory not found: {grouped_dir}")
            console.print("Please run the smart grouping pipeline first.")
            raise typer.Exit(1)

        # Show configuration
        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Data Directory: {data_dir}")
        console.print(f"  Assignment: {assignment_name}")
        console.print(f"  Target: {'Single student: ' + student if student else 'All students'}")
        console.print(f"  Question Mapping: {not no_question_mapping}")
        console.print(f"  HTML Reports: {not no_html}")
        console.print(f"  Verbose: {verbose}")
        console.print(f"  Dry Run: {dry_run}\n")

        if dry_run:
            # Show what would be processed
            _show_dry_run_info(grouped_dir, student)
            return

        # Confirm before processing
        if not typer.confirm(
            "This will process exam images and may incur LLM API costs. Continue?",
            default=True
        ):
            console.print("[yellow]Transcription cancelled.[/yellow]")
            raise typer.Exit(0)

        # Process transcription
        if student:
            # Single student transcription
            result = transcribe_single_student(
                student_folder=student,
                assignment_name=assignment_name,
                data_dir=data_dir,
                use_question_mapping=not no_question_mapping,
                generate_html_reports=not no_html
            )
            _display_single_student_results(result)
        else:
            # All students transcription
            results = transcribe_all_students(
                assignment_name=assignment_name,
                data_dir=data_dir,
                use_question_mapping=not no_question_mapping,
                generate_html_reports=not no_html
            )
            _display_batch_results(results)

        console.print("\n[bold green]✅ Transcription completed successfully![/bold green]")

        # Display cost summary
        if cost_tracker:
            cost_tracker.save()  # Ensure final save
            cost_tracker.print_summary(detailed=False)

    except TranscriptionError as e:
        console.print(f"\n[red]Transcription Error:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Transcription interrupted by user.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def status(
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory to check",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    )
):
    """
    Show status of AITA pipelines and data.

    Displays information about:
    - Available student folders
    - Transcription results
    - Processing statistics
    """
    if data_dir is None:
        data_dir = Path.cwd() / "data"

    console.print("\n[bold cyan]AITA Pipeline Status[/bold cyan]\n")

    # Check directories
    grouped_dir = data_dir / "grouped"
    transcription_dir = data_dir / "intermediateproduct" / "transcription_results"

    # Show directory status
    _show_directory_status(data_dir, grouped_dir, transcription_dir)

    # Show student folders
    if grouped_dir.exists():
        _show_student_folders(grouped_dir)

    # Show transcription results
    if transcription_dir.exists():
        _show_transcription_status(transcription_dir)


@app.command()
def generate_rubric(
    assignment_name: str = typer.Option(
        "exam1",
        "--assignment",
        "-a",
        help="Assignment name"
    ),
    user_rubrics: Optional[Path] = typer.Option(
        None,
        "--user-rubrics",
        "-r",
        help="Path to user-provided rubrics JSON file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    instructions: Optional[Path] = typer.Option(
        None,
        "--instructions",
        "-i",
        help="Path to grading instructions text file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory containing exam specification",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force regeneration even if rubrics already exist"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be generated without actually doing it"
    )
):
    """
    Generate answer keys and grading rubrics for exam questions.

    Creates detailed step-by-step answer keys and comprehensive grading rubrics
    using LLM analysis. Supports user-provided rubrics and custom grading instructions.

    Examples:

        # Generate rubrics for default assignment
        aita generate-rubric

        # Generate with custom instructions
        aita generate-rubric --instructions grading_rules.txt

        # Use user-provided rubrics for some questions
        aita generate-rubric --user-rubrics my_rubrics.json

        # Force regeneration
        aita generate-rubric --force --assignment "BMI541_Midterm"
    """
    setup_logging(verbose)

    try:
        # Determine data directory first
        if data_dir is None:
            data_dir = Path.cwd() / "data"

        # Initialize cost tracking
        cost_tracker = init_global_tracker(
            data_dir=_get_cost_tracking_dir(data_dir)
        )

        # Display header
        console.print("\n[bold cyan]AITA Rubric Generation Pipeline[/bold cyan]")
        console.print("Generating answer keys and grading rubrics using AI analysis\n")

        # Validate data directory and exam spec
        if not data_dir.exists():
            console.print(f"[red]Error:[/red] Data directory not found: {data_dir}")
            raise typer.Exit(1)

        exam_spec_file = data_dir / "results" / "exam_spec.json"
        if not exam_spec_file.exists():
            console.print(f"[red]Error:[/red] Exam specification not found: {exam_spec_file}")
            console.print("Please run the question extraction pipeline first.")
            raise typer.Exit(1)

        # Show configuration
        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Data Directory: {data_dir}")
        console.print(f"  Assignment: {assignment_name}")
        console.print(f"  User Rubrics: {user_rubrics or 'None'}")
        console.print(f"  Instructions: {instructions or 'None'}")
        console.print(f"  Force Regenerate: {force}")
        console.print(f"  Verbose: {verbose}")
        console.print(f"  Dry Run: {dry_run}\n")

        if dry_run:
            # Show what would be processed
            _show_rubric_dry_run_info(data_dir, user_rubrics, instructions)
            return

        # Confirm before processing
        if not typer.confirm(
            "This will generate rubrics and may incur LLM API costs. Continue?",
            default=True
        ):
            console.print("[yellow]Rubric generation cancelled.[/yellow]")
            raise typer.Exit(0)

        # Generate rubrics
        answer_keys, rubrics = generate_rubrics_for_assignment(
            assignment_name=assignment_name,
            user_rubrics_file=str(user_rubrics) if user_rubrics else None,
            instructions_file=str(instructions) if instructions else None,
            force_regenerate=force,
            data_dir=data_dir
        )

        # Display results
        _display_rubric_results(answer_keys, rubrics)

        console.print("\n[bold green]✅ Rubric generation completed successfully![/bold green]")

        # Display cost summary
        if cost_tracker:
            cost_tracker.save()  # Ensure final save
            cost_tracker.print_summary(detailed=False)

    except RubricGenerationError as e:
        console.print(f"\n[red]Rubric Generation Error:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Rubric generation interrupted by user.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def grade(
    assignment_name: str = typer.Option(
        "exam_grading",
        "--assignment",
        "-a",
        help="Assignment name for organization"
    ),
    student: Optional[str] = typer.Option(
        None,
        "--student",
        "-s",
        help="Grade only a specific student (folder name)"
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory containing exam data",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be graded without actually doing it"
    )
):
    """
    Grade student exam responses using LLM and generated rubrics.

    Uses previously generated rubrics and answer keys to grade transcribed
    student responses. Produces detailed grading results with feedback,
    scoring, and comprehensive statistics.

    Prerequisites:
    - Run 'aita generate-rubric' first to create rubrics and answer keys
    - Run 'aita transcribe' to create student transcriptions

    Examples:

        # Grade all students
        aita grade

        # Grade specific assignment
        aita grade --assignment "BMI541_Midterm"

        # Grade single student
        aita grade --student "Turner, Evan"

        # Use custom data directory
        aita grade --data-dir /path/to/exam/data

        # Show what would be graded (dry run)
        aita grade --dry-run
    """
    setup_logging(verbose)

    try:
        # Determine data directory first
        if data_dir is None:
            data_dir = Path.cwd() / "data"

        # Initialize cost tracking
        cost_tracker = init_global_tracker(
            data_dir=_get_cost_tracking_dir(data_dir)
        )

        # Display header
        console.print("\n[bold cyan]AITA Grading Pipeline[/bold cyan]")
        console.print("Grading student exam responses using AI-powered evaluation\n")

        # Validate prerequisites
        if not data_dir.exists():
            console.print(f"[red]Error:[/red] Data directory not found: {data_dir}")
            raise typer.Exit(1)

        # Check for rubrics and answer keys
        intermediate_dir = data_dir.parent / "intermediateproduct"
        rubrics_dir = intermediate_dir / "rubrics"
        rubrics_file = rubrics_dir / "generated_rubrics.json"
        answer_keys_file = rubrics_dir / "generated_answer_keys.json"

        if not rubrics_file.exists():
            console.print(f"[red]Error:[/red] Rubrics not found: {rubrics_file}")
            console.print("Please run 'aita generate-rubric' first.")
            raise typer.Exit(1)

        if not answer_keys_file.exists():
            console.print(f"[red]Error:[/red] Answer keys not found: {answer_keys_file}")
            console.print("Please run 'aita generate-rubric' first.")
            raise typer.Exit(1)

        # Check for transcription results
        transcription_dir = intermediate_dir / "transcription_results"
        if not transcription_dir.exists():
            console.print(f"[red]Error:[/red] Transcription results not found: {transcription_dir}")
            console.print("Please run 'aita transcribe' first.")
            raise typer.Exit(1)

        # Show configuration
        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Data Directory: {data_dir}")
        console.print(f"  Assignment: {assignment_name}")
        console.print(f"  Target: {'Single student: ' + student if student else 'All students'}")
        console.print(f"  Verbose: {verbose}")
        console.print(f"  Dry Run: {dry_run}\n")

        if dry_run:
            # Show what would be processed
            _show_grading_dry_run_info(transcription_dir, student, rubrics_file, answer_keys_file)
            return

        # Confirm before processing
        if not typer.confirm(
            "This will grade student responses and may incur LLM API costs. Continue?",
            default=True
        ):
            console.print("[yellow]Grading cancelled.[/yellow]")
            raise typer.Exit(0)

        # Process grading
        if student:
            # Single student grading
            result = grade_single_student(
                student_name=student,
                assignment_name=assignment_name,
                data_dir=data_dir
            )
            _display_single_student_grading_results(result)
        else:
            # All students grading
            results = grade_all_students(
                assignment_name=assignment_name,
                data_dir=data_dir
            )
            _display_batch_grading_results(results)

        console.print("\n[bold green]✅ Grading completed successfully![/bold green]")

        # Display cost summary
        if cost_tracker:
            cost_tracker.save()  # Ensure final save
            cost_tracker.print_summary(detailed=False)

    except GradingError as e:
        console.print(f"\n[red]Grading Error:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Grading interrupted by user.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def costs(
    session_id: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Specific session ID to analyze"
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed cost breakdown per API call"
    ),
    recent: int = typer.Option(
        5,
        "--recent",
        "-r",
        help="Show N most recent sessions"
    )
):
    """
    Display LLM API cost tracking information.

    Shows cost summaries for LLM API usage across different operations
    like name extraction, transcription, question extraction, and grading.

    Examples:

        # Show most recent session costs
        aita costs

        # Show detailed breakdown
        aita costs --detailed

        # Show last 10 sessions
        aita costs --recent 10

        # Show specific session
        aita costs --session session_20250115_143022_abc123
    """
    # Get cost tracking directory using relative paths
    cost_dir = _get_cost_tracking_dir()

    if not cost_dir.exists():
        console.print("[yellow]No cost tracking data found.[/yellow]")
        console.print(f"Cost tracking directory does not exist: {cost_dir}")
        return

    if session_id:
        # Load and display specific session
        session_file = cost_dir / f"{session_id}.json"
        if not session_file.exists():
            console.print(f"[red]Session not found: {session_id}[/red]")
            return

        from .services.llm.cost_tracker import CostTracker
        tracker = CostTracker(data_dir=cost_dir, session_id=session_id)
        tracker.load(session_file)
        tracker.print_summary(detailed=detailed)

    else:
        # Show recent sessions
        import json
        from datetime import datetime

        session_files = sorted(cost_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

        if not session_files:
            console.print("[yellow]No cost tracking sessions found.[/yellow]")
            return

        # Create summary table
        table = Table(title=f"Recent {min(recent, len(session_files))} Cost Tracking Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Date/Time", style="green")
        table.add_column("Total Calls", justify="right")
        table.add_column("Total Tokens", justify="right")
        table.add_column("Total Cost", justify="right", style="yellow")
        table.add_column("Primary Operation")

        total_cost_all = 0.0

        for session_file in session_files[:recent]:
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)

                session_id = data.get("session_id", "unknown")
                start_time = data.get("start_time", "")
                if start_time:
                    dt = datetime.fromisoformat(start_time)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_time = "Unknown"

                total_calls = data.get("total_calls", 0)
                total_tokens = data.get("total_input_tokens", 0) + data.get("total_output_tokens", 0)
                total_cost = data.get("total_cost", 0.0)
                total_cost_all += total_cost

                # Find primary operation
                cost_by_op = data.get("cost_by_operation", {})
                if cost_by_op:
                    primary_op = max(cost_by_op.items(), key=lambda x: x[1])[0]
                else:
                    primary_op = "general"

                table.add_row(
                    session_id[:40] + "..." if len(session_id) > 40 else session_id,
                    formatted_time,
                    str(total_calls),
                    f"{total_tokens:,}",
                    f"${total_cost:.6f}",
                    primary_op
                )

            except Exception as e:
                console.print(f"[yellow]Error reading session file {session_file.name}: {e}[/yellow]")

        console.print(table)
        console.print(f"\n[bold]Total Cost Across All Sessions: ${total_cost_all:.6f} USD[/bold]")

        # Show tip for detailed view
        if session_files:
            console.print(f"\n💡 Tip: Use 'aita costs --session {session_files[0].stem}' to see detailed breakdown")


@app.command(name="report-transcription")
def report_transcription(
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory containing transcription results",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    student: Optional[str] = typer.Option(
        None,
        "--student",
        "-s",
        help="Generate report for specific student only"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    Generate HTML transcription reports from existing transcription data.

    This command regenerates HTML reports without re-running transcription.
    Useful if you want to update report templates or regenerate reports after
    transcription is complete.

    Examples:

        # Generate reports for all students
        aita report-transcription

        # Generate report for specific student
        aita report-transcription --student "Smith, John"

        # Use custom data directory
        aita report-transcription --data-dir /path/to/exam/data
    """
    setup_logging(verbose)

    try:
        from .report.transcription_report import generate_transcription_reports

        # Determine data directory
        if data_dir is None:
            data_dir = Path.cwd() / "data"

        # Display header
        console.print("\n[bold cyan]AITA Transcription Report Generator[/bold cyan]")
        console.print("Generating HTML reports from transcription data\n")

        # Validate data directory
        if not data_dir.exists():
            console.print(f"[red]Error:[/red] Data directory not found: {data_dir}")
            raise typer.Exit(1)

        transcription_dir = data_dir / "intermediateproduct" / "transcription_results"
        if not transcription_dir.exists():
            console.print(f"[red]Error:[/red] Transcription results not found: {transcription_dir}")
            console.print("Please run the transcription pipeline first.")
            raise typer.Exit(1)

        # Show configuration
        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Data Directory: {data_dir}")
        console.print(f"  Target: {'Single student: ' + student if student else 'All students'}")
        console.print(f"  Verbose: {verbose}\n")

        # Generate reports
        student_names = [student] if student else None

        with console.status("[bold green]Generating reports..."):
            report_paths = generate_transcription_reports(
                data_dir=data_dir,
                student_names=student_names
            )

        # Display results
        console.print(f"\n[bold green]✅ Generated {len(report_paths)} HTML reports[/bold green]\n")

        if verbose and report_paths:
            console.print("[bold]Generated Reports:[/bold]")
            for report_path in report_paths[:10]:  # Show first 10
                console.print(f"  - {report_path}")
            if len(report_paths) > 10:
                console.print(f"  ... and {len(report_paths) - 10} more")

    except Exception as e:
        console.print(f"\n[red]Report generation error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def version():
    """Show AITA version information."""
    console.print("\n[bold cyan]AITA - Automatic Image Test Analysis[/bold cyan]")
    console.print("Version: 0.1.0")
    console.print("AI-powered exam grading system\n")

    # Show component status
    console.print("[bold]Available Components:[/bold]")
    console.print("  ✅ Smart Grouping Pipeline")
    console.print("  ✅ Question Extraction Pipeline")
    console.print("  ✅ Transcription Pipeline (with HTML Reports)")
    console.print("  ✅ Rubric Generation Pipeline")
    console.print("  ✅ Grading Pipeline")
    console.print("  ⏳ Grade Report Generation\n")


def _show_dry_run_info(grouped_dir: Path, student: Optional[str]):
    """Show what would be processed in dry run mode."""
    console.print("[bold yellow]DRY RUN - No actual processing will occur[/bold yellow]\n")

    if student:
        student_dir = grouped_dir / student
        if student_dir.exists():
            images = list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.png"))
            console.print(f"Would transcribe {len(images)} images for student: {student}")
        else:
            console.print(f"[red]Student folder not found: {student}[/red]")
    else:
        student_folders = [d for d in grouped_dir.iterdir() if d.is_dir()]
        total_images = 0

        console.print(f"Would process {len(student_folders)} students:")
        for folder in student_folders:
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            total_images += len(images)
            console.print(f"  - {folder.name}: {len(images)} images")

        console.print(f"\nTotal images to process: {total_images}")


def _display_single_student_results(result):
    """Display results for single student transcription."""
    console.print(f"\n[bold]Results for {result.student_name}:[/bold]")
    console.print(f"  Pages Processed: {result.total_pages}")
    console.print(f"  Successful: {result.successful_pages}")
    console.print(f"  Failed: {result.failed_pages}")
    console.print(f"  Success Rate: {result.success_rate:.1f}%")
    console.print(f"  Average Confidence: {result.average_confidence:.2f}")
    console.print(f"  Processing Time: {result.processing_time:.1f} seconds")

    if result.errors:
        console.print(f"\n[yellow]Errors ({len(result.errors)}):[/yellow]")
        for error in result.errors:
            console.print(f"  - {error}")


def _display_batch_results(results):
    """Display results for batch transcription."""
    console.print(f"\n[bold]Transcription Results Summary:[/bold]")
    console.print(f"  Students Processed: {results.total_students}")
    console.print(f"  Students Successful: {results.successful_students}")
    console.print(f"  Total Pages: {results.total_pages}")
    console.print(f"  Successful Pages: {results.successful_pages}")
    console.print(f"  Overall Success Rate: {results.success_rate:.1f}%")
    console.print(f"  Average Confidence: {results.average_confidence:.2f}")
    console.print(f"  Total Processing Time: {results.processing_time:.1f} seconds")

    # Show per-student breakdown
    if results.student_results:
        table = Table(title="Per-Student Results")
        table.add_column("Student", style="cyan")
        table.add_column("Pages", justify="right")
        table.add_column("Success Rate", justify="right", style="green")
        table.add_column("Avg Confidence", justify="right", style="blue")

        for student_result in results.student_results:
            table.add_row(
                student_result.student_name,
                f"{student_result.successful_pages}/{student_result.total_pages}",
                f"{student_result.success_rate:.1f}%",
                f"{student_result.average_confidence:.2f}"
            )

        console.print(table)


def _show_directory_status(data_dir: Path, grouped_dir: Path, transcription_dir: Path):
    """Show status of key directories."""
    table = Table(title="Directory Status")
    table.add_column("Directory", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Path")

    directories = [
        ("Data Directory", data_dir),
        ("Grouped Students", grouped_dir),
        ("Transcription Results", transcription_dir)
    ]

    for name, path in directories:
        status = "✅ Exists" if path.exists() else "❌ Missing"
        table.add_row(name, status, str(path))

    console.print(table)


def _show_student_folders(grouped_dir: Path):
    """Show available student folders."""
    student_folders = [d for d in grouped_dir.iterdir() if d.is_dir()]

    if not student_folders:
        console.print("\n[yellow]No student folders found in grouped directory.[/yellow]")
        return

    console.print(f"\n[bold]Student Folders ({len(student_folders)}):[/bold]")

    table = Table()
    table.add_column("Student", style="cyan")
    table.add_column("Images", justify="right", style="green")
    table.add_column("Last Modified", style="dim")

    for folder in sorted(student_folders, key=lambda x: x.name):
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        from datetime import datetime
        modified = datetime.fromtimestamp(folder.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        table.add_row(folder.name, str(len(images)), modified)

    console.print(table)


def _show_transcription_status(transcription_dir: Path):
    """Show transcription results status."""
    summary_file = transcription_dir / "summary_report.json"

    if summary_file.exists():
        import json
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            console.print(f"\n[bold]Last Transcription Run:[/bold]")
            console.print(f"  Assignment: {summary.get('assignment_name', 'Unknown')}")
            console.print(f"  Date: {summary.get('timestamp', 'Unknown')}")
            console.print(f"  Students: {summary.get('successful_students', 0)}/{summary.get('total_students', 0)}")
            console.print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")

        except Exception as e:
            console.print(f"\n[yellow]Could not read transcription summary: {e}[/yellow]")
    else:
        console.print(f"\n[yellow]No transcription results found.[/yellow]")


def _show_rubric_dry_run_info(data_dir: Path, user_rubrics: Optional[Path], instructions: Optional[Path]):
    """Show what would be processed in rubric generation dry run mode."""
    console.print("[bold yellow]DRY RUN - No actual processing will occur[/bold yellow]\n")

    # Load exam spec to show what would be processed
    try:
        from .domain.models import ExamSpec
        exam_spec_file = data_dir / "results" / "exam_spec.json"
        exam_spec = ExamSpec.load_from_file(exam_spec_file)

        console.print(f"Would generate rubrics for: {exam_spec.exam_name}")
        console.print(f"Questions to process: {len(exam_spec.questions)}")
        console.print(f"Total points: {exam_spec.total_points}")

        # Show question breakdown
        console.print(f"\nQuestions:")
        for question in exam_spec.questions:
            console.print(f"  - {question.question_id}: {question.points} points ({question.question_type.value})")

        # Show user input status
        console.print(f"\nUser Inputs:")
        console.print(f"  Custom Rubrics: {'Yes' if user_rubrics else 'No'}")
        console.print(f"  Instructions File: {'Yes' if instructions else 'No'}")

        # Check for default files using relative paths
        rubrics_dir = data_dir.parent / "intermediateproduct" / "rubrics"
        default_rubrics = rubrics_dir / "user_rubrics.json"
        default_instructions = rubrics_dir / "rubric_instructions.txt"

        if default_rubrics.exists():
            console.print(f"  Default Rubrics Found: {default_rubrics}")
        if default_instructions.exists():
            console.print(f"  Default Instructions Found: {default_instructions}")

    except Exception as e:
        console.print(f"[red]Error loading exam specification: {e}[/red]")


def _display_rubric_results(answer_keys, rubrics):
    """Display rubric generation results."""
    console.print(f"\n[bold]Rubric Generation Results:[/bold]")

    # Summary table
    table = Table(title="Generated Content Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Total Points", justify="right", style="yellow")

    answer_key_count = len(answer_keys)
    rubric_count = len(rubrics)
    total_points = sum(r.total_points for r in rubrics)

    table.add_row("Answer Keys", str(answer_key_count), "-")
    table.add_row("Rubrics", str(rubric_count), f"{total_points:.1f}")

    console.print(table)

    # Detail breakdown
    if rubrics:
        console.print(f"\n[bold]Rubric Details:[/bold]")
        detail_table = Table()
        detail_table.add_column("Question", style="cyan")
        detail_table.add_column("Points", justify="right", style="yellow")
        detail_table.add_column("Criteria Count", justify="right", style="green")
        detail_table.add_column("Answer Key", style="blue")

        answer_key_dict = {ak.question_id: ak for ak in answer_keys}

        for rubric in rubrics:
            answer_key = answer_key_dict.get(rubric.question_id)
            has_answer = "✅" if answer_key else "❌"

            detail_table.add_row(
                rubric.question_id,
                f"{rubric.total_points:.1f}",
                str(len(rubric.criteria)),
                has_answer
            )

        console.print(detail_table)


def _show_grading_dry_run_info(
    transcription_dir: Path,
    student: Optional[str],
    rubrics_file: Path,
    answer_keys_file: Path
):
    """Show what would be processed in grading dry run mode."""
    console.print("[bold yellow]DRY RUN - No actual grading will occur[/bold yellow]\n")

    # Load grading data to show what would be processed
    try:
        import json

        # Show rubrics and answer keys info
        with open(rubrics_file, 'r') as f:
            rubrics_data = json.load(f)
        with open(answer_keys_file, 'r') as f:
            answer_keys_data = json.load(f)

        console.print("[bold]Available Grading Data:[/bold]")
        console.print(f"  Rubrics: {len(rubrics_data.get('rubrics', []))}")
        console.print(f"  Answer Keys: {len(answer_keys_data.get('answer_keys', []))}")

        # Show rubric breakdown
        console.print(f"\n[bold]Questions to Grade:[/bold]")
        for rubric in rubrics_data.get('rubrics', []):
            console.print(f"  - {rubric['question_id']}: {rubric['total_points']} points")

        # Show student transcription data
        if student:
            student_dir = transcription_dir / student
            if student_dir.exists():
                transcription_file = student_dir / "question_transcriptions.json"
                if transcription_file.exists():
                    with open(transcription_file, 'r') as f:
                        trans_data = json.load(f)
                    console.print(f"\n[bold]Student Data ({student}):[/bold]")
                    console.print(f"  Transcribed Questions: {len(trans_data.get('question_answers', []))}")
                else:
                    console.print(f"\n[red]No transcription data found for {student}[/red]")
            else:
                console.print(f"\n[red]Student folder not found: {student}[/red]")
        else:
            student_dirs = [d for d in transcription_dir.iterdir() if d.is_dir()]
            console.print(f"\n[bold]Students to Grade:[/bold]")
            console.print(f"  Total Students: {len(student_dirs)}")

            valid_students = 0
            for student_dir in student_dirs:
                transcription_file = student_dir / "question_transcriptions.json"
                if transcription_file.exists():
                    valid_students += 1
                    if valid_students <= 5:  # Show first 5
                        console.print(f"  - {student_dir.name}")

            if len(student_dirs) > 5:
                console.print(f"  ... and {len(student_dirs) - 5} more")

            console.print(f"  Students with valid data: {valid_students}/{len(student_dirs)}")

    except Exception as e:
        console.print(f"[red]Error loading grading data: {e}[/red]")


def _display_single_student_grading_results(result):
    """Display results for single student grading."""
    console.print(f"\n[bold]Grading Results for {result.student_name}:[/bold]")
    console.print(f"  Total Score: {result.total_score:.1f}/{result.total_possible:.1f}")
    console.print(f"  Percentage: {result.percentage:.1f}%")
    console.print(f"  Letter Grade: {result.grade_letter}")
    console.print(f"  Questions Graded: {len(result.question_grades)}")
    console.print(f"  Processing Time: {result.processing_time:.1f} seconds")

    # Show question breakdown
    if result.question_grades:
        console.print(f"\n[bold]Question Breakdown:[/bold]")
        table = Table()
        table.add_column("Question", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="blue")
        table.add_column("Feedback", style="dim")

        for grade in result.question_grades:
            feedback_preview = grade.feedback[:50] + "..." if len(grade.feedback) > 50 else grade.feedback
            table.add_row(
                grade.question_id,
                f"{grade.points_earned:.1f}/{grade.points_possible:.1f}",
                f"{grade.percentage:.1f}%",
                feedback_preview
            )

        console.print(table)

    if result.errors:
        console.print(f"\n[yellow]Errors ({len(result.errors)}):[/yellow]")
        for error in result.errors:
            console.print(f"  - {error}")


def _display_batch_grading_results(results):
    """Display results for batch grading."""
    console.print(f"\n[bold]Batch Grading Results Summary:[/bold]")
    console.print(f"  Students Processed: {results.total_students}")
    console.print(f"  Students Successful: {results.successful_students}")
    console.print(f"  Success Rate: {results.success_rate:.1f}%")
    console.print(f"  Total Questions: {results.total_questions}")
    console.print(f"  Average Score: {results.average_score:.1f}%")
    console.print(f"  Total Processing Time: {results.processing_time:.1f} seconds")

    # Show per-student breakdown
    if results.student_results:
        console.print(f"\n[bold]Per-Student Results:[/bold]")
        table = Table(title="Student Grades")
        table.add_column("Student", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="blue")
        table.add_column("Letter Grade", justify="center", style="bold")
        table.add_column("Questions", justify="right")

        for student_result in results.student_results:
            table.add_row(
                student_result.student_name,
                f"{student_result.total_score:.1f}/{student_result.total_possible:.1f}",
                f"{student_result.percentage:.1f}%",
                student_result.grade_letter,
                str(len(student_result.question_grades))
            )

        console.print(table)

        # Show grade distribution
        grade_letters = [r.grade_letter for r in results.student_results if r.percentage > 0]
        if grade_letters:
            console.print(f"\n[bold]Grade Distribution:[/bold]")
            from collections import Counter
            grade_counts = Counter(grade_letters)

            dist_table = Table()
            dist_table.add_column("Letter Grade", style="bold")
            dist_table.add_column("Count", justify="right", style="green")
            dist_table.add_column("Percentage", justify="right", style="blue")

            total_graded = len(grade_letters)
            for grade in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F"]:
                count = grade_counts.get(grade, 0)
                if count > 0:
                    percentage = (count / total_graded) * 100
                    dist_table.add_row(grade, str(count), f"{percentage:.1f}%")

            console.print(dist_table)

    if results.errors:
        console.print(f"\n[yellow]Overall Errors ({len(results.errors)}):[/yellow]")
        for error in results.errors[:5]:  # Show first 5 errors
            console.print(f"  - {error}")
        if len(results.errors) > 5:
            console.print(f"  ... and {len(results.errors) - 5} more errors")


# Global error handler
def _handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler for better error display."""
    if issubclass(exc_type, KeyboardInterrupt):
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        return

    console.print(f"\n[red]Unexpected error:[/red] {exc_value}")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


# Set global exception handler
sys.excepthook = _handle_exception


if __name__ == "__main__":
    app()