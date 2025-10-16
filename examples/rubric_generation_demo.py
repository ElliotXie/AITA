#!/usr/bin/env python3
"""
Rubric Generation Pipeline Demo

This script demonstrates how to use the AITA rubric generation pipeline
to create answer keys and grading rubrics for exam questions.

Requirements:
- Run question extraction pipeline first to generate exam_spec.json
- Set up OpenRouter API key in environment
- Configure Google Cloud Storage (optional, for additional features)
"""

import sys
from pathlib import Path

# Add the AITA package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aita.pipelines.generate_rubric import (
    create_rubric_generation_pipeline,
    generate_rubrics_for_assignment,
    RubricGenerationError
)
from aita.domain.models import ExamSpec, Question, QuestionType

console = Console()


def demo_rubric_generation():
    """Demonstrate the rubric generation pipeline."""

    console.print(Panel.fit(
        "[bold cyan]AITA Rubric Generation Demo[/bold cyan]\n"
        "Generating answer keys and grading rubrics using AI",
        border_style="cyan"
    ))

    try:
        # Check if exam specification exists
        data_dir = Path("../data")
        exam_spec_file = data_dir / "results" / "exam_spec.json"

        if not exam_spec_file.exists():
            console.print("[yellow]⚠️  No exam specification found.[/yellow]")
            console.print("Creating a sample exam specification for demo purposes...")

            # Create sample exam spec for demo
            create_sample_exam_spec(data_dir)
            console.print("[green]✅ Sample exam specification created.[/green]\n")

        # Show what we're working with
        console.print("[bold]📋 Loading Exam Specification[/bold]")
        exam_spec = ExamSpec.load_from_file(exam_spec_file)

        console.print(f"Exam: {exam_spec.exam_name}")
        console.print(f"Questions: {len(exam_spec.questions)}")
        console.print(f"Total Points: {exam_spec.total_points}\n")

        # Show questions table
        questions_table = Table(title="Exam Questions")
        questions_table.add_column("Question ID", style="cyan")
        questions_table.add_column("Type", style="blue")
        questions_table.add_column("Points", justify="right", style="yellow")
        questions_table.add_column("Question Text", style="white")

        for question in exam_spec.questions:
            questions_table.add_row(
                question.question_id,
                question.question_type.value,
                f"{question.points:.1f}",
                question.question_text[:60] + "..." if len(question.question_text) > 60 else question.question_text
            )

        console.print(questions_table)
        console.print()

        # Generate rubrics and answer keys
        console.print("[bold]🤖 Generating Answer Keys and Rubrics[/bold]")
        console.print("This will use the LLM to generate step-by-step solutions and detailed rubrics...\n")

        # Create pipeline
        pipeline = create_rubric_generation_pipeline("demo_exam", data_dir)

        # Generate with user inputs if they exist
        user_inputs = {
            'instructions': '',
            'question_instructions': {},
            'rubrics': {}
        }

        # Check for example files
        rubrics_dir = Path("../intermediateproduct/rubrics")
        if (rubrics_dir / "rubric_instructions.txt.example").exists():
            console.print("[blue]📝 Found example instructions file[/blue]")
        if (rubrics_dir / "user_rubrics.json.example").exists():
            console.print("[blue]📊 Found example user rubrics file[/blue]")

        answer_keys, rubrics = pipeline.generate_from_exam_spec(
            exam_spec=exam_spec,
            assignment_name="demo_exam",
            user_rubrics_file=None,
            instructions_file=None,
            force_regenerate=True
        )

        # Display results
        console.print("\n[bold green]✅ Generation Complete![/bold green]\n")

        # Show answer keys
        console.print("[bold]🔑 Generated Answer Keys[/bold]")
        for answer_key in answer_keys:
            console.print(f"\n[cyan]Question {answer_key.question_id}:[/cyan]")
            console.print(f"Answer: {answer_key.correct_answer}")
            if answer_key.solution_steps:
                console.print("Steps:")
                for i, step in enumerate(answer_key.solution_steps, 1):
                    console.print(f"  {i}. {step}")
            if answer_key.grading_notes:
                console.print(f"Notes: {answer_key.grading_notes}")

        # Show rubrics
        console.print(f"\n[bold]📊 Generated Grading Rubrics[/bold]")
        for rubric in rubrics:
            console.print(f"\n[cyan]Question {rubric.question_id} ({rubric.total_points} points):[/cyan]")
            for criterion in rubric.criteria:
                console.print(f"  • {criterion.points} pts: {criterion.description}")
                if criterion.examples:
                    console.print(f"    Examples: {', '.join(criterion.examples[:2])}")

        # Show file locations
        console.print(f"\n[bold]💾 Files Saved To:[/bold]")
        console.print(f"  • Answer Keys: {data_dir}/results/answer_key.json")
        console.print(f"  • Rubrics: {data_dir}/results/rubric.json")
        console.print(f"  • Intermediate: {rubrics_dir}/generated_*.json")

        console.print(f"\n[green]🎉 Demo completed successfully![/green]")
        console.print(f"You can now use these rubrics for grading student responses.")

    except RubricGenerationError as e:
        console.print(f"\n[red]❌ Rubric Generation Error:[/red] {e}")
        console.print("Check your API configuration and try again.")
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected Error:[/red] {e}")
        console.print_exception()


def create_sample_exam_spec(data_dir: Path):
    """Create a sample exam specification for demo purposes."""

    # Ensure results directory exists
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create sample questions
    questions = [
        Question(
            question_id="1a",
            question_text="Calculate the derivative of f(x) = x² + 3x + 2",
            points=10.0,
            question_type=QuestionType.CALCULATION,
            page_number=1
        ),
        Question(
            question_id="1b",
            question_text="Explain the geometric interpretation of the derivative at a point",
            points=8.0,
            question_type=QuestionType.SHORT_ANSWER,
            page_number=1
        ),
        Question(
            question_id="2",
            question_text="Prove that the limit of (sin x)/x as x approaches 0 equals 1",
            points=15.0,
            question_type=QuestionType.LONG_ANSWER,
            page_number=2
        ),
        Question(
            question_id="3a",
            question_text="Find the critical points of g(x) = x³ - 6x² + 9x + 1",
            points=12.0,
            question_type=QuestionType.CALCULATION,
            page_number=2
        ),
        Question(
            question_id="3b",
            question_text="Classify the critical points found in part (a) as local maxima, minima, or saddle points",
            points=8.0,
            question_type=QuestionType.SHORT_ANSWER,
            page_number=2
        )
    ]

    # Create exam specification
    exam_spec = ExamSpec(
        exam_name="Calculus Midterm Exam - Demo",
        total_pages=2,
        questions=questions
    )

    # Save to file
    exam_spec.save_to_file(results_dir / "exam_spec.json")


def show_example_files():
    """Show the example configuration files."""

    console.print(Panel.fit(
        "[bold cyan]Example Configuration Files[/bold cyan]\n"
        "Templates for customizing rubric generation",
        border_style="cyan"
    ))

    rubrics_dir = Path("../intermediateproduct/rubrics")

    examples = [
        ("user_rubrics.json.example", "User-provided rubrics for specific questions"),
        ("rubric_instructions.txt.example", "General grading instructions and rules"),
        ("question_specific_instructions.json.example", "Per-question grading guidance")
    ]

    for filename, description in examples:
        file_path = rubrics_dir / filename
        if file_path.exists():
            console.print(f"\n[green]📄 {filename}[/green]")
            console.print(f"   {description}")
            console.print(f"   Location: {file_path}")
        else:
            console.print(f"\n[yellow]📄 {filename}[/yellow] (not found)")

    console.print(f"\n[bold]Usage:[/bold]")
    console.print("1. Copy .example files and remove the .example extension")
    console.print("2. Customize the content for your specific exam")
    console.print("3. Run: aita generate-rubric --user-rubrics user_rubrics.json --instructions rubric_instructions.txt")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        show_example_files()
    else:
        demo_rubric_generation()