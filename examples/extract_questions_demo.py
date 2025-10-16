"""
Question Extraction Demo

Demonstrates how to extract question structure from exam images.

This script shows the complete workflow:
1. Load exam images from a student folder
2. Upload images to Google Cloud Storage
3. Use LLM vision to extract question structure
4. Save the exam specification to JSON

Requirements:
- .env file with GCS and LLM credentials
- Student folder with exam images (e.g., data/grouped/Student_001/)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aita.pipelines.extract_questions import extract_questions_from_student
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    """Run question extraction demo."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]         AITA Question Extraction Demo                   [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

    # Configuration
    student_folder = "data/grouped/Student_001"
    assignment_name = "BMI541_Midterm"
    exam_name = "BMI541 Biostatistics Midterm Exam"

    console.print(f"📂 Student Folder: [cyan]{student_folder}[/cyan]")
    console.print(f"📝 Assignment: [cyan]{assignment_name}[/cyan]")
    console.print(f"📋 Exam Name: [cyan]{exam_name}[/cyan]\n")

    # Check if folder exists
    if not Path(student_folder).exists():
        console.print(f"[red]❌ Error: Student folder not found: {student_folder}[/red]")
        console.print("\n[yellow]💡 Tip: Make sure you have exam images in the data/grouped/ directory[/yellow]")
        return

    try:
        # Extract questions
        console.print("[bold]Starting question extraction...[/bold]\n")

        exam_spec = extract_questions_from_student(
            student_folder=student_folder,
            assignment_name=assignment_name,
            exam_name=exam_name
        )

        # Display results
        display_results(exam_spec)

    except Exception as e:
        console.print(f"\n[red]❌ Extraction failed: {e}[/red]")
        console.print("\n[yellow]💡 Troubleshooting:[/yellow]")
        console.print("   1. Check your .env file has valid credentials")
        console.print("   2. Verify GCS bucket is accessible")
        console.print("   3. Ensure LLM API key is valid")
        console.print("   4. Check that image files exist and are readable")
        raise


def display_results(exam_spec):
    """Display extraction results in a nice format."""
    console.print("\n[bold green]✅ Extraction Successful![/bold green]\n")

    # Summary table
    summary_table = Table(title="Exam Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Property", style="cyan")
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Exam Name", exam_spec.exam_name)
    summary_table.add_row("Total Pages", str(exam_spec.total_pages))
    summary_table.add_row("Total Questions", str(len(exam_spec.questions)))
    summary_table.add_row("Total Points", f"{exam_spec.total_points:.1f}")

    console.print(summary_table)

    # Questions table
    questions_table = Table(
        title="\nExtracted Questions",
        show_header=True,
        header_style="bold magenta"
    )
    questions_table.add_column("ID", style="cyan", width=8)
    questions_table.add_column("Question Text", style="white", width=60)
    questions_table.add_column("Points", style="green", width=8, justify="right")
    questions_table.add_column("Page", style="yellow", width=6, justify="center")
    questions_table.add_column("Type", style="blue", width=15)

    for q in exam_spec.questions:
        # Truncate long question text
        question_text = q.question_text[:57] + "..." if len(q.question_text) > 60 else q.question_text

        questions_table.add_row(
            q.question_id,
            question_text,
            f"{q.points:.1f}",
            str(q.page_number) if q.page_number else "-",
            q.question_type.value.replace("_", " ").title()
        )

    console.print(questions_table)

    # Point distribution
    console.print("\n[bold]📊 Point Distribution by Question Type:[/bold]")
    type_points = {}
    for q in exam_spec.questions:
        type_name = q.question_type.value.replace("_", " ").title()
        type_points[type_name] = type_points.get(type_name, 0) + q.points

    for type_name, points in sorted(type_points.items(), key=lambda x: x[1], reverse=True):
        percentage = (points / exam_spec.total_points) * 100
        console.print(f"   {type_name:20s}: {points:5.1f} pts ({percentage:5.1f}%)")

    # Page distribution
    console.print("\n[bold]📄 Questions per Page:[/bold]")
    page_questions = {}
    for q in exam_spec.questions:
        if q.page_number:
            page_questions[q.page_number] = page_questions.get(q.page_number, 0) + 1

    if page_questions:
        for page_num in sorted(page_questions.keys()):
            count = page_questions[page_num]
            console.print(f"   Page {page_num}: {count} question(s)")
    else:
        console.print("   [yellow]No page number information available[/yellow]")

    # Next steps
    console.print("\n[bold cyan]📝 Next Steps:[/bold cyan]")
    console.print("   1. Review the extracted questions for accuracy")
    console.print("   2. Edit data/results/exam_spec.json if needed")
    console.print("   3. Run rubric generation pipeline (Step 3)")
    console.print("   4. Process all student exams for transcription")
    console.print("   5. Run grading pipeline with rubrics\n")

    # File location
    console.print(f"[bold]💾 Exam spec saved to:[/bold] [cyan]data/results/exam_spec.json[/cyan]\n")


if __name__ == "__main__":
    main()
