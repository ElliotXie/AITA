"""
Test Script: Run Question Extraction Pipeline (Step 2)

This script runs the actual question extraction pipeline on real exam data.
It will:
1. Use one student's exam images as a template
2. Upload images to Google Cloud Storage
3. Extract question structure using LLM vision (Gemini 2.5 Flash)
4. Save the exam specification to data/results/exam_spec.json

Requirements:
- .env file with GCS and OpenRouter credentials
- Student folders in data/grouped/ with exam images
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aita.pipelines.extract_questions import extract_questions_from_student
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

console = Console()


def main():
    """Run question extraction on real exam data."""
    console.print("\n" + "="*70)
    console.print("[bold cyan]AITA Step 2: Question Extraction Pipeline Test[/bold cyan]")
    console.print("="*70 + "\n")

    # Configuration
    data_dir = Path("C:/Users/ellio/OneDrive - UW-Madison/AITA/data")
    student_folder = data_dir / "grouped" / "Mei, Elizabeth"
    assignment_name = "BMI541_Exam"
    exam_name = "BMI541 Biostatistics Exam"

    # Display configuration
    console.print(Panel.fit(
        f"[cyan]Student Folder:[/cyan] {student_folder}\n"
        f"[cyan]Assignment:[/cyan] {assignment_name}\n"
        f"[cyan]Exam Name:[/cyan] {exam_name}",
        title="Configuration",
        border_style="blue"
    ))

    # Check if folder exists
    if not student_folder.exists():
        console.print(f"\n[red]❌ Error: Student folder not found: {student_folder}[/red]")
        console.print("[yellow]Available students:[/yellow]")
        grouped_dir = data_dir / "grouped"
        if grouped_dir.exists():
            for folder in grouped_dir.iterdir():
                if folder.is_dir():
                    console.print(f"   - {folder.name}")
        return 1

    # Count images
    image_files = list(student_folder.glob("*.jpg")) + list(student_folder.glob("*.png"))
    console.print(f"\n[green]✓[/green] Found {len(image_files)} exam images")
    for img in sorted(image_files):
        console.print(f"   • {img.name}")

    # Confirm before proceeding
    console.print("\n[yellow]⚠️  This will:[/yellow]")
    console.print("   1. Upload images to Google Cloud Storage")
    console.print("   2. Call OpenRouter LLM API (Gemini 2.5 Flash)")
    console.print("   3. Extract question structure from exam")
    console.print("   4. Save results to data/results/exam_spec.json")
    console.print("\n[yellow]💰 Estimated cost: ~$0.01-0.02 USD[/yellow]\n")

    # Auto-confirm for automated testing
    console.print("[green]Auto-confirming for test run...[/green]\n")

    try:
        # Run extraction
        console.print("\n[bold]Starting extraction...[/bold]\n")

        exam_spec = extract_questions_from_student(
            student_folder=str(student_folder),
            assignment_name=assignment_name,
            exam_name=exam_name
        )

        # Display results
        display_results(exam_spec)

        # Save detailed output
        save_detailed_output(exam_spec, data_dir)

        console.print("\n[bold green]✅ SUCCESS! Question extraction completed.[/bold green]\n")
        return 0

    except Exception as e:
        console.print(f"\n[red]❌ Extraction failed: {e}[/red]")
        console.print("\n[yellow]💡 Troubleshooting:[/yellow]")
        console.print("   1. Check .env file has valid credentials:")
        console.print("      - OPENROUTER_API_KEY")
        console.print("      - GCS_PROJECT_ID, GCS_BUCKET_NAME, GCS_CREDENTIALS_PATH")
        console.print("   2. Verify GCS bucket is accessible")
        console.print("   3. Check that images are readable")
        console.print("   4. Review logs above for specific errors")

        # Print stack trace for debugging
        import traceback
        console.print("\n[red]Full error:[/red]")
        console.print(traceback.format_exc())
        return 1


def display_results(exam_spec):
    """Display extraction results in a nice format."""
    console.print("\n" + "="*70)
    console.print("[bold green]EXTRACTION RESULTS[/bold green]")
    console.print("="*70 + "\n")

    # Summary
    summary_table = Table(title="Exam Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Property", style="yellow", width=20)
    summary_table.add_column("Value", style="white", width=40)

    summary_table.add_row("Exam Name", exam_spec.exam_name)
    summary_table.add_row("Total Pages", str(exam_spec.total_pages))
    summary_table.add_row("Total Questions", str(len(exam_spec.questions)))
    summary_table.add_row("Total Points", f"{exam_spec.total_points:.1f}")

    console.print(summary_table)

    # Questions table
    console.print()
    questions_table = Table(
        title="Extracted Questions",
        show_header=True,
        header_style="bold cyan"
    )
    questions_table.add_column("Q ID", style="cyan", width=8)
    questions_table.add_column("Question Text", style="white", width=50)
    questions_table.add_column("Points", style="green", width=8, justify="right")
    questions_table.add_column("Page", style="yellow", width=6, justify="center")
    questions_table.add_column("Type", style="magenta", width=15)

    for q in exam_spec.questions:
        # Truncate long question text
        question_text = q.question_text[:47] + "..." if len(q.question_text) > 50 else q.question_text

        questions_table.add_row(
            q.question_id,
            question_text,
            f"{q.points:.1f}",
            str(q.page_number) if q.page_number else "-",
            q.question_type.value.replace("_", " ").title()
        )

    console.print(questions_table)

    # Statistics
    console.print("\n[bold cyan]📊 Statistics:[/bold cyan]")

    # Point distribution by question type
    type_points = {}
    type_counts = {}
    for q in exam_spec.questions:
        type_name = q.question_type.value.replace("_", " ").title()
        type_points[type_name] = type_points.get(type_name, 0) + q.points
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    console.print("\n[bold]Points by Question Type:[/bold]")
    for type_name in sorted(type_points.keys()):
        points = type_points[type_name]
        count = type_counts[type_name]
        percentage = (points / exam_spec.total_points) * 100 if exam_spec.total_points > 0 else 0
        console.print(f"   {type_name:20s}: {points:5.1f} pts ({count} questions, {percentage:5.1f}%)")

    # Questions per page
    console.print("\n[bold]Questions per Page:[/bold]")
    page_questions = {}
    page_points = {}
    for q in exam_spec.questions:
        if q.page_number:
            page_questions[q.page_number] = page_questions.get(q.page_number, 0) + 1
            page_points[q.page_number] = page_points.get(q.page_number, 0) + q.points

    if page_questions:
        for page_num in sorted(page_questions.keys()):
            count = page_questions[page_num]
            points = page_points[page_num]
            console.print(f"   Page {page_num}: {count} question(s), {points:.1f} pts")
    else:
        console.print("   [yellow]No page number information available[/yellow]")


def save_detailed_output(exam_spec, data_dir):
    """Save detailed output for review."""
    output_file = data_dir / "results" / "exam_spec_detailed.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create detailed output
    detailed_output = {
        "exam_metadata": {
            "exam_name": exam_spec.exam_name,
            "total_pages": exam_spec.total_pages,
            "total_points": exam_spec.total_points,
            "total_questions": len(exam_spec.questions)
        },
        "questions": []
    }

    for q in exam_spec.questions:
        detailed_output["questions"].append({
            "question_id": q.question_id,
            "question_text": q.question_text,
            "points": q.points,
            "page_number": q.page_number,
            "question_type": q.question_type.value,
            "image_bounds": q.image_bounds
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_output, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]💾 Detailed output saved to:[/green] {output_file}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
