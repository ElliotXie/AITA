"""
Real-world test of question extraction with actual exam images.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from aita.pipelines.extract_questions import extract_questions_from_student
from rich.console import Console

console = Console()

def main():
    console.print("\n[bold cyan]Testing Question Extraction with Real Images[/bold cyan]\n")

    # Use one of the actual student folders as sample
    student_folder = "data/grouped/Mei, Elizabeth"
    assignment_name = "BMI541_Exam"

    console.print(f"📂 Using: {student_folder}")
    console.print(f"📝 Assignment: {assignment_name}\n")

    try:
        # Run extraction
        exam_spec = extract_questions_from_student(
            student_folder=student_folder,
            assignment_name=assignment_name,
            exam_name="BMI541 Biostatistics Exam"
        )

        # Display results
        console.print("\n[green]✅ SUCCESS![/green]\n")
        console.print(f"Extracted {len(exam_spec.questions)} questions")
        console.print(f"Total points: {exam_spec.total_points}")

        console.print("\n[bold]Questions:[/bold]")
        for q in exam_spec.questions:
            console.print(f"  {q.question_id}: {q.points} pts - {q.question_text[:60]}...")

        return True

    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        console.print("\n[yellow]Check your .env file has:[/yellow]")
        console.print("  - GCS_PROJECT_ID")
        console.print("  - GCS_BUCKET_NAME")
        console.print("  - GCS_CREDENTIALS_PATH")
        console.print("  - OPENROUTER_API_KEY")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
