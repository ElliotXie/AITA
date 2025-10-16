# Question Extraction Pipeline - User Guide

## Overview

The Question Extraction Pipeline is a core component of AITA that automatically extracts the structure of an exam from images. It uses LLM vision capabilities to identify:

- **Question identifiers** (e.g., "1a", "1b", "2", "3c")
- **Question text/content**
- **Point values** for each question
- **Page numbers** where questions appear
- **Question types** (calculation, short answer, etc.)

## How It Works

```
Student Folder → Upload to GCS → LLM Vision Analysis → Parse Structure → Validate → Save JSON
```

### Detailed Process

1. **Image Discovery**: Scans student folder for exam image files (JPG, PNG, etc.)
2. **Cloud Upload**: Uploads all pages to Google Cloud Storage (GCS) for public access
3. **LLM Analysis**: Sends all image URLs to Gemini 2.5 Flash vision model
4. **Structure Extraction**: LLM analyzes images and returns JSON with question structure
5. **Validation**: Checks for completeness, uniqueness, and validity
6. **Persistence**: Saves exam specification to `data/results/exam_spec.json`

## Prerequisites

### 1. Environment Configuration

Ensure your `.env` file has:

```bash
# Google Cloud Storage
GCS_PROJECT_ID=your-project-id
GCS_BUCKET_NAME=your-bucket-name
GCS_CREDENTIALS_PATH=path/to/credentials.json

# OpenRouter LLM API
OPENROUTER_API_KEY=your-api-key
OPENROUTER_MODEL=google/gemini-2.5-flash
```

### 2. Student Folder Structure

```
data/grouped/
└── Student_001/
    ├── page_1.jpg
    ├── page_2.jpg
    ├── page_3.jpg
    ├── page_4.jpg
    └── page_5.jpg
```

Images should be:
- Named in sequential order (will be sorted alphabetically)
- Clear and readable
- Standard formats: JPG, PNG, BMP, TIFF, WebP

## Usage

### Method 1: Python API (Recommended)

```python
from aita.pipelines.extract_questions import extract_questions_from_student

# Extract from one student's exam as sample
exam_spec = extract_questions_from_student(
    student_folder="data/grouped/Student_001",
    assignment_name="BMI541_Midterm",
    exam_name="BMI541 Biostatistics Midterm Exam"  # Optional
)

# Access results
print(f"Exam: {exam_spec.exam_name}")
print(f"Total Questions: {len(exam_spec.questions)}")
print(f"Total Points: {exam_spec.total_points}")

# Iterate through questions
for question in exam_spec.questions:
    print(f"{question.question_id}: {question.points} pts - {question.question_text[:50]}...")
```

### Method 2: Demo Script

```bash
python examples/extract_questions_demo.py
```

The demo script provides a rich interactive display showing:
- Extraction progress
- Summary statistics
- Detailed question table
- Point distribution analysis
- Next steps

### Method 3: Advanced - Custom Pipeline

```python
from pathlib import Path
from aita.pipelines.extract_questions import QuestionExtractionPipeline
from aita.services.llm.openrouter import create_openrouter_client
from aita.services.storage import create_storage_service

# Create services
llm_client = create_openrouter_client()
storage_service = create_storage_service()

# Create pipeline with custom configuration
pipeline = QuestionExtractionPipeline(
    llm_client=llm_client,
    storage_service=storage_service,
    data_dir=Path("data"),
    max_parse_retries=3  # Custom retry count
)

# Run extraction
exam_spec = pipeline.extract_from_student(
    student_folder=Path("data/grouped/Student_001"),
    assignment_name="exam1"
)
```

## Output Format

### ExamSpec JSON Structure

```json
{
  "exam_name": "BMI541 Biostatistics Midterm Exam",
  "total_pages": 5,
  "total_points": 100.0,
  "questions": [
    {
      "question_id": "1a",
      "question_text": "Calculate the mean and standard deviation...",
      "points": 5.0,
      "page_number": 1,
      "question_type": "calculation"
    },
    {
      "question_id": "1b",
      "question_text": "Interpret the results in context...",
      "points": 3.0,
      "page_number": 1,
      "question_type": "short_answer"
    }
  ]
}
```

### Question Types

- `multiple_choice` - Multiple choice questions
- `short_answer` - Brief answer questions
- `long_answer` - Essay-style questions
- `calculation` - Mathematical/statistical calculations
- `diagram` - Drawing or graphical questions

## Error Handling

### Common Issues and Solutions

#### 1. No Images Found

**Error**: `QuestionExtractionError: No image files found`

**Solution**:
- Verify student folder path is correct
- Check that images have valid extensions (.jpg, .png, etc.)
- Ensure images aren't in subdirectories

#### 2. GCS Upload Failure

**Error**: `QuestionExtractionError: All image uploads failed`

**Solution**:
- Check GCS credentials are valid
- Verify bucket exists and is accessible
- Ensure bucket has public read permissions (for LLM vision API)
- Check network connectivity

#### 3. JSON Parse Error

**Error**: `QuestionExtractionError: Failed to parse LLM response`

**Solution**:
- This typically auto-retries up to 2 times
- Check LLM API key is valid
- Verify image quality is sufficient
- Try with different/clearer images

#### 4. Validation Failure

**Error**: `QuestionExtractionError: Exam spec validation failed`

**Details**: Shows specific validation errors

**Solution**:
- Review error messages for specific issues
- Common problems:
  - Duplicate question IDs
  - Negative or zero points
  - Empty question text
  - Invalid page numbers
- May need to manually edit the exam_spec.json

### Retry Logic

The pipeline includes automatic retry mechanisms:

- **LLM API calls**: 3 retries with exponential backoff (handled by base LLM client)
- **JSON parsing**: 2 retries with enhanced prompts
- **GCS uploads**: Individual failures don't block entire pipeline

## Tips for Best Results

### 1. Image Quality

- **Resolution**: Higher is better, but 1500x2000+ pixels usually sufficient
- **Lighting**: Even lighting, no shadows or glare
- **Focus**: Sharp, not blurry
- **Orientation**: Upright, not rotated

### 2. Exam Layout

The pipeline works best when:
- Question numbers/letters are clearly visible
- Point values are written near questions
- Questions follow standard formatting (1, 1a, 1b, 2, 2a, etc.)

### 3. Which Student to Use?

- Use **any student's complete exam** as the sample
- All students should have the same exam questions
- Choose a student with clear, legible handwriting for best results
- The pipeline only extracts the question structure, not student answers

### 4. Cost Optimization

- **Gemini 2.5 Flash** is very cost-effective (~$0.02 per exam)
- Extract questions only once per exam (use for all students)
- Results are cached in `data/results/exam_spec.json`

## Advanced Features

### Custom Validation Rules

```python
from aita.pipelines.extract_questions import validate_exam_spec

# Run validation separately
errors = validate_exam_spec(exam_spec)

if errors:
    print("Validation errors found:")
    for error in errors:
        print(f"  - {error}")
```

### Manual Editing

After extraction, you can manually edit `data/results/exam_spec.json`:

```python
from aita.domain.models import ExamSpec

# Load existing spec
exam_spec = ExamSpec.load_from_file("data/results/exam_spec.json")

# Modify as needed
exam_spec.questions[0].points = 10.0
exam_spec.questions[0].question_text = "Updated text..."

# Save changes
exam_spec.save_to_file("data/results/exam_spec.json")
```

### Batch Processing Multiple Exams

```python
from pathlib import Path
from aita.pipelines.extract_questions import extract_questions_from_student

student_folders = Path("data/grouped").glob("Student_*")

for student_folder in student_folders:
    try:
        exam_spec = extract_questions_from_student(
            student_folder=str(student_folder),
            assignment_name="exam1"
        )
        print(f"✓ Extracted from {student_folder.name}")
        break  # Only need one successful extraction
    except Exception as e:
        print(f"✗ Failed {student_folder.name}: {e}")
        continue
```

## Integration with Other Pipelines

The extracted ExamSpec is used by:

1. **Rubric Generation** (Step 3): Creates answer keys and grading rubrics
2. **Transcription** (Step 4): Maps transcribed text to question IDs
3. **Grading** (Step 5): Applies rubrics to grade student answers
4. **Reports** (Step 6): Generates formatted grade reports

### Example Workflow

```python
# Step 2: Extract questions
exam_spec = extract_questions_from_student("data/grouped/Student_001", "exam1")

# Step 3: Generate rubrics (coming next)
from aita.pipelines.generate_rubric import generate_rubrics
rubrics = generate_rubrics(exam_spec)

# Step 4: Transcribe all students (coming next)
from aita.pipelines.transcribe import transcribe_all_students
transcriptions = transcribe_all_students(exam_spec, "data/grouped")

# Step 5: Grade all students (coming next)
from aita.pipelines.grade import grade_all_students
grades = grade_all_students(exam_spec, transcriptions, rubrics)
```

## Troubleshooting

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run extraction - will show detailed logs
exam_spec = extract_questions_from_student(...)
```

### Inspecting LLM Responses

The pipeline logs the raw LLM response. Check logs:

```
2024-10-15 14:30:21 - aita.pipelines.extract_questions - DEBUG - LLM response: {"exam_name": ...}
```

### Testing Without Real LLM Calls

Use mocked services for testing:

```python
from unittest.mock import Mock
import json

# Create mock LLM client
mock_llm = Mock()
mock_llm.complete_with_images = Mock(return_value=json.dumps({
    "exam_name": "Test",
    "total_pages": 2,
    "questions": [...]
}))

# Use in pipeline
pipeline = QuestionExtractionPipeline(
    llm_client=mock_llm,
    storage_service=mock_storage,
    data_dir=Path("data")
)
```

## Performance

### Typical Extraction Times

- **5-page exam**: 30-60 seconds
  - Upload: 5-10 seconds
  - LLM analysis: 20-40 seconds
  - Parsing/validation: <1 second

### Cost Estimates

Using Gemini 2.5 Flash (as of October 2024):
- **Input**: ~$0.001 per image (5 images = $0.005)
- **Output**: ~$0.005 per response
- **Total per exam**: ~$0.01-0.02

## Next Steps

After successful question extraction:

1. ✅ Review `data/results/exam_spec.json` for accuracy
2. ✅ Make any necessary manual corrections
3. ➡️ **Proceed to Step 3**: Rubric Generation Pipeline
4. ➡️ Then continue with transcription and grading

## API Reference

See the module docstrings for detailed API documentation:

```python
from aita.pipelines import extract_questions
help(extract_questions)
```

Key classes and functions:
- `QuestionExtractionPipeline` - Main pipeline class
- `extract_questions_from_student()` - High-level convenience function
- `parse_question_extraction_response()` - Response parser
- `validate_exam_spec()` - Validation function

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Enable debug logging
4. Check the test suite for examples: `tests/pipelines/test_extract_questions.py`
5. Review the demo script: `examples/extract_questions_demo.py`
