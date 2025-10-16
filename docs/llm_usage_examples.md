# LLM Image Processing Usage Examples

This document provides comprehensive examples of how to use the AITA LLM module for image processing and analysis tasks.

## Basic Setup

First, ensure you have your OpenRouter API key configured:

```bash
# Set up environment variables
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
export OPENROUTER_MODEL="google/gemini-2.5-flash"
```

Or create a `.env` file based on `.env.example`.

## Creating an LLM Client

```python
from aita.services.llm.openrouter import OpenRouterClient, create_openrouter_client
from aita.utils.prompts import get_name_extraction_prompt, get_question_extraction_prompt

# Method 1: Direct instantiation
client = OpenRouterClient(
    api_key="your_api_key",
    model="google/gemini-2.5-flash"
)

# Method 2: Factory function (uses environment variables)
client = create_openrouter_client()

# Method 3: Using configuration
from aita.config import get_config
config = get_config()
client = OpenRouterClient(**config.openrouter.__dict__)
```

## Basic Text Processing

```python
# Simple text completion
response = client.complete_text("What is the capital of France?")
print(response)  # "The capital of France is Paris."

# With system prompt
response = client.complete_text(
    "Explain photosynthesis in simple terms.",
    system_prompt="You are a biology teacher explaining to high school students."
)
```

## Single Image Analysis

### Name Extraction from Exam Headers

```python
from aita.utils.prompts import get_name_extraction_prompt

# Extract student name from exam header
image_url = "https://storage.googleapis.com/your-bucket/exam_page_1.jpg"
prompt = get_name_extraction_prompt()

student_name = client.complete_with_image(prompt, image_url)
print(f"Extracted name: {student_name}")
```

### Question Structure Extraction

```python
from aita.utils.prompts import get_question_extraction_prompt

# Extract questions from a complete exam
image_url = "https://storage.googleapis.com/your-bucket/complete_exam.jpg"
prompt = get_question_extraction_prompt()

questions_json = client.complete_with_image(prompt, image_url)

# Parse the JSON response
import json
questions = json.loads(questions_json)
for question in questions["questions"]:
    print(f"Question {question['question_id']}: {question['points']} points")
```

### Answer Transcription

```python
from aita.utils.prompts import get_transcription_prompt

# Transcribe handwritten student answer
question_text = "Explain the water cycle."
image_url = "https://storage.googleapis.com/your-bucket/student_answer.jpg"
prompt = get_transcription_prompt(question_text)

transcription_json = client.complete_with_image(prompt, image_url)
transcription = json.loads(transcription_json)

print(f"Transcribed text: {transcription['transcribed_text']}")
print(f"Confidence: {transcription['confidence']}")
```

## Multiple Image Analysis

### Analyzing Complete Multi-Page Exams

```python
# Process multiple pages of an exam
exam_pages = [
    "https://storage.googleapis.com/your-bucket/page_1.jpg",
    "https://storage.googleapis.com/your-bucket/page_2.jpg",
    "https://storage.googleapis.com/your-bucket/page_3.jpg"
]

# Extract questions from all pages
prompt = get_question_extraction_prompt()
complete_exam_analysis = client.complete_with_images(prompt, exam_pages)

print("Complete exam structure:", complete_exam_analysis)
```

### Comparing Student Responses

```python
# Compare multiple student answers to the same question
student_answers = [
    "https://storage.googleapis.com/your-bucket/student1_q1.jpg",
    "https://storage.googleapis.com/your-bucket/student2_q1.jpg",
    "https://storage.googleapis.com/your-bucket/student3_q1.jpg"
]

prompt = """
Compare these three student answers to the same question.
Identify common mistakes and highlight the best response.
Provide your analysis in JSON format.
"""

comparison = client.complete_with_images(prompt, student_answers)
```

## Complete Grading Workflow

### End-to-End Example

```python
from aita.utils.prompts import (
    get_name_extraction_prompt,
    get_question_extraction_prompt,
    get_transcription_prompt,
    get_grading_prompt
)
import json

def grade_student_exam(client, exam_image_urls, answer_key):
    """Complete grading workflow for a single student"""

    # Step 1: Extract student name from first page
    name_prompt = get_name_extraction_prompt()
    student_name = client.complete_with_image(name_prompt, exam_image_urls[0])

    # Step 2: Extract question structure (if not already done)
    question_prompt = get_question_extraction_prompt()
    questions_json = client.complete_with_images(question_prompt, exam_image_urls)
    questions = json.loads(questions_json)

    # Step 3: Transcribe and grade each question
    results = {
        "student_name": student_name.strip(),
        "grades": []
    }

    for question in questions["questions"]:
        # Find the page containing this question
        page_url = exam_image_urls[question["page_number"] - 1]

        # Transcribe the answer
        transcription_prompt = get_transcription_prompt(question["question_text"])
        transcription_json = client.complete_with_image(transcription_prompt, page_url)
        transcription = json.loads(transcription_json)

        # Grade the answer
        grading_prompt = get_grading_prompt(
            question["question_text"],
            answer_key[question["question_id"]]["correct_answer"],
            answer_key[question["question_id"]]["rubric"],
            transcription["transcribed_text"]
        )
        grade_json = client.complete_text(grading_prompt)
        grade = json.loads(grade_json)

        results["grades"].append({
            "question_id": question["question_id"],
            "transcription": transcription,
            "grade": grade
        })

    return results

# Example usage
exam_urls = [
    "https://storage.googleapis.com/your-bucket/student1_page1.jpg",
    "https://storage.googleapis.com/your-bucket/student1_page2.jpg"
]

answer_key = {
    "1a": {
        "correct_answer": "The mitochondria is the powerhouse of the cell.",
        "rubric": "5 points for correct identification, partial credit for related concepts."
    }
}

results = grade_student_exam(client, exam_urls, answer_key)
print(f"Student: {results['student_name']}")
for grade in results["grades"]:
    print(f"Question {grade['question_id']}: {grade['grade']['points_earned']}/{grade['grade']['points_possible']}")
```

## Error Handling and Best Practices

### Robust Error Handling

```python
import time
from requests.exceptions import RequestException

def safe_llm_call(client, prompt, image_url, max_retries=3):
    """Make LLM call with comprehensive error handling"""

    for attempt in range(max_retries):
        try:
            response = client.complete_with_image(prompt, image_url)
            return response

        except RequestException as e:
            if "401" in str(e):
                raise ValueError("Invalid API key")
            elif "429" in str(e):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
            else:
                print(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return None

        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                raise

# Usage
try:
    result = safe_llm_call(client, prompt, image_url)
    if result:
        print("Success:", result)
    else:
        print("Failed to get valid response")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Cost Monitoring

```python
def estimate_grading_cost(client, num_students, pages_per_student, questions_per_page):
    """Estimate the cost of grading an entire class"""

    # Rough token estimates
    tokens_per_image = 1000  # Conservative estimate
    tokens_per_response = 200

    total_images = num_students * pages_per_student
    total_questions = num_students * pages_per_student * questions_per_page

    # Operations: name extraction, question extraction, transcription, grading
    name_extractions = num_students
    question_extractions = 1  # Done once per exam
    transcriptions = total_questions
    gradings = total_questions

    total_input_tokens = (
        (name_extractions + transcriptions) * tokens_per_image +
        question_extractions * tokens_per_image * pages_per_student +
        gradings * 100  # Text-only grading prompts
    )

    total_output_tokens = (
        (name_extractions + question_extractions + transcriptions + gradings) * tokens_per_response
    )

    cost_info = client.estimate_cost(total_input_tokens, total_output_tokens)

    if "error" not in cost_info:
        print(f"Estimated cost: ${cost_info['total_cost']:.2f}")
        print(f"Cost per student: ${cost_info['total_cost'] / num_students:.2f}")
    else:
        print("Could not estimate cost:", cost_info["error"])

# Example usage
estimate_grading_cost(client, num_students=30, pages_per_student=5, questions_per_page=2)
```

## Configuration Examples

### Using Environment Variables

```bash
# .env file
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=google/gemini-2.5-flash
AITA_MAX_RETRIES=5
AITA_RETRY_DELAY=1.5
```

### Programmatic Configuration

```python
from aita.config import AITAConfig, OpenRouterConfig

# Create custom configuration
openrouter_config = OpenRouterConfig(
    api_key="your_key",
    model="google/gemini-2.5-flash",
    max_retries=5,
    retry_delay=1.0
)

# Use with client
client = OpenRouterClient(**openrouter_config.__dict__)
```

## Testing Your Setup

### Quick Validation Test

```python
def test_llm_setup():
    """Quick test to validate LLM setup"""

    try:
        client = create_openrouter_client()

        # Test basic text completion
        response = client.complete_text("Say 'Hello, AITA!'")
        print("Text completion test:", response)

        # Test image analysis with a public test image
        test_image = "https://storage.googleapis.com/exam1_uwmadison/courses/test_assignment/2025-10-14/test_student/page_1.jpg"
        response = client.complete_with_image("Describe this image briefly.", test_image)
        print("Image analysis test:", response[:100] + "...")

        print("✅ LLM setup is working correctly!")

    except Exception as e:
        print(f"❌ LLM setup failed: {e}")
        print("Check your OPENROUTER_API_KEY and internet connection.")

# Run the test
test_llm_setup()
```

## Next Steps

After verifying your LLM setup works:

1. **Set up Google Cloud Storage** for hosting exam images
2. **Implement the ingest pipeline** for grouping student exams
3. **Create grading rubrics** for your specific exam questions
4. **Run the complete AITA workflow** on a sample exam

For more information, see:
- `docs/implementation_plan.md` for the overall project roadmap
- `aita/utils/prompts.py` for all available prompt templates
- `tests/services/test_llm_integration.py` for more examples