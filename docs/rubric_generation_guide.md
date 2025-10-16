# Rubric and Answer Key Generation Guide

The AITA rubric generation pipeline creates detailed answer keys and grading rubrics for exam questions using advanced LLM analysis. This guide covers setup, usage, and customization options.

## Overview

The rubric generation system:
- **Generates step-by-step answer keys** with detailed solutions
- **Creates comprehensive grading rubrics** with point breakdowns
- **Supports user-provided rubrics** and custom grading instructions
- **Integrates with existing pipelines** and saves results for grading

## Prerequisites

1. **Question Extraction Complete**: Run the question extraction pipeline first to generate `exam_spec.json`
2. **API Configuration**: Set up OpenRouter API key in your environment
3. **Dependencies**: Install required packages with `pip install -e .`

## Quick Start

### Basic Usage

Generate rubrics for the default assignment:

```bash
aita generate-rubric
```

### With Custom Instructions

Provide general grading instructions:

```bash
aita generate-rubric --instructions grading_rules.txt
```

### With User-Provided Rubrics

Use custom rubrics for specific questions:

```bash
aita generate-rubric --user-rubrics my_rubrics.json --instructions grading_rules.txt
```

### Force Regeneration

Overwrite existing rubrics:

```bash
aita generate-rubric --force --assignment "BMI541_Midterm"
```

## Configuration Files

The system supports three types of user input files in the `intermediateproduct/rubrics/` directory:

### 1. User Rubrics (`user_rubrics.json`)

Provide complete rubrics for specific questions:

```json
{
  "1a": {
    "total_points": 10,
    "criteria": [
      {
        "points": 5,
        "description": "Correctly applies the derivative formula",
        "examples": [
          "Uses power rule: d/dx(x²) = 2x",
          "Identifies f'(x) = 2x + 3"
        ]
      },
      {
        "points": 3,
        "description": "Shows clear work and steps",
        "examples": ["Lists each step", "Work is organized"]
      },
      {
        "points": 2,
        "description": "Final answer is correct",
        "examples": ["f'(x) = 2x + 3"]
      }
    ]
  }
}
```

### 2. General Instructions (`rubric_instructions.txt`)

Define overall grading rules and focus areas:

```text
General Grading Instructions for Math Exam

POINT DEDUCTION RULES:
- Deduct 0.5 points for arithmetic errors if approach is correct
- Deduct 1 point for incorrect final answer if methodology is sound

PARTIAL CREDIT GUIDELINES:
- Give partial credit for correct setup even if execution fails
- Award points for clear work shown, even if incomplete

FOCUS AREAS:
- Problem-solving approach and logical reasoning
- Correct application of mathematical concepts
- Clarity of explanation and organization
```

### 3. Question-Specific Instructions (`question_specific_instructions.json`)

Provide targeted guidance for individual questions:

```json
{
  "1a": "Focus on correct application of derivative rules. Award full credit for f'(x) = 2x + 3 with clear work.",
  "2b": "Look for accurate definition of limits and intuitive understanding with examples.",
  "3c": "Application problem requiring setup (30%), calculation (50%), interpretation (20%)."
}
```

## Input Hierarchy

The system applies user inputs in this priority order:

1. **Complete user rubrics** → Used directly if provided for a question
2. **Question-specific instructions** → Used to guide LLM generation
3. **General instructions** → Applied to all generated rubrics
4. **Default generation** → LLM generates based on question content alone

## File Locations

### Input Files
- `intermediateproduct/rubrics/user_rubrics.json` - Custom rubrics
- `intermediateproduct/rubrics/rubric_instructions.txt` - General instructions
- `intermediateproduct/rubrics/question_specific_instructions.json` - Per-question guidance

### Output Files
- `data/results/answer_key.json` - Final answer keys
- `data/results/rubric.json` - Final rubrics
- `intermediateproduct/rubrics/generated_answer_keys.json` - Intermediate answer keys
- `intermediateproduct/rubrics/generated_rubrics.json` - Intermediate rubrics

## CLI Options

| Option | Description |
|--------|-------------|
| `--assignment, -a` | Assignment name (default: "exam1") |
| `--user-rubrics, -r` | Path to user-provided rubrics JSON file |
| `--instructions, -i` | Path to grading instructions text file |
| `--useradjust, -u` | Path to natural language adjustment instructions |
| `--data-dir, -d` | Data directory containing exam specification |
| `--force, -f` | Force regeneration even if rubrics exist |
| `--verbose, -v` | Enable verbose logging |
| `--dry-run` | Preview what would be generated |

## Rubric Adjustments

The `--useradjust` option allows you to iteratively refine existing rubrics using natural language instructions. This feature is particularly useful after reviewing generated rubrics and identifying areas for improvement.

### How It Works

1. **Generate Initial Rubrics**: First run the standard rubric generation
2. **Create Adjustment File**: Write natural language instructions describing desired changes
3. **Apply Adjustments**: Use `--useradjust` to intelligently update rubrics
4. **Review Changes**: Check the updated rubrics and adjustment logs

### Adjustment File Format

Create a plain text file with natural language instructions:

```text
For question 1a and 1b about derivatives, add 2 points for showing work.

Question 2 needs clearer partial credit - split the 10 points into:
- 3 points for correct approach/setup
- 4 points for accurate calculation
- 3 points for final answer

Add a criterion for question 3 that awards 1 point for including correct units.

Questions 4a through 4c should have more emphasis on explanation - increase
the explanation component from 2 to 4 points and reduce calculation points accordingly.
```

### Examples

```bash
# Initial generation
aita generate-rubric --assignment "calculus_midterm"

# After review, apply adjustments
aita generate-rubric --assignment "calculus_midterm" --useradjust adjustments.txt

# Apply additional refinements
echo "Question 5 should award partial credit for dimensional analysis" >> adjustments.txt
aita generate-rubric --assignment "calculus_midterm" --useradjust adjustments.txt
```

### Adjustment Types Supported

- **Add Criteria**: "Add a criterion for showing work (2 points)"
- **Modify Points**: "Increase question 2 from 10 to 12 points"
- **Split Criteria**: "Split the 6-point calculation into setup (2 pts) and execution (4 pts)"
- **Clarify Descriptions**: "Make the explanation criteria more specific"
- **Redistribute Points**: "Move 1 point from calculation to explanation"

### Safety Features

- **Automatic Backup**: Original rubrics are backed up before changes
- **Point Validation**: Ensures criteria points always sum to question totals
- **Audit Trail**: All adjustments are logged with timestamps
- **Rollback Capability**: Backups allow reverting to previous versions

### Output Structure

After adjustments, you'll find:

```
intermediateproduct/rubrics/
├── generated_rubrics.json              # Updated rubrics
├── generated_answer_keys.json          # Original answer keys
├── adjustment_history/
│   ├── rubrics_backup_20250116_143022.json    # Original rubrics
│   └── adjustment_log_20250116_143022.json    # Change log
└── user_adjustments.txt               # Latest adjustment instructions
```

### Best Practices

1. **Be Specific**: Reference exact question IDs and point values
2. **One Change Per Line**: Separate different adjustments for clarity
3. **Review First**: Always review generated rubrics before adjusting
4. **Test Incrementally**: Make small adjustments and verify results
5. **Keep Records**: Save adjustment files for documentation

## Answer Key Format

Generated answer keys include:

```json
{
  "question_id": "1a",
  "correct_answer": "f'(x) = 2x + 3",
  "solution_steps": [
    "Apply the power rule to x²: d/dx(x²) = 2x",
    "Apply the power rule to 3x: d/dx(3x) = 3",
    "Combine results: f'(x) = 2x + 3"
  ],
  "alternative_answers": ["dy/dx = 2x + 3"],
  "explanation": "Using basic differentiation rules for polynomials",
  "grading_notes": "Accept equivalent forms and notation"
}
```

## Rubric Format

Generated rubrics include:

```json
{
  "question_id": "1a",
  "total_points": 10,
  "criteria": [
    {
      "points": 5,
      "description": "Correct application of differentiation rules",
      "examples": ["Uses power rule correctly", "Applies linearity"]
    }
  ]
}
```

## Programming Interface

### High-Level Function

```python
from aita.pipelines.generate_rubric import generate_rubrics_for_assignment

answer_keys, rubrics = generate_rubrics_for_assignment(
    assignment_name="BMI541_Midterm",
    user_rubrics_file="custom_rubrics.json",
    instructions_file="grading_instructions.txt",
    force_regenerate=True
)
```

### Pipeline Class

```python
from aita.pipelines.generate_rubric import create_rubric_generation_pipeline
from aita.domain.models import ExamSpec

# Create pipeline
pipeline = create_rubric_generation_pipeline("exam1", data_dir)

# Load exam specification
exam_spec = ExamSpec.load_from_file("data/results/exam_spec.json")

# Generate rubrics
answer_keys, rubrics = pipeline.generate_from_exam_spec(
    exam_spec=exam_spec,
    assignment_name="exam1",
    force_regenerate=False
)
```

## Best Practices

### Writing Effective Instructions

1. **Be Specific**: Clear point deduction rules work better than vague guidance
2. **Consider Partial Credit**: Specify how to award points for incomplete work
3. **Include Examples**: Show what constitutes good vs. poor responses
4. **Focus on Learning Objectives**: Align rubrics with what you want to assess

### Organizing User Rubrics

1. **Start Small**: Provide rubrics for key questions, let LLM handle others
2. **Use Consistent Structure**: Follow the same point breakdown patterns
3. **Include Examples**: Concrete examples help with consistent grading
4. **Test and Iterate**: Review generated rubrics and refine inputs

### Quality Assurance

1. **Review Generated Content**: Always check LLM-generated rubrics before use
2. **Test with Sample Responses**: Validate rubrics with actual student work
3. **Adjust as Needed**: Modify instructions based on grading experience
4. **Document Changes**: Keep track of rubric modifications

## Troubleshooting

### Common Issues

**"No exam specification found"**
- Run question extraction pipeline first
- Check that `data/results/exam_spec.json` exists

**"Invalid JSON in user rubrics"**
- Validate JSON syntax in configuration files
- Use example files as templates

**"LLM API errors"**
- Check OpenRouter API key configuration
- Verify sufficient API credits
- Try with fewer questions if rate limited

### Getting Help

1. **Dry Run Mode**: Use `--dry-run` to preview without API costs
2. **Verbose Logging**: Add `--verbose` for detailed error information
3. **Example Files**: Check `intermediateproduct/rubrics/*.example` for templates
4. **Demo Script**: Run `examples/rubric_generation_demo.py` for testing

## Integration with Other Pipelines

The rubric generation pipeline integrates with:

- **Question Extraction**: Requires exam specification as input
- **Transcription Pipeline**: Answer keys help validate transcribed responses
- **Grading Pipeline**: Uses generated rubrics for automated grading
- **Report Generation**: Incorporates rubric criteria in grade reports

## Cost Considerations

- **Token Usage**: Generates detailed prompts for each question
- **Retry Logic**: May make multiple API calls if parsing fails
- **Optimization**: Use user rubrics to reduce LLM generation costs
- **Monitoring**: Check cost tracking with `aita costs` command

## Next Steps

After generating rubrics:

1. **Review Generated Content**: Verify answer keys and rubric quality
2. **Test with Sample Data**: Use transcribed responses to validate rubrics
3. **Run Grading Pipeline**: Apply rubrics to grade all student responses
4. **Generate Reports**: Create detailed grade reports for distribution