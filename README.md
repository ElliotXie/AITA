# AITA - Automatic Image Test Analysis

An intelligent system for automatically grading handwritten exams using OCR and Large Language Models.

## Features

- **Smart Grouping**: Automatically groups exam pages by student using OCR and fuzzy name matching
- **Question Extraction**: Uses LLM to extract question structure, points, and content from exam images
- **Rubric Generation**: Automatically generates grading rubrics and answer keys
- **Answer Transcription**: Converts handwritten student answers to text
- **Automated Grading**: Grades student responses using AI with detailed feedback

## System Architecture

```
aita/
├─ aita/                              # Main package
│  ├─ config.py                       # Configuration management
│  ├─ main.py                         # CLI interface
│  ├─ domain/                         # Domain models
│  ├─ services/                       # Core services (OCR, LLM, Storage)
│  ├─ pipelines/                      # Processing pipelines
│  ├─ utils/                          # Utilities
│  └─ report/                         # Report generation
├─ data/                              # Input/output data
└─ tests/                             # Test suite
```

## Installation

```bash
pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env`
2. Configure your API keys and settings:
   - OpenRouter API key for LLM access
   - Google Cloud Storage credentials
   - Adjust processing parameters as needed

## Usage

```bash
# Initialize data structure
aita init-data

# Process exam images
aita ingest path/to/exam/images/

# Extract questions from one student's exam
aita extract-questions

# Generate rubric and answer key
aita generate-rubric

# Transcribe all student answers
aita transcribe

# Grade all exams
aita grade

# Generate reports
aita report
```

## Requirements

- Python 3.8+
- OpenRouter API access
- Google Cloud Storage bucket (public)
- EasyOCR dependencies

## License

MIT