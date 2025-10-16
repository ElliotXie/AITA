# AITA User Guide - Complete Exam Grading Pipeline

AITA (Automatic Image Test Analysis) is an AI-powered system that automatically grades handwritten exams. This guide walks you through the complete 5-step pipeline from raw exam images to detailed graded results.

## 📋 Prerequisites

1. **Install AITA**:
   ```bash
   cd AITA
   pip install -e .
   ```

2. **Set up environment** (create `.env` file):
   ```bash
   OPENROUTER_API_KEY=your_openrouter_key
   GOOGLE_APPLICATION_CREDENTIALS=path/to/gcs-credentials.json
   GCS_BUCKET_NAME=your_bucket_name
   ```

3. **Prepare data structure**:
   ```
   data/
   ├── raw_images/          # Place all exam images here
   └── student_roster.txt   # One student name per line
   ```

## 🚀 Complete Pipeline Workflow

### Step 1: Smart Grouping
**Purpose**: Organize exam images by student using AI name recognition.

```bash
# Group all exam images by student
aita smart-group

# Use custom data directory
aita smart-group --data-dir /path/to/exam/data

# Preview what would be processed
aita smart-group --dry-run
```

**Output**: `data/grouped/` with folders for each student containing their exam pages.

---

### Step 2: Question Extraction
**Purpose**: Analyze exam structure and extract question details.

```bash
# Extract questions from exam images
aita extract-questions

# Process specific assignment
aita extract-questions --assignment "BMI541_Midterm"

# Preview extraction
aita extract-questions --dry-run
```

**Output**: `data/results/exam_spec.json` with question structure and points.

---

### Step 3: Generate Rubrics & Answer Keys
**Purpose**: Create AI-generated grading rubrics and detailed answer keys.

```bash
# Generate rubrics and answer keys
aita generate-rubric

# Use custom grading instructions
aita generate-rubric --instructions grading_rules.txt

# Use user-provided rubrics for some questions
aita generate-rubric --user-rubrics my_rubrics.json

# Apply natural language adjustments to existing rubrics
aita generate-rubric --useradjust rubric_adjustments.txt

# Force regeneration
aita generate-rubric --force
```

**Adjusting Existing Rubrics**:
After generating initial rubrics, you can refine them using natural language instructions:

1. Create an adjustment file with your instructions:
```bash
echo "For question 1a and 1b about derivatives, add 2 points for showing work.
Question 2 needs clearer partial credit - split the 10 points into 3 for approach,
4 for calculation, and 3 for final answer.
Add a criterion for question 3 that awards 1 point for correct units." > rubric_adjustments.txt
```

2. Apply the adjustments:
```bash
aita generate-rubric --useradjust rubric_adjustments.txt
```

**Output**:
- `intermediateproduct/rubrics/generated_rubrics.json` (updated with adjustments)
- `intermediateproduct/rubrics/generated_answer_keys.json`
- `intermediateproduct/rubrics/adjustment_history/` (backup and logs)

---

### Step 4: Transcription
**Purpose**: Convert handwritten answers to text using AI vision.

```bash
# Transcribe all students
aita transcribe

# Transcribe specific student
aita transcribe --student "Turner, Evan"

# Disable HTML report generation
aita transcribe --no-html

# Preview transcription
aita transcribe --dry-run
```

**Output**: `intermediateproduct/transcription_results/` with text transcriptions per student.

---

### Step 5: Grading
**Purpose**: Grade student responses using AI evaluation with rubrics.

```bash
# Grade all students
aita grade

# Grade specific student
aita grade --student "Turner, Evan"

# Grade specific assignment
aita grade --assignment "BMI541_Midterm"

# Preview grading
aita grade --dry-run
```

**Output**: `intermediateproduct/grading_results/` with detailed grades and feedback.

## 📊 Monitoring & Analysis

### Check Pipeline Status
```bash
# View overall system status
aita status

# Show version and available components
aita version
```

### Cost Tracking
```bash
# View recent API costs
aita costs

# Detailed cost breakdown
aita costs --detailed

# View specific session
aita costs --session session_20250115_143022_abc123
```

### Generate Reports
```bash
# Generate HTML transcription reports
aita report-transcription

# Generate report for specific student
aita report-transcription --student "Smith, John"
```

## 📁 Complete Output Structure

After running all 5 steps, your directory structure will be:

```
AITA/
├── data/
│   ├── raw_images/                    # Input: Original exam images
│   ├── grouped/                       # Step 1: Images organized by student
│   │   ├── Turner, Evan/
│   │   │   ├── page_001.jpg
│   │   │   └── page_002.jpg
│   │   └── Mei, Elizabeth/
│   │       ├── page_001.jpg
│   │       └── page_002.jpg
│   ├── results/
│   │   └── exam_spec.json             # Step 2: Question structure
│   └── student_roster.txt
├── intermediateproduct/
│   ├── rubrics/                       # Step 3: Rubrics & answer keys
│   │   ├── generated_rubrics.json
│   │   └── generated_answer_keys.json
│   ├── transcription_results/         # Step 4: Text transcriptions
│   │   ├── Turner, Evan/
│   │   │   ├── question_transcriptions.json
│   │   │   └── transcription_report.html
│   │   └── summary_report.json
│   ├── grading_results/               # Step 5: Final grades
│   │   ├── students/
│   │   │   ├── Turner, Evan/
│   │   │   │   ├── detailed_grades.json
│   │   │   │   └── grade_summary.json
│   │   │   └── Mei, Elizabeth/
│   │   │       ├── detailed_grades.json
│   │   │       └── grade_summary.json
│   │   ├── summary_report.json
│   │   └── grade_distribution.json
│   └── cost_tracking/                 # API cost monitoring
│       └── session_*.json
└── docs/
    └── user_guide.md                  # This guide
```

## 🎯 Example: Complete Workflow

Here's a complete example for grading a BMI541 midterm exam:

```bash
# 1. Set up data
mkdir -p data/raw_images
# Copy exam images to data/raw_images/
echo -e "Turner, Evan\nMei, Elizabeth\nSmith, John" > data/student_roster.txt

# 2. Run complete pipeline
aita smart-group --assignment "BMI541_Midterm"
aita extract-questions --assignment "BMI541_Midterm"
aita generate-rubric --assignment "BMI541_Midterm"
aita transcribe --assignment "BMI541_Midterm"
aita grade --assignment "BMI541_Midterm"

# 3. Check results
aita status
aita costs
```

## 🔧 Troubleshooting

### Common Issues

**"No student folders found"**
- Ensure step 1 (smart-group) completed successfully
- Check that `data/grouped/` contains student folders

**"Rubrics not found"**
- Run step 3 (generate-rubric) before step 5 (grade)
- Check that `intermediateproduct/rubrics/` contains rubric files

**"Transcription results not found"**
- Run step 4 (transcribe) before step 5 (grade)
- Check that `intermediateproduct/transcription_results/` contains student data

**API Errors**
- Verify your `.env` file has correct API keys
- Check internet connection and API quotas
- Use `aita costs` to monitor usage

### Getting Help

```bash
# Get help for any command
aita --help
aita grade --help

# Check system status
aita status

# View detailed logs
aita transcribe --verbose
```

## 🎉 Success!

Once all 5 steps complete successfully, you'll have:

- ✅ **Organized Images**: All exam pages sorted by student
- ✅ **Question Structure**: Complete exam specification with points
- ✅ **Answer Keys**: AI-generated solutions for each question
- ✅ **Rubrics**: Detailed grading criteria
- ✅ **Transcriptions**: Handwriting converted to searchable text
- ✅ **Grades**: Complete scoring with detailed feedback
- ✅ **Reports**: HTML reports and statistics
- ✅ **Cost Tracking**: Detailed API usage monitoring

Your exam is now fully graded with comprehensive AI-powered analysis!