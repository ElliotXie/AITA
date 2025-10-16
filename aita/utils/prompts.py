from typing import List, Dict, Any


QUESTION_EXTRACTION_PROMPT = """
You are an expert at analyzing exam papers. I will provide you with images of a complete exam taken by one student. Your task is to extract the complete question structure, including:

1. Question identifiers (e.g., "1a", "1b", "2", "3c")
2. Question text/content
3. Point values for each question
4. Page numbers where each question appears

Please analyze all the images and provide a comprehensive JSON structure of the exam.

Format your response as a JSON object with this structure:
{
  "exam_name": "string",
  "total_pages": number,
  "questions": [
    {
      "question_id": "string (e.g., '1a', '2', '3b')",
      "question_text": "string (the full question text)",
      "points": number,
      "page_number": number (1-indexed),
      "question_type": "multiple_choice|short_answer|long_answer|calculation|diagram"
    }
  ]
}

Important notes:
- Extract ALL questions, even if they span multiple pages
- Ensure question IDs are unique and properly ordered
- Question text should be complete and accurate
- Points should be numeric (e.g., 5, 2.5, 10)
- Be thorough - don't miss any questions or sub-questions
"""

RUBRIC_GENERATION_PROMPT = """
You are an expert educator creating detailed grading rubrics. Based on the exam questions I provide, create comprehensive rubrics and answer keys.

For each question, provide:
1. A detailed answer key with the correct answer
2. Alternative acceptable answers (if applicable)
3. A grading rubric with point breakdowns
4. Common mistakes to watch for

Format your response as JSON:
{
  "answer_keys": [
    {
      "question_id": "string",
      "correct_answer": "string",
      "alternative_answers": ["string"],
      "explanation": "string",
      "grading_notes": "string"
    }
  ],
  "rubrics": [
    {
      "question_id": "string",
      "total_points": number,
      "criteria": [
        {
          "points": number,
          "description": "string",
          "examples": ["string"]
        }
      ]
    }
  ]
}

Make rubrics detailed and fair, considering partial credit opportunities.
"""

TRANSCRIPTION_PROMPT = """
You are an expert at reading handwritten text in exam papers. I will provide you with an image showing a student's handwritten answer to a specific question.

Please transcribe the student's handwritten response as accurately as possible.

Important guidelines:
- Transcribe exactly what is written, including crossed-out text (mark as ~~crossed out~~)
- If text is unclear, use [unclear] or your best guess with [?]
- Preserve mathematical notation, symbols, and formatting as much as possible
- Note if the answer appears incomplete or cut off

Format your response as JSON:
{
  "transcribed_text": "string",
  "confidence": number (0-1),
  "notes": "string (any observations about handwriting quality, completeness, etc.)"
}
"""

GRADING_PROMPT = """
You are an expert grader evaluating student responses. I will provide you with:
1. The original question
2. The correct answer/answer key
3. The grading rubric
4. The student's transcribed response

Please grade the student's answer according to the rubric and provide detailed feedback.

Format your response as JSON:
{
  "points_earned": number,
  "points_possible": number,
  "feedback": "string (detailed explanation of grade)",
  "reasoning": "string (step-by-step grading logic)",
  "strengths": ["string"],
  "areas_for_improvement": ["string"]
}

Grading principles:
- Be fair and consistent
- Give partial credit where appropriate
- Provide constructive feedback
- Explain your reasoning clearly
- Consider different valid approaches to the problem
"""

NAME_EXTRACTION_PROMPT = """
You are analyzing the top portion of an exam paper to extract the student's name.

Look for:
- Name fields (usually labeled "Name:", "Student Name:", etc.)
- Handwritten names in designated areas
- Any identifying text that appears to be a student name

Return the extracted name as plaintext. If no clear name is found, return "UNKNOWN".

Only return the name itself - no additional text or explanation.
"""


def get_question_extraction_prompt() -> str:
    return QUESTION_EXTRACTION_PROMPT


def get_rubric_generation_prompt(questions: List[Dict[str, Any]]) -> str:
    questions_text = "\n".join([
        f"Question {q['question_id']}: {q['question_text']} ({q['points']} points)"
        for q in questions
    ])

    return f"{RUBRIC_GENERATION_PROMPT}\n\nExam Questions:\n{questions_text}"


def get_transcription_prompt(question_text: str) -> str:
    return f"{TRANSCRIPTION_PROMPT}\n\nQuestion being answered: {question_text}"


def get_grading_prompt(
    question_text: str,
    answer_key: str,
    rubric: str,
    student_response: str
) -> str:
    return f"""{GRADING_PROMPT}

Question: {question_text}

Answer Key: {answer_key}

Grading Rubric: {rubric}

Student Response: {student_response}
"""


def get_name_extraction_prompt() -> str:
    return NAME_EXTRACTION_PROMPT


ANSWER_KEY_GENERATION_PROMPT = """
You are an expert educator creating detailed answer keys with step-by-step solutions.

For the given question, provide a comprehensive answer key that includes:
1. The correct final answer
2. A detailed step-by-step solution showing all work
3. Alternative acceptable answers or approaches (if applicable)
4. Important grading notes for partial credit

Format your response as JSON:
{
  "correct_answer": "string (the final correct answer)",
  "solution_steps": [
    "..."
  ],
  "alternative_answers": ["string (alternative acceptable answers)"],
  "explanation": "string (detailed explanation of the solution approach)",
  "grading_notes": "string (important notes for graders about partial credit)"
}

Guidelines:
- Show ALL steps needed to reach the solution
- Include mathematical notation and formulas where relevant
- Consider multiple valid solution approaches
- Provide clear grading guidance for partial credit
- Keep grading notes concise and focused on what students should demonstrate


"""

SINGLE_RUBRIC_GENERATION_PROMPT = """
You are an expert educator creating detailed grading rubrics.

For the given question and answer key, create a comprehensive grading rubric that:
1. Breaks down the total points into specific criteria
2. Provides clear descriptions for each point value
3. Includes examples of student work at different levels
4. Considers partial credit opportunities

Format your response as JSON:
{
  "total_points": number,
  "criteria": [
    {
      "points": number,
      "description": "string (what earns these points)",
      "examples": ["string (examples of work that would earn these points)"]
    }
  ]
}

CRITICAL REQUIREMENT:
- The sum of ALL criteria points MUST EXACTLY EQUAL the total_points value
- Verify that criteria[0].points + criteria[1].points + ... = total_points
- Do NOT create criteria that sum to more or less than the question's point value

Grading principles:
- Be fair and consistent
- Provide clear, objective criteria
- Allow for partial credit where appropriate
- Consider different valid approaches to the problem

Example: For a 4-point question, if you have 3 items worth 1.5, 1.5, and 1.0 points, they sum to 4.0. ✓
Example: For a 4-point question, if you have items worth 1, 2, and 2 points, they sum to 5.0. ✗ WRONG - Must equal 4.0
"""

GRADING_NOTES_FORMAT_PROMPT = """
You are an expert grader preparing structured guidance for other graders. Rewrite the raw grading notes into STRICT XML using the exact structure shown below:

<gradingNotes>
  <partialCreditRules>
    <item>
      <description>string</description>
      <points>number</points>
    </item>
  </partialCreditRules>
</gradingNotes>

CRITICAL REQUIREMENTS:
- The points in ALL <item> elements MUST sum EXACTLY to the question's total points
- Each <item> must have a numeric <points> value (cannot be empty)
- The sum of all points values across all items must equal the question's point value
- Respond with XML ONLY. Do not add markdown fences or commentary.
- Within the container, output one <item> per partial credit criterion
- Keep descriptions concise but clear
- Sometimes the grade notes will state the full points first, dont count that!
"""


def get_answer_key_generation_prompt(
    question,
    general_instructions: str = "",
    question_instructions: str = ""
) -> str:
    """Generate prompt for answer key creation with user instructions."""
    base_prompt = ANSWER_KEY_GENERATION_PROMPT

    # Add question details
    question_info = f"""
Question {question.question_id} ({question.points} points):
{question.question_text}

Question Type: {question.question_type.value}
"""

    # Add user instructions if provided
    instructions_section = ""
    if general_instructions:
        instructions_section += f"\nGeneral Grading Instructions:\n{general_instructions}\n"

    if question_instructions:
        instructions_section += f"\nSpecific Instructions for this Question:\n{question_instructions}\n"

    return f"{base_prompt}\n{question_info}{instructions_section}"


def get_single_rubric_generation_prompt(
    question,
    answer_key = None,
    general_instructions: str = "",
    question_instructions: str = ""
) -> str:
    """Generate prompt for single rubric creation with user instructions."""
    base_prompt = SINGLE_RUBRIC_GENERATION_PROMPT

    # Add question details
    question_info = f"""
Question {question.question_id} ({question.points} points):
{question.question_text}

Question Type: {question.question_type.value}
"""

    # Add answer key if available
    answer_key_section = ""
    if answer_key:
        answer_key_section = f"""
Correct Answer: {answer_key.correct_answer}

Solution Steps:
{chr(10).join(f"- {step}" for step in getattr(answer_key, 'solution_steps', []))}

Grading Notes: {answer_key.grading_notes}
"""

    # Add user instructions if provided
    instructions_section = ""
    if general_instructions:
        instructions_section += f"\nGeneral Grading Instructions:\n{general_instructions}\n"

    if question_instructions:
        instructions_section += f"\nSpecific Instructions for this Question:\n{question_instructions}\n"

    return f"{base_prompt}\n{question_info}{answer_key_section}{instructions_section}"


def get_grading_notes_format_prompt(question, grading_notes: str) -> str:
    """Generate prompt instructing the LLM to normalize grading notes."""
    question_info = f"""
Question {question.question_id} ({question.points} points):
{question.question_text}

Question Type: {question.question_type.value}
"""

    raw_notes = grading_notes.strip()

    return (
        f"{GRADING_NOTES_FORMAT_PROMPT.strip()}\n\n"
        f"{question_info}\n"
        f"Raw grading notes:\n{raw_notes}"
    )
