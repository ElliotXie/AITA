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


RUBRIC_ADJUSTMENT_IDENTIFICATION_PROMPT = """
You are an expert grading assistant helping to parse natural language instructions for adjusting exam rubrics.

Your task is to analyze user instructions and identify:
1. Which specific questions they want to modify
2. What type of adjustment they want to make
3. The level of specification provided by the user

ADJUSTMENT TYPE CLASSIFICATION:
- **intelligent_replacement**: User provides criteria (with or without points) that should intelligently replace/complete the rubric
- **add_criterion**: User wants to add specific criteria to existing rubric
- **modify_points**: User wants to change point distribution
- **clarify_description**: User wants to improve descriptions without structural changes

SPECIFICATION LEVEL DETECTION:
- **complete_with_points**: User provides all criteria with specific point values
- **partial_with_points**: User provides some criteria with points, others missing
- **conceptual_only**: User mentions criteria concepts without specific points
- **point_mismatch**: User's specified points don't match the original question total

Parse the user's adjustment instructions and return a structured JSON response:

Format your response as JSON:
{
  "adjustments": [
    {
      "target_questions": ["question_id1", "question_id2"],
      "adjustment_type": "string (intelligent_replacement, add_criterion, modify_points, clarify_description)",
      "specification_level": "string (complete_with_points, partial_with_points, conceptual_only, point_mismatch)",
      "description": "string (clear description of what to change)",
      "user_criteria": [
        {
          "description": "string (user's description, however brief)",
          "points": number or null,
          "priority": "high/medium/low"
        }
      ],
      "total_points_specified": number or null
    }
  ]
}

Guidelines:
- Be specific about which questions are referenced
- Use EXACT question IDs from the available questions list (e.g., "1a", "1b", "2", "5", NOT "Question 1a")
- If a question is mentioned as "question 1a" or "Q1a", map it to just "1a"
- If instructions are general (e.g., "all calculation questions"), identify all matching questions
- When user lists multiple criteria with points for a question, use adjustment_type: "complete_replacement"
- Break down complex instructions into multiple specific adjustments
- Preserve the intent and specificity of the user's instructions
- CRITICAL: target_questions must contain only the exact question_id values, not prefixed with "Question"
"""


RUBRIC_ADJUSTMENT_APPLICATION_PROMPT = """
You are an expert grading assistant creating intelligent rubric adjustments that are faithful to user input while being helpful and robust.

You will receive:
1. The current rubric for a question
2. A specific adjustment instruction with user specifications
3. Optional answer key context

POINT NOTATION FLEXIBILITY:
Accept and normalize these equivalent terms: "pts", "points", "score", "pt" (all case-insensitive)

CORE DECISION LOGIC - REPLACE vs MERGE:

**COMPLETE SPECIFICATION → REPLACE ENTIRELY**:
- If user provides criteria that sum to the original question's total points
- User has provided a complete rubric specification
- Action: Replace the entire rubric with user's criteria (expanded and enhanced)

**PARTIAL SPECIFICATION → INTELLIGENT MERGE**:
- If user provides criteria that sum to LESS than the original total points
- User has provided partial rubric guidance
- Action: Keep user criteria exactly as specified, complete the rubric by:
  - Adding complementary criteria from the original rubric (adjusted/modified as needed)
  - Ensuring total points match the original question total
  - Maintaining logical flow and avoiding duplication

**OVER-SPECIFICATION → USE USER TOTAL**:
- If user provides criteria that sum to MORE than original total
- User may be requesting a point total change
- Action: Use user's total if reasonable, otherwise discuss in output

INTELLIGENT ADJUSTMENT GUIDELINES:

1. **PRESERVE USER INTENT**: User-specified criteria are sacred and must be included exactly
2. **EXPAND CONCISE DESCRIPTIONS**: Transform brief user criteria ("normalization 1pts") into full, actionable descriptions
3. **SMART COMPLETION**: When merging, select the most relevant original criteria to complete the rubric
4. **AVOID DUPLICATION**: Don't duplicate concepts between user criteria and completed criteria

POINT DISTRIBUTION LOGIC:
- User-specified points are NEVER changed
- Remaining points (original total - user total) are distributed among completion criteria
- Each criterion must have positive points

Format your response as JSON:
{
  "question_id": "string",
  "total_points": number,
  "adjustment_strategy": "string (complete_replace|intelligent_merge|use_user_total)",
  "reasoning": "string (brief explanation of why this strategy was chosen)",
  "criteria": [
    {
      "points": number,
      "description": "string (clear, specific description)",
      "examples": ["string (examples of work that earns these points)"],
      "source": "string (user_specified|original_adapted|completion)"
    }
  ]
}

CRITICAL REQUIREMENTS:
- The sum of all criteria points MUST EXACTLY EQUAL the total_points value
- Each criterion must have positive points
- User-specified criteria must be preserved exactly and expanded, never changed
- Descriptions must be clear and actionable for graders
- Add examples that illustrate what earns full/partial credit
- Include "source" field to track origin of each criterion

DECISION EXAMPLES:
- **User provides 3pts for 3pt question**: complete_replace → Use only user criteria (expanded)
- **User provides 2pts for 5pt question**: intelligent_merge → Keep user 2pts + add 3pts from original rubric
- **User provides 6pts for 3pt question**: use_user_total → Use user's 6pt total (question total increased)
- **User says "check normalization" (no points)**: intelligent_merge → Distribute original total logically
"""


def get_rubric_adjustment_identification_prompt(
    adjustment_text: str,
    question_context: str
) -> str:
    """Generate prompt for identifying target questions from user adjustments."""
    return f"""{RUBRIC_ADJUSTMENT_IDENTIFICATION_PROMPT}

Available Questions:
{question_context}

User Adjustment Instructions:
{adjustment_text}

Please analyze the instructions and identify which questions to modify and how."""


def get_rubric_adjustment_application_prompt(
    current_rubric: dict,
    adjustment,
    answer_key = None
) -> str:
    """Generate prompt for applying specific adjustment to a rubric."""
    # Format current rubric
    rubric_text = f"""
Current Rubric for {current_rubric['question_id']} ({current_rubric['total_points']} points):
"""
    for i, criterion in enumerate(current_rubric.get('criteria', []), 1):
        rubric_text += f"\n{i}. {criterion['points']} points: {criterion['description']}"
        if criterion.get('examples'):
            rubric_text += f"\n   Examples: {', '.join(criterion['examples'])}"

    # Format adjustment with intelligent context
    adjustment_text = f"""
Adjustment to Apply:
Type: {adjustment.adjustment_type}
Specification Level: {adjustment.specification_level}
Description: {adjustment.description}
"""

    # Add user criteria information
    if hasattr(adjustment, 'user_criteria') and adjustment.user_criteria:
        adjustment_text += "\nUser-Specified Criteria:\n"
        user_total = 0
        for i, criterion in enumerate(adjustment.user_criteria, 1):
            points_text = f" ({criterion.points} points)" if criterion.points is not None else " (points not specified)"
            adjustment_text += f"{i}. {criterion.description}{points_text}\n"
            if criterion.points is not None:
                user_total += criterion.points

        if user_total > 0:
            adjustment_text += f"Total User-Specified Points: {user_total}\n"

    if hasattr(adjustment, 'total_points_specified') and adjustment.total_points_specified:
        adjustment_text += f"User's Target Total Points: {adjustment.total_points_specified}\n"

    # Legacy support
    if hasattr(adjustment, 'criteria_changes') and adjustment.criteria_changes:
        adjustment_text += "\nLegacy Criteria Changes:\n"
        for change in adjustment.criteria_changes:
            adjustment_text += f"- {change.get('action', 'modify')}: {change.get('description', '')}"
            if change.get('points'):
                adjustment_text += f" ({change['points']} points)"
            adjustment_text += "\n"

    if hasattr(adjustment, 'point_changes') and adjustment.point_changes:
        adjustment_text += f"\nLegacy Point Changes: {adjustment.point_changes}\n"

    # Add answer key context if available
    answer_key_section = ""
    if answer_key:
        answer_key_section = f"""

Answer Key Context:
Correct Answer: {getattr(answer_key, 'correct_answer', 'N/A')}
Explanation: {getattr(answer_key, 'explanation', 'N/A')}
"""

    return f"""{RUBRIC_ADJUSTMENT_APPLICATION_PROMPT}

{rubric_text}

{adjustment_text}{answer_key_section}

Please apply the adjustment to create an improved rubric."""
