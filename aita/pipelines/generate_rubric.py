"""
Rubric and Answer Key Generation Pipeline

Generates comprehensive grading rubrics and detailed answer keys using LLM.
Supports user-provided rubrics and custom grading instructions.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from aita.domain.models import ExamSpec, Question, AnswerKey, Rubric, RubricCriterion
from aita.domain.exam_reconstruct import ExamReconstructor
from aita.services.llm.base import BaseLLMClient
from aita.utils.prompts import (
    get_answer_key_generation_prompt,
    get_single_rubric_generation_prompt,
    get_grading_notes_format_prompt
)
from aita.utils.parallel_executor import ParallelLLMExecutor
from aita.utils.llm_task_implementations import (
    create_answer_key_tasks,
    create_rubric_tasks,
    AnswerKeyGenerationTask,
    RubricGenerationTask
)

logger = logging.getLogger(__name__)
console = Console()


def _extract_json_from_response(response_text: str) -> str:
    """Extract JSON from LLM response, handling markdown wrapping."""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return response_text.strip()


def _extract_xml_from_response(response_text: str) -> str:
    """Extract XML from LLM response, handling markdown wrapping."""
    xml_match = re.search(
        r'```(?:xml)?\s*(<gradingNotes[\s\S]*?</gradingNotes>)\s*```',
        response_text,
        re.IGNORECASE
    )
    if xml_match:
        return xml_match.group(1)

    xml_match = re.search(r'<gradingNotes[\s\S]*?</gradingNotes>', response_text, re.IGNORECASE)
    if xml_match:
        return xml_match.group(0)

    return response_text.strip()


def _parse_optional_number(value: Optional[str]) -> Optional[float]:
    """Convert a string value to int/float when possible, otherwise return None."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _grading_notes_xml_to_dict(xml_text: str) -> Dict[str, Any]:
    """Convert grading notes XML into a dictionary matching the JSON schema."""
    root = ET.fromstring(xml_text)
    if root.tag != "gradingNotes":
        raise ValueError("Root element must be <gradingNotes>")

    data: Dict[str, Any] = {}

    data["partial_credit_rules"] = []
    partial_credit_container = root.find("partialCreditRules")
    if partial_credit_container is not None:
        for item in partial_credit_container.findall("item"):
            points = _parse_optional_number(item.findtext("points"))
            if points is None:
                logger.warning("Found partial credit item with no points value, skipping")
                continue
            data["partial_credit_rules"].append({
                "description": (item.findtext("description") or "").strip(),
                "points": points
            })

    return data


def _parse_answer_key_text(response_text: str, question_id: str) -> AnswerKey:
    """Parse raw LLM response text into an AnswerKey."""
    data = json.loads(_extract_json_from_response(response_text))
    return AnswerKey(
        question_id=question_id,
        correct_answer=data.get('correct_answer', ''),
        alternative_answers=data.get('alternative_answers', []),
        explanation=data.get('explanation', ''),
        grading_notes=data.get('grading_notes', ''),
        solution_steps=data.get('solution_steps', [])
    )


def _parse_rubric_text(response_text: str, question: Question) -> Rubric:
    """Parse raw LLM response text into a Rubric."""
    data = json.loads(_extract_json_from_response(response_text))
    criteria = []
    for criterion_data in data.get('criteria', []):
        criteria.append(RubricCriterion(
            points=criterion_data['points'],
            description=criterion_data['description'],
            examples=criterion_data.get('examples', [])
        ))

    return Rubric(
        question_id=question.question_id,
        total_points=data.get('total_points', question.points),
        criteria=criteria
    )


class RubricGenerationError(Exception):
    """Raised when rubric generation fails."""
    pass


class RubricGenerationPipeline:
    """
    Pipeline for generating answer keys and grading rubrics from exam specifications.

    Features:
    - Generates step-by-step answer keys using LLM
    - Creates detailed grading rubrics with point breakdowns
    - Supports user-provided rubrics and grading instructions
    - Saves results to intermediate and final directories
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        data_dir: Path,
        intermediate_dir: Path,
        max_parse_retries: int = 2,
        parallel_executor: Optional[ParallelLLMExecutor] = None
    ):
        """
        Initialize the rubric generation pipeline.

        Args:
            llm_client: LLM client for generating rubrics and answer keys
            data_dir: Base data directory
            intermediate_dir: Intermediate products directory
            max_parse_retries: Maximum retries for JSON parsing failures (deprecated, use executor)
            parallel_executor: Optional parallel executor (will create one if not provided)
        """
        self.llm_client = llm_client
        self.data_dir = Path(data_dir)
        self.intermediate_dir = Path(intermediate_dir)
        self.max_parse_retries = max_parse_retries  # Kept for backward compatibility
        self.reconstructor = ExamReconstructor(data_dir)

        # Setup parallel executor
        if parallel_executor:
            self.executor = parallel_executor
        else:
            # Create executor from config
            from aita.config import get_config
            config = get_config()
            self.executor = ParallelLLMExecutor(
                llm_client=llm_client,
                max_workers=config.parallel_execution.max_workers,
                rate_limit=config.parallel_execution.rate_limit_rps,
                enable_checkpointing=config.parallel_execution.enable_checkpointing,
                checkpoint_interval=config.parallel_execution.checkpoint_interval,
                max_retries=max_parse_retries,
                show_progress=config.parallel_execution.show_progress
            )

        # Ensure directories exist
        self.rubrics_dir = self.intermediate_dir / "rubrics"
        self.rubrics_dir.mkdir(parents=True, exist_ok=True)

        logger.info("RubricGenerationPipeline initialized with parallel executor")

    def generate_from_exam_spec(
        self,
        exam_spec: ExamSpec,
        assignment_name: str,
        user_rubrics_file: Optional[Path] = None,
        instructions_file: Optional[Path] = None,
        force_regenerate: bool = False
    ) -> Tuple[List[AnswerKey], List[Rubric]]:
        """
        Generate answer keys and rubrics from exam specification.

        Args:
            exam_spec: ExamSpec object with questions
            assignment_name: Name of assignment for file organization
            user_rubrics_file: Optional path to user-provided rubrics JSON
            instructions_file: Optional path to grading instructions text file
            force_regenerate: Whether to regenerate existing files

        Returns:
            Tuple of (answer_keys, rubrics) lists

        Raises:
            RubricGenerationError: If generation fails
        """
        console.print(f"\n📋 [bold cyan]Rubric Generation Pipeline[/bold cyan]")
        console.print(f"   Assignment: {assignment_name}")
        console.print(f"   Questions: {len(exam_spec.questions)}\n")

        try:
            # Step 1: Check if already exists (unless forcing)
            if not force_regenerate and self._check_existing_files():
                console.print("   ✅ Rubrics already exist (use --force to regenerate)")
                return self._load_existing_files()

            # Step 2: Load user inputs
            user_inputs = self._load_user_inputs(user_rubrics_file, instructions_file)
            console.print(f"   📂 Loaded user inputs: {len(user_inputs['rubrics'])} rubrics, {len(user_inputs['instructions'])} instructions")

            # Step 3: Generate answer keys
            raw_answer_keys, answer_keys = self._generate_answer_keys(exam_spec, user_inputs, force_regenerate)
            console.print(f"   ✅ Generated {len(answer_keys)} answer keys")

            # Step 4: Generate rubrics
            rubrics = self._generate_rubrics(exam_spec, user_inputs, answer_keys, force_regenerate)
            console.print(f"   ✅ Generated {len(rubrics)} rubrics")

            # Step 5: Save results
            self._save_results(raw_answer_keys, answer_keys, rubrics)
            console.print(f"   💾 Saved to intermediate and results directories")

            # Success summary
            total_points = sum(r.total_points for r in rubrics)
            console.print(f"\n✅ [bold green]Generation Complete![/bold green]")
            console.print(f"   📝 Answer Keys: {len(answer_keys)}")
            console.print(f"   📊 Rubrics: {len(rubrics)}")
            console.print(f"   💯 Total Points: {total_points}\n")

            return answer_keys, rubrics

        except Exception as e:
            logger.error(f"Rubric generation failed: {e}", exc_info=True)
            raise RubricGenerationError(f"Generation failed: {e}") from e

    def _check_existing_files(self) -> bool:
        """Check if rubric files already exist."""
        raw_answer_key_file = self.rubrics_dir / "generated_answer_keys_raw.json"
        answer_key_file = self.rubrics_dir / "generated_answer_keys.json"
        rubrics_file = self.rubrics_dir / "generated_rubrics.json"
        return raw_answer_key_file.exists() and answer_key_file.exists() and rubrics_file.exists()

    def _load_existing_files(self) -> Tuple[List[AnswerKey], List[Rubric]]:
        """Load existing answer keys and rubrics."""
        answer_keys = self.reconstructor.load_answer_keys()
        rubrics = self.reconstructor.load_rubrics()
        return answer_keys, rubrics

    def _load_user_inputs(
        self,
        user_rubrics_file: Optional[Path],
        instructions_file: Optional[Path]
    ) -> Dict[str, Any]:
        """
        Load user-provided rubrics and instructions.

        Returns:
            Dict with 'rubrics', 'instructions', and 'question_instructions' keys
        """
        user_inputs = {
            'rubrics': {},
            'instructions': "",
            'question_instructions': {}
        }

        # Load user rubrics
        if user_rubrics_file and user_rubrics_file.exists():
            try:
                with open(user_rubrics_file, 'r', encoding='utf-8') as f:
                    user_inputs['rubrics'] = json.load(f)
                logger.info(f"Loaded user rubrics from {user_rubrics_file}")
            except Exception as e:
                logger.warning(f"Failed to load user rubrics: {e}")

        # Try default location if not provided
        elif not user_rubrics_file:
            default_rubrics = self.rubrics_dir / "user_rubrics.json"
            if default_rubrics.exists():
                try:
                    with open(default_rubrics, 'r', encoding='utf-8') as f:
                        user_inputs['rubrics'] = json.load(f)
                    logger.info(f"Loaded user rubrics from default location")
                except Exception as e:
                    logger.warning(f"Failed to load default rubrics: {e}")

        # Load general instructions
        if instructions_file and instructions_file.exists():
            try:
                with open(instructions_file, 'r', encoding='utf-8') as f:
                    user_inputs['instructions'] = f.read().strip()
                logger.info(f"Loaded instructions from {instructions_file}")
            except Exception as e:
                logger.warning(f"Failed to load instructions: {e}")

        # Try default location if not provided
        elif not instructions_file:
            default_instructions = self.rubrics_dir / "rubric_instructions.txt"
            if default_instructions.exists():
                try:
                    with open(default_instructions, 'r', encoding='utf-8') as f:
                        user_inputs['instructions'] = f.read().strip()
                    logger.info(f"Loaded instructions from default location")
                except Exception as e:
                    logger.warning(f"Failed to load default instructions: {e}")

        # Load question-specific instructions
        question_instructions_file = self.rubrics_dir / "question_specific_instructions.json"
        if question_instructions_file.exists():
            try:
                with open(question_instructions_file, 'r', encoding='utf-8') as f:
                    user_inputs['question_instructions'] = json.load(f)
                logger.info(f"Loaded question-specific instructions")
            except Exception as e:
                logger.warning(f"Failed to load question instructions: {e}")

        return user_inputs

    def _generate_answer_keys(
        self,
        exam_spec: ExamSpec,
        user_inputs: Dict[str, Any],
        force_regenerate: bool = False
    ) -> Tuple[List[AnswerKey], List[AnswerKey]]:
        """Generate answer keys for all questions using parallel executor.

        Args:
            exam_spec: Exam specification with questions
            user_inputs: User-provided instructions and rubrics
            force_regenerate: Whether to ignore existing checkpoints

        Returns:
            Tuple of (raw_answer_keys, formatted_answer_keys)
        """
        # Create tasks for all questions
        tasks = create_answer_key_tasks(
            questions=exam_spec.questions,
            general_instructions=user_inputs['instructions'],
            question_instructions=user_inputs['question_instructions']
        )

        # Execute in parallel (skip checkpoint resume if forcing regeneration)
        result = self.executor.execute_batch(
            tasks=tasks,
            checkpoint_name=f"answer_keys_{len(tasks)}_questions",
            resume_from_checkpoint=not force_regenerate
        )

        # Convert task results to answer keys
        raw_answer_keys: List[AnswerKey] = []
        for task_result in result.task_results:
            if task_result.success:
                answer = task_result.result
                if isinstance(answer, dict):
                    answer = AnswerKey.from_dict(answer)
                raw_answer_keys.append(answer)
            else:
                # Create fallback answer key for failed tasks
                question_id = task_result.task_id.replace("answer_key_", "")
                answer_key = AnswerKey(
                    question_id=question_id,
                    correct_answer="[Answer key generation failed - manual review required]",
                    explanation="Failed to generate automatically",
                    grading_notes=f"Error: {task_result.error}"
                )
                raw_answer_keys.append(answer_key)
                logger.error(f"Failed to generate answer key for {question_id}: {task_result.error}")

        # Normalize grading notes for downstream validation
        formatted_answer_keys = self._refine_grading_notes(
            exam_spec,
            [AnswerKey.from_dict(ak.to_dict()) for ak in raw_answer_keys]
        )

        # Sort by question order
        question_order = {q.question_id: i for i, q in enumerate(exam_spec.questions)}
        raw_answer_keys.sort(key=lambda ak: question_order.get(ak.question_id, 999))
        formatted_answer_keys.sort(key=lambda ak: question_order.get(ak.question_id, 999))

        return raw_answer_keys, formatted_answer_keys

    def _generate_single_answer_key(
        self,
        question: Question,
        user_inputs: Dict[str, Any]
    ) -> AnswerKey:
        """Generate answer key for a single question."""
        question_instructions = user_inputs['question_instructions'].get(question.question_id, "")
        general_instructions = user_inputs['instructions']

        return generate_answer_for_question(
            llm_client=self.llm_client,
            question=question,
            general_instructions=general_instructions,
            question_instructions=question_instructions,
            max_retries=self.max_parse_retries
        )

    def _parse_answer_key_response(self, response_text: str, question_id: str) -> AnswerKey:
        """Parse LLM JSON response into AnswerKey object."""
        return _parse_answer_key_text(response_text, question_id)

    def _refine_grading_notes(
        self,
        exam_spec: ExamSpec,
        answer_keys: List[AnswerKey]
    ) -> List[AnswerKey]:
        """Run grading notes through a formatting pass to ensure consistent structure."""
        question_lookup = {q.question_id: q for q in exam_spec.questions}
        refined: List[AnswerKey] = []

        for answer_key in answer_keys:
            notes = answer_key.grading_notes or ""
            if not notes.strip():
                refined.append(answer_key)
                continue

            if notes.strip().startswith("{"):
                refined.append(answer_key)
                continue

            if "[Answer key generation failed" in notes:
                refined.append(answer_key)
                continue

            question = question_lookup.get(answer_key.question_id)
            if not question:
                refined.append(answer_key)
                continue

            try:
                formatted_notes = format_grading_notes(
                    llm_client=self.llm_client,
                    question=question,
                    grading_notes=notes,
                    max_retries=self.max_parse_retries
                )
                if formatted_notes:
                    answer_key.grading_notes = formatted_notes
            except Exception as exc:
                logger.warning(
                    f"Failed to format grading notes for {answer_key.question_id}: {exc}"
                )

            refined.append(answer_key)

        return refined

    def _generate_rubrics(
        self,
        exam_spec: ExamSpec,
        user_inputs: Dict[str, Any],
        answer_keys: List[AnswerKey],
        force_regenerate: bool = False
    ) -> List[Rubric]:
        """Generate rubrics for all questions using parallel executor.

        Args:
            exam_spec: Exam specification with questions
            user_inputs: User-provided instructions and rubrics
            answer_keys: Generated answer keys
            force_regenerate: Whether to ignore existing checkpoints
        """
        rubrics = []
        answer_key_dict = {ak.question_id: ak for ak in answer_keys}

        # Separate user-provided rubrics from questions that need generation
        questions_to_generate = []
        for question in exam_spec.questions:
            if question.question_id in user_inputs['rubrics']:
                # Use user-provided rubric
                rubric = self._create_rubric_from_user_input(
                    question,
                    user_inputs['rubrics'][question.question_id]
                )
                rubrics.append(rubric)
                logger.info(f"Using user-provided rubric for {question.question_id}")
            else:
                questions_to_generate.append(question)

        # Generate rubrics for remaining questions in parallel
        if questions_to_generate:
            tasks = create_rubric_tasks(
                questions=questions_to_generate,
                answer_keys=answer_key_dict,
                general_instructions=user_inputs['instructions'],
                question_instructions=user_inputs['question_instructions']
            )

            # Execute in parallel (skip checkpoint resume if forcing regeneration)
            result = self.executor.execute_batch(
                tasks=tasks,
                checkpoint_name=f"rubrics_{len(tasks)}_questions",
                resume_from_checkpoint=not force_regenerate
            )

            # Convert task results to rubrics
            for task_result in result.task_results:
                if task_result.success:
                    rubric = task_result.result
                    if isinstance(rubric, dict):
                        rubric = Rubric.from_dict(rubric)
                    rubrics.append(rubric)
                else:
                    # Create fallback rubric for failed tasks
                    question_id = task_result.task_id.replace("rubric_", "")
                    question = next((q for q in questions_to_generate if q.question_id == question_id), None)
                    if question:
                        rubric = Rubric(
                            question_id=question_id,
                            total_points=question.points,
                            criteria=[
                                RubricCriterion(
                                    points=question.points,
                                    description=f"[Rubric generation failed - manual review required] Error: {task_result.error}"
                                )
                            ]
                        )
                        rubrics.append(rubric)
                        logger.error(f"Failed to generate rubric for {question_id}: {task_result.error}")

        # Sort by question order
        question_order = {q.question_id: i for i, q in enumerate(exam_spec.questions)}
        rubrics.sort(key=lambda r: question_order.get(r.question_id, 999))

        return rubrics

    def _create_rubric_from_user_input(self, question: Question, user_rubric: Dict[str, Any]) -> Rubric:
        """Create Rubric object from user-provided data."""
        criteria = []
        for criterion_data in user_rubric.get('criteria', []):
            criteria.append(RubricCriterion(
                points=criterion_data['points'],
                description=criterion_data['description'],
                examples=criterion_data.get('examples', [])
            ))

        return Rubric(
            question_id=question.question_id,
            total_points=user_rubric.get('total_points', question.points),
            criteria=criteria
        )

    def _generate_single_rubric(
        self,
        question: Question,
        user_inputs: Dict[str, Any],
        answer_key: Optional[AnswerKey]
    ) -> Rubric:
        """Generate rubric for a single question."""
        question_instructions = user_inputs['question_instructions'].get(question.question_id, "")
        general_instructions = user_inputs['instructions']

        return generate_rubric_for_question(
            llm_client=self.llm_client,
            question=question,
            answer_key=answer_key,
            general_instructions=general_instructions,
            question_instructions=question_instructions,
            max_retries=self.max_parse_retries
        )

    def _parse_rubric_response(self, response_text: str, question: Question) -> Rubric:
        """Parse LLM JSON response into Rubric object."""
        return _parse_rubric_text(response_text, question)

    def _save_results(
        self,
        raw_answer_keys: List[AnswerKey],
        answer_keys: List[AnswerKey],
        rubrics: List[Rubric]
    ) -> None:
        """Save raw and formatted answer keys plus rubrics to intermediate and final directories."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("💾 Saving results...", total=None)

            # Save to intermediate directory
            intermediate_raw_answer_keys = self.rubrics_dir / "generated_answer_keys_raw.json"
            intermediate_answer_keys = self.rubrics_dir / "generated_answer_keys.json"
            intermediate_rubrics = self.rubrics_dir / "generated_rubrics.json"

            # Save raw answer keys to intermediate
            with open(intermediate_raw_answer_keys, 'w', encoding='utf-8') as f:
                data = {"answer_keys": [ak.to_dict() for ak in raw_answer_keys]}
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Save answer keys to intermediate
            with open(intermediate_answer_keys, 'w', encoding='utf-8') as f:
                data = {"answer_keys": [ak.to_dict() for ak in answer_keys]}
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Save rubrics to intermediate
            with open(intermediate_rubrics, 'w', encoding='utf-8') as f:
                data = {"rubrics": [r.to_dict() for r in rubrics]}
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Save to final results directory using ExamReconstructor
            self.reconstructor.save_answer_keys(answer_keys)
            self.reconstructor.save_rubrics(rubrics)

            # Save raw answer keys to results for auditing
            raw_results_file = self.reconstructor.results_dir / "answer_key_raw.json"
            with open(raw_results_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {"answer_keys": [ak.to_dict() for ak in raw_answer_keys]},
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            progress.update(task, completed=True)
            logger.info(f"Saved {len(answer_keys)} answer keys and {len(rubrics)} rubrics")

    def adjust_existing_rubrics(
        self,
        exam_spec: ExamSpec,
        assignment_name: str,
        adjustment_file: Path,
        force_regenerate: bool = False
    ) -> Tuple[List[AnswerKey], List[Rubric]]:
        """
        Apply natural language adjustments to existing rubrics.

        Args:
            exam_spec: ExamSpec object with questions
            assignment_name: Name of assignment for file organization
            adjustment_file: Path to text file containing adjustment instructions
            force_regenerate: Whether to ignore existing checkpoints

        Returns:
            Tuple of (answer_keys, adjusted_rubrics) lists

        Raises:
            RubricGenerationError: If adjustment fails
        """
        from aita.utils.rubric_adjustment import (
            parse_user_adjustments,
            identify_target_questions,
            create_backup_rubrics,
            validate_adjustment_compatibility,
            RubricAdjustmentError
        )
        from aita.utils.llm_task_implementations import create_rubric_adjustment_tasks

        console.print(f"\n🔧 [bold cyan]Rubric Adjustment Pipeline[/bold cyan]")
        console.print(f"   Assignment: {assignment_name}")
        console.print(f"   Adjustment File: {adjustment_file}")
        console.print(f"   Questions: {len(exam_spec.questions)}\n")

        try:
            # Step 1: Check if existing rubrics exist
            if not self._check_existing_files():
                raise RubricGenerationError(
                    "No existing rubrics found. Please run 'aita generate-rubric' first."
                )

            # Step 2: Load existing rubrics and answer keys
            answer_keys, existing_rubrics = self._load_existing_files()
            console.print(f"   📂 Loaded {len(existing_rubrics)} existing rubrics")

            # Step 3: Parse user adjustment instructions
            adjustment_text = parse_user_adjustments(adjustment_file)
            console.print(f"   📝 Parsed adjustment instructions ({len(adjustment_text)} characters)")

            # Step 4: Identify target questions using LLM
            adjustments = identify_target_questions(
                adjustment_text=adjustment_text,
                questions=exam_spec.questions,
                llm_client=self.llm_client
            )
            console.print(f"   🎯 Identified {len(adjustments)} adjustments")

            # Step 5: Validate adjustment compatibility
            errors, warnings = validate_adjustment_compatibility(adjustments, existing_rubrics)
            if errors:
                error_msg = "\n".join(f"  - {error}" for error in errors)
                raise RubricGenerationError(f"Adjustment validation failed:\n{error_msg}")

            if warnings:
                console.print("[yellow]⚠️  Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"   {warning}")

            # Step 6: Create backup of original rubrics
            backup_dir = self.rubrics_dir / "adjustment_history"
            backup_file = create_backup_rubrics(existing_rubrics, backup_dir)
            console.print(f"   💾 Created backup: {backup_file.name}")

            # Step 7: Apply adjustments using parallel executor
            answer_key_dict = {ak.question_id: ak for ak in answer_keys}
            tasks = create_rubric_adjustment_tasks(
                rubrics=existing_rubrics,
                adjustments=adjustments,
                answer_keys=answer_key_dict
            )

            result = self.executor.execute_batch(
                tasks=tasks,
                checkpoint_name=f"rubric_adjustments_{len(tasks)}_tasks",
                resume_from_checkpoint=not force_regenerate
            )

            # Step 8: Collect adjusted rubrics
            adjusted_rubrics = list(existing_rubrics)  # Start with originals
            rubric_dict = {r.question_id: r for r in adjusted_rubrics}

            successful_adjustments = 0
            for task_result in result.task_results:
                if task_result.success:
                    adjusted_rubric = task_result.result
                    if isinstance(adjusted_rubric, dict):
                        adjusted_rubric = Rubric.from_dict(adjusted_rubric)

                    # Replace original with adjusted version
                    rubric_dict[adjusted_rubric.question_id] = adjusted_rubric
                    successful_adjustments += 1
                else:
                    logger.error(f"Failed to adjust rubric: {task_result.error}")

            adjusted_rubrics = list(rubric_dict.values())
            console.print(f"   ✅ Successfully adjusted {successful_adjustments}/{len(tasks)} rubrics")

            # Step 9: Save adjusted results
            self._save_results(answer_keys, answer_keys, adjusted_rubrics)
            console.print(f"   💾 Saved adjusted rubrics")

            # Step 10: Save adjustment log
            self._save_adjustment_log(adjustments, adjustment_file, backup_file)

            # Success summary
            total_points = sum(r.total_points for r in adjusted_rubrics)
            console.print(f"\n✅ [bold green]Adjustment Complete![/bold green]")
            console.print(f"   🔧 Applied Adjustments: {len(adjustments)}")
            console.print(f"   📊 Adjusted Rubrics: {successful_adjustments}")
            console.print(f"   💯 Total Points: {total_points}")
            console.print(f"   📁 Backup: {backup_file}\n")

            return answer_keys, adjusted_rubrics

        except RubricAdjustmentError as e:
            logger.error(f"Rubric adjustment failed: {e}", exc_info=True)
            raise RubricGenerationError(f"Adjustment failed: {e}") from e
        except Exception as e:
            logger.error(f"Rubric adjustment failed: {e}", exc_info=True)
            raise RubricGenerationError(f"Adjustment failed: {e}") from e

    def _save_adjustment_log(
        self,
        adjustments: List,
        adjustment_file: Path,
        backup_file: Path
    ) -> None:
        """Save a log of the adjustments applied."""
        from datetime import datetime
        import json

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "adjustment_file": str(adjustment_file),
            "backup_file": str(backup_file),
            "adjustments": [
                {
                    "target_questions": adj.target_questions,
                    "adjustment_type": adj.adjustment_type,
                    "description": adj.description
                }
                for adj in adjustments
            ]
        }

        log_file = self.rubrics_dir / "adjustment_history" / f"adjustment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved adjustment log to {log_file}")


# Factory and convenience functions


def generate_answer_for_question(
    llm_client: BaseLLMClient,
    question: Question,
    general_instructions: str = "",
    question_instructions: str = "",
    max_retries: int = 2
) -> AnswerKey:
    """
    Generate an answer key for a single question.

    Args:
        llm_client: LLM client used for completion
        question: Question to answer
        general_instructions: General grading instructions
        question_instructions: Question-specific instructions
        max_retries: Maximum number of JSON parse retries
    """
    base_prompt = get_answer_key_generation_prompt(
        question=question,
        general_instructions=general_instructions,
        question_instructions=question_instructions
    )
    prompt = base_prompt
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Generating answer key for {question.question_id} (attempt {attempt + 1})"
            )
            response = llm_client.complete(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1200
            )
            response_text = getattr(response, "content", response)
            logger.debug(
                f"LLM response for {question.question_id}: {str(response_text)[:300]}..."
            )
            return _parse_answer_key_text(response_text, question.question_id)

        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                f"JSON parsing failed for {question.question_id} "
                f"(attempt {attempt + 1}): {exc}"
            )

            if attempt < max_retries - 1:
                prompt = (
                    "Your previous response was not valid JSON. "
                    "Please respond ONLY with valid JSON, no markdown formatting "
                    "or additional text.\n\n"
                    + base_prompt
                )
            continue

    raise RubricGenerationError(
        f"Failed to parse answer key for {question.question_id} after {max_retries} attempts"
    ) from last_error


def generate_rubric_for_question(
    llm_client: BaseLLMClient,
    question: Question,
    answer_key: Optional[AnswerKey],
    general_instructions: str = "",
    question_instructions: str = "",
    max_retries: int = 2
) -> Rubric:
    """
    Generate a rubric for a single question.

    Args:
        llm_client: LLM client used for completion
        question: Question to grade
        answer_key: Optional answer key to provide additional context
        general_instructions: General grading instructions
        question_instructions: Question-specific instructions
        max_retries: Maximum number of JSON parse retries
    """
    base_prompt = get_single_rubric_generation_prompt(
        question=question,
        answer_key=answer_key,
        general_instructions=general_instructions,
        question_instructions=question_instructions
    )
    prompt = base_prompt
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Generating rubric for {question.question_id} (attempt {attempt + 1})"
            )
            response = llm_client.complete(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1500
            )
            response_text = getattr(response, "content", response)
            logger.debug(
                f"LLM response for rubric {question.question_id}: {str(response_text)[:300]}..."
            )
            return _parse_rubric_text(response_text, question)

        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                f"JSON parsing failed for rubric {question.question_id} "
                f"(attempt {attempt + 1}): {exc}"
            )

            if attempt < max_retries - 1:
                prompt = (
                    "Your previous response was not valid JSON. "
                    "Please respond ONLY with valid JSON, no markdown formatting "
                    "or additional text.\n\n"
                    + base_prompt
                )
            continue

    raise RubricGenerationError(
        f"Failed to parse rubric for {question.question_id} after {max_retries} attempts"
    ) from last_error


def format_grading_notes(
    llm_client: BaseLLMClient,
    question: Question,
    grading_notes: str,
    max_retries: int = 2
) -> str:
    """
    Normalize grading notes by requesting structured XML and converting it into JSON.

    Args:
        llm_client: LLM client used for completion
        question: Question context for the notes
        grading_notes: Raw grading notes string from the primary generation step
        max_retries: Maximum attempts for obtaining valid XML output

    Returns:
        JSON-formatted string containing structured grading guidance
    """
    if not grading_notes or not grading_notes.strip():
        return grading_notes

    prompt_base = get_grading_notes_format_prompt(question, grading_notes)
    prompt = prompt_base
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response_text = llm_client.complete_text(
                prompt=prompt,
                temperature=0.0,
                max_tokens=900
            )
            logger.debug(
                f"LLM response for grading notes {question.question_id}: "
                f"{str(response_text)[:300]}..."
            )

            xml_text = _extract_xml_from_response(response_text)
            data = _grading_notes_xml_to_dict(xml_text)
            normalized = _ensure_grading_notes_structure(data)

            partial_values = [
                item["points"]
                for item in normalized["partial_credit_rules"]
                if item.get("points") is not None
            ]
            partial_sum = sum(partial_values) if partial_values else None
            normalized["partial_credit_points_sum"] = partial_sum

            question_points = getattr(question, "points", None)
            if question_points is not None:
                normalized["question_total_points"] = question_points
                if partial_sum is not None:
                    matches_total = abs(partial_sum - question_points) < 1e-6
                    normalized["partial_credit_points_match_total"] = matches_total
                    if not matches_total:
                        logger.warning(
                            "Partial credit points sum (%.2f) does not match total points "
                            "(%.2f) for %s",
                            partial_sum,
                            question_points,
                            question.question_id
                        )
                else:
                    normalized["partial_credit_points_match_total"] = None
            else:
                normalized["question_total_points"] = None
                normalized["partial_credit_points_match_total"] = None

            return json.dumps(normalized, indent=2, ensure_ascii=False)

        except (ET.ParseError, ValueError) as exc:
            last_error = exc
            logger.warning(
                f"XML parsing failed for grading notes {question.question_id} "
                f"(attempt {attempt + 1}): {exc}"
            )

            if attempt < max_retries - 1:
                prompt = (
                    "Your previous response was not valid XML. "
                    "Respond ONLY with XML that matches the specified structure.\n\n"
                    + prompt_base
                )
            continue

        except Exception as exc:
            last_error = exc
            logger.warning(
                f"Unexpected error formatting grading notes for {question.question_id}: {exc}"
            )
            break

    raise RubricGenerationError(
        f"Failed to format grading notes for {question.question_id} after {max_retries} attempts"
    ) from last_error


def _ensure_grading_notes_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all expected fields exist with the correct types."""
    structured: Dict[str, Any] = {}

    structured["partial_credit_rules"] = []
    for item in data.get("partial_credit_rules", []):
        points = item.get("points")
        if points is not None:
            structured["partial_credit_rules"].append({
                "description": str(item.get("description", "")).strip(),
                "points": points
            })

    return structured


def create_rubric_generation_pipeline(
    assignment_name: str,
    data_dir: Optional[Path] = None,
    intermediate_dir: Optional[Path] = None
) -> RubricGenerationPipeline:
    """
    Create rubric generation pipeline with services from config.

    Args:
        assignment_name: Name of assignment
        data_dir: Optional data directory (uses config default if not provided)
        intermediate_dir: Optional intermediate directory (uses config default if not provided)

    Returns:
        Configured RubricGenerationPipeline
    """
    from aita.config import get_config
    from aita.services.llm.openrouter import create_openrouter_client

    config = get_config()
    llm_client = create_openrouter_client()

    return RubricGenerationPipeline(
        llm_client=llm_client,
        data_dir=data_dir or Path(config.data_dir),
        intermediate_dir=intermediate_dir or Path(config.intermediate_dir)
    )


def generate_rubrics_for_assignment(
    assignment_name: str = "exam1",
    user_rubrics_file: Optional[str] = None,
    instructions_file: Optional[str] = None,
    useradjust_file: Optional[str] = None,
    force_regenerate: bool = False,
    data_dir: Optional[Path] = None
) -> Tuple[List[AnswerKey], List[Rubric]]:
    """
    High-level function to generate rubrics and answer keys for an assignment.

    Args:
        assignment_name: Name of assignment
        user_rubrics_file: Optional path to user-provided rubrics JSON
        instructions_file: Optional path to grading instructions text file
        useradjust_file: Optional path to natural language adjustment instructions
        force_regenerate: Whether to regenerate existing files
        data_dir: Optional data directory

    Returns:
        Tuple of (answer_keys, rubrics) lists

    Example:
        >>> answer_keys, rubrics = generate_rubrics_for_assignment(
        ...     assignment_name="BMI541_Midterm",
        ...     instructions_file="grading_instructions.txt"
        ... )
        >>> print(f"Generated {len(answer_keys)} answer keys and {len(rubrics)} rubrics")

        # Apply adjustments to existing rubrics
        >>> answer_keys, rubrics = generate_rubrics_for_assignment(
        ...     assignment_name="BMI541_Midterm",
        ...     useradjust_file="rubric_adjustments.txt"
        ... )
    """
    pipeline = create_rubric_generation_pipeline(assignment_name, data_dir)

    # Load exam spec
    exam_spec = pipeline.reconstructor.load_exam_spec()
    if not exam_spec:
        raise RubricGenerationError("No exam specification found. Run question extraction first.")

    # Check if this is an adjustment operation
    if useradjust_file:
        adjustment_path = Path(useradjust_file)
        return pipeline.adjust_existing_rubrics(
            exam_spec=exam_spec,
            assignment_name=assignment_name,
            adjustment_file=adjustment_path,
            force_regenerate=force_regenerate
        )
    else:
        # Regular generation
        user_rubrics_path = Path(user_rubrics_file) if user_rubrics_file else None
        instructions_path = Path(instructions_file) if instructions_file else None

        return pipeline.generate_from_exam_spec(
            exam_spec=exam_spec,
            assignment_name=assignment_name,
            user_rubrics_file=user_rubrics_path,
            instructions_file=instructions_path,
            force_regenerate=force_regenerate
        )
