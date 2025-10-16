"""
Rubric Adjustment Utilities

Handles parsing and application of natural language adjustments to existing rubrics.
Provides functions to identify target questions and merge user suggestions with current rubrics.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from aita.domain.models import Rubric, RubricCriterion, Question
from aita.services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class UserCriterion:
    """Represents a user-specified criterion."""
    description: str  # User's description (however brief)
    points: Optional[float] = None  # Specified points, if any
    priority: str = "high"  # Priority level (high/medium/low)


@dataclass
class RubricAdjustment:
    """Represents a single rubric adjustment instruction."""
    target_questions: List[str]  # Question IDs to apply adjustment to
    adjustment_type: str  # Type of adjustment (intelligent_replacement, add_criterion, etc.)
    specification_level: str  # Level of detail provided (complete_with_points, partial_with_points, etc.)
    description: str  # Human description of the adjustment
    user_criteria: List[UserCriterion] = None  # User-specified criteria
    total_points_specified: Optional[float] = None  # Total points if specified by user
    # Legacy fields for backward compatibility
    criteria_changes: List[Dict[str, Any]] = None
    point_changes: Dict[str, float] = None


@dataclass
class AdjustmentResult:
    """Result of applying rubric adjustments."""
    success: bool
    original_rubrics: List[Rubric]
    adjusted_rubrics: List[Rubric]
    applied_adjustments: List[RubricAdjustment]
    errors: List[str]
    warnings: List[str]


class RubricAdjustmentError(Exception):
    """Raised when rubric adjustment fails."""
    pass


def normalize_point_notation(text: str) -> Optional[float]:
    """
    Extract and normalize point values from various notations.

    Handles: "1 pts", "2 points", "3 score", "1.5pt", "2points", etc.

    Args:
        text: Text containing point notation

    Returns:
        Extracted point value as float, or None if not found
    """
    if not text:
        return None

    # Normalize to lowercase for case-insensitive matching
    text = text.lower().strip()

    # Pattern to match number followed by point notation
    # Matches: "1pts", "1 pts", "2 points", "1.5 score", "3pt", etc.
    pattern = r'(\d+(?:\.\d+)?)\s*(?:pts?|points?|scores?)\b'

    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    # Also try to extract standalone numbers if followed by common point indicators
    standalone_pattern = r'(\d+(?:\.\d+)?)'
    matches = re.findall(standalone_pattern, text)
    if matches and any(word in text for word in ['point', 'pt', 'score']):
        try:
            return float(matches[0])  # Take the first number found
        except ValueError:
            return None

    return None


def parse_user_adjustments(adjustment_file: Path) -> str:
    """
    Parse user adjustment instructions from file.

    Args:
        adjustment_file: Path to text file containing adjustment instructions

    Returns:
        Raw text content of the adjustment file

    Raises:
        RubricAdjustmentError: If file cannot be read
    """
    try:
        with open(adjustment_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            raise RubricAdjustmentError("Adjustment file is empty")

        logger.info(f"Loaded {len(content)} characters of adjustment instructions")
        return content

    except Exception as e:
        raise RubricAdjustmentError(f"Failed to read adjustment file: {e}") from e


def identify_target_questions(
    adjustment_text: str,
    questions: List[Question],
    llm_client: BaseLLMClient
) -> List[RubricAdjustment]:
    """
    Use LLM to identify which questions the user is referring to and parse adjustments.

    Args:
        adjustment_text: Raw user adjustment instructions
        questions: List of available questions
        llm_client: LLM client for parsing

    Returns:
        List of parsed adjustments with identified target questions

    Raises:
        RubricAdjustmentError: If parsing fails
    """
    from aita.utils.prompts import get_rubric_adjustment_identification_prompt

    try:
        # Create question context for LLM
        question_context = "\n".join([
            f"Question {q.question_id}: {q.question_text} ({q.points} points)"
            for q in questions
        ])

        prompt = get_rubric_adjustment_identification_prompt(
            adjustment_text=adjustment_text,
            question_context=question_context
        )

        from aita.services.llm.base import LLMMessage

        response = llm_client.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.1,
            max_tokens=1000
        )

        response_text = getattr(response, "content", response)
        adjustments_data = json.loads(_extract_json_from_response(response_text))

        # Convert parsed data to RubricAdjustment objects
        adjustments = []
        for adj_data in adjustments_data.get('adjustments', []):
            # Parse user criteria with point notation normalization
            user_criteria = []
            for crit_data in adj_data.get('user_criteria', []):
                description = crit_data.get('description', '')
                points = crit_data.get('points')

                # If points not explicitly provided, try to extract from description
                if points is None:
                    points = normalize_point_notation(description)

                # If we found points in description, clean them from description
                if points is not None and normalize_point_notation(description) is not None:
                    # Remove point notation from description to clean it up
                    cleaned_desc = re.sub(r'\s*\d+(?:\.\d+)?\s*(?:pts?|points?|scores?)\b', '', description, flags=re.IGNORECASE).strip()
                    description = cleaned_desc if cleaned_desc else description

                user_criteria.append(UserCriterion(
                    description=description,
                    points=points,
                    priority=crit_data.get('priority', 'high')
                ))

            adjustment = RubricAdjustment(
                target_questions=adj_data.get('target_questions', []),
                adjustment_type=adj_data.get('adjustment_type', 'intelligent_replacement'),
                specification_level=adj_data.get('specification_level', 'conceptual_only'),
                description=adj_data.get('description', ''),
                user_criteria=user_criteria,
                total_points_specified=adj_data.get('total_points_specified'),
                # Legacy fields for backward compatibility
                criteria_changes=adj_data.get('criteria_changes'),
                point_changes=adj_data.get('point_changes')
            )
            adjustments.append(adjustment)

        logger.info(f"Parsed {len(adjustments)} adjustments targeting {sum(len(a.target_questions) for a in adjustments)} questions")
        return adjustments

    except Exception as e:
        raise RubricAdjustmentError(f"Failed to parse adjustments: {e}") from e


def apply_adjustment_to_rubric(
    rubric: Rubric,
    adjustment: RubricAdjustment,
    answer_key: Optional[Any],
    llm_client: BaseLLMClient
) -> Rubric:
    """
    Apply a specific adjustment to a rubric using intelligent merge vs replace logic.

    Args:
        rubric: Original rubric to modify
        adjustment: Adjustment instruction to apply
        answer_key: Optional answer key for context
        llm_client: LLM client for generating adjustments

    Returns:
        New adjusted rubric

    Raises:
        RubricAdjustmentError: If adjustment cannot be applied
    """
    from aita.utils.prompts import get_rubric_adjustment_application_prompt

    try:
        # Calculate user-specified total points for strategy determination
        user_total_points = 0
        if adjustment.user_criteria:
            user_total_points = sum(
                c.points for c in adjustment.user_criteria
                if c.points is not None
            )

        logger.info(
            f"Applying adjustment to {rubric.question_id}: "
            f"User specified {user_total_points}/{rubric.total_points} points"
        )

        # Serialize current rubric for LLM context
        current_rubric_data = rubric.to_dict()

        prompt = get_rubric_adjustment_application_prompt(
            current_rubric=current_rubric_data,
            adjustment=adjustment,
            answer_key=answer_key
        )

        from aita.services.llm.base import LLMMessage

        response = llm_client.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.1,
            max_tokens=1500
        )

        response_text = getattr(response, "content", response)
        adjusted_data = json.loads(_extract_json_from_response(response_text))

        # Validate enhanced response format
        strategy = adjusted_data.get('adjustment_strategy', 'unknown')
        reasoning = adjusted_data.get('reasoning', 'No reasoning provided')

        logger.info(f"LLM chose strategy '{strategy}' for {rubric.question_id}: {reasoning}")

        # Create new rubric from adjusted data
        criteria = []
        for criterion_data in adjusted_data.get('criteria', []):
            criteria.append(RubricCriterion(
                points=criterion_data['points'],
                description=criterion_data['description'],
                examples=criterion_data.get('examples', [])
            ))

        adjusted_rubric = Rubric(
            question_id=rubric.question_id,
            total_points=adjusted_data.get('total_points', rubric.total_points),
            criteria=criteria
        )

        # Enhanced validation with strategy awareness
        _validate_adjusted_rubric(adjusted_rubric, rubric, strategy, user_total_points)

        logger.info(
            f"Successfully applied {strategy} adjustment to rubric {rubric.question_id} "
            f"({rubric.total_points} → {adjusted_rubric.total_points} points)"
        )
        return adjusted_rubric

    except Exception as e:
        raise RubricAdjustmentError(f"Failed to apply adjustment to {rubric.question_id}: {e}") from e


def create_backup_rubrics(rubrics: List[Rubric], backup_dir: Path) -> Path:
    """
    Create a timestamped backup of original rubrics before applying adjustments.

    Args:
        rubrics: List of rubrics to back up
        backup_dir: Directory to store backup

    Returns:
        Path to the backup file
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"rubrics_backup_{timestamp}.json"

    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_data = {
        "timestamp": timestamp,
        "rubrics": [rubric.to_dict() for rubric in rubrics]
    }

    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Created rubric backup at {backup_file}")
    return backup_file


def validate_adjustment_compatibility(
    adjustments: List[RubricAdjustment],
    rubrics: List[Rubric]
) -> Tuple[List[str], List[str]]:
    """
    Validate that adjustments are compatible with existing rubrics.

    Args:
        adjustments: List of parsed adjustments
        rubrics: List of existing rubrics

    Returns:
        Tuple of (errors, warnings) lists
    """
    errors = []
    warnings = []
    rubric_ids = {r.question_id for r in rubrics}

    for adjustment in adjustments:
        # Check if target questions exist
        for question_id in adjustment.target_questions:
            if question_id not in rubric_ids:
                errors.append(f"Target question '{question_id}' not found in existing rubrics")

        # Validate point changes if specified
        if adjustment.point_changes:
            for question_id, new_points in adjustment.point_changes.items():
                if question_id in rubric_ids:
                    current_rubric = next(r for r in rubrics if r.question_id == question_id)
                    if new_points != current_rubric.total_points:
                        warnings.append(
                            f"Point change for {question_id}: {current_rubric.total_points} → {new_points}"
                        )

    return errors, warnings


def _extract_json_from_response(response_text: str) -> str:
    """Extract JSON from LLM response, handling markdown wrapping."""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return response_text.strip()


def _validate_adjusted_rubric(
    adjusted_rubric: Rubric,
    original_rubric: Rubric,
    strategy: str = "unknown",
    user_total_points: float = 0
) -> None:
    """
    Validate that an adjusted rubric maintains structural integrity with strategy awareness.

    Args:
        adjusted_rubric: The adjusted rubric to validate
        original_rubric: The original rubric for comparison
        strategy: The adjustment strategy used (complete_replace|intelligent_merge|use_user_total)
        user_total_points: Points specified by the user

    Raises:
        RubricAdjustmentError: If validation fails
    """
    # Check that criteria points sum to total
    criteria_sum = sum(c.points for c in adjusted_rubric.criteria)
    if abs(criteria_sum - adjusted_rubric.total_points) > 0.01:
        raise RubricAdjustmentError(
            f"Criteria points ({criteria_sum}) do not sum to total points ({adjusted_rubric.total_points})"
        )

    # Check that all criteria have positive points
    for criterion in adjusted_rubric.criteria:
        if criterion.points <= 0:
            raise RubricAdjustmentError(f"Criterion has non-positive points: {criterion.points}")

    # Check that rubric has at least one criterion
    if not adjusted_rubric.criteria:
        raise RubricAdjustmentError("Adjusted rubric has no criteria")

    # Strategy-specific validation
    if strategy == "complete_replace":
        # For complete replacement, expect total points to match original (unless user specified otherwise)
        if user_total_points > 0 and abs(adjusted_rubric.total_points - user_total_points) > 0.01:
            logger.warning(
                f"Complete replacement but total points don't match user specification: "
                f"user={user_total_points}, result={adjusted_rubric.total_points}"
            )

    elif strategy == "intelligent_merge":
        # For intelligent merge, total should usually match original
        if abs(adjusted_rubric.total_points - original_rubric.total_points) > 0.01:
            logger.warning(
                f"Intelligent merge changed total points for {adjusted_rubric.question_id}: "
                f"{original_rubric.total_points} → {adjusted_rubric.total_points}"
            )

    elif strategy == "use_user_total":
        # For user total, expect it to match user specification
        if user_total_points > 0 and abs(adjusted_rubric.total_points - user_total_points) > 0.01:
            logger.warning(
                f"use_user_total strategy but result doesn't match user total: "
                f"user={user_total_points}, result={adjusted_rubric.total_points}"
            )

    # General warning for significant point changes
    if abs(adjusted_rubric.total_points - original_rubric.total_points) > 0.01:
        logger.info(
            f"Total points changed for {adjusted_rubric.question_id} using {strategy}: "
            f"{original_rubric.total_points} → {adjusted_rubric.total_points}"
        )