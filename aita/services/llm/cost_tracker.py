"""Cost tracking for LLM API calls."""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from threading import Lock, RLock
import traceback

from .model_pricing import get_model_pricing, calculate_cost

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Single cost entry for an LLM call."""
    timestamp: str
    session_id: str
    model: str
    operation_type: str  # name_extraction, transcription, question_extraction, grading, etc.
    input_tokens: int
    output_tokens: int
    image_count: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    image_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    stack_trace: Optional[str] = None


@dataclass
class SessionSummary:
    """Summary of costs for a session."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_images: int
    total_cost: float
    cost_by_operation: Dict[str, float]
    cost_by_model: Dict[str, float]
    entries: List[CostEntry]


class CostTracker:
    """Tracks costs for LLM API calls."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        Initialize cost tracker.

        Args:
            data_dir: Directory to save cost tracking files
            session_id: Session identifier (auto-generated if not provided)
            auto_save: Automatically save after each entry
        """
        self.data_dir = data_dir or Path("C:/Users/ellio/OneDrive - UW-Madison/AITA/intermediateproduct/cost_tracking")
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.session_file = self.data_dir / f"{self.session_id}.json"

        self.auto_save = auto_save
        self.entries: List[CostEntry] = []
        # Use reentrant lock because save()/get_summary() are called while holding the lock
        self.lock = RLock()  # Thread safety for concurrent LLM calls

        # Track session start
        self.start_time = datetime.now().isoformat()

        logger.info(f"Cost tracker initialized for session: {self.session_id}")
        logger.info(f"Cost tracking file: {self.session_file}")

    def track_call(
        self,
        model: str,
        operation_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        image_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        usage_data: Optional[Dict[str, Any]] = None
    ) -> CostEntry:
        """
        Track a single LLM API call.

        Args:
            model: Model identifier
            operation_type: Type of operation (name_extraction, transcription, etc.)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            image_count: Number of images processed
            metadata: Additional metadata to store
            usage_data: Raw usage data from API response

        Returns:
            CostEntry object
        """
        with self.lock:
            try:
                # Extract token counts from usage_data if provided
                if usage_data:
                    input_tokens = usage_data.get("prompt_tokens", input_tokens)
                    output_tokens = usage_data.get("completion_tokens", output_tokens)

                # Calculate costs
                cost_info = calculate_cost(model, input_tokens, output_tokens, image_count)

                # Create entry
                entry = CostEntry(
                    timestamp=datetime.now().isoformat(),
                    session_id=self.session_id,
                    model=model,
                    operation_type=operation_type,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    image_count=image_count,
                    input_cost=cost_info.get("input_cost", 0.0),
                    output_cost=cost_info.get("output_cost", 0.0),
                    image_cost=cost_info.get("image_cost", 0.0),
                    total_cost=cost_info.get("total_cost", 0.0),
                    currency=cost_info.get("currency", "USD"),
                    metadata=metadata or {},
                    error=cost_info.get("error")
                )

                self.entries.append(entry)

                # Log the cost
                logger.info(
                    f"Cost tracked: {operation_type} using {model} - "
                    f"${entry.total_cost:.6f} ({input_tokens} in, {output_tokens} out, {image_count} imgs)"
                )

                # Auto-save if enabled
                if self.auto_save:
                    self.save()

                return entry

            except Exception as e:
                logger.error(f"Error tracking cost: {e}")
                # Create error entry
                entry = CostEntry(
                    timestamp=datetime.now().isoformat(),
                    session_id=self.session_id,
                    model=model,
                    operation_type=operation_type,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    image_count=image_count,
                    metadata=metadata or {},
                    error=str(e),
                    stack_trace=traceback.format_exc()
                )
                self.entries.append(entry)
                return entry

    def get_summary(self) -> SessionSummary:
        """
        Get summary of the current session.

        Returns:
            SessionSummary object
        """
        with self.lock:
            cost_by_operation = {}
            cost_by_model = {}
            total_input = 0
            total_output = 0
            total_images = 0
            total_cost = 0.0

            for entry in self.entries:
                # Aggregate by operation
                if entry.operation_type not in cost_by_operation:
                    cost_by_operation[entry.operation_type] = 0.0
                cost_by_operation[entry.operation_type] += entry.total_cost

                # Aggregate by model
                if entry.model not in cost_by_model:
                    cost_by_model[entry.model] = 0.0
                cost_by_model[entry.model] += entry.total_cost

                # Totals
                total_input += entry.input_tokens
                total_output += entry.output_tokens
                total_images += entry.image_count
                total_cost += entry.total_cost

            return SessionSummary(
                session_id=self.session_id,
                start_time=self.start_time,
                end_time=datetime.now().isoformat() if self.entries else None,
                total_calls=len(self.entries),
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_images=total_images,
                total_cost=round(total_cost, 6),
                cost_by_operation={k: round(v, 6) for k, v in cost_by_operation.items()},
                cost_by_model={k: round(v, 6) for k, v in cost_by_model.items()},
                entries=self.entries
            )

    def save(self, file_path: Optional[Path] = None) -> None:
        """
        Save session data to JSON file.

        Args:
            file_path: Optional custom file path
        """
        file_path = file_path or self.session_file

        with self.lock:
            try:
                summary = self.get_summary()
                data = {
                    "session_id": summary.session_id,
                    "start_time": summary.start_time,
                    "end_time": summary.end_time,
                    "total_calls": summary.total_calls,
                    "total_input_tokens": summary.total_input_tokens,
                    "total_output_tokens": summary.total_output_tokens,
                    "total_images": summary.total_images,
                    "total_cost": summary.total_cost,
                    "cost_by_operation": summary.cost_by_operation,
                    "cost_by_model": summary.cost_by_model,
                    "entries": [asdict(entry) for entry in summary.entries]
                }

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                logger.debug(f"Cost data saved to: {file_path}")

            except Exception as e:
                logger.error(f"Failed to save cost data: {e}")

    def load(self, file_path: Optional[Path] = None) -> None:
        """
        Load session data from JSON file.

        Args:
            file_path: Optional custom file path
        """
        file_path = file_path or self.session_file

        with self.lock:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.session_id = data.get("session_id", self.session_id)
                self.start_time = data.get("start_time", self.start_time)

                self.entries = []
                for entry_data in data.get("entries", []):
                    entry = CostEntry(**entry_data)
                    self.entries.append(entry)

                logger.info(f"Loaded {len(self.entries)} cost entries from: {file_path}")

            except FileNotFoundError:
                logger.info(f"No existing cost file found: {file_path}")
            except Exception as e:
                logger.error(f"Failed to load cost data: {e}")

    def print_summary(self, detailed: bool = False) -> None:
        """
        Print a formatted summary to console.

        Args:
            detailed: Include detailed breakdown
        """
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print(f"💰 LLM Cost Summary - Session: {summary.session_id}")
        print("=" * 60)
        print(f"📅 Start: {summary.start_time}")
        print(f"📅 End:   {summary.end_time or 'In progress'}")
        print(f"📊 Total API Calls: {summary.total_calls}")
        print(f"📝 Total Tokens: {summary.total_input_tokens:,} in / {summary.total_output_tokens:,} out")
        if summary.total_images > 0:
            print(f"🖼️  Total Images: {summary.total_images}")
        print(f"💵 Total Cost: ${summary.total_cost:.6f} USD")

        if summary.cost_by_operation:
            print("\n📋 Cost by Operation Type:")
            for op_type, cost in sorted(summary.cost_by_operation.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {op_type}: ${cost:.6f}")

        if summary.cost_by_model:
            print("\n🤖 Cost by Model:")
            for model, cost in sorted(summary.cost_by_model.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {model}: ${cost:.6f}")

        if detailed and summary.entries:
            print("\n📜 Detailed Call Log:")
            for i, entry in enumerate(summary.entries, 1):
                print(f"\n  Call {i}:")
                print(f"    Time: {entry.timestamp}")
                print(f"    Operation: {entry.operation_type}")
                print(f"    Model: {entry.model}")
                print(f"    Tokens: {entry.input_tokens} in / {entry.output_tokens} out")
                if entry.image_count > 0:
                    print(f"    Images: {entry.image_count}")
                print(f"    Cost: ${entry.total_cost:.6f}")
                if entry.error:
                    print(f"    ⚠️ Error: {entry.error}")

        print("=" * 60 + "\n")

    @staticmethod
    def infer_operation_type(context: Dict[str, Any]) -> str:
        """
        Infer operation type from context.

        Args:
            context: Context dictionary with clues about operation

        Returns:
            Inferred operation type
        """
        # Check for specific patterns in prompts or metadata
        prompt = str(context.get("prompt", "")).lower()
        metadata = context.get("metadata", {})

        # Pattern matching for operation types
        if "extract" in prompt and "name" in prompt:
            return "name_extraction"
        elif "transcribe" in prompt or "handwriting" in prompt:
            return "transcription"
        elif "question" in prompt and ("extract" in prompt or "identify" in prompt):
            return "question_extraction"
        elif "grade" in prompt or "rubric" in prompt:
            return "grading"
        elif "generate" in prompt and "rubric" in prompt:
            return "rubric_generation"

        # Check metadata
        if "operation" in metadata:
            return metadata["operation"]

        # Default
        return "general"


# Global instance for easy access
_global_tracker: Optional[CostTracker] = None
_global_lock = Lock()


def get_global_tracker() -> Optional[CostTracker]:
    """Get the global cost tracker instance."""
    return _global_tracker


def set_global_tracker(tracker: CostTracker) -> None:
    """Set the global cost tracker instance."""
    global _global_tracker
    with _global_lock:
        _global_tracker = tracker


def init_global_tracker(
    data_dir: Optional[Path] = None,
    session_id: Optional[str] = None,
    auto_save: bool = True
) -> CostTracker:
    """
    Initialize and set the global cost tracker.

    Returns:
        The initialized tracker
    """
    tracker = CostTracker(data_dir, session_id, auto_save)
    set_global_tracker(tracker)
    return tracker
