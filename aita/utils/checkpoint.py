"""
Checkpoint System for Parallel Execution

Provides save/load functionality for execution state, allowing pipelines
to resume from the last checkpoint after failures or interruptions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ExecutionCheckpoint:
    """
    Checkpoint data for parallel execution state.

    Tracks completed tasks, failed tasks, and partial results
    to enable resume functionality.
    """

    version: str = "1.0"
    checkpoint_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    total_tasks: int = 0
    completed_task_ids: Set[str] = field(default_factory=set)
    failed_task_ids: Dict[str, str] = field(default_factory=dict)  # task_id -> error_msg
    results: Dict[str, Any] = field(default_factory=dict)  # task_id -> result_data

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure set type for completed_task_ids
        if isinstance(self.completed_task_ids, list):
            self.completed_task_ids = set(self.completed_task_ids)

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (len(self.completed_task_ids) / self.total_tasks) * 100

    @property
    def num_completed(self) -> int:
        """Number of completed tasks."""
        return len(self.completed_task_ids)

    @property
    def num_failed(self) -> int:
        """Number of failed tasks."""
        return len(self.failed_task_ids)

    @property
    def num_pending(self) -> int:
        """Number of pending tasks."""
        return self.total_tasks - self.num_completed - self.num_failed

    def is_task_completed(self, task_id: str) -> bool:
        """Check if a task is already completed."""
        return task_id in self.completed_task_ids

    def mark_completed(self, task_id: str, result: Any = None):
        """Mark a task as completed and optionally store its result."""
        self.completed_task_ids.add(task_id)
        if result is not None:
            self.results[task_id] = result
        self.updated_at = datetime.now().isoformat()

    def mark_failed(self, task_id: str, error_msg: str):
        """Mark a task as failed with error message."""
        self.failed_task_ids[task_id] = error_msg
        self.updated_at = datetime.now().isoformat()

    def get_result(self, task_id: str) -> Optional[Any]:
        """Get result for a completed task."""
        return self.results.get(task_id)

    def get_pending_task_ids(self, all_task_ids: List[str]) -> List[str]:
        """Get list of task IDs that haven't been completed or failed."""
        completed_or_failed = self.completed_task_ids | set(self.failed_task_ids.keys())
        return [tid for tid in all_task_ids if tid not in completed_or_failed]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert set to list for JSON serialization
        data['completed_task_ids'] = list(self.completed_task_ids)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionCheckpoint":
        """Create from dictionary."""
        # Convert list back to set
        if 'completed_task_ids' in data and isinstance(data['completed_task_ids'], list):
            data['completed_task_ids'] = set(data['completed_task_ids'])
        return cls(**data)


class CheckpointManager:
    """
    Manager for saving and loading execution checkpoints.

    Handles file I/O, versioning, and cleanup of checkpoint files.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files (default: ./checkpoints)
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def save(self, checkpoint: ExecutionCheckpoint, checkpoint_file: Optional[Path] = None) -> Path:
        """
        Save checkpoint to file.

        Args:
            checkpoint: ExecutionCheckpoint to save
            checkpoint_file: Optional specific file path

        Returns:
            Path to saved checkpoint file
        """
        if checkpoint_file is None:
            # Generate default filename
            safe_name = checkpoint.checkpoint_name.replace(' ', '_').replace('/', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.checkpoint_dir / f"{safe_name}_{timestamp}.checkpoint.json"
        else:
            checkpoint_file = Path(checkpoint_file)

        # Ensure parent directory exists
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        checkpoint.updated_at = datetime.now().isoformat()

        # Save to file
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(
                f"Checkpoint saved: {checkpoint_file.name} "
                f"({checkpoint.num_completed}/{checkpoint.total_tasks} completed, "
                f"{checkpoint.num_failed} failed)"
            )
            return checkpoint_file

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load(self, checkpoint_file: Path) -> ExecutionCheckpoint:
        """
        Load checkpoint from file.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            ExecutionCheckpoint object

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint format is invalid
        """
        checkpoint_file = Path(checkpoint_file)

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            checkpoint = ExecutionCheckpoint.from_dict(data)
            logger.info(
                f"Checkpoint loaded: {checkpoint_file.name} "
                f"({checkpoint.num_completed}/{checkpoint.total_tasks} completed, "
                f"{checkpoint.num_failed} failed, "
                f"{checkpoint.num_pending} pending)"
            )
            return checkpoint

        except json.JSONDecodeError as e:
            logger.error(f"Invalid checkpoint format: {e}")
            raise ValueError(f"Invalid checkpoint file format: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def find_latest_checkpoint(self, checkpoint_name: str) -> Optional[Path]:
        """
        Find the most recent checkpoint file for a given name.

        Args:
            checkpoint_name: Name of the checkpoint to search for

        Returns:
            Path to most recent checkpoint file, or None if not found
        """
        safe_name = checkpoint_name.replace(' ', '_').replace('/', '_')
        pattern = f"{safe_name}_*.checkpoint.json"

        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        if not checkpoint_files:
            return None

        # Sort by modification time, most recent first
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = checkpoint_files[0]

        logger.info(f"Found latest checkpoint: {latest.name}")
        return latest

    def cleanup(self, checkpoint_file: Path, keep_on_completion: bool = False):
        """
        Clean up checkpoint file.

        Args:
            checkpoint_file: Checkpoint file to remove
            keep_on_completion: If True, keep checkpoint even on completion
        """
        if keep_on_completion:
            logger.debug(f"Keeping checkpoint file: {checkpoint_file.name}")
            return

        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"Cleaned up checkpoint: {checkpoint_file.name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint: {e}")

    def cleanup_old_checkpoints(self, checkpoint_name: str, keep_recent: int = 5):
        """
        Clean up old checkpoint files, keeping only the most recent ones.

        Args:
            checkpoint_name: Name of checkpoints to clean
            keep_recent: Number of recent checkpoints to keep
        """
        safe_name = checkpoint_name.replace(' ', '_').replace('/', '_')
        pattern = f"{safe_name}_*.checkpoint.json"

        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        if len(checkpoint_files) <= keep_recent:
            return

        # Sort by modification time, most recent first
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Remove old checkpoints
        for old_checkpoint in checkpoint_files[keep_recent:]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint.name}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint: {e}")

        logger.info(f"Cleaned up {len(checkpoint_files) - keep_recent} old checkpoints")


def create_checkpoint(
    checkpoint_name: str,
    total_tasks: int,
    metadata: Optional[Dict[str, Any]] = None
) -> ExecutionCheckpoint:
    """
    Convenience function to create a new checkpoint.

    Args:
        checkpoint_name: Name for the checkpoint
        total_tasks: Total number of tasks to execute
        metadata: Optional metadata to store

    Returns:
        New ExecutionCheckpoint object
    """
    return ExecutionCheckpoint(
        checkpoint_name=checkpoint_name,
        total_tasks=total_tasks,
        metadata=metadata or {}
    )
