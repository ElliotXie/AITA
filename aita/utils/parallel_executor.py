"""
Parallel LLM Executor

Executes LLM tasks in parallel with rate limiting, checkpointing,
progress tracking, and comprehensive error handling.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)

from aita.services.llm.base import BaseLLMClient
from aita.utils.llm_tasks import LLMTask, TaskResult
from aita.utils.checkpoint import ExecutionCheckpoint, CheckpointManager, create_checkpoint
from aita.utils.rate_limiter import TokenBucketRateLimiter, NoOpRateLimiter

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ExecutionMetrics:
    """Metrics collected during parallel execution."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    retried_tasks: int = 0

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    total_execution_time: float = 0.0
    min_task_time: float = float('inf')
    max_task_time: float = 0.0
    avg_task_time: float = 0.0

    throughput: float = 0.0  # tasks per second

    def update_task_time(self, task_time: float):
        """Update timing metrics with a new task execution time."""
        self.min_task_time = min(self.min_task_time, task_time)
        self.max_task_time = max(self.max_task_time, task_time)

        # Update running average
        total = self.completed_tasks + self.failed_tasks
        if total > 0:
            self.avg_task_time = (
                (self.avg_task_time * (total - 1) + task_time) / total
            )

    def finalize(self):
        """Finalize metrics at end of execution."""
        self.end_time = time.time()
        self.total_execution_time = self.end_time - self.start_time

        if self.total_execution_time > 0:
            self.throughput = self.completed_tasks / self.total_execution_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "retried_tasks": self.retried_tasks,
            "total_execution_time": self.total_execution_time,
            "min_task_time": self.min_task_time if self.min_task_time != float('inf') else 0.0,
            "max_task_time": self.max_task_time,
            "avg_task_time": self.avg_task_time,
            "throughput": self.throughput
        }


@dataclass
class BatchExecutionResult:
    """Result of executing a batch of LLM tasks."""

    task_results: List[TaskResult]
    metrics: ExecutionMetrics
    checkpoint: Optional[ExecutionCheckpoint] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if not self.task_results:
            return 0.0
        successful = sum(1 for r in self.task_results if r.success)
        return (successful / len(self.task_results)) * 100

    @property
    def successful_results(self) -> List[TaskResult]:
        """Get only successful results."""
        return [r for r in self.task_results if r.success]

    @property
    def failed_results(self) -> List[TaskResult]:
        """Get only failed results."""
        return [r for r in self.task_results if not r.success]

    def get_result_by_id(self, task_id: str) -> Optional[TaskResult]:
        """Get result for a specific task ID."""
        for result in self.task_results:
            if result.task_id == task_id:
                return result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_results": [r.to_dict() for r in self.task_results],
            "metrics": self.metrics.to_dict(),
            "success_rate": self.success_rate
        }


class ParallelLLMExecutor:
    """
    Executes LLM tasks in parallel with advanced features.

    Features:
    - Parallel execution with configurable concurrency
    - Rate limiting to avoid API throttling
    - Checkpoint/resume functionality
    - Progress tracking with Rich
    - Comprehensive error handling and retries
    - Performance metrics collection
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_workers: int = 5,
        rate_limit: Optional[float] = None,
        enable_checkpointing: bool = True,
        checkpoint_interval: int = 10,
        checkpoint_dir: Optional[Path] = None,
        max_retries: int = 2,
        show_progress: bool = True
    ):
        """
        Initialize parallel executor.

        Args:
            llm_client: LLM client for making API calls
            max_workers: Maximum concurrent tasks (default: 5)
            rate_limit: Rate limit in requests/second (None = no limit)
            enable_checkpointing: Enable checkpoint/resume functionality
            checkpoint_interval: Save checkpoint every N completed tasks
            checkpoint_dir: Directory for checkpoint files
            max_retries: Maximum retries per task (default: 2)
            show_progress: Show progress bars (default: True)
        """
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.show_progress = show_progress

        # Rate limiting
        if rate_limit and rate_limit > 0:
            self.rate_limiter = TokenBucketRateLimiter(rate=rate_limit)
            logger.info(f"Rate limiting enabled: {rate_limit} req/s")
        else:
            self.rate_limiter = NoOpRateLimiter()

        # Checkpointing
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_manager = CheckpointManager(checkpoint_dir) if enable_checkpointing else None

        logger.info(
            f"ParallelLLMExecutor initialized: "
            f"workers={max_workers}, rate_limit={rate_limit}, "
            f"checkpointing={enable_checkpointing}"
        )

    def execute_batch(
        self,
        tasks: List[LLMTask],
        checkpoint_name: Optional[str] = None,
        resume_from_checkpoint: bool = True,
        on_task_complete: Optional[Callable[[TaskResult], None]] = None
    ) -> BatchExecutionResult:
        """
        Execute a batch of LLM tasks in parallel.

        Args:
            tasks: List of LLMTask objects to execute
            checkpoint_name: Name for checkpoint (default: auto-generated)
            resume_from_checkpoint: Try to resume from existing checkpoint
            on_task_complete: Optional callback called after each task completes

        Returns:
            BatchExecutionResult with all task results and metrics
        """
        if not tasks:
            logger.warning("No tasks to execute")
            return BatchExecutionResult(task_results=[], metrics=ExecutionMetrics())

        logger.info(f"Executing batch of {len(tasks)} tasks")

        # Initialize metrics
        metrics = ExecutionMetrics(total_tasks=len(tasks))

        # Setup checkpoint
        checkpoint = self._init_checkpoint(
            tasks=tasks,
            checkpoint_name=checkpoint_name or f"batch_{len(tasks)}_tasks",
            resume=resume_from_checkpoint
        )

        # Filter out already completed tasks
        pending_tasks = self._get_pending_tasks(tasks, checkpoint)
        logger.info(f"Pending tasks: {len(pending_tasks)} (resumed: {len(tasks) - len(pending_tasks)})")

        # Execute tasks
        task_results = []

        if checkpoint:
            # Add results from checkpoint
            for task in tasks:
                if checkpoint.is_task_completed(task.task_id):
                    cached_result = checkpoint.get_result(task.task_id)
                    if cached_result:
                        task_results.append(TaskResult.from_dict(cached_result))

        # Execute pending tasks
        if pending_tasks:
            new_results = self._execute_tasks_parallel(
                tasks=pending_tasks,
                checkpoint=checkpoint,
                metrics=metrics,
                on_task_complete=on_task_complete
            )
            task_results.extend(new_results)

        # Finalize metrics
        metrics.completed_tasks = sum(1 for r in task_results if r.success)
        metrics.failed_tasks = sum(1 for r in task_results if not r.success)
        metrics.finalize()

        # Cleanup checkpoint on success
        if self.enable_checkpointing and checkpoint:
            keep_checkpoint = metrics.failed_tasks > 0  # Keep if there were failures
            if hasattr(checkpoint, '_checkpoint_file') and checkpoint._checkpoint_file:
                self.checkpoint_manager.cleanup(
                    checkpoint._checkpoint_file,
                    keep_on_completion=keep_checkpoint
                )

        logger.info(
            f"Batch execution complete: {metrics.completed_tasks} successful, "
            f"{metrics.failed_tasks} failed, {metrics.total_execution_time:.1f}s"
        )

        return BatchExecutionResult(
            task_results=task_results,
            metrics=metrics,
            checkpoint=checkpoint if metrics.failed_tasks > 0 else None
        )

    def _init_checkpoint(
        self,
        tasks: List[LLMTask],
        checkpoint_name: str,
        resume: bool
    ) -> Optional[ExecutionCheckpoint]:
        """Initialize or load checkpoint."""
        if not self.enable_checkpointing:
            return None

        # Try to load existing checkpoint
        if resume:
            checkpoint_file = self.checkpoint_manager.find_latest_checkpoint(checkpoint_name)
            if checkpoint_file:
                try:
                    checkpoint = self.checkpoint_manager.load(checkpoint_file)
                    checkpoint._checkpoint_file = checkpoint_file  # Store for cleanup
                    logger.info(f"Resuming from checkpoint: {checkpoint.num_completed}/{checkpoint.total_tasks} completed")
                    return checkpoint
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}, starting fresh")

        # Create new checkpoint
        checkpoint = create_checkpoint(
            checkpoint_name=checkpoint_name,
            total_tasks=len(tasks),
            metadata={"started_at": datetime.now().isoformat()}
        )
        return checkpoint

    def _get_pending_tasks(
        self,
        tasks: List[LLMTask],
        checkpoint: Optional[ExecutionCheckpoint]
    ) -> List[LLMTask]:
        """Filter out already completed tasks from checkpoint."""
        if not checkpoint:
            return tasks

        all_task_ids = [task.task_id for task in tasks]
        pending_ids = set(checkpoint.get_pending_task_ids(all_task_ids))

        return [task for task in tasks if task.task_id in pending_ids]

    def _execute_tasks_parallel(
        self,
        tasks: List[LLMTask],
        checkpoint: Optional[ExecutionCheckpoint],
        metrics: ExecutionMetrics,
        on_task_complete: Optional[Callable[[TaskResult], None]]
    ) -> List[TaskResult]:
        """Execute tasks in parallel with thread pool."""
        results = []
        tasks_since_checkpoint = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._execute_single_task, task): task
                for task in tasks
            }

            # Setup progress bar
            if self.show_progress:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=console
                )
                progress.start()
                progress_task = progress.add_task(
                    f"🚀 Executing {len(tasks)} tasks...",
                    total=len(tasks)
                )
            else:
                progress = None

            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    result = future.result()
                    results.append(result)

                    # Update metrics
                    if result.success:
                        metrics.completed_tasks += 1
                    else:
                        metrics.failed_tasks += 1
                    metrics.retried_tasks += result.retry_count
                    metrics.update_task_time(result.execution_time)

                    # Update checkpoint
                    if checkpoint:
                        if result.success:
                            checkpoint.mark_completed(task.task_id, result.to_dict())
                        else:
                            checkpoint.mark_failed(task.task_id, result.error or "Unknown error")

                        tasks_since_checkpoint += 1

                        # Periodic checkpoint save
                        if tasks_since_checkpoint >= self.checkpoint_interval:
                            self._save_checkpoint(checkpoint)
                            tasks_since_checkpoint = 0

                    # Call completion callback
                    if on_task_complete:
                        try:
                            on_task_complete(result)
                        except Exception as e:
                            logger.warning(f"Task completion callback failed: {e}")

                    # Update progress
                    if progress:
                        progress.update(progress_task, advance=1)

                except Exception as e:
                    logger.error(f"Future failed for task {task.task_id}: {e}")
                    results.append(TaskResult(
                        task_id=task.task_id,
                        success=False,
                        error=str(e)
                    ))

            # Stop progress bar
            if progress:
                progress.stop()

        # Final checkpoint save
        if checkpoint and tasks_since_checkpoint > 0:
            self._save_checkpoint(checkpoint)

        return results

    def _execute_single_task(self, task: LLMTask) -> TaskResult:
        """Execute a single LLM task with retry logic."""
        task_start_time = time.time()
        retry_count = 0
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                self.rate_limiter.acquire()

                # Build messages
                messages = task.build_messages()

                # Get LLM parameters
                llm_params = task.get_llm_params()

                # Call LLM
                response = self.llm_client.complete(messages=messages, **llm_params)

                # Parse response
                parsed_result = task.parse_response(response.content)

                # Success!
                execution_time = time.time() - task_start_time

                task.on_success(parsed_result)

                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    result=parsed_result,
                    raw_response=response.content,
                    retry_count=retry_count,
                    execution_time=execution_time
                )

            except Exception as e:
                last_error = e
                retry_count += 1

                logger.warning(
                    f"Task {task.task_id} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )

                # Check if should retry
                if attempt < self.max_retries and task.should_retry(e, retry_count):
                    # Exponential backoff
                    wait_time = min(2 ** attempt, 10)  # Max 10 seconds
                    time.sleep(wait_time)
                    continue
                else:
                    break

        # All retries failed
        execution_time = time.time() - task_start_time

        task.on_failure(last_error)

        return TaskResult(
            task_id=task.task_id,
            success=False,
            error=str(last_error),
            retry_count=retry_count,
            execution_time=execution_time
        )

    def _save_checkpoint(self, checkpoint: ExecutionCheckpoint):
        """Save checkpoint to file."""
        try:
            checkpoint_file = getattr(checkpoint, '_checkpoint_file', None)
            saved_file = self.checkpoint_manager.save(checkpoint, checkpoint_file)

            # Store for future saves
            if not hasattr(checkpoint, '_checkpoint_file'):
                checkpoint._checkpoint_file = saved_file

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")


def create_executor(
    llm_client: BaseLLMClient,
    max_workers: int = 5,
    rate_limit: Optional[float] = 10.0,
    enable_checkpointing: bool = True
) -> ParallelLLMExecutor:
    """
    Convenience function to create a ParallelLLMExecutor.

    Args:
        llm_client: LLM client
        max_workers: Maximum concurrent tasks
        rate_limit: Rate limit in requests/second
        enable_checkpointing: Enable checkpointing

    Returns:
        Configured ParallelLLMExecutor
    """
    return ParallelLLMExecutor(
        llm_client=llm_client,
        max_workers=max_workers,
        rate_limit=rate_limit,
        enable_checkpointing=enable_checkpointing
    )
