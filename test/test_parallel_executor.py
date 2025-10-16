"""
Tests for Parallel LLM Execution Framework
"""

import pytest
import time
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import Future

from aita.utils.rate_limiter import TokenBucketRateLimiter, NoOpRateLimiter
from aita.utils.checkpoint import ExecutionCheckpoint, CheckpointManager, create_checkpoint
from aita.utils.llm_tasks import LLMTask, SimpleLLMTask, TaskResult
from aita.utils.parallel_executor import ParallelLLMExecutor, ExecutionMetrics
from aita.services.llm.base import LLMMessage, LLMResponse


class TestRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_init(self):
        """Test rate limiter initialization."""
        limiter = TokenBucketRateLimiter(rate=10.0)
        assert limiter.rate == 10.0
        assert limiter.capacity == 20  # 2x rate by default
        assert limiter.tokens == 20

    def test_acquire_immediate(self):
        """Test immediate token acquisition."""
        limiter = TokenBucketRateLimiter(rate=10.0)
        assert limiter.acquire(tokens=5, blocking=False)
        assert limiter.tokens == 15

    def test_acquire_blocking(self):
        """Test blocking token acquisition."""
        limiter = TokenBucketRateLimiter(rate=10.0)

        # Deplete tokens
        limiter.acquire(tokens=20)

        # Next acquisition should wait
        start = time.time()
        assert limiter.acquire(tokens=1, blocking=True, timeout=0.2)
        elapsed = time.time() - start
        assert elapsed >= 0.05  # Should wait at least 0.05s for 1 token at 10/s

    def test_acquire_timeout(self):
        """Test acquisition timeout."""
        limiter = TokenBucketRateLimiter(rate=1.0)  # 1 token per second
        limiter.acquire(tokens=2)  # Deplete

        # Try to acquire with short timeout
        assert not limiter.acquire(tokens=10, blocking=True, timeout=0.1)

    def test_token_refill(self):
        """Test token refill over time."""
        limiter = TokenBucketRateLimiter(rate=10.0)
        limiter.acquire(tokens=20)  # Deplete

        time.sleep(0.5)  # Wait for 5 tokens to refill

        available = limiter.get_available_tokens()
        assert available >= 4 and available <= 6  # Allow some timing variance

    def test_reset(self):
        """Test rate limiter reset."""
        limiter = TokenBucketRateLimiter(rate=10.0)
        limiter.acquire(tokens=10)

        limiter.reset()
        assert limiter.tokens == limiter.capacity

    def test_no_op_limiter(self):
        """Test NoOpRateLimiter always allows requests."""
        limiter = NoOpRateLimiter()
        assert limiter.acquire(tokens=1000000)
        assert limiter.try_acquire(tokens=1000000)
        assert limiter.get_wait_time(1000000) == 0.0


class TestCheckpoint:
    """Tests for ExecutionCheckpoint and CheckpointManager."""

    def test_checkpoint_init(self):
        """Test checkpoint initialization."""
        checkpoint = create_checkpoint("test_checkpoint", total_tasks=10)
        assert checkpoint.checkpoint_name == "test_checkpoint"
        assert checkpoint.total_tasks == 10
        assert checkpoint.num_completed == 0
        assert checkpoint.num_failed == 0

    def test_mark_completed(self):
        """Test marking tasks as completed."""
        checkpoint = create_checkpoint("test", total_tasks=5)
        checkpoint.mark_completed("task1", result={"data": "test"})

        assert checkpoint.is_task_completed("task1")
        assert checkpoint.get_result("task1") == {"data": "test"}
        assert checkpoint.num_completed == 1

    def test_mark_failed(self):
        """Test marking tasks as failed."""
        checkpoint = create_checkpoint("test", total_tasks=5)
        checkpoint.mark_failed("task1", "Test error")

        assert checkpoint.num_failed == 1
        assert "task1" in checkpoint.failed_task_ids

    def test_get_pending_tasks(self):
        """Test getting pending task IDs."""
        checkpoint = create_checkpoint("test", total_tasks=5)
        all_ids = ["task1", "task2", "task3", "task4", "task5"]

        checkpoint.mark_completed("task1")
        checkpoint.mark_completed("task3")
        checkpoint.mark_failed("task4", "error")

        pending = checkpoint.get_pending_task_ids(all_ids)
        assert set(pending) == {"task2", "task5"}

    def test_completion_rate(self):
        """Test completion rate calculation."""
        checkpoint = create_checkpoint("test", total_tasks=10)
        checkpoint.mark_completed("task1")
        checkpoint.mark_completed("task2")

        assert checkpoint.completion_rate == 20.0

    def test_checkpoint_serialization(self):
        """Test checkpoint to_dict and from_dict."""
        checkpoint = create_checkpoint("test", total_tasks=5)
        checkpoint.mark_completed("task1", result={"data": 1})
        checkpoint.mark_failed("task2", "error")

        data = checkpoint.to_dict()
        restored = ExecutionCheckpoint.from_dict(data)

        assert restored.checkpoint_name == checkpoint.checkpoint_name
        assert restored.total_tasks == checkpoint.total_tasks
        assert restored.is_task_completed("task1")
        assert "task2" in restored.failed_task_ids

    def test_checkpoint_manager_save_load(self, tmp_path):
        """Test saving and loading checkpoints."""
        manager = CheckpointManager(tmp_path)

        checkpoint = create_checkpoint("test_save", total_tasks=10)
        checkpoint.mark_completed("task1")
        checkpoint.mark_completed("task2")

        # Save
        file_path = manager.save(checkpoint)
        assert file_path.exists()

        # Load
        loaded = manager.load(file_path)
        assert loaded.checkpoint_name == "test_save"
        assert loaded.num_completed == 2

    def test_find_latest_checkpoint(self, tmp_path):
        """Test finding latest checkpoint."""
        manager = CheckpointManager(tmp_path)

        checkpoint1 = create_checkpoint("test", total_tasks=5)
        checkpoint2 = create_checkpoint("test", total_tasks=5)

        file1 = manager.save(checkpoint1)
        time.sleep(0.1)  # Ensure different timestamps
        file2 = manager.save(checkpoint2)

        latest = manager.find_latest_checkpoint("test")
        assert latest == file2

    def test_cleanup(self, tmp_path):
        """Test checkpoint cleanup."""
        manager = CheckpointManager(tmp_path)

        checkpoint = create_checkpoint("test", total_tasks=5)
        file_path = manager.save(checkpoint)

        assert file_path.exists()
        manager.cleanup(file_path, keep_on_completion=False)
        assert not file_path.exists()


class TestLLMTasks:
    """Tests for LLM task abstractions."""

    def test_simple_task_creation(self):
        """Test creating a simple LLM task."""
        task = SimpleLLMTask(
            task_id="test_task",
            prompt="What is 2+2?",
            system_prompt="You are a calculator",
            temperature=0.5
        )

        assert task.task_id == "test_task"
        assert task.get_llm_params()["temperature"] == 0.5

    def test_simple_task_messages(self):
        """Test message building."""
        task = SimpleLLMTask(
            task_id="test",
            prompt="Test prompt",
            system_prompt="System message"
        )

        messages = task.build_messages()
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "System message"
        assert messages[1].role == "user"
        assert messages[1].content == "Test prompt"

    def test_simple_task_parse(self):
        """Test response parsing with custom parser."""
        def parse_fn(text):
            return {"parsed": text.upper()}

        task = SimpleLLMTask(
            task_id="test",
            prompt="test",
            parse_fn=parse_fn
        )

        result = task.parse_response("hello")
        assert result == {"parsed": "HELLO"}

    def test_task_result_creation(self):
        """Test TaskResult creation."""
        result = TaskResult(
            task_id="test",
            success=True,
            result={"answer": 42},
            execution_time=1.5
        )

        assert result.task_id == "test"
        assert result.success
        assert result.result["answer"] == 42

    def test_task_result_serialization(self):
        """Test TaskResult serialization."""
        result = TaskResult(
            task_id="test",
            success=True,
            result={"data": "value"},
            raw_response="response text",
            retry_count=2
        )

        data = result.to_dict()
        restored = TaskResult.from_dict(data)

        assert restored.task_id == result.task_id
        assert restored.success == result.success
        assert restored.retry_count == result.retry_count


class TestParallelExecutor:
    """Tests for ParallelLLMExecutor."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        client.complete = Mock(return_value=LLMResponse(
            content="test response",
            usage={"input_tokens": 10, "output_tokens": 20},
            model="test-model"
        ))
        return client

    @pytest.fixture
    def simple_tasks(self):
        """Create list of simple tasks."""
        return [
            SimpleLLMTask(
                task_id=f"task_{i}",
                prompt=f"Task {i}",
                parse_fn=lambda x: {"result": x}
            )
            for i in range(5)
        ]

    def test_executor_init(self, mock_llm_client):
        """Test executor initialization."""
        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            max_workers=3,
            rate_limit=5.0,
            enable_checkpointing=False
        )

        assert executor.llm_client == mock_llm_client
        assert executor.max_workers == 3

    def test_execute_batch_success(self, mock_llm_client, simple_tasks):
        """Test successful batch execution."""
        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            max_workers=2,
            enable_checkpointing=False,
            show_progress=False
        )

        result = executor.execute_batch(simple_tasks)

        assert len(result.task_results) == 5
        assert result.metrics.completed_tasks == 5
        assert result.metrics.failed_tasks == 0
        assert result.success_rate == 100.0

    def test_execute_batch_with_failures(self, mock_llm_client, simple_tasks):
        """Test batch execution with some failures."""
        # Make some calls fail
        mock_llm_client.complete.side_effect = [
            LLMResponse(content="success"),
            Exception("API error"),
            LLMResponse(content="success"),
            Exception("Timeout"),
            LLMResponse(content="success")
        ]

        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            max_workers=2,
            enable_checkpointing=False,
            show_progress=False,
            max_retries=0  # No retries for test speed
        )

        result = executor.execute_batch(simple_tasks)

        assert len(result.task_results) == 5
        assert result.metrics.failed_tasks == 2
        assert 40.0 < result.success_rate < 60.0  # Approximately 60%

    def test_execute_batch_empty(self, mock_llm_client):
        """Test executing empty batch."""
        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            enable_checkpointing=False,
            show_progress=False
        )

        result = executor.execute_batch([])
        assert len(result.task_results) == 0

    def test_rate_limiting(self, mock_llm_client, simple_tasks):
        """Test that rate limiting is applied."""
        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            max_workers=5,
            rate_limit=5.0,  # 5 requests per second
            enable_checkpointing=False,
            show_progress=False
        )

        start_time = time.time()
        result = executor.execute_batch(simple_tasks)
        elapsed = time.time() - start_time

        # 5 tasks at 5/sec should take at least 0.8 seconds
        # (allowing some overhead)
        assert elapsed >= 0.6

    def test_metrics_collection(self, mock_llm_client):
        """Test execution metrics collection."""
        tasks = [SimpleLLMTask(
            task_id=f"task_{i}",
            prompt=f"Test {i}",
            parse_fn=lambda x: x
        ) for i in range(3)]

        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            enable_checkpointing=False,
            show_progress=False
        )

        result = executor.execute_batch(tasks)

        assert result.metrics.total_tasks == 3
        assert result.metrics.throughput > 0
        assert result.metrics.avg_task_time > 0

    @patch('aita.utils.parallel_executor.CheckpointManager')
    def test_checkpointing(self, mock_checkpoint_manager, mock_llm_client, simple_tasks, tmp_path):
        """Test checkpoint functionality."""
        executor = ParallelLLMExecutor(
            llm_client=mock_llm_client,
            enable_checkpointing=True,
            checkpoint_dir=tmp_path,
            checkpoint_interval=2,
            show_progress=False
        )

        result = executor.execute_batch(simple_tasks, checkpoint_name="test_batch")

        # Checkpoint manager should have been used
        assert executor.checkpoint_manager is not None


class TestExecutionMetrics:
    """Tests for ExecutionMetrics."""

    def test_metrics_init(self):
        """Test metrics initialization."""
        metrics = ExecutionMetrics(total_tasks=10)
        assert metrics.total_tasks == 10
        assert metrics.completed_tasks == 0

    def test_update_task_time(self):
        """Test updating task timing metrics."""
        metrics = ExecutionMetrics()
        metrics.completed_tasks = 1

        metrics.update_task_time(1.5)
        metrics.completed_tasks = 2
        metrics.update_task_time(2.5)

        assert metrics.min_task_time == 1.5
        assert metrics.max_task_time == 2.5
        assert metrics.avg_task_time == 2.0

    def test_finalize(self):
        """Test metrics finalization."""
        metrics = ExecutionMetrics(total_tasks=10)
        metrics.completed_tasks = 8
        time.sleep(0.1)

        metrics.finalize()

        assert metrics.end_time is not None
        assert metrics.total_execution_time > 0
        assert metrics.throughput > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
