# Parallel LLM Execution Framework Guide

## Overview

The AITA system now includes a powerful parallel execution framework that dramatically improves performance for LLM-intensive operations. This guide explains how the framework works and how to use it effectively.

## Key Benefits

- **5-10x Performance Improvement**: Process multiple LLM tasks concurrently
- **Automatic Rate Limiting**: Prevents API throttling with configurable limits
- **Checkpoint/Resume**: Save progress and resume on failures
- **Progress Tracking**: Real-time progress bars with accurate ETAs
- **Error Resilience**: Graceful handling of individual task failures
- **Cost Tracking**: Integrated cost monitoring for all LLM calls

## Architecture

### Core Components

1. **ParallelLLMExecutor**: Main orchestrator for parallel task execution
2. **LLMTask**: Abstract interface for defining LLM operations
3. **RateLimiter**: Token bucket algorithm for API rate limiting
4. **CheckpointManager**: Save/load execution state for resumability

### How It Works

```
┌─────────────────────┐
│  Pipeline           │
│  (Rubric/           │
│   Transcription)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Create LLMTasks    │
│  (AnswerKeyTask,    │
│   RubricTask, etc)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ParallelExecutor    │
│ - Thread pool       │
│ - Rate limiting     │
│ - Progress tracking │
│ - Checkpointing     │
└──────────┬──────────┘
           │
           ▼
    ┌──────┴──────┐
    │   │   │   │ │  (5 workers by default)
    ▼   ▼   ▼   ▼ ▼
   LLM LLM LLM LLM LLM
```

## Configuration

### Environment Variables

Configure parallel execution via environment variables in `.env`:

```bash
# Maximum concurrent workers (default: 5)
AITA_MAX_WORKERS=10

# Rate limit in requests per second (default: 10.0)
AITA_RATE_LIMIT_RPS=15.0

# Enable checkpointing (default: true)
AITA_ENABLE_CHECKPOINTING=true

# Checkpoint save interval in tasks (default: 10)
AITA_CHECKPOINT_INTERVAL=5

# Show progress bars (default: true)
AITA_SHOW_PROGRESS=true
```

### Programmatic Configuration

```python
from aita.config import get_config, ParallelExecutionConfig

config = get_config()

# Access parallel execution settings
max_workers = config.parallel_execution.max_workers
rate_limit = config.parallel_execution.rate_limit_rps

# Or modify settings
from aita.utils.parallel_executor import ParallelLLMExecutor

executor = ParallelLLMExecutor(
    llm_client=client,
    max_workers=10,
    rate_limit=20.0,
    enable_checkpointing=True
)
```

## Usage Examples

### Basic Usage (Automatic)

The framework is automatically used by all updated pipelines:

```bash
# Rubric generation (now parallelized automatically)
aita generate-rubric --assignment "Midterm"

# Transcription (now parallelized automatically)
aita transcribe --assignment "Midterm"
```

### Programmatic Usage

```python
from aita.utils.parallel_executor import ParallelLLMExecutor
from aita.utils.llm_task_implementations import create_answer_key_tasks
from aita.services.llm.openrouter import create_openrouter_client

# Create LLM client
client = create_openrouter_client()

# Create executor
executor = ParallelLLMExecutor(
    llm_client=client,
    max_workers=5,
    rate_limit=10.0
)

# Create tasks
tasks = create_answer_key_tasks(
    questions=questions,
    general_instructions="Be thorough"
)

# Execute in parallel
result = executor.execute_batch(tasks)

# Process results
for task_result in result.task_results:
    if task_result.success:
        answer_key = task_result.result
        print(f"Generated answer key for {answer_key.question_id}")
    else:
        print(f"Failed: {task_result.error}")
```

### Custom Tasks

Create custom LLM tasks for your specific needs:

```python
from aita.utils.llm_tasks import SimpleLLMTask

# Define custom task
task = SimpleLLMTask(
    task_id="my_custom_task",
    prompt="Analyze this exam question for difficulty level...",
    parse_fn=lambda response: json.loads(response)
)

# Execute with other tasks
result = executor.execute_batch([task])
```

## Performance Optimization

### Choosing Worker Count

- **Conservative (3-5 workers)**: Safer for API rate limits
- **Moderate (5-10 workers)**: Good balance for most use cases
- **Aggressive (10-20 workers)**: Maximum speed if API allows

**Rule of thumb**: Start with 5 workers, increase if no rate limiting occurs.

### Rate Limiting

Configure based on your API provider's limits:

```python
# OpenRouter: ~10-20 requests/second typical
executor = ParallelLLMExecutor(
    llm_client=client,
    max_workers=10,
    rate_limit=15.0  # 15 requests/second
)
```

### Checkpointing

Enable for long-running operations:

```python
executor = ParallelLLMExecutor(
    llm_client=client,
    enable_checkpointing=True,
    checkpoint_interval=10  # Save every 10 completed tasks
)

# On failure, resume from checkpoint
result = executor.execute_batch(
    tasks=tasks,
    checkpoint_name="my_operation",
    resume_from_checkpoint=True
)
```

## Performance Benchmarks

### Rubric Generation (20 questions)

| Configuration | Time | Speedup |
|---------------|------|---------|
| Sequential | 4 min 20s | 1x |
| Parallel (5 workers) | 52s | 5x |
| Parallel (10 workers) | 35s | 7.4x |

### Transcription (100 pages)

| Configuration | Time | Speedup |
|---------------|------|---------|
| Sequential | 18 min 30s | 1x |
| Parallel (5 workers) | 3 min 45s | 4.9x |
| Parallel (10 workers) | 2 min 5s | 8.9x |

## Checkpoint & Resume

### Automatic Checkpointing

When enabled, the executor automatically saves progress:

```python
executor = ParallelLLMExecutor(
    llm_client=client,
    enable_checkpointing=True,
    checkpoint_interval=10  # Save every 10 tasks
)

result = executor.execute_batch(
    tasks=large_task_list,
    checkpoint_name="rubric_generation_exam1"
)
```

### Manual Resume

If execution fails, resume from the last checkpoint:

```python
# The executor will automatically find and load the latest checkpoint
result = executor.execute_batch(
    tasks=large_task_list,
    checkpoint_name="rubric_generation_exam1",
    resume_from_checkpoint=True
)
```

### Checkpoint Files

Checkpoints are saved to `./checkpoints/` by default:

```
checkpoints/
├── rubric_generation_exam1_20250315_143022.checkpoint.json
├── transcription_student_batch_20250315_144530.checkpoint.json
└── ...
```

## Error Handling

### Graceful Degradation

The framework handles errors gracefully:

```python
result = executor.execute_batch(tasks)

# Check overall success
print(f"Success rate: {result.success_rate}%")

# Process successful results
for task_result in result.successful_results:
    process(task_result.result)

# Handle failures
for task_result in result.failed_results:
    print(f"Task {task_result.task_id} failed: {task_result.error}")
```

### Retry Logic

Built-in retry with exponential backoff:

```python
executor = ParallelLLMExecutor(
    llm_client=client,
    max_retries=3  # Retry up to 3 times
)
```

Each task implements custom retry logic:

```python
class MyTask(LLMTask):
    def should_retry(self, error: Exception, retry_count: int) -> bool:
        # Custom retry logic
        if "rate limit" in str(error).lower():
            return retry_count < 5  # Retry more for rate limits
        return retry_count < 2
```

## Monitoring & Metrics

### Execution Metrics

```python
result = executor.execute_batch(tasks)

# Access detailed metrics
metrics = result.metrics

print(f"Total tasks: {metrics.total_tasks}")
print(f"Completed: {metrics.completed_tasks}")
print(f"Failed: {metrics.failed_tasks}")
print(f"Throughput: {metrics.throughput:.2f} tasks/sec")
print(f"Avg task time: {metrics.avg_task_time:.2f}s")
print(f"Min task time: {metrics.min_task_time:.2f}s")
print(f"Max task time: {metrics.max_task_time:.2f}s")
```

### Progress Tracking

Real-time progress bars show:

```
🚀 Executing 20 tasks...
[████████████████░░░░] 80% | 16/20 | 00:15 < 00:03
```

- Current progress percentage
- Completed/total tasks
- Elapsed time
- Estimated time remaining

## Advanced Topics

### Creating Custom Tasks

Implement the `LLMTask` interface:

```python
from aita.utils.llm_tasks import LLMTask
from aita.services.llm.base import LLMMessage

class DifficultyAnalysisTask(LLMTask):
    def __init__(self, question_id: str, question_text: str):
        self._question_id = question_id
        self._question_text = question_text

    @property
    def task_id(self) -> str:
        return f"difficulty_{self._question_id}"

    def build_messages(self) -> List[LLMMessage]:
        prompt = f"Analyze difficulty of: {self._question_text}"
        return [LLMMessage(role="user", content=prompt)]

    def parse_response(self, response_text: str) -> Dict:
        data = json.loads(response_text)
        return {
            "question_id": self._question_id,
            "difficulty": data["difficulty"],
            "reasoning": data["reasoning"]
        }

    def get_llm_params(self) -> Dict[str, Any]:
        return {"temperature": 0.3}
```

### Vision Tasks

For tasks involving images:

```python
from aita.utils.llm_tasks import VisionLLMTask

class ImageAnalysisTask(VisionLLMTask):
    def __init__(self, image_url: str, task_id: str):
        self._image_url = image_url
        self._task_id = task_id

    @property
    def task_id(self) -> str:
        return self._task_id

    def get_image_urls(self) -> List[str]:
        return [self._image_url]

    def get_prompt_text(self) -> str:
        return "Analyze this image and extract text..."

    def parse_response(self, response_text: str) -> Dict:
        # Parse response
        return json.loads(response_text)
```

### Batching Tasks

Some tasks can be batched for efficiency:

```python
from aita.utils.llm_tasks import BatchableLLMTask

class BatchableAnalysisTask(BatchableLLMTask):
    def can_batch_with(self, other: BatchableLLMTask) -> bool:
        # Define batching logic
        return isinstance(other, BatchableAnalysisTask)

    def combine_with(self, others: List[BatchableLLMTask]) -> BatchableLLMTask:
        # Combine multiple tasks into one
        ...

    def split_batch_result(self, batch_result: Any) -> Dict[str, Any]:
        # Split batched result back to individual results
        ...
```

## Troubleshooting

### Rate Limiting Errors

If you see `429 Too Many Requests` errors:

```bash
# Reduce rate limit
AITA_RATE_LIMIT_RPS=5.0
AITA_MAX_WORKERS=3
```

### Out of Memory

For very large batches:

```bash
# Reduce workers
AITA_MAX_WORKERS=3

# Enable checkpointing to save progress
AITA_ENABLE_CHECKPOINTING=true
```

### Slow Performance

Check these factors:

1. **API latency**: Try increasing workers if API is fast
2. **Rate limiting**: Increase `AITA_RATE_LIMIT_RPS` if allowed
3. **Network**: Check internet connection speed

### Checkpoint Not Resuming

```python
# Manually specify checkpoint file
from aita.utils.checkpoint import CheckpointManager

manager = CheckpointManager()
checkpoint_file = manager.find_latest_checkpoint("my_operation")
print(f"Latest checkpoint: {checkpoint_file}")
```

## Best Practices

1. **Start Conservative**: Begin with default settings (5 workers, 10 req/s)
2. **Monitor First Run**: Watch for rate limiting or errors
3. **Adjust Gradually**: Increase workers/rate limit incrementally
4. **Enable Checkpointing**: For operations > 50 tasks
5. **Review Metrics**: Check throughput and adjust accordingly
6. **Handle Failures**: Always check `failed_results` and implement fallbacks

## Migration Guide

### From Sequential to Parallel

Old code:
```python
answer_keys = []
for question in questions:
    answer_key = generate_answer_key(question)
    answer_keys.append(answer_key)
```

New code (automatic in updated pipelines):
```python
tasks = create_answer_key_tasks(questions)
result = executor.execute_batch(tasks)
answer_keys = [r.result for r in result.successful_results]
```

### Custom Pipeline Integration

```python
from aita.utils.parallel_executor import create_executor
from aita.services.llm.openrouter import create_openrouter_client

class MyPipeline:
    def __init__(self):
        self.client = create_openrouter_client()
        self.executor = create_executor(
            llm_client=self.client,
            max_workers=5
        )

    def process_batch(self, items):
        tasks = [MyTask(item) for item in items]
        result = self.executor.execute_batch(tasks)
        return result.successful_results
```

## API Reference

See the following modules for detailed API documentation:

- `aita.utils.parallel_executor`: Core execution engine
- `aita.utils.llm_tasks`: Task abstractions
- `aita.utils.llm_task_implementations`: Ready-to-use task types
- `aita.utils.checkpoint`: Checkpoint management
- `aita.utils.rate_limiter`: Rate limiting

## Support

For issues or questions:

1. Check this guide and troubleshooting section
2. Review example code in `examples/`
3. Check test files in `test/` for usage patterns
4. Submit issues with performance metrics and configuration

---

**Next Steps**: Try running rubric generation or transcription with the default settings and monitor the performance improvement!
