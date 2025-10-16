"""
LLM Task Abstraction

Defines the protocol and base classes for LLM tasks that can be executed
in parallel. Each task knows how to build its prompt, parse responses,
and serialize for checkpointing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass, field
import logging

from aita.services.llm.base import LLMMessage

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """
    Result of executing an LLM task.

    Contains the parsed output, raw response, and metadata about execution.
    """

    task_id: str
    success: bool
    result: Any = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Serialize result if it has a to_dict method
        serialized_result = self.result
        if hasattr(self.result, 'to_dict') and callable(self.result.to_dict):
            serialized_result = self.result.to_dict()

        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": serialized_result,
            "raw_response": self.raw_response,
            "error": self.error,
            "retry_count": self.retry_count,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        """Create from dictionary."""
        return cls(**data)


class LLMTask(ABC):
    """
    Abstract base class for LLM tasks that can be executed in parallel.

    Each task must implement methods to:
    - Generate LLM prompts
    - Parse LLM responses
    - Serialize/deserialize for checkpointing
    """

    @property
    @abstractmethod
    def task_id(self) -> str:
        """Unique identifier for this task."""
        pass

    @abstractmethod
    def build_messages(self) -> List[LLMMessage]:
        """
        Build the LLM messages for this task.

        Returns:
            List of LLMMessage objects
        """
        pass

    @abstractmethod
    def parse_response(self, response_text: str) -> Any:
        """
        Parse the LLM response into a structured result.

        Args:
            response_text: Raw text response from LLM

        Returns:
            Parsed result object

        Raises:
            ValueError: If response cannot be parsed
        """
        pass

    def get_llm_params(self) -> Dict[str, Any]:
        """
        Get LLM parameters for this task (temperature, max_tokens, etc.).

        Returns:
            Dictionary of LLM parameters
        """
        return {
            "temperature": 0.1,
            "max_tokens": None
        }

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get data for checkpointing this task.

        Returns:
            Dictionary of serializable checkpoint data
        """
        return {
            "task_id": self.task_id,
            "task_type": self.__class__.__name__
        }

    @classmethod
    def from_checkpoint(cls, data: Dict[str, Any]) -> "LLMTask":
        """
        Recreate task from checkpoint data.

        Args:
            data: Checkpoint data dictionary

        Returns:
            Reconstructed LLMTask instance
        """
        raise NotImplementedError(f"{cls.__name__} does not support checkpoint restoration")

    def on_success(self, result: Any):
        """
        Hook called when task completes successfully.

        Args:
            result: Parsed result from parse_response
        """
        pass

    def on_failure(self, error: Exception):
        """
        Hook called when task fails.

        Args:
            error: Exception that caused the failure
        """
        pass

    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """
        Determine if task should be retried after failure.

        Args:
            error: Exception that occurred
            retry_count: Number of retries so far

        Returns:
            True if should retry, False otherwise
        """
        # By default, allow retries for common transient errors
        transient_errors = (
            "timeout",
            "connection",
            "rate limit",
            "503",
            "502",
            "429"
        )

        error_msg = str(error).lower()
        is_transient = any(err in error_msg for err in transient_errors)

        return is_transient and retry_count < 3


class SimpleLLMTask(LLMTask):
    """
    Simple implementation of LLMTask for quick prototyping.

    Useful for one-off tasks or testing.
    """

    def __init__(
        self,
        task_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        parse_fn=None,
        temperature: float = 0.1
    ):
        """
        Initialize simple task.

        Args:
            task_id: Unique task identifier
            prompt: User prompt text
            system_prompt: Optional system prompt
            parse_fn: Optional function to parse response (default: identity)
            temperature: LLM temperature parameter
        """
        self._task_id = task_id
        self._prompt = prompt
        self._system_prompt = system_prompt
        self._parse_fn = parse_fn or (lambda x: x)
        self._temperature = temperature

    @property
    def task_id(self) -> str:
        return self._task_id

    def build_messages(self) -> List[LLMMessage]:
        messages = []
        if self._system_prompt:
            messages.append(LLMMessage(role="system", content=self._system_prompt))
        messages.append(LLMMessage(role="user", content=self._prompt))
        return messages

    def parse_response(self, response_text: str) -> Any:
        return self._parse_fn(response_text)

    def get_llm_params(self) -> Dict[str, Any]:
        return {
            "temperature": self._temperature,
            "max_tokens": None
        }


class VisionLLMTask(LLMTask):
    """
    Base class for LLM tasks that include image inputs.

    Handles multimodal message construction with images.
    """

    @abstractmethod
    def get_image_urls(self) -> List[str]:
        """
        Get list of image URLs for this task.

        Returns:
            List of image URLs
        """
        pass

    @abstractmethod
    def get_prompt_text(self) -> str:
        """
        Get the text prompt for this task.

        Returns:
            Prompt text string
        """
        pass

    def build_messages(self) -> List[LLMMessage]:
        """Build multimodal messages with text and images."""
        image_urls = self.get_image_urls()

        # Build user content with text and images
        user_content = [
            {"type": "text", "text": self.get_prompt_text()}
        ]

        for image_url in image_urls:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        return [LLMMessage(role="user", content=user_content)]


class BatchableLLMTask(LLMTask):
    """
    Protocol for tasks that can be batched together.

    Some LLM tasks can be combined into a single request for efficiency.
    """

    @abstractmethod
    def can_batch_with(self, other: "BatchableLLMTask") -> bool:
        """
        Check if this task can be batched with another.

        Args:
            other: Another BatchableLLMTask

        Returns:
            True if tasks can be combined
        """
        pass

    @abstractmethod
    def combine_with(self, others: List["BatchableLLMTask"]) -> "BatchableLLMTask":
        """
        Combine this task with others into a single batch task.

        Args:
            others: List of other tasks to batch

        Returns:
            New BatchableLLMTask representing the batch
        """
        pass

    @abstractmethod
    def split_batch_result(self, batch_result: Any) -> Dict[str, Any]:
        """
        Split a batched result back into individual task results.

        Args:
            batch_result: Result from combined batch task

        Returns:
            Dictionary mapping task_id to individual result
        """
        pass


def create_simple_task(
    task_id: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    parse_fn=None,
    temperature: float = 0.1
) -> SimpleLLMTask:
    """
    Convenience function to create a simple LLM task.

    Args:
        task_id: Unique task identifier
        prompt: User prompt
        system_prompt: Optional system prompt
        parse_fn: Optional response parser
        temperature: LLM temperature

    Returns:
        SimpleLLMTask instance
    """
    return SimpleLLMTask(
        task_id=task_id,
        prompt=prompt,
        system_prompt=system_prompt,
        parse_fn=parse_fn,
        temperature=temperature
    )
