from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time
import logging
import inspect

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    role: str  # "system", "user", "assistant"
    content: Union[str, List[Dict[str, Any]]]  # text or multimodal content


@dataclass
class LLMResponse:
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None


class BaseLLMClient(ABC):
    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_cost_tracking: bool = True,
        **kwargs
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_cost_tracking = enable_cost_tracking
        self._cost_tracker = None

    @abstractmethod
    def _make_request(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        pass

    def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        last_exception = None
        image_count = self._count_images_in_messages(messages)
        operation_type = kwargs.pop("operation_type", None)

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM request attempt {attempt + 1}/{self.max_retries}")
                response = self._make_request(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                logger.debug(f"LLM request successful on attempt {attempt + 1}")

                # Track cost if enabled
                if self.enable_cost_tracking:
                    self._track_cost(
                        response=response,
                        image_count=image_count,
                        operation_type=operation_type,
                        messages=messages
                    )

                return response

            except Exception as e:
                last_exception = e
                logger.warning(f"LLM request failed on attempt {attempt + 1}: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        logger.error(f"All {self.max_retries} LLM request attempts failed")
        raise last_exception

    def complete_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        messages = []

        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))

        messages.append(LLMMessage(role="user", content=prompt))

        response = self.complete(messages, **kwargs)
        return response.content

    def complete_with_image(
        self,
        prompt: str,
        image_url: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        messages = []

        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))

        # Multimodal content with text and image
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]

        messages.append(LLMMessage(role="user", content=user_content))

        response = self.complete(messages, **kwargs)
        return response.content

    def complete_with_images(
        self,
        prompt: str,
        image_urls: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        messages = []

        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))

        # Multimodal content with text and multiple images
        user_content = [{"type": "text", "text": prompt}]

        for image_url in image_urls:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        messages.append(LLMMessage(role="user", content=user_content))

        response = self.complete(messages, **kwargs)
        return response.content

    def set_cost_tracker(self, tracker) -> None:
        """
        Set the cost tracker for this client.

        Args:
            tracker: CostTracker instance or None
        """
        self._cost_tracker = tracker

    def _count_images_in_messages(self, messages: List[LLMMessage]) -> int:
        """
        Count the number of images in messages.

        Args:
            messages: List of LLMMessage objects

        Returns:
            Number of images found
        """
        count = 0
        for msg in messages:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        count += 1
        return count

    def _infer_operation_type(self, messages: List[LLMMessage]) -> str:
        """
        Infer operation type from messages and call stack.

        Args:
            messages: List of LLMMessage objects

        Returns:
            Inferred operation type
        """
        # Check call stack for pipeline context
        for frame_info in inspect.stack():
            filename = frame_info.filename.lower()
            function_name = frame_info.function.lower()

            # Check for known pipeline files
            if "ingest" in filename:
                return "name_extraction"
            elif "transcribe" in filename or "transcription" in function_name:
                return "transcription"
            elif "extract_question" in filename or "question" in function_name:
                return "question_extraction"
            elif "grade" in filename or "grading" in function_name:
                return "grading"
            elif "rubric" in filename:
                return "rubric_generation"

        # Check message content for patterns
        for msg in messages:
            content_str = str(msg.content).lower()
            if "extract" in content_str and "name" in content_str:
                return "name_extraction"
            elif "transcribe" in content_str or "handwriting" in content_str:
                return "transcription"
            elif "question" in content_str and ("extract" in content_str or "identify" in content_str):
                return "question_extraction"
            elif "grade" in content_str or "score" in content_str:
                return "grading"
            elif "rubric" in content_str:
                return "rubric_generation"

        return "general"

    def _track_cost(
        self,
        response: LLMResponse,
        image_count: int = 0,
        operation_type: Optional[str] = None,
        messages: Optional[List[LLMMessage]] = None
    ) -> None:
        """
        Track cost for an LLM call.

        Args:
            response: LLMResponse object with usage data
            image_count: Number of images processed
            operation_type: Type of operation (if not provided, will be inferred)
            messages: Original messages for context
        """
        try:
            # Get global tracker first, then fall back to instance tracker
            from .cost_tracker import get_global_tracker
            tracker = get_global_tracker() or self._cost_tracker

            if not tracker:
                return  # No tracker configured

            # Infer operation type if not provided
            if not operation_type and messages:
                operation_type = self._infer_operation_type(messages)
            elif not operation_type:
                operation_type = "general"

            # Track the cost
            tracker.track_call(
                model=response.model or self.model,
                operation_type=operation_type,
                usage_data=response.usage,
                image_count=image_count,
                metadata={
                    "finish_reason": response.finish_reason
                }
            )

        except Exception as e:
            logger.debug(f"Cost tracking error (non-fatal): {e}")