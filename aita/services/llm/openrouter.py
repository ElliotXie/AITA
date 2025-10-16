import os
import requests
from typing import List, Dict, Any, Optional
import logging

from .base import BaseLLMClient, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseLLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash",
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/aita-grading/aita",
            "X-Title": "AITA Automatic Grading System"
        }

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        converted = []

        for msg in messages:
            converted_msg = {
                "role": msg.role,
                "content": msg.content
            }
            converted.append(converted_msg)

        return converted

    def _make_request(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.debug(f"Making request to OpenRouter: {url}")
        logger.debug(f"Model: {self.model}")
        logger.debug(f"Messages count: {len(messages)}")

        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=30  # Reduced from 120 to 30 seconds
        )

        if not response.ok:
            logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
            response.raise_for_status()

        data = response.json()

        if "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            logger.error(f"OpenRouter returned error: {error_msg}")
            raise Exception(f"OpenRouter API error: {error_msg}")

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]

            usage = data.get("usage", {})
            finish_reason = choice.get("finish_reason")

            return LLMResponse(
                content=content,
                usage=usage,
                model=data.get("model", self.model),
                finish_reason=finish_reason
            )

        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from OpenRouter: {data}")
            raise Exception(f"Failed to parse OpenRouter response: {e}")

    def get_available_models(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/models"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        data = response.json()
        return data.get("data", [])

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None
    ) -> Dict[str, float]:
        models = self.get_available_models()
        model_name = model or self.model

        for model_info in models:
            if model_info["id"] == model_name:
                pricing = model_info.get("pricing", {})
                prompt_cost = float(pricing.get("prompt", "0"))
                completion_cost = float(pricing.get("completion", "0"))

                # Costs are per million tokens
                total_cost = (
                    (input_tokens * prompt_cost / 1_000_000) +
                    (output_tokens * completion_cost / 1_000_000)
                )

                return {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_tokens * prompt_cost / 1_000_000,
                    "output_cost": output_tokens * completion_cost / 1_000_000,
                    "total_cost": total_cost,
                    "currency": "USD"
                }

        return {"error": f"Model {model_name} not found"}


def create_openrouter_client(**kwargs) -> OpenRouterClient:
    model = kwargs.pop("model", None) or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    return OpenRouterClient(model=model, **kwargs)