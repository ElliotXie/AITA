"""Model pricing data for cost tracking."""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for a model."""
    input_per_million: float  # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens
    image_per_thousand: Optional[float] = None  # USD per 1K images
    currency: str = "USD"


# Classic model pricing data (as of Jun 2025)
MODEL_PRICING: Dict[str, ModelPricing] = {
    # Google Gemini Models via OpenRouter
    "google/gemini-2.5-flash": ModelPricing(
        input_per_million=0.30,  # $0.30/M input tokens
        output_per_million=2.50,  # $2.50/M output tokens
        image_per_thousand=1.238  # $1.238/K input images
    ),
    "google/gemini-2.0-flash-exp": ModelPricing(
        input_per_million=0.30,
        output_per_million=2.50,
        image_per_thousand=1.238
    ),
    "google/gemini-pro": ModelPricing(
        input_per_million=1.25,
        output_per_million=5.00,
        image_per_thousand=2.50
    ),
    "google/gemini-pro-vision": ModelPricing(
        input_per_million=1.25,
        output_per_million=5.00,
        image_per_thousand=2.50
    ),

    # OpenAI Models
    "openai/gpt-4-turbo": ModelPricing(
        input_per_million=10.00,
        output_per_million=30.00,
        image_per_thousand=None  # Vision pricing integrated in tokens
    ),
    "openai/gpt-4": ModelPricing(
        input_per_million=30.00,
        output_per_million=60.00
    ),
    "openai/gpt-3.5-turbo": ModelPricing(
        input_per_million=0.50,
        output_per_million=1.50
    ),
    "openai/gpt-4o": ModelPricing(
        input_per_million=5.00,
        output_per_million=15.00,
        image_per_thousand=None
    ),
    "openai/gpt-4o-mini": ModelPricing(
        input_per_million=0.15,
        output_per_million=0.60,
        image_per_thousand=None
    ),

    # Anthropic Claude Models
    "anthropic/claude-3-opus": ModelPricing(
        input_per_million=15.00,
        output_per_million=75.00
    ),
    "anthropic/claude-3-sonnet": ModelPricing(
        input_per_million=3.00,
        output_per_million=15.00
    ),
    "anthropic/claude-3-haiku": ModelPricing(
        input_per_million=0.25,
        output_per_million=1.25
    ),
    "anthropic/claude-3.5-sonnet": ModelPricing(
        input_per_million=3.00,
        output_per_million=15.00
    ),

    # Meta Llama Models
    "meta-llama/llama-3-70b-instruct": ModelPricing(
        input_per_million=0.80,
        output_per_million=0.80
    ),
    "meta-llama/llama-3-8b-instruct": ModelPricing(
        input_per_million=0.10,
        output_per_million=0.10
    ),

    # Mistral Models
    "mistral/mistral-large": ModelPricing(
        input_per_million=4.00,
        output_per_million=12.00
    ),
    "mistral/mistral-medium": ModelPricing(
        input_per_million=2.70,
        output_per_million=8.10
    ),
    "mistral/mixtral-8x7b-instruct": ModelPricing(
        input_per_million=0.60,
        output_per_million=0.60
    ),

    # Cohere Models
    "cohere/command-r-plus": ModelPricing(
        input_per_million=3.00,
        output_per_million=15.00
    ),
    "cohere/command-r": ModelPricing(
        input_per_million=0.50,
        output_per_million=1.50
    ),
}


def get_model_pricing(model_name: str) -> Optional[ModelPricing]:
    """
    Get pricing information for a model.

    Args:
        model_name: The model identifier

    Returns:
        ModelPricing object if found, None otherwise
    """
    return MODEL_PRICING.get(model_name)


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    image_count: int = 0
) -> Dict[str, float]:
    """
    Calculate the cost for a model usage.

    Args:
        model_name: The model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        image_count: Number of images processed

    Returns:
        Dictionary with cost breakdown
    """
    pricing = get_model_pricing(model_name)

    if not pricing:
        return {
            "error": f"No pricing data for model: {model_name}",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "image_count": image_count,
            "total_cost": 0.0
        }

    # Calculate token costs
    input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million

    # Calculate image costs if applicable
    image_cost = 0.0
    if image_count > 0 and pricing.image_per_thousand is not None:
        image_cost = (image_count / 1_000) * pricing.image_per_thousand

    total_cost = input_cost + output_cost + image_cost

    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "image_count": image_count,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "image_cost": round(image_cost, 6),
        "total_cost": round(total_cost, 6),
        "currency": pricing.currency
    }