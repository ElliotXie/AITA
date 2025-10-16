"""
Rate Limiter for LLM API Calls

Implements a token bucket algorithm for rate limiting API requests.
Thread-safe and supports both synchronous and asynchronous usage.
"""

import time
import threading
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    Allows bursts of requests while maintaining an average rate limit.
    Each request consumes one token. Tokens refill at a constant rate.

    Example:
        >>> limiter = TokenBucketRateLimiter(rate=10.0)  # 10 requests per second
        >>> limiter.acquire()  # Blocks if rate limit exceeded
    """

    def __init__(
        self,
        rate: float,
        capacity: Optional[int] = None,
        initial_tokens: Optional[int] = None
    ):
        """
        Initialize the rate limiter.

        Args:
            rate: Maximum requests per second
            capacity: Maximum burst capacity (default: 2x rate)
            initial_tokens: Initial token count (default: full capacity)
        """
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")

        self.rate = rate
        self.capacity = capacity or int(rate * 2)  # Allow 2x burst by default
        self.tokens = initial_tokens if initial_tokens is not None else self.capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

        logger.debug(f"RateLimiter initialized: {rate} req/s, capacity={self.capacity}")

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire (default: 1)
            blocking: If True, wait for tokens; if False, return immediately
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if tokens were acquired, False otherwise

        Raises:
            ValueError: If tokens > capacity
        """
        if tokens > self.capacity:
            raise ValueError(f"Requested {tokens} tokens exceeds capacity {self.capacity}")

        start_time = time.time() if timeout else None

        while True:
            with self.lock:
                # Refill tokens based on elapsed time
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(f"Acquired {tokens} token(s), {self.tokens:.2f} remaining")
                    return True

                # Not enough tokens
                if not blocking:
                    logger.debug(f"Rate limit: need {tokens}, have {self.tokens:.2f}")
                    return False

                # Check timeout
                if timeout and (time.time() - start_time) >= timeout:
                    logger.debug(f"Rate limit timeout after {timeout}s")
                    return False

                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate

            # Wait outside the lock
            logger.debug(f"Rate limit: waiting {wait_time:.3f}s for {tokens_needed:.2f} tokens")
            time.sleep(min(wait_time, 0.1))  # Sleep max 100ms at a time

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if acquired, False if rate limit exceeded
        """
        return self.acquire(tokens=tokens, blocking=False)

    def reset(self):
        """Reset the rate limiter to full capacity."""
        with self.lock:
            self.tokens = self.capacity
            self.last_update = time.time()
        logger.debug("RateLimiter reset to full capacity")

    def get_available_tokens(self) -> float:
        """Get the current number of available tokens."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            available = min(self.capacity, self.tokens + elapsed * self.rate)
            return available

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Calculate how long to wait before tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds (0 if tokens are available now)
        """
        available = self.get_available_tokens()
        if available >= tokens:
            return 0.0

        tokens_needed = tokens - available
        return tokens_needed / self.rate


class NoOpRateLimiter:
    """
    No-operation rate limiter for testing or when rate limiting is disabled.
    """

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        return True

    def try_acquire(self, tokens: int = 1) -> bool:
        return True

    def reset(self):
        pass

    def get_available_tokens(self) -> float:
        return float('inf')

    def get_wait_time(self, tokens: int = 1) -> float:
        return 0.0
