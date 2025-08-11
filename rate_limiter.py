"""
Rate Limiter for Gemini API
Handles free plan limitations (15 requests per minute) with smart queuing and retry logic
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from config import get_enhanced_config

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""

    max_requests_per_minute: int = 15  # Free plan limit
    max_requests_per_second: int = 1  # Conservative approach
    retry_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    jitter: bool = True  # Add random jitter to avoid thundering herd

    @classmethod
    def from_enhanced_config(cls, config):
        """Create RateLimitConfig from enhanced system config"""
        try:
            rate_settings = config.rate_limits
            return cls(
                max_requests_per_minute=rate_settings.max_requests_per_minute,
                max_requests_per_second=rate_settings.max_requests_per_second,
                retry_attempts=rate_settings.retry_attempts,
                base_delay=rate_settings.base_delay,
                max_delay=rate_settings.max_delay,
                jitter=rate_settings.jitter,
            )
        except Exception as e:
            logger.warning(f"Error creating RateLimitConfig from enhanced config: {e}")
            # Return default configuration
            return cls()


class RateLimiter:
    """Smart rate limiter for Gemini API with exponential backoff"""

    def __init__(self, config: RateLimitConfig = None):
        if config is None:
            # Get configuration from enhanced system config
            enhanced_config = get_enhanced_config()
            config = RateLimitConfig.from_enhanced_config(enhanced_config)

        self.config = config
        self.request_times: List[float] = []
        self.current_requests = 0
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Acquire permission to make a request, returns delay needed"""
        async with self.lock:
            now = time.time()

            # Clean old request times (older than 1 minute)
            self.request_times = [t for t in self.request_times if now - t < 60]

            # Check if we can make a request now
            if len(self.request_times) < self.config.max_requests_per_minute:
                # Add jitter to avoid synchronized requests
                delay = 0
                if self.config.jitter:
                    delay = random.uniform(0, 0.5)

                self.request_times.append(now + delay)
                return delay

            # Calculate delay needed
            oldest_request = min(self.request_times)
            time_since_oldest = now - oldest_request

            if time_since_oldest < 60:
                delay_needed = 60 - time_since_oldest
                # Add jitter
                if self.config.jitter:
                    delay_needed += random.uniform(0, 2)
                return delay_needed

            # Shouldn't reach here, but just in case
            return 1.0

    async def wait_for_permission(self):
        """Wait until we can make a request"""
        delay = await self.acquire()
        if delay > 0:
            logger.info(f"Rate limit: waiting {delay:.2f} seconds")
            await asyncio.sleep(delay)

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for retries"""
        delay = self.config.base_delay * (2**attempt)
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            delay *= random.uniform(0.5, 1.5)

        return delay


class APIClient:
    """Enhanced API client with rate limiting and retry logic"""

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.request_count = 0
        self.error_count = 0
        self.last_success = time.time()

    async def make_request(
        self, request_func: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        """Make a rate-limited API request with retry logic"""

        for attempt in range(self.rate_limiter.config.retry_attempts):
            try:
                # Wait for rate limit permission
                await self.rate_limiter.wait_for_permission()

                # Make the actual request
                start_time = time.time()
                result = await request_func(*args, **kwargs)

                # Validate the result
                if result is None:
                    raise ValueError("API returned None result")

                # Check if result is empty or just whitespace (for string results)
                if isinstance(result, str) and (not result.strip() or result.isspace()):
                    raise ValueError("API returned empty or whitespace-only response")

                # Check if result is an empty dict/list
                if isinstance(result, (dict, list)) and not result:
                    raise ValueError("API returned empty data structure")

                # Success - reset error count and update stats
                self.error_count = 0
                self.request_count += 1
                self.last_success = start_time

                logger.info(f"API request successful (attempt {attempt + 1})")
                return result

            except Exception as e:
                self.error_count += 1
                error_msg = str(e)

                # Handle specific error types
                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.warning(
                        f"Rate limit exceeded (attempt {attempt + 1}): {error_msg}"
                    )
                    # Wait longer for rate limit errors
                    delay = self.rate_limiter.get_retry_delay(attempt) * 2
                    await asyncio.sleep(delay)

                elif "list indices" in error_msg:
                    logger.error(
                        f"Programming error (attempt {attempt + 1}): {error_msg}"
                    )
                    # Don't retry programming errors
                    break

                else:
                    logger.error(f"API error (attempt {attempt + 1}): {error_msg}")
                    delay = self.rate_limiter.get_retry_delay(attempt)
                    await asyncio.sleep(delay)

                # If this was the last attempt, return error
                if attempt == self.rate_limiter.config.retry_attempts - 1:
                    return {
                        "error": f"Failed after {self.rate_limiter.config.retry_attempts} attempts: {error_msg}",
                        "attempts": attempt + 1,
                        "last_error": error_msg,
                    }

        return {"error": "Unexpected error in request handling"}


# Global rate limiter instance
_rate_limiter = RateLimiter()
_api_client = APIClient(_rate_limiter)


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance"""
    return _rate_limiter


def get_api_client() -> APIClient:
    """Get the global API client instance"""
    return _api_client
