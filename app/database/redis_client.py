import asyncio

from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError

from app.config import get_settings

_client: Redis | None = None
_client_lock = asyncio.Lock()


def _create_client() -> Redis:
    return Redis.from_url(
        get_settings().redis_url,
        decode_responses=True,
        health_check_interval=15,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry=Retry(ExponentialBackoff(base=0.1, cap=1), retries=3),
        retry_on_error=[ConnectionError, TimeoutError],
    )


async def get_redis() -> Redis:
    global _client

    if _client is not None:
        return _client

    async with _client_lock:
        if _client is None:
            _client = _create_client()
            try:
                await _client.ping()
            except Exception:
                await _client.aclose()
                _client = None
                raise

        return _client


async def close_redis() -> None:
    global _client

    async with _client_lock:
        if _client is not None:
            await _client.aclose()
            _client = None
