from redis.asyncio import Redis
from app.config import get_settings

_client: Redis | None = None

async def get_redis() -> Redis:
    global _client
    if _client is None:
        _client = Redis.from_url(get_settings().redis_url, decode_responses=True, health_check_interval=30)
        await _client.ping()
    return _client

async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
