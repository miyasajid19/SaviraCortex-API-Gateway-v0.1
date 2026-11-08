import asyncio
import json
import time
import uuid

import httpx
import pytest
from redis.asyncio import Redis

from app.config import get_settings
from app.services.key_manager import KeyManager, fingerprint
from app.services.failover import JinaFailover, GatewayError, retry_after_seconds


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv(
        "JINA_API_KEYS",
        json.dumps(["jina-test-key-01", "jina-test-key-02", "jina-test-key-03"]),
    )
    monkeypatch.setenv("GATEWAY_API_KEY", "g" * 32)
    monkeypatch.setenv("GATEWAY_ADMIN_API_KEY", "a" * 32)
    monkeypatch.setenv("GATEWAY_MAX_ATTEMPTS", "3")
    # Each test gets an isolated namespace in the real Aiven Redis instance.
    monkeypatch.setenv("GATEWAY_REDIS_PREFIX", f"savira:gateway:pytest:{uuid.uuid4().hex}")
    get_settings.cache_clear()
    value = get_settings()
    yield value
    get_settings.cache_clear()


@pytest.fixture
async def redis(configured):
    client = Redis.from_url(
        configured.redis_url,
        decode_responses=True,
        socket_connect_timeout=8,
        socket_timeout=8,
    )
    await client.ping()
    yield client
    # Delete only this test's unique namespace; never flush the Redis database.
    keys = [key async for key in client.scan_iter(match=f"{configured.redis_prefix}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()


@pytest.mark.asyncio
async def test_round_robin_and_expired_cooldown_rejoins(configured, redis):
    manager = KeyManager(redis)
    reservations = [await manager.reserve() for _ in range(4)]
    assert [item[1] for item in reservations] == list(configured.keys) + [configured.keys[0]]
    kid = fingerprint(configured.keys[1])
    await manager.set_cooldown(kid, 1, "test rate limit")
    assert 0 < await redis.ttl(manager.status_key(kid)) <= 1
    await asyncio.sleep(1.1)
    selected = await manager.reserve()
    assert selected[1] == configured.keys[1]
    snapshot = {item["id"]: item for item in await manager.snapshot()}
    assert snapshot[kid]["status"] == "ACTIVE"


@pytest.mark.asyncio
async def test_concurrent_reservations_are_atomic(configured, redis):
    manager = KeyManager(redis)
    count = 90
    results = await asyncio.gather(*(manager.reserve() for _ in range(count)))
    selected = [item[1] for item in results if item]
    assert len(selected) == count
    assert set(selected) == set(configured.keys)
    assert max(selected.count(key) for key in configured.keys) - min(
        selected.count(key) for key in configured.keys
    ) <= 1


@pytest.mark.asyncio
async def test_429_fails_over_and_respects_retry_after(configured, redis):
    seen = []

    async def handler(request):
        key = request.headers["Authorization"].removeprefix("Bearer ")
        seen.append(key)
        if key == configured.keys[0]:
            return httpx.Response(429, headers={"Retry-After": "60"}, json={"error": "rate limited"})
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JinaFailover(redis, client).post("/embeddings", {"input": ["hello"]})
    assert result[0] == 200
    assert seen == [configured.keys[0], configured.keys[1]]
    states = {item["id"]: item for item in await KeyManager(redis).snapshot()}
    first = states[fingerprint(configured.keys[0])]
    assert first["status"] == "COOLDOWN"
    assert first["cooldown_remaining_seconds"] > 0
    assert await redis.ttl(KeyManager(redis).status_key(first["id"])) > 0


@pytest.mark.asyncio
async def test_multiple_consecutive_429s_then_success(configured, redis):
    seen = []

    async def handler(request):
        key = request.headers["Authorization"].split()[-1]
        seen.append(key)
        if key != configured.keys[2]:
            return httpx.Response(429, headers={"Retry-After": "45"}, json={"error": "limited"})
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JinaFailover(redis, client).post(
            "/rerank", {"query": "q", "documents": ["d"]}
        )
    assert result[0] == 200
    assert seen == list(configured.keys)


@pytest.mark.asyncio
async def test_authentication_failure_disables_key(configured, redis):
    async def handler(request):
        if request.headers["Authorization"].endswith(configured.keys[0]):
            return httpx.Response(401, json={"error": "invalid api key"})
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JinaFailover(redis, client).post("/embeddings", {"input": ["x"]})
    assert result[0] == 200
    states = {item["id"]: item for item in await KeyManager(redis).snapshot()}
    disabled = states[fingerprint(configured.keys[0])]
    assert disabled["status"] == "DISABLED"
    assert await redis.ttl(KeyManager(redis).status_key(disabled["id"])) == -1


@pytest.mark.asyncio
async def test_shared_quota_does_not_rotate_through_keys(configured, redis):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(
            429,
            headers={"Retry-After": "90", "X-RateLimit-Scope": "organization"},
            json={"error": "quota"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as caught:
            await JinaFailover(redis, client).post("/embeddings", {"input": ["x"]})
    assert len(calls) == 1
    assert caught.value.retry_after == 90
    assert int(await redis.get(f"{configured.redis_prefix}:gateway:metrics:shared_quota_limited")) == 1
    assert 0 < await redis.ttl(f"{configured.redis_prefix}:jina:provider:cooldown_until") <= 90


@pytest.mark.asyncio
async def test_client_400_is_returned_without_rotation(configured, redis):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"error": "bad request"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JinaFailover(redis, client).post("/embeddings", {"input": ["x"]})
    assert result[0] == 400
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_all_keys_cooling_down_returns_retry_after(configured, redis):
    async def handler(request):
        return httpx.Response(429, headers={"Retry-After": "25"}, json={"error": "limited"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as caught:
            await JinaFailover(redis, client).post("/embeddings", {"input": ["x"]})
    assert caught.value.status_code == 503
    # Redis TTL counts down while each request traverses the real network.
    assert 1 <= caught.value.retry_after <= 25


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/embeddings", {"input": ["test"], "model": "fixed-embedding-model", "dimensions": 1024}),
        ("/rerank", {"query": "test", "documents": ["one", "two"], "model": "fixed-reranker"}),
    ],
)
def test_retry_after_parser(path, payload):
    assert path.startswith("/")
    assert payload
    assert retry_after_seconds(httpx.Headers({"Retry-After": "37"})) == 37


@pytest.mark.asyncio
async def test_provider_network_errors_fail_over(configured, redis):
    seen = []

    async def handler(request):
        seen.append(request.headers["Authorization"])
        if len(seen) == 1:
            raise httpx.ConnectError("simulated connection reset", request=request)
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JinaFailover(redis, client).post("/embeddings", {"input": ["x"]})
    assert result[0] == 200
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_embeddings_route_uses_fixed_model_and_dimensions(configured, redis):
    from fastapi import FastAPI
    from app.routers.proxy import router
    from app.database.redis_client import get_redis

    app = FastAPI()
    app.include_router(router)

    async def fake_redis():
        return redis

    app.dependency_overrides[get_redis] = fake_redis
    provider_seen = {}

    async def handler(request):
        provider_seen["url"] = str(request.url)
        provider_seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * configured.jina_embedding_dimensions}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as provider_client:
        app.state.http = provider_client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                headers={"Authorization": "Bearer " + "g" * 32},
                json={"model": "caller-model", "input": ["hello"]},
            )
    assert response.status_code == 200
    assert provider_seen["url"].endswith("/embeddings")
    assert provider_seen["payload"]["model"] == configured.jina_embedding_model
    assert provider_seen["payload"]["dimensions"] == configured.jina_embedding_dimensions


@pytest.mark.asyncio
async def test_rerank_route_uses_configured_reranker(configured, redis):
    from fastapi import FastAPI
    from app.routers.proxy import router
    from app.database.redis_client import get_redis

    app = FastAPI()
    app.include_router(router)

    async def fake_redis():
        return redis

    app.dependency_overrides[get_redis] = fake_redis
    provider_seen = {}

    async def handler(request):
        provider_seen["url"] = str(request.url)
        provider_seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as provider_client:
        app.state.http = provider_client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/rerank",
                headers={"Authorization": "Bearer " + "g" * 32},
                json={"model": "caller-model", "query": "q", "documents": ["d"]},
            )
    assert response.status_code == 200
    assert provider_seen["url"].endswith("/rerank")
    assert provider_seen["payload"]["model"] == configured.jina_reranker_model


@pytest.mark.asyncio
async def test_metrics_have_configured_ttl(configured, redis):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))) as client:
        gateway = JinaFailover(redis, client)
        await gateway.metric("ttl_check")
    ttl = await redis.ttl(f"{configured.redis_prefix}:gateway:metrics:ttl_check")
    assert 0 < ttl <= configured.metrics_ttl_seconds
