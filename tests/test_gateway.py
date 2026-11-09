import asyncio
import json
import time
import uuid

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis

from app.config import get_settings
from app.database.redis_client import get_redis
from app.routers.admin import router as admin_router
from app.services.key_manager import KeyManager, fingerprint
from app.services.failover import JinaFailover, GatewayError, retry_after_seconds
from app.services.fastocr_failover import FastOCRFailover
from app.services.fastocr_key_manager import FastOCRKeyManager


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv(
        "JINA_API_KEYS",
        json.dumps(["jina-test-key-01", "jina-test-key-02", "jina-test-key-03"]),
    )
    monkeypatch.setenv("GATEWAY_API_KEY", "g" * 32)
    monkeypatch.setenv("GATEWAY_ADMIN_API_KEY", "a" * 32)
    monkeypatch.setenv("GATEWAY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("FASTOCR_API_KEYS", json.dumps(["fastocr-account-key-01", "fastocr-account-key-02", "fastocr-account-key-03"]))
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_IDS", json.dumps(["cf-account-01", "cf-account-02"]))
    monkeypatch.setenv("CLOUDFLARE_API_TOKENS", json.dumps(["cf-token-01", "cf-token-02"]))
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


@pytest.mark.asyncio
async def test_cloudflare_rerank_maps_payload_and_fails_over_by_paired_account(configured, redis):
    from fastapi import FastAPI
    from app.routers.proxy import router
    from app.database.redis_client import get_redis

    app = FastAPI()
    app.include_router(router)

    async def fake_redis():
        return redis

    app.dependency_overrides[get_redis] = fake_redis
    calls = []

    async def handler(request):
        calls.append({
            "url": str(request.url),
            "token": request.headers["Authorization"],
            "payload": json.loads(request.content),
        })
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"success": False, "errors": [{"message": "rate limited"}]})
        return httpx.Response(200, json={"success": True, "result": {"response": [{"id": 1, "score": 0.91}]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as provider_client:
        app.state.http = provider_client
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            response = await client.post(
                "/v1/cloudflare/rerank",
                headers={"Authorization": "Bearer " + "g" * 32},
                json={"query": "question", "documents": ["first", "second"], "top_n": 1},
            )

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0]["url"].endswith("/accounts/cf-account-01/ai/run/@cf/baai/bge-reranker-base")
    assert calls[0]["token"] == "Bearer cf-token-01"
    assert calls[1]["url"].endswith("/accounts/cf-account-02/ai/run/@cf/baai/bge-reranker-base")
    assert calls[1]["token"] == "Bearer cf-token-02"
    assert calls[0]["payload"] == {
        "query": "question",
        "contexts": [{"text": "first"}, {"text": "second"}],
        "top_k": 1,
    }
    assert response.json()["result"]["response"][0]["id"] == 1


def test_cloudflare_account_and_token_arrays_pair_by_index(configured):
    assert configured.cloudflare_accounts == (
        ("cf-account-01", "cf-token-01"),
        ("cf-account-02", "cf-token-02"),
    )
    assert configured.cloudflare_reranker_model == "@cf/baai/bge-reranker-base"


def test_cloudflare_account_and_token_array_lengths_must_match(configured, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_IDS", json.dumps(["only-one-account"]))
    monkeypatch.setenv("CLOUDFLARE_API_TOKENS", json.dumps(["token-one", "token-two"]))
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="equal, non-zero lengths"):
            get_settings()
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_cloudflare_embeddings_route_uses_paired_accounts_and_fixed_model(configured, redis):
    from fastapi import FastAPI
    from app.routers.proxy import router
    from app.database.redis_client import get_redis

    app = FastAPI()
    app.include_router(router)

    async def fake_redis():
        return redis

    app.dependency_overrides[get_redis] = fake_redis
    calls = []

    async def handler(request):
        calls.append({
            "url": str(request.url),
            "token": request.headers["Authorization"],
            "payload": json.loads(request.content),
        })
        if len(calls) == 1:
            return httpx.Response(404, json={"errors": [{"code": 7003, "message": "invalid account route"}]})
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1] * 1024}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as provider_client:
        app.state.http = provider_client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/cloudflare/embeddings",
                headers={"Authorization": "Bearer " + "g" * 32},
                json={"model": "caller-model", "input": ["hello"]},
            )

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0]["url"].endswith("/accounts/cf-account-01/ai/v1/embeddings")
    assert calls[0]["token"] == "Bearer cf-token-01"
    assert calls[1]["url"].endswith("/accounts/cf-account-02/ai/v1/embeddings")
    assert calls[1]["token"] == "Bearer cf-token-02"
    assert calls[1]["payload"] == {
        "model": "@cf/qwen/qwen3-embedding-0.6b",
        "input": ["hello"],
    }
    assert len(response.json()["data"][0]["embedding"]) == 1024



def test_cloudflare_account_and_token_lists_accept_unquoted_env_values(configured, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_IDS", "[cf-account-a, cf-account-b]")
    monkeypatch.setenv("CLOUDFLARE_API_TOKENS", "[cf-token-a, cf-token-b]")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.cloudflare_accounts == (
            ("cf-account-a", "cf-token-a"),
            ("cf-account-b", "cf-token-b"),
        )
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_fastocr_create_rotates_on_per_key_throttle_and_pins_document(configured, redis):
    seen = []

    async def handler(request):
        key = request.headers["Authorization"].removeprefix("Bearer ")
        seen.append(key)
        if key == configured.fastocr_keys[0]:
            return httpx.Response(
                429,
                headers={"Retry-After": "60"},
                json={"message": "API key request rate limit exceeded"},
            )
        return httpx.Response(201, json={"id": "fastocr-doc-001"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await FastOCRFailover(redis, client).request(
            "POST",
            "/v1/documents",
            content=b'{"filename":"selected-pages.pdf"}',
            headers={
                "content-type": "application/json",
                "idempotency-key": "fastocr-create-001",
            },
            remember_document=True,
        )

    assert result[0] == 201
    assert seen == list(configured.fastocr_keys[:2])
    manager = FastOCRKeyManager(redis)
    assert await manager.document_key_id("fastocr-doc-001") == fingerprint(configured.fastocr_keys[1])
    assert await manager.key_for_idempotency("fastocr-create-001") == fingerprint(configured.fastocr_keys[1])
    states = {item["id"]: item for item in await manager.snapshot()}
    assert states[fingerprint(configured.fastocr_keys[0])]["status"] == "COOLDOWN"


@pytest.mark.asyncio
async def test_fastocr_document_requests_stay_on_creating_account(configured, redis):
    manager = FastOCRKeyManager(redis)
    expected_key = configured.fastocr_keys[1]
    await manager.bind_document("fastocr-doc-sticky", fingerprint(expected_key))
    seen = []

    async def handler(request):
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json={"id": "fastocr-doc-sticky", "status": "completed"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await FastOCRFailover(redis, client).request(
            "GET",
            "/v1/documents/fastocr-doc-sticky",
            document_id="fastocr-doc-sticky",
        )

    assert result[0] == 200
    assert seen == [f"Bearer {expected_key}"]


@pytest.mark.asyncio
async def test_fastocr_source_address_limit_applies_provider_wide(configured, redis):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(
            429,
            headers={"Retry-After": "30", "X-RateLimit-Scope": "source_address"},
            json={"error": {"code": "throttled", "message": "source address throttle"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = FastOCRFailover(redis, client)
        first = await gateway.request("GET", "/v1/documents")
        assert first[0] == 429
        with pytest.raises(GatewayError) as caught:
            await gateway.request("GET", "/v1/documents")

    assert len(calls) == 1
    assert caught.value.status_code == 429
    assert 0 < caught.value.retry_after <= 30
    assert await FastOCRKeyManager(redis).provider_cooldown_remaining() > 0


@pytest.mark.asyncio
async def test_fastocr_account_concurrency_limit_keeps_document_key(configured, redis):
    manager = FastOCRKeyManager(redis)
    expected_key = configured.fastocr_keys[0]
    kid = fingerprint(expected_key)
    await manager.bind_document("fastocr-doc-busy", kid)
    seen = []

    async def handler(request):
        seen.append(request.headers["Authorization"])
        return httpx.Response(
            429,
            headers={"Retry-After": "20"},
            json={"error": {"code": "concurrency_limit", "message": "two documents already processing"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await FastOCRFailover(redis, client).request(
            "POST",
            "/v1/documents/fastocr-doc-busy/start",
            document_id="fastocr-doc-busy",
        )

    assert result[0] == 429
    assert seen == [f"Bearer {expected_key}"]
    assert (await manager.key_state(kid))["status"] == "COOLDOWN"
    assert await manager.provider_cooldown_remaining() == 0


@pytest.mark.asyncio
async def test_fastocr_create_network_retry_reuses_idempotent_account(configured, redis):
    seen = []

    async def handler(request):
        key = request.headers["Authorization"].removeprefix("Bearer ")
        seen.append(key)
        if len(seen) == 1:
            raise httpx.ConnectError("ambiguous create timeout", request=request)
        return httpx.Response(201, json={"id": "fastocr-doc-retry"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = FastOCRFailover(redis, client)
        kwargs = {
            "method": "POST",
            "path": "/v1/documents",
            "content": b'{"filename":"selected-pages.pdf"}',
            "headers": {
                "content-type": "application/json",
                "idempotency-key": "fastocr-create-retry",
            },
            "remember_document": True,
        }
        with pytest.raises(GatewayError) as caught:
            await gateway.request(**kwargs)
        assert caught.value.status_code == 503
        pinned = await FastOCRKeyManager(redis).key_for_idempotency("fastocr-create-retry")
        result = await gateway.request(**kwargs)

    assert result[0] == 201
    assert seen == [configured.fastocr_keys[0], configured.fastocr_keys[0]]
    assert pinned == fingerprint(configured.fastocr_keys[0])


@pytest.mark.asyncio
async def test_fastocr_document_route_uses_gateway_auth_and_rotated_provider_key(configured, redis):
    from fastapi import FastAPI
    from app.database.redis_client import get_redis
    from app.routers.proxy import router

    app = FastAPI()
    app.include_router(router)

    async def fake_redis():
        return redis

    app.dependency_overrides[get_redis] = fake_redis
    provider_seen = {}

    async def handler(request):
        provider_seen["url"] = str(request.url)
        provider_seen["authorization"] = request.headers["Authorization"]
        provider_seen["idempotency_key"] = request.headers.get("idempotency-key")
        provider_seen["body"] = request.content
        return httpx.Response(201, json={"id": "fastocr-route-doc"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as provider_client:
        app.state.fastocr_http = provider_client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/documents",
                headers={
                    "Authorization": "Bearer " + "g" * 32,
                    "Idempotency-Key": "fastocr-route-idem",
                },
                json={"filename": "selected-pages.pdf", "size_bytes": 1234},
            )

    assert response.status_code == 201
    assert provider_seen["url"] == "https://api.fastocr.org/v1/documents"
    assert provider_seen["authorization"] == f"Bearer {configured.fastocr_keys[0]}"
    assert provider_seen["idempotency_key"] == "fastocr-route-idem"
    assert json.loads(provider_seen["body"])["filename"] == "selected-pages.pdf"
    assert await FastOCRKeyManager(redis).document_key_id("fastocr-route-doc") == fingerprint(configured.fastocr_keys[0])


@pytest.mark.asyncio
async def test_fastocr_page_quota_failure_does_not_rotate_accounts(configured, redis):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(
            403,
            json={"error": {"code": "buy_pages", "message": "monthly page balance exhausted"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await FastOCRFailover(redis, client).request(
            "POST",
            "/v1/documents",
            content=b'{"filename":"selected-pages.pdf"}',
            headers={
                "content-type": "application/json",
                "idempotency-key": "fastocr-page-quota",
            },
            remember_document=True,
        )

    assert result[0] == 403
    assert len(calls) == 1
    states = {item["id"]: item for item in await FastOCRKeyManager(redis).snapshot()}
    assert all(item["status"] == "ACTIVE" for item in states.values())
    assert int(await redis.get(f"{configured.redis_prefix}:gateway:metrics:fastocr_quota_exhausted")) == 1


@pytest.mark.asyncio
async def test_fastocr_concurrency_cooldown_preserves_retry_code(configured, redis):
    manager = FastOCRKeyManager(redis)
    kid = fingerprint(configured.fastocr_keys[0])
    await manager.bind_document("fastocr-doc-concurrency-code", kid)

    async def handler(request):
        return httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={
                "error": {
                    "code": "concurrency_limit",
                    "message": "two documents already processing",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = FastOCRFailover(redis, client)
        first = await gateway.request(
            "POST",
            "/v1/documents/fastocr-doc-concurrency-code/start",
            document_id="fastocr-doc-concurrency-code",
        )
        with pytest.raises(GatewayError) as caught:
            await gateway.request(
                "POST",
                "/v1/documents/fastocr-doc-concurrency-code/start",
                document_id="fastocr-doc-concurrency-code",
            )

    assert first[0] == 429
    assert caught.value.status_code == 429
    assert caught.value.code == "concurrency_limit"
    assert caught.value.retry_after > 0


# FastOCR admin endpoint coverage shares the configured/Redis fixtures above.
@pytest.mark.asyncio
async def test_fastocr_admin_status_and_reenable_key(configured, redis):
    app = FastAPI()
    app.include_router(admin_router)

    async def fake_redis():
        return redis

    app.dependency_overrides[get_redis] = fake_redis
    manager = FastOCRKeyManager(redis)
    disabled_kid = fingerprint(configured.fastocr_keys[0])
    await manager.disable(disabled_kid, "test invalid key")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway",
    ) as client:
        status = await client.get(
            "/admin/fastocr/status",
            headers={"X-Admin-Key": configured.admin_api_key},
        )
        enabled = await client.post(
            f"/admin/fastocr/keys/{disabled_kid}/enable",
            headers={"X-Admin-Key": configured.admin_api_key},
        )

    assert status.status_code == 200
    body = status.json()
    assert body["provider"] == "FastOCR"
    assert body["configured"] is True
    assert body["key_counts"]["total"] == len(configured.fastocr_keys)
    assert body["key_counts"]["disabled"] == 1
    assert "concurrency_limit" in body["metrics"]
    assert all(configured_key not in str(body) for configured_key in configured.fastocr_keys)
    assert enabled.status_code == 200
    assert enabled.json() == {"id": disabled_kid, "status": "ACTIVE"}
    assert (await manager.key_state(disabled_kid))["status"] == "ACTIVE"


@pytest.mark.asyncio
async def test_fastocr_proxy_preserves_concurrency_error_code(configured, redis):
    from app.routers.proxy import router as proxy_router

    app = FastAPI()
    app.include_router(proxy_router)

    async def fake_redis():
        return redis

    app.dependency_overrides[get_redis] = fake_redis
    manager = FastOCRKeyManager(redis)
    kid = fingerprint(configured.fastocr_keys[0])
    await manager.bind_document("fastocr-doc-proxy-cooldown", kid)
    await manager.set_cooldown(kid, 30, "account concurrency limit")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500)
        )
    ) as provider_client:
        app.state.fastocr_http = provider_client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
        ) as client:
            response = await client.post(
                "/v1/documents/fastocr-doc-proxy-cooldown/start",
                headers={"Authorization": "Bearer " + "g" * 32},
            )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "concurrency_limit"
    assert int(response.headers["Retry-After"]) > 0


@pytest.mark.asyncio
async def test_fastocr_sticky_monthly_page_quota_does_not_disable_key(configured, redis):
    manager = FastOCRKeyManager(redis)
    kid = fingerprint(configured.fastocr_keys[0])
    await manager.bind_document("fastocr-doc-sticky-buy-pages", kid)

    async def handler(request):
        return httpx.Response(
            403,
            json={
                "error": {
                    "code": "buy_pages",
                    "message": "monthly page allowance exhausted",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await FastOCRFailover(redis, client).request(
            "POST",
            "/v1/documents/fastocr-doc-sticky-buy-pages/start",
            document_id="fastocr-doc-sticky-buy-pages",
        )

    assert result[0] == 403
    assert (await manager.key_state(kid))["status"] == "ACTIVE"
    assert int(
        await redis.get(
            f"{configured.redis_prefix}:gateway:metrics:fastocr_quota_exhausted"
        )
    ) == 1
