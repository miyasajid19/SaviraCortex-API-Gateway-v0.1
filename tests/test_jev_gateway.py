from __future__ import annotations

import base64
from types import SimpleNamespace

import httpx
import pytest

from app.services.jev_gateway import JEVGateway


class FakeRedis:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def expire(self, key: str, seconds: int) -> bool:
        return True

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self.values[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += int(key in self.values)
            self.values.pop(key, None)
        return removed


def settings(*, opencode_keys=("open-key-1", "open-key-2"), cloudflare_accounts=(("cf-account", "cf-token"),)):
    return SimpleNamespace(
        redis_prefix="test:gateway",
        rotation_ttl_seconds=3600,
        metrics_ttl_seconds=3600,
        opencode_keys=tuple(opencode_keys),
        opencode_jev_model="jev-1.13-free",
        opencode_jev_base_url="https://opencode.ai/inference/systemone/v1/systemone",
        opencode_jev_timeout_seconds=30,
        cloudflare_accounts=tuple(cloudflare_accounts),
        cloudflare_jev_model="@cf/cloudflare/clef",
        cloudflare_jev_selector="clef",
        cloudflare_jev_fast_model="@cf/cloudflare/clef-flash",
        cloudflare_jev_fast_selector="clef-flash",
        cloudflare_jev_omni_model="@cf/cloudflare/clef-omni",
        cloudflare_jev_omni_selector="clef-omni",
        jev_retry_cooldown_seconds=5,
    )


def decision_payload(images=None):
    payload = {
        "state": {"user_input": "Hello", "context": []},
        "questions": {
            "safe": {
                "type": "noul",
                "instructions": "Is this request safe?",
            }
        },
    }
    if images is not None:
        payload["images"] = images
    return payload


@pytest.mark.asyncio
async def test_opencode_primary_success_uses_server_model_and_bearer_key():
    calls = []

    async def handler(request: httpx.Request):
        calls.append(request)
        return httpx.Response(200, json={"model": "jev-1.13-free", "answers": {"safe": {"type": "noul", "noul": 0.99}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JEVGateway(FakeRedis(), client, settings()).judge(decision_payload())

    assert result.provider == "opencode"
    assert result.model == "jev-1.13-free"
    assert result.fallback is False
    assert calls[0].url == "https://opencode.ai/inference/systemone/v1/systemone"
    assert calls[0].headers["authorization"] == "Bearer open-key-1"
    assert b'"model":"jev-1.13-free"' in calls[0].content


@pytest.mark.asyncio
async def test_all_opencode_rate_limits_fall_back_to_cloudflare_clef():
    calls = []

    async def handler(request: httpx.Request):
        calls.append(request)
        if "opencode.ai" in str(request.url):
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "rate limited"})
        return httpx.Response(200, json={
            "result": {"model": "clef", "answers": {"safe": {"type": "noul", "noul": 0.98}}, "usage": {}},
            "success": True,
            "errors": [],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JEVGateway(FakeRedis(), client, settings()).judge(decision_payload())

    assert result.provider == "cloudflare"
    assert result.model == "@cf/cloudflare/clef"
    assert result.fallback is True
    assert b'"answers"' in result.body
    assert b'"result"' not in result.body
    assert len(calls) == 3
    assert calls[-1].url.path.endswith("/ai/run/@cf/cloudflare/clef")
    assert calls[-1].headers["authorization"] == "Bearer cf-token"
    assert b'"model":"clef"' in calls[-1].content


@pytest.mark.asyncio
async def test_non_retryable_primary_validation_error_does_not_call_fallback():
    calls = []

    async def handler(request: httpx.Request):
        calls.append(request)
        return httpx.Response(422, json={"detail": "invalid request"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JEVGateway(FakeRedis(), client, settings()).judge(decision_payload())

    assert result.provider == "opencode"
    assert result.status_code == 422
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_cloudflare_only_configuration_works_and_preserves_images():
    image_bytes = b"fake-png-image"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    calls = []

    async def handler(request: httpx.Request):
        calls.append(request)
        return httpx.Response(200, json={"model": "clef", "answers": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JEVGateway(
            FakeRedis(),
            client,
            settings(opencode_keys=(), cloudflare_accounts=(("cf-account", "cf-token"),)),
        ).judge(decision_payload(images=[{"content_type": "image/png", "base64": encoded}]))

    assert result.provider == "cloudflare"
    assert result.fallback is True
    assert b"data:image/png;base64," + encoded.encode() in calls[0].content


@pytest.mark.asyncio
async def test_opencode_image_request_uses_documented_multipart_assets_opt_in():
    image_bytes = b"fake-webp-image"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    calls = []

    async def handler(request: httpx.Request):
        calls.append(request)
        return httpx.Response(200, json={"model": "jev-1.13-free", "answers": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JEVGateway(
            FakeRedis(),
            client,
            settings(opencode_keys=("open-key-1",), cloudflare_accounts=()),
        ).judge(decision_payload(images=[{"content_type": "image/webp", "base64": encoded}]))

    assert result.provider == "opencode"
    assert calls[0].headers["typeSafe-danger-zone-images"] == "true"
    assert "multipart/form-data" in calls[0].headers["content-type"]
    assert b"jev_image_0" in calls[0].content
    assert image_bytes in calls[0].content


@pytest.mark.asyncio
async def test_ingestion_task_uses_clefflash_as_primary():
    calls = []

    async def handler(request: httpx.Request):
        calls.append(request)
        return httpx.Response(200, json={
            "result": {"model": "clef-flash", "answers": {"safe": {"type": "noul", "noul": 0.99}}},
            "success": True,
            "errors": [],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JEVGateway(
            FakeRedis(),
            client,
            settings(),
        ).judge(decision_payload(), task="ingestion")

    assert result.provider == "cloudflare"
    assert result.model == "@cf/cloudflare/clef-flash"
    assert result.fallback is False
    assert len(calls) == 1
    assert calls[0].url.path.endswith("/ai/run/@cf/cloudflare/clef-flash")
    assert b'"model":"clef-flash"' in calls[0].content


@pytest.mark.asyncio
async def test_ingestion_falls_back_to_opencode_when_clefflash_pool_fails():
    calls = []

    async def handler(request: httpx.Request):
        calls.append(request)
        if "api.cloudflare.com" in str(request.url):
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"errors": ["busy"]})
        return httpx.Response(200, json={"model": "jev-1.13-free", "answers": {"safe": {"type": "noul", "noul": 0.99}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JEVGateway(FakeRedis(), client, settings()).judge(
            decision_payload(),
            task="ingestion",
        )

    assert result.provider == "opencode"
    assert result.model == "jev-1.13-free"
    assert result.fallback is True
    assert len(calls) == 2
    assert calls[0].url.path.endswith("/ai/run/@cf/cloudflare/clef-flash")
    assert calls[1].url.host == "opencode.ai"


@pytest.mark.asyncio
async def test_unknown_task_is_rejected_before_provider_calls():
    gateway = JEVGateway(FakeRedis(), httpx.AsyncClient(), settings())
    with pytest.raises(ValueError, match="task must be either"):
        await gateway.judge(decision_payload(), task="arbitrary-model")
    await gateway.client.aclose()
