from __future__ import annotations

import email.utils
import time
from datetime import timezone

import httpx
from redis.asyncio import Redis

from app.config import get_settings
from app.services.key_manager import KeyManager


class GatewayError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        retry_after: int | None = None,
        code: str | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after
        self.code = code
        super().__init__(message)


def retry_after_seconds(headers: httpx.Headers) -> int | None:
    raw = headers.get("retry-after")
    if raw:
        try:
            return max(1, int(float(raw)))
        except ValueError:
            try:
                dt = email.utils.parsedate_to_datetime(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return max(1, int(dt.timestamp() - time.time()))
            except (TypeError, ValueError, OverflowError):
                pass
    for name in (
        "x-ratelimit-reset",
        "ratelimit-reset",
        "x-rate-limit-reset",
        "x-ratelimit-reset-requests",
    ):
        raw = headers.get(name)
        if raw:
            try:
                value = float(raw)
                return max(1, int(value - time.time())) if value > 1_000_000_000 else max(1, int(value))
            except ValueError:
                continue
    return None


def is_shared_quota(headers: httpx.Headers, body: bytes) -> bool:
    scope = (headers.get("x-ratelimit-scope") or headers.get("x-quota-scope") or "").lower()
    if scope in {"org", "organization", "account", "project", "provider"}:
        return True
    text = body[:4000].decode("utf-8", "ignore").lower()
    return any(
        term in text
        for term in (
            "organization_quota",
            "organization quota",
            "account quota",
            "shared quota",
            "billing hard limit",
            "org-wide quota",
        )
    )


class JinaFailover:
    def __init__(self, redis: Redis, client: httpx.AsyncClient):
        self.redis, self.client = redis, client
        self.settings = get_settings()
        self.keys = KeyManager(redis)

    async def metric(self, name: str, amount: int = 1):
        key = f"{self.settings.redis_prefix}:gateway:metrics:{name}"
        await self.redis.incrby(key, amount)
        await self.redis.expire(key, self.settings.metrics_ttl_seconds)

    async def post(self, path: str, payload: dict) -> tuple[int, dict, bytes]:
        started = time.perf_counter()
        await self.metric("requests")
        attempted: set[str] = set()
        last_error: str | None = None
        last_retry_after: int | None = None

        for _ in range(self.settings.max_attempts):
            selected = await self.keys.reserve()
            if selected is None:
                break
            kid, secret = selected
            if kid in attempted:
                break
            attempted.add(kid)
            try:
                response = await self.client.post(
                    self.settings.jina_base_url + path,
                    headers={
                        "Authorization": f"Bearer {secret}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                raw = response.content
                if response.status_code < 400:
                    await self.keys.clear_backoff(kid)
                    await self.metric("success")
                    await self.metric(
                        "latency_ms_total",
                        int((time.perf_counter() - started) * 1000),
                    )
                    return response.status_code, dict(response.headers), raw

                status = response.status_code
                if status == 429:
                    shared = is_shared_quota(response.headers, raw)
                    wait = retry_after_seconds(response.headers)
                    if wait is None:
                        wait = await self.keys.next_backoff(kid)
                    last_retry_after = wait
                    await self.keys.set_cooldown(
                        kid, wait, "HTTP 429 rate limit", provider_wide=shared
                    )
                    await self.metric("rate_limited")
                    if shared:
                        await self.metric("shared_quota_limited")
                        break
                    await self.metric("automatic_failovers")
                    continue
                if status in (401, 403):
                    await self.keys.disable(
                        kid, f"HTTP {status}: invalid or unauthorized credential"
                    )
                    await self.metric("disabled_keys")
                    await self.metric("automatic_failovers")
                    continue
                if status in (500, 502, 503, 504):
                    await self.keys.set_cooldown(
                        kid,
                        self.settings.transient_cooldown_seconds,
                        f"temporary provider HTTP {status}",
                    )
                    await self.metric("provider_errors")
                    await self.metric("automatic_failovers")
                    continue

                # Client errors (including 400) are not retried or rotated.
                await self.metric("client_errors" if status < 500 else "failed")
                return status, dict(response.headers), raw

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = type(exc).__name__
                await self.metric("network_errors")
                await self.keys.set_cooldown(
                    kid,
                    self.settings.transient_cooldown_seconds,
                    f"network failure: {last_error}",
                )
                await self.metric("automatic_failovers")

        await self.metric("failed")
        snapshot = await self.keys.snapshot()
        waits = [
            key["cooldown_remaining_seconds"]
            for key in snapshot
            if key["status"] == "COOLDOWN" and key["cooldown_remaining_seconds"] > 0
        ]
        if last_error and not waits:
            raise GatewayError(
                503, f"Provider network error ({last_error}); no eligible key remains."
            )
        if waits:
            last_retry_after = min(waits)
        if last_retry_after:
            raise GatewayError(
                503,
                "No Jina key is currently eligible; rate-limit cooldown is active.",
                last_retry_after,
            )
        raise GatewayError(
            503, "No Jina API keys are configured or all keys are disabled."
        )
