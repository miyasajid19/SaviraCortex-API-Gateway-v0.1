from __future__ import annotations

import json
import time
from typing import Iterable

import httpx
from redis.asyncio import Redis

from app.config import get_settings
from app.services.failover import GatewayError, retry_after_seconds
from app.services.fastocr_key_manager import FastOCRKeyManager


class FastOCRFailover:
    """Rotate FastOCR credentials on safe, per-key failures.

    A document is pinned to the key/account that created it. Later start, poll,
    and output calls stay on that account so document ownership is never mixed.
    """

    def __init__(self, redis: Redis, client: httpx.AsyncClient):
        self.redis = redis
        self.client = client
        self.settings = get_settings()
        self.keys = FastOCRKeyManager(redis)

    async def metric(self, name: str, amount: int = 1) -> None:
        key = f"{self.settings.redis_prefix}:gateway:metrics:fastocr_{name}"
        await self.redis.incrby(key, amount)
        await self.redis.expire(key, self.settings.metrics_ttl_seconds)

    @staticmethod
    def _error_fields(body: bytes) -> tuple[str, str]:
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "", ""
        if not isinstance(parsed, dict):
            return "", ""
        nested = parsed.get("error")
        fields = nested if isinstance(nested, dict) else {}
        code = str(fields.get("code") or parsed.get("code") or "").lower()
        message = str(
            fields.get("message")
            or parsed.get("message")
            or (nested if isinstance(nested, str) else "")
        ).lower()
        return code, message

    @classmethod
    def _is_source_address_limit(
        cls, headers: httpx.Headers, body: bytes
    ) -> bool:
        scope = (
            headers.get("x-ratelimit-scope")
            or headers.get("x-quota-scope")
            or ""
        ).lower()
        if any(term in scope for term in ("ip", "address", "source", "shared")):
            return True
        code, message = cls._error_fields(body)
        if code in {
            "throttled",
            "source_address_throttled",
            "source_ip_throttled",
            "ip_rate_limited",
        }:
            # FastOCR documents code=throttled for its shared gateway throttle;
            # per-key and per-address throttles return a message without a code.
            return True
        return any(
            term in message
            for term in (
                "source address",
                "source ip",
                "ip address rate",
                "ip rate limit",
                "per-ip",
                "shared throttle",
                "shared rate limit",
                "shared gateway limit",
            )
        )

    @staticmethod
    def _response_tuple(response: httpx.Response) -> tuple[int, dict, bytes]:
        allowed = {"content-type", "x-request-id", "retry-after"}
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in allowed
        }
        return response.status_code, headers, response.content

    async def _send(
        self,
        method: str,
        path: str,
        *,
        secret: str,
        params: Iterable[tuple[str, str]],
        content: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        outgoing_headers = dict(headers)
        outgoing_headers["Authorization"] = f"Bearer {secret}"
        return await self.client.request(
            method,
            self.settings.fastocr_base_url + path,
            params=list(params),
            content=content,
            headers=outgoing_headers,
        )

    async def _sticky_request(
        self,
        method: str,
        path: str,
        *,
        kid: str,
        secret: str,
        params: Iterable[tuple[str, str]],
        content: bytes,
        headers: dict[str, str],
        document_id: str | None,
    ) -> tuple[int, dict, bytes]:
        state = await self.keys.key_state(kid)
        if state["status"] == "DISABLED":
            raise GatewayError(
                503,
                "The FastOCR key assigned to this document is disabled; re-enable it before retrying.",
            )
        if state["status"] == "COOLDOWN":
            raise GatewayError(
                429,
                "The FastOCR key assigned to this document is cooling down.",
                state["cooldown_remaining_seconds"] or 1,
                code=(
                    "concurrency_limit"
                    if "account concurrency limit" in state["last_reason"]
                    else None
                ),
            )

        try:
            response = await self._send(
                method, path, secret=secret,
                params=params, content=content, headers=headers,
            )
        except httpx.RequestError as exc:
            wait = self.settings.fastocr_transient_cooldown_seconds
            await self.keys.set_cooldown(kid, wait, f"network failure: {type(exc).__name__}")
            await self.metric("network_errors")
            raise GatewayError(
                503,
                f"FastOCR network error ({type(exc).__name__}); retry this document with its assigned account.",
                wait,
            ) from exc

        status = response.status_code
        if status < 400:
            await self.metric("success")
            await self.keys.clear_backoff(kid)
            if method == "DELETE" and document_id:
                await self.keys.unbind_document(document_id)
            return self._response_tuple(response)

        if status == 429:
            wait = retry_after_seconds(response.headers)
            if wait is None:
                wait = await self.keys.next_backoff(kid)
            if self._is_source_address_limit(response.headers, response.content):
                await self.keys.set_cooldown(
                    kid, wait, "shared source-address throttle", provider_wide=True
                )
                await self.metric("shared_rate_limited")
            else:
                code, _ = self._error_fields(response.content)
                reason = (
                    "account concurrency limit"
                    if code == "concurrency_limit"
                    else "per-key request rate limit"
                )
                await self.keys.set_cooldown(kid, wait, reason)
                await self.metric("rate_limited")
                if code == "concurrency_limit":
                    await self.metric("concurrency_limit")
            return self._response_tuple(response)

        if status in (401, 403):
            code, message = self._error_fields(response.content)
            if (
                code in {"buy_pages", "page_quota_exceeded", "page_limit_exceeded"}
                or "buy_pages" in message
            ):
                await self.metric("quota_exhausted")
                return self._response_tuple(response)
            await self.keys.disable(kid, f"FastOCR HTTP {status}: invalid or unauthorized key")
            await self.metric("disabled_keys")
        elif status in (500, 502, 503, 504):
            await self.keys.set_cooldown(
                kid,
                self.settings.fastocr_transient_cooldown_seconds,
                f"temporary FastOCR HTTP {status}",
            )
            await self.metric("provider_errors")
        else:
            await self.metric("client_errors")
        return self._response_tuple(response)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Iterable[tuple[str, str]] = (),
        content: bytes = b"",
        headers: dict[str, str] | None = None,
        document_id: str | None = None,
        remember_document: bool = False,
    ) -> tuple[int, dict, bytes]:
        if not self.settings.fastocr_keys:
            raise GatewayError(
                503,
                "No FastOCR API keys configured. Set FASTOCR_API_KEYS in api-gateway/.env.",
            )

        started = time.perf_counter()
        await self.metric("requests")
        request_headers = headers or {}

        provider_wait = await self.keys.provider_cooldown_remaining()
        if provider_wait:
            raise GatewayError(
                429,
                "FastOCR is cooling down after a shared source-address rate limit.",
                provider_wait,
                code="throttled",
            )

        if document_id:
            sticky_kid = await self.keys.document_key_id(document_id)
            if sticky_kid:
                secret = self.keys.secret_for(sticky_kid)
                if not secret:
                    raise GatewayError(503, "The FastOCR key assigned to this document is no longer configured.")
                result = await self._sticky_request(
                    method, path, kid=sticky_kid, secret=secret,
                    params=params, content=content, headers=request_headers,
                    document_id=document_id,
                )
                if result[0] < 400:
                    await self.metric(
                        "latency_ms_total",
                        int((time.perf_counter() - started) * 1000),
                    )
                return result

        attempted: set[str] = set()
        last_error: str | None = None
        create_request = method.upper() == "POST" and path == "/v1/documents"
        idempotency_key = next(
            (
                value
                for key, value in request_headers.items()
                if key.lower() == "idempotency-key"
            ),
            None,
        )
        idempotency_kid = (
            await self.keys.key_for_idempotency(idempotency_key)
            if create_request and idempotency_key
            else None
        )
        if idempotency_kid:
            pinned_secret = self.keys.secret_for(idempotency_kid)
            if not pinned_secret:
                raise GatewayError(
                    503,
                    "The FastOCR key pinned to this idempotent create request is no longer configured.",
                )
            pinned_state = await self.keys.key_state(idempotency_kid)
            if pinned_state["status"] == "DISABLED":
                raise GatewayError(
                    503,
                    "The FastOCR key pinned to this idempotent create request is disabled.",
                )
            if pinned_state["status"] == "COOLDOWN":
                raise GatewayError(
                    429,
                    "The FastOCR key pinned to this idempotent create request is cooling down.",
                    pinned_state["cooldown_remaining_seconds"] or 1,
                    code=(
                        "concurrency_limit"
                        if "account concurrency limit" in pinned_state["last_reason"]
                        else None
                    ),
                )

        for _ in range(self.settings.fastocr_max_attempts):
            if idempotency_kid:
                selected = (idempotency_kid, self.keys.secret_for(idempotency_kid))
                if selected[1] is None:
                    raise GatewayError(503, "The pinned FastOCR key is no longer configured.")
            else:
                selected = await self.keys.reserve()
            if selected is None:
                break
            kid, secret = selected
            if kid in attempted:
                break
            attempted.add(kid)
            if create_request and idempotency_key:
                await self.keys.bind_idempotency(idempotency_key, kid)
                idempotency_kid = kid

            try:
                response = await self._send(
                    method, path, secret=secret,
                    params=params, content=content, headers=request_headers,
                )
            except httpx.RequestError as exc:
                last_error = type(exc).__name__
                await self.metric("network_errors")
                if create_request:
                    # Do not retry document creation with another account after
                    # an ambiguous network result; that could create a duplicate.
                    raise GatewayError(
                        503,
                        f"FastOCR network error ({last_error}); retry the same idempotent request.",
                        self.settings.fastocr_transient_cooldown_seconds,
                    ) from exc
                await self.keys.set_cooldown(
                    kid,
                    self.settings.fastocr_transient_cooldown_seconds,
                    f"network failure: {last_error}",
                )
                await self.metric("automatic_failovers")
                continue

            status = response.status_code
            if status < 400:
                await self.keys.clear_backoff(kid)
                await self.metric("success")
                if remember_document:
                    try:
                        body = response.json()
                        new_document_id = body.get("id") if isinstance(body, dict) else None
                    except ValueError:
                        new_document_id = None
                    if isinstance(new_document_id, str) and new_document_id:
                        await self.keys.bind_document(new_document_id, kid)
                await self.metric(
                    "latency_ms_total",
                    int((time.perf_counter() - started) * 1000),
                )
                return self._response_tuple(response)

            if status == 429:
                wait = retry_after_seconds(response.headers)
                if wait is None:
                    wait = await self.keys.next_backoff(kid)
                if self._is_source_address_limit(response.headers, response.content):
                    await self.keys.set_cooldown(
                        kid, wait, "shared source-address throttle", provider_wide=True
                    )
                    await self.metric("shared_rate_limited")
                    return self._response_tuple(response)

                code, message = self._error_fields(response.content)
                if code in {"buy_pages", "page_quota_exceeded", "page_limit_exceeded"} or "buy_pages" in message:
                    await self.metric("quota_exhausted")
                    return self._response_tuple(response)
                if code == "concurrency_limit":
                    await self.keys.set_cooldown(kid, wait, "account concurrency limit")
                    await self.metric("rate_limited")
                    return self._response_tuple(response)

                await self.keys.set_cooldown(kid, wait, "per-key request rate limit")
                await self.metric("rate_limited")
                # A 429 on create is definitive: FastOCR rejected it before
                # creating a document, so another account can safely be tried.
                if create_request:
                    if idempotency_key:
                        await self.keys.unbind_idempotency(idempotency_key)
                    idempotency_kid = None
                    await self.metric("automatic_failovers")
                    continue
                if method.upper() == "GET" and path == "/v1/documents":
                    await self.metric("automatic_failovers")
                    continue
                return self._response_tuple(response)

            if status in (401, 403):
                code, message = self._error_fields(response.content)
                if code in {"buy_pages", "page_quota_exceeded", "page_limit_exceeded"} or "buy_pages" in message:
                    await self.metric("quota_exhausted")
                    return self._response_tuple(response)
                await self.keys.disable(kid, f"FastOCR HTTP {status}: invalid or unauthorized key")
                await self.metric("disabled_keys")
                if create_request and idempotency_key:
                    await self.keys.unbind_idempotency(idempotency_key)
                    idempotency_kid = None
                await self.metric("automatic_failovers")
                continue

            if status in (500, 502, 503, 504):
                await self.metric("provider_errors")
                if create_request:
                    # Keep the same account for idempotent create retries.
                    return self._response_tuple(response)
                await self.keys.set_cooldown(
                    kid,
                    self.settings.fastocr_transient_cooldown_seconds,
                    f"temporary FastOCR HTTP {status}",
                )
                await self.metric("automatic_failovers")
                continue

            await self.metric("client_errors")
            return self._response_tuple(response)

        await self.metric("failed")
        snapshot = await self.keys.snapshot()
        waits = [
            item["cooldown_remaining_seconds"]
            for item in snapshot
            if item["status"] == "COOLDOWN" and item["cooldown_remaining_seconds"] > 0
        ]
        if waits:
            last_retry_after = min(waits)
            raise GatewayError(
                429,
                "No FastOCR key is currently eligible; per-key cooldown is active.",
                last_retry_after,
            )
        if last_error:
            raise GatewayError(
                503,
                f"FastOCR network error ({last_error}); no eligible key remains.",
            )
        raise GatewayError(
            503,
            "No FastOCR API keys are active; check configured keys or re-enable a disabled key.",
        )
