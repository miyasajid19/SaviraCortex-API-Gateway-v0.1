from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from redis.asyncio import Redis

from app.config import Settings, get_settings


_DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$", re.I)
_ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
_RETRYABLE_STATUSES = {401, 403, 404, 408, 425, 429, 500, 502, 503, 504, 529}


@dataclass(frozen=True)
class JEVResult:
    status_code: int
    body: bytes
    headers: dict[str, str]
    provider: str
    model: str
    fallback: bool


class JEVGatewayError(Exception):
    def __init__(self, message: str, retry_after: int | None = None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(message)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalize_images(images: Any) -> list[dict[str, str]]:
    if images is None:
        return []
    if not isinstance(images, list) or len(images) > 4:
        raise ValueError("images must be a list containing at most four images.")

    normalized: list[dict[str, str]] = []
    total_bytes = 0
    for index, image in enumerate(images):
        if isinstance(image, str):
            match = _DATA_URL.fullmatch(image.strip())
            if not match:
                raise ValueError(f"images[{index}] must be a PNG, JPEG, or WebP base64 data URL.")
            content_type, encoded = match.groups()
        elif isinstance(image, dict):
            content_type = str(image.get("content_type", "")).lower().strip()
            encoded = str(image.get("base64", "")).strip()
        else:
            raise ValueError(f"images[{index}] must be a data URL or image object.")

        if content_type not in _ALLOWED_IMAGE_TYPES or not encoded:
            raise ValueError(f"images[{index}] has an unsupported type or empty base64 data.")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError(f"images[{index}] contains invalid base64 data.") from exc
        if not raw:
            raise ValueError(f"images[{index}] is empty.")
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError(f"images[{index}] exceeds the 4 MiB limit.")
        total_bytes += len(raw)
        normalized.append({"content_type": content_type, "base64": encoded, "_bytes": raw})  # type: ignore[dict-item]

    if total_bytes > 8 * 1024 * 1024:
        raise ValueError("Combined image data exceeds the 8 MiB limit.")
    return normalized


def _retry_after(response: httpx.Response, default: int) -> int:
    raw = response.headers.get("retry-after", "").strip()
    try:
        return max(1, min(3600, int(float(raw)))) if raw else default
    except ValueError:
        return default


class JEVGateway:
    """OpenCode Jev primary with Cloudflare Clef fallback and Redis-backed credential rotation."""

    def __init__(
        self,
        redis: Redis,
        client: httpx.AsyncClient,
        settings: Settings | None = None,
    ):
        self.redis = redis
        self.client = client
        self.settings = settings or get_settings()
        self.prefix = f"{self.settings.redis_prefix}:jev"

    def _state_key(self, provider: str, credential_id: str) -> str:
        return f"{self.prefix}:{provider}:credential:{credential_id}:state"

    async def _metric(self, name: str) -> None:
        key = f"{self.settings.redis_prefix}:gateway:metrics:jev_{name}"
        await self.redis.incr(key)
        await self.redis.expire(key, self.settings.metrics_ttl_seconds)

    async def _ordered_credentials(
        self, provider: str, credentials: list[tuple[str, Any]]
    ) -> list[tuple[str, Any]]:
        if not credentials:
            return []
        cursor_key = f"{self.prefix}:{provider}:rotation:cursor"
        cursor = int(await self.redis.incr(cursor_key))
        await self.redis.expire(cursor_key, self.settings.rotation_ttl_seconds)
        start = (cursor - 1) % len(credentials)
        ordered = credentials[start:] + credentials[:start]

        eligible: list[tuple[str, Any]] = []
        for credential_id, credential in ordered:
            state = await self.redis.get(self._state_key(provider, credential_id))
            if state in {"DISABLED", "COOLDOWN"}:
                continue
            eligible.append((credential_id, credential))
        return eligible

    async def _mark_failure(
        self,
        provider: str,
        credential_id: str,
        response: httpx.Response | None = None,
    ) -> None:
        key = self._state_key(provider, credential_id)
        if response is None:
            await self.redis.set(key, "COOLDOWN", ex=self.settings.jev_retry_cooldown_seconds)
            return
        if response.status_code in {401, 403}:
            await self.redis.set(key, "DISABLED")
        elif response.status_code in {429, 529}:
            await self.redis.set(
                key,
                "COOLDOWN",
                ex=_retry_after(response, self.settings.jev_retry_cooldown_seconds),
            )
        elif response.status_code >= 500 or response.status_code in {408, 425}:
            await self.redis.set(key, "COOLDOWN", ex=self.settings.jev_retry_cooldown_seconds)

    async def _clear_state(self, provider: str, credential_id: str) -> None:
        await self.redis.delete(self._state_key(provider, credential_id))

    @staticmethod
    def _response_headers(response: httpx.Response) -> dict[str, str]:
        allowed = {"content-type", "x-request-id", "retry-after"}
        return {key: value for key, value in response.headers.items() if key.lower() in allowed}

    @staticmethod
    def _cloudflare_result_body(response: httpx.Response) -> bytes:
        # Cloudflare's REST API wraps Workers AI output in {result, success, errors}.
        # Unwrap result so the gateway presents the same System One response shape
        # regardless of which provider answered.
        try:
            data = response.json()
        except ValueError:
            return response.content
        if isinstance(data, dict) and isinstance(data.get("result"), dict):
            return json.dumps(data["result"], separators=(",", ":")).encode("utf-8")
        return response.content

    async def _call_opencode(
        self,
        payload: dict[str, Any],
        images: list[dict[str, Any]],
        *,
        fallback: bool = False,
    ) -> JEVResult | httpx.Response | None:
        credentials = [(_fingerprint(key), key) for key in self.settings.opencode_keys]
        eligible = await self._ordered_credentials("opencode", credentials)
        if not eligible:
            return None

        for credential_id, api_key in eligible:
            request_payload = {
                "model": self.settings.opencode_jev_model,
                "state": payload["state"],
                "questions": payload["questions"],
            }
            headers = {"Authorization": f"Bearer {api_key}"}
            try:
                if images:
                    # Jev image support is experimental and requires the documented opt-in header.
                    assets: dict[str, dict[str, str]] = {}
                    files: list[tuple[str, tuple[Any, ...]]] = [
                        ("request", (None, "", "application/json"))
                    ]
                    image_state: dict[str, Any] = {"input": payload["state"], "images": []}
                    for index, image in enumerate(images):
                        reference = f"jev_image_{index}"
                        part_name = f"jev_image_file_{index}"
                        assets[reference] = {"type": "image", "part": part_name}
                        image_state["images"].append(reference)
                        extension = image["content_type"].split("/", 1)[1]
                        files.append(
                            (part_name, (f"image_{index}.{extension}", image["_bytes"], image["content_type"]))
                        )
                    request_payload["state"] = image_state
                    request_payload["assets"] = assets
                    files[0] = ("request", (None, json.dumps(request_payload), "application/json"))
                    headers["TypeSafe-Danger-Zone-Images"] = "true"
                    response = await self.client.post(
                        self.settings.opencode_jev_base_url,
                        headers=headers,
                        files=files,
                    )
                else:
                    headers["Content-Type"] = "application/json"
                    response = await self.client.post(
                        self.settings.opencode_jev_base_url,
                        headers=headers,
                        json=request_payload,
                    )
            except (httpx.TimeoutException, httpx.NetworkError):
                await self._mark_failure("opencode", credential_id)
                await self._metric("opencode_network_failures")
                continue

            if response.status_code < 400:
                await self._clear_state("opencode", credential_id)
                await self._metric("opencode_success")
                return JEVResult(
                    response.status_code,
                    response.content,
                    self._response_headers(response),
                    "opencode",
                    self.settings.opencode_jev_model,
                    fallback,
                )
            if response.status_code not in _RETRYABLE_STATUSES:
                return response

            await self._mark_failure("opencode", credential_id, response)
            await self._metric(f"opencode_http_{response.status_code}")
        return None

    async def _call_cloudflare(
        self,
        payload: dict[str, Any],
        images: list[dict[str, Any]],
        *,
        model_path: str | None = None,
        model_selector: str | None = None,
        fallback: bool = True,
    ) -> JEVResult | httpx.Response | None:
        selected_model = model_path or self.settings.cloudflare_jev_model
        selected_selector = model_selector or self.settings.cloudflare_jev_selector
        credentials = [
            (_fingerprint(f"{account_id}:{token}"), (account_id, token))
            for account_id, token in self.settings.cloudflare_accounts
        ]
        eligible = await self._ordered_credentials("cloudflare", credentials)
        if not eligible:
            return None

        for credential_id, pair in eligible:
            account_id, token = pair
            request_payload: dict[str, Any] = {
                "model": selected_selector,
                "state": payload["state"],
                "questions": payload["questions"],
            }
            if images:
                request_payload["images"] = [
                    f"data:{image['content_type']};base64,{image['base64']}"
                    for image in images
                ]
            url = (
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{account_id}/ai/run/{selected_model}"
            )
            try:
                response = await self.client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                )
            except (httpx.TimeoutException, httpx.NetworkError):
                await self._mark_failure("cloudflare", credential_id)
                await self._metric("cloudflare_network_failures")
                continue

            if response.status_code < 400:
                await self._clear_state("cloudflare", credential_id)
                await self._metric("cloudflare_success")
                return JEVResult(
                    response.status_code,
                    self._cloudflare_result_body(response),
                    self._response_headers(response),
                    "cloudflare",
                    selected_model,
                    fallback,
                )
            if response.status_code not in _RETRYABLE_STATUSES:
                return response

            await self._mark_failure("cloudflare", credential_id, response)
            await self._metric(f"cloudflare_http_{response.status_code}")
        return None

    async def judge(self, payload: dict[str, Any], task: str = "runtime") -> JEVResult:
        if task not in {"ingestion", "runtime"}:
            raise ValueError("task must be either 'ingestion' or 'runtime'.")
        images = _normalize_images(payload.get("images"))
        await self._metric("requests")

        if task == "ingestion":
            # Ingestion is a high-volume path: prioritize Cloudflare Clef-flash latency.
            primary = await self._call_cloudflare(
                payload,
                images,
                model_path=self.settings.cloudflare_jev_fast_model,
                model_selector=self.settings.cloudflare_jev_fast_selector,
                fallback=False,
            )
            if isinstance(primary, JEVResult):
                return primary
            if isinstance(primary, httpx.Response):
                return JEVResult(
                    primary.status_code,
                    primary.content,
                    self._response_headers(primary),
                    "cloudflare",
                    self.settings.cloudflare_jev_fast_model,
                    False,
                )

            fallback = await self._call_opencode(payload, images, fallback=True)
            if isinstance(fallback, JEVResult):
                return fallback
            if isinstance(fallback, httpx.Response):
                return JEVResult(
                    fallback.status_code,
                    fallback.content,
                    self._response_headers(fallback),
                    "opencode",
                    self.settings.opencode_jev_model,
                    True,
                )
        else:
            # Runtime decisions prioritize Jev; standard Clef is the resilient fallback.
            primary = await self._call_opencode(payload, images, fallback=False)
            if isinstance(primary, JEVResult):
                return primary
            if isinstance(primary, httpx.Response):
                return JEVResult(
                    primary.status_code,
                    primary.content,
                    self._response_headers(primary),
                    "opencode",
                    self.settings.opencode_jev_model,
                    False,
                )

            fallback = await self._call_cloudflare(payload, images, fallback=True)
            if isinstance(fallback, JEVResult):
                return fallback
            if isinstance(fallback, httpx.Response):
                return JEVResult(
                    fallback.status_code,
                    fallback.content,
                    self._response_headers(fallback),
                    "cloudflare",
                    self.settings.cloudflare_jev_model,
                    True,
                )

        await self._metric("unavailable")
        if not self.settings.opencode_keys and not self.settings.cloudflare_accounts:
            raise JEVGatewayError(
                "No JEV provider credentials configured. Set OPENCODE_API_KEYS and/or Cloudflare account/token pairs."
            )
        raise JEVGatewayError(
            "All configured JEV credentials are disabled, cooling down, or unavailable.",
            self.settings.jev_retry_cooldown_seconds,
        )
