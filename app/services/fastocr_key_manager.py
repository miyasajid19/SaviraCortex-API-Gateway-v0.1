from __future__ import annotations

import hashlib
import json
import time

from redis.asyncio import Redis

from app.config import get_settings
from app.services.key_manager import RESERVE_SCRIPT, fingerprint


class FastOCRKeyManager:
    """Redis-backed FastOCR key rotation and per-document account affinity."""

    def __init__(self, redis: Redis):
        self.redis = redis
        self.settings = get_settings()
        self.prefix = f"{self.settings.redis_prefix}:fastocr"
        self.keys = tuple(self.settings.fastocr_keys)

    def _key(self, suffix: str) -> str:
        return f"{self.prefix}:{suffix}"

    def status_key(self, kid: str) -> str:
        return self._key(f"key:{kid}:status")

    def reason_key(self, kid: str) -> str:
        return self._key(f"key:{kid}:last_reason")

    def backoff_key(self, kid: str) -> str:
        return self._key(f"key:{kid}:backoff")

    def document_key_key(self, document_id: str) -> str:
        return self._key(f"document:{document_id}:key")

    def idempotency_key_key(self, idempotency_key: str) -> str:
        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        return self._key(f"idempotency:{digest}:key")

    @property
    def ids(self) -> list[str]:
        return [fingerprint(key) for key in self.keys]

    def secret_for(self, kid: str) -> str | None:
        return next((key for key in self.keys if fingerprint(key) == kid), None)

    async def reserve(self) -> tuple[str, str] | None:
        mapping = {fingerprint(key): key for key in self.keys}
        if not mapping:
            return None
        state_keys = [self.status_key(kid) for kid in mapping]
        result = await self.redis.eval(
            RESERVE_SCRIPT,
            3 + len(state_keys),
            self._key("rotation:cursor"),
            self._key("provider:cooldown_until"),
            self._key("last_selected"),
            *state_keys,
            json.dumps(list(mapping)),
            time.time(),
            self.settings.rotation_ttl_seconds,
        )
        if not result:
            return None
        kid = result[0]
        return kid, mapping[kid]

    async def key_state(self, kid: str) -> dict:
        status = await self.redis.get(self.status_key(kid)) or "ACTIVE"
        ttl = int(await self.redis.ttl(self.status_key(kid)))
        reason = await self.redis.get(self.reason_key(kid)) or ""
        if status == "COOLDOWN" and ttl < 0:
            await self.redis.delete(self.status_key(kid), self.reason_key(kid))
            status, ttl, reason = "ACTIVE", -2, ""
        return {
            "status": status,
            "cooldown_remaining_seconds": max(0, ttl) if status == "COOLDOWN" else 0,
            "last_reason": reason,
        }

    async def document_key_id(self, document_id: str) -> str | None:
        return await self.redis.get(self.document_key_key(document_id))

    async def bind_document(self, document_id: str, kid: str) -> None:
        await self.redis.set(
            self.document_key_key(document_id),
            kid,
            ex=self.settings.fastocr_document_key_ttl_seconds,
        )

    async def unbind_document(self, document_id: str) -> None:
        await self.redis.delete(self.document_key_key(document_id))

    async def key_for_idempotency(self, idempotency_key: str) -> str | None:
        return await self.redis.get(self.idempotency_key_key(idempotency_key))

    async def bind_idempotency(self, idempotency_key: str, kid: str) -> None:
        await self.redis.set(
            self.idempotency_key_key(idempotency_key),
            kid,
            ex=self.settings.fastocr_document_key_ttl_seconds,
        )

    async def unbind_idempotency(self, idempotency_key: str) -> None:
        await self.redis.delete(self.idempotency_key_key(idempotency_key))

    async def provider_cooldown_remaining(self) -> int:
        until = float(await self.redis.get(self._key("provider:cooldown_until")) or 0)
        return max(0, round(until - time.time()))

    async def set_cooldown(
        self,
        kid: str,
        seconds: int,
        reason: str,
        *,
        provider_wide: bool = False,
    ) -> None:
        wait = max(1, int(seconds))
        if provider_wide:
            await self.redis.set(
                self._key("provider:cooldown_until"),
                time.time() + wait,
                ex=wait,
            )
            return
        await self.redis.set(self.status_key(kid), "COOLDOWN", ex=wait)
        await self.redis.set(self.reason_key(kid), reason[:180], ex=wait)

    async def disable(self, kid: str, reason: str) -> None:
        await self.redis.set(self.status_key(kid), "DISABLED")
        await self.redis.set(self.reason_key(kid), reason[:180])

    async def enable(self, kid: str) -> None:
        if kid not in self.ids:
            raise KeyError(kid)
        await self.redis.delete(
            self.status_key(kid),
            self.reason_key(kid),
            self.backoff_key(kid),
        )

    async def next_backoff(self, kid: str) -> int:
        key = self.backoff_key(kid)
        count = await self.redis.incr(key)
        await self.redis.expire(key, self.settings.rotation_ttl_seconds)
        return min(
            self.settings.fastocr_cooldown_max_seconds,
            self.settings.fastocr_cooldown_base_seconds
            * (2 ** min(count - 1, 10)),
        )

    async def clear_backoff(self, kid: str) -> None:
        await self.redis.delete(self.backoff_key(kid))

    async def snapshot(self) -> list[dict]:
        mapping = {fingerprint(key): key for key in self.keys}
        pipe = self.redis.pipeline(transaction=False)
        for kid in mapping:
            pipe.get(self.status_key(kid))
            pipe.ttl(self.status_key(kid))
            pipe.get(self.reason_key(kid))
        values = await pipe.execute()

        result = []
        for index, (kid, secret) in enumerate(mapping.items()):
            status = values[index * 3] or "ACTIVE"
            ttl = int(values[index * 3 + 1])
            reason = values[index * 3 + 2] or ""
            if status == "COOLDOWN" and ttl < 0:
                await self.redis.delete(self.status_key(kid), self.reason_key(kid))
                status, ttl, reason = "ACTIVE", -2, ""
            result.append({
                "id": kid,
                "masked": f"{secret[:3]}...{secret[-3:]}" if len(secret) > 8 else "********",
                "status": status,
                "cooldown_remaining_seconds": max(0, ttl) if status == "COOLDOWN" else 0,
                "last_reason": reason,
            })
        return result
