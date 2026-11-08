from __future__ import annotations

import hashlib
import json
import time

from redis.asyncio import Redis

from app.config import get_settings


RESERVE_SCRIPT = """
local ids = cjson.decode(ARGV[1])
local now = tonumber(ARGV[2])
local rotation_ttl = tonumber(ARGV[3])
local n = #ids
if n == 0 then return nil end

local provider_until = tonumber(redis.call('GET', KEYS[2]) or '0')
if provider_until > now then return nil end

local cursor = tonumber(redis.call('GET', KEYS[1]) or '0')
for offset = 0, n - 1 do
  local idx = (cursor + offset) % n + 1
  local id = ids[idx]
  local status = redis.call('GET', KEYS[3 + idx])
  -- Missing state means ACTIVE. COOLDOWN keys expire automatically;
  -- DISABLED keys intentionally persist until an administrator re-enables them.
  if (not status) or status == 'ACTIVE' then
    redis.call('SET', KEYS[1], idx % n, 'EX', rotation_ttl)
    redis.call('SET', KEYS[3], id, 'EX', rotation_ttl)
    return {id, tostring(idx)}
  end
end
return nil
"""


def fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]


class KeyManager:
    def __init__(self, redis: Redis, prefix: str | None = None):
        self.redis = redis
        self.settings = get_settings()
        self.prefix = (prefix or self.settings.redis_prefix).strip().strip(":")

    def _key(self, suffix: str) -> str:
        return f"{self.prefix}:{suffix}"

    def status_key(self, kid: str) -> str:
        return self._key(f"jina:key:{kid}:status")

    def reason_key(self, kid: str) -> str:
        return self._key(f"jina:key:{kid}:last_reason")

    def backoff_key(self, kid: str) -> str:
        return self._key(f"jina:key:{kid}:backoff")

    @property
    def ids(self) -> list[str]:
        return [fingerprint(key) for key in self.settings.keys]

    async def reserve(self) -> tuple[str, str] | None:
        mapping = {fingerprint(key): key for key in self.settings.keys}
        if not mapping:
            return None
        state_keys = [self.status_key(kid) for kid in mapping]
        result = await self.redis.eval(
            RESERVE_SCRIPT,
            3 + len(state_keys),
            self._key("jina:rotation:cursor"),
            self._key("jina:provider:cooldown_until"),
            self._key("jina:last_selected"),
            *state_keys,
            json.dumps(list(mapping)),
            time.time(),
            self.settings.rotation_ttl_seconds,
        )
        if not result:
            return None
        kid = result[0]
        return kid, mapping[kid]

    async def set_cooldown(
        self, kid: str, seconds: int, reason: str, provider_wide: bool = False
    ):
        wait = max(1, int(seconds))
        until = time.time() + wait
        if provider_wide:
            await self.redis.set(
                self._key("jina:provider:cooldown_until"), until, ex=wait
            )
        else:
            await self.redis.set(self.status_key(kid), "COOLDOWN", ex=wait)
            await self.redis.set(self.reason_key(kid), reason[:180], ex=wait)

    async def disable(self, kid: str, reason: str):
        # Invalid credentials stay disabled until explicit admin action.
        await self.redis.set(self.status_key(kid), "DISABLED")
        await self.redis.set(self.reason_key(kid), reason[:180])

    async def enable(self, kid: str):
        if kid not in self.ids:
            raise KeyError(kid)
        await self.redis.delete(self.status_key(kid), self.reason_key(kid), self.backoff_key(kid))

    async def next_backoff(self, kid: str) -> int:
        key = self.backoff_key(kid)
        count = await self.redis.incr(key)
        await self.redis.expire(key, self.settings.rotation_ttl_seconds)
        return min(
            self.settings.cooldown_max_seconds,
            self.settings.cooldown_base_seconds * (2 ** min(count - 1, 10)),
        )

    async def clear_backoff(self, kid: str):
        await self.redis.delete(self.backoff_key(kid))

    async def snapshot(self) -> list[dict]:
        mapping = {fingerprint(key): key for key in self.settings.keys}
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
                # Defensive cleanup for legacy or manually altered state.
                await self.redis.delete(self.status_key(kid), self.reason_key(kid))
                status, reason, ttl = "ACTIVE", "", -2
            result.append(
                {
                    "id": kid,
                    "masked": f"{secret[:3]}…{secret[-3:]}" if len(secret) > 8 else "••••",
                    "status": status,
                    "cooldown_remaining_seconds": max(0, ttl) if status == "COOLDOWN" else 0,
                    "last_reason": reason,
                }
            )
        return result
