from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# This service reads only its own environment file.
_GATEWAY_DIR = Path(__file__).resolve().parents[1]
load_dotenv(_GATEWAY_DIR / ".env", override=False)


def _parse_keys() -> tuple[str, ...]:
    raw = os.getenv("JINA_API_KEYS", "").strip() or os.getenv("JINA_API_KEY", "").strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return tuple(dict.fromkeys(str(v).strip() for v in parsed if str(v).strip()))
    except json.JSONDecodeError:
        pass
    return tuple(dict.fromkeys(v.strip() for v in raw.split(",") if v.strip()))


@dataclass(frozen=True)
class Settings:
    gateway_api_key: str = field(repr=False)
    admin_api_key: str = field(repr=False)
    redis_url: str = field(repr=False)
    redis_prefix: str
    rotation_ttl_seconds: int
    metrics_ttl_seconds: int
    jina_base_url: str
    jina_embedding_model: str
    jina_embedding_dimensions: int
    jina_reranker_model: str
    jina_timeout_seconds: float
    max_attempts: int
    cooldown_base_seconds: int
    cooldown_max_seconds: int
    transient_cooldown_seconds: int
    keys: tuple[str, ...] = field(repr=False)


@lru_cache
def get_settings() -> Settings:
    keys = _parse_keys()
    redis_url = os.getenv("AIVEN_REDIS_URI", "").strip() or os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        raise RuntimeError(
            "Redis URI is not configured. Set AIVEN_REDIS_URI in api-gateway/.env."
        )
    if not keys:
        raise RuntimeError(
            "No Jina provider keys configured. Set JINA_API_KEYS or JINA_API_KEY "
            "in api-gateway/.env."
        )
    prefix = os.getenv("GATEWAY_REDIS_PREFIX", "savira:gateway").strip().strip(":")
    if not prefix:
        raise RuntimeError("GATEWAY_REDIS_PREFIX must not be empty.")
    return Settings(
        gateway_api_key=os.getenv("GATEWAY_API_KEY", ""),
        admin_api_key=os.getenv("GATEWAY_ADMIN_API_KEY", ""),
        redis_url=redis_url,
        redis_prefix=prefix,
        rotation_ttl_seconds=max(60, int(os.getenv("GATEWAY_ROTATION_TTL_SECONDS", "86400"))),
        metrics_ttl_seconds=max(60, int(os.getenv("GATEWAY_METRICS_TTL_SECONDS", "2592000"))),
        jina_base_url=os.getenv("JINA_BASE_URL", "https://api.jina.ai/v1").rstrip("/"),
        jina_embedding_model=os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small"),
        jina_embedding_dimensions=int(os.getenv("JINA_EMBEDDING_DIMENSIONS", "1024")),
        jina_reranker_model=os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v2-base-multilingual"),
        jina_timeout_seconds=float(os.getenv("JINA_TIMEOUT_SECONDS", "90")),
        max_attempts=max(1, int(os.getenv("GATEWAY_MAX_ATTEMPTS", str(max(1, len(keys)))))),
        cooldown_base_seconds=max(1, int(os.getenv("COOLDOWN_BASE_SECONDS", "60"))),
        cooldown_max_seconds=max(1, int(os.getenv("COOLDOWN_MAX_SECONDS", "3600"))),
        transient_cooldown_seconds=max(1, int(os.getenv("TRANSIENT_COOLDOWN_SECONDS", "15"))),
        keys=keys,
    )
