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


def _parse_key_list(primary_name: str, fallback_name: str) -> tuple[str, ...]:
    raw = os.getenv(primary_name, "").strip() or os.getenv(fallback_name, "").strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return tuple(dict.fromkeys(str(v).strip() for v in parsed if str(v).strip()))
    except json.JSONDecodeError:
        pass
    return tuple(dict.fromkeys(v.strip() for v in raw.split(",") if v.strip()))


def _parse_keys() -> tuple[str, ...]:
    return _parse_key_list("JINA_API_KEYS", "JINA_API_KEY")


def _parse_fastocr_keys() -> tuple[str, ...]:
    return _parse_key_list("FASTOCR_API_KEYS", "FASTOCR_API_KEY")


def _parse_opencode_keys() -> tuple[str, ...]:
    return _parse_key_list("OPENCODE_API_KEYS", "OPENCODE_API_KEY")


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
    opencode_keys: tuple[str, ...] = field(repr=False)
    opencode_jev_model: str
    opencode_jev_base_url: str
    opencode_jev_timeout_seconds: float
    cloudflare_jev_model: str
    cloudflare_jev_selector: str
    cloudflare_jev_fast_model: str
    cloudflare_jev_fast_selector: str
    cloudflare_jev_omni_model: str
    cloudflare_jev_omni_selector: str
    jev_retry_cooldown_seconds: int
    fastocr_keys: tuple[str, ...] = field(repr=False)
    fastocr_base_url: str
    fastocr_timeout_seconds: float
    fastocr_max_attempts: int
    fastocr_cooldown_base_seconds: int
    fastocr_cooldown_max_seconds: int
    fastocr_transient_cooldown_seconds: int
    fastocr_document_key_ttl_seconds: int
    cloudflare_accounts: tuple[tuple[str, str], ...] = field(repr=False)
    cloudflare_reranker_model: str
    cloudflare_embedding_model: str


def _parse_cloudflare_accounts() -> tuple[tuple[str, str], ...]:
    raw_ids = os.getenv("CLOUDFLARE_ACCOUNT_IDS", "").strip()
    raw_tokens = os.getenv("CLOUDFLARE_API_TOKENS", "").strip()
    if not raw_ids and not raw_tokens:
        return ()
    if not raw_ids or not raw_tokens:
        raise RuntimeError("Set both CLOUDFLARE_ACCOUNT_IDS and CLOUDFLARE_API_TOKENS.")
    def parse_list(raw: str, variable: str) -> list[str]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Also accept bracketed/comma-separated .env values without quotes.
            normalized = raw.strip()
            if normalized.startswith("[") and normalized.endswith("]"):
                normalized = normalized[1:-1]
            parsed = [item.strip().strip("\"\'") for item in normalized.split(",")]
        if not isinstance(parsed, list) or not parsed:
            raise RuntimeError(f"{variable} must be a non-empty list.")
        return parsed

    account_ids = parse_list(raw_ids, "CLOUDFLARE_ACCOUNT_IDS")
    api_tokens = parse_list(raw_tokens, "CLOUDFLARE_API_TOKENS")
    if not account_ids or len(account_ids) != len(api_tokens):
        raise RuntimeError("Cloudflare account ID and API token arrays must have equal, non-zero lengths.")
    pairs = []
    for index, (account_id, token) in enumerate(zip(account_ids, api_tokens, strict=True)):
        if not isinstance(account_id, str) or not account_id.strip():
            raise RuntimeError(f"CLOUDFLARE_ACCOUNT_IDS[{index}] must be a non-empty string.")
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError(f"CLOUDFLARE_API_TOKENS[{index}] must be a non-empty string.")
        pairs.append((account_id.strip(), token.strip()))
    return tuple(pairs)


@lru_cache
def get_settings() -> Settings:
    keys = _parse_keys()
    opencode_keys = _parse_opencode_keys()
    fastocr_keys = _parse_fastocr_keys()
    cloudflare_accounts = _parse_cloudflare_accounts()
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
        opencode_keys=opencode_keys,
        opencode_jev_model=os.getenv("OPENCODE_JEV_MODEL", "jev-1.13-free").strip(),
        opencode_jev_base_url=os.getenv(
            "OPENCODE_JEV_BASE_URL",
            "https://opencode.ai/inference/systemone/v1/systemone",
        ).strip().rstrip("/"),
        opencode_jev_timeout_seconds=max(1.0, float(os.getenv("OPENCODE_JEV_TIMEOUT_SECONDS", "60"))),
        cloudflare_jev_model=os.getenv("CLOUDFLARE_JEV_MODEL", "@cf/cloudflare/clef").strip(),
        cloudflare_jev_selector=os.getenv("CLOUDFLARE_JEV_SELECTOR", "clef").strip(),
        cloudflare_jev_fast_model=os.getenv("CLOUDFLARE_JEV_FAST_MODEL", "@cf/cloudflare/clef-flash").strip(),
        cloudflare_jev_fast_selector=os.getenv("CLOUDFLARE_JEV_FAST_SELECTOR", "clef-flash").strip(),
        cloudflare_jev_omni_model=os.getenv("CLOUDFLARE_JEV_OMNI_MODEL", "@cf/cloudflare/clef-omni").strip(),
        cloudflare_jev_omni_selector=os.getenv("CLOUDFLARE_JEV_OMNI_SELECTOR", "clef-omni").strip(),
        jev_retry_cooldown_seconds=max(1, int(os.getenv("JEV_RETRY_COOLDOWN_SECONDS", "30"))),
        fastocr_keys=fastocr_keys,
        fastocr_base_url=os.getenv("FASTOCR_BASE_URL", "https://api.fastocr.org").rstrip("/"),
        fastocr_timeout_seconds=max(1.0, float(os.getenv("FASTOCR_TIMEOUT_SECONDS", "90"))),
        fastocr_max_attempts=max(1, int(os.getenv("FASTOCR_MAX_ATTEMPTS", str(max(1, len(fastocr_keys)))))),
        fastocr_cooldown_base_seconds=max(1, int(os.getenv("FASTOCR_COOLDOWN_BASE_SECONDS", "30"))),
        fastocr_cooldown_max_seconds=max(1, int(os.getenv("FASTOCR_COOLDOWN_MAX_SECONDS", "900"))),
        fastocr_transient_cooldown_seconds=max(1, int(os.getenv("FASTOCR_TRANSIENT_COOLDOWN_SECONDS", "15"))),
        fastocr_document_key_ttl_seconds=max(300, int(os.getenv("FASTOCR_DOCUMENT_KEY_TTL_SECONDS", "86400"))),
        cloudflare_accounts=cloudflare_accounts,
        cloudflare_reranker_model=os.getenv("CLOUDFLARE_RERANKER_MODEL", "@cf/baai/bge-reranker-base"),
        cloudflare_embedding_model=os.getenv("CLOUDFLARE_EMBEDDING_MODEL", "@cf/qwen/qwen3-embedding-0.6b"),
    )
