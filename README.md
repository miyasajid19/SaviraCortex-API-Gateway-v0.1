# Savare Cortex API Gateway

A standalone FastAPI gateway for Jina embeddings/reranking and optional Cloudflare Workers AI embeddings. It keeps provider API credentials private and uses **Aiven Redis as its runtime state store** for Jina key rotation, cooldowns, disabled-key state, backoff, and metrics.

No in-memory Redis or Docker is used by the running gateway. The gateway connects directly to the configured Aiven Redis instance through `AIVEN_REDIS_URI`.

## 1. Environment isolation

The gateway reads only `api-gateway/.env`; it never loads or falls back to `backend/.env`. The backend reads only `backend/.env`. Values are not copied or synchronized.

Required gateway settings:

```dotenv
AIVEN_REDIS_URI=rediss://<username>:<password>@<aiven-host>:<port>/<db>
GATEWAY_API_KEY=<private-client-secret-at-least-24-characters>
GATEWAY_ADMIN_API_KEY=<different-admin-secret-at-least-24-characters>
JINA_API_KEYS=["jina_key_01","jina_key_02","jina_key_03"]
```

Optional Cloudflare Workers AI embedding settings:

```dotenv
# Parallel JSON arrays: account ID at index N is paired only with token at index N.
# Both lists must be non-empty and have equal lengths.
CLOUDFLARE_ACCOUNT_IDS=["account_id_01","account_id_02"]
CLOUDFLARE_API_TOKENS=["api_token_01","api_token_02"]
CLOUDFLARE_EMBEDDING_MODEL=@cf/qwen/qwen3-embedding-0.6b
CLOUDFLARE_TIMEOUT_SECONDS=90
```

Cloudflare is enabled only when both JSON arrays are set. The lists must have equal, non-zero lengths; they accept JSON arrays or bracketed comma-separated values. Index N in `CLOUDFLARE_ACCOUNT_IDS` is paired only with index N in `CLOUDFLARE_API_TOKENS`. The gateway rotates the starting pair across requests and tries other pairs on authentication, rate-limit, server, or network failures. Each token must have permission to run Workers AI models. Keep these credentials only in `api-gateway/.env`; never put them in backend configuration or source control.

Optional Redis lifecycle settings:

```dotenv
GATEWAY_REDIS_PREFIX=savira:gateway
GATEWAY_ROTATION_TTL_SECONDS=86400
GATEWAY_METRICS_TTL_SECONDS=2592000
```

- Rotation cursor and last-selected-key metadata expire after the rotation TTL.
- Per-key cooldown state and its reason expire at the cooldown deadline.
- Provider-wide cooldown expires at the provider's retry deadline.
- Metrics expire after the metrics TTL.
- Exponential-backoff counters expire after the rotation TTL.
- A credential rejected with HTTP 401/403 is stored as `DISABLED` without a TTL. This is intentional: an invalid credential must not silently become active again. It is removed only by the admin re-enable action.
- Raw Jina API keys are **never stored in Redis**. They remain in `api-gateway/.env`; Redis stores only hashed key identifiers and operational state.

Use the exact TLS `rediss://` URI provided by Aiven. Do not paste credentials into source code, logs, README, or chat.

## 2. Install and run (Windows PowerShell)

From `api-gateway`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8090
```

The service listens on `http://127.0.0.1:8090`. Startup connects to Aiven Redis and verifies it with PING.

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8090/health
```

## 3. Client endpoints

- `GET /health` — gateway, Redis, and provider configuration health.
- `GET /v1/models` — configured Jina and Cloudflare models.
- `POST /v1/embeddings` — existing Jina embeddings proxy (unchanged).
- `POST /v1/cloudflare/embeddings` — Cloudflare Workers AI embeddings using Qwen3 Embedding 0.6B.
- `POST /v1/rerank` — existing Jina reranking proxy (unchanged).
- `POST /v1/cloudflare/rerank` — Cloudflare Workers AI BGE reranking using `@cf/baai/bge-reranker-base`.
- `GET /admin/status` — Redis-backed Jina key state and metrics; requires `X-Admin-Key`.
- `POST /admin/keys/{fingerprint}/enable` — re-enable a corrected Jina key; requires admin authentication.
- `GET /admin/dashboard` — admin dashboard.

Client requests use `Authorization: Bearer <GATEWAY_API_KEY>`. Provider credentials are never returned to the caller.

### Jina embeddings (existing endpoint)

```python
import requests

response = requests.post(
    "http://127.0.0.1:8090/v1/embeddings",
    headers={"Authorization": "Bearer YOUR_GATEWAY_API_KEY"},
    json={"input": ["A document passage"]},
    timeout=90,
)
response.raise_for_status()
vectors = response.json()["data"]
```

### Cloudflare Qwen3 embeddings

This endpoint accepts the same OpenAI-style `input` shape, but uses the separately configured Cloudflare model and credentials:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8090/v1/cloudflare/embeddings",
    headers={"Authorization": "Bearer YOUR_GATEWAY_API_KEY"},
    json={"input": ["A document passage"]},
    timeout=90,
)
response.raise_for_status()
vectors = response.json()["data"]
```

The gateway fixes the model to `@cf/qwen/qwen3-embedding-0.6b` by default and returns Cloudflare's OpenAI-compatible response. The model produces native 1,024-dimensional vectors. Caller-supplied `model` and `dimensions` values are not forwarded.

### Cloudflare BGE reranking

Cloudflare account IDs and API tokens are configured as parallel JSON arrays in `api-gateway/.env`. The account ID at index N is paired only with the token at index N; both arrays must be non-empty and equal in length. The gateway rotates the starting pair in Redis and tries the remaining pairs after authentication, rate-limit, server, timeout, or network failures.

The gateway accepts the familiar `query` + `documents` request shape and maps documents to Cloudflare's native `query` + `contexts` format:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8090/v1/cloudflare/rerank",
    headers={"Authorization": "Bearer YOUR_GATEWAY_API_KEY"},
    json={
        "query": "Which passage answers the question?",
        "documents": ["First passage", "Second passage"],
        "top_n": 2,
    },
    timeout=90,
)
response.raise_for_status()
ranked = response.json()
```

Cloudflare's native response is passed through unchanged. Context IDs in the response refer to the zero-based positions of the submitted documents. The existing Jina `/v1/rerank` route continues to use the configured Jina reranker.

### Jina reranking

```python
response = requests.post(
    "http://127.0.0.1:8090/v1/rerank",
    headers={"Authorization": "Bearer YOUR_GATEWAY_API_KEY"},
    json={"query": "What is the answer?", "documents": ["passage one", "passage two"], "top_n": 2},
    timeout=90,
)
response.raise_for_status()
results = response.json()["results"]
```

## 4. Connect SaviraCortex backend

The backend can select either provider without changing the gateway client credential. Jina remains the default; set `EMBEDDING_PROVIDER=cloudflare` in `backend/.env` to route ingestion, query, and summary embeddings through Cloudflare Qwen3. The Cloudflare provider uses separate chunk and summary Qdrant collections with a `_cloudflare_qwen3_1024` suffix, so Jina vectors are never mixed with Qwen vectors. Switching providers requires re-embedding and re-indexing into the selected collections. The existing gateway settings remain:

```dotenv
JINA_API_BASE_URL=http://127.0.0.1:8090/v1
JINA_API_KEY=<same-value-as-GATEWAY_API_KEY>
```

To use Cloudflare from a client, call `/v1/cloudflare/embeddings` explicitly with the same gateway client credential. The backend's `JINA_API_KEY` is the gateway client credential, not a provider key. Keep actual Jina and Cloudflare provider credentials only in `api-gateway/.env`.

**Important:** Jina and Cloudflare Qwen3 embeddings are different vector spaces even though both are 1,024 dimensions. Do not mix vectors from the two models in one Qdrant collection. Switching a collection to Qwen3 requires a separate collection and a full re-embedding/re-indexing of its documents and query path.

This gateway does not proxy generation traffic. Jina keeps its Redis-backed key rotation and failover. Cloudflare uses its own account token and independent request path; it does not consume or rotate Jina keys.

## 5. Jina failover behavior

- HTTP 429: cooldown uses `Retry-After` or provider reset metadata, otherwise capped exponential backoff; then rotates to another eligible key.
- Shared organization/account quota exhaustion: applies a provider-wide TTL cooldown instead of cycling individual keys.
- HTTP 401/403: disables that credential until an administrator explicitly re-enables it.
- HTTP 500/502/503/504 and network errors: bounded failover with short per-key cooldown.
- HTTP 400 and other non-retryable client errors: returned without rotating.
- Expired cooldowns become eligible automatically; no worker or restart is required.
- Redis Lua script makes reservation atomic across concurrent gateway workers.

Cloudflare errors/status codes are passed through to the gateway client; network and timeout failures return a gateway 503 response.

## 6. Verification

The automated test suite uses the configured Redis endpoint with unique per-test namespaces and cleanup. It does not flush the Redis database. Install development test tools with `pip install -r requirements-dev.txt`, then run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover Redis-backed atomic reservations, Jina key rotation, rate-limit handling, cooldown expiry, TTL behavior, gateway authentication, and provider request routing. For live provider connectivity, use the provider dashboards/usage tools and gateway health endpoint; do not run quota-exhaustion tests against production credentials.

## 7. Deployment

Run Uvicorn under a process supervisor and place it behind an HTTPS reverse proxy. Restrict admin access to a VPN or trusted IP range. Store the environment file with restrictive permissions and never commit it. No Docker or local Redis server is required.
