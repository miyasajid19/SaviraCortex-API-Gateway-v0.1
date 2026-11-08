# Savare Cortex API Gateway

A standalone FastAPI gateway for Jina embeddings and reranking. It keeps provider API keys private and uses **Aiven Redis as its runtime state store** for atomic key rotation, cooldowns, disabled-key state, backoff, and metrics.

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

- `GET /health` — gateway and Redis health.
- `GET /v1/models` — configured embedding and reranker models.
- `POST /v1/embeddings` — Jina embeddings proxy.
- `POST /v1/rerank` — Jina reranking proxy.
- `GET /admin/status` — Redis-backed key state and metrics; requires `X-Admin-Key`.
- `POST /admin/keys/{fingerprint}/enable` — re-enable a corrected key; requires admin authentication.
- `GET /admin/dashboard` — admin dashboard.

Client requests use `Authorization: Bearer <GATEWAY_API_KEY>`. The selected provider key is never returned to the caller.

Embedding example:

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

Rerank example:

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

The backend should call the gateway, not Jina directly. Set these values in `backend/.env` only:

```dotenv
JINA_API_BASE_URL=http://127.0.0.1:8090/v1
JINA_API_KEY=<same-value-as-GATEWAY_API_KEY>
```

The backend's `JINA_API_KEY` is the gateway client credential, not a provider key. Keep actual Jina provider keys only in `api-gateway/.env`.

Keep the gateway embedding model and dimensions identical to the existing Qdrant collection. Changing either requires a separate collection and re-indexing.

This gateway is deliberately narrow: it handles Jina embeddings and reranking key rotation. It does not proxy OpenCode or generation traffic. The goal is to make provider rate limiting transparent to SaviraCortex's backend without adding unnecessary orchestration to the backend.

## 5. Failover behavior

- HTTP 429: cooldown uses `Retry-After` or provider reset metadata, otherwise capped exponential backoff; then rotates to another eligible key.
- Shared organization/account quota exhaustion: applies a provider-wide TTL cooldown instead of cycling individual keys.
- HTTP 401/403: disables that credential until an administrator explicitly re-enables it.
- HTTP 500/502/503/504 and network errors: bounded failover with short per-key cooldown.
- HTTP 400 and other non-retryable client errors: returned without rotating.
- Expired cooldowns become eligible automatically; no worker or restart is required.
- Redis Lua script makes reservation atomic across concurrent gateway workers.

## 6. Verification

The automated test suite uses the configured Redis endpoint with unique per-test namespaces and cleanup. It does not flush the Redis database. Install development test tools with `pip install -r requirements-dev.txt`, then run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover Redis-backed atomic reservations, key rotation, rate-limit handling, cooldown expiry, TTL behavior, and gateway authentication. For live provider connectivity, use the Jina provider's own dashboard/usage tools and the gateway health and admin endpoints; do not run quota-exhaustion tests against production credentials.

## 7. Deployment

Run Uvicorn under a process supervisor and place it behind an HTTPS reverse proxy. Restrict admin access to a VPN or trusted IP range. Store the environment file with restrictive permissions and never commit it. No Docker or local Redis server is required.
