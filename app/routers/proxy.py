from __future__ import annotations
import hmac
import httpx
from urllib.parse import quote
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from redis.asyncio import Redis
from app.config import get_settings
from app.database.redis_client import get_redis
from app.services.failover import JinaFailover, GatewayError
from app.services.fastocr_failover import FastOCRFailover

router = APIRouter(prefix="/v1")

def require_gateway_key(value: str | None):
    expected = get_settings().gateway_api_key
    if not expected or not value or not value.lower().startswith("bearer "):
        raise HTTPException(401, detail={"error":{"message":"Invalid gateway API key","type":"authentication_error"}})
    supplied = value[7:].strip()
    if not hmac.compare_digest(expected, supplied):
        raise HTTPException(401, detail={"error":{"message":"Invalid gateway API key","type":"authentication_error"}})

async def _proxy(request: Request, path: str, payload: dict, authorization: str | None, redis: Redis):
    require_gateway_key(authorization)
    if not get_settings().keys:
        raise HTTPException(503, detail={"error":{"message":"No Jina provider keys configured","type":"configuration_error"}})
    engine = JinaFailover(redis, request.app.state.http)
    try:
        status, headers, raw = await engine.post(path, payload)
    except GatewayError as exc:
        out = JSONResponse(status_code=exc.status_code, content={"error":{"message":exc.message,"type":"provider_unavailable"}})
        if exc.retry_after:
            out.headers["Retry-After"] = str(exc.retry_after)
        return out
    response_headers = {k:v for k,v in headers.items() if k.lower() in {"content-type","x-request-id","retry-after"}}
    return Response(content=raw, status_code=status, headers=response_headers, media_type=headers.get("content-type"))

@router.post("/embeddings")
async def embeddings(request: Request, authorization: str | None=Header(default=None), redis: Redis=Depends(get_redis)):
    payload = await request.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("input"), (str, list)):
        raise HTTPException(400, "Request must contain an input string or list.")
    payload["model"] = get_settings().jina_embedding_model
    payload["dimensions"] = get_settings().jina_embedding_dimensions
    return await _proxy(request, "/embeddings", payload, authorization, redis)

@router.post("/rerank")
async def rerank(request: Request, authorization: str | None=Header(default=None), redis: Redis=Depends(get_redis)):
    payload = await request.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("query"), str) or not isinstance(payload.get("documents"), list):
        raise HTTPException(400, "Request must contain query and documents.")
    payload["model"] = get_settings().jina_reranker_model
    return await _proxy(request, "/rerank", payload, authorization, redis)


async def _fastocr_proxy(
    request: Request,
    path: str,
    authorization: str | None,
    redis: Redis,
    *,
    document_id: str | None = None,
    remember_document: bool = False,
):
    require_gateway_key(authorization)
    if not get_settings().fastocr_keys:
        raise HTTPException(
            503,
            detail={
                "error": {
                    "message": "No FastOCR API keys configured. Set FASTOCR_API_KEYS in api-gateway/.env.",
                    "type": "configuration_error",
                }
            },
        )

    forwarded_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() in {"content-type", "accept", "idempotency-key", "user-agent"}
    }
    engine = FastOCRFailover(redis, request.app.state.fastocr_http)
    try:
        status, headers, raw = await engine.request(
            request.method,
            path,
            params=list(request.query_params.multi_items()),
            content=await request.body(),
            headers=forwarded_headers,
            document_id=document_id,
            remember_document=remember_document,
        )
    except GatewayError as exc:
        error = {"message": exc.message, "type": "provider_unavailable"}
        if exc.code:
            error["code"] = exc.code
        response = JSONResponse(
            status_code=exc.status_code,
            content={"error": error},
        )
        if exc.retry_after:
            response.headers["Retry-After"] = str(exc.retry_after)
        return response

    return Response(
        content=raw,
        status_code=status,
        headers=headers,
    )


@router.api_route("/documents", methods=["GET", "POST"])
async def fastocr_documents(
    request: Request,
    authorization: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    return await _fastocr_proxy(
        request,
        "/v1/documents",
        authorization,
        redis,
        remember_document=request.method.upper() == "POST",
    )


@router.api_route("/documents/{document_id}", methods=["GET", "DELETE"])
async def fastocr_document(
    document_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    return await _fastocr_proxy(
        request,
        f"/v1/documents/{quote(document_id, safe='')}",
        authorization,
        redis,
        document_id=document_id,
    )


@router.post("/documents/{document_id}/start")
async def fastocr_start_document(
    document_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    return await _fastocr_proxy(
        request,
        f"/v1/documents/{quote(document_id, safe='')}/start",
        authorization,
        redis,
        document_id=document_id,
    )


@router.get("/documents/{document_id}/output")
async def fastocr_document_output(
    document_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    return await _fastocr_proxy(
        request,
        f"/v1/documents/{quote(document_id, safe='')}/output",
        authorization,
        redis,
        document_id=document_id,
    )


@router.post("/cloudflare/rerank")
async def cloudflare_rerank(
    request: Request,
    authorization: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    require_gateway_key(authorization)
    settings = get_settings()
    if not settings.cloudflare_accounts:
        raise HTTPException(
            503,
            detail={"error": {"message": "Cloudflare account/token pairs are not configured.", "type": "configuration_error"}},
        )

    payload = await request.json()
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("query"), str)
        or not payload["query"].strip()
        or not isinstance(payload.get("documents"), list)
        or not payload["documents"]
        or any(not isinstance(document, str) or not document.strip() for document in payload["documents"])
    ):
        raise HTTPException(400, "Request must contain a non-empty query and a non-empty list of document strings.")

    top_n = payload.get("top_n")
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
        raise HTTPException(400, "top_n must be a positive integer.")

    # Workers AI reranking accepts query + contexts, where each context is an object.
    provider_payload = {
        "query": payload["query"],
        "contexts": [{"text": document} for document in payload["documents"]],
    }
    if top_n is not None:
        provider_payload["top_k"] = top_n

    cursor_key = f"{settings.redis_prefix}:cloudflare:rerank:cursor"
    cursor = await redis.incr(cursor_key)
    await redis.expire(cursor_key, settings.rotation_ttl_seconds)
    pairs = settings.cloudflare_accounts
    start = (cursor - 1) % len(pairs)
    client = request.app.state.http
    retryable_statuses = {401, 403, 404, 429, 500, 502, 503, 504}
    last_response = None
    last_error = None

    for offset in range(len(pairs)):
        account_id, token = pairs[(start + offset) % len(pairs)]
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/ai/run/{settings.cloudflare_reranker_model}"
        )
        try:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=provider_payload,
            )
            last_response = response
            if response.status_code < 400:
                await redis.incr(f"{settings.redis_prefix}:gateway:metrics:cloudflare_rerank_success")
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers={k: v for k, v in response.headers.items() if k.lower() in {"content-type", "x-request-id"}},
                    media_type=response.headers.get("content-type"),
                )
            if response.status_code not in retryable_statuses:
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers={k: v for k, v in response.headers.items() if k.lower() in {"content-type", "x-request-id"}},
                    media_type=response.headers.get("content-type"),
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = type(exc).__name__
            continue

    if last_response is not None:
        return Response(
            content=last_response.content,
            status_code=last_response.status_code,
            headers={k: v for k, v in last_response.headers.items() if k.lower() in {"content-type", "retry-after", "x-request-id"}},
            media_type=last_response.headers.get("content-type"),
        )
    return JSONResponse(
        status_code=503,
        content={"error": {"message": f"Cloudflare reranker unavailable ({last_error or 'all account pairs failed'}).", "type": "provider_unavailable"}},
    )


@router.post("/cloudflare/embeddings")
async def cloudflare_embeddings(
    request: Request,
    authorization: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    require_gateway_key(authorization)
    settings = get_settings()
    if not settings.cloudflare_accounts:
        raise HTTPException(
            503,
            detail={"error": {"message": "Cloudflare account/token pairs are not configured.", "type": "configuration_error"}},
        )

    payload = await request.json()
    input_value = payload.get("input") if isinstance(payload, dict) else None
    valid_input = (
        isinstance(input_value, str) and bool(input_value.strip())
    ) or (
        isinstance(input_value, list)
        and bool(input_value)
        and all(isinstance(item, str) and item.strip() for item in input_value)
    )
    if not valid_input:
        raise HTTPException(400, "Request must contain a non-empty input string or list of strings.")

    provider_payload = {
        "model": settings.cloudflare_embedding_model,
        "input": input_value,
    }
    cursor_key = f"{settings.redis_prefix}:cloudflare:embeddings:cursor"
    cursor = await redis.incr(cursor_key)
    await redis.expire(cursor_key, settings.rotation_ttl_seconds)
    pairs = settings.cloudflare_accounts
    start = (cursor - 1) % len(pairs)
    client = request.app.state.http
    retryable_statuses = {401, 403, 404, 429, 500, 502, 503, 504}
    last_response = None
    last_error = None

    for offset in range(len(pairs)):
        account_id, token = pairs[(start + offset) % len(pairs)]
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/embeddings"
        try:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=provider_payload,
            )
            last_response = response
            if response.status_code < 400:
                await redis.incr(f"{settings.redis_prefix}:gateway:metrics:cloudflare_embeddings_success")
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers={k: v for k, v in response.headers.items() if k.lower() in {"content-type", "x-request-id"}},
                    media_type=response.headers.get("content-type"),
                )
            if response.status_code not in retryable_statuses:
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers={k: v for k, v in response.headers.items() if k.lower() in {"content-type", "x-request-id"}},
                    media_type=response.headers.get("content-type"),
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = type(exc).__name__
            continue

    if last_response is not None:
        return Response(
            content=last_response.content,
            status_code=last_response.status_code,
            headers={k: v for k, v in last_response.headers.items() if k.lower() in {"content-type", "retry-after", "x-request-id"}},
            media_type=last_response.headers.get("content-type"),
        )
    return JSONResponse(
        status_code=503,
        content={"error": {"message": f"Cloudflare embeddings unavailable ({last_error or 'all account pairs failed'}).", "type": "provider_unavailable"}},
    )


@router.get("/models")
async def models(authorization: str | None=Header(default=None)):
    require_gateway_key(authorization)
    settings = get_settings()
    return {
        "provider": "jina",
        "embeddings": {"model": settings.jina_embedding_model, "dimensions": settings.jina_embedding_dimensions},
        "reranking": {"model": settings.jina_reranker_model},
        "cloudflare": {
            "configured": bool(settings.cloudflare_accounts),
            "account_pairs": len(settings.cloudflare_accounts),
            "embeddings": {"model": settings.cloudflare_embedding_model, "dimensions": 1024},
            "reranking": {"model": settings.cloudflare_reranker_model},
        },
    }
