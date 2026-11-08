from __future__ import annotations
import hmac
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from redis.asyncio import Redis
from app.config import get_settings
from app.database.redis_client import get_redis
from app.services.failover import JinaFailover, GatewayError

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


@router.get("/models")
async def models(authorization: str | None=Header(default=None)):
    require_gateway_key(authorization)
    settings = get_settings()
    return {
        "provider": "jina",
        "embeddings": {"model": settings.jina_embedding_model, "dimensions": settings.jina_embedding_dimensions},
        "reranking": {"model": settings.jina_reranker_model},
    }
