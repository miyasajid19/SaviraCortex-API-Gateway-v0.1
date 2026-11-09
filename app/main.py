from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI
from app.config import get_settings
from app.database.redis_client import get_redis, close_redis
from app.routers.proxy import router as proxy_router
from app.routers.admin import router as admin_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if len(settings.gateway_api_key) < 24:
        raise RuntimeError("Set GATEWAY_API_KEY to a random secret of at least 24 characters.")
    if len(settings.admin_api_key) < 24:
        raise RuntimeError("Set GATEWAY_ADMIN_API_KEY to a separate random secret of at least 24 characters.")
    if not settings.keys:
        raise RuntimeError("Configure JINA_API_KEYS or JINA_API_KEY.")
    if settings.jina_embedding_dimensions < 1:
        raise RuntimeError("JINA_EMBEDDING_DIMENSIONS must be positive.")
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(settings.jina_timeout_seconds, connect=10), follow_redirects=False)
    await get_redis()
    yield
    await app.state.http.aclose()
    await close_redis()

app = FastAPI(title="SaviraCortex Provider Gateway", version="1.0.0", lifespan=lifespan)
app.include_router(proxy_router)
app.include_router(admin_router)

@app.get("/health")
async def health():
    redis = await get_redis()
    await redis.ping()
    settings = get_settings()
    return {
        "status": "ok",
        "redis": "ok",
        "provider": "jina",
        "providers": {
            "jina": True,
            "cloudflare": bool(settings.cloudflare_accounts),
        },
        "cloudflare_account_pairs": len(settings.cloudflare_accounts),
    }
