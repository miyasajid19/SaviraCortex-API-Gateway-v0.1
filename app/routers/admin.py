from __future__ import annotations

import hmac
import hashlib
import time

from fastapi import APIRouter, Depends, Header, HTTPException
from redis.asyncio import Redis

from app.config import get_settings
from app.database.redis_client import get_redis
from app.services.key_manager import KeyManager
from app.services.fastocr_key_manager import FastOCRKeyManager

router = APIRouter(prefix="/admin")


def authorize(token: str | None):
    expected = get_settings().admin_api_key
    if not expected or not token or not hmac.compare_digest(expected, token):
        raise HTTPException(403, "Valid X-Admin-Key required.")


@router.get("/status")
async def status(x_admin_key: str | None = Header(default=None), redis: Redis = Depends(get_redis)):
    authorize(x_admin_key)
    settings = get_settings()
    keys = await KeyManager(redis).snapshot()
    provider_until = float(
        await redis.get(f"{settings.redis_prefix}:jina:provider:cooldown_until") or 0
    )
    selected = await redis.get(f"{settings.redis_prefix}:jina:last_selected")
    names = (
        "requests", "success", "failed", "rate_limited", "shared_quota_limited",
        "automatic_failovers", "disabled_keys", "provider_errors", "network_errors",
        "client_errors", "quota_exhausted", "concurrency_limit", "latency_ms_total",
    )
    metrics = {
        name: int(
            await redis.get(f"{settings.redis_prefix}:gateway:metrics:{name}") or 0
        )
        for name in names
    }
    metrics["average_latency_ms"] = round(
        metrics["latency_ms_total"] / max(1, metrics["success"]), 2
    )
    return {
        "provider": "Jina",
        "key_counts": {
            "total": len(keys),
            "active": sum(key["status"] == "ACTIVE" for key in keys),
            "cooldown": sum(key["status"] == "COOLDOWN" for key in keys),
            "disabled": sum(key["status"] == "DISABLED" for key in keys),
        },
        "current_selected_key": next(
            (key for key in keys if key["id"] == selected), None
        ),
        "provider_cooldown_remaining_seconds": max(
            0, round(provider_until - time.time())
        ),
        "keys": keys,
        "metrics": metrics,
    }


@router.post("/keys/{kid}/enable")
async def enable(
    kid: str,
    x_admin_key: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    authorize(x_admin_key)
    try:
        await KeyManager(redis).enable(kid)
    except KeyError:
        raise HTTPException(404, "Unknown configured key fingerprint.")
    return {"id": kid, "status": "ACTIVE"}


@router.get("/fastocr/status")
async def fastocr_status(
    x_admin_key: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    authorize(x_admin_key)
    settings = get_settings()
    manager = FastOCRKeyManager(redis)
    keys = await manager.snapshot()
    selected = await redis.get(f"{settings.redis_prefix}:fastocr:last_selected")
    names = (
        "requests", "success", "failed", "rate_limited", "shared_rate_limited",
        "automatic_failovers", "disabled_keys", "provider_errors", "network_errors",
        "client_errors", "quota_exhausted", "concurrency_limit", "latency_ms_total",
    )
    metrics = {
        name: int(
            await redis.get(
                f"{settings.redis_prefix}:gateway:metrics:fastocr_{name}"
            ) or 0
        )
        for name in names
    }
    metrics["average_latency_ms"] = round(
        metrics["latency_ms_total"] / max(1, metrics["success"]), 2
    )
    return {
        "provider": "FastOCR",
        "configured": bool(settings.fastocr_keys),
        "key_counts": {
            "total": len(keys),
            "active": sum(key["status"] == "ACTIVE" for key in keys),
            "cooldown": sum(key["status"] == "COOLDOWN" for key in keys),
            "disabled": sum(key["status"] == "DISABLED" for key in keys),
        },
        "current_selected_key": next(
            (key for key in keys if key["id"] == selected), None
        ),
        "provider_cooldown_remaining_seconds": (
            await manager.provider_cooldown_remaining()
        ),
        "keys": keys,
        "metrics": metrics,
    }


@router.post("/fastocr/keys/{kid}/enable")
async def enable_fastocr_key(
    kid: str,
    x_admin_key: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    authorize(x_admin_key)
    try:
        await FastOCRKeyManager(redis).enable(kid)
    except KeyError:
        raise HTTPException(404, "Unknown configured FastOCR key fingerprint.")
    return {"id": kid, "status": "ACTIVE"}


@router.get("/jev/status")
async def jev_status(
    x_admin_key: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    authorize(x_admin_key)
    settings = get_settings()

    def fingerprint(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    async def snapshot(provider: str, credentials: list[tuple[str, str]]):
        result = []
        for kid, _label in credentials:
            key = f"{settings.redis_prefix}:jev:{provider}:credential:{kid}:state"
            state = await redis.get(key) or "ACTIVE"
            ttl = int(await redis.ttl(key))
            result.append({
                "id": kid,
                "status": state,
                "cooldown_remaining_seconds": max(0, ttl) if state == "COOLDOWN" else 0,
            })
        return result

    opencode = [(fingerprint(key), "OpenCode key") for key in settings.opencode_keys]
    cloudflare = [
        (fingerprint(f"{account_id}:{token}"), "Cloudflare account/token pair")
        for account_id, token in settings.cloudflare_accounts
    ]
    names = (
        "requests", "opencode_success", "cloudflare_success", "opencode_network_failures",
        "cloudflare_network_failures", "unavailable",
    )
    metrics = {
        name: int(await redis.get(f"{settings.redis_prefix}:gateway:metrics:jev_{name}") or 0)
        for name in names
    }
    return {
        "providers": {
            "opencode": {
                "model": settings.opencode_jev_model,
                "configured": bool(opencode),
                "credentials": await snapshot("opencode", opencode),
            },
            "cloudflare": {
                "standard_model": settings.cloudflare_jev_model,
                "fast_model": settings.cloudflare_jev_fast_model,
                "omni_model": settings.cloudflare_jev_omni_model,
                "configured": bool(cloudflare),
                "credentials": await snapshot("cloudflare", cloudflare),
            },
        },
        "task_profiles": {
            "ingestion": {
                "primary": "cloudflare",
                "primary_model": settings.cloudflare_jev_fast_model,
                "fallback": "opencode",
                "fallback_model": settings.opencode_jev_model,
            },
            "runtime": {
                "primary": "opencode",
                "primary_model": settings.opencode_jev_model,
                "fallback": "cloudflare",
                "fallback_model": settings.cloudflare_jev_model,
            },
        },
        "metrics": metrics,
    }


@router.post("/jev/credentials/{provider}/{kid}/enable")
async def enable_jev_credential(
    provider: str,
    kid: str,
    x_admin_key: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    authorize(x_admin_key)
    settings = get_settings()

    def fingerprint(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    if provider == "opencode":
        configured = {fingerprint(key) for key in settings.opencode_keys}
    elif provider == "cloudflare":
        configured = {
            fingerprint(f"{account_id}:{token}")
            for account_id, token in settings.cloudflare_accounts
        }
    else:
        raise HTTPException(404, "Unknown JEV provider.")

    if kid not in configured:
        raise HTTPException(404, "Unknown configured JEV credential fingerprint.")
    await redis.delete(f"{settings.redis_prefix}:jev:{provider}:credential:{kid}:state")
    return {"provider": provider, "id": kid, "status": "ACTIVE"}


@router.get("/dashboard")
async def dashboard():
    from fastapi.responses import HTMLResponse

    return HTMLResponse("""<!doctype html><html><head><meta charset=utf-8><title>Jina Gateway</title><style>body{font:15px system-ui;max-width:1050px;margin:32px auto;padding:0 18px;background:#111827;color:#e5e7eb}input,button{padding:9px;background:#263244;color:white;border:1px solid #526078;border-radius:6px}button#refresh{cursor:pointer;transition:background .15s,border-color .15s}button#refresh:active,button#refresh.is-loading{background:#2563eb;border-color:#60a5fa;color:white}button#refresh:disabled{cursor:wait;opacity:1}table{width:100%;border-collapse:collapse;margin:18px 0}td,th{text-align:left;padding:9px;border-bottom:1px solid #374151}pre{white-space:pre-wrap}</style></head><body><h1>Jina API Gateway</h1><p>Admin status · secrets are masked</p><input id=t type=password placeholder='Admin key' autocomplete='current-password'><button id=refresh onclick=load()>Refresh</button><div id=o></div><script>function esc(s){return String(s??'').replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}async function load(){const o=document.querySelector('#o'),b=document.querySelector('#refresh'),key=document.querySelector('#t').value.trim();if(!key){o.textContent='Enter the admin key first.';return}b.disabled=true;b.classList.add('is-loading');b.textContent='Loading...';o.textContent='Loading gateway status...';try{const r=await fetch('/admin/status',{headers:{'X-Admin-Key':key}});const d=await r.json();if(!r.ok)throw new Error(d.detail||('HTTP '+r.status));let h='<h2>Key counts</h2><pre>'+esc(JSON.stringify(d.key_counts,null,2))+'</pre><h2>Provider cooldown</h2><pre>'+esc(d.provider_cooldown_remaining_seconds)+' seconds</pre><h2>Current selected key</h2><pre>'+esc(JSON.stringify(d.current_selected_key,null,2))+'</pre><h2>Metrics</h2><pre>'+esc(JSON.stringify(d.metrics,null,2))+'</pre><h2>Configured keys</h2><table><tr><th>Fingerprint</th><th>Masked</th><th>Status</th><th>Cooldown</th><th>Last reason</th><th>Action</th></tr>';for(const k of d.keys){h+='<tr><td>'+esc(k.id)+'</td><td>'+esc(k.masked)+'</td><td>'+esc(k.status)+'</td><td>'+esc(k.cooldown_remaining_seconds)+'s</td><td>'+esc(k.last_reason||'—')+'</td><td>'+(k.status==='DISABLED'?'<button type=button data-k="'+esc(k.id)+'">Re-enable</button>':'—')+'</td></tr>'}o.innerHTML=h+'</table>';o.querySelectorAll('[data-k]').forEach(function(x){x.addEventListener('click',function(){enable(x.getAttribute('data-k'))})})}catch(e){o.textContent='Unable to load gateway status: '+e.message}finally{b.classList.remove('is-loading');b.disabled=false;b.textContent='Refresh'}}async function enable(k){const o=document.querySelector('#o');o.textContent='Re-enabling key...';try{const r=await fetch('/admin/keys/'+encodeURIComponent(k)+'/enable',{method:'POST',headers:{'X-Admin-Key':document.querySelector('#t').value.trim()}});const d=await r.json();if(!r.ok)throw new Error(d.detail||('HTTP '+r.status));await load()}catch(e){o.textContent='Unable to re-enable key: '+e.message}}</script></body></html>""")
