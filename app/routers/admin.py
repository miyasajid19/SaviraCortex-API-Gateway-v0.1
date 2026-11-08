from __future__ import annotations

import hmac
import time

from fastapi import APIRouter, Depends, Header, HTTPException
from redis.asyncio import Redis

from app.config import get_settings
from app.database.redis_client import get_redis
from app.services.key_manager import KeyManager

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
        "client_errors", "latency_ms_total",
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


@router.get("/dashboard")
async def dashboard():
    from fastapi.responses import HTMLResponse

    return HTMLResponse("""<!doctype html><html><head><meta charset=utf-8><title>Jina Gateway</title><style>body{font:15px system-ui;max-width:1050px;margin:32px auto;padding:0 18px;background:#111827;color:#e5e7eb}input,button{padding:9px;background:#263244;color:white;border:1px solid #526078;border-radius:6px}table{width:100%;border-collapse:collapse;margin:18px 0}td,th{text-align:left;padding:9px;border-bottom:1px solid #374151}pre{white-space:pre-wrap}</style></head><body><h1>Jina API Gateway</h1><p>Admin status · secrets are masked</p><input id=t type=password placeholder='Admin key'><button onclick=load()>Refresh</button><div id=o></div><script>async function load(){let r=await fetch('/admin/status',{headers:{'X-Admin-Key':document.querySelector('#t').value}}),d=await r.json();if(!r.ok){document.querySelector('#o').innerText=JSON.stringify(d);return}let h='<h2>Key counts</h2><pre>'+JSON.stringify(d.key_counts,null,2)+'</pre><h2>Current selected key</h2><pre>'+JSON.stringify(d.current_selected_key,null,2)+'</pre><h2>Metrics</h2><pre>'+JSON.stringify(d.metrics,null,2)+'</pre><table><tr><th>Fingerprint</th><th>Masked</th><th>Status</th><th>Cooldown</th><th></th></tr>';for(let k of d.keys)h+='<tr><td>'+k.id+'</td><td>'+k.masked+'</td><td>'+k.status+'</td><td>'+k.cooldown_remaining_seconds+'s</td><td>'+(k.status==='DISABLED'?'<button onclick="enable(\''+k.id+'\')">Re-enable</button>':'')+'</td></tr>';document.querySelector('#o').innerHTML=h+'</table>'}async function enable(k){await fetch('/admin/keys/'+k+'/enable',{method:'POST',headers:{'X-Admin-Key':document.querySelector('#t').value}});load()}</script></body></html>""")
