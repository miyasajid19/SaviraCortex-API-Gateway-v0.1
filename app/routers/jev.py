from __future__ import annotations

import hmac
import re
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from redis.asyncio import Redis

from app.config import get_settings
from app.database.redis_client import get_redis
from app.services.jev_gateway import JEVGateway, JEVGatewayError

router = APIRouter(prefix="/v1")
_QUESTION_ID = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def _require_gateway_key(value: str | None) -> None:
    expected = get_settings().gateway_api_key
    if not expected or not value or not value.lower().startswith("bearer "):
        raise HTTPException(401, detail={"error": {"message": "Invalid gateway API key", "type": "authentication_error"}})
    supplied = value[7:].strip()
    if not hmac.compare_digest(expected, supplied):
        raise HTTPException(401, detail={"error": {"message": "Invalid gateway API key", "type": "authentication_error"}})


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")
    if "state" not in payload or not isinstance(payload["state"], (str, dict, list)):
        raise HTTPException(422, "state must be a string, object, or array.")
    if not isinstance(payload.get("questions"), dict) or not 1 <= len(payload["questions"]) <= 64:
        raise HTTPException(422, "questions must contain between 1 and 64 typed questions.")

    for question_id, question in payload["questions"].items():
        if not isinstance(question_id, str) or not _QUESTION_ID.fullmatch(question_id):
            raise HTTPException(422, "Question IDs may contain letters, digits, underscores, dots, and hyphens (max 100 characters).")
        if not isinstance(question, dict):
            raise HTTPException(422, f"Question '{question_id}' must be an object.")
        question_type = question.get("type")
        if question_type not in {"noul", "choice", "score"}:
            raise HTTPException(422, f"Question '{question_id}' type must be noul, choice, or score.")
        instructions = question.get("instructions")
        if instructions is None or instructions == "" or instructions == [] or instructions == {}:
            raise HTTPException(422, f"Question '{question_id}' requires non-empty instructions.")
        criteria = question.get("criteria")
        if question_type == "choice" and (
            not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255
        ):
            raise HTTPException(422, f"Question '{question_id}' choice criteria must contain 2 to 255 options.")
        if question_type == "score" and (
            not isinstance(criteria, list) or not 2 <= len(criteria) <= 10
        ):
            raise HTTPException(422, f"Question '{question_id}' score criteria must contain 2 to 10 levels.")

    task = payload.get("task", "runtime")
    if task not in {"ingestion", "runtime"}:
        raise HTTPException(422, "task must be either 'ingestion' or 'runtime'.")

    # The task is an allowlisted routing hint. Upstream model IDs remain server-side.
    return {
        "task": task,
        "state": payload["state"],
        "questions": payload["questions"],
        "images": payload.get("images"),
    }


@router.post("/judge")
async def judge(
    request: Request,
    authorization: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
):
    _require_gateway_key(authorization)
    try:
        incoming = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Request body must be valid JSON.") from None
    payload = _validate_payload(incoming)

    try:
        result = await JEVGateway(redis, request.app.state.jev_http).judge(payload, task=payload["task"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except JEVGatewayError as exc:
        response = JSONResponse(
            status_code=503,
            content={"error": {"message": exc.message, "type": "provider_unavailable"}},
        )
        if exc.retry_after:
            response.headers["Retry-After"] = str(exc.retry_after)
        return response

    headers = dict(result.headers)
    headers["X-JEV-Provider"] = result.provider
    headers["X-JEV-Model"] = result.model
    headers["X-JEV-Fallback"] = "true" if result.fallback else "false"
    headers["X-JEV-Task"] = payload["task"]
    return Response(
        content=result.body,
        status_code=result.status_code,
        headers=headers,
        media_type=result.headers.get("content-type"),
    )
