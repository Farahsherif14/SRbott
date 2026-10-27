"""Short-lived per-chat draft storage (Redis when available, local fallback otherwise)."""
import json
from redis import Redis
from redis.exceptions import RedisError
from app.config import settings

redis = Redis.from_url(settings.redis_url, decode_responses=True)
_fallback: dict[str, dict] = {}

def _key(chat_id: str) -> str:
    return f"service-request:session:{chat_id}"

def load_session(chat_id: str, user_id: str) -> dict:
    try:
        raw = redis.get(_key(chat_id))
        value = json.loads(raw) if raw else None
    except (RedisError, ValueError):
        value = _fallback.get(chat_id)
    if not value or value.get("user_id") != user_id:
        return {}
    return value

def save_session(chat_id: str, user_id: str, state: dict) -> None:
    value = {"user_id": user_id, "fields": state.get("fields") or state.get("previous_fields") or {},
             "phase": state.get("phase"), "missing_fields": state.get("missing_fields", []),
             "response_language": state.get("response_language", "en")}
    _fallback[chat_id] = value
    try:
        redis.setex(_key(chat_id), settings.session_ttl_seconds, json.dumps(value, ensure_ascii=False))
    except RedisError:
        pass

def clear_session(chat_id: str) -> None:
    _fallback.pop(chat_id, None)
    try:
        redis.delete(_key(chat_id))
    except RedisError:
        pass
