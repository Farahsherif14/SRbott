import json

from . import llm
from .config import MAX_TURNS, REDIS_URL, SESSION_TTL_SECONDS
from .messages import FIELDS

_redis = None


def _r():
    global _redis
    if _redis is None:
        import redis
        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _key(chat_id: int) -> str:
    return f"session:{chat_id}"


def new_session() -> dict:
    return {"step": "collecting", "lang": "en", "fields": {k: None for k in FIELDS},
            "history": [], "summary": "", "asked_saved_unit": False, "pending_unit": None}


def load(chat_id: int) -> dict:
    raw = _r().get(_key(chat_id))
    return json.loads(raw) if raw else new_session()


def save(chat_id: int, s: dict):
    keep = MAX_TURNS * 2
    if len(s["history"]) > keep:                      # fixed budget: summarize older turns
        overflow, s["history"] = s["history"][:-keep], s["history"][-keep:]
        s["summary"] = llm.summarize(overflow, s.get("summary", ""))
    _r().set(_key(chat_id), json.dumps(s, ensure_ascii=False), ex=SESSION_TTL_SECONDS)


def clear(chat_id: int):
    _r().delete(_key(chat_id))