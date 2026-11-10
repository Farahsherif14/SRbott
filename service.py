"""Orchestrator: load memory -> run the LangGraph flow -> persist memory/history."""
import logging

from . import db, memory
from .graph import graph

log = logging.getLogger("srbot.service")  # logs ids and steps only, never message text
MAX_LEN = 1500


def handle_message(user_id: int, chat_id: int, text: str) -> str:
    text = (text or "").strip()[:MAX_LEN]
    session = memory.load(chat_id)
    profile = db.get_profile(user_id)          # only THIS user's profile is ever passed on
    result = graph.invoke({"session": session, "user_id": user_id, "text": text, "profile": profile})
    reply, session = result.get("reply", ""), result["session"]
    turns = session["history"] + [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    if result.get("done"):
        db.save_history(user_id, result["sr_id"], turns)
        memory.clear(chat_id)                  # cleared on submit
    else:
        session["history"] = turns
        memory.save(chat_id, session)
    log.info("turn user=%s step=%s done=%s", user_id, session.get("step"), bool(result.get("done")))
    return reply


def reset(chat_id: int):
    memory.clear(chat_id)


def forget(user_id: int, chat_id: int):
    memory.clear(chat_id)
    db.forget(user_id)