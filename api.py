import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import db, service, stt
from .messages import FORGET_DONE, RESET_DONE, t

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("srbot.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_schema()
    yield


app = FastAPI(title="Service Request Bot API", lifespan=lifespan)


class Msg(BaseModel):
    user_id: int
    chat_id: int
    text: str


class Ids(BaseModel):
    user_id: int
    chat_id: int


def _fail(e: Exception):
    log.error("request failed: %s", type(e).__name__)  # type only: no personal data in logs
    raise HTTPException(status_code=503, detail="backend error")


@app.post("/message")
def message(m: Msg):
    try:
        return {"reply": service.handle_message(m.user_id, m.chat_id, m.text)}
    except Exception as e:
        _fail(e)


@app.post("/voice")
async def voice(request: Request, user_id: int, chat_id: int):
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        transcript = await run_in_threadpool(stt.transcribe, audio)
        if not transcript:
            return {"reply": t("en", "voice_unclear") + "\n" + t("ar", "voice_unclear"), "transcript": ""}
        reply = await run_in_threadpool(service.handle_message, user_id, chat_id, transcript)
        return {"reply": reply, "transcript": transcript}
    except Exception as e:
        _fail(e)


@app.get("/status/{ticket}")
def status(ticket: str, user_id: int):
    row = db.get_sr_status(ticket.strip().upper(), user_id)   # only the owner can see a ticket
    if not row:
        return {"reply": t("en", "status_none") + "\n" + t("ar", "status_none")}
    return {"reply": t("en", "status", ticket=row["ticket"], status=row["status"], category=row["category"],
                       urgency=row["urgency"], created=row["created_at"].strftime("%Y-%m-%d %H:%M"))}


@app.post("/reset")
def reset(ids: Ids):
    service.reset(ids.chat_id)
    return {"reply": RESET_DONE}


@app.post("/forget")
def forget(ids: Ids):
    service.forget(ids.user_id, ids.chat_id)
    return {"reply": FORGET_DONE}