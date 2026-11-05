from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import select, func
from app.config import settings
from app.db import init_db, SessionLocal, ServiceRequest
from app.schemas import IntakeMessage, ConfirmRequest, StatusRequest
from app.flow import run_request_flow
from app.session import clear_session

app = FastAPI(title="Service Request Bot", version="0.1.0")
redis = Redis.from_url(settings.redis_url, decode_responses=True)

@app.on_event("startup")
def startup(): init_db()

def remember(chat_id: str, role: str, text: str):
    try:
        redis.rpush(f"history:{chat_id}", f"{role}: {text[:1000]}")
        redis.ltrim(f"history:{chat_id}", -12, -1)
        redis.expire(f"history:{chat_id}", settings.session_ttl_seconds)
    except RedisError: pass

@app.get("/health")
def health(): return {"status": "ok", "provider": "Groq"}

@app.post("/intake")
def intake(payload: IntakeMessage):
    remember(payload.chat_id, "user", payload.text)
    try:
        result = run_request_flow(chat_id=payload.chat_id, user_id=payload.user_id, message=payload.text)
    except Exception as exc:
        raise HTTPException(503, f"Request flow failed: {type(exc).__name__}: {exc}")
    remember(payload.chat_id, "assistant", result.get("response", ""))
    fields = result.get("fields") or result.get("previous_fields") or {}
    return {"fields": fields, "complete": not result.get("missing_fields"),
            "needs_confirmation": result.get("phase") == "awaiting_confirmation",
            "phase": result.get("phase"), "response": result.get("response")}

@app.post("/confirm")
def confirm(payload: ConfirmRequest):
    try:
        result = run_request_flow(chat_id=payload.chat_id, user_id=payload.user_id,
            confirmed=payload.confirmed, fields=payload.fields.model_dump())
    except Exception as exc:
        raise HTTPException(503, f"Request flow failed: {type(exc).__name__}: {exc}")
    return {"saved": result.get("phase") == "saved", "phase": result.get("phase"),
            "ticket_number": result.get("ticket_number"), "message": result.get("response")}

@app.post("/status")
def status(payload: StatusRequest):
    with SessionLocal() as db:
        sr = db.scalar(select(ServiceRequest).where(ServiceRequest.ticket_number == payload.ticket_number.upper()))
        if not sr: raise HTTPException(404, "Ticket not found")
        return {"ticket_number": sr.ticket_number, "status": sr.status, "created_at": sr.created_at.isoformat()}

@app.delete("/profile/{user_id}")
def forget(user_id: str):
    with SessionLocal() as db:
        db.query(ServiceRequest).filter(ServiceRequest.telegram_user_id == user_id).update({ServiceRequest.name: None, ServiceRequest.phone: None})
        db.commit()
    clear_session(user_id)
    return {"forgotten": True}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    with SessionLocal() as db:
        total = db.scalar(select(func.count()).select_from(ServiceRequest)) or 0
        categories = db.execute(select(ServiceRequest.category, func.count()).group_by(ServiceRequest.category)).all()
        urgencies = db.execute(select(ServiceRequest.urgency, func.count()).group_by(ServiceRequest.urgency)).all()
        recent = db.scalars(select(ServiceRequest).order_by(ServiceRequest.created_at.desc()).limit(25)).all()
    rows = "".join(f"<tr><td>{s.ticket_number}</td><td>{s.created_at:%Y-%m-%d %H:%M}</td><td>{s.compound or ''} / {s.building or ''} / {s.unit_number or ''}</td><td>{s.category}</td><td>{s.urgency}</td><td>{s.status}</td></tr>" for s in recent)
    cats = " · ".join(f"{k}: {v}" for k,v in categories) or "No requests yet"
    urg = " · ".join(f"{k}: {v}" for k,v in urgencies) or "No requests yet"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Service Requests</title><style>body{{font:16px system-ui;background:#f5f6fa;color:#19212e;margin:40px}}main{{max-width:1100px;margin:auto}}section,table{{background:white;padding:20px;border-radius:14px;margin:16px 0;width:100%;box-sizing:border-box}}table{{border-collapse:collapse}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #eee}}small{{color:#64748b}}</style></head><body><main><h1>Service Requests</h1><small>Live operational snapshot</small><section><h2>{total} total requests</h2><p><b>Categories</b> · {cats}</p><p><b>Urgency</b> · {urg}</p></section><table><thead><tr><th>Ticket</th><th>Created</th><th>Unit</th><th>Category</th><th>Urgency</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></main></body></html>"""
