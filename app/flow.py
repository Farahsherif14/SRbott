"""LangGraph orchestration for one service-request conversation."""
from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from app.ai import extract_request
from app.classifier import classify
from app.db import ConversationTurn, ServiceRequest, SessionLocal
from app.schemas import ExtractedRequest
from app.session import load_session, save_session, clear_session


class RequestState(TypedDict, total=False):
    chat_id: str
    user_id: str
    message: str
    transcript: str
    response_language: str | None
    previous_fields: dict[str, Any]
    fields: dict[str, Any]
    missing_fields: list[str]
    confirmation: bool | None
    response: str
    phase: str
    prediction: dict[str, Any]
    ticket_number: str


def transcribe_node(state: RequestState) -> dict:
    # Telegram voice is transcribed by faster-whisper before entering the graph.
    transcript = state.get("transcript") or state.get("message", "")
    language = state.get("response_language")
    if language is None:
        language = "ar" if has_arabic(transcript) else "en"
    return {"transcript": transcript, "response_language": language, "phase": "transcribed"}


def has_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" or "\u0750" <= char <= "\u077f" or "\u08a0" <= char <= "\u08ff" for char in text)


def extract_node(state: RequestState) -> dict:
    text = state.get("transcript", "")
    previous = state.get("previous_fields") or {}
    with SessionLocal() as db:
        db.add(ConversationTurn(chat_id=state["chat_id"], role="user", text=text))
        db.commit()
    extracted = extract_request(text, previous_fields=previous,
        response_language=state.get("response_language") or "en")
    new = extracted.model_dump()
    old_unit = previous.get("unit") or {}
    new_unit = new.get("unit") or {}
    merged = {**previous, **new}
    merged["unit"] = {**old_unit, **{k: v for k, v in new_unit.items() if v not in (None, "")}}
    for key in ("name", "phone", "issue_description"):
        if new.get(key) in (None, "") and previous.get(key):
            merged[key] = previous[key]
    required = {"compound": merged["unit"].get("compound"),
                "building": merged["unit"].get("building"),
                "unit_number": merged["unit"].get("unit_number"),
                "issue_description": merged.get("issue_description")}
    missing = [key for key, value in required.items() if not value]
    if merged.get("issue_needs_detail"):
        missing.append("issue_details")
    merged["missing_fields"] = missing
    return {"fields": merged, "missing_fields": missing, "confirmation": None, "phase": "extracted"}


def route_after_transcribe(state: RequestState) -> str:
    message = state.get("message", "").strip().casefold()
    existing = state.get("previous_fields") or {}
    complete = not any(not v for v in [
        (existing.get("unit") or {}).get("compound"),
        (existing.get("unit") or {}).get("building"),
        (existing.get("unit") or {}).get("unit_number"),
        existing.get("issue_description"),
    ])
    yes = {"yes", "confirm", "confirmed", "نعم", "أؤكد", "موافق", "تأكيد"}
    no = {"no", "edit", "change", "لا", "تعديل"}
    if state.get("confirmation") is not None:
        return "classify" if state["confirmation"] else "declined"
    if message in yes and complete:
        return "classify"
    if message in no and complete:
        return "declined"
    return "extract"


def missing_or_confirmation(state: RequestState) -> str:
    return "follow_up" if state.get("missing_fields") else "confirm_request"


def follow_up_node(state: RequestState) -> dict:
    arabic = state.get("response_language") == "ar"
    missing = state.get("missing_fields", [])
    details_needed = "issue_details" in missing
    other_missing = [key for key in missing if key != "issue_details"]
    if arabic:
        labels = {"compound": "اسم الكومباوند", "building": "اسم أو رقم المبنى",
                  "unit_number": "رقم الوحدة", "issue_description": "وصف المشكلة"}
        requested = "، ".join(labels.get(key, "") for key in other_missing if labels.get(key))
        question = f"من فضلك أرسل: {requested}." if requested else ""
        if details_needed:
            details_prompt = "من فضلك وضّح المشكلة أكثر: ما العَرَض، وأين يظهر، ومتى بدأ؟"
            question = f"{question}\n{details_prompt}" if question else details_prompt
    else:
        labels = {"compound": "compound", "building": "building", "unit_number": "unit number",
                  "issue_description": "problem description"}
        requested = ", ".join(labels[key] for key in other_missing if key in labels)
        question = f"Please share: {requested}." if requested else ""
        if details_needed:
            details_prompt = "Could you describe what is happening, where you notice it, and when it started?"
            question = f"{question}\n{details_prompt}" if question else details_prompt
    return {"response": question, "phase": "awaiting_fields"}


def confirmation_node(state: RequestState) -> dict:
    fields = state.get("fields") or state.get("previous_fields") or {}
    unit = fields.get("unit") or {}
    transcript = state.get("transcript") or state.get("message", "")
    arabic = state.get("response_language") == "ar" or has_arabic(transcript)
    if arabic:
        text = (
            "من فضلك راجع البيانات وأكدها:\n"
            f"المشكلة: {fields.get('issue_description') or '—'}\n"
            f"الكومباوند: {unit.get('compound') or '—'}\n"
            f"المبنى: {unit.get('building') or '—'}\n"
            f"رقم الوحدة: {unit.get('unit_number') or '—'}\n\n"
            "اكتب نعم للتأكيد أو أرسل التصحيح."
        )
    else:
        text = (
            "Please review and confirm the request:\n"
            f"Problem: {fields.get('issue_description') or '—'}\n"
            f"Compound: {unit.get('compound') or '—'}\n"
            f"Building: {unit.get('building') or '—'}\n"
            f"Unit number: {unit.get('unit_number') or '—'}\n\n"
            "Reply YES to confirm or send a correction."
        )
    return {"response": text, "phase": "awaiting_confirmation"}


def declined_node(state: RequestState) -> dict:
    response = ("حسنًا، أرسل التفاصيل المصححة وسأحدّث الطلب."
                if state.get("response_language") == "ar"
                else "Okay, send the corrected details and I’ll update the request.")
    return {"fields": state.get("previous_fields") or state.get("fields") or {},
            "response": response,
            "phase": "awaiting_corrections", "confirmation": None}


def classify_node(state: RequestState) -> dict:
    fields = state.get("fields") or state.get("previous_fields") or {}
    try:
        prediction = classify(fields.get("issue_description") or "")
    except NotImplementedError:
        response = ("تم تأكيد التفاصيل، لكن التصنيف غير جاهز بعد، لذلك لم يتم حفظ الطلب."
                    if state.get("response_language") == "ar"
                    else "Your details are confirmed, but classification is not implemented yet, so the request was not saved.")
        return {"phase": "classification_not_implemented",
                "response": response}
    return {"prediction": prediction, "phase": "classified"}


def save_node(state: RequestState) -> dict:
    fields = ExtractedRequest.model_validate(state.get("fields") or state.get("previous_fields") or {})
    prediction = state["prediction"]
    ticket = "SR-" + uuid4().hex[:8].upper()
    with SessionLocal() as db:
        db.add(ServiceRequest(ticket_number=ticket, telegram_user_id=state["user_id"],
            compound=fields.unit.compound, building=fields.unit.building,
            unit_number=fields.unit.unit_number, name=fields.name, phone=fields.phone,
            issue_description=fields.issue_description or "", category=prediction["category"],
            urgency=prediction.get("urgency", "medium"), confidence=prediction["confidence"],
            predictions=prediction))
        db.commit()
    response = (f"تم حفظ الطلب. رقم التذكرة {ticket}." if state.get("response_language") == "ar"
                else f"Request saved. Ticket {ticket}.")
    return {"ticket_number": ticket, "phase": "saved", "response": response}


builder = StateGraph(RequestState)
builder.add_node("transcribe", transcribe_node)
builder.add_node("extract", extract_node)
builder.add_node("follow_up", follow_up_node)
builder.add_node("confirm_request", confirmation_node)
builder.add_node("declined", declined_node)
builder.add_node("classify", classify_node)
builder.add_node("save", save_node)
builder.add_edge(START, "transcribe")
builder.add_conditional_edges("transcribe", route_after_transcribe,
    {"extract": "extract", "classify": "classify", "declined": "declined"})
builder.add_conditional_edges("extract", missing_or_confirmation,
    {"follow_up": "follow_up", "confirm_request": "confirm_request"})
builder.add_edge("follow_up", END)
builder.add_edge("confirm_request", END)
builder.add_edge("declined", END)
builder.add_conditional_edges("classify", lambda s: "save" if s.get("phase") == "classified" else END,
    {"save": "save", END: END})
builder.add_edge("save", END)
request_graph = builder.compile()


def run_request_flow(*, chat_id: str, user_id: str, message: str = "",
                     transcript: str | None = None, confirmed: bool | None = None,
                     fields: dict[str, Any] | None = None) -> dict[str, Any]:
    saved = load_session(chat_id, user_id)
    prior = fields or saved.get("fields") or {}
    result = request_graph.invoke({"chat_id": chat_id, "user_id": user_id,
        "message": message, "transcript": transcript, "previous_fields": prior,
        "fields": prior, "confirmation": confirmed,
        "response_language": saved.get("response_language")})
    if result.get("phase") == "saved":
        clear_session(chat_id)
    else:
        save_session(chat_id, user_id, result)
    return result
