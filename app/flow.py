"""LangGraph orchestration for one service-request conversation."""
from __future__ import annotations

import logging
from typing import Any, TypedDict
import re
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from app.ai import correct_voice_transcript, extract_request
from app.classifier import classify
from app.db import ConversationTurn, ServiceRequest, SessionLocal
from app.schemas import ExtractedRequest
from app.session import load_session, save_session, clear_session

logger = logging.getLogger(__name__)


COMPOUND_NAMES = {
    "marassi": "MARASSI",
    "soul": "SOUL",
    "uptown cairo": "UPTOWN CAIRO",
    "mivida": "MIVIDA",
    "bellevie": "BELLEVIE",
    "cairo gate": "CAIRO GATE",
}
IDENTIFIER_PATTERN = re.compile(r"^(?:[A-Za-z]?\d+[A-Za-z]?)$", re.ASCII)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def recover_explicit_voice_fields(text: str) -> dict[str, Any]:
    """Recover high-confidence location details from noisy Egyptian ASR text.

    This is deliberately conservative: compounds must match the supported list,
    and identifiers must follow an explicit building/unit cue.
    """
    normalized = text.translate(ARABIC_DIGITS)
    folded = " ".join(normalized.casefold().split())
    recovered: dict[str, Any] = {}

    # Match Arabic pronunciations and common ASR spellings to canonical names.
    compound_aliases = {
        "marassi": ("مراسي", "مراسى", "ماراسي", "مارسى"),
        "soul": ("سول", "سولْ"),
        "uptown cairo": ("اب تاون كايرو", "أب تاون كايرو", "ابتاون كايرو", "اب تاون القاهرة"),
        "mivida": ("ميفيدا", "مي فيدا"),
        "bellevie": ("بيل في", "بيل فيو", "بيلفيف", "بيلڤي"),
        "cairo gate": ("كايرو جيت", "القاهرة جيت", "كائرو جيت", "كايرو gate", "كائرو gate",
                       "القاهرة gate", "cairo gate"),
    }
    for canonical, aliases in compound_aliases.items():
        if re.search(r"(?<!\w)" + re.escape(canonical) + r"(?!\w)", folded):
            recovered["compound"] = COMPOUND_NAMES[canonical]
            break
        if any(alias in folded for alias in aliases):
            recovered["compound"] = COMPOUND_NAMES[canonical]
            break
        if canonical == "cairo gate" and re.search(
                r"(?:كايرو|كائرو|القاهرة)\s*(?:gate|جيت)|cairo\s*gate", folded, re.IGNORECASE):
            recovered["compound"] = COMPOUND_NAMES[canonical]
            break

    # Strip a spoken Arabic letter name following a digit, e.g. ٧ بي -> 7B.
    cues = {
        "building": r"(?:عمارة|مبنى|مبني|building)\s*[:：-]?\s*([A-Za-z]?\s*\d+\s*(?:[A-Za-z]|بي|سي|دي|إيه|اي|إف|اف|جي|كي|إل|إم|ام|إن|ان)?)",
        "unit_number": r"(?:شقة|وحدة|رقم\s+الوحدة|unit)\s*[:：-]?\s*(\d+\s*(?:[A-Za-z]|بي|سي|دي|إيه|اي|إف|اف|جي|كي|إل|إم|ام|إن|ان)?)",
    }
    for key, pattern in cues.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            value = match.group(1).translate(ARABIC_DIGITS).strip().upper()
            value = re.sub(r"\s+", "", value)
            for spoken, letter in (("بي", "B"), ("سي", "C"), ("دي", "D"),
                                   ("إيه", "A"), ("اي", "A"), ("إف", "F"),
                                   ("اف", "F"), ("جي", "G"), ("كي", "K"),
                                   ("إل", "L"), ("إم", "M"), ("ام", "M"),
                                   ("إن", "N"), ("ان", "N")):
                value = value.replace(spoken.upper(), letter)
            value = valid_identifier(value)
            if value:
                recovered[key] = value

    # Egyptian Arabic ASR often joins the English code “four G” into “فورج”.
    if "unit_number" not in recovered:
        spoken_four_g = re.search(
            r"(?:شقة|وحدة|رقم\s+الوحدة)\s*[:：-]?\s*(?:فور\s*جي|فورج|four\s*[- ]?g)(?![A-Za-z])",
            normalized, flags=re.IGNORECASE)
        if spoken_four_g:
            recovered["unit_number"] = "4G"

    # Whisper commonly renders the Egyptian phrase "عمارة ٤٠، شقة 7B"
    # as a run-on sound-alike like "عمره 40 شأة 7B". Recover the ordered
    # number/code pair only when both occur after a recognized compound.
    if "compound" in recovered and not {"building", "unit_number"}.issubset(recovered):
        tail = normalized
        for alias in compound_aliases.get(next(k for k, v in COMPOUND_NAMES.items()
                                                if v == recovered["compound"]), ()):
            if alias in folded:
                tail = normalized[folded.index(alias) + len(alias):]
                break
        pair = re.search(r"(?<![A-Za-z0-9])(\d{1,4})\s+(?:شقة|شأة|شقه|وحدة)?\s*([A-Za-z](?![A-Za-z]))", tail, re.IGNORECASE)
        if pair:
            recovered.setdefault("building", pair.group(1))
            recovered.setdefault("unit_number", (pair.group(1) + pair.group(2)).upper())
    return recovered


def canonical_compound(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().casefold().split())
    return COMPOUND_NAMES.get(normalized)


def valid_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", "", value.translate(ARABIC_DIGITS).strip())
    if not IDENTIFIER_PATTERN.fullmatch(normalized):
        return None
    return normalized.upper()


def recover_single_missing_answer(text: str, previous: dict[str, Any]) -> dict[str, Any]:
    """Map a bare typed reply to the next missing identifier field."""
    unit = previous.get("unit") or {}
    answer = text.strip()
    identifier = valid_identifier(answer)
    if identifier:
        # The bot asks for the building before the unit, so a bare code answers
        # the first still-missing identifier in that same order.
        for field in ("building", "unit_number"):
            if not unit.get(field):
                return {field: identifier}
    return {}


def recover_typed_location_answers(text: str) -> dict[str, Any]:
    """Read explicit typed building/unit answers without relying on the LLM."""
    normalized = text.translate(ARABIC_DIGITS)
    recovered: dict[str, Any] = {}
    patterns = {
        "building": r"(?:building|عمارة|مبنى|مبني|رقم\s+المبنى)\s*[:：#-]?\s*([A-Za-z]?\s*\d+\s*[A-Za-z]?)",
        "unit_number": r"(?:unit(?:\s+number)?|apartment|flat|وحدة|رقم\s+الوحدة|شقة)\s*[:：#-]?\s*(\d+\s*[A-Za-z]?)",
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            value = valid_identifier(match.group(1))
            if value:
                recovered[field] = value
    return recovered


class RequestState(TypedDict, total=False):
    chat_id: str
    user_id: str
    message: str
    transcript: str
    corrected_transcript: str
    is_voice: bool
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
    # Telegram voice has already been transcribed locally by Vosk.
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
    extraction_text = text
    if state.get("is_voice") and text:
        try:
            extraction_text = correct_voice_transcript(
                text, response_language=state.get("response_language") or "ar")
        except Exception as exc:
            # Transcript correction is an enhancement. Preserve intake availability
            # by extracting from the original Whisper transcript if it fails.
            detail = str(exc) if str(exc).startswith(("Transcript correction relay error:",
                                                       "Transcript correction relay is unreachable")) else type(exc).__name__
            logger.warning("Voice transcript correction failed: %s", detail)
    explicit_voice_fields = {}
    if state.get("is_voice"):
        # Whisper/Vosk may hear a code correctly while the proofreading pass
        # drops or alters it. Prefer values explicitly present in the raw ASR.
        explicit_voice_fields.update(recover_explicit_voice_fields(extraction_text))
        explicit_voice_fields.update(recover_explicit_voice_fields(text))
    typed_location_answers = {}
    if not state.get("is_voice"):
        typed_location_answers.update(recover_explicit_voice_fields(text))
        typed_location_answers.update(recover_typed_location_answers(text))
    short_text_answer = (recover_single_missing_answer(text, previous)
                         if not state.get("is_voice") else {})
    extracted = extract_request(extraction_text, previous_fields=previous,
        response_language=state.get("response_language") or "en")
    new = extracted.model_dump()
    old_unit = previous.get("unit") or {}
    new_unit = new.get("unit") or {}
    merged = {**previous, **new}
    merged["unit"] = {**old_unit, **{k: v for k, v in new_unit.items() if v not in (None, "")}}
    # Deterministic, explicitly cued values from voice transcripts take
    # precedence over an LLM's missing or malformed interpretation.
    merged["unit"].update(explicit_voice_fields)
    # A one-value text reply is an answer to the one field we just requested.
    # Apply it after model output so a null/misassigned extraction cannot cause
    # the same follow-up prompt to repeat.
    for key in ("compound", "building", "unit_number"):
        if key in typed_location_answers:
            merged["unit"][key] = typed_location_answers[key]
        elif key in short_text_answer:
            merged["unit"][key] = short_text_answer[key]
    if short_text_answer.get("issue_description"):
        merged["issue_description"] = short_text_answer["issue_description"]
    # Do not present a failed ASR correction as a polished, certain issue.
    # Keep the raw transcript visible and request a focused correction instead.
    if state.get("is_voice") and extraction_text == text and not previous.get("issue_description"):
        if has_arabic(text) and re.search(r"(?:الباطخ|البثيت|حاولي بلاصة|زن جلد|الشأة)", text):
            merged["issue_description"] = None
            merged["issue_needs_detail"] = True
    # Normalize and validate location identifiers before confirmation so a
    # spoken word or unsupported compound can never appear as accepted data.
    merged["unit"]["compound"] = canonical_compound(merged["unit"].get("compound"))
    merged["unit"]["building"] = valid_identifier(merged["unit"].get("building"))
    merged["unit"]["unit_number"] = valid_identifier(merged["unit"].get("unit_number"))
    for key in ("name", "phone", "issue_description"):
        if new.get(key) in (None, "") and previous.get(key):
            merged[key] = previous[key]
    # A prior vague-issue flag must not override a clearer description supplied
    # in the latest message or merged from the current draft.
    if merged.get("issue_description") and not new.get("issue_needs_detail"):
        merged["issue_needs_detail"] = False
    required = {"compound": merged["unit"].get("compound"),
                "building": merged["unit"].get("building"),
                "unit_number": merged["unit"].get("unit_number"),
                "issue_description": merged.get("issue_description")}
    missing = [key for key, value in required.items() if not value]
    if merged.get("issue_needs_detail"):
        missing.append("issue_details")
    merged["missing_fields"] = missing
    return {"fields": merged, "missing_fields": missing, "confirmation": None,
            "corrected_transcript": extraction_text if state.get("is_voice") else "",
            "phase": "extracted"}


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
        labels = {"compound": "اسم الكومباوند (Marassi أو Soul أو Uptown Cairo أو Mivida أو Bellevie أو Cairo Gate)",
                  "building": "رقم المبنى (رقم أو رقم مع حرف واحد)",
                  "unit_number": "رقم الوحدة (رقم أو رقم مع حرف واحد)", "issue_description": "وصف المشكلة"}
        requested = "، ".join(labels.get(key, "") for key in other_missing if labels.get(key))
        question = f"من فضلك أرسل أو قل في رسالة صوتية: {requested}." if requested else ""
        if details_needed:
            details_prompt = "لم أتمكن من تحديد المشكلة من الرسالة. من فضلك أعد وصفها كتابةً أو في رسالة صوتية واضحة، مع ذكر العَرَض ومكانه ومتى بدأ."
            question = f"{question}\n{details_prompt}" if question else details_prompt
    else:
        labels = {"compound": "compound (Marassi, Soul, Uptown Cairo, Mivida, Bellevie, or Cairo Gate)",
                  "building": "building number (digits, optionally followed by one letter)",
                  "unit_number": "unit number (digits, optionally followed by one letter)",
                  "issue_description": "problem description"}
        requested = ", ".join(labels[key] for key in other_missing if key in labels)
        question = f"Please type or say in a voice message: {requested}." if requested else ""
        if details_needed:
            details_prompt = "I couldn’t determine the problem from your message. Please describe it again in text or a clear voice message, including the symptom, where it appears, and when it started."
            question = f"{question}\n{details_prompt}" if question else details_prompt
    transcript = (state.get("corrected_transcript") or state.get("transcript") or "").strip()
    if transcript and state.get("is_voice"):
        heard = (f"سمعت من رسالتك: «{transcript}».\n" if arabic
                 else f'I heard: “{transcript}”.\n')
        question = heard + question
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
                     fields: dict[str, Any] | None = None,
                     is_voice: bool = False) -> dict[str, Any]:
    saved = load_session(chat_id, user_id)
    prior = fields or saved.get("fields") or {}
    result = request_graph.invoke({"chat_id": chat_id, "user_id": user_id,
        "message": message, "transcript": transcript, "previous_fields": prior,
        "is_voice": is_voice,
        "fields": prior, "confirmation": confirmed,
        "response_language": saved.get("response_language")})
    if result.get("phase") == "saved":
        clear_session(chat_id)
    else:
        save_session(chat_id, user_id, result)
    return result
