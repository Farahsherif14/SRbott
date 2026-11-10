"""LangGraph orchestration for one service-request conversation."""
from __future__ import annotations

import logging
from typing import Any, TypedDict
import re
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from app.ai import correct_voice_transcript, extract_request
from app.classifier import ClassifierUnavailable, classify
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
BUILDING_PATTERN = re.compile(r"^(?:\d+[A-Z]?|[A-Z]\d+)$", re.ASCII)
UNIT_PATTERN = re.compile(r"^\d+$", re.ASCII)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

ENGLISH_NUMBERS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}
ENGLISH_NUMBERS.update({
    "زيرو": 0, "وان": 1, "تو": 2, "ثري": 3, "فور": 4, "فايف": 5,
    "ساكس": 6, "سيكس": 6, "سفن": 7, "سيفن": 7, "ايت": 8,
    "إيت": 8, "ناين": 9, "تين": 10, "إليفن": 11, "اليفن": 11,
    "تويلف": 12, "ثيرتين": 13, "فورتي": 40, "توينتي": 20,
    "ثيرتي": 30, "فيفتي": 50, "سيكستي": 60, "سيفنتي": 70,
    "ايتي": 80, "ناينتي": 90,
})
ARABIC_NUMBERS = {
    "صفر": 0, "واحد": 1, "واحدة": 1, "واحده": 1, "احد": 1,
    "اتنين": 2, "اثنين": 2, "اثنان": 2, "تنين": 2,
    "تلاتة": 3, "تلاته": 3, "ثلاثة": 3, "ثلاثه": 3,
    "اربعة": 4, "اربعه": 4, "أربعة": 4, "أربعه": 4,
    "خمسة": 5, "خمسه": 5, "ستة": 6, "سته": 6,
    "سبعة": 7, "سبعه": 7, "تمانية": 8, "تمانيه": 8, "ثمانية": 8,
    "ثمانيه": 8, "تسعة": 9, "تسعه": 9, "عشرة": 10, "عشره": 10,
    "حداشر": 11, "احدعشر": 11, "حداشر": 11, "اتناشر": 12,
    "اثناشر": 12, "تلتاشر": 13, "تلاتاشر": 13, "اربعتاشر": 14,
    "خمستاشر": 15, "ستاشر": 16, "سبعتاشر": 17, "تمنتاشر": 18,
    "تسعتاشر": 19, "عشرين": 20, "عشرون": 20, "عشرين": 20,
    "ثلاثين": 30, "تلاتين": 30, "اربعين": 40, "أربعين": 40,
    "خمسين": 50, "ستين": 60, "سبعين": 70, "تمانين": 80,
    "ثمانين": 80, "تسعين": 90, "مائة": 100, "مية": 100,
    "مئة": 100, "الف": 1000, "ألف": 1000,
}
SPOKEN_LETTERS = {
    "a": "A", "ay": "A", "اي": "A", "إيه": "A", "أيه": "A",
    "b": "B", "bee": "B", "بي": "B", "بى": "B",
    "c": "C", "see": "C", "سي": "C", "سى": "C",
    "d": "D", "dee": "D", "دي": "D", "دى": "D",
    "e": "E", "ee": "E", "إي": "E", "اي": "E",
    "f": "F", "eff": "F", "إف": "F", "اف": "F",
    "g": "G", "gee": "G", "جي": "G", "جى": "G",
    "h": "H", "aitch": "H", "إتش": "H", "اتش": "H",
    "j": "J", "jay": "J", "جاي": "J", "كي": "K", "kay": "K",
    "l": "L", "el": "L", "إل": "L", "m": "M", "em": "M",
    "إم": "M", "n": "N", "en": "N", "إن": "N",
}


def _number_words_to_int(phrase: str) -> int | None:
    """Convert a clear Arabic or English spoken cardinal number to digits."""
    phrase = re.sub(r"[ًٌٍَُِّْـ]", "", phrase.casefold())
    tokens = re.findall(r"[A-Za-z]+|[\u0600-\u06ff]+", phrase)
    if not tokens:
        return None
    values: list[int] = []
    for token in tokens:
        token = token.strip()
        if token in {"and", "و"}:
            continue
        value = ENGLISH_NUMBERS.get(token, ARABIC_NUMBERS.get(token))
        if value is None and token.startswith("و"):
            value = ENGLISH_NUMBERS.get(token[1:], ARABIC_NUMBERS.get(token[1:]))
        if value is None:
            return None
        values.append(value)
    if not values:
        return None
    if len(values) > 1 and all(value <= 9 for value in values):
        return int("".join(str(value) for value in values))
    total = current = 0
    for value in values:
        if value == 1000:
            total += max(current, 1) * value
            current = 0
        elif value == 100:
            current = max(current, 1) * value
        else:
            current += value
    result = total + current
    return result if 0 <= result <= 9999 else None


def valid_building(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", "", value.translate(ARABIC_DIGITS).strip()).upper()
    return normalized if BUILDING_PATTERN.fullmatch(normalized) else None


def valid_unit_number(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", "", value.translate(ARABIC_DIGITS).strip())
    return normalized if UNIT_PATTERN.fullmatch(normalized) else None


def _parse_cued_number(segment: str, *, allow_letter: bool) -> str | None:
    """Parse digits or clearly spoken Arabic/English numbers after a location cue."""
    tokens = re.findall(r"[A-Za-z]+|\d+|[\u0600-\u06ff]+", segment.translate(ARABIC_DIGITS))
    while tokens and tokens[0].casefold() in {"رقم", "هو", "هي", "بتاع", "بتاعة", "number", "no"}:
        tokens.pop(0)
    if not tokens:
        return None

    candidate: list[str] = []
    letter = ""
    letter_prefix = False
    for token in tokens:
        folded = token.casefold()
        is_number = token.isdigit() or folded in ENGLISH_NUMBERS or folded in ARABIC_NUMBERS
        if not is_number and folded.startswith("و"):
            is_number = folded[1:] in ENGLISH_NUMBERS or folded[1:] in ARABIC_NUMBERS
        if is_number:
            candidate.append(token)
            continue
        spoken_letter = SPOKEN_LETTERS.get(folded)
        if not allow_letter and (spoken_letter or (len(token) == 1 and token.isascii() and token.isalpha())):
            return None
        if allow_letter and not letter and spoken_letter:
            letter = spoken_letter
            letter_prefix = not candidate
        elif allow_letter and not letter and len(token) == 1 and token.isascii() and token.isalpha():
            letter = token.upper()
            letter_prefix = not candidate
        break

    if not candidate:
        return None
    if len(candidate) == 1 and candidate[0].isdigit():
        number = int(candidate[0])
    else:
        number = _number_words_to_int(" ".join(candidate))
    if number is None:
        return None
    value = (letter + str(number)) if letter and letter_prefix else (str(number) + letter)
    return valid_building(value) if allow_letter else valid_unit_number(value)


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

    cues = {
        "building": r"(?:رقم\s+)?(?:عمارة|عماره|عمره|مبنى|مبني|building)\s*(?:رقم|number|no)?\s*[:：#-]?\s*",
        "unit_number": r"(?:رقم\s+)?(?:وحدة|وحده|شقة|شقه|flat|apartment|unit)\s*(?:number|no|رقم)?\s*[:：#-]?\s*",
    }
    found = []
    for key, pattern in cues.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            found.append((match.start(), match.end(), key))
    found.sort()
    for index, (start, end, key) in enumerate(found):
        segment_end = found[index + 1][0] if index + 1 < len(found) else len(normalized)
        segment = normalized[end:segment_end].strip(" \t,،.;:!?؟-_")
        value = _parse_cued_number(segment, allow_letter=(key == "building"))
        if value:
            recovered[key] = value
    return recovered


def canonical_compound(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().casefold().split())
    return COMPOUND_NAMES.get(normalized)


def recover_single_missing_answer(text: str, previous: dict[str, Any]) -> dict[str, Any]:
    """Map a bare typed reply to the next missing identifier field."""
    unit = previous.get("unit") or {}
    answer = text.strip().translate(ARABIC_DIGITS)
    if answer.isdigit():
        identifier = answer
        # The bot asks for the building before the unit, so a bare code answers
        # the first still-missing identifier in that same order.
        for field in ("building", "unit_number"):
            if not unit.get(field):
                return {field: (valid_building(identifier) if field == "building"
                                else valid_unit_number(identifier))}
    return {}


def recover_typed_location_answers(text: str) -> dict[str, Any]:
    """Read explicit typed building/unit answers without relying on the LLM."""
    normalized = text.translate(ARABIC_DIGITS)
    recovered: dict[str, Any] = {}
    recovered.update(recover_explicit_voice_fields(normalized))
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
    previous_phase: str
    previous_location_confirmed: dict[str, str]
    location_to_confirm: dict[str, str]
    location_confirmed: dict[str, str]
    fields: dict[str, Any]
    missing_fields: list[str]
    confirmation: bool | None
    response: str
    phase: str
    prediction: dict[str, Any]
    ticket_number: str
    location_confirmation_required: bool


def transcribe_node(state: RequestState) -> dict:
    # Telegram voice has already been transcribed locally by Tawasul.
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
            # by extracting from the original Tawasul transcript if it fails.
            detail = str(exc) if str(exc).startswith(("Transcript correction relay error:",
                                                       "Transcript correction relay is unreachable")) else type(exc).__name__
            logger.warning("Voice transcript correction failed: %s", detail)
    explicit_voice_fields = {}
    if state.get("is_voice"):
        # Tawasul may hear a code correctly while the proofreading pass
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
    merged["unit"]["building"] = valid_building(merged["unit"].get("building"))
    merged["unit"]["unit_number"] = valid_unit_number(merged["unit"].get("unit_number"))
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
    previously_confirmed = state.get("previous_location_confirmed") or {}
    location_to_confirm = {}
    if state.get("is_voice"):
        for key in ("building", "unit_number"):
            value = merged["unit"].get(key)
            if value and value != previously_confirmed.get(key):
                location_to_confirm[key] = value
    return {"fields": merged, "missing_fields": missing, "confirmation": None,
            "corrected_transcript": extraction_text if state.get("is_voice") else "",
            "location_confirmed": previously_confirmed,
            "location_to_confirm": location_to_confirm,
            "location_confirmation_required": bool(location_to_confirm),
            "phase": "extracted"}


def route_after_transcribe(state: RequestState) -> str:
    message = state.get("message", "").strip().casefold()
    existing = state.get("previous_fields") or {}
    session_phase = state.get("previous_phase")
    yes = {"yes", "confirm", "confirmed", "نعم", "أؤكد", "موافق", "تأكيد", "صح", "صحيح"}
    no = {"no", "لا", "غلط", "مش صحيح", "غير صحيح", "تعديل"}
    if session_phase == "awaiting_location_confirmation":
        if message in yes:
            return "accept_location"
        if message in no:
            return "request_location_correction"
    complete = not any(not v for v in [
        (existing.get("unit") or {}).get("compound"),
        (existing.get("unit") or {}).get("building"),
        (existing.get("unit") or {}).get("unit_number"),
        existing.get("issue_description"),
    ])
    if state.get("confirmation") is not None:
        return "classify" if state["confirmation"] else "declined"
    if message in yes and complete:
        return "classify"
    if message in no and complete:
        return "declined"
    return "extract"


def missing_or_confirmation(state: RequestState) -> str:
    if state.get("location_confirmation_required"):
        return "confirm_location"
    return "follow_up" if state.get("missing_fields") else "confirm_request"


def confirm_location_node(state: RequestState) -> dict:
    location = state.get("location_to_confirm") or {}
    if state.get("response_language") == "ar":
        recognized = []
        if location.get("building"):
            recognized.append(f"رقم المبنى {location['building']}")
        if location.get("unit_number"):
            recognized.append(f"رقم الوحدة {location['unit_number']}")
        response = (f"فهمت {' و'.join(recognized)}.\n"
                    "هل الرقم صحيح؟ اكتب «نعم» للتأكيد، أو أرسل التصحيح. "
                    "رقم الوحدة أرقام فقط، ورقم المبنى يمكن أن يكون حرفًا واحدًا قبل الأرقام أو بعدها.")
    else:
        recognized = []
        if location.get("building"):
            recognized.append(f"building {location['building']}")
        if location.get("unit_number"):
            recognized.append(f"unit {location['unit_number']}")
        response = (f"I heard {' and '.join(recognized)}.\n"
                    "Is that correct? Reply YES to confirm or send a correction. "
                    "A unit number contains digits only; a building number may have one letter before or after the digits.")
    return {"response": response, "phase": "awaiting_location_confirmation"}


def accept_location_node(state: RequestState) -> dict:
    fields = state.get("previous_fields") or {}
    previously_confirmed = state.get("previous_location_confirmed") or {}
    newly_confirmed = state.get("location_to_confirm") or {}
    if not newly_confirmed:
        # Backward compatibility for a pending confirmation created before
        # confirmed values were tracked separately in the session.
        unit = fields.get("unit") or {}
        newly_confirmed = {key: unit[key] for key in ("building", "unit_number") if unit.get(key)}
    confirmed = {**previously_confirmed, **newly_confirmed}
    required = {"compound": (fields.get("unit") or {}).get("compound"),
                "building": (fields.get("unit") or {}).get("building"),
                "unit_number": (fields.get("unit") or {}).get("unit_number"),
                "issue_description": fields.get("issue_description")}
    missing = [key for key, value in required.items() if not value]
    if fields.get("issue_needs_detail"):
        missing.append("issue_details")
    return {"fields": fields, "missing_fields": missing,
            "location_confirmed": confirmed, "location_to_confirm": {},
            "location_confirmation_required": False, "phase": "location_confirmed"}


def request_location_correction_node(state: RequestState) -> dict:
    if state.get("response_language") == "ar":
        response = "من فضلك أرسل رقم المبنى ورقم الوحدة الصحيحين، مثل: المبنى B40 أو 40B، والوحدة 7."
    else:
        response = "Please send the correct building and unit numbers, for example: building B40 or 40B, unit 7."
    return {"fields": state.get("previous_fields") or {},
            "response": response, "phase": "awaiting_location_correction"}


def follow_up_node(state: RequestState) -> dict:
    arabic = state.get("response_language") == "ar"
    missing = state.get("missing_fields", [])
    details_needed = "issue_details" in missing
    other_missing = [key for key in missing if key != "issue_details"]
    if arabic:
        labels = {"compound": "اسم الكومباوند (Marassi أو Soul أو Uptown Cairo أو Mivida أو Bellevie أو Cairo Gate)",
                  "building": "رقم المبنى (حرف واحد قبل الأرقام أو بعدها، مثل B40 أو 40B)",
                  "unit_number": "رقم الوحدة (أرقام فقط بدون حروف)", "issue_description": "وصف المشكلة"}
        requested = "، ".join(labels.get(key, "") for key in other_missing if labels.get(key))
        question = f"من فضلك أرسل أو قل في رسالة صوتية: {requested}." if requested else ""
        if details_needed:
            details_prompt = "لم أتمكن من تحديد المشكلة من الرسالة. من فضلك أعد وصفها كتابةً أو في رسالة صوتية واضحة، مع ذكر العَرَض ومكانه ومتى بدأ."
            question = f"{question}\n{details_prompt}" if question else details_prompt
    else:
        labels = {"compound": "compound (Marassi, Soul, Uptown Cairo, Mivida, Bellevie, or Cairo Gate)",
                  "building": "building number (one letter before or after the digits, e.g. B40 or 40B)",
                  "unit_number": "unit number (digits only, no letters)",
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
    except ClassifierUnavailable as exc:
        logger.error("Request classifier unavailable: %s", exc)
        response = (
            "تم تأكيد طلبك، لكن التصنيف المحلي غير متاح الآن. شغّل Ollama وتأكد من تثبيت "
            "qwen3-embedding:4b، ثم أرسل «نعم» لإعادة المحاولة. لم يتم حفظ الطلب بعد."
            if state.get("response_language") == "ar" else
            "Your request is confirmed, but local classification is unavailable. Start Ollama "
            "and ensure qwen3-embedding:4b is installed, then reply YES to retry. The request "
            "has not been saved yet."
        )
        return {"phase": "classification_unavailable", "response": response}
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
    if state.get("response_language") == "ar":
        categories = {"plumbing": "سباكة", "electricity": "كهرباء", "AC": "تكييف",
                      "carpentry": "نجارة", "pest control": "مكافحة آفات", "general": "عام"}
        urgencies = {"low": "منخفضة", "medium": "متوسطة", "high": "عالية", "emergency": "طارئة"}
        category_text = categories.get(prediction["category"], prediction["category"])
        urgency_text = urgencies.get(prediction.get("urgency", "medium"), "متوسطة")
        confidence_note = " (مبدئي)" if prediction.get("needs_review") else ""
        response = (f"تم حفظ الطلب وتصنيفه{confidence_note}: {category_text}، "
                    f"الأولوية {urgency_text}. رقم التذكرة {ticket}.")
    else:
        confidence_note = " (provisional)" if prediction.get("needs_review") else ""
        response = (f"Request saved and classified{confidence_note}: {prediction['category']}, "
                    f"priority {prediction.get('urgency', 'medium')}. Ticket {ticket}.")
    return {"ticket_number": ticket, "phase": "saved", "response": response}


builder = StateGraph(RequestState)
builder.add_node("transcribe", transcribe_node)
builder.add_node("extract", extract_node)
builder.add_node("follow_up", follow_up_node)
builder.add_node("confirm_request", confirmation_node)
builder.add_node("confirm_location", confirm_location_node)
builder.add_node("accept_location", accept_location_node)
builder.add_node("request_location_correction", request_location_correction_node)
builder.add_node("declined", declined_node)
builder.add_node("classify", classify_node)
builder.add_node("save", save_node)
builder.add_edge(START, "transcribe")
builder.add_conditional_edges("transcribe", route_after_transcribe,
    {"extract": "extract", "classify": "classify", "declined": "declined",
     "accept_location": "accept_location",
     "request_location_correction": "request_location_correction"})
builder.add_conditional_edges("extract", missing_or_confirmation,
    {"follow_up": "follow_up", "confirm_request": "confirm_request",
     "confirm_location": "confirm_location"})
builder.add_conditional_edges("accept_location", missing_or_confirmation,
    {"follow_up": "follow_up", "confirm_request": "confirm_request"})
builder.add_edge("follow_up", END)
builder.add_edge("confirm_request", END)
builder.add_edge("confirm_location", END)
builder.add_edge("request_location_correction", END)
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
        "previous_phase": saved.get("phase", ""),
        "previous_location_confirmed": saved.get("location_confirmed", {}),
        "location_confirmed": saved.get("location_confirmed", {}),
        "location_to_confirm": saved.get("location_to_confirm", {}),
        "is_voice": is_voice,
        "fields": prior, "confirmation": confirmed,
        "response_language": saved.get("response_language")})
    if result.get("phase") == "saved":
        clear_session(chat_id)
    else:
        save_session(chat_id, user_id, result)
    return result
