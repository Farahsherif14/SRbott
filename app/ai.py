import json
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from app.config import settings
from app.schemas import ExtractedRequest


def correct_voice_transcript(text: str, response_language: str = "ar") -> str:
    """Use the LLM to clean ASR errors before extracting any request fields."""
    if not settings.groq_api_key:
        raise RuntimeError("Set GROQ_API_KEY to enable transcript correction.")
    prompt = f"""You are an Arabic (especially Egyptian dialect) and English speech-transcript proofreader for residential maintenance requests.
Correct Tawasul ASR spelling, sound-alike, word-boundary, and speech-recognition errors in the transcript before extracting any information. You are especially good at Egyptian Arabic colloquial speech. Use grammar, the surrounding sentence, and residential maintenance context to restore the words the speaker most likely said; do not preserve obviously nonsensical ASR output just because it is uncertain in isolation. Common maintenance terms include حوض المطبخ، حوض الحمام، تسريب، الحنفية، الصرف، باب الشقة، بيحك في الأرض، صوت زنة/زقزقة، and durations such as نص ساعة. For example, “شغل في الحوض الباطخ، في البثيت السريب، بيقع حاولي بلاصة ساعة تحت الحوض” in a maintenance report should be corrected toward “شغل في حوض المطبخ، بيبقى فيه تسريب بيقعد حوالي نص ساعة تحت الحوض,” preserving only the facts supported by the audio/transcript context. Resolve likely sound-alikes from context, but do not add locations, timing, causes, severity, or symptoms that are not supported. Preserve the speaker's meaning, dialect, language, code-switching, all facts, proper names, the six compound names (MARASSI, SOUL, UPTOWN CAIRO, MIVIDA, BELLEVIE, CAIRO GATE), and numeric/alphanumeric codes. Normalize a clearly spoken digit/letter code such as “seven bee” to “7B” only when unambiguous. Never invent, omit, summarize, or translate. If a phrase remains genuinely ambiguous after using context, preserve just that uncertain phrase and do not turn it into a confident fact. Return the corrected transcript in the same language(s) and script style as the input. Intended conversation language: {"Arabic" if response_language == "ar" else "English"}.
Transcript from speech recognition:
{text}"""
    prompt += "\nReturn a JSON object with exactly one key: corrected_text (string)."
    body = json.dumps({"model": settings.groq_model, "prompt": prompt,
                       "json_object": True}).encode()
    request = Request(settings.groq_relay_url, data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        try:
            error_payload = json.loads(exc.read())
            error_type = error_payload.get("error", "ProviderError")
            provider_status = error_payload.get("status_code")
        except Exception:
            error_type, provider_status = "ProviderError", None
        detail = f"{error_type}/{provider_status}" if provider_status else error_type
        raise RuntimeError(f"Transcript correction relay error: HTTP {exc.code} ({detail})") from exc
    except URLError as exc:
        raise RuntimeError("Transcript correction relay is unreachable") from exc
    corrected = payload.get("corrected_text")
    if not isinstance(corrected, str) or not corrected.strip():
        raise RuntimeError("Groq returned an empty corrected transcript")
    return corrected.strip()


def extract_request(text: str, previous_fields: dict | None = None, profile: dict | None = None,
                    response_language: str = "en") -> ExtractedRequest:
    if not settings.groq_api_key:
        raise RuntimeError("Set GROQ_API_KEY to enable Groq extraction.")
    prompt = f"""You are a careful Arabic/English residential maintenance intake assistant. A message may be typed text or a noisy automatic speech-recognition transcript of spoken Arabic (including Egyptian dialect), English, or code-switching.
First interpret the message as a whole, correcting only high-confidence transcription errors using grammar, dialect, and maintenance context. Normalize clear spoken numbers and letter/number combinations when their intended value is unambiguous. Do not treat the transcript as authoritative when it contains an obvious sound-alike or malformed phrase. Do not over-correct: never replace a plausible place/person name or technical term with a more familiar one merely because it is common. If a word or value has multiple plausible readings and it matters to the request, ask one short clarification in the conversation language; do not silently guess or invent details.
Treat the current transcript as the user's actual message and actively extract every stated fact from it. A voice message may contain all required information in one utterance, in any order, with Arabic instructions and English names/codes mixed together. Do not ask the user to re-send information that is present and understandable in the transcript. Only mark a required field missing when you cannot confidently recover it. Never infer unit or contact details. Preserve all meaningful facts, including what is wrong, exact location within the home, onset/timing, and impact, without adding unstated details.
The compound must be exactly one of these canonical names, written in ALL UPPERCASE in the confirmation: MARASSI, SOUL, UPTOWN CAIRO, MIVIDA, BELLEVIE, CAIRO GATE. Match what the user said or typed to the closest item only when the match is clear, including Arabic pronunciation/transcription and mixed Arabic-English speech (for example مراسي → MARASSI). Never choose based on unrelated context; if unclear or not in this list, ask which compound they mean. Do not put compound names in building or vice versa.
Building and unit_number have different formats and must be extracted only from their own explicit cues. A building number follows “عمارة”, “مبنى”, or “building” and must be digits, optionally followed by one Latin letter (examples: 40, 40B). A unit number follows “وحدة”, “شقة”, “flat”, or “apartment” and must contain digits only (examples: 7, 204); never attach a letter to a unit. The speaker may say the numbers in Arabic or English words or use Arabic/Latin digits. Convert only clear spoken number words to digits (for example: أربعين/forty → 40; سبعة/seven → 7). For a building letter, convert a clearly spoken trailing letter name (for example بي/bee → B); never infer a letter from surrounding context. Do not swap the building and unit, copy one value into the other field, or guess a value from a nearby number. If the cue or spoken number is ambiguous, leave that field missing and ask a focused follow-up. Keep the compound, building, and unit in their separate fields.
Required fields: compound, building, unit_number, and a sufficiently detailed issue description. Name and phone are optional. Mark issue_needs_detail=true and ask a concise follow-up in the user's language when the description is vague (for example, “something is broken”, “AC problem”, or “there is a problem”). Ask what is wrong, what symptom they see, and any useful location, timing, or impact; ask only what is still unclear. A clear report such as “water is leaking under the kitchen sink” does not need more detail. On follow-up, incorporate new facts into the prior issue description and set issue_needs_detail=false only when there is enough detail to understand the maintenance problem. Never invent facts. Offer profile details for confirmation; never silently change them.
Keep issue_description and every generated follow_up_question in the conversation language selected for this request: {"Arabic" if response_language == "ar" else "English"}. Do not switch languages even if the user's latest answer contains a short phrase in another language. Translate the combined issue description into the selected conversation language when needed, preserving all stated facts and proper names.
Previously collected fields for this same request: {previous_fields or {}}
Saved profile for this user only: {profile or {}}
Current user message: {text}"""
    body = json.dumps({"model": settings.groq_model, "prompt": prompt,
        "schema": ExtractedRequest.model_json_schema()}).encode()
    request = Request(settings.groq_relay_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        try:
            error_payload = json.loads(exc.read())
            error_type = error_payload.get("error", "ProviderError")
            provider_status = error_payload.get("status_code")
        except Exception:
            error_type, provider_status = "ProviderError", None
        detail = f"{error_type}/{provider_status}" if provider_status else error_type
        raise RuntimeError(f"Extraction relay error: HTTP {exc.code} ({detail})") from exc
    except URLError as exc:
        raise RuntimeError("Extraction relay is unreachable") from exc
    return ExtractedRequest.model_validate(payload)
