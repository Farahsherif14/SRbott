import json
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from app.config import settings
from app.schemas import ExtractedRequest

def extract_request(text: str, previous_fields: dict | None = None, profile: dict | None = None,
                    response_language: str = "en") -> ExtractedRequest:
    if not settings.groq_api_key:
        raise RuntimeError("Set GROQ_API_KEY to enable Groq extraction.")
    prompt = f"""You are a careful Arabic/English residential maintenance intake assistant.
Extract only information stated by the current user, the prior messages in this same chat session, or this user's saved profile. Never infer unit or contact details.
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
    except (URLError, HTTPError) as exc:
        raise RuntimeError(f"Local Groq relay unavailable at {settings.groq_relay_url}: {exc}") from exc
    return ExtractedRequest.model_validate(payload)
