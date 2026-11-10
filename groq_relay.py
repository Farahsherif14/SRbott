"""Local relay that app/ai.py talks to (settings.groq_relay_url = http://127.0.0.1:8765/extract).

Request : POST /extract  {"model": ..., "prompt": ..., "schema": {...}}   or   {..., "json_object": true}
Response: the JSON object produced by the model (ExtractedRequest fields, or {"corrected_text": ...}).
On provider errors it answers HTTP 502 with {"error": ..., "status_code": ...}, as app/ai.py expects.
Run from the project root:  python -m uvicorn groq_relay:app --port 8765
Message text is never logged."""
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

load_dotenv()
app = FastAPI(title="Groq relay")
_client = None


def _groq():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


@app.get("/health")
def health():
    return {"status": "ok", "has_key": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/extract")
async def extract(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    schema = body.get("schema")
    if schema:
        prompt += ("\n\nReturn ONLY a JSON object that follows this JSON schema "
                   "(use null for unknown values):\n" + json.dumps(schema, ensure_ascii=False))
    model = body.get("model") or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    def call():
        return _groq().chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])

    try:
        resp = await run_in_threadpool(call)
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        return JSONResponse(status_code=502, content={
            "error": type(exc).__name__, "status_code": getattr(exc, "status_code", None)})
