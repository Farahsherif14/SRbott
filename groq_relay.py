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


<<<<<<< HEAD
def _groq():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client
=======
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/extract":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 100_000:
                self.send_error(413)
                return
            request = json.loads(self.rfile.read(size))
            response_format = ({"type": "json_object"} if request.get("json_object") else {
                "type": "json_schema", "json_schema": {
                    "name": "service_request_extraction", "strict": False,
                    "schema": request["schema"],
                }
            })
            completion_options = {
                "model": request["model"],
                "messages": [
                    {"role": "system", "content": "Return only valid JSON matching the supplied schema. Never invent missing information."},
                    {"role": "user", "content": request["prompt"]},
                ],
                "response_format": response_format,
                # GPT-OSS expects its native temperature; JSON response modes
                # provide structure without forcing unsupported sampling.
                "temperature": 1,
                "include_reasoning": False,
            }
            response = Groq(api_key=settings.groq_api_key).chat.completions.create(**completion_options)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Groq returned an empty response")
            result = json.loads(content)
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            # Log only the exception class/status: never log prompts, transcripts,
            # API keys, or provider response bodies.
            error_type = type(exc).__name__
            status_code = getattr(exc, "status_code", None)
            detail = str(exc).replace("\n", " ")[:500]
            print(f"Groq relay error: {error_type}; status={status_code}; detail={detail}")
            body = json.dumps({"error": error_type, "status_code": status_code}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("Groq relay: request received")
>>>>>>> origin/main


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
