"""Local-only bridge for Groq calls from the app's restricted network context."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from groq import Groq
from app.config import settings


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
            response = Groq(api_key=settings.groq_api_key).chat.completions.create(
                model=request["model"],
                messages=[
                    {"role": "system", "content": "Return only valid JSON matching the supplied schema. Never invent missing information."},
                    {"role": "user", "content": request["prompt"]},
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "service_request_extraction", "strict": False,
                    "schema": request["schema"],
                }},
                temperature=0,
            )
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
            # Avoid logging request text or credentials.
            body = json.dumps({"error": type(exc).__name__}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("Groq relay: request received")


if __name__ == "__main__":
    if not settings.groq_api_key:
        raise SystemExit("Set GROQ_API_KEY in .env before starting the relay.")
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Groq relay listening on http://127.0.0.1:8765 (local machine only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Groq relay")
        server.server_close()
