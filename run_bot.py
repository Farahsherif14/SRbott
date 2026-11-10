"""Run the local Groq relay and Telegram bot as one managed process."""
from __future__ import annotations

import subprocess
import socket
import json
import os
import sys
import time
import shutil
from urllib.parse import urlsplit
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV = Path(sys.executable)
def ollama_base_url() -> str:
    configured = os.environ.get("OLLAMA_HOST")
    if configured:
        return configured.rstrip("/")
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OLLAMA_HOST="):
                return line.split("=", 1)[1].strip().strip("\"'").rstrip("/")
    return "http://localhost:11434"


def ollama_models() -> set[str] | None:
    try:
        with urlopen(f"{ollama_base_url()}/api/tags", timeout=0.5) as response:
            payload = json.loads(response.read())
        return {item.get("name", "") for item in payload.get("models", [])}
    except (URLError, TimeoutError, ValueError):
        return None


def ensure_ollama(children: list[tuple[str, subprocess.Popen]]) -> None:
    models = ollama_models()
    if models is None:
        hostname = urlsplit(ollama_base_url()).hostname
        if hostname not in {"localhost", "127.0.0.1", "::1"}:
            print(f"Configured Ollama service is not reachable at {ollama_base_url()}.",
                  file=sys.stderr)
            return
        executable = shutil.which("ollama")
        if not executable:
            print("Ollama is not installed. ML classification needs Ollama with qwen3-embedding:4b.",
                  file=sys.stderr)
            return
        process = subprocess.Popen([executable, "serve"], cwd=ROOT)
        children.append(("Ollama", process))
        for _ in range(40):
            time.sleep(0.5)
            if process.poll() is not None:
                print("Ollama stopped during startup; classification will be unavailable.",
                      file=sys.stderr)
                return
            models = ollama_models()
            if models is not None:
                break
        else:
            print("Ollama did not become ready; classification will be unavailable.",
                  file=sys.stderr)
            return

    if not any(name == "qwen3-embedding:4b" or name.startswith("qwen3-embedding:4b-")
               for name in models or set()):
        print("ML embedding model is not installed. Run `ollama pull qwen3-embedding:4b` once; "
              "classification will be unavailable until then.", file=sys.stderr)


def main() -> int:
    if not ENV.is_file():
        print(f"Virtual environment not found: {ENV}\nCreate it and install requirements first.", file=sys.stderr)
        return 1

    try:
        with socket.create_connection(("127.0.0.1", 8765), timeout=0.3):
            print(
                "A Groq relay is already listening on port 8765.\n"
                "Stop the old relay with Ctrl+C in the Terminal where it is running, "
                "then run `python run_bot.py` again.",
                file=sys.stderr,
            )
            return 1
    except OSError:
        pass

    children: list[tuple[str, subprocess.Popen]] = []
    try:
        ensure_ollama(children)
        relay = subprocess.Popen(
            [str(ENV), "-m", "uvicorn", "groq_relay:app", "--host", "127.0.0.1", "--port", "8765"],
            cwd=ROOT,
        )
        children.append(("Groq relay", relay))

        # Give the relay a moment to catch an invalid key or occupied port.
        time.sleep(0.5)
        if relay.poll() is not None:
            print("Groq relay exited during startup; see its error above.", file=sys.stderr)
            return int(relay.returncode or 1)

        bot = subprocess.Popen([str(ENV), "-m", "app.bot"], cwd=ROOT)
        children.append(("Telegram bot", bot))
        print("\nOllama (if needed), relay, and bot are running. Press Ctrl+C to stop processes "
              "started by this launcher.\n", flush=True)

        while True:
            for name, process in children:
                code = process.poll()
                if code is not None:
                    print(f"{name} exited with status {code}; stopping the other process.", file=sys.stderr)
                    return int(code)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping services started by this launcher...", flush=True)
        return 0
    finally:
        for _, process in reversed(children):
            if process.poll() is None:
                process.terminate()
        for _, process in reversed(children):
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
