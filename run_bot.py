"""Run the local Groq relay and Telegram bot as one managed process."""
from __future__ import annotations

import subprocess
import socket
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv" / "bin" / "python"


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
    bot_started = False
    try:
        relay = subprocess.Popen([str(ENV), "groq_relay.py"], cwd=ROOT)
        children.append(("Groq relay", relay))

        # Give the relay a moment to catch an invalid key or occupied port.
        time.sleep(0.5)
        if relay.poll() is not None:
            print("Groq relay exited during startup; see its error above.", file=sys.stderr)
            return int(relay.returncode or 1)

        bot = subprocess.Popen([str(ENV), "-m", "app.bot"], cwd=ROOT)
        children.append(("Telegram bot", bot))
        bot_started = True
        print("\nRelay and bot started. Press Ctrl+C to stop both.\n", flush=True)

        while True:
            for name, process in children:
                code = process.poll()
                if code is not None:
                    print(f"{name} exited with status {code}; stopping the other process.", file=sys.stderr)
                    return int(code)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping relay and bot...", flush=True)
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
