# Service Request Bot

Initial bilingual (Arabic/English) service request prototype. The attached brief provides project requirements; its Azure references are replaced with Groq per the owner's instruction.

## First iteration

- FastAPI intake API for text requests and transcribed voice.
- Groq structured extraction with missing-field follow-ups.
- LangGraph orchestration for transcription handoff, extraction, missing-field loops, confirmation, classification, and saving.
- Empty classifier placeholder for category and urgency; classification is intentionally deferred.
- PostgreSQL ticket and conversation storage; Redis short-lived session history.
- Dashboard showing counts, breakdowns, and recent tickets.
- Telegram adapter for text, voice notes via local Whisper, status, reset, and forget.

## Run locally

1. Set `GROQ_API_KEY` and `TELEGRAM_BOT_TOKEN` in `.env`.
2. In a normal macOS Terminal, activate `.venv` and run `python groq_relay.py`.
3. In another Terminal, run `uvicorn app.main:app --reload`. Local development uses SQLite in `service_requests.db`, so PostgreSQL and Redis are not required to open the API and dashboard. Docker Compose overrides this with PostgreSQL.
4. Open <http://localhost:8000/dashboard> or <http://localhost:8000/docs>.

The graph is `transcribe → extract → check required fields → ask follow-up or request confirmation → classify → save`. Missing-field answers resume from the draft stored by chat ID. The response language is retained with that draft, so Arabic conversations receive Arabic prompts and confirmation summaries. Classification is reachable only after user confirmation; saving is reachable only after classification succeeds. The classifier is intentionally empty, so confirmed requests currently stop with a clear placeholder response and are not saved.

The app sends extraction requests to a loopback-only relay (`127.0.0.1:8765`) started from your regular Terminal. The relay makes outbound Groq requests from that process, keeping the key on your machine and out of app logs. Drafts are stored in Redis when available, keyed by chat ID with a 30-minute expiry; local development falls back to process memory if Redis is unavailable. Voice audio is transcribed by Whisper in the Telegram adapter before entering the graph. The API and dashboard can run without Telegram credentials. Don't enter real resident data in this prototype. This is a foundation, not the complete 10-week deliverable; data sourcing, evaluation, polished analytics, deployment and demo remain follow-on work.
