# Service Request Bot

Initial bilingual (Arabic/English) service request prototype. The attached brief provides project requirements; its Azure references are replaced with Groq per the owner's instruction.

## First iteration

- FastAPI intake API for text requests and transcribed voice.
- Groq structured extraction with missing-field follow-ups.
- LLM proofreading of voice transcripts before structured field extraction.
- LangGraph orchestration for transcription handoff, extraction, missing-field loops, confirmation, classification, and saving.
- Empty classifier placeholder for category and urgency; classification is intentionally deferred.
- PostgreSQL ticket and conversation storage; Redis short-lived session history.
- Dashboard showing counts, breakdowns, and recent tickets.
- Telegram adapter for text, voice notes via local Vosk, status, reset, and forget.

## Run locally

1. Set `GROQ_API_KEY` and `TELEGRAM_BOT_TOKEN` in `.env`. Download and extract the Arabic Vosk model into `models/vosk-model-ar-mgb2-0.4`, or set `VOSK_MODEL_PATH` to its extracted folder.
2. In Terminal window 1, run `cd "/path/to/SRbott"`, then `source .venv/bin/activate`, then `python groq_relay.py`. Keep this window open.
3. In Terminal window 2, run the same `cd` and `source` commands, then run `python -m app.bot`. Keep this window open. Startup prints the build ID and exact source path; `/version` in Telegram reports the same values. The bot creates its local database tables on startup and prevents a second copy from starting.
4. If you also want the dashboard/API, use Terminal window 3 to run the same `cd` and `source` commands, then `uvicorn app.main:app --reload`. Open <http://localhost:8000/dashboard> or <http://localhost:8000/docs>. Local development uses SQLite in `service_requests.db`; Docker Compose overrides this with PostgreSQL.

Stop a running process with `Ctrl+C` in its own Terminal window. Start exactly one bot process; a second start is blocked by the bot's local lock.

The graph is `transcribe → extract → check required fields → ask follow-up or request confirmation → classify → save`. Missing-field answers resume from the draft stored by chat ID. The response language is retained with that draft, so Arabic conversations receive Arabic prompts and confirmation summaries. Classification is reachable only after user confirmation; saving is reachable only after classification succeeds. The classifier is intentionally empty, so confirmed requests currently stop with a clear placeholder response and are not saved.

The app sends text-processing requests to a loopback-only relay (`127.0.0.1:8765`) started from your regular Terminal. The relay makes outbound Groq requests from that process, keeping the key on your machine and out of app logs. Voice notes are transcribed locally by Vosk; audio is not sent to an STT provider and does not require Google Cloud or billing. The configured Arabic model is `vosk-model-ar-mgb2-0.4` (about 318 MB); it is a general Arabic model, not one specifically trained for Egyptian dialect, so accuracy may be lower than Whisper on colloquial Egyptian speech. Install project dependencies and place the extracted model at the configured path before starting the bot. Before extracting fields from a voice transcript, the LLM makes a separate proofreading pass to correct clear spelling and sound-alike errors while preserving names, codes, dialect, and stated facts. The corrected text is then passed to structured extraction and shown when clarification is needed. Drafts are stored in Redis when available, keyed by chat ID with a 30-minute expiry; local development falls back to process memory if Redis is unavailable. Extraction preserves plausible names and technical terms and asks for clarification when a key word or location is ambiguous. Compound names are matched to MARASSI, SOUL, UPTOWN CAIRO, MIVIDA, BELLEVIE, and CAIRO GATE and displayed in uppercase. Building and unit identifiers must be digits optionally followed by one Latin letter; spoken English letters and numbers in Arabic messages are normalized when clear, and ambiguous values trigger clarification. The API and dashboard can run without Telegram credentials. Don't enter real resident data in this prototype. This is a foundation, not the complete 10-week deliverable; data sourcing, evaluation, polished analytics, deployment and demo remain follow-on work.
