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
- Telegram adapter for text, voice notes via local Tawasul ASR, status, reset, and forget.

## Run locally

1. Set `GROQ_API_KEY` and `TELEGRAM_BOT_TOKEN` in `.env`. Place the Tawasul checkpoint at `tawasul_egy_stt.nemo`, or set `TAWASUL_MODEL_PATH` to its location. The checkpoint is available from [TawasulAI/tawasul-egy-stt](https://huggingface.co/TawasulAI/tawasul-egy-stt).
2. Start the relay and Telegram bot together with one command: `python run_bot.py`. It uses the project's `.venv`, starts both processes, and stops both when you press `Ctrl+C`. Startup prints the bot build ID and exact source path; `/version` in Telegram reports the same values. The bot creates its local database tables on startup and prevents a second copy from starting.
3. If you also want the dashboard/API, open another Terminal window, run the same `cd` and `source` commands, then `uvicorn app.main:app --reload`. Open <http://localhost:8000/dashboard> or <http://localhost:8000/docs>. Local development uses SQLite in `service_requests.db`; Docker Compose overrides this with PostgreSQL.

Stop the combined relay and bot with `Ctrl+C` in the window running `python run_bot.py`. Start exactly one bot process; a second start is blocked by the bot's local lock.

The graph is `transcribe → extract → check required fields → ask follow-up or request confirmation → classify → save`. Missing-field answers resume from the draft stored by chat ID. The response language is retained with that draft, so Arabic conversations receive Arabic prompts and confirmation summaries. Classification is reachable only after user confirmation; saving is reachable only after classification succeeds. The classifier is intentionally empty, so confirmed requests currently stop with a clear placeholder response and are not saved.

The app sends text-processing requests to a loopback-only relay (`127.0.0.1:8765`) started from your regular Terminal. The relay makes outbound Groq requests from that process, keeping the key on your machine and out of app logs. Voice notes are transcribed locally by Tawasul using NVIDIA NeMo; audio is not sent to an STT provider. Inputs are decoded to mono 16 kHz WAV, as required by the model. Tawasul supports Arabic dialects with a focus on Egyptian Arabic, is non-streaming, and returns unvowelled Arabic text. Install project dependencies and place the checkpoint at the configured path before starting the bot. Before extracting fields from a voice transcript, the LLM makes a separate proofreading pass to correct clear spelling and sound-alike errors while preserving names, codes, dialect, and stated facts. The corrected text is then passed to structured extraction and shown when clarification is needed. Drafts are stored in Redis when available, keyed by chat ID with a 30-minute expiry; local development falls back to process memory if Redis is unavailable. Extraction preserves plausible names and technical terms and asks for clarification when a key word or location is ambiguous. Compound names are matched to MARASSI, SOUL, UPTOWN CAIRO, MIVIDA, BELLEVIE, and CAIRO GATE and displayed in uppercase. Building and unit identifiers must be digits optionally followed by one Latin letter; spoken English letters and numbers in Arabic messages are normalized when clear, and ambiguous values trigger clarification. The API and dashboard can run without Telegram credentials. Don't enter real resident data in this prototype. This is a foundation, not the complete 10-week deliverable; data sourcing, evaluation, polished analytics, deployment and demo remain follow-on work.
