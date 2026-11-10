# Service Request Bot

Bilingual (Arabic and English) Telegram service request bot with a local FastAPI API and Streamlit dashboard. Confirmed service requests are stored in SQLite with ticket numbers.

## Project layout

- `app/` — API, bot, conversation flow, extraction, database, and classifier integration.
- `dashboard/app.py` — local Streamlit analytics dashboard.
- `ml/` — model training, evaluation, data preparation, and `reports/` outputs.
- `models/` — locally used trained classifier files.
- `docs/` — project brief, data card, contracts, and run logs.
- `service_requests.db` — local SQLite database.
- `tawasul_egy_stt.nemo` — local speech recognition checkpoint.

## Install and run locally (no Docker)

Activate the existing virtual environment and install project requirements if needed:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell), activate with `.venv\Scripts\activate` instead; `run_bot.py` then uses `.venv\Scripts\python.exe`.

Set `GROQ_API_KEY` and `TELEGRAM_BOT_TOKEN` in `.env`. Keep this file private. The `.env.example` file lists the expected settings. Place the Tawasul checkpoint at `tawasul_egy_stt.nemo`, or set `TAWASUL_MODEL_PATH`.

Start the Telegram bot and its local Groq relay:

```bash
python run_bot.py
```

Start the dashboard in a second Terminal window from the project folder:

```bash
source .venv/bin/activate
python -m streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501). The dashboard reads `service_requests.db` directly and refreshes its cached data every 30 seconds; use Streamlit's rerun control for an immediate refresh.

To run the API instead, start another terminal and run:

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --reload
```

Then open [http://localhost:8000/dashboard](http://localhost:8000/dashboard) or [http://localhost:8000/docs](http://localhost:8000/docs). The API and dashboard can run without Telegram credentials.

## Request handling

The request flow is `transcribe → extract → clarify missing fields → request confirmation → classify → save`. Classification and saving happen only after the requester confirms. The bot returns the generated ticket number as confirmation.

Text processing uses the local loopback Groq relay (`127.0.0.1:8765`); the relay reads the Groq key from `.env` and sends outbound requests. Voice notes are transcribed locally with Tawasul/NVIDIA NeMo. Ollama provides the `qwen3-embedding:4b` embedding model when needed. Drafts use Redis when available and fall back to process memory.

The local prototype contains resident request data. Keep `.env`, `service_requests.db`, downloaded models, and speech checkpoints private; do not commit them or enter real resident data into shared environments. See `docs/DATA_CARD.md` and `docs/contracts.md` for project details.
