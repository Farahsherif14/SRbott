import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sr:sr@localhost:5432/srbot")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")

SESSION_TTL_SECONDS = 30 * 60   # session cleared after 30 minutes idle
MAX_TURNS = 8                   # last N turns kept verbatim; older ones summarized
REVIEW_THRESHOLD = 0.6          # below this confidence -> flagged for review