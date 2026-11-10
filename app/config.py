"""App settings, read from environment variables and the .env file.

Teammates import the ready object: `from app.config import settings`.
New code can call `get_settings()`; both give the same object.
"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (the folder that holds pyproject.toml), so .env is found from any working folder
ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Every setting has a default or may be missing; entry points call `require()`."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "KEY=" with no value means "not set", so the default is used
        extra="ignore",  # other variables in .env are not an error
    )

    # Secrets: SecretStr shows "**********" when printed or logged.
    # Read the real value only where it is used: settings.telegram_bot_token.get_secret_value()
    groq_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    # URLs can hold a password (e.g. postgresql://user:password@host/db), so they are secrets too.
    # The SQLite path starts at the current working folder (the project root when run as usual).
    database_url: SecretStr = SecretStr("sqlite:///./service_requests.db")
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    # Not secret
    groq_model: str = "openai/gpt-oss-20b"
    groq_relay_url: str = "http://127.0.0.1:8765/extract"
    tawasul_model_path: str = "tawasul_egy_stt.nemo"
    ollama_host: str = "http://localhost:11434"
    classifier_path: str = "models/category_clf.joblib"

    # Session memory. PositiveInt = a whole number above 0, so "0", "-5" or "abc" in .env
    # stops the app at startup with a clear error.
    # Seconds without a new message before a chat's session is deleted (1800 = 30 minutes).
    session_ttl_seconds: PositiveInt = 1800
    # How many recent messages (user or bot) the history keeps for the LLM.
    history_max_messages: PositiveInt = 10

    def require(self, *names: str) -> None:
        """Fail fast: stop if any of the named settings is not set.

        Example: settings.require("telegram_bot_token")
        """
        missing = [name.upper() for name in names if getattr(self, name) is None]
        if missing:
            raise RuntimeError(
                f"Missing setting(s) in .env: {', '.join(missing)}. See .env.example."
            )

    def log_secrets(self) -> list[str]:
        """The values the logs must hide: every SecretStr setting that is set, and the password
        inside each URL (a driver error can show the password alone).

        Example: setup_logging(secrets=settings.log_secrets())
        """
        values: set[str] = set()
        for name in type(self).model_fields:
            value = getattr(self, name)
            if not isinstance(value, SecretStr):
                continue
            secret = value.get_secret_value()
            values.add(secret)
            if "://" in secret:
                values.add(urlsplit(secret).password or "")
        values.discard("")
        # Longest first: if a short secret were hidden first, it could cut a longer one in two,
        # and the rest of the longer one would no longer match.
        return sorted(values, key=len, reverse=True)


@lru_cache
def get_settings() -> Settings:
    """Create the settings once, then return the same object on every call."""
    return Settings()


# The object teammates' code imports. It is the same object get_settings() returns.
settings = get_settings()
