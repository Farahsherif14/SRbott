"""Tests for app/config.py. They use fake values only and never read the real .env."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.config import ROOT_DIR, Settings, get_settings, settings

# Every setting and its default. Teammates' code and old .env files use these exact names,
# so a renamed setting or a changed default must fail here first.
DEFAULTS = {
    "groq_api_key": None,
    "telegram_bot_token": None,
    "database_url": None,  # PostgreSQL, set in .env (Step 4: the old SQLite default moved below)
    "legacy_database_url": "sqlite:///./service_requests.db",
    "redis_url": "redis://localhost:6379/0",
    "groq_model": "openai/gpt-oss-20b",
    "groq_relay_url": "http://127.0.0.1:8765/extract",
    "tawasul_model_path": "tawasul_egy_stt.nemo",
    "ollama_host": "http://localhost:11434",
    "classifier_path": "models/category_clf.joblib",
    "session_ttl_seconds": 1800,
    "history_max_messages": 10,
}

FAKE_SECRETS = {
    "TELEGRAM_BOT_TOKEN": "123456789:fake-telegram-token",
    "GROQ_API_KEY": "fake-groq-key",
    "DATABASE_URL": "postgresql://user:fake-db-password@localhost:5432/db",
    "REDIS_URL": "redis://:fake-redis-password@localhost:6379/0",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove our variables from the environment, so every test starts empty, like CI."""
    for name in DEFAULTS:
        monkeypatch.delenv(name.upper(), raising=False)


def plain(value: object) -> object:
    """The real value of a SecretStr, so it can be compared with a normal string."""
    return value.get_secret_value() if isinstance(value, SecretStr) else value


def test_every_setting_keeps_its_name_and_default() -> None:
    loaded = Settings(_env_file=None)  # _env_file=None: ignore the real .env

    assert set(Settings.model_fields) == set(DEFAULTS)
    for name, default in DEFAULTS.items():
        assert plain(getattr(loaded, name)) == default, name


def test_secrets_hidden_when_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in FAKE_SECRETS.items():
        monkeypatch.setenv(name, value)
    loaded = Settings(_env_file=None)

    printed = str(loaded) + repr(loaded)
    for value in FAKE_SECRETS.values():
        assert value not in printed
    assert loaded.groq_api_key is not None
    assert loaded.groq_api_key.get_secret_value() == "fake-groq-key"


def test_empty_value_means_not_set(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=\nSESSION_TTL_SECONDS=\nREDIS_URL=\n", encoding="utf-8")

    loaded = Settings(_env_file=env_file)

    assert loaded.telegram_bot_token is None
    assert loaded.session_ttl_seconds == 1800  # empty = the default, as .env.example says
    assert loaded.redis_url.get_secret_value() == DEFAULTS["redis_url"]


def test_env_file_is_found_from_the_project_root() -> None:
    # Not ".env" in the current folder: starting the app from another folder must still work
    assert Settings.model_config["env_file"] == ROOT_DIR / ".env"
    assert (ROOT_DIR / "pyproject.toml").is_file()


def test_require_names_the_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:fake-telegram-token")
    loaded = Settings(_env_file=None)

    loaded.require("telegram_bot_token")  # set, so no error

    with pytest.raises(RuntimeError, match="GROQ_API_KEY") as error:
        loaded.require("telegram_bot_token", "groq_api_key")
    assert "TELEGRAM_BOT_TOKEN" not in str(error.value)


def test_session_memory_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SESSION_TTL_SECONDS", "60")
    monkeypatch.setenv("HISTORY_MAX_MESSAGES", "4")

    loaded = Settings(_env_file=None)

    assert loaded.session_ttl_seconds == 60
    assert loaded.history_max_messages == 4


@pytest.mark.parametrize("name", ["SESSION_TTL_SECONDS", "HISTORY_MAX_MESSAGES"])
@pytest.mark.parametrize("value", ["0", "-5", "abc"])  # zero / negative / not a number
def test_session_memory_settings_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name.lower()):
        Settings(_env_file=None)


def test_log_secrets_has_every_secret_and_url_password(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in FAKE_SECRETS.items():
        monkeypatch.setenv(name, value)

    secrets = Settings(_env_file=None).log_secrets()

    assert set(FAKE_SECRETS.values()) <= set(secrets)
    assert "fake-db-password" in secrets  # the password alone, too
    assert "fake-redis-password" in secrets
    assert [len(secret) for secret in secrets] == sorted(map(len, secrets), reverse=True)


def test_log_secrets_skips_settings_that_are_not_set() -> None:
    secrets = Settings(_env_file=None).log_secrets()

    # Only the 2 default URLs: no token, no key, no empty string, no password
    assert sorted(secrets) == sorted([DEFAULTS["legacy_database_url"], DEFAULTS["redis_url"]])


def test_settings_object_is_the_cached_one() -> None:
    # Teammates import `settings`; new code calls get_settings(). Both must be the same object.
    assert settings is get_settings()
