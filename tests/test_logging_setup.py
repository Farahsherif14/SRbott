"""Tests for app/logging_setup.py: secrets and personal data never reach a log line.

Fake values only. The tests that call setup_logging() run in a new Python process, because
pytest adds its own handlers to the root logger, and then logging.basicConfig() does nothing.
"""

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.logging_setup import LOG_FORMAT, RedactingFormatter

ROOT = Path(__file__).resolve().parents[1]
FAKE_TOKEN = "123456789:FAKE-token-for-tests-only"
FAKE_NAME = "Fake Resident"
FAKE_PHONE = "0100000000"
# What asyncpg's error text looks like when a CHECK rule refuses a row
REFUSED_ROW = (
    'new row for relation "users" violates check constraint "ck_users_phone_e164"\n'
    f"DETAIL:  Failing row contains (7, 123, {FAKE_NAME}, {FAKE_PHONE}, 2026-10-31 12:00:00+00)."
)


def make_record(msg: str, *args: object, exc_info=None) -> logging.LogRecord:
    """A log record like the one logger.error(msg, *args) would create."""
    return logging.LogRecord("test", logging.ERROR, __file__, 1, msg, args, exc_info)


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run code in a new Python process from the project root, with this test's environment."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


# --- The formatter (ported from sr2-bot) ---


def test_token_hidden_in_message() -> None:
    formatter = RedactingFormatter(LOG_FORMAT, secrets=[FAKE_TOKEN])
    record = make_record("url=https://api.telegram.org/bot%s/getUpdates", FAKE_TOKEN)

    line = formatter.format(record)

    assert FAKE_TOKEN not in line
    assert "bot***/getUpdates" in line


def test_token_hidden_in_traceback() -> None:
    formatter = RedactingFormatter(LOG_FORMAT, secrets=[FAKE_TOKEN])
    try:
        raise ValueError(f"bad url https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage")
    except ValueError:
        record = make_record("request failed", exc_info=sys.exc_info())

    line = formatter.format(record)

    assert "Traceback" in line
    assert FAKE_TOKEN not in line


def test_postgres_detail_hidden_in_message() -> None:
    formatter = RedactingFormatter(LOG_FORMAT, secrets=[])

    line = formatter.format(make_record("saving the user failed: %s", REFUSED_ROW))

    assert FAKE_NAME not in line
    assert FAKE_PHONE not in line
    assert "DETAIL: [hidden: may contain personal data]" in line
    assert "ck_users_phone_e164" in line  # the rule's name stays: enough to debug


def test_postgres_detail_hidden_in_chained_traceback() -> None:
    """The DETAIL hides in the CAUSE of the error; a traceback prints every link of the chain."""

    class FakeDriverError(Exception):
        pass

    formatter = RedactingFormatter(LOG_FORMAT, secrets=[])
    try:
        try:
            raise FakeDriverError(REFUSED_ROW)
        except FakeDriverError as cause:
            raise RuntimeError("insert failed") from cause
    except RuntimeError:
        record = make_record("saving the user failed", exc_info=sys.exc_info())

    line = formatter.format(record)

    assert "direct cause" in line  # the chain is printed ...
    assert FAKE_NAME not in line  # ... but not the row
    assert FAKE_PHONE not in line


@pytest.mark.parametrize(
    "error_text",
    [
        # asyncpg's text for a value of the wrong type: one row, and many rows at once
        f"invalid input for query argument $3: '{FAKE_PHONE}' (expected str, got int)",
        (  # one string in two parts (parentheses: not a missing comma)
            f"invalid input for query argument $2 in element #0 of executemany() sequence: "
            f"'{FAKE_NAME}' ('str' object cannot be interpreted as an integer)"
        ),
    ],
)
def test_wrong_type_value_hidden(error_text: str) -> None:
    formatter = RedactingFormatter(LOG_FORMAT, secrets=[])

    line = formatter.format(make_record("saving failed: %s", error_text))

    assert FAKE_NAME not in line
    assert FAKE_PHONE not in line
    assert "query argument $" in line  # which argument it was stays: enough to debug
    assert "[hidden: may contain personal data]" in line


def test_empty_secret_is_ignored() -> None:
    formatter = RedactingFormatter(LOG_FORMAT, secrets=[""])

    line = formatter.format(make_record("normal line"))

    assert line.endswith("normal line")
    assert "***" not in line


# --- setup_logging() in a clean process, like the bot and the API ---


def test_setup_logging_hides_secret_and_row_in_the_output() -> None:
    result = run_python(f"""
        import logging
        from app.logging_setup import setup_logging

        setup_logging(secrets=[{FAKE_TOKEN!r}])
        logging.getLogger("app.demo").error("sending failed: bot%s\\n%s", {FAKE_TOKEN!r}, {REFUSED_ROW!r})
    """)

    assert result.returncode == 0, result.stderr
    assert "ERROR app.demo: sending failed: bot***" in result.stderr
    assert "DETAIL: [hidden: may contain personal data]" in result.stderr
    for private in (FAKE_TOKEN, FAKE_NAME, FAKE_PHONE):
        assert private not in result.stderr


def test_setup_logging_covers_uvicorn_crash_tracebacks() -> None:
    """uvicorn logs API crashes through its own loggers (propagate=False), not the root one."""
    result = run_python(f"""
        import logging
        import logging.config
        from uvicorn.config import LOGGING_CONFIG
        from app.logging_setup import setup_logging

        logging.config.dictConfig(LOGGING_CONFIG)  # what uvicorn does when it starts
        setup_logging(secrets=[{FAKE_TOKEN!r}])  # what the API does at startup
        try:
            raise ValueError("token " + {FAKE_TOKEN!r} + "\\n" + {REFUSED_ROW!r})
        except ValueError:
            logging.getLogger("uvicorn.error").exception("Exception in ASGI application")
        logging.getLogger("uvicorn.access").info("%s - %s %s", "127.0.0.1:5000", "GET /boom", 500)
    """)

    assert result.returncode == 0, result.stderr
    assert "ERROR uvicorn.error: Exception in ASGI application" in result.stderr  # our format
    assert "ValueError: token ***" in result.stderr
    assert "INFO uvicorn.access: 127.0.0.1:5000 - GET /boom 500" in result.stderr
    for private in (FAKE_TOKEN, FAKE_NAME, FAKE_PHONE):
        assert private not in result.stderr


def test_entry_points_leave_the_root_logger_unconfigured() -> None:
    """If an import added a root handler, basicConfig() in setup_logging() would do nothing,
    and the filter would be silently off in the bot and the API."""
    result = run_python("""
        import logging
        import app.bot
        import app.main

        print(len(logging.getLogger().handlers))
    """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"
