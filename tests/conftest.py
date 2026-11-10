"""Shared test setup. pytest loads this file before it imports any test file.

aiogram checks the token's shape when app/bot.py is imported (Bot(...) at the top),
and app.config reads the settings only once, at its first import. So the fake token must be in
the environment before any test file imports app code; here it always is.
An environment variable beats .env, so the tests never use a real token (CI has no .env at all).
"""

import os

FAKE_TOKEN = "123456789:FAKE-TOKEN-FOR-TESTS"
os.environ["TELEGRAM_BOT_TOKEN"] = FAKE_TOKEN
