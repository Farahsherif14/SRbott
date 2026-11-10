"""Smoke test: every app module can be imported (Step 1, K1 + K2).

An import error is the first thing that breaks when files move or two names clash.
This test finds it in seconds, without Telegram, a database, Redis or an LLM.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest

# The fake Telegram token comes from tests/conftest.py (aiogram checks it on import).
MODULES = [
    "app.config",
    "app.schemas",
    "app.extraction.schema",
    "app.session",
    "app.ai",
    "app.db",
    "app.classifier",
    "app.flow",
    "app.main",
]
BOT_FILE = Path(__file__).resolve().parents[1] / "app" / "bot.py"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_bot_file_loads() -> None:
    # "import app.bot" gives the folder app/bot/, never the file app/bot.py: load it by path.
    # This runs its top part (imports, Bot(...), handlers). Its __name__ is not "__main__",
    # so the guard at the bottom stays false and no bot starts.
    spec = importlib.util.spec_from_file_location("bot_file_under_test", BOT_FILE)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_python_m_app_bot_finds_something_to_run() -> None:
    # "python -m app.bot" (run_bot.py) runs the module app/bot.py, or app/bot/__main__.py if
    # app/bot/ is a package (has __init__.py). find_spec only looks; it runs nothing, no bot starts.
    spec = importlib.util.find_spec("app.bot")
    assert spec is not None
    if spec.submodule_search_locations is not None:  # a package: it needs a __main__.py
        assert importlib.util.find_spec("app.bot.__main__") is not None
