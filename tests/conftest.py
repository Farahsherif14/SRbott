"""Shared test setup. pytest loads this file before it imports any test file.

1. The fake bot token. aiogram checks the token's shape when app/bot.py is imported (Bot(...) at
   the top), and app.config reads the settings only once, at its first import. So the fake token
   must be in the environment before any app code is imported, including the imports below.
   An environment variable beats .env, so the tests never use a real token (CI has no .env).

2. The PostgreSQL test database. A test that doesn't ask for `db`, `test_db_url` or
   `alembic_config` never touches a database.
   - Which database: DATABASE_URL from .env with "_test" added to the name (srbott -> srbott_test).
     Your dev database is never used: every helper refuses a name that doesn't end in "_test".
   - No DATABASE_URL (CI), or an old SQLite one: the database tests are skipped.
   - A PostgreSQL DATABASE_URL, but Postgres is off: they fail and say what to start. A silent
     skip would hide a stopped Docker.
   - Once per test run: create srbott_test if it is missing, then `alembic upgrade head` on it.
     So the tests also prove that the migrations build the tables the code expects.
   - Async without pytest-asyncio: each test runs its scenario with asyncio.run().
"""

import os

FAKE_TOKEN = "123456789:FAKE-TOKEN-FOR-TESTS"
os.environ["TELEGRAM_BOT_TOKEN"] = FAKE_TOKEN

# The imports below come after the fake token on purpose (see point 1). Ruff allows imports
# after an os.environ line, so no rule needs to be switched off.
import asyncio
from collections.abc import Awaitable, Callable

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import ROOT_DIR, get_settings
from app.db.models import Compound, Unit, User
from app.db.session import database_url, make_engine

TEST_SUFFIX = "_test"
# Fake values for the test rows (all fictional)
FAKE_NAME = "Test Resident"
FAKE_PHONE = "+0000000000"  # E.164 shape, but no country has the code 0
START_POSTGRES = "Start Rancher Desktop, then: docker compose up -d --wait postgres"

# A test scenario: an async function that gets a session. RunDb runs one (the `db` fixture).
# Test files import these two names for their type hints.
Scenario = Callable[[AsyncSession], Awaitable[None]]
RunDb = Callable[[Scenario], None]


def require_test_database(url: URL) -> URL:
    """The safety guard: stop unless the database name ends in "_test"."""
    if not (url.database or "").endswith(TEST_SUFFIX):
        raise RuntimeError(
            f"Refusing to use the database {url.database!r} in tests: "
            f"its name must end in {TEST_SUFFIX!r}, so dev data can never be wiped"
        )
    return url


def make_alembic_config(url: URL) -> Config:
    """Alembic's settings from alembic.ini, pointed at the test database (env.py reads this)."""
    config = Config(ROOT_DIR / "alembic.ini")
    config.attributes["database_url"] = require_test_database(url)
    return config


async def create_database_if_missing(url: URL) -> None:
    """Create the test database once. Connects to the built-in "postgres" database for that.

    AUTOCOMMIT: CREATE DATABASE can't run inside a transaction.
    """
    name = require_test_database(url).database
    engine = make_engine(url.set(database="postgres"), pooled=False)
    try:
        async with engine.execution_options(isolation_level="AUTOCOMMIT").connect() as connection:
            exists = await connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
            )
            if not exists:
                # A name can't be a query parameter. It is our own .env value + "_test"
                await connection.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        await engine.dispose()


async def run_in_rollback(url: URL, scenario: Scenario) -> None:
    """Run one scenario in a transaction that is rolled back at the end: nothing is kept.

    join_transaction_mode="create_savepoint": the session runs inside our outer transaction.
    Its commit() only ends a savepoint, so the outer rollback still removes everything, and
    `async with session.begin_nested():` makes a SAVEPOINT for an insert that must fail.
    """
    engine = make_engine(require_test_database(url), pooled=False)
    try:
        async with engine.connect() as connection:
            outer = await connection.begin()
            session = AsyncSession(
                bind=connection,
                join_transaction_mode="create_savepoint",
                expire_on_commit=False,
            )
            try:
                await scenario(session)
            finally:
                await session.close()
                await outer.rollback()
    finally:
        await engine.dispose()


async def add_test_unit(session: AsyncSession) -> Unit:
    """A fake compound with one unit, B-204. The test database has no registry loaded."""
    compound = Compound(name_en="Test Compound", name_ar="Ù…Ø¬Ù…Ø¹ ØªØ¬Ø±ÙŠØ¨ÙŠ")
    session.add(compound)
    await session.flush()  # now compound.id is known
    unit = Unit(compound_id=compound.id, building="B", unit_number="B-204")
    session.add(unit)
    await session.flush()
    return unit


async def add_test_user(session: AsyncSession, telegram_user_id: int = 1) -> User:
    """A fake resident."""
    user = User(telegram_user_id=telegram_user_id, name=FAKE_NAME, phone=FAKE_PHONE)
    session.add(user)
    await session.flush()
    return user


@pytest.fixture(scope="session")
def test_db_url() -> URL:
    """The test database's URL. Created and migrated once per test run (scope="session")."""
    if get_settings().database_url is None:
        pytest.skip("DATABASE_URL is not set: database tests skipped")
    try:
        dev_url = database_url()
    except ValueError:
        # An .env from before PostgreSQL (DATABASE_URL=sqlite:...). No URL in the message
        pytest.skip("DATABASE_URL is not a PostgreSQL URL: database tests skipped")
    url = require_test_database(dev_url.set(database=f"{dev_url.database}{TEST_SUFFIX}"))
    try:
        asyncio.run(create_database_if_missing(url))
    except (OSError, SQLAlchemyError) as error:
        # Only the error type, never the URL: it contains the password
        pytest.fail(
            f"Can't reach Postgres for the database tests ({type(error).__name__}). "
            f"{START_POSTGRES}",
            pytrace=False,
        )
    command.upgrade(make_alembic_config(url), "head")
    return url


@pytest.fixture
def alembic_config(test_db_url: URL) -> Config:
    """Alembic's settings for the test database, for the migration tests."""
    return make_alembic_config(test_db_url)


@pytest.fixture
def db(test_db_url: URL) -> RunDb:
    """Run a test scenario with a session; everything it writes is rolled back.

    Use:
        def test_something(db: RunDb) -> None:
            async def scenario(session: AsyncSession) -> None:
                ...
            db(scenario)
    """

    def run(scenario: Scenario) -> None:
        asyncio.run(run_in_rollback(test_db_url, scenario))

    return run
