"""The connection to PostgreSQL for the new database code: the URL, the async engine, sessions.

- Engine: a pool of open connections to Postgres. Opening a connection is slow, so the engine
  keeps a few open and lends them out. Make ONE engine per program, at startup, and share it.
- Session: one unit of work. Add or change objects, then `await session.commit()` saves them
  all in one transaction, or `await session.rollback()` undoes them all.

Used by Alembic (migrations/env.py), the tests, and later the seed script and the app.
The URL contains the password: it stays a SecretStr in app/config.py until here, and a
SQLAlchemy URL object prints the password as *** (str() and repr() both hide it).
"""

from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import get_settings

DRIVER = "postgresql+asyncpg"  # the dialect (postgresql) + the driver (asyncpg)


def database_url(url: str | None = None) -> URL:
    """DATABASE_URL with the asyncpg driver.

    .env keeps the plain URL (postgresql://...), which other tools understand too; this adds
    "+asyncpg". A URL that already has "+asyncpg" is kept as it is. Any other driver is an
    error: our code is written for asyncpg.

    url: a URL to use instead of DATABASE_URL (for tests). None = read DATABASE_URL from .env.
    """
    if url is None:
        settings = get_settings()
        settings.require("database_url")  # fail fast with the name of the missing .env line
        # require() checked it already; the assert tells the type checker it isn't None
        assert settings.database_url is not None
        url = settings.database_url.get_secret_value()

    parsed = make_url(url)
    if parsed.drivername == "postgresql":
        return parsed.set(drivername=DRIVER)
    if parsed.drivername == DRIVER:
        return parsed
    # Only the scheme in the message, never the URL: it contains the password
    message = (
        f"DATABASE_URL must start with postgresql:// or {DRIVER}://, not {parsed.drivername}://"
    )
    if parsed.drivername.startswith("sqlite"):
        # An .env from before the PostgreSQL code still has the old SQLite URL here
        message += ". The SQLite URL of the current bot goes in LEGACY_DATABASE_URL (.env.example)"
    raise ValueError(message)


def make_engine(url: URL | None = None, *, pooled: bool = True) -> AsyncEngine:
    """The async engine. It doesn't connect yet: the first query opens the first connection.

    url: None = database_url() from .env.
    pooled=False: no pool, every session opens its own connection and closes it at the end.
        For short programs and tests: Alembic, the seed script, and tests (where each test runs
        its own event loop, and an asyncpg connection works only in the loop that opened it).
    hide_parameters=True: when a query fails, SQLAlchemy's error message and log show only the
        SQL with placeholders ($1, $2), never the values we sent (a name, a phone).
    """
    return create_async_engine(
        url if url is not None else database_url(),
        hide_parameters=True,
        poolclass=None if pooled else NullPool,  # None = SQLAlchemy's normal pool
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """A factory for sessions. Use: `async with factory() as session: ...`.

    expire_on_commit=False: after a commit, objects keep their values in memory. With the
    default (True), SQLAlchemy forgets them, and reading `sr.ticket_number` after the commit
    would start a hidden query. Async code can't run hidden queries, so that fails
    (MissingGreenlet).
    """
    return async_sessionmaker(engine, expire_on_commit=False)
