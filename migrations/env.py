"""Alembic's entry point: it runs on every `uv run alembic ...` command.

It connects to the database and runs the migration scripts in migrations/versions/.
- The URL: DATABASE_URL from .env through app/db/session.py, never from alembic.ini.
  Tests pass the test database's URL in config.attributes["database_url"].
- target_metadata = our new tables (app/db/models.py), not the old SQLite models in
  app/db/__init__.py. `alembic revision --autogenerate` compares them with the database and
  writes the difference as a new migration. Always read it: it misses some changes (CHECK
  rules, views).
- Logging: our setup_logging() with secrets hidden, instead of alembic.ini's logging.
"""

import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from app.config import get_settings
from app.db.models import Base
from app.db.session import database_url, make_engine
from app.logging_setup import setup_logging

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Offline mode (`alembic upgrade head --sql`): print the SQL instead of running it.

    Needs no database and no URL, only the dialect: which SQL language to write.
    """
    setup_logging(secrets=[])
    context.configure(
        dialect_name="postgresql",
        target_metadata=target_metadata,
        literal_binds=True,  # write the values into the SQL text: nothing is sent to a database
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run the migrations on one connection, all in one transaction.

    Postgres can undo CREATE TABLE too (transactional DDL), so a migration that fails half-way
    leaves nothing half-done: the database stays at the version before.
    """
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    url = config.attributes.get("database_url") or database_url()
    # Every secret from .env, and the password of the URL in use (the test database's in tests)
    setup_logging(secrets=[*get_settings().log_secrets(), url.password or ""])
    engine = make_engine(url, pooled=False)  # a short program: no pool needed
    async with engine.connect() as connection:
        # Alembic's own code is not async; run_sync runs it with this async connection
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    """Online mode (the normal one): connect to the database and run the migrations."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
