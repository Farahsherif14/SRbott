"""Tests for app/db/session.py. No database needed: they only check URLs and engine settings.

They use fake URLs and never read the real .env, so they run in CI too.
"""

import pytest
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import session
from app.db.session import DRIVER, database_url, make_engine

FAKE_PASSWORD = "fake-db-password"
PLAIN_URL = f"postgresql://srbott:{FAKE_PASSWORD}@127.0.0.1:5432/srbott"


def test_plain_url_gets_the_asyncpg_driver() -> None:
    url = database_url(PLAIN_URL)

    assert url.drivername == DRIVER
    assert (url.host, url.port, url.database) == ("127.0.0.1", 5432, "srbott")


def test_url_with_asyncpg_is_kept_not_doubled() -> None:
    url = database_url(f"postgresql+asyncpg://srbott:{FAKE_PASSWORD}@127.0.0.1:5432/srbott")

    assert url.drivername == "postgresql+asyncpg"
    assert url.render_as_string(hide_password=False).startswith("postgresql+asyncpg://srbott:")


@pytest.mark.parametrize(
    "scheme", ["postgresql+psycopg", "postgresql+psycopg2", "postgres", "sqlite", "mysql"]
)
def test_other_drivers_are_refused_without_showing_the_password(scheme: str) -> None:
    with pytest.raises(ValueError, match="must start with postgresql://") as error:
        database_url(f"{scheme}://srbott:{FAKE_PASSWORD}@127.0.0.1:5432/srbott")

    assert FAKE_PASSWORD not in str(error.value)


def test_an_old_sqlite_url_says_where_it_moved() -> None:
    # An .env from before the PostgreSQL code still has DATABASE_URL=sqlite:...
    with pytest.raises(ValueError, match="LEGACY_DATABASE_URL"):
        database_url("sqlite:///./service_requests.db")


def test_printing_the_url_hides_the_password() -> None:
    url = database_url(PLAIN_URL)

    assert FAKE_PASSWORD not in str(url)
    assert FAKE_PASSWORD not in repr(url)
    assert "***" in str(url)


def test_reads_database_url_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", PLAIN_URL)
    monkeypatch.setattr(session, "get_settings", lambda: Settings(_env_file=None))

    assert database_url().drivername == DRIVER


def test_missing_database_url_names_the_env_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(session, "get_settings", lambda: Settings(_env_file=None))

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        database_url()


def test_engine_hides_query_values_and_uses_asyncpg() -> None:
    engine = make_engine(database_url(PLAIN_URL))  # no connection is opened here

    assert engine.url.drivername == DRIVER
    assert engine.sync_engine.hide_parameters is True
    assert not isinstance(engine.pool, NullPool)  # the normal pool


def test_unpooled_engine_has_no_pool() -> None:
    engine = make_engine(database_url(PLAIN_URL), pooled=False)

    assert isinstance(engine.pool, NullPool)
