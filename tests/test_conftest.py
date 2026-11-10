"""Tests for the test setup itself (tests/conftest.py): the "_test" guard and the rollback.

The guard tests need no database, so they run in CI too. The rollback test uses the test
database (skipped without a PostgreSQL DATABASE_URL).
"""

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Compound
from tests.conftest import RunDb, add_test_unit, make_alembic_config, require_test_database


@pytest.mark.parametrize("name", ["srbott", "srbott_testing", "test_srbott", "", None])
def test_guard_refuses_a_database_not_ending_in_test(name: str | None) -> None:
    url = make_url("postgresql+asyncpg://user:fake-password@127.0.0.1:5432").set(database=name)

    with pytest.raises(RuntimeError, match="must end in '_test'"):
        require_test_database(url)


def test_guard_accepts_a_test_database() -> None:
    url = make_url("postgresql+asyncpg://user:fake-password@127.0.0.1:5432/srbott_test")

    assert require_test_database(url) is url


def test_alembic_is_never_pointed_at_a_dev_database() -> None:
    url = make_url("postgresql+asyncpg://user:fake-password@127.0.0.1:5432/srbott")

    with pytest.raises(RuntimeError, match="must end in '_test'"):
        make_alembic_config(url)


def test_rows_are_rolled_back_after_each_scenario(db: RunDb) -> None:
    async def count_compounds(session: AsyncSession) -> int:
        return await session.scalar(select(func.count()).select_from(Compound)) or 0

    async def write(session: AsyncSession) -> None:
        await add_test_unit(session)
        await session.commit()  # even a commit only ends a savepoint inside the outer transaction
        assert await count_compounds(session) == 1

    async def read_again(session: AsyncSession) -> None:
        assert await count_compounds(session) == 0  # the row from write() is gone

    db(write)
    db(read_again)
