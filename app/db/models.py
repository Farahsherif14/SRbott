"""The PostgreSQL tables of the new database code, as Python classes (SQLAlchemy ORM).

7 tables: compounds, units, users, user_units, service_requests, model_predictions and
conversation_messages. Names follow the team contract (docs/contracts.md, app/db_unused/schema.sql)
where the meaning is the same; the Step 4 pull request lists every difference from it.

These classes only describe the tables. Alembic compares them with the database and writes a
migration (migrations/versions/), and the migration creates or changes the tables. Never change
a table by hand in psql.

The rules the database itself enforces (constraints) are written here too, so wrong data can't
get in even if the Python code has a bug. Every constraint and index gets a fixed name
(NAMING_CONVENTION), so a later migration can find it, and an error says which rule failed.

Personal data: users.name, users.phone, service_requests.issue_description and
conversation_messages.content. They have a comment in the database too. Never log a row of these
tables.

This file is separate from the old models in app/db/__init__.py (SQLite), which the current bot
and API use until the flow moves to PostgreSQL.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Identity,
    MetaData,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeEngine

from app.labels import CATEGORIES, LANGUAGES, MESSAGE_ROLES, URGENCIES

# Allowed values of the status column. Labels, languages and message roles: app/labels.py
SR_STATUSES = ("open", "in_progress", "closed")

PERSONAL_DATA = "Personal data: never in logs, traces or git"

# How SQLAlchemy names constraints and indexes. It fills in the %(...)s parts, e.g.
# CheckConstraint(..., name="phone_e164") on the table users -> ck_users_phone_e164
NAMING_CONVENTION = {
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
}


class Base(DeclarativeBase):
    """The parent of every table class: the settings they share.

    Plain DeclarativeBase, not MappedAsDataclass: a dataclass would print every column in its
    repr(), so a logged object would show a name or a phone.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    # Python type -> Postgres type, for each Mapped[...] that doesn't name its own type.
    # ClassVar: a setting of the class, not a column (SQLAlchemy skips ClassVar attributes)
    type_annotation_map: ClassVar[dict[type, TypeEngine]] = {
        str: Text(),  # text: no length limit to guess (Postgres stores varchar(n) the same way)
        datetime: DateTime(timezone=True),  # timestamptz: saved as UTC
        # double precision: 8 bytes like Python's float, so a value comes back exactly as saved.
        # real (4 bytes) would give back 0.87 as 0.8700000047683716
        float: Double(),
    }


def one_of(column: str, values: Sequence[str]) -> str:
    """SQL for a CHECK rule: the column holds one of these values.

    Example: one_of("urgency", URGENCIES) -> "urgency IN ('low', 'medium', 'high')".
    Only for our own constant lists, never for user input (the values are pasted into SQL).
    """
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({quoted})"


def label_checks() -> list[CheckConstraint]:
    """The label rules of service_requests and model_predictions. A new set for each table.

    Labels are text + CHECK, not a Postgres ENUM. A label change needs a hand-written migration.
    """
    return [
        CheckConstraint(one_of("category", CATEGORIES), name="category"),
        CheckConstraint(one_of("urgency", URGENCIES), name="urgency"),
    ]


# Mapped[str] = NOT NULL, Mapped[str | None] = NULL allowed. Labels are Mapped[str], not
# Mapped[Category]: SQLAlchemy would turn a Literal type into a Postgres ENUM.
# Identity(always=True) = GENERATED ALWAYS AS IDENTITY: Postgres numbers the rows and refuses an
# id from us. Only service_requests uses BY DEFAULT (see there).


class Compound(Base):
    """The 5 fictional compounds of the unit registry (ml/make_mock_units.py)."""

    __tablename__ = "compounds"

    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    name_en: Mapped[str] = mapped_column(unique=True)  # "Palm Gardens"
    name_ar: Mapped[str] = mapped_column(unique=True)  # the same name in Arabic letters


class Unit(Base):
    """The 960 units of the mock registry. Read-only for the bot."""

    __tablename__ = "units"
    __table_args__ = (
        # A-101 exists in every compound, so a unit number is unique only inside its compound
        UniqueConstraint("compound_id", "unit_number"),
        CheckConstraint("building ~ '^[A-F]$'", name="building_letter"),
        CheckConstraint("unit_number ~ '^[A-F]-[0-9]{3}$'", name="unit_number_format"),
        # building copies the first letter (for the "top buildings" chart), so they must agree
        CheckConstraint("left(unit_number, 1) = building", name="unit_number_in_building"),
    )

    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    compound_id: Mapped[int] = mapped_column(ForeignKey("compounds.id", ondelete="RESTRICT"))
    building: Mapped[str]  # "A"
    unit_number: Mapped[str]  # "A-204" = building A, floor 2, apartment 04


class User(Base):
    """A resident's profile, recognized by the Telegram user ID."""

    __tablename__ = "users"
    __table_args__ = (
        # A second safety net: the app normalizes the phone first. NULL passes a CHECK, so a
        # user without a phone is allowed; a phone that is given must look like +201XXXXXXXXX
        CheckConstraint(r"phone ~ '^\+[0-9]{7,15}$'", name="phone_e164"),
    )

    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    # bigint: Telegram IDs can be bigger than an integer holds (about 2.1 billion)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    # Optional for now: whether name and phone are required is still a team decision
    name: Mapped[str | None] = mapped_column(comment=PERSONAL_DATA)
    phone: Mapped[str | None] = mapped_column(comment=PERSONAL_DATA)  # E.164: +201XXXXXXXXX
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class UserUnit(Base):
    """Which user lives in which unit (many-to-many). Saved only after the user confirms it."""

    __tablename__ = "user_units"

    # The two columns together are the primary key (a composite key): one row per pair
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    unit_id: Mapped[int] = mapped_column(
        ForeignKey("units.id", ondelete="RESTRICT"), primary_key=True
    )
    confirmed_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ServiceRequest(Base):
    """One confirmed service request. The dashboard reads mostly this table (through a view)."""

    __tablename__ = "service_requests"
    __table_args__ = (
        *label_checks(),
        CheckConstraint(one_of("language", LANGUAGES), name="language"),
        CheckConstraint(one_of("status", SR_STATUSES), name="status"),
        # SR-YYYYMMDD-NNNN: the Cairo date + the id, at least 4 digits
        CheckConstraint("ticket_number ~ '^SR-[0-9]{8}-[0-9]{4,}$'", name="ticket_number_format"),
        # closed_at is set exactly when the SR is closed ...
        CheckConstraint(
            "(status = 'closed') = (closed_at IS NOT NULL)", name="closed_at_if_closed"
        ),
        # ... and never before it was created. NULL >= x is NULL, and NULL passes a CHECK,
        # so open SRs (closed_at NULL) pass this rule
        CheckConstraint("closed_at >= created_at", name="closed_after_created"),
    )

    # BY DEFAULT, not ALWAYS: the app inserts the id it took from the sequence for the ticket
    id: Mapped[int] = mapped_column(Identity(always=False), primary_key=True)
    ticket_number: Mapped[str] = mapped_column(unique=True)
    # Empty after /forget (ON DELETE SET NULL). Indexed for /status and /forget
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    # Always a real registry unit. Indexed for the "top units" chart
    unit_id: Mapped[int] = mapped_column(ForeignKey("units.id", ondelete="RESTRICT"), index=True)
    # The problem in the user's words; "[deleted]" after /forget
    issue_description: Mapped[str] = mapped_column(comment=PERSONAL_DATA)
    language: Mapped[str]
    # The FINAL labels. What each model said is in model_predictions
    category: Mapped[str]
    urgency: Mapped[str]
    status: Mapped[str] = mapped_column(server_default="open")
    # When the user confirmed. Indexed for the charts by date
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), index=True)
    closed_at: Mapped[datetime | None]


class ModelPrediction(Base):
    """Every model answer for an SR: one row per SR per model version."""

    __tablename__ = "model_predictions"
    __table_args__ = (
        # Starts with sr_id, so its index also serves searches by sr_id: no own index needed
        UniqueConstraint("sr_id", "model_name", "model_version"),
        *label_checks(),
        CheckConstraint("category_confidence BETWEEN 0 AND 1", name="category_confidence_range"),
        CheckConstraint("urgency_confidence BETWEEN 0 AND 1", name="urgency_confidence_range"),
        CheckConstraint("latency_ms >= 0", name="latency_ms_not_negative"),
    )

    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    sr_id: Mapped[int] = mapped_column(ForeignKey("service_requests.id", ondelete="CASCADE"))
    model_name: Mapped[str]  # e.g. "qwen3-embedding-logreg"
    model_version: Mapped[str]  # e.g. "1.0.0"
    category: Mapped[str]
    category_confidence: Mapped[float]
    urgency: Mapped[str]
    urgency_confidence: Mapped[float]
    needs_review: Mapped[bool]  # a confidence below the threshold
    # How long the model took, in milliseconds (for the ML vs LLM comparison). Optional
    latency_ms: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ConversationMessage(Base):
    """The conversation history kept in PostgreSQL. The short-term copy for the LLM is in Redis."""

    __tablename__ = "conversation_messages"
    __table_args__ = (CheckConstraint(one_of("role", MESSAGE_ROLES), name="role"),)

    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    # No user_id: the first messages come before a profile exists. /forget deletes by chat ID,
    # so it is indexed. In a private chat, the chat ID = the user's Telegram ID
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    # Filled when the SR is saved; a message may belong to no SR
    sr_id: Mapped[int | None] = mapped_column(ForeignKey("service_requests.id", ondelete="CASCADE"))
    role: Mapped[str]  # "user" or "assistant"
    # For a voice note: the confirmed transcript. Audio is never stored
    content: Mapped[str] = mapped_column(comment=PERSONAL_DATA)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
