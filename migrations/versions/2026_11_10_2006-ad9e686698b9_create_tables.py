"""create tables

Migration 1: the 7 tables of app/db/models.py (compounds, users, units, service_requests,
user_units, conversation_messages, model_predictions) with their rules and 4 indexes. Written by
`alembic revision --autogenerate` and read line by line before it was applied. Only the
comments were changed by hand.

The label lists in the CHECK rules are written out on purpose, not imported from app/labels.py:
this file must always build the tables as they were on this day. A later label change gets its
own migration.

Revision ID: ad9e686698b9
Revises:
Create Date: 2026-11-10 20:06:09.901843

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ad9e686698b9"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Parents before children: a foreign key can only point to a table that already exists
    op.create_table(
        "compounds",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("name_en", sa.Text(), nullable=False),
        sa.Column("name_ar", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_compounds")),
        sa.UniqueConstraint("name_ar", name=op.f("uq_compounds_name_ar")),
        sa.UniqueConstraint("name_en", name=op.f("uq_compounds_name_en")),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "name", sa.Text(), nullable=True, comment="Personal data: never in logs, traces or git"
        ),
        sa.Column(
            "phone", sa.Text(), nullable=True, comment="Personal data: never in logs, traces or git"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("phone ~ '^\\+[0-9]{7,15}$'", name=op.f("ck_users_phone_e164")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("telegram_user_id", name=op.f("uq_users_telegram_user_id")),
    )
    op.create_table(
        "units",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("compound_id", sa.Integer(), nullable=False),
        sa.Column("building", sa.Text(), nullable=False),
        sa.Column("unit_number", sa.Text(), nullable=False),
        sa.CheckConstraint("building ~ '^[A-F]$'", name=op.f("ck_units_building_letter")),
        sa.CheckConstraint(
            "unit_number ~ '^[A-F]-[0-9]{3}$'", name=op.f("ck_units_unit_number_format")
        ),
        sa.CheckConstraint(
            "left(unit_number, 1) = building", name=op.f("ck_units_unit_number_in_building")
        ),
        sa.ForeignKeyConstraint(
            ["compound_id"],
            ["compounds.id"],
            name=op.f("fk_units_compound_id_compounds"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_units")),
        sa.UniqueConstraint(
            "compound_id", "unit_number", name=op.f("uq_units_compound_id_unit_number")
        ),
    )
    op.create_table(
        "service_requests",
        sa.Column("id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("ticket_number", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column(
            "issue_description",
            sa.Text(),
            nullable=False,
            comment="Personal data: never in logs, traces or git",
        ),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("urgency", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="open", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'closed') = (closed_at IS NOT NULL)",
            name=op.f("ck_service_requests_closed_at_if_closed"),
        ),
        sa.CheckConstraint(
            "category IN ('plumbing', 'electricity', 'ac', 'carpentry', 'pest_control', 'general')",
            name=op.f("ck_service_requests_category"),
        ),
        sa.CheckConstraint("language IN ('ar', 'en')", name=op.f("ck_service_requests_language")),
        sa.CheckConstraint(
            "status IN ('open', 'in_progress', 'closed')", name=op.f("ck_service_requests_status")
        ),
        sa.CheckConstraint(
            "ticket_number ~ '^SR-[0-9]{8}-[0-9]{4,}$'",
            name=op.f("ck_service_requests_ticket_number_format"),
        ),
        sa.CheckConstraint(
            "urgency IN ('low', 'medium', 'high')", name=op.f("ck_service_requests_urgency")
        ),
        sa.CheckConstraint(
            "closed_at >= created_at", name=op.f("ck_service_requests_closed_after_created")
        ),
        sa.ForeignKeyConstraint(
            ["unit_id"],
            ["units.id"],
            name=op.f("fk_service_requests_unit_id_units"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_service_requests_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_service_requests")),
        sa.UniqueConstraint("ticket_number", name=op.f("uq_service_requests_ticket_number")),
    )
    op.create_index(
        op.f("ix_service_requests_created_at"), "service_requests", ["created_at"], unique=False
    )
    op.create_index(
        op.f("ix_service_requests_unit_id"), "service_requests", ["unit_id"], unique=False
    )
    op.create_index(
        op.f("ix_service_requests_user_id"), "service_requests", ["user_id"], unique=False
    )
    op.create_table(
        "user_units",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column(
            "confirmed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["unit_id"], ["units.id"], name=op.f("fk_user_units_unit_id_units"), ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_user_units_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "unit_id", name=op.f("pk_user_units")),
    )
    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("sr_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
            comment="Personal data: never in logs, traces or git",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant')", name=op.f("ck_conversation_messages_role")
        ),
        sa.ForeignKeyConstraint(
            ["sr_id"],
            ["service_requests.id"],
            name=op.f("fk_conversation_messages_sr_id_service_requests"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversation_messages")),
    )
    op.create_index(
        op.f("ix_conversation_messages_chat_id"), "conversation_messages", ["chat_id"], unique=False
    )
    op.create_table(
        "model_predictions",
        sa.Column("id", sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column("sr_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("category_confidence", sa.Double(), nullable=False),
        sa.Column("urgency", sa.Text(), nullable=False),
        sa.Column("urgency_confidence", sa.Double(), nullable=False),
        sa.Column("needs_review", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "category IN ('plumbing', 'electricity', 'ac', 'carpentry', 'pest_control', 'general')",
            name=op.f("ck_model_predictions_category"),
        ),
        sa.CheckConstraint(
            "urgency IN ('low', 'medium', 'high')", name=op.f("ck_model_predictions_urgency")
        ),
        sa.CheckConstraint(
            "category_confidence BETWEEN 0 AND 1",
            name=op.f("ck_model_predictions_category_confidence_range"),
        ),
        sa.CheckConstraint(
            "latency_ms >= 0", name=op.f("ck_model_predictions_latency_ms_not_negative")
        ),
        sa.CheckConstraint(
            "urgency_confidence BETWEEN 0 AND 1",
            name=op.f("ck_model_predictions_urgency_confidence_range"),
        ),
        sa.ForeignKeyConstraint(
            ["sr_id"],
            ["service_requests.id"],
            name=op.f("fk_model_predictions_sr_id_service_requests"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_predictions")),
        sa.UniqueConstraint(
            "sr_id",
            "model_name",
            "model_version",
            name=op.f("uq_model_predictions_sr_id_model_name_model_version"),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Children before parents: the reverse order of upgrade(). This deletes every row too
    op.drop_table("model_predictions")
    op.drop_index(op.f("ix_conversation_messages_chat_id"), table_name="conversation_messages")
    op.drop_table("conversation_messages")
    op.drop_table("user_units")
    op.drop_index(op.f("ix_service_requests_user_id"), table_name="service_requests")
    op.drop_index(op.f("ix_service_requests_unit_id"), table_name="service_requests")
    op.drop_index(op.f("ix_service_requests_created_at"), table_name="service_requests")
    op.drop_table("service_requests")
    op.drop_table("units")
    op.drop_table("users")
    op.drop_table("compounds")
