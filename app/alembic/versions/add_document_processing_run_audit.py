"""document processing_run_id, phase json, last error; admin_audit_log; cancelled -> cancelled_by_user

Revision ID: doc_run_audit
Revises: add_ke_confidence
Create Date: 2026-04-15

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "doc_run_audit"
down_revision = "add_ke_confidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document",
        sa.Column("processing_run_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "document",
        sa.Column("phase_status_json", JSONB(), nullable=True),
    )
    op.add_column(
        "document",
        sa.Column("last_processing_error", sa.String(), nullable=True),
    )
    op.execute(
        "UPDATE document SET processing_status = 'cancelled_by_user' "
        "WHERE processing_status = 'cancelled'"
    )
    op.create_table(
        "adminauditlog",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("detail_json", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_adminauditlog_user_id"), "adminauditlog", ["user_id"], unique=False
    )
    op.create_index(
        op.f("ix_adminauditlog_action"), "adminauditlog", ["action"], unique=False
    )
    op.create_index(
        op.f("ix_adminauditlog_created_at"),
        "adminauditlog",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.execute(
        "UPDATE document SET processing_status = 'cancelled' "
        "WHERE processing_status = 'cancelled_by_user'"
    )
    op.drop_index(op.f("ix_adminauditlog_created_at"), table_name="adminauditlog")
    op.drop_index(op.f("ix_adminauditlog_action"), table_name="adminauditlog")
    op.drop_index(op.f("ix_adminauditlog_user_id"), table_name="adminauditlog")
    op.drop_table("adminauditlog")
    op.drop_column("document", "last_processing_error")
    op.drop_column("document", "phase_status_json")
    op.drop_column("document", "processing_run_id")
