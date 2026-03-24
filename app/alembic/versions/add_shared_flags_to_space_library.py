"""add shared/global flags to space and library

Revision ID: add_shared_flags_space_library
Revises: add_document_is_paid
Create Date: 2026-03-24 14:30:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "add_shared_flags_space_library"
down_revision = "add_document_is_paid"
branch_labels = None
depends_on = None


def _get_columns(table_name: str) -> set[str]:
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = :table_name
            """
        ),
        {"table_name": table_name},
    )
    return {row[0] for row in result}


def upgrade() -> None:
    space_columns = _get_columns("space")
    library_columns = _get_columns("library")

    if "is_shared" not in space_columns:
        op.add_column(
            "space",
            sa.Column("is_shared", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        )
        op.execute(sa.text("UPDATE space SET is_shared = true WHERE is_shared IS NULL"))
        op.alter_column("space", "is_shared", server_default=None)

    if "is_global" not in library_columns:
        op.add_column(
            "library",
            sa.Column("is_global", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        )
        op.execute(sa.text("UPDATE library SET is_global = false WHERE is_global IS NULL"))
        op.alter_column("library", "is_global", server_default=None)

    # Permettre une bibliothèque globale sans propriétaire explicite.
    if "user_id" in library_columns:
        op.alter_column("library", "user_id", existing_type=sa.Integer(), nullable=True)

    # Permettre des espaces partagés sans propriétaire explicite.
    if "user_id" in space_columns:
        op.alter_column("space", "user_id", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    space_columns = _get_columns("space")
    library_columns = _get_columns("library")

    if "is_shared" in space_columns:
        op.drop_column("space", "is_shared")

    if "is_global" in library_columns:
        op.drop_column("library", "is_global")
