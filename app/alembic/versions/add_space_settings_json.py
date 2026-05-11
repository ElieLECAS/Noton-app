"""add space settings_json

Revision ID: add_space_settings_json
Revises: add_document_source
Create Date: 2026-05-11 09:45:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "add_space_settings_json"
down_revision = "add_document_source"
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
    cols = _get_columns("space")
    if "settings_json" not in cols:
        op.add_column("space", sa.Column("settings_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    cols = _get_columns("space")
    if "settings_json" in cols:
        op.drop_column("space", "settings_json")

