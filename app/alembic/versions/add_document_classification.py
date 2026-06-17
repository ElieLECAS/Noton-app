"""add document classification metadata columns

Revision ID: add_document_classification
Revises: add_category_to_message_feedback
Create Date: 2026-06-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "add_document_classification"
down_revision = "add_category_to_message_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document",
        sa.Column(
            "product_types",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
    )
    op.add_column(
        "document",
        sa.Column(
            "materials",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
    )
    op.add_column(
        "document",
        sa.Column("coulissant_galandage", sa.String(length=8), nullable=True),
    )
    op.add_column(
        "document",
        sa.Column(
            "proferm_gammes",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
    )
    op.add_column(
        "document",
        sa.Column(
            "classification_status",
            sa.String(length=16),
            nullable=False,
            server_default="incomplete",
        ),
    )
    op.create_index(
        "ix_document_classification_status",
        "document",
        ["classification_status"],
    )
    op.execute(
        "CREATE INDEX ix_document_product_types ON document USING GIN (product_types)"
    )
    op.execute(
        "CREATE INDEX ix_document_materials ON document USING GIN (materials)"
    )
    op.execute(
        "CREATE INDEX ix_document_proferm_gammes ON document USING GIN (proferm_gammes)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_proferm_gammes")
    op.execute("DROP INDEX IF EXISTS ix_document_materials")
    op.execute("DROP INDEX IF EXISTS ix_document_product_types")
    op.drop_index("ix_document_classification_status", table_name="document")
    op.drop_column("document", "classification_status")
    op.drop_column("document", "proferm_gammes")
    op.drop_column("document", "coulissant_galandage")
    op.drop_column("document", "materials")
    op.drop_column("document", "product_types")
