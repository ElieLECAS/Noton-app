"""drop KAG tables (knowledge graph)

Revision ID: drop_kag_tables
Revises: add_document_source
"""

from alembic import op
from sqlalchemy import inspect


revision = "drop_kag_tables"
down_revision = "add_document_source"
branch_labels = None
depends_on = None

_KAG_TABLES = (
    "chunkentityrelation",
    "entityalias",
    "entityentityrelation",
    "knowledgeentity",
)


def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    existing = set(inspector.get_table_names(schema="public"))
    for table in _KAG_TABLES:
        if table in existing:
            op.drop_table(table)


def downgrade() -> None:
    # Tables recréées par migrations historiques add_kag_* si besoin de rollback manuel.
    pass
