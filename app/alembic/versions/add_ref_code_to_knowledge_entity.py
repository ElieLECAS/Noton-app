"""add ref_code identity column to knowledgeentity

Pivot de la résolution d'identité KAG : un produit = un nœud, keyé par son code
(« 6111 », « RAL:7016 »…). Colonne nullable + index de lookup (space_id, ref_code).
Pas de backfill : l'état propre vient du retraitement KAG (le pipeline remplit ref_code).

Revision ID: add_ref_code_to_knowledge_entity
Revises: add_two_level_taxonomy
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "add_ref_code_to_knowledge_entity"
down_revision = "add_two_level_taxonomy"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    conn = op.get_bind()
    inspector = inspect(conn)
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    if "ref_code" not in _columns("knowledgeentity"):
        op.add_column(
            "knowledgeentity",
            sa.Column("ref_code", sa.String(length=64), nullable=True),
        )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledgeentity_ref_code "
        "ON knowledgeentity (ref_code)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledgeentity_space_ref_code "
        "ON knowledgeentity (space_id, ref_code)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_knowledgeentity_space_ref_code")
    op.execute("DROP INDEX IF EXISTS ix_knowledgeentity_ref_code")
    if "ref_code" in _columns("knowledgeentity"):
        op.drop_column("knowledgeentity", "ref_code")
