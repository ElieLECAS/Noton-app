"""add match_symptoms column to guidedtree (SAV diagnostic matching)

Revision ID: add_match_symptoms_to_guidedtree
Revises: add_category_candidates
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "add_match_symptoms_to_guidedtree"
down_revision = "add_category_candidates"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    conn = op.get_bind()
    inspector = inspect(conn)
    if table not in set(inspector.get_table_names(schema="public")):
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    cols = _columns("guidedtree")
    if cols and "match_symptoms" not in cols:
        op.add_column("guidedtree", sa.Column("match_symptoms", sa.JSON(), nullable=True))


def downgrade() -> None:
    cols = _columns("guidedtree")
    if "match_symptoms" in cols:
        op.drop_column("guidedtree", "match_symptoms")
