"""guided tree layout: positions libres des cas dans le graphe (déplacement à la souris)

Les positions sont normalisées (0..1) et stockées sur l'arbre :
{node_key: {"nx": 0.42, "ny": 0.15}}. Absent → placement automatique par niveaux.

Revision ID: guided_tree_layout
Revises: guided_runtime_v2
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSONB


revision = "guided_tree_layout"
down_revision = "guided_runtime_v2"
branch_labels = None
depends_on = None


def _existing_columns(table: str) -> set[str]:
    return {c["name"] for c in inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    if "layout" not in _existing_columns("guidedtree"):
        op.add_column("guidedtree", sa.Column("layout", JSONB, nullable=True))


def downgrade() -> None:
    if "layout" in _existing_columns("guidedtree"):
        op.drop_column("guidedtree", "layout")
