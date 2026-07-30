"""guided runtime v2: session pinning/feedback/photos, gaps backlog, permission seed

Refonte "Arbre SAV" (plan docs/plan_refonte_arbre_sav_2026-07-30.md) :
- guidedsession : épinglage de la version d'arbre, feedback de résolution, photos client ;
- guidedgap : backlog des angles morts (symptômes/questions sans arbre publié) ;
- seed idempotent de la permission guided_trees:manage.

Revision ID: guided_runtime_v2
Revises: guided_tree_authoring
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSONB


revision = "guided_runtime_v2"
down_revision = "guided_tree_authoring"
branch_labels = None
depends_on = None


def _inspector():
    return inspect(op.get_bind())


def _existing_tables() -> set[str]:
    return set(_inspector().get_table_names(schema="public"))


def _existing_columns(table: str) -> set[str]:
    return {c["name"] for c in _inspector().get_columns(table)}


def upgrade() -> None:
    cols = _existing_columns("guidedsession")
    if "tree_version" not in cols:
        op.add_column("guidedsession", sa.Column("tree_version", sa.Integer(), nullable=True))
    if "resolved_feedback" not in cols:
        op.add_column("guidedsession", sa.Column("resolved_feedback", sa.Boolean(), nullable=True))
    if "uploaded_files" not in cols:
        op.add_column("guidedsession", sa.Column("uploaded_files", JSONB, nullable=True))

    if "guidedgap" not in _existing_tables():
        op.create_table(
            "guidedgap",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("space_id", sa.Integer(), nullable=False),
            sa.Column("detected_symptom", sa.String(120), nullable=True),
            sa.Column("query_text", sa.Text(), nullable=False, server_default=""),
            sa.Column("count", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("status", sa.String(20), nullable=False, server_default="open"),
            sa.Column("sample_conversation_ids", JSONB, nullable=True),
            sa.Column("first_seen", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("last_seen", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["space_id"], ["space.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_guidedgap_space_id", "guidedgap", ["space_id"])
        op.create_index("ix_guidedgap_detected_symptom", "guidedgap", ["detected_symptom"])
        op.create_index("ix_guidedgap_status", "guidedgap", ["status"])

    # Seed de la permission d'édition (idempotent — schéma réel de app/models/permission.py).
    # Gardé par l'existence de la table : le RBAC n'est pas couvert par les migrations
    # (créé par le create_all du startup) — sur une base vierge, la permission sera créée
    # via l'admin ; le rôle admin a de toute façon accès (require_sav_editor).
    if "permission" in _existing_tables():
        op.execute(
            "INSERT INTO permission (code, name, description, category, is_system, created_at) "
            "SELECT 'guided_trees:manage', 'Gérer les arbres SAV', "
            "'Créer et éditer les arbres SAV (publication réservée admin)', 'sav', true, now() "
            "WHERE NOT EXISTS (SELECT 1 FROM permission WHERE code = 'guided_trees:manage')"
        )


def downgrade() -> None:
    if "guidedgap" in _existing_tables():
        op.drop_table("guidedgap")
    cols = _existing_columns("guidedsession")
    for c in ("uploaded_files", "resolved_feedback", "tree_version"):
        if c in cols:
            op.drop_column("guidedsession", c)
    if "permission" in _existing_tables():
        op.execute("DELETE FROM permission WHERE code = 'guided_trees:manage'")
