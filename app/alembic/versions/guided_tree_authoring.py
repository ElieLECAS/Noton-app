"""guided tree authoring: status/perimeter, versions, attachments, aliases, entry index

Refonte "Arbre SAV" (plan docs/plan_refonte_arbre_sav_2026-07-30.md) :
- guidedtree : autorat (status draft/published/archived, symptôme d'entrée, périmètre,
  version courante, auteurs) — is_active backfillée puis supprimée ;
- guidedtreenode : champs d'autorat (titre court, note interne, photo, texte libre,
  condition de périmètre, outillage) ;
- guidedtreeversion : snapshots publiés (le runtime ne lit que ça) ;
- guidednodeattachment : pièces jointes bibliothèque (document + plage de pages) ;
- guidedsymptomalias : vocabulaire client des symptômes ;
- guidedentryindex : index d'entrée sémantique N0 (reconstruit à la publication).

Revision ID: guided_tree_authoring
Revises: add_ref_code_to_knowledge_entity
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector


revision = "guided_tree_authoring"
down_revision = "add_ref_code_to_knowledge_entity"
branch_labels = None
depends_on = None

# Dimension figée au moment de la migration (mistral-embed) ; les migrations
# update_embedding_dim_* historiques gèrent les changements de dimension.
_EMBEDDING_DIM = 1024


def _inspector():
    return inspect(op.get_bind())


def _existing_tables() -> set[str]:
    return set(_inspector().get_table_names(schema="public"))


def _existing_columns(table: str) -> set[str]:
    return {c["name"] for c in _inspector().get_columns(table)}


def upgrade() -> None:
    existing = _existing_tables()

    # --- guidedtree : autorat ---
    cols = _existing_columns("guidedtree")
    if "status" not in cols:
        op.add_column(
            "guidedtree",
            sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
        )
        if "is_active" in cols:
            op.execute("UPDATE guidedtree SET status = 'published' WHERE is_active = true")
        op.create_index("ix_guidedtree_status", "guidedtree", ["status"])
    if "entry_symptom" not in cols:
        op.add_column("guidedtree", sa.Column("entry_symptom", sa.String(120), nullable=True))
        op.execute(
            "UPDATE guidedtree SET entry_symptom = match_symptoms->>0 "
            "WHERE match_symptoms IS NOT NULL "
            "AND jsonb_typeof(match_symptoms::jsonb) = 'array' "
            "AND jsonb_array_length(match_symptoms::jsonb) > 0"
        )
    if "description" not in cols:
        op.add_column(
            "guidedtree", sa.Column("description", sa.Text(), nullable=False, server_default="")
        )
    if "perimeter" not in cols:
        op.add_column("guidedtree", sa.Column("perimeter", JSONB, nullable=True))
    if "current_version" not in cols:
        op.add_column(
            "guidedtree",
            sa.Column("current_version", sa.Integer(), nullable=False, server_default="0"),
        )
    if "created_by" not in cols:
        op.add_column("guidedtree", sa.Column("created_by", sa.Integer(), nullable=True))
    if "updated_by" not in cols:
        op.add_column("guidedtree", sa.Column("updated_by", sa.Integer(), nullable=True))
    if "is_active" in cols:
        op.execute("DROP INDEX IF EXISTS ix_guidedtree_is_active")
        op.drop_column("guidedtree", "is_active")

    # --- guidedtreenode : autorat ---
    cols = _existing_columns("guidedtreenode")
    node_columns = [
        ("title", sa.Column("title", sa.String(200), nullable=False, server_default="")),
        (
            "internal_note",
            sa.Column("internal_note", sa.Text(), nullable=False, server_default=""),
        ),
        (
            "ask_photo",
            sa.Column("ask_photo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        ),
        (
            "allow_free_text",
            sa.Column(
                "allow_free_text", sa.Boolean(), nullable=False, server_default=sa.text("true")
            ),
        ),
        ("perimeter_condition", sa.Column("perimeter_condition", JSONB, nullable=True)),
        ("tools_hint", sa.Column("tools_hint", sa.String(200), nullable=False, server_default="")),
    ]
    for name, col in node_columns:
        if name not in cols:
            op.add_column("guidedtreenode", col)

    # --- versions publiées (snapshots) ---
    if "guidedtreeversion" not in existing:
        op.create_table(
            "guidedtreeversion",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("tree_id", sa.Integer(), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("snapshot", JSONB, nullable=False),
            sa.Column("note", sa.String(300), nullable=False, server_default=""),
            sa.Column("published_by", sa.Integer(), nullable=True),
            sa.Column(
                "published_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
            ),
            sa.ForeignKeyConstraint(["tree_id"], ["guidedtree.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("tree_id", "version", name="uq_guidedtreeversion_tree_version"),
        )
        op.create_index("ix_guidedtreeversion_tree_id", "guidedtreeversion", ["tree_id"])

    # --- pièces jointes bibliothèque par nœud ---
    if "guidednodeattachment" not in existing:
        op.create_table(
            "guidednodeattachment",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("node_id", sa.Integer(), nullable=False),
            sa.Column("document_id", sa.Integer(), nullable=False),
            sa.Column("page_start", sa.Integer(), nullable=True),
            sa.Column("page_end", sa.Integer(), nullable=True),
            sa.Column("caption", sa.String(300), nullable=False, server_default=""),
            sa.Column("kind", sa.String(20), nullable=False, server_default="notice"),
            sa.Column("display_order", sa.Integer(), nullable=False, server_default="0"),
            sa.ForeignKeyConstraint(["node_id"], ["guidedtreenode.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["document_id"], ["document.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_guidednodeattachment_node_id", "guidednodeattachment", ["node_id"])
        op.create_index(
            "ix_guidednodeattachment_document_id", "guidednodeattachment", ["document_id"]
        )

    # --- alias de symptômes (vocabulaire client) ---
    if "guidedsymptomalias" not in existing:
        op.create_table(
            "guidedsymptomalias",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symptom_slug", sa.String(120), nullable=False),
            sa.Column("alias", sa.String(200), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("symptom_slug", "alias", name="uq_guidedsymptomalias_slug_alias"),
        )
        op.create_index(
            "ix_guidedsymptomalias_symptom_slug", "guidedsymptomalias", ["symptom_slug"]
        )

    # --- index d'entrée sémantique N0 (reconstruit à la publication) ---
    if "guidedentryindex" not in existing:
        op.create_table(
            "guidedentryindex",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("space_id", sa.Integer(), nullable=True),
            sa.Column("tree_id", sa.Integer(), nullable=True),
            sa.Column("tree_version", sa.Integer(), nullable=True),
            sa.Column("entry_kind", sa.String(20), nullable=False),
            sa.Column("ref_key", sa.String(160), nullable=False),
            sa.Column("label", sa.String(300), nullable=False, server_default=""),
            sa.Column("text", sa.Text(), nullable=False, server_default=""),
            sa.Column("filters", JSONB, nullable=True),
            sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=True),
            sa.ForeignKeyConstraint(["tree_id"], ["guidedtree.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_guidedentryindex_space_id", "guidedentryindex", ["space_id"])
        op.create_index("ix_guidedentryindex_tree_id", "guidedentryindex", ["tree_id"])


def downgrade() -> None:
    existing = _existing_tables()
    for t in ("guidedentryindex", "guidedsymptomalias", "guidednodeattachment", "guidedtreeversion"):
        if t in existing:
            op.drop_table(t)

    cols = _existing_columns("guidedtreenode")
    for c in ("tools_hint", "perimeter_condition", "allow_free_text", "ask_photo", "internal_note", "title"):
        if c in cols:
            op.drop_column("guidedtreenode", c)

    cols = _existing_columns("guidedtree")
    if "is_active" not in cols:
        op.add_column(
            "guidedtree",
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        )
        op.execute("UPDATE guidedtree SET is_active = true WHERE status = 'published'")
        op.create_index("ix_guidedtree_is_active", "guidedtree", ["is_active"])
    op.execute("DROP INDEX IF EXISTS ix_guidedtree_status")
    for c in ("updated_by", "created_by", "current_version", "perimeter", "description", "entry_symptom", "status"):
        op.execute(f"ALTER TABLE guidedtree DROP COLUMN IF EXISTS {c}")
