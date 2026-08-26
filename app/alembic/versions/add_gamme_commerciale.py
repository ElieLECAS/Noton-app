"""couche de connaissance métier : fiches des gammes commerciales Proferm

Le vocabulaire utilisateur (« Perform 76 ») et le vocabulaire documentaire
(« TROCAL 76 ADVANCED ») ne se recoupent nulle part. Ces fiches portent ce pont, pour
injection dans les prompts de compréhension et de génération.

Idempotente : create_all tourne aussi au démarrage de l'app.

Revision ID: add_gamme_commerciale
Revises: remove_dense_text_embeddings
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import ARRAY


revision = "add_gamme_commerciale"
down_revision = "remove_dense_text_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if "gamme_commerciale" in inspect(bind).get_table_names():
        return

    op.create_table(
        "gamme_commerciale",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("nom", sa.String(length=120), nullable=False),
        sa.Column("accroche", sa.String(length=255), nullable=True),
        sa.Column("materiau", sa.String(length=32), nullable=True),
        sa.Column("familles", ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("alias_utilisateur", ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("termes_documentaires", ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("fournisseurs", ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("discriminants", sa.Text(), nullable=True),
        sa.Column("document_ids", ARRAY(sa.Integer()), nullable=False, server_default="{}"),
        sa.Column("a_valider", sa.Text(), nullable=True),
        sa.Column("statut", sa.String(length=16), nullable=False, server_default="brouillon"),
        sa.Column("ordre", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.Integer(), sa.ForeignKey("user.id"), nullable=True),
    )
    op.create_index("ix_gamme_commerciale_slug", "gamme_commerciale", ["slug"], unique=True)
    op.create_index("ix_gamme_commerciale_statut", "gamme_commerciale", ["statut"])


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS gamme_commerciale;")
