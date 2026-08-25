"""remove dense text embeddings (retriever = ColPali + BM25 seulement)

Le retriever texte dense (pgvector sur documentchunk.embedding) est supprimé : il
lisait la même évidence textuelle que BM25 (tsv_content, généré automatiquement) et
votait deux fois à la fusion RRF sans rien ajouter que ColPali ne couvre déjà pour le
visuel. Cf. docs/plan_retriever_colpali_bm25_2026-08-25.md.

document.embedding n'était déjà écrit nulle part (vestige) — supprimé avec le même
mouvement. L'extension pgvector reste active (guidedentryindex, knowledgeentity).

Revision ID: remove_dense_text_embeddings
Revises: guided_tree_layout
"""

from alembic import op


revision = "remove_dense_text_embeddings"
down_revision = "guided_tree_layout"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documentchunk_embedding_hnsw;")
    op.execute("ALTER TABLE documentchunk DROP COLUMN IF EXISTS embedding;")
    op.execute("ALTER TABLE document DROP COLUMN IF EXISTS embedding;")


def downgrade() -> None:
    # Reconstitue la forme du schéma (pas les données : les vecteurs sont perdus).
    op.execute("ALTER TABLE documentchunk ADD COLUMN IF NOT EXISTS embedding vector(1024);")
    op.execute("ALTER TABLE document ADD COLUMN IF NOT EXISTS embedding vector(1024);")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_documentchunk_embedding_hnsw
        ON documentchunk USING hnsw (embedding vector_cosine_ops);
        """
    )
