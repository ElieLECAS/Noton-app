"""remove note_chunk table (optional migration for simplified RAG architecture)

Revision ID: remove_note_chunk_table
Revises: embedding_dim_384
Create Date: 2024-01-04 00:00:00.000000

NOTE: Cette migration est OPTIONNELLE. 
Le système fonctionne parfaitement sans l'exécuter.
Elle supprime simplement la table note_chunk qui n'est plus utilisée
dans la nouvelle architecture RAG simplifiée.

Pour l'appliquer:
    cd app
    alembic upgrade remove_note_chunk_table

"""
from typing import Sequence, Union

import os
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'remove_note_chunk_table'
down_revision: Union[str, None] = 'embedding_dim_384'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Supprimer la table note_chunk qui n'est plus utilisée dans la nouvelle architecture.
    
    ATTENTION: Cette opération supprime définitivement tous les chunks existants.
    Assurez-vous d'avoir exécuté le script de régénération des embeddings avant:
        python -m app.scripts.regenerate_all_embeddings
    """
    conn = op.get_bind()
    notechunk_exists = conn.execute(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'notechunk'
            )
            """
        )
    ).scalar()

    # Sécurité : cette migration est "optionnelle".
    # Par défaut, on ne supprime plus notechunk pour éviter de casser
    # les branches de migration qui la manipulent ensuite.
    force_drop = os.getenv("ALEMBIC_FORCE_DROP_NOTECHUNK", "false").lower() == "true"
    if not force_drop:
        print(
            "remove_note_chunk_table: skip (set ALEMBIC_FORCE_DROP_NOTECHUNK=true to force drop)."
        )
        return

    if not notechunk_exists:
        print("remove_note_chunk_table: notechunk absente, rien à supprimer.")
        return

    op.execute("DROP INDEX IF EXISTS notechunk_embedding_idx")
    op.execute("DROP INDEX IF EXISTS ix_notechunk_note_id")
    op.drop_table("notechunk")

    print("""
╔══════════════════════════════════════════════════════════════╗
║  Table note_chunk supprimée avec succès                      ║
╠══════════════════════════════════════════════════════════════╣
║  Suppression forcée (ALEMBIC_FORCE_DROP_NOTECHUNK=true).     ║
║  Vérifiez que votre branche de migrations ne dépend pas      ║
║  ensuite de la table notechunk.                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def downgrade() -> None:
    """
    Recréer la table note_chunk (en cas de rollback).
    
    NOTE: Les données des chunks ne seront PAS restaurées.
    Vous devrez exécuter un script de re-chunking si nécessaire.
    """
    # Recréer la table notechunk
    op.create_table(
        'notechunk',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('note_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('start_char', sa.Integer(), nullable=False),
        sa.Column('end_char', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['note_id'], ['note.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Recréer l'index sur note_id
    op.create_index('ix_notechunk_note_id', 'notechunk', ['note_id'])
    
    # Recréer l'index HNSW sur les embeddings
    # Note: Nécessite l'extension pgvector
    op.execute('CREATE INDEX IF NOT EXISTS notechunk_embedding_idx ON notechunk USING hnsw (embedding vector_cosine_ops)')
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Table note_chunk restaurée                                  ║
╠══════════════════════════════════════════════════════════════╣
║  ATTENTION: La table est vide, vous devez re-créer les      ║
║  chunks avec un script de migration si nécessaire.           ║
╚══════════════════════════════════════════════════════════════╝
    """)

