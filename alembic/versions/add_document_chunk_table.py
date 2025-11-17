"""add document_chunk table and update note for docling integration

Revision ID: add_document_chunk_table
Revises: remove_note_chunk_table
Create Date: 2025-11-17 21:00:00.000000

Cette migration ajoute le système DocumentChunk pour la gestion unifiée
des documents (notes manuelles et fichiers importés via Docling).

Changements:
- Nouvelle table document_chunk avec métadonnées enrichies
- Ajout de champs source_file_path et source_file_type à note
- Support pour embeddings pgvector sur les chunks

Pour l'appliquer depuis Docker:
    docker-compose exec app alembic upgrade head

Après la migration du schéma, exécuter le script de migration des données:
    docker-compose exec app python -m app.scripts.migrate_to_docling --migrate
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'add_document_chunk_table'
down_revision: Union[str, None] = 'remove_note_chunk_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Créer la table document_chunk et ajouter les champs pour les documents uploadés.
    """
    # Créer la table document_chunk
    op.create_table(
        'document_chunk',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('note_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('chunk_type', sa.String(), nullable=False, server_default='text'),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('section_title', sa.String(), nullable=True),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('start_char', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('end_char', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('embedding', Vector(384), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['note_id'], ['note.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Créer les index
    op.create_index('ix_document_chunk_note_id', 'document_chunk', ['note_id'])
    op.create_index('ix_document_chunk_chunk_index', 'document_chunk', ['chunk_index'])
    
    # Créer l'index HNSW pour la recherche vectorielle rapide
    # Note: Nécessite l'extension pgvector (déjà installée)
    op.execute('''
        CREATE INDEX IF NOT EXISTS document_chunk_embedding_hnsw_idx 
        ON document_chunk 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    ''')
    
    # Ajouter les colonnes pour les documents uploadés à la table note
    op.add_column('note', sa.Column('source_file_path', sa.String(), nullable=True))
    op.add_column('note', sa.Column('source_file_type', sa.String(), nullable=True))
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Migration DocumentChunk appliquée avec succès               ║
╠══════════════════════════════════════════════════════════════╣
║  ✅ Table document_chunk créée                               ║
║  ✅ Index vectoriels (HNSW) créés                            ║
║  ✅ Champs source_file_* ajoutés à note                      ║
║                                                              ║
║  📋 Prochaines étapes:                                       ║
║  1. Migrer les notes existantes vers chunks:                ║
║     docker-compose exec app python -m \\                     ║
║       app.scripts.migrate_to_docling --migrate              ║
║                                                              ║
║  2. Vérifier la migration:                                   ║
║     docker-compose exec app python -m \\                     ║
║       app.scripts.migrate_to_docling --verify               ║
╚══════════════════════════════════════════════════════════════╝
    """)


def downgrade() -> None:
    """
    Supprimer la table document_chunk et les champs ajoutés.
    
    ATTENTION: Cette opération supprime tous les chunks et les données
    de documents uploadés.
    """
    # Supprimer les index
    op.execute('DROP INDEX IF EXISTS document_chunk_embedding_hnsw_idx')
    op.drop_index('ix_document_chunk_chunk_index', table_name='document_chunk')
    op.drop_index('ix_document_chunk_note_id', table_name='document_chunk')
    
    # Supprimer la table
    op.drop_table('document_chunk')
    
    # Supprimer les colonnes ajoutées à note
    op.drop_column('note', 'source_file_type')
    op.drop_column('note', 'source_file_path')
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Rollback effectué                                           ║
╠══════════════════════════════════════════════════════════════╣
║  ⚠️  Table document_chunk supprimée                          ║
║  ⚠️  Champs source_file_* supprimés                          ║
║                                                              ║
║  Les données des chunks et documents sont perdues.           ║
╚══════════════════════════════════════════════════════════════╝
    """)

