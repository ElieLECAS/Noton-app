"""add source column to document and documentchunk

Revision ID: add_document_source
Revises: doc_run_audit
Create Date: 2026-04-20
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "add_document_source"
down_revision = "doc_run_audit"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column("document", sa.Column("source", sa.String(), nullable=True))
    op.add_column("documentchunk", sa.Column("source", sa.String(), nullable=True))
    op.create_index("ix_document_source", "document", ["source"])
    op.create_index("ix_documentchunk_source", "documentchunk", ["source"])

def downgrade() -> None:
    op.drop_index("ix_documentchunk_source", table_name="documentchunk")
    op.drop_index("ix_document_source", table_name="document")
    op.drop_column("documentchunk", "source")
    op.drop_column("document", "source")
