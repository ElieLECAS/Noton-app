"""Le mode d'une conversation : ``chat`` (l'écran d'accueil) ou ``vocal`` (l'assistant vocal).

Une colonne, idempotente (``IF NOT EXISTS``), comme ``create_all`` au démarrage. Les
conversations existantes sont des conversations de chat.

Revision ID: lia_vocal_mode
Revises: lia_wiki_schema
Create Date: 2026-09-22
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "lia_vocal_mode"
down_revision = "lia_wiki_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.get_bind().execute(
        text("ALTER TABLE conversation ADD COLUMN IF NOT EXISTS mode VARCHAR(20) NOT NULL DEFAULT 'chat'")
    )


def downgrade() -> None:
    op.get_bind().execute(text("ALTER TABLE conversation DROP COLUMN IF EXISTS mode"))
