"""two-level taxonomy: general families (axis task_group) + parent_slug + sharpened descriptions

Matérialise les 5 familles générales (niveau 1, axis=task_group, jamais assignées
directement aux chunks) et renseigne ``parent_slug`` sur les 16 catégories précises
(niveau 2, axis=task). Met aussi à jour les descriptions désambiguïsées en base (elles
alimentent le prompt de classification). Idempotent.

Revision ID: add_two_level_taxonomy
Revises: add_ccr_is_primary
"""

from alembic import op
from sqlalchemy import text

from app.services.category_catalog import (
    CONTENT_CATEGORY_SLUGS,
    DEFAULT_CATEGORY_DESCRIPTIONS,
)
from app.services.theme_tree_catalog import THEME_FAMILIES, family_for_task_slug


revision = "add_two_level_taxonomy"
down_revision = "add_ccr_is_primary"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Familles générales (niveau 1) — upsert idempotent, axis=task_group.
    for fam in THEME_FAMILIES:
        bind.execute(
            text(
                """
                INSERT INTO documentcategory
                    (slug, label, description, axis, parent_slug, is_active, created_at, updated_at)
                VALUES (:slug, :label, :description, 'task_group', NULL, true, now(), now())
                ON CONFLICT (slug) DO UPDATE
                SET label = EXCLUDED.label,
                    axis = 'task_group',
                    is_active = true,
                    updated_at = now()
                """
            ),
            {
                "slug": fam["slug"],
                "label": fam["label"],
                "description": f"Famille générale : {fam['label']}",
            },
        )

    # 2. Catégories précises (niveau 2) : parent_slug + description désambiguïsée.
    for slug in CONTENT_CATEGORY_SLUGS:
        bind.execute(
            text(
                """
                UPDATE documentcategory
                SET parent_slug = :parent,
                    description = :description,
                    updated_at = now()
                WHERE slug = :slug
                """
            ),
            {
                "slug": slug,
                "parent": family_for_task_slug(slug),
                "description": DEFAULT_CATEGORY_DESCRIPTIONS.get(slug, ""),
            },
        )


def downgrade() -> None:
    bind = op.get_bind()
    # Retire le rattachement parent des 16 précises.
    for slug in CONTENT_CATEGORY_SLUGS:
        bind.execute(
            text("UPDATE documentcategory SET parent_slug = NULL WHERE slug = :slug"),
            {"slug": slug},
        )
    # Désactive les familles générales (on ne les supprime pas pour préserver l'historique).
    for fam in THEME_FAMILIES:
        bind.execute(
            text(
                "UPDATE documentcategory SET is_active = false WHERE slug = :slug AND axis = 'task_group'"
            ),
            {"slug": fam["slug"]},
        )
