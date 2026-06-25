"""add axis facet to document categories + seed doc_type / lifecycle_phase / symptom

Revision ID: add_axis_to_document_categories
Revises: add_guided_sessions_and_trees
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "add_axis_to_document_categories"
down_revision = "add_guided_sessions_and_trees"
branch_labels = None
depends_on = None


# axis -> [(slug, label, description)]
SEED_BY_AXIS = {
    "doc_type": [
        ("notice_pose", "Notice de pose", "Procédure de pose pas à pas, séquence de montage sur chantier"),
        ("fiche_technique", "Fiche technique", "Caractéristiques techniques, cotes, performances produit"),
        ("doc_commerciale", "Document commercial", "Dépliant, argumentaire, brochure marketing"),
        ("pv_certification", "PV / certification", "Procès-verbal d'essai, certificat, marquage CE, attestation"),
        ("conditions_garantie", "Conditions de garantie", "Durée, conditions, exclusions de garantie"),
        ("guide_sav", "Guide SAV", "Guide de dépannage, diagnostic, intervention après-vente"),
        ("nomenclature", "Nomenclature / pièces", "Liste de composants, références pièces détachées, éclatés"),
        ("dtu_norme", "DTU / norme", "Document normatif, DTU, NF EN, réglementation"),
    ],
    "lifecycle_phase": [
        ("avant_vente", "Avant-vente", "Choix produit, devis, conseil avant achat"),
        ("chantier_pose", "Chantier / pose", "Phase de pose et mise en œuvre sur chantier"),
        ("apres_vente_sav", "Après-vente / SAV", "Usage, maintenance, dépannage après installation"),
    ],
    "symptom": [
        ("infiltration_eau", "Infiltration d'eau", "Entrée d'eau, fuite, défaut d'étanchéité à l'eau"),
        ("condensation", "Condensation", "Buée, condensation sur vitrage ou profilé"),
        ("blocage_manoeuvre", "Blocage de manœuvre", "Ouvrant dur, bloqué, manœuvre difficile"),
        ("deformation", "Déformation", "Profilé déformé, voilé, gauchi"),
        ("defaut_etancheite_air", "Défaut d'étanchéité à l'air", "Courant d'air, sifflement, perméabilité à l'air"),
        ("casse_quincaillerie", "Casse quincaillerie", "Pièce cassée : gâche, roulette, charnière défaillante"),
        ("bruit", "Bruit", "Grincement, claquement, bruit de manœuvre ou au vent"),
        ("desalignement_ouvrant", "Désalignement d'ouvrant", "Ouvrant désaligné, frottement, mauvais affleurement"),
    ],
}


def _columns(table: str) -> set[str]:
    conn = op.get_bind()
    inspector = inspect(conn)
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    cols = _columns("documentcategory")

    if "axis" not in cols:
        op.add_column(
            "documentcategory",
            sa.Column("axis", sa.String(length=32), nullable=False, server_default="task"),
        )
        op.execute("CREATE INDEX IF NOT EXISTS ix_documentcategory_axis ON documentcategory (axis)")
    if "parent_slug" not in cols:
        op.add_column(
            "documentcategory",
            sa.Column("parent_slug", sa.String(length=64), nullable=True),
        )

    # Les 16 catégories historiques restent sur l'axe `task` (zéro régression du boost/KAG).
    op.execute("UPDATE documentcategory SET axis = 'task' WHERE axis IS NULL OR axis = ''")

    # Seed des nouveaux axes (idempotent).
    for axis, rows in SEED_BY_AXIS.items():
        for slug, label, description in rows:
            escaped_desc = description.replace("'", "''")
            escaped_label = label.replace("'", "''")
            op.execute(
                f"""
                INSERT INTO documentcategory (slug, label, description, axis, is_active, created_at, updated_at)
                VALUES ('{slug}', '{escaped_label}', '{escaped_desc}', '{axis}', true, NOW(), NOW())
                ON CONFLICT (slug) DO NOTHING
                """
            )


def downgrade() -> None:
    # Purger d'abord les relations chunk↔catégorie des axes non-`task` (contrainte FK),
    # puis les catégories elles-mêmes, avant de retirer la colonne.
    op.execute(
        """
        DELETE FROM chunkcategoryrelation
        WHERE category_id IN (SELECT id FROM documentcategory WHERE axis <> 'task')
        """
    )
    op.execute("DELETE FROM documentcategory WHERE axis <> 'task'")
    op.execute("DROP INDEX IF EXISTS ix_documentcategory_axis")

    cols = _columns("documentcategory")
    if "parent_slug" in cols:
        op.drop_column("documentcategory", "parent_slug")
    if "axis" in cols:
        op.drop_column("documentcategory", "axis")
