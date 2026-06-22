"""add document categories and chunk category relations

Revision ID: add_document_categories
Revises: recreate_kag_tables
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "add_document_categories"
down_revision = "recreate_kag_tables"
branch_labels = None
depends_on = None

SEED_CATEGORIES = [
    (
        "mounting",
        "Pose / montage",
        "Séquences de pose, assemblage, fixation, calage, pattes, gondage",
    ),
    (
        "hardware_adjustment",
        "Réglage quincaillerie",
        "Roulettes, gâches, réglages de manœuvre, alignement ouvrant",
    ),
    (
        "sealing",
        "Étanchéité / calfeutrement",
        "Bavettes, joints, remontées, infiltration, étanchéité air/eau",
    ),
    (
        "drilling_constraints",
        "Perçage / interdictions",
        "Ce qu'on peut/ne peut pas percer ; formulations « interdit »",
    ),
    (
        "dimensions_tolerances",
        "Cotes / tolérances",
        "Faux aplomb, mm/m, cotes chiffrées, tolérances de pose",
    ),
    (
        "load_capacity",
        "Charge / limites structurelles",
        "Poids max, report de charge, entraxe pattes, limites vantail",
    ),
    (
        "material_profile",
        "Matériau / profilé",
        "PVC, alu, hybride, coupe de profil, composition matériau",
    ),
    (
        "glazing",
        "Vitrage / performances",
        "Vitrages, thermique, acoustique, spécifications de vitrage",
    ),
    (
        "parts_references",
        "Références pièces / codes",
        "Codes Txxx, SEC-xxx, nomenclature, liste de composants",
    ),
    (
        "product_range",
        "Gamme / produit",
        "Identification SOLEAL, LUMEAL, Perform, variante, famille",
    ),
    (
        "regulatory",
        "Normes / conformité",
        "DTU, NF EN, obligations normatives, PMR (exigences réglementaires)",
    ),
    (
        "warranty",
        "Garanties",
        "Durée, conditions de garantie, exclusions, couverture",
    ),
    (
        "certification",
        "Certifications / marquage",
        "CE, labels, attestations, conformité produit",
    ),
    (
        "commercial",
        "Contenu commercial",
        "Dépliants, arguments design/performance, contenu marketing",
    ),
    (
        "product_comparison",
        "Comparatif / aide au choix",
        "LUMEAL vs SOLEAL, différences produit, aide à la décision",
    ),
    (
        "troubleshooting",
        "Dépannage / diagnostic",
        "Symptômes client, causes probables, SAV, diagnostic",
    ),
]


def _existing_tables() -> set[str]:
    conn = op.get_bind()
    inspector = inspect(conn)
    return set(inspector.get_table_names(schema="public"))


def upgrade() -> None:
    existing = _existing_tables()

    if "documentcategory" not in existing:
        op.create_table(
            "documentcategory",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("slug", sa.String(length=64), nullable=False),
            sa.Column("label", sa.String(length=200), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("slug"),
        )
        op.create_index("ix_documentcategory_slug", "documentcategory", ["slug"])
        op.create_index("ix_documentcategory_is_active", "documentcategory", ["is_active"])
    else:
        op.execute("CREATE INDEX IF NOT EXISTS ix_documentcategory_slug ON documentcategory (slug)")
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_documentcategory_is_active ON documentcategory (is_active)"
        )

    if "chunkcategoryrelation" not in existing:
        op.create_table(
            "chunkcategoryrelation",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("chunk_id", sa.Integer(), nullable=False),
            sa.Column("category_id", sa.Integer(), nullable=False),
            sa.Column("space_id", sa.Integer(), nullable=False),
            sa.Column("document_id", sa.Integer(), nullable=False),
            sa.Column("page_no", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["category_id"], ["documentcategory.id"]),
            sa.ForeignKeyConstraint(["chunk_id"], ["documentchunk.id"]),
            sa.ForeignKeyConstraint(["space_id"], ["space.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_chunkcategoryrelation_chunk_id", "chunkcategoryrelation", ["chunk_id"])
        op.create_index("ix_chunkcategoryrelation_category_id", "chunkcategoryrelation", ["category_id"])
        op.create_index("ix_chunkcategoryrelation_space_id", "chunkcategoryrelation", ["space_id"])
        op.create_index("ix_chunkcategoryrelation_document_id", "chunkcategoryrelation", ["document_id"])
        op.create_index(
            "uq_chunkcategoryrelation_chunk_category",
            "chunkcategoryrelation",
            ["chunk_id", "category_id"],
            unique=True,
        )
    else:
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunkcategoryrelation_chunk_id "
            "ON chunkcategoryrelation (chunk_id)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunkcategoryrelation_category_id "
            "ON chunkcategoryrelation (category_id)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunkcategoryrelation_space_id "
            "ON chunkcategoryrelation (space_id)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunkcategoryrelation_document_id "
            "ON chunkcategoryrelation (document_id)"
        )
        op.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_chunkcategoryrelation_chunk_category
            ON chunkcategoryrelation (chunk_id, category_id)
            """
        )

    for slug, label, description in SEED_CATEGORIES:
        escaped_desc = description.replace("'", "''")
        escaped_label = label.replace("'", "''")
        op.execute(
            f"""
            INSERT INTO documentcategory (slug, label, description, is_active, created_at, updated_at)
            VALUES ('{slug}', '{escaped_label}', '{escaped_desc}', true, NOW(), NOW())
            ON CONFLICT (slug) DO NOTHING
            """
        )


def downgrade() -> None:
    existing = _existing_tables()
    if "chunkcategoryrelation" in existing:
        op.execute("DROP INDEX IF EXISTS uq_chunkcategoryrelation_chunk_category")
        op.execute("DROP INDEX IF EXISTS ix_chunkcategoryrelation_document_id")
        op.execute("DROP INDEX IF EXISTS ix_chunkcategoryrelation_space_id")
        op.execute("DROP INDEX IF EXISTS ix_chunkcategoryrelation_category_id")
        op.execute("DROP INDEX IF EXISTS ix_chunkcategoryrelation_chunk_id")
        op.drop_table("chunkcategoryrelation")
    if "documentcategory" in existing:
        op.execute("DROP INDEX IF EXISTS ix_documentcategory_is_active")
        op.execute("DROP INDEX IF EXISTS ix_documentcategory_slug")
        op.drop_table("documentcategory")
