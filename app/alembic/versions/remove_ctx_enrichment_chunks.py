"""remove contextual enrichment chunks (la matière lue redevient le texte source)

La couche « chunks contextuels » (content_type = contextual_enrichment) était une
réécriture par LLM de fenêtres de 3 pages. Elle est supprimée le 2026-09-16 :

  * elle RÉÉCRIVAIT le document au lieu de le transcrire (son prompt disait « ta mission
    n'est PAS de recopier ») et entrait dans la matière lue par le générateur, mêlée au
    texte source — le modèle pouvait citer une reformulation comme s'il s'agissait du
    document ;
  * elle ne pesait que 0,7 % du corpus de recherche (87 chunks sur 12 439 feuilles en
    base locale) et n'existait que sur des brochures commerciales, jamais sur les
    catalogues techniques où les questions sont difficiles ;
  * son propre garde-fou (_drop_ungrounded_numbers, qui rejetait les chunks portant un
    nombre absent du texte source) était l'aveu qu'elle fabriquait des valeurs.

Cette migration purge les lignes existantes. Les relations FK (chunkentityrelation,
chunkcategoryrelation, entityentityrelation) n'ont PAS de ON DELETE CASCADE : elles sont
purgées d'abord, sinon la suppression échoue. Chaque table est traitée dans un bloc
tolérant — certaines ont pu être supprimées avec le KAG.

Idempotente : relancée, elle ne trouve plus rien à supprimer.

Revision ID: remove_ctx_enrichment_chunks
Revises: add_gamme_commerciale
"""

from alembic import op

revision = "remove_ctx_enrichment_chunks"
down_revision = "add_gamme_commerciale"
branch_labels = None
depends_on = None

_TARGET = """
    SELECT id FROM documentchunk
    WHERE COALESCE(
        metadata_json->>'content_type',
        metadata_->>'content_type',
        ''
    ) = 'contextual_enrichment'
"""


def upgrade() -> None:
    conn = op.get_bind()

    for table, column in (
        ("chunkentityrelation", "chunk_id"),
        ("chunkcategoryrelation", "chunk_id"),
        ("entityentityrelation", "source_chunk_id"),
    ):
        # Table absente (purgée avec le KAG) → sans objet, on continue.
        op.execute(
            f"""
            DO $$
            BEGIN
                IF to_regclass('public.{table}') IS NOT NULL THEN
                    DELETE FROM {table} WHERE {column} IN ({_TARGET});
                END IF;
            END $$;
            """
        )

    result = conn.exec_driver_sql(
        f"DELETE FROM documentchunk WHERE id IN ({_TARGET})"
    )
    print(f"[migration] {result.rowcount or 0} chunk(s) contextuel(s) supprimé(s)")


def downgrade() -> None:
    # Irréversible par nature : ces chunks étaient du texte GÉNÉRÉ, pas extrait. Les
    # recréer supposerait de relancer un service qui n'existe plus.
    pass
