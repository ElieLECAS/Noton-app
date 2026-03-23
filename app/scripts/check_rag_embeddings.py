"""
Script de diagnostic pour le RAG : vérifie la dimension pgvector et les embeddings.
Exécuter après une migration de dimension pour s'assurer que les données sont cohérentes.
"""
from __future__ import annotations

from sqlalchemy import text

from app.database import engine
from app.embedding_config import EMBEDDING_DIMENSION


def main() -> None:
    print(f"Dimension attendue (EMBEDDING_DIMENSION): {EMBEDDING_DIMENSION}")
    print("-" * 50)

    with engine.connect() as conn:
        # Vérifier si la table notechunk existe
        result = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'notechunk'
                )
                """
            )
        )
        if not result.scalar():
            print("Table notechunk introuvable.")
            return

        # Dimension de la colonne embedding (via vector_dims si des données existent)
        result = conn.execute(
            text(
                """
                SELECT vector_dims(embedding) AS dim
                FROM notechunk
                WHERE embedding IS NOT NULL
                LIMIT 1
                """
            )
        )
        row = result.fetchone()
        if row and row[0]:
            actual_dim = row[0]
            print(f"Dimension pgvector (notechunk.embedding): {actual_dim}")
            if actual_dim != EMBEDDING_DIMENSION:
                print(
                    f"  ⚠️  INCOHÉRENCE: attendu {EMBEDDING_DIMENSION}, "
                    f"trouvé {actual_dim}. Exécuter les migrations puis réindexer."
                )
            else:
                print(f"  ✓ Cohérent avec EMBEDDING_DIMENSION")
        else:
            print(
                "Dimension: N/A (aucun embedding non NULL). "
                "Réindexer pour peupler les embeddings."
            )

        # Nombre de chunks avec embeddings
        result = conn.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS avec_embedding,
                    COUNT(*) FILTER (WHERE embedding IS NULL) AS sans_embedding,
                    COUNT(*) FILTER (WHERE is_leaf = true AND embedding IS NOT NULL) AS feuilles_indexees
                FROM notechunk
                """
            )
        )
        row = result.fetchone()
        if row:
            avec, sans, feuilles = row[0], row[1], row[2]
            total = avec + sans
            print(f"\nChunks notechunk: {total} total")
            print(f"  - Avec embedding: {avec}")
            print(f"  - Sans embedding: {sans}")
            print(f"  - Feuilles indexées (is_leaf=true + embedding): {feuilles}")
            if feuilles == 0:
                print(
                    "\n  ⚠️  Aucune feuille indexée. La recherche vectorielle retournera 0 résultats."
                )
                print("  → Exécuter: python -m app.scripts.reindex_notes [--project-id ID]")
            elif avec < total:
                print(
                    f"\n  ⚠️  {sans} chunks sans embedding. Réindexer pour compléter."
                )


if __name__ == "__main__":
    main()
