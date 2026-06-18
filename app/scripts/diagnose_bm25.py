"""
Diagnostic BM25 — vérifier pourquoi la recherche lexicale retourne 0 résultat.

Usage:
    python -m app.scripts.diagnose_bm25 <space_id> "<query>"
"""
from __future__ import annotations

import sys

from sqlalchemy import text

from app.database import engine


def diagnose_bm25(space_id: int, query: str) -> None:
    print(f"=== Diagnostic BM25 — space_id={space_id} ===")
    print(f"Requête: {query!r}\n")

    with engine.connect() as conn:
        check_col = text("""
            SELECT
                COUNT(*) AS total,
                COUNT(tsv_content) AS with_tsv,
                COUNT(CASE WHEN tsv_content IS NOT NULL THEN 1 END) AS non_null_tsv,
                COUNT(CASE WHEN content IS NULL OR content = '' THEN 1 END) AS empty_content
            FROM documentchunk dc
            INNER JOIN document d ON dc.document_id = d.id
            INNER JOIN document_space ds ON d.id = ds.document_id
            WHERE ds.space_id = :space_id AND dc.is_leaf = true
        """)
        row = conn.execute(check_col, {"space_id": space_id}).fetchone()
        print(
            f"Chunks L1: total={row.total}, avec tsv_content={row.with_tsv}, "
            f"content vide={row.empty_content}"
        )

        check_semantic = text("""
            SELECT COUNT(*) AS semantic_leaves
            FROM documentchunk dc
            INNER JOIN document d ON dc.document_id = d.id
            INNER JOIN document_space ds ON d.id = ds.document_id
            WHERE ds.space_id = :space_id
              AND dc.is_leaf = true
              AND COALESCE(dc.metadata_json->>'content_type', dc.metadata_->>'content_type', '') = 'semantic_leaf'
        """)
        semantic_count = conn.execute(check_semantic, {"space_id": space_id}).scalar()
        print(f"Chunks semantic_leaf: {semantic_count}")

        test_plain = text("SELECT plainto_tsquery('french', :query) AS tsquery")
        tsquery_plain = conn.execute(test_plain, {"query": query}).scalar()
        print(f"\nplainto_tsquery: {tsquery_plain}")

        test_web = text("SELECT websearch_to_tsquery('french', :query) AS tsquery")
        tsquery_web = conn.execute(test_web, {"query": query}).scalar()
        print(f"websearch_to_tsquery: {tsquery_web}")

        count_plain = text("""
            SELECT COUNT(DISTINCT dc.id) AS matches
            FROM documentchunk dc
            INNER JOIN document d ON dc.document_id = d.id
            INNER JOIN document_space ds ON d.id = ds.document_id
            WHERE ds.space_id = :space_id
              AND dc.is_leaf = true
              AND dc.tsv_content @@ plainto_tsquery('french', :query)
        """)
        matches_plain = conn.execute(count_plain, {"space_id": space_id, "query": query}).scalar()
        print(f"\nMatches plainto_tsquery: {matches_plain}")

        count_web = text("""
            SELECT COUNT(DISTINCT dc.id) AS matches
            FROM documentchunk dc
            INNER JOIN document d ON dc.document_id = d.id
            INNER JOIN document_space ds ON d.id = ds.document_id
            WHERE ds.space_id = :space_id
              AND dc.is_leaf = true
              AND COALESCE(dc.metadata_json->>'content_type', dc.metadata_->>'content_type', '') = 'semantic_leaf'
              AND dc.tsv_content @@ websearch_to_tsquery('french', :query)
        """)
        matches_web = conn.execute(count_web, {"space_id": space_id, "query": query}).scalar()
        print(f"Matches websearch_to_tsquery (semantic_leaf): {matches_web}")

        missing_page = text("""
            SELECT COUNT(*) AS cnt
            FROM documentchunk dc
            INNER JOIN document d ON dc.document_id = d.id
            INNER JOIN document_space ds ON d.id = ds.document_id
            WHERE ds.space_id = :space_id
              AND dc.is_leaf = true
              AND COALESCE(dc.metadata_json->>'content_type', dc.metadata_->>'content_type', '') = 'semantic_leaf'
              AND COALESCE(
                  (dc.metadata_json->>'page_no')::int,
                  (dc.metadata_json->>'page_start')::int,
                  (dc.metadata_->>'page_no')::int,
                  (dc.metadata_->>'page_start')::int
              ) IS NULL
        """)
        no_page = conn.execute(missing_page, {"space_id": space_id}).scalar()
        print(f"\nChunks semantic_leaf sans page_no/page_start: {no_page}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python -m app.scripts.diagnose_bm25 <space_id> \"<query>\"")
        sys.exit(1)
    space_id = int(sys.argv[1])
    query = sys.argv[2]
    diagnose_bm25(space_id, query)


if __name__ == "__main__":
    main()
