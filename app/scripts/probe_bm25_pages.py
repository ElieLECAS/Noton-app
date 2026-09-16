"""Sonde BM25 — diagnostiquer le canal lexical page par page, sans passer par le chat.

Pourquoi une sonde dédiée : juger BM25 depuis le chat est indirect. Le 2026-09-16, une
réponse parfaitement juste a été produite alors que BM25 désignait la mauvaise page — le
document tenait entier dans le budget de packing, et ColPali avait joint la bonne image.
Sur un catalogue de 200 pages, aucun de ces deux filets n'existe.

Ce que la sonde mesure, par question :
  * le palier réellement emprunté (AND strict, OU thésaurus, OU websearch) et la requête
    effectivement envoyée à Postgres — c'est là que les mots vides manquants se voient ;
  * le rang de la page attendue dans le classement BM25 ;
  * la couverture des termes de la question dans le corpus INDEXÉ (chunks) et dans le
    markdown augmenté de la même page, pour distinguer « BM25 classe mal » de « le mot
    n'existe pas dans ce qui est indexé » ;
  * le classement qu'on obtiendrait si les fragments étaient dérivés du markdown.

Usage (dans le conteneur) ::

    docker compose exec web python -m app.scripts.probe_bm25_pages
    docker compose exec web python -m app.scripts.probe_bm25_pages --jeu tests/fixtures/bm25/roto.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_JEU = "tests/fixtures/bm25/roto_eneo.json"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text)


def _termes(question: str, stopwords: set) -> List[str]:
    return sorted({w for w in _norm(question).split() if len(w) > 2 and w not in stopwords})


def _couverture(texte: str, termes: List[str]) -> int:
    corpus = _norm(texte)
    return sum(1 for t in termes if t in corpus)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jeu", default=DEFAULT_JEU)
    parser.add_argument("--limit", type=int, default=5, help="pages rendues par BM25")
    args = parser.parse_args()

    from sqlmodel import Session

    from app.database import engine
    from app.services.page_markdown_service import load_parsed
    from app.services.page_retrieval_service import (
        _BM25_STOPWORDS,
        retrieve_bm25_pages,
    )

    jeu = json.loads(Path(args.jeu).read_text(encoding="utf-8"))
    cas: List[Dict[str, Any]] = jeu["cas"]
    doc_ids: List[int] = jeu["documents"]

    # Texte indexé par (document, page) — c'est le corpus que BM25 interroge.
    from sqlalchemy import text as sql

    indexe: Dict[Tuple[int, int], str] = {}
    with engine.connect() as c:
        for row in c.execute(
            sql(
                """
                SELECT document_id,
                       COALESCE(metadata_json->>'page_no', metadata_json->>'page_start') p,
                       string_agg(content, ' ') AS contenu
                FROM documentchunk
                WHERE document_id = ANY(:ids) AND is_leaf
                  AND COALESCE(metadata_json->>'content_type','') = 'semantic_leaf'
                GROUP BY 1, 2
                """
            ),
            {"ids": doc_ids},
        ):
            doc_id, page_no, contenu = row[0], row[1], row[2]
            if page_no:
                indexe[(int(doc_id), int(page_no))] = contenu or ""

    # Markdown augmenté par (document, page), quand il existe.
    markdown: Dict[Tuple[int, int], str] = {}
    for did in doc_ids:
        parsed = load_parsed(did)
        if parsed:
            for page in parsed.pages:
                markdown[(did, page.page_pdf)] = page.body

    # Le palier emprunté (AND strict / OU thésaurus / OU websearch) n'est pas retourné :
    # il n'existe que dans les journaux du service. On l'y capture.
    import logging

    class _CaptureurPalier(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.lignes: List[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.lignes.append(record.getMessage())

    capteur = _CaptureurPalier()
    logger_service = logging.getLogger("app.services.page_retrieval_service")
    logger_service.addHandler(capteur)
    logger_service.setLevel(logging.INFO)

    resultats: List[Dict[str, Any]] = []
    with Session(engine) as session:
        for cas_i in cas:
            question = cas_i["question"]
            att_doc, att_page = int(cas_i["document_id"]), int(cas_i["page"])
            termes = _termes(question, _BM25_STOPWORDS)

            capteur.lignes.clear()
            hits = retrieve_bm25_pages(session, doc_ids, question, args.limit)
            palier, requete = "AND strict", question
            for ligne in capteur.lignes:
                m = re.search(r"mode=(\S+) query='([^']*)'", ligne)
                if m:
                    palier, requete = m.group(1), m.group(2)
            classement = [(h.document_id, h.page_no) for h in hits]
            rang = (
                classement.index((att_doc, att_page)) + 1
                if (att_doc, att_page) in classement
                else None
            )

            cov_idx = _couverture(indexe.get((att_doc, att_page), ""), termes)
            cov_md = _couverture(markdown.get((att_doc, att_page), ""), termes)

            # Classement simulé sur le markdown : nombre de termes distincts présents.
            simule = sorted(
                markdown.items(), key=lambda kv: -_couverture(kv[1], termes)
            )
            rang_md = None
            for i, (cle, _) in enumerate(simule, start=1):
                if cle == (att_doc, att_page):
                    rang_md = i
                    break

            resultats.append(
                {
                    "id": cas_i.get("id"),
                    "question": question,
                    "attendu": f"{att_doc}:p{att_page}",
                    "rang_bm25": rang,
                    "top_bm25": [f"{d}:p{p}" for d, p in classement[:3]],
                    "termes": termes,
                    "couverture_indexe": f"{cov_idx}/{len(termes)}",
                    "couverture_markdown": f"{cov_md}/{len(termes)}" if markdown else "—",
                    "rang_markdown_simule": rang_md,
                    "palier": palier,
                    "requete_envoyee": requete,
                }
            )

    # ------------------------------------------------------------------ rapport
    print(f"jeu : {args.jeu} — {len(cas)} question(s), documents {doc_ids}\n")
    largeur = 78
    for r in resultats:
        print("─" * largeur)
        print(f"[{r['id']}] {r['question']}")
        print(f"  attendu        : {r['attendu']}")
        marque = "OK" if r["rang_bm25"] == 1 else ("rang " + str(r["rang_bm25"]) if r["rang_bm25"] else "ABSENT")
        print(f"  BM25           : {marque:10s} top3 = {', '.join(r['top_bm25']) or '(vide)'}")
        print(f"  palier         : {r['palier']}")
        if r["requete_envoyee"] != r["question"]:
            print(f"  requête envoyée: {r['requete_envoyee']}")
        print(f"  couverture     : indexé {r['couverture_indexe']}  ·  markdown {r['couverture_markdown']}")
        print(f"  markdown simulé: rang {r['rang_markdown_simule']}")
        print(f"  termes retenus : {' '.join(r['termes'])}")
    print("─" * largeur)

    n = len(resultats)
    ok = sum(1 for r in resultats if r["rang_bm25"] == 1)
    ok_md = sum(1 for r in resultats if r["rang_markdown_simule"] == 1)
    top3 = sum(1 for r in resultats if r["rang_bm25"] and r["rang_bm25"] <= 3)
    absents = sum(1 for r in resultats if r["rang_bm25"] is None)
    print(f"BM25 actuel        : {ok}/{n} en 1re position, {top3}/{n} dans le top 3, {absents} absente(s)")
    print(f"markdown simulé    : {ok_md}/{n} en 1re position")
    paliers: Dict[str, int] = {}
    for r in resultats:
        paliers[r["palier"]] = paliers.get(r["palier"], 0) + 1
    print("paliers empruntés  : " + ", ".join(f"{k} ×{v}" for k, v in sorted(paliers.items())))
    mots_vides_manques = sorted(
        {t for r in resultats for t in r["termes"] if len(t) <= 4 and t in {"sont", "avec", "quel", "cest"}}
    )
    if mots_vides_manques:
        print("mots vides manquants dans _BM25_STOPWORDS : " + ", ".join(mots_vides_manques))
    return 0


if __name__ == "__main__":
    sys.exit(main())
