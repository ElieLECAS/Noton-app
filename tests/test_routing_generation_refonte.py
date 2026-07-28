"""Refonte routage/génération (plan 2026-07-21) — C1/C2/C3/C4/C6.

Couvre :
  - extraction des codes de référence d'un message (C1) ;
  - bloc COUVERTURE : statuts ok/partiel/vide, codes TROUVÉE/ABSENTE (C3) ;
  - prompt unique : sections SUJET/FORMAT/GROUNDING DUR/ANTI-DIGRESSION (C4) ;
  - suppression de la route fiche technique (C2, garde anti-régression) ;
  - chunk pinning : sélection des chunks-autorité par ref_code (C6, DB).
"""
from __future__ import annotations

import uuid

from sqlmodel import Session

from app.services.coverage_service import (
    build_coverage_block,
    coverage_status,
    extract_message_reference_codes,
)


# ---------------------------------------------------------------------------
# C1 — extraction des codes du message
# ---------------------------------------------------------------------------


class TestExtractMessageReferenceCodes:
    def test_relational_question_extracts_pivot_code(self):
        codes = extract_message_reference_codes("LES SEUILS compatibles avec 6111")
        assert codes == ["6111"]

    def test_alnum_reference(self):
        assert extract_message_reference_codes("où trouver la TGY3702 ?") == ["TGY3702"]

    def test_short_numbers_are_dimensions_not_codes(self):
        # « 155 » est une cote : numérique pur < 4 chiffres → exclu.
        assert extract_message_reference_codes("longueur de 155 mm") == []

    def test_years_excluded(self):
        assert extract_message_reference_codes("catalogue 2024 des gammes") == []

    def test_domain_words_excluded(self):
        assert extract_message_reference_codes("fenêtre PVC avec seuil PMR") == []

    def test_dedup_and_multi_sources(self):
        codes = extract_message_reference_codes("le 6111", "profil 6111 et 6127", None)
        assert codes == ["6111", "6127"]

    def test_empty_input(self):
        assert extract_message_reference_codes(None, "") == []


# ---------------------------------------------------------------------------
# C3 — bloc COUVERTURE
# ---------------------------------------------------------------------------


def _passages(n: int, score: float = 0.9, title: str = "DTA 6/16-2335_V5"):
    return [
        {"score": score, "document_title": title, "passage": "…"}
        for _ in range(n)
    ]


class TestBuildCoverageBlock:
    def test_status_ok_when_code_found(self):
        block = build_coverage_block(
            context_text="Le profilé 6111 a une longueur totale de 155 mm.",
            requested_codes=["6111"],
            doc_passages=_passages(3),
        )
        assert "Référence 6111 : TROUVÉE" in block
        assert "Statut couverture : ok" in block
        assert "Passages pertinents trouvés : 3" in block

    def test_status_partiel_and_warning_when_code_missing(self):
        block = build_coverage_block(
            context_text="Le profilé 6110 mesure 135 mm.",
            requested_codes=["6111"],
            doc_passages=_passages(2),
        )
        assert "Référence 6111 : ABSENTE" in block
        assert "référence voisine" in block  # interdiction d'extrapoler depuis 6110
        assert "Statut couverture : partiel" in block

    def test_word_boundary_no_false_positive(self):
        # « 61110 » ne doit PAS valider « 6111 ».
        block = build_coverage_block(
            context_text="référence 61110 uniquement",
            requested_codes=["6111"],
            doc_passages=_passages(1),
        )
        assert "Référence 6111 : ABSENTE" in block

    def test_status_vide_without_passages(self):
        block = build_coverage_block(
            context_text="", requested_codes=[], doc_passages=[]
        )
        assert "Statut couverture : vide" in block
        assert "Passages pertinents trouvés : 0" in block

    def test_pinned_suffix(self):
        block = build_coverage_block(
            context_text="6111 : longueur 155 mm",
            requested_codes=["6111"],
            doc_passages=_passages(1),
            pinned_codes=["6111"],
        )
        assert "chunk de référence épinglé" in block

    def test_document_titles_listed(self):
        block = build_coverage_block(
            context_text="6111",
            requested_codes=[],
            doc_passages=_passages(2, title="Catalogue PROFINE"),
        )
        assert "Catalogue PROFINE" in block

    def test_structured_status(self):
        result = coverage_status(
            context_text="rien ici",
            requested_codes=["6111", "9F67"],
            doc_passages=_passages(1),
        )
        assert result["status"] == "partiel"
        assert result["missing_codes"] == ["6111", "9F67"]


# ---------------------------------------------------------------------------
# C4 — prompt unique / C2 — route fiche supprimée
# ---------------------------------------------------------------------------


class TestUnifiedPromptAndRouting:
    def test_prompt_contains_new_sections(self):
        from app.routers.chat import SPACE_CHAT_SYSTEM_PROMPT as prompt

        assert "### SUJET DEMANDÉ" in prompt
        assert "### FORMAT ADAPTATIF" in prompt
        assert "### GROUNDING DUR" in prompt
        assert "### ANTI-DIGRESSION" in prompt
        # La règle relationnelle est explicite (le bug « seuils compatibles 6111 »).
        assert "le sujet est X" in prompt
        # Le bloc couverture fait foi.
        assert "COUVERTURE" in prompt

    def test_fiche_fast_path_removed_from_chat(self):
        # Garde anti-régression C2 : le routage ne doit plus jamais court-circuiter la
        # compréhension via le regex fiche.
        import inspect

        import app.routers.chat as chat_module

        source = inspect.getsource(chat_module)
        assert "detect_reference_query" not in source
        assert "prepare_fiche_technique" not in source

    def test_no_standalone_route_llm_fallback(self):
        # C1 : plus d'appel decide_retrieval_route autonome dans chat.py (il ne vit que
        # comme nœud interne de la compréhension fusionnée).
        import inspect

        import app.routers.chat as chat_module

        source = inspect.getsource(chat_module)
        assert "import decide_retrieval_route" not in source
        assert "await decide_retrieval_route" not in source


# ---------------------------------------------------------------------------
# C6 — chunk pinning (DB)
# ---------------------------------------------------------------------------


def _setup_space_with_chunks(db_session: Session):
    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk
    from app.models.library import Library
    from app.models.space import Space
    from tests.conftest import create_test_user

    user = create_test_user(db_session, "responsable")
    library = Library(name="Lib pin", user_id=user.id, is_global=False)
    db_session.add(library)
    db_session.commit()
    db_session.refresh(library)

    space = Space(name=f"Pin {uuid.uuid4().hex[:8]}", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    doc = Document(
        title="DTA test pinning",
        document_type="written",
        processing_status="completed",
        library_id=library.id,
        user_id=user.id,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    def make_chunk(idx: int, content: str) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=idx,
            content=content,
            text=content,
            is_leaf=True,
            hierarchy_level=0,
            node_id=f"n_{uuid.uuid4().hex[:8]}",
            start_char=0,
            end_char=len(content),
            metadata_json={"page_no": idx + 1},
        )
        db_session.add(chunk)
        db_session.commit()
        db_session.refresh(chunk)
        return chunk

    return user, space, doc, make_chunk


class TestEpinglageParReference:
    """L'épinglage passe désormais par reference_pinning_service (SQL, sans graphe).

    Les tests détaillés — frontières de code, densité de spécification, classement —
    sont dans tests/test_reference_pinning.py. On vérifie ici l'intégration réelle en
    base : le chunk le plus « spécifiant » est retenu et le bloc est verbatim + sourcé.
    """

    def test_chunk_avec_cotes_prime_sur_simple_mention(self, db_session: Session):
        from app.services.reference_pinning_service import select_authority_chunks

        user, space, doc, make_chunk = _setup_space_with_chunks(db_session)
        spec_chunk = make_chunk(0, "Profilé TST6111 : longueur totale de 155 mm.")
        make_chunk(1, "Le TST6111 est cité dans la nomenclature générale du chapitre.")

        picked = select_authority_chunks(db_session, [doc.id], "TST6111")

        assert picked, "au moins un chunk-autorité attendu"
        assert picked[0]["chunk_id"] == spec_chunk.id
        assert "155 mm" in picked[0]["content"]

    def test_code_inconnu_rend_vide(self, db_session: Session):
        from app.services.reference_pinning_service import select_authority_chunks

        user, space, doc, make_chunk = _setup_space_with_chunks(db_session)
        make_chunk(0, "Profilé TST6111 : longueur totale de 155 mm.")

        assert select_authority_chunks(db_session, [doc.id], "ZZZ9999") == []

    def test_bloc_verbatim_et_source(self, db_session: Session):
        from app.services.reference_pinning_service import build_pinned_reference_block

        user, space, doc, make_chunk = _setup_space_with_chunks(db_session)
        make_chunk(0, "Seuil TST9F67 : épaisseur 20 mm, compatible dormant TST6111.")

        block, pinned = build_pinned_reference_block(db_session, [doc.id], ["TST9F67"])

        assert pinned == ["TST9F67"]
        assert "EXTRAIT DE RÉFÉRENCE — TST9F67" in block
        assert "épaisseur 20 mm" in block
