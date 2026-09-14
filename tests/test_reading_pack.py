"""Pack de LECTURE — le lecteur ne reçoit que des pages LUES en image.

La garantie centrale, vérifiée ici : **le texte indexé n'entre JAMAIS dans le contexte du
générateur quand la page a pu être vue.** Sur ce corpus, ce texte est soit une transcription
vision sans aucune cote (le dossier Perform76 n'a pas une seule couche texte sur 26 pages),
soit une reconstruction de tableau qui fabrique des associations fausses. Il sert à trouver
la page (BM25), pas à répondre.
"""
from __future__ import annotations

import uuid

import pytest
from sqlmodel import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.library import Library
from app.services.page_reader_service import PageReading
from app.services.reading_pack_service import (
    PageRead,
    build_reading_pack,
    render_page_block,
    select_pages_to_read,
)
from tests.conftest import create_test_user

SYS = "SYSTEME"
TEXTE_INDEXE = "TRANSCRIPTION_VISION_SANS_COTE"


# ---------------------------------------------------------------------------
# Choix des pages à lire
# ---------------------------------------------------------------------------


def test_les_documents_sont_servis_a_tour_de_role():
    """Un document élu en complément doit TOUJOURS obtenir au moins une page lue.

    Servir « le premier document d'abord » rendait les co-élus muets — c'est ce que le
    plancher d'images par document assurait avant sa suppression le 02/09.
    """
    pages = select_pages_to_read(
        [
            (438, {"matched_pages": {8: 0.9, 5: 0.5, 11: 0.3}}),
            (424, {"matched_pages": {24: 0.4}}),
        ],
        max_readings=4,
        max_per_doc=5,
    )
    assert pages[0] == (438, 8), "la meilleure page du dominant passe en premier"
    assert pages[1] == (424, 24), "le co-élu est servi au tour suivant, pas après tout le reste"
    assert pages == [(438, 8), (424, 24), (438, 5), (438, 11)]


def test_le_plafond_global_coupe_la_liste():
    pages = select_pages_to_read(
        [(1, {"matched_pages": {p: 1.0 / p for p in range(1, 20)}})],
        max_readings=3,
        max_per_doc=10,
    )
    assert len(pages) == 3


def test_le_plafond_par_document_bride_un_gros_document():
    pages = select_pages_to_read(
        [
            (1, {"matched_pages": {p: 1.0 / p for p in range(1, 20)}}),
            (2, {"matched_pages": {7: 0.5}}),
        ],
        max_readings=10,
        max_per_doc=2,
    )
    assert sum(1 for d, _ in pages if d == 1) == 2
    assert (2, 7) in pages


def test_pages_ordonnees_par_score_pas_par_numero():
    pages = select_pages_to_read(
        [(1, {"matched_pages": {2: 0.1, 89: 0.9, 40: 0.5}})], max_readings=3, max_per_doc=3
    )
    assert [p for _, p in pages] == [89, 40, 2]


# ---------------------------------------------------------------------------
# Rendu d'une page
# ---------------------------------------------------------------------------


def _lecture(**kw) -> PageReading:
    base = dict(document_id=438, page_no=8)
    base.update(kw)
    return PageReading(**base)


def test_une_page_qui_repond_dit_la_valeur_et_sa_provenance():
    entry = PageRead(
        document_id=438,
        page_no=8,
        reading=_lecture(
            answer="30",
            survey=[{"repere": "Parclose 2636", "valeurs": [{"valeur": "30", "couleur": "bleu"}]}],
        ),
    )
    bloc = render_page_block(entry, needle="2636")
    assert "LUE EN IMAGE" in bloc
    assert "→ 30" in bloc
    assert "Parclose 2636" in bloc


def test_une_absence_ne_propose_aucune_valeur():
    """Les pages hors sujet le disent : c'est ce qui empêche le modèle d'aller chercher une
    valeur sur la mauvaise page. Mais l'absence de RÉPONSE ne vide pas la page."""
    bloc = render_page_block(PageRead(document_id=438, page_no=5, reading=_lecture(absent=True)))
    assert "n'y a pas trouvé ce qui est demandé" in bloc
    assert "→" not in bloc


def test_une_page_sans_reponse_montre_quand_meme_son_contenu():
    """LA régression du 13/09 (« hauteur de poignée pour un ouvrant de 700 mm ») : la page
    qui porte le tableau de correspondance concluait « absent », parce que 700 n'est le
    repère de rien — et le pack ne montrait alors PLUS RIEN de cette page. Le croisement de
    plage est le travail du lecteur principal ; encore faut-il lui laisser le tableau."""
    tableau = "| Hauteur ouvrant | Axe poignée |\n| 601 mm | 900 mm | 220 mm |"
    bloc = render_page_block(
        PageRead(document_id=438, page_no=6, reading=_lecture(absent=True, content=tableau))
    )
    assert "601 mm | 900 mm | 220 mm" in bloc
    assert "Contenu de la page, restitué depuis l'image" in bloc


def test_une_ambiguite_ne_propose_aucune_valeur():
    bloc = render_page_block(
        PageRead(document_id=1, page_no=2, reading=_lecture(ambiguous=True, answer=""))
    )
    assert "la lecture ne tranche pas" in bloc


def test_le_texte_imprime_est_joint_mais_annonce_comme_non_interprete():
    """La couche texte NATIVE accompagne la lecture pour vérifier l'orthographe d'une
    référence (TGY3702 ≠ TGY3704), jamais pour en déduire une valeur."""
    entry = PageRead(
        document_id=400,
        page_no=172,
        reading=_lecture(document_id=400, page_no=172, answer="TGY3704"),
        native_text="TGA3817 Cale de vitrage\nTGY3600 Equerre 11x28",
    )
    bloc = render_page_block(entry)
    assert "TGY3600 Equerre 11x28" in bloc
    assert "pas à déduire une valeur" in bloc


def test_sans_pdf_le_texte_indexe_est_rendu_mais_marque_comme_non_verifie():
    """Seul cas où le texte indexé apparaît : la page n'a PAS pu être vue."""
    bloc = render_page_block(
        PageRead(document_id=9, page_no=3, reading=None, fallback_text=TEXTE_INDEXE)
    )
    assert "PAGE NON LUE" in bloc
    assert "NON vérifié" in bloc
    assert TEXTE_INDEXE in bloc


# ---------------------------------------------------------------------------
# Pack complet
# ---------------------------------------------------------------------------


def _doc_avec_texte_indexe(session: Session, tmp_path, *, titre="Dossier technique") -> Document:
    user = create_test_user(session, "responsable")
    lib = Library(name="Lib", user_id=user.id, is_global=False)
    session.add(lib)
    session.commit()
    session.refresh(lib)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 factice")

    doc = Document(
        title=titre,
        document_type="written",
        processing_status="completed",
        library_id=lib.id,
        user_id=user.id,
        source_file_path=str(pdf),
        proferm_gammes=["Perform 76"],
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    session.add(
        DocumentChunk(
            document_id=doc.id,
            chunk_index=8,
            content=TEXTE_INDEXE,
            text=TEXTE_INDEXE,
            is_leaf=True,
            hierarchy_level=0,
            node_id=f"n_{uuid.uuid4().hex[:8]}",
            start_char=0,
            end_char=len(TEXTE_INDEXE),
            metadata_json={"page_no": 8, "content_type": "page_raw_enriched"},
        )
    )
    session.commit()
    from app.services.context_packer_service import invalidate_document_fulltext_cache

    invalidate_document_fulltext_cache(doc.id)
    return doc


@pytest.mark.asyncio
async def test_le_texte_indexe_nentre_jamais_dans_le_pack(db_session: Session, tmp_path, monkeypatch):
    """LA garantie du module. La page est lue ; sa transcription reste hors du contexte."""
    doc = _doc_avec_texte_indexe(db_session, tmp_path)

    async def _fausse_lecture(**kwargs):
        return PageReading(
            document_id=kwargs["document_id"],
            page_no=kwargs["page_no"],
            answer="30",
            survey=[{"repere": "Parclose 2636", "valeurs": [{"valeur": "30", "couleur": "bleu"}]}],
        )

    monkeypatch.setattr(
        "app.services.reading_pack_service.read_page_image", _fausse_lecture
    )

    pack = await build_reading_pack(
        db_session,
        [{"document_id": doc.id, "document_title": doc.title, "page_no": 8, "score": 0.9}],
        question="épaisseur de vitrage parclose 2636",
        system_prompt=SYS,
        needle="2636",
    )

    assert TEXTE_INDEXE not in pack["content"], "la transcription vision a fui dans le pack"
    assert "→ 30" in pack["content"]
    assert "Parclose 2636" in pack["content"]
    assert "Perform 76" in pack["content"], "l'en-tête de gamme reste (anti-croisement)"
    assert pack["cag_documents"][0]["document_id"] == doc.id
    assert pack["cag_documents"][0]["pages"] == [8]
    assert pack["reading_trace"]["pages_answered"] == 1


@pytest.mark.asyncio
async def test_une_page_illisible_retombe_sur_le_texte_indexe_marque(
    db_session: Session, tmp_path, monkeypatch
):
    """PDF absent : on ne peut pas VOIR la page. Le texte indexé revient, mais annoncé."""
    doc = _doc_avec_texte_indexe(db_session, tmp_path)
    doc.source_file_path = str(tmp_path / "inexistant.pdf")
    db_session.add(doc)
    db_session.commit()

    pack = await build_reading_pack(
        db_session,
        [{"document_id": doc.id, "document_title": doc.title, "page_no": 8, "score": 0.9}],
        question="épaisseur de vitrage parclose 2636",
        system_prompt=SYS,
    )

    assert "PAGE NON LUE" in pack["content"]
    assert TEXTE_INDEXE in pack["content"]
    assert pack["reading_trace"]["pages_failed"] == 1


@pytest.mark.asyncio
async def test_pack_sans_passage_reste_exploitable(db_session: Session):
    pack = await build_reading_pack(
        db_session, [], question="q", system_prompt=SYS
    )
    assert pack["cag_documents"] == []
    assert pack["cag_document_blocks"] == []
    assert "Aucune page trouvée" in pack["content"]


@pytest.mark.asyncio
async def test_les_pages_sont_lues_en_parallele(db_session: Session, tmp_path, monkeypatch):
    """Le parallélisme est ce qui rend la lecture systématique payable : mesuré 6 pages en
    8,5 s de mur pour 26,4 s d'appels cumulés. En série, le tour serait inutilisable."""
    import asyncio

    doc = _doc_avec_texte_indexe(db_session, tmp_path)
    en_vol = {"max": 0, "courant": 0}

    async def _lecture_lente(**kwargs):
        en_vol["courant"] += 1
        en_vol["max"] = max(en_vol["max"], en_vol["courant"])
        await asyncio.sleep(0.05)
        en_vol["courant"] -= 1
        return PageReading(
            document_id=kwargs["document_id"], page_no=kwargs["page_no"], answer="ok"
        )

    monkeypatch.setattr("app.services.reading_pack_service.read_page_image", _lecture_lente)

    passages = [
        {"document_id": doc.id, "document_title": doc.title, "page_no": p, "score": 1.0 / p}
        for p in (8, 5, 11, 17)
    ]
    await build_reading_pack(
        db_session, passages, question="q", system_prompt=SYS, max_readings=4
    )
    assert en_vol["max"] >= 2, "les lectures doivent partir ensemble, pas l'une après l'autre"


# ---------------------------------------------------------------------------
# Fiabilité — deux pages peuvent répondre, elles ne se valent pas
# ---------------------------------------------------------------------------
def test_une_absence_ne_porte_aucune_mention_de_fiabilite():
    bloc = render_page_block(PageRead(document_id=1, page_no=2, reading=_lecture(absent=True)))
    assert "fiabilité" not in bloc


def test_le_releve_cede_la_place_au_contenu_restitue():
    """Sur une page de tableau, le relevé plat ferait doublon — et c'est lui qui aplatissait
    les lignes. Le contenu restitué prime ; le relevé reste utile aux planches cotées."""
    survey = [{"repere": "Parclose 2636", "valeurs": [{"valeur": "30", "couleur": "bleu"}]}]
    avec = render_page_block(
        PageRead(document_id=1, page_no=1, reading=_lecture(answer="x", content="| a | b |", survey=survey))
    )
    sans = render_page_block(
        PageRead(document_id=1, page_no=1, reading=_lecture(answer="x", survey=survey))
    )
    assert "repères lus sur la page" not in avec
    assert "repères lus sur la page" in sans
