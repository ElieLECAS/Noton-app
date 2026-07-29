"""
Vue « source citée » : page PDF + tout le texte extrait, sans filtre mot-clé.

Alimente la modale ouverte depuis « Documents consultés » dans le chat.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.lexical_search_service import (
    _contains_any,
    _list_document_pages_all,
    get_space_source_page_detail,
)


def _nav_page(page_no, chunk_count=1):
    """Entrée de navigation telle que la produit _list_document_pages_all."""
    return {
        "document_id": 7,
        "document_title": "Dépliant INNOSLIDE",
        "page_no": page_no,
        "chunk_count": chunk_count,
        "has_source_file": True,
    }


class _Chunk:
    """Stub minimal de DocumentChunk (les `or` des builders excluent MagicMock)."""

    def __init__(self, chunk_id, content, metadata=None, chunk_index=0):
        self.id = chunk_id
        self.content = content
        self.text = None
        self.metadata_json = metadata or {}
        self.metadata_ = {}
        self.chunk_index = chunk_index


def test_contains_any_none_needles_disables_filter():
    """None = pas de filtre : c'est ce qui distingue la vue source de la recherche."""
    assert _contains_any("n'importe quel texte", None) is True
    assert _contains_any("", None) is True
    # Comportement historique préservé pour une vraie liste de mots-clés.
    assert _contains_any("Perform 70", ["perform"]) is True
    assert _contains_any("Perform 70", ["lumine"]) is False
    assert _contains_any("Perform 70", []) is False


def _patched_detail(source_chunks, enrichment_chunks, pages, doc=None, space_docs=(7,)):
    session = MagicMock()
    document = doc if doc is not None else MagicMock(
        id=7, title="Dépliant INNOSLIDE", source_file_path=None
    )
    session.get.return_value = document
    with (
        patch(
            "app.services.lexical_search_service.get_space_document_ids",
            return_value=list(space_docs),
        ),
        patch(
            "app.services.lexical_search_service.load_l1_chunks_for_page",
            return_value=source_chunks,
        ),
        patch(
            "app.services.lexical_search_service.load_enrichment_chunks_for_pages",
            return_value=enrichment_chunks,
        ),
        patch(
            "app.services.lexical_search_service.build_consolidated_page_text",
            return_value="md",
        ),
        patch(
            "app.services.lexical_search_service._list_document_pages_all",
            return_value=pages,
        ),
    ):
        return get_space_source_page_detail(session, space_id=1, document_id=7, page_no=2)


def test_source_page_returns_all_chunks_without_keyword_filter():
    """Aucun chunk ne doit être écarté : la page entière est voulue."""
    payload = _patched_detail(
        source_chunks=[
            _Chunk(1, "LE COULISSANT, AUTREMENT", {"heading": "Titre"}, 6),
            _Chunk(2, "ET SANS EFFORT", {"heading": "Autre"}, 7),
            _Chunk(3, "Texte sans rapport avec un quelconque mot-clé", {}, 8),
        ],
        enrichment_chunks=[_Chunk(9, "Synthèse IA", {"theme": "pose"}, 20)],
        pages=[_nav_page(1, 3), _nav_page(2, 3), _nav_page(4)],
    )
    assert payload is not None
    assert len(payload["chunks"]) == 3
    assert len(payload["enrichment_chunks"]) == 1
    assert payload["page_no"] == 2
    assert payload["query"] == ""
    assert payload["document"]["title"] == "Dépliant INNOSLIDE"


def test_source_page_navigation_follows_document_pages():
    payload = _patched_detail(
        source_chunks=[_Chunk(1, "x", {}, 0)],
        enrichment_chunks=[],
        pages=[_nav_page(1), _nav_page(2), _nav_page(4)],
    )
    nav = payload["navigation"]
    assert nav["total"] == 3
    assert nav["current_index"] == 1
    assert nav["prev"]["page_no"] == 1
    assert nav["next"]["page_no"] == 4


def test_source_page_payload_matches_api_contract():
    """
    Valide le payload via le modèle de réponse, pas seulement le dict : c'est ce qui
    manquait quand les références de navigation sortaient sans `document_title`
    (service correct, mais HTTP 500 à la sérialisation).
    """
    from app.routers.spaces import SpaceSearchPageDetailResponse

    payload = _patched_detail(
        source_chunks=[_Chunk(1, "Texte", {"heading": "T"}, 0)],
        enrichment_chunks=[_Chunk(9, "Synthèse", {"theme": "pose"}, 20)],
        pages=[_nav_page(1), _nav_page(2), _nav_page(4)],
    )
    model = SpaceSearchPageDetailResponse.model_validate(payload)
    assert model.navigation.prev.document_title == "Dépliant INNOSLIDE"
    assert model.navigation.next.document_title == "Dépliant INNOSLIDE"
    assert model.page_no == 2


def test_list_document_pages_all_carries_navigation_fields():
    """Chaque page doit porter les champs exigés par SpaceCategoryPageNavRef."""
    session = MagicMock()
    session.execute.return_value.all.return_value = [(1, 3), (2, 5)]
    pages = _list_document_pages_all(session, 7, "Mon document", True)
    assert pages == [
        {
            "document_id": 7,
            "document_title": "Mon document",
            "page_no": 1,
            "chunk_count": 3,
            "has_source_file": True,
        },
        {
            "document_id": 7,
            "document_title": "Mon document",
            "page_no": 2,
            "chunk_count": 5,
            "has_source_file": True,
        },
    ]


def test_source_page_without_indexed_text_still_returns_detail():
    """
    Page muette (planche d'illustrations retenue par ColPali) : on renvoie quand même
    le détail pour que le rendu PDF reste consultable, contrairement à la vue recherche
    qui répond 404 quand aucun chunk ne correspond.
    """
    payload = _patched_detail(
        source_chunks=[],
        enrichment_chunks=[],
        pages=[_nav_page(1, 2)],
    )
    assert payload is not None
    assert payload["chunks"] == []
    assert payload["enrichment_chunks"] == []
    assert payload["navigation"]["current_index"] is None


def test_source_page_rejects_document_outside_space():
    payload = _patched_detail(
        source_chunks=[_Chunk(1, "x", {}, 0)],
        enrichment_chunks=[],
        pages=[_nav_page(2)],
        space_docs=(99,),
    )
    assert payload is None


def test_source_page_rejects_invalid_page_number():
    session = MagicMock()
    with patch(
        "app.services.lexical_search_service.get_space_document_ids",
        return_value=[7],
    ) as spy:
        assert get_space_source_page_detail(session, 1, 7, 0) is None
        # Court-circuit avant toute requête.
        spy.assert_not_called()


def test_source_page_endpoint_404_when_space_not_found(client, responsable_headers):
    r = client.get("/api/spaces/999999/pages/1/1", headers=responsable_headers)
    assert r.status_code == 404
