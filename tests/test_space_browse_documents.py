"""
Modale de consultation des documents d'un espace.

Remplace l'ancien parcours par catégories : la modale liste les documents de l'espace,
et la recherche par mot-clé porte soit sur tout l'espace, soit sur un seul document.
Ces tests couvrent la portée documentaire, qui est la seule vraie nouveauté côté
service — le rendu page PDF + texte extrait est inchangé (cf. test_space_source_page).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.lexical_search_service import (
    list_space_documents_overview,
    search_space_pages,
)


def _rows_to_session(rows):
    session = MagicMock()
    session.execute.return_value.all.return_value = rows
    return session


def test_documents_overview_maps_rows_and_defaults():
    """Un document sans titre reste listable, et le compte de pages est un entier."""
    session = _rows_to_session(
        [
            (7, "Dépliant INNOSLIDE", "Technal", True, 12),
            (9, None, None, False, None),
        ]
    )

    items = list_space_documents_overview(session, space_id=3)

    assert items == [
        {
            "document_id": 7,
            "title": "Dépliant INNOSLIDE",
            "source": "Technal",
            "has_source_file": True,
            "page_count": 12,
        },
        {
            "document_id": 9,
            "title": "Document sans titre",
            "source": "",
            "has_source_file": False,
            "page_count": 0,
        },
    ]


def test_search_scoped_to_document_passes_the_filter():
    """La portée « ce document » doit atteindre la requête SQL, pas être filtrée après."""
    with (
        patch(
            "app.services.lexical_search_service.get_space_document_ids",
            return_value=[7, 9],
        ),
        patch(
            "app.services.lexical_search_service._list_space_pages",
            return_value=[],
        ) as list_pages,
    ):
        search_space_pages(MagicMock(), space_id=3, query="crémone", document_id=7)

    assert list_pages.call_args.kwargs["scope_document_id"] == 7


def test_search_without_scope_covers_the_whole_space():
    with (
        patch(
            "app.services.lexical_search_service.get_space_document_ids",
            return_value=[7, 9],
        ),
        patch(
            "app.services.lexical_search_service._list_space_pages",
            return_value=[],
        ) as list_pages,
    ):
        search_space_pages(MagicMock(), space_id=3, query="crémone")

    assert list_pages.call_args.kwargs["scope_document_id"] is None


def test_search_scoped_to_a_foreign_document_returns_nothing():
    """Un document hors de l'espace ne doit jamais servir de portée de recherche."""
    with (
        patch(
            "app.services.lexical_search_service.get_space_document_ids",
            return_value=[7],
        ),
        patch("app.services.lexical_search_service._list_space_pages") as list_pages,
    ):
        result = search_space_pages(
            MagicMock(), space_id=3, query="crémone", document_id=999
        )

    assert result["page_count"] == 0
    assert result["pages"] == []
    list_pages.assert_not_called()
