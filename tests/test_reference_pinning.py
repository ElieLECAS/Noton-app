"""
Épinglage par référence sans graphe (reference_pinning_service).

Remplace l'ancien chemin KAG qui exigeait que le code ait été extrait comme
``ref_code`` d'une entité par un LLM. Ici c'est du SQL + un classement par densité
de spécification.
"""
from unittest.mock import MagicMock

import pytest

from app.services.reference_codes import code_in_text, spec_density
from app.services.reference_pinning_service import (
    build_pinned_reference_block,
    select_authority_chunks,
)


def _row(chunk_id, content, title="Notice Perform 70", page_no=12):
    row = MagicMock()
    row.id = chunk_id
    row.content = content
    row.document_title = title
    row.page_no = page_no
    return row


def _session(rows):
    session = MagicMock()
    session.execute.return_value.all.return_value = rows
    return session


class TestFrontieresDeCode:
    def test_code_exact_trouve(self):
        assert code_in_text("6111", "le profil 6111 mesure 70 mm")

    def test_pas_de_faux_positif_sur_prefixe(self):
        """« 6111 » ne doit PAS matcher « 61110 » — le bug du matching sans frontières."""
        assert not code_in_text("6111", "reference 61110")
        assert not code_in_text("6111", "x6111y")

    def test_insensible_a_la_casse(self):
        assert code_in_text("tgy3702", "Profil TGY3702 en aluminium")

    def test_ponctuation_est_une_frontiere(self):
        assert code_in_text("6111", "profil 6111, largeur 70 mm")
        assert code_in_text("6111", "(6111)")


class TestDensiteSpecification:
    def test_compte_les_valeurs_unitaires(self):
        assert spec_density("largeur 70 mm, hauteur 1,5 m, couple 12 daN") == 3

    def test_ignore_les_nombres_nus(self):
        assert spec_density("reference 6111 et 4003") == 0


class TestSelectionAutorite:
    def test_privilegie_la_densite_de_specification(self):
        rows = [
            _row(1, "Le profil 6111 est disponible au catalogue."),
            _row(2, "Profil 6111 : largeur 70 mm, hauteur 45 mm, inertie 12 cm."),
        ]
        picked = select_authority_chunks(_session(rows), [1], "6111")
        assert len(picked) == 1
        assert picked[0]["chunk_id"] == 2

    def test_a_densite_egale_privilegie_le_plus_court(self):
        rows = [
            _row(1, "Profil 6111 largeur 70 mm. " + "Texte de remplissage. " * 20),
            _row(2, "Profil 6111 largeur 70 mm."),
        ]
        picked = select_authority_chunks(_session(rows), [1], "6111")
        assert picked[0]["chunk_id"] == 2

    def test_ecarte_les_faux_positifs_du_like_sql(self):
        """Le ILIKE SQL est large ; les frontières sont vérifiées en Python."""
        rows = [_row(1, "reference 61110 largeur 70 mm")]
        assert select_authority_chunks(_session(rows), [1], "6111") == []

    def test_ecarte_les_chunks_trop_longs(self):
        rows = [_row(1, "Profil 6111 largeur 70 mm. " + "x" * 5000)]
        assert select_authority_chunks(_session(rows), [1], "6111") == []

    def test_sans_document_ni_code(self):
        assert select_authority_chunks(_session([]), [], "6111") == []
        assert select_authority_chunks(_session([]), [1], "") == []

    def test_page_no_illisible_ne_casse_pas(self):
        row = _row(1, "Profil 6111 largeur 70 mm", page_no="n/a")
        picked = select_authority_chunks(_session([row]), [1], "6111")
        assert picked[0]["page_no"] is None


class TestBlocEpingle:
    def test_bloc_sourceé_et_verbatim(self):
        rows = [_row(7, "Profil 6111 : largeur 70 mm.", title="Catalogue Perform", page_no=34)]
        block, pinned = build_pinned_reference_block(_session(rows), [1], ["6111"])

        assert pinned == ["6111"]
        assert "EXTRAIT DE RÉFÉRENCE — 6111" in block
        assert "Catalogue Perform" in block
        assert "p.34" in block
        assert "« Profil 6111 : largeur 70 mm. »" in block

    def test_plafond_de_codes(self):
        rows = [_row(1, "Profil 6111 largeur 70 mm")]
        session = _session(rows)
        _, pinned = build_pinned_reference_block(
            session, [1], ["6111", "6112", "6113"], max_codes=2
        )
        # Le même row est renvoyé pour chaque requête mockée : on vérifie le PLAFOND.
        assert len(pinned) <= 2

    def test_aucun_code_trouve(self):
        block, pinned = build_pinned_reference_block(_session([]), [1], ["9999"])
        assert block == ""
        assert pinned == []

    def test_erreur_sql_est_avalee_par_code(self):
        session = MagicMock()
        session.execute.side_effect = RuntimeError("db down")
        block, pinned = build_pinned_reference_block(session, [1], ["6111"])
        assert block == ""
        assert pinned == []
