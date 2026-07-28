"""Tests de la dérivation de santé d'indexation (texte / ColPali sync / KAG)."""

import pytest

from app.config import settings
from app.services.indexing_health_service import (
    _colpali_health,
    _overall_and_mode,
    _text_health,
    build_indexing_health_issues,
)


@pytest.fixture(autouse=True)
def _colpali_on(monkeypatch):
    monkeypatch.setattr(settings, "COLPALI_ENABLED", True)


def _mk_colpali(**kwargs):
    defaults = dict(
        is_pdf=True,
        lancedb_ids=set(),
        all_ids=set(),
        anchor_ids=set(),
        leaf_ids=set(),
    )
    defaults.update(kwargs)
    return _colpali_health(**defaults)


class TestColpaliHealth:
    def test_ok_quand_chaque_anchor_a_ses_patches(self):
        h = _mk_colpali(
            lancedb_ids={1, 2, 3},
            all_ids={1, 2, 3, 10},
            anchor_ids={1, 2, 3},
            leaf_ids={10},
        )
        assert h["status"] == "ok"
        assert h["indexed_pages"] == 3
        assert h["orphan_count"] == 0

    def test_desync_sur_patches_orphelins(self):
        # Le cas du log de prod : patches LanceDB pointant vers des chunks supprimés.
        h = _mk_colpali(
            lancedb_ids={1, 2, 900, 901},
            all_ids={1, 2},
            anchor_ids={1, 2},
            leaf_ids=set(),
        )
        assert h["status"] == "desync"
        assert h["orphan_count"] == 2

    def test_partial_sur_pages_manquantes(self):
        h = _mk_colpali(
            lancedb_ids={1},
            all_ids={1, 2, 3},
            anchor_ids={1, 2, 3},
            leaf_ids=set(),
        )
        assert h["status"] == "partial"
        assert h["missing_count"] == 2

    def test_missing_quand_aucun_patch(self):
        h = _mk_colpali(lancedb_ids=set(), all_ids={1}, anchor_ids={1}, leaf_ids=set())
        assert h["status"] == "missing"

    def test_legacy_mode_sur_feuilles_sans_anchor(self):
        # Ancien pipeline : patches liés aux chunks feuilles, pas d'anchors.
        h = _mk_colpali(
            lancedb_ids={5, 6},
            all_ids={5, 6, 7},
            anchor_ids=set(),
            leaf_ids={5, 6},
        )
        assert h["status"] == "ok"
        assert h["legacy_mode"] is True

    def test_desync_sur_patches_vers_chunks_non_cibles(self):
        # Patches vers des chunks existants mais qui ne sont plus la cible (anchors présents).
        h = _mk_colpali(
            lancedb_ids={10, 11},
            all_ids={1, 2, 10, 11},
            anchor_ids={1, 2},
            leaf_ids={10, 11},
        )
        assert h["status"] == "desync"
        assert h["legacy_count"] == 2

    def test_unknown_si_scan_lancedb_en_echec(self):
        h = _mk_colpali(lancedb_ids=None, all_ids={1}, anchor_ids={1}, leaf_ids=set())
        assert h["status"] == "unknown"

    def test_not_applicable_pour_non_pdf_sans_rien(self):
        h = _mk_colpali(is_pdf=False)
        assert h["status"] == "not_applicable"

    def test_disabled_quand_colpali_off(self, monkeypatch):
        monkeypatch.setattr(settings, "COLPALI_ENABLED", False)
        h = _mk_colpali(lancedb_ids={1}, all_ids={1}, anchor_ids={1}, leaf_ids=set())
        assert h["status"] == "disabled"


class TestTextHealth:
    def test_ok(self):
        h = _text_health({"chunk_count": 10, "leaf_count": 8, "leaves_with_embedding": 8})
        assert h["status"] == "ok"

    def test_partial(self):
        h = _text_health({"chunk_count": 10, "leaf_count": 8, "leaves_with_embedding": 5})
        assert h["status"] == "partial"
        assert h["missing_embeddings"] == 3

    def test_missing_sans_chunks(self):
        h = _text_health({"chunk_count": 0, "leaf_count": 0, "leaves_with_embedding": 0})
        assert h["status"] == "missing"


class TestOverallAndMode:
    def _h(self, text="ok", colpali="ok", enrichment="ok"):
        return (
            {"status": text, "missing_embeddings": 0},
            {"status": colpali},
            {"status": enrichment},
        )

    def test_tout_ok(self):
        t, c, e = self._h()
        assert _overall_and_mode(t, c, e) == ("ok", None)

    def test_texte_casse_impose_full(self):
        t, c, e = self._h(text="missing")
        assert _overall_and_mode(t, c, e) == ("error", "full")

    def test_colpali_desync_suggere_colpali_only(self):
        t, c, e = self._h(colpali="desync")
        assert _overall_and_mode(t, c, e) == ("warning", "colpali_only")

    def test_synthese_manquante_suggere_enrichment_only(self):
        """Remplace l'ancienne suggestion kag_only : ce qui manque après un passage
        text_only, ce sont les chunks contextuels."""
        t, c, e = self._h(enrichment="missing")
        assert _overall_and_mode(t, c, e) == ("warning", "enrichment_only")

    def test_colpali_et_synthese_casses_suggerent_full(self):
        t, c, e = self._h(colpali="missing", enrichment="missing")
        assert _overall_and_mode(t, c, e) == ("warning", "full")

    def test_synthese_desactivee_ne_declenche_rien(self):
        t, c, e = self._h(enrichment="disabled")
        assert _overall_and_mode(t, c, e) == ("ok", None)


class TestIssues:
    def test_issues_desync_mentionne_colpali_only(self, monkeypatch):
        monkeypatch.setattr(settings, "KAG_ENABLED", True)
        health = {
            "text": {"status": "ok"},
            "colpali": {"status": "desync", "orphan_count": 35, "legacy_count": 0},
            "kag": {"status": "ok"},
            "categories_ok": True,
        }
        issues = build_indexing_health_issues(health)
        assert any("colpali_only" in i for i in issues)
        assert any("35" in i for i in issues)
