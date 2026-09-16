"""Le badge de source doit ouvrir le PDF sur la page qui porte la réponse.

Incident du 2026-09-16 : « huile a metre sur les ferrure 1 fois par an cest le client ou
nous ». Le retriever avait classé la page 10 en tête et le document entier était packé,
mais le modèle a émis ``<sources>`` SANS pages. Le repli prenait alors le premier élément
de ``matched_pages``, rangé par numéro croissant — donc la page 1, la couverture.

Le classement du retriever sait quelle page porte la réponse : c'est lui qui doit servir de
repli, pas l'ordre d'affichage.
"""
from app.services.context_packer_service import build_document_sources


def _doc(**kw):
    base = {
        "index": 1,
        "document_id": 390,
        "document_title": "Notice Eneo CC",
        "pages": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "matched_pages": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "best_page": 10,
        "full_document": True,
        "score": 0.04,
        "has_source_file": True,
    }
    base.update(kw)
    return base


class TestPageDAtterrissage:
    def test_sans_pages_du_modele_on_atterrit_sur_la_mieux_notee(self):
        """Le cas de l'incident : bloc <sources> sans pages sur un document complet."""
        (source,) = build_document_sources([_doc()], {1: []})
        assert source["page_no"] == 10

    def test_les_pages_du_modele_priment_sur_le_classement(self):
        (source,) = build_document_sources([_doc()], {1: [7]})
        assert source["page_no"] == 7
        assert source["used_pages"] == [7]

    def test_sans_page_mieux_notee_on_retombe_sur_les_pages_matchees(self):
        (source,) = build_document_sources([_doc(best_page=None)], {1: []})
        assert source["page_no"] == 1

    def test_une_page_mieux_notee_hors_du_pack_est_ignoree(self):
        """``best_page`` est calculée sur les pages RÉELLEMENT packées."""
        d = _doc(pages=[4, 5, 6], matched_pages=[4, 5, 6], best_page=6)
        (source,) = build_document_sources([d], {1: []})
        assert source["page_no"] == 6

    def test_l_etendue_affichee_reste_celle_du_document(self):
        """La page d'atterrissage ne doit pas rétrécir le libellé « pages 1-10 »."""
        (source,) = build_document_sources([_doc()], {1: []})
        assert source["page_start"] == 1 and source["page_end"] == 10
        assert "1-10" in source["excerpt"]

    def test_un_document_sans_aucune_page_n_atterrit_nulle_part(self):
        d = _doc(pages=[], matched_pages=[], best_page=None)
        (source,) = build_document_sources([d], {1: []})
        assert source["page_no"] is None


class TestPackerRenseigneLaMeilleurePage:
    def test_la_cle_best_page_est_produite_par_le_packer(self):
        """Garde-fou de contrat : le packer doit poser la clé que le badge consomme."""
        import inspect

        from app.services import context_packer_service

        source = inspect.getsource(context_packer_service.build_cag_context)
        assert '"best_page"' in source
