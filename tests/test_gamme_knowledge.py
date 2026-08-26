"""Couche de connaissance métier — fiches des gammes commerciales Proferm.

Ces fiches portent le pont entre le vocabulaire des utilisateurs (« Perform 76 ») et
celui des documents fournisseurs (« TROCAL 76 ADVANCED »), pont qui n'existe dans aucun
chunk. Elles sont injectées dans les prompts pour router la recherche.

Les invariants testés ici sont ceux qui font la différence avec le KAG (désactivé) :
l'amorçage ne peut pas écraser une relecture métier, et une fiche ne prétend jamais faire
autorité sur une valeur.
"""
import pytest
from sqlmodel import select

from app.models.gamme_commerciale import STATUT_BROUILLON, STATUT_VALIDE, GammeCommerciale
from app.services.gamme_knowledge_service import (
    build_gamme_knowledge_block,
    list_gammes,
    seed_gammes,
)


@pytest.fixture
def _seeded(db_session):
    seed_gammes(db_session)
    return db_session


class TestAmorcage:
    def test_cree_les_cinq_gammes(self, db_session):
        res = seed_gammes(db_session)
        assert res["created"] == 5
        slugs = {g.slug for g in list_gammes(db_session)}
        assert slugs == {"perform", "hybride", "textural", "lumine", "innoslide"}

    def test_tout_naît_en_brouillon(self, _seeded):
        """Une fiche pré-remplie depuis les documents n'est PAS validée : c'est
        exactement ce que le KAG ne faisait pas."""
        assert all(g.statut == STATUT_BROUILLON for g in list_gammes(_seeded))

    def test_rejouer_lamorcage_necrase_rien(self, _seeded):
        """Une fois relue par le métier, c'est la base qui fait foi, pas le code."""
        perform = _seeded.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "perform")
        ).first()
        perform.description = "RELECTURE MÉTIER"
        perform.statut = STATUT_VALIDE
        _seeded.add(perform)
        _seeded.commit()

        res = seed_gammes(_seeded)
        assert res["created"] == 0 and res["skipped"] == 5

        perform = _seeded.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "perform")
        ).first()
        assert perform.description == "RELECTURE MÉTIER"
        assert perform.statut == STATUT_VALIDE

    def test_overwrite_explicite_reecrit(self, _seeded):
        perform = _seeded.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "perform")
        ).first()
        perform.description = "à écraser"
        _seeded.add(perform)
        _seeded.commit()

        res = seed_gammes(_seeded, overwrite=True)
        assert res["updated"] == 5

        perform = _seeded.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "perform")
        ).first()
        assert "GREENLINE" in (perform.description or "")


class TestPontDeVocabulaire:
    def test_perform_traduit_vers_les_termes_fournisseur(self, _seeded):
        """Le cœur du problème : « Perform » ne se trouve dans aucun document."""
        perform = _seeded.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "perform")
        ).first()
        assert "Perform 76" in perform.alias_utilisateur
        assert any("TROCAL 76 ADVANCED" in t for t in perform.termes_documentaires)
        assert any("KÖMMERLING" in t or "KOMMERLING" in t for t in perform.termes_documentaires)

    def test_lumine_porte_les_prefixes_technal(self, _seeded):
        lumine = _seeded.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "lumine")
        ).first()
        assert lumine.materiau == "aluminium"
        assert "TGY" in (lumine.discriminants or "")
        assert "TFY" in (lumine.discriminants or "")

    def test_les_univers_sexcluent(self, _seeded):
        """« je demande de l'alu, il répond sur du PVC » : la règle doit être écrite."""
        lumine = _seeded.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "lumine")
        ).first()
        assert "JAMAIS" in (lumine.discriminants or "")


class TestBlocPrompt:
    def test_bloc_vide_si_aucune_fiche(self, db_session):
        """Sans fiche, le prompt reste strictement identique — pas de bloc orphelin.

        On vide explicitement : la session de test n'est pas isolée entre tests, et un
        amorçage d'un test voisin suffirait à masquer la régression.
        """
        for g in db_session.exec(select(GammeCommerciale)).all():
            db_session.delete(g)
        db_session.commit()
        assert build_gamme_knowledge_block(db_session) == ""

    def test_bloc_contient_les_deux_vocabulaires(self, _seeded):
        bloc = build_gamme_knowledge_block(_seeded)
        assert "Dit par l'utilisateur" in bloc
        assert "À CHERCHER dans les documents" in bloc

    def test_bloc_interdit_de_faire_autorite_sur_une_valeur(self, _seeded):
        """Garde-fou central : la fiche route, le document tranche."""
        bloc = build_gamme_knowledge_block(_seeded)
        assert "ne font JAMAIS autorité" in bloc

    def test_bloc_rendu_en_entier_meme_en_brouillon(self, _seeded):
        """Pour écarter une référence Technal quand on cherche du PVC, le modèle doit
        connaître la règle Technal : l'injection sélective casserait la discrimination."""
        bloc = build_gamme_knowledge_block(_seeded)
        for nom in ("PERFORM", "HYBRIDE", "TEXTURAL", "LUMINE", "INNOSLIDE"):
            assert nom in bloc

    def test_only_valid_filtre_les_brouillons(self, _seeded):
        assert build_gamme_knowledge_block(_seeded, only_valid=True) == ""

    def test_brouillon_signale_dans_le_bloc(self, _seeded):
        assert "brouillon" in build_gamme_knowledge_block(_seeded)


class TestEndpoints:
    """Les tests ci-dessus ne touchaient que le service : c'est par là que sont passés
    un helper JS inexistant et un risque d'ordre de routes non détectés."""

    def test_liste_vide_sans_fiche(self, client, admin_headers, db_session):
        for g in db_session.exec(select(GammeCommerciale)).all():
            db_session.delete(g)
        db_session.commit()
        r = client.get("/api/admin/gammes", headers=admin_headers)
        assert r.status_code == 200
        assert r.json() == []

    def test_seed_puis_liste(self, client, admin_headers, db_session):
        for g in db_session.exec(select(GammeCommerciale)).all():
            db_session.delete(g)
        db_session.commit()

        r = client.post("/api/admin/gammes/seed", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert r.json()["created"] == 5

        r = client.get("/api/admin/gammes", headers=admin_headers)
        fiches = r.json()
        assert len(fiches) == 5
        perform = next(f for f in fiches if f["slug"] == "perform")
        # Le front lit ces clés en .join(", ") : une chaîne au lieu d'une liste casse l'écran.
        assert isinstance(perform["alias_utilisateur"], list)
        assert isinstance(perform["termes_documentaires"], list)
        assert isinstance(perform["familles"], list)

    def test_preview_prompt_nest_pas_avalee_par_la_route_id(self, client, admin_headers, _seeded):
        """« preview-prompt » est déclarée après « /{gamme_id} » : si le routage la
        capturait, l'aperçu renverrait un 422 sur un id non entier."""
        r = client.get("/api/admin/gammes/preview-prompt", headers=admin_headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["chars"] > 0 and body["tokens_estimes"] > 0
        assert "PERFORM" in body["block"]

    def test_maj_dune_fiche(self, client, admin_headers, _seeded, db_session):
        perform = db_session.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "perform")
        ).first()
        r = client.put(
            f"/api/admin/gammes/{perform.id}",
            headers=admin_headers,
            json={
                "description": "relu par le métier",
                "alias_utilisateur": ["Perform", "Perform 76"],
                "statut": STATUT_VALIDE,
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["statut"] == STATUT_VALIDE

        db_session.expire_all()
        perform = db_session.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "perform")
        ).first()
        assert perform.description == "relu par le métier"
        assert perform.alias_utilisateur == ["Perform", "Perform 76"]

    def test_lecteur_ne_peut_pas_editer(self, client, lecteur_headers, _seeded, db_session):
        perform = db_session.exec(
            select(GammeCommerciale).where(GammeCommerciale.slug == "perform")
        ).first()
        r = client.put(
            f"/api/admin/gammes/{perform.id}",
            headers=lecteur_headers,
            json={"description": "pas censé passer"},
        )
        assert r.status_code in (401, 403)

    def test_fiche_inconnue_renvoie_404(self, client, admin_headers, _seeded):
        r = client.put(
            "/api/admin/gammes/999999", headers=admin_headers, json={"description": "x"}
        )
        assert r.status_code == 404
