"""Garde-fou déterministe des nombres : un chiffre absent du texte source a été lu
sur l'image (hors mandat) ou inventé.

Ce contrôle vivait dans ``contextual_enrichment_service``, supprimé le 2026-09-16 avec
la couche de chunks contextuels. L'extraction SAV en était le second consommateur et en
est désormais le seul propriétaire : les tests suivent la fonction.
"""
import pytest

from app.services.sav_extraction_service import _ungrounded_numbers, build_sav_batches


class TestDetectionNombresNonAncres:
    SOURCE = "Le profil 6111 a une largeur de 70 mm et une hauteur de 45 mm."

    def test_cote_presente_est_ancree(self):
        assert _ungrounded_numbers("Largeur 70 mm.", self.SOURCE) == []

    def test_cote_absente_est_signalee(self):
        assert "1200 mm" in _ungrounded_numbers("Largeur 1200 mm.", self.SOURCE)

    def test_virgule_decimale_toleree(self):
        """« 1,5 m » est ancré par « 1.5 m » dans la source."""
        assert _ungrounded_numbers("Hauteur 1,5 m.", "hauteur 1.5 m maxi") == []

    def test_unite_differente_mais_valeur_ancree(self):
        """On compare la partie numérique : l'unité peut être écrite autrement."""
        assert _ungrounded_numbers("70 mm", "largeur 70mm") == []

    def test_reference_longue_non_ancree_signalee(self):
        assert _ungrounded_numbers("Voir le profil 9999.", self.SOURCE) == ["9999"]

    def test_reference_presente_non_signalee(self):
        assert _ungrounded_numbers("Voir le profil 6111.", self.SOURCE) == []

    def test_petits_entiers_ignores(self):
        """Numéros d'étape / de repère : légitimes sans figurer tels quels."""
        assert _ungrounded_numbers("Étape 3 puis étape 4.", self.SOURCE) == []

    def test_contenu_vide(self):
        assert _ungrounded_numbers("", self.SOURCE) == []


class TestFenetresDePages:
    """La fenêtre glissante a migré KAG → enrichissement → SAV sans changer de règle."""

    def test_recouvrement_par_defaut_des_reglages_sav(self):
        # 8 pages, overlap 1 → stride 7
        assert build_sav_batches(list(range(1, 16))) == [
            list(range(1, 9)),
            list(range(8, 16)),
        ]

    def test_taille_et_recouvrement_explicites(self):
        assert build_sav_batches(
            [1, 2, 3, 4, 5, 6, 7], batch_size=3, overlap=1
        ) == [[1, 2, 3], [3, 4, 5], [5, 6, 7]]

    def test_vide_et_page_unique(self):
        assert build_sav_batches([], batch_size=3, overlap=1) == []
        assert build_sav_batches([4], batch_size=3, overlap=1) == [[4]]

    def test_recouvrement_borne_a_la_taille(self):
        """Un recouvrement >= taille ferait une boucle infinie : il est plafonné."""
        assert build_sav_batches([1, 2, 3, 4], batch_size=2, overlap=5) == [
            [1, 2],
            [2, 3],
            [3, 4],
        ]

    def test_pages_desordonnees_et_doublons(self):
        assert build_sav_batches([3, 1, 2, 2], batch_size=2, overlap=0) == [[1, 2], [3]]


class TestCoucheSupprimee:
    def test_le_service_d_enrichissement_n_existe_plus(self):
        with pytest.raises(ModuleNotFoundError):
            import app.services.contextual_enrichment_service  # noqa: F401
