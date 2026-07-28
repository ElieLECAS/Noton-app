"""
Chunks contextuels : vision systématique + garde-fou déterministe des nombres.

La vision est désormais TOUJOURS fournie à l'enrichissement (c'est le seul étage qui
voit une page avec droit de synthétiser). Le garde-fou n'est donc plus une porte
d'entrée mais un contrôle de SORTIE : tout nombre absent du texte transcrit signifie
qu'il a été lu sur une image — hors mandat — et fait rejeter le chunk.
"""
from unittest.mock import patch

import pytest

from app.services.contextual_enrichment_service import (
    EnrichmentChunkItem,
    _coerce_enrichment_response,
    _drop_ungrounded_numbers,
    _ungrounded_numbers,
)


class TestDetectionNombresNonAncres:
    SOURCE = "Le profil 6111 a une largeur de 70 mm et une hauteur de 45 mm."

    def test_cote_presente_est_ancree(self):
        assert _ungrounded_numbers("Largeur 70 mm.", self.SOURCE) == []

    def test_cote_absente_est_signalee(self):
        assert "1200 mm" in _ungrounded_numbers("Largeur 1200 mm.", self.SOURCE)

    def test_virgule_decimale_toleree(self):
        """« 1,5 m » dans la synthèse est ancré par « 1.5 m » dans la source."""
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


class TestRejetDesChunks:
    SOURCE = "Serrer le ressort avec une clé de 4 mm."

    def _item(self, content):
        return EnrichmentChunkItem(
            category_slug="mounting", theme="t", content=content, source_page=1
        )

    def test_chunk_ancre_conserve(self):
        kept, rejected = _drop_ungrounded_numbers(
            [self._item("Utiliser une clé de 4 mm.")], self.SOURCE
        )
        assert len(kept) == 1
        assert rejected == 0

    def test_chunk_avec_cote_inventee_rejete(self):
        kept, rejected = _drop_ungrounded_numbers(
            [self._item("Serrer à 12 daN sur 250 mm.")], self.SOURCE
        )
        assert kept == []
        assert rejected == 1

    def test_tri_partiel(self):
        kept, rejected = _drop_ungrounded_numbers(
            [
                self._item("Clé de 4 mm."),
                self._item("Couple de 35 daN."),
                self._item("Serrage à la clé."),
            ],
            self.SOURCE,
        )
        assert len(kept) == 2
        assert rejected == 1


class TestCoercionAvecGardeFou:
    SOURCE = "Profil 6111, largeur 70 mm."

    def _payload(self, content):
        return {
            "enrichment_chunks": [
                {
                    "category_slug": "specifications",
                    "theme": "Profil 6111",
                    "content": content,
                    "source_page": 1,
                }
            ]
        }

    def test_chunk_ancre_passe(self):
        response = _coerce_enrichment_response(
            self._payload("Le profil 6111 fait 70 mm de large."),
            [1],
            frozenset({"specifications"}),
            source_text=self.SOURCE,
        )
        assert len(response.enrichment_chunks) == 1

    def test_tous_rejetes_leve_une_erreur(self):
        """Erreur explicite plutôt qu'une réponse vide silencieuse : le batch sera loggé."""
        with pytest.raises(ValueError, match="non ancrées"):
            _coerce_enrichment_response(
                self._payload("Le profil fait 1200 mm de large."),
                [1],
                frozenset({"specifications"}),
                source_text=self.SOURCE,
            )

    def test_sans_source_le_garde_fou_ne_tourne_pas(self):
        """Rétrocompatibilité : appelé sans source_text, aucun rejet."""
        response = _coerce_enrichment_response(
            self._payload("Le profil fait 1200 mm de large."),
            [1],
            frozenset({"specifications"}),
        )
        assert len(response.enrichment_chunks) == 1


class TestVisionSystematique:
    def test_porte_visuelle_supprimee(self):
        import app.services.contextual_enrichment_service as ces

        assert not hasattr(ces, "_batch_is_visual")
        assert not hasattr(ces, "_VISUAL_SECTION_TYPES")

    def test_images_rendues_a_300_dpi(self):
        """Le rendu d'enrichissement suit PAGE_EXTRACTION_DPI (300), pas les 150 dpi
        de la génération : une cote fine doit rester lisible."""
        import app.services.contextual_enrichment_service as ces

        with patch(
            "app.services.multimodal_page_service.render_page_png_cached",
            return_value=b"png",
        ) as render:
            ces._render_batch_images("/tmp/x.pdf", [1, 2])

        assert render.call_count == 2
        for call in render.call_args_list:
            assert call.kwargs["dpi"] == 300

    def test_rendu_png_en_echec_est_non_bloquant(self):
        import app.services.contextual_enrichment_service as ces

        with patch(
            "app.services.multimodal_page_service.render_page_png_cached",
            side_effect=RuntimeError("pdf illisible"),
        ):
            assert ces._render_batch_images("/tmp/x.pdf", [1, 2]) == []
