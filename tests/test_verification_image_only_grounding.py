"""Hard grounding en mode 100 % PNG (CAG_IMAGE_ONLY) — contrôles programmatiques.

Contexte (01/09) : la vérification était désactivée en mode image-only « faute de texte à
confronter », alors que le packer calcule le texte des pages dans les deux modes
(``cag_document_blocks``). Dans ce mode, le modèle ne lit que des images et fabrique le
plus volontiers : une liste de teintes RAL absentes de tout le corpus est partie à
l'utilisateur sans être rattrapée. Ces tests verrouillent les deux briques qui rendent le
contrôle possible ET honnête :

  1. les teintes RAL sont des affirmations vérifiables (même écrites « RAL 9016 ») ;
  2. un code que l'utilisateur a lui-même cité n'est jamais compté comme une invention —
     sur une planche CAO la référence n'existe qu'en image, et le texte extrait ne peut
     pas réfuter ce qu'il ne contient pas.
"""
from app.services.response_verification_service import (
    check_grounding,
    check_reference_grounding,
    extract_verifiable_claims,
    unsupported_reference_codes,
)

_REPONSE_RAL_INVENTEE = (
    "Pour la gamme Perform 76, les couleurs disponibles sont : Blanc (RAL 9016), "
    "Ivory (RAL 1015), Gris anthracite (RAL 7016), Gris clair (RAL 7035), Noir (RAL 9005)."
)
_CONTEXTE_SANS_CES_RAL = (
    "[page 3]\nProfilés Perform 76 : intercalaire TGI coloris noir. Finitions RAL au choix.\n"
    "[page 12]\nTeinté dans la masse : Blanc 9016. Plaxé : Chêne doré, Acajou."
)


class TestTeintesRAL:
    def test_les_ral_sont_extraits_meme_avec_une_espace(self):
        claims = extract_verifiable_claims(_REPONSE_RAL_INVENTEE)
        assert "RAL 9016" in claims
        assert "RAL 1015" in claims
        assert len([c for c in claims if c.upper().startswith("RAL")]) == 5

    def test_les_ral_absents_du_contexte_sont_signales(self):
        unsupported = check_grounding(_REPONSE_RAL_INVENTEE, _CONTEXTE_SANS_CES_RAL)
        codes = {c.upper().replace(" ", "") for c in unsupported}
        assert {"RAL1015", "RAL7016", "RAL7035", "RAL9005"} <= codes

    def test_un_ral_present_dans_le_contexte_nest_pas_signale(self):
        """« RAL 9016 » n'apparaît que sous la forme « Blanc 9016 » : ce contrôle est un test
        de présence LITTÉRALE (normalisée), il ne devine pas — c'est le juge LLM qui tranche
        les formulations. On vérifie ici qu'une forme présente n'est pas accusée."""
        ctx = _CONTEXTE_SANS_CES_RAL + "\n[page 18]\nBlanc RAL 9016 satiné."
        unsupported = check_grounding("La teinte standard est le RAL 9016.", ctx)
        assert unsupported == []

    def test_ral_ecrit_sans_espace_dans_le_contexte_est_reconnu(self):
        unsupported = check_grounding("Disponible en RAL 7016.", "[page 2]\nGris anthracite RAL7016.")
        assert unsupported == []


class TestExemptionDesCodesDeLaQuestion:
    _QUESTION = "comment installer la rallonge TGY3704 sur une cremone TGY3702 ?"
    _REPONSE = "Pour installer la rallonge TGY3704 sur la crémone TGY3702 : retirer le capot…"
    _CONTEXTE_MUET = "[page 109]\n(planche CAO : quelques cotes) 25 mm 40 mm"

    def test_sans_exemption_les_codes_de_la_question_seraient_accuses(self):
        assert check_reference_grounding(self._REPONSE, self._CONTEXTE_MUET) == ["TGY3704", "TGY3702"]

    def test_avec_exemption_ils_ne_le_sont_plus(self):
        assert unsupported_reference_codes(
            self._REPONSE, self._CONTEXTE_MUET, question=self._QUESTION
        ) == []

    def test_le_message_utilisateur_brut_compte_aussi(self):
        assert unsupported_reference_codes(
            self._REPONSE, self._CONTEXTE_MUET, user_message="rallonge TGY3704 crémone TGY3702"
        ) == []

    def test_un_code_ni_dans_le_contexte_ni_dans_la_question_reste_signale(self):
        reponse = self._REPONSE + " Prévoir aussi la platine TGY3710."
        assert unsupported_reference_codes(
            reponse, self._CONTEXTE_MUET, question=self._QUESTION
        ) == ["TGY3710"]

    def test_lexemption_est_a_frontieres_de_mot(self):
        """« TGY371 » cité dans la question n'exempte pas « TGY3710 » dans la réponse."""
        assert unsupported_reference_codes(
            "Utiliser la platine TGY3710.", self._CONTEXTE_MUET, question="la TGY371 ?"
        ) == ["TGY3710"]
