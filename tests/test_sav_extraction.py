"""Extraction d'atomes SAV (lot L1) — fonctions pures, sans DB ni réseau."""
import json

from app.services import sav_extraction_service as svc
from app.services.sav_extraction_service import (
    SavAtom,
    build_pivot_from_atoms,
    build_sav_batches,
    canonicalize_symptom_groups,
    citation_is_verbatim,
    coerce_atoms,
    dedupe_atoms,
    derive_tree_title,
    drop_ungrounded_atoms,
    extract_atoms_for_batch,
    format_batch_text,
    parse_atoms_payload,
    validate_groups,
)


def _atom(**kw):
    base = {
        "symptome": "le volet s'arrête avant le bas",
        "cause": "fin de course basse déréglée",
        "verification": "il s'arrête toujours au même endroit",
        "geste": "reprendre le réglage de fin de course",
        "citation": "Le réglage de fin de course basse se reprend au dos du moteur.",
        "page": 34,
        "produit": "Oximo 40 WF RTS",
        "qui": "poseur",
        "effort": "visuel",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def test_batches_couvrent_toutes_les_pages():
    pages = list(range(1, 21))
    batches = build_sav_batches(pages, batch_size=8, overlap=1)

    vues = {p for batch in batches for p in batch}
    assert vues == set(pages), "aucune page ne doit être oubliée"


def test_batches_se_recouvrent_dune_page():
    batches = build_sav_batches(list(range(1, 17)), batch_size=8, overlap=1)

    assert batches[0] == list(range(1, 9))
    # stride 7 : le batch suivant repart sur la dernière page du précédent
    assert batches[1][0] == batches[0][-1]


def test_batch_unique_si_document_court():
    assert build_sav_batches([1, 2, 3], batch_size=8, overlap=1) == [[1, 2, 3]]


def test_batches_vides_si_aucune_page():
    assert build_sav_batches([], batch_size=8, overlap=1) == []


# ---------------------------------------------------------------------------
# Lecture de la réponse modèle
# ---------------------------------------------------------------------------


def test_reponse_valide_est_lue():
    raw = json.dumps({"atomes": [_atom()]})
    rows, truncated = parse_atoms_payload(raw)

    assert truncated is False
    assert len(rows) == 1


def test_reponse_en_fence_markdown_est_lue():
    raw = "```json\n" + json.dumps({"atomes": [_atom()]}) + "\n```"
    rows, truncated = parse_atoms_payload(raw)

    assert truncated is False
    assert len(rows) == 1


def test_objet_sans_atomes_vaut_rien_a_signaler():
    """« Ces pages ne contiennent rien d'utile » est un résultat normal, pas une panne."""
    rows, truncated = parse_atoms_payload(json.dumps({"commentaire": "rien ici"}))

    assert truncated is False
    assert rows == []


def test_reponse_tronquee_est_signalee_et_non_reparee():
    """Une réparation rendrait un JSON valide mais amputé : les atomes perdus le
    resteraient en silence. On préfère signaler pour re-découper."""
    raw = '{"atomes": [{"symptome": "le volet s\'arrête", "cause": "fin de cou'
    rows, truncated = parse_atoms_payload(raw)

    assert truncated is True
    assert rows is None


# ---------------------------------------------------------------------------
# Validation des atomes
# ---------------------------------------------------------------------------


def test_atome_hors_batch_est_rejete():
    atoms, rejected = coerce_atoms([_atom(page=99)], allowed_pages=[33, 34, 35])

    assert atoms == []
    assert any("99" in motif for motif in rejected)


def test_atome_sans_symptome_est_rejete():
    atoms, rejected = coerce_atoms([_atom(symptome="  ")], allowed_pages=[34])

    assert atoms == []
    assert rejected


def test_valeurs_hors_nomenclature_sont_ramenees_au_defaut():
    atoms, _ = coerce_atoms(
        [_atom(qui="technicien-chef", effort="tres_dur")], allowed_pages=[34]
    )

    assert atoms[0].qui == "poseur"
    assert atoms[0].effort == "visuel"


def test_entree_non_objet_est_ignoree():
    atoms, rejected = coerce_atoms(["pas un objet"], allowed_pages=[34])

    assert atoms == []
    assert rejected


# ---------------------------------------------------------------------------
# Garde-fou des valeurs chiffrées
# ---------------------------------------------------------------------------


def test_cote_absente_du_texte_source_fait_tomber_latome():
    """Une cote lue sur l'image (ou inventée) ne doit pas entrer dans l'arbre."""
    source = "--- PAGE 34 ---\nReprendre le réglage de fin de course au dos du moteur."
    atom = SavAtom.model_validate(_atom(geste="serrer la vis à 12 mm du bord"))

    kept, dropped = drop_ungrounded_atoms([atom], source)

    assert kept == []
    assert len(dropped) == 1


def test_cote_presente_dans_le_texte_source_est_conservee():
    source = "--- PAGE 34 ---\nSerrer la vis à 12 mm du bord de la coulisse."
    atom = SavAtom.model_validate(_atom(geste="serrer la vis à 12 mm du bord"))

    kept, dropped = drop_ungrounded_atoms([atom], source)

    assert len(kept) == 1
    assert dropped == []


# ---------------------------------------------------------------------------
# Dédoublonnage du recouvrement
# ---------------------------------------------------------------------------


def test_atome_vu_deux_fois_par_le_recouvrement_est_fusionne():
    a = SavAtom.model_validate(_atom(page=8, citation="Phrase courte."))
    b = SavAtom.model_validate(_atom(page=8, citation="Phrase nettement plus longue et complète."))

    result = dedupe_atoms([a, b])

    assert len(result) == 1
    assert result[0].citation == "Phrase nettement plus longue et complète."


def test_deux_symptomes_distincts_ne_sont_pas_fusionnes():
    a = SavAtom.model_validate(_atom())
    b = SavAtom.model_validate(_atom(symptome="le volet ne descend pas du tout"))

    assert len(dedupe_atoms([a, b])) == 2


def test_dedoublonnage_ignore_casse_et_accents():
    a = SavAtom.model_validate(_atom(symptome="Le volet s'arrête avant le bas"))
    b = SavAtom.model_validate(_atom(symptome="le volet s'arrete avant le bas"))

    assert len(dedupe_atoms([a, b])) == 1


# ---------------------------------------------------------------------------
# Batch complet (appel modèle simulé)
# ---------------------------------------------------------------------------


def test_pages_muettes_ne_declenchent_aucun_appel(monkeypatch):
    appels = []
    monkeypatch.setattr(
        svc, "_call_extraction_api", lambda *a, **k: appels.append(a) or "{}"
    )

    atoms, report = extract_atoms_for_batch([1, 2], "Notice", {1: "", 2: "   "}, None)

    assert atoms == []
    assert report["status"] == "pages_muettes"
    assert appels == []


def test_batch_tronque_est_redecoupe_en_deux(monkeypatch):
    """Le re-découpage est le garde-fou des gros batches : sans lui, une réponse coupée
    par le plafond de sortie perd des atomes sans que rien ne le signale."""
    pages_text = {p: f"Texte de la page {p}." for p in range(1, 5)}
    vus = []

    def faux_appel(title, batch_pages, page_text, images):
        vus.append(list(batch_pages))
        if len(batch_pages) > 2:
            return '{"atomes": [{"symptome": "coupé au milieu'  # tronqué
        page = batch_pages[0]
        return json.dumps(
            {"atomes": [_atom(page=page, symptome=f"symptome page {page}", geste="")]}
        )

    monkeypatch.setattr(svc, "_call_extraction_api", faux_appel)

    atoms, report = extract_atoms_for_batch([1, 2, 3, 4], "Notice", pages_text, None)

    assert report["status"] == "decoupe"
    assert [1, 2, 3, 4] in vus and [1, 2] in vus and [3, 4] in vus
    assert len(atoms) == 2, "les deux moitiés doivent avoir été relancées"


def test_batch_nominal_retient_les_atomes(monkeypatch):
    pages_text = {34: "Le réglage de fin de course basse se reprend au dos du moteur."}
    monkeypatch.setattr(
        svc,
        "_call_extraction_api",
        lambda *a, **k: json.dumps({"atomes": [_atom(geste="reprendre le réglage")]}),
    )

    atoms, report = extract_atoms_for_batch([34], "Notice", pages_text, None)

    assert len(atoms) == 1
    assert report["status"] == "ok"
    assert report["atoms"] == 1


def test_erreur_dappel_ne_fait_pas_tomber_le_batch(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(svc, "_call_extraction_api", boom)

    atoms, report = extract_atoms_for_batch([1], "Notice", {1: "du texte"}, None)

    assert atoms == []
    assert report["status"] == "erreur"


def test_texte_du_batch_porte_les_numeros_de_page():
    """Le modèle doit citer le rang de page fourni, pas un numéro imprimé lu sur l'image."""
    texte = format_batch_text([7, 8], {7: "contenu sept", 8: "contenu huit"})

    assert "--- PAGE 7 ---" in texte
    assert "--- PAGE 8 ---" in texte


def _children_of(pivot, parent_id):
    return [c for c in pivot["cas"] if parent_id in (c.get("parents") or [])]


# ---------------------------------------------------------------------------
# L2b — B1 : conservation des atomes au regroupement
# ---------------------------------------------------------------------------


def test_groupes_valides_couvrant_tous_les_atomes():
    ok, reason = validate_groups(
        [{"label": "A", "atomes": [0, 1]}, {"label": "B", "atomes": [2]}], 3
    )
    assert ok, reason


def test_groupe_qui_perd_un_atome_est_rejete():
    """Regrouper ne doit JAMAIS perdre un constat : c'est ce qui autorise à faire
    confiance au temps B."""
    ok, reason = validate_groups([{"label": "A", "atomes": [0, 1]}], 3)
    assert not ok
    assert "non group" in reason


def test_atome_dans_deux_groupes_est_rejete():
    ok, reason = validate_groups(
        [{"label": "A", "atomes": [0, 1]}, {"label": "B", "atomes": [1, 2]}], 3
    )
    assert not ok
    assert "deux groupes" in reason


def test_indice_hors_bornes_est_rejete():
    ok, _ = validate_groups([{"label": "A", "atomes": [0, 99]}], 2)
    assert not ok


def test_groupe_sans_label_est_rejete():
    ok, _ = validate_groups([{"label": "  ", "atomes": [0]}], 1)
    assert not ok


def test_regroupement_llm_invalide_retombe_sur_none(monkeypatch):
    monkeypatch.setattr(
        svc,
        "_call_grouping_api",
        lambda listing: json.dumps({"groupes": [{"label": "A", "atomes": [0]}]}),
    )
    atoms = [
        SavAtom.model_validate(_atom(page=1)),
        SavAtom.model_validate(_atom(page=2, symptome="autre")),
    ]

    # 2 atomes, 1 seul groupé -> rejeté -> None (repli sur le regroupement exact)
    assert canonicalize_symptom_groups(atoms) is None


def test_regroupement_llm_valide_est_retenu(monkeypatch):
    monkeypatch.setattr(
        svc,
        "_call_grouping_api",
        lambda listing: json.dumps(
            {"groupes": [{"label": "Telecommande muette", "atomes": [0, 1]}]}
        ),
    )
    atoms = [
        SavAtom.model_validate(_atom(page=1)),
        SavAtom.model_validate(_atom(page=2, symptome="autre")),
    ]

    groups = canonicalize_symptom_groups(atoms)
    assert groups == [{"label": "Telecommande muette", "atomes": [0, 1]}]


def test_panne_dappel_du_regroupement_nest_pas_bloquante(monkeypatch):
    def boom(_listing):
        raise RuntimeError("429")

    monkeypatch.setattr(svc, "_call_grouping_api", boom)
    atoms = [SavAtom.model_validate(_atom(page=1)), SavAtom.model_validate(_atom(page=2))]

    assert canonicalize_symptom_groups(atoms) is None


def test_groupes_fusionnent_des_symptomes_formules_differemment():
    """Le défaut mesuré sur Eneo CC : 3 formulations d'une même panne de télécommande
    devenaient 3 symptômes frères."""
    atoms = [
        _atom(
            symptome="La porte ne se deverrouille pas avec la telecommande",
            cause="c1",
            constat_cause="aucun voyant",
        ),
        _atom(
            symptome="La telecommande ne declenche pas l ouverture",
            cause="c2",
            constat_cause="voyant clignote",
        ),
        _atom(
            symptome="La telecommande ne fonctionne pas apres association",
            cause="c3",
            constat_cause="bip long",
        ),
    ]
    groups = [{"label": "La telecommande ne fait rien", "atomes": [0, 1, 2]}]

    pivot, warnings = build_pivot_from_atoms(
        atoms, document_id=12, document_title="Notice", groups=groups
    )

    tops = [c for c in pivot["cas"] if c["parents"] == []]
    assert len(tops) == 1
    assert tops[0]["nom"] == "La telecommande ne fait rien"
    assert warnings == []


def test_regroupement_invalide_passe_en_avertissement_sans_bloquer():
    atoms = [_atom(page=1), _atom(page=2, symptome="autre")]
    pivot, warnings = build_pivot_from_atoms(
        atoms,
        document_id=12,
        document_title="Notice",
        groups=[{"label": "A", "atomes": [0]}],  # atome 1 manquant
    )

    assert pivot["cas"], "l'assemblage continue malgre le regroupement invalide"
    assert any("Regroupement ignoré" in w for w in warnings)


# ---------------------------------------------------------------------------
# L2b — B2/B3 : constat observable, checklist, effondrement, familles
# ---------------------------------------------------------------------------


def test_le_bouton_porte_le_constat_observable_pas_la_cause():
    atoms = [
        _atom(cause="Polarite inversee des 24 volts", constat_cause="aucun bruit du tout"),
        _atom(cause="Fin de course dereglee", constat_cause="il s arrete au meme endroit"),
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice")

    noms = {c["nom"] for c in pivot["cas"] if "_c" in c["id"]}
    assert noms == {"aucun bruit du tout", "il s arrete au meme endroit"}
    # la cause technique n'est pas perdue : elle vit dans la description
    desc = " ".join(c.get("description", "") for c in pivot["cas"])
    assert "Polarite inversee des 24 volts" in desc


def test_causes_indiscernables_sont_fusionnees_en_une_checklist():
    """Les 4 causes electriques d'Eneo CC : indistinguables sans multimetre. En faire 4
    boutons, c'est demander au client de deviner."""
    atoms = [
        _atom(cause="pas de 220 V au primaire", constat_cause="", qui="sav"),
        _atom(cause="pas de 24 V au secondaire", constat_cause=""),
        _atom(cause="24 V absents a la serrure", constat_cause=""),
        _atom(cause="polarite inversee", constat_cause=""),
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice")

    feuilles = [c for c in pivot["cas"] if c["type"] != "aiguillage" and c["id"] != "sav_global"]
    assert len(feuilles) == 1, "une seule feuille pour les 4 causes indiscernables"
    desc = feuilles[0]["description"]
    for cause in ("220 V", "24 V au secondaire", "polarite inversee"):
        assert cause in desc, f"{cause} doit rester visible dans la checklist"
    # une seule cause reservee au SAV suffit a reserver la feuille au SAV
    assert feuilles[0]["type"] == "sav"


def test_symptome_a_une_seule_cause_devient_la_feuille():
    """14 symptomes sur 17 etaient dans ce cas sur le premier arbre reel : un clic pour
    un seul choix n'apprend rien."""
    pivot, _ = build_pivot_from_atoms(
        [_atom(constat_cause="il s arrete au meme endroit")],
        document_id=12,
        document_title="Notice",
    )

    tops = [c for c in pivot["cas"] if c["parents"] == [] and c["id"] != "sav_global"]
    assert len(tops) == 1
    assert tops[0]["type"] != "aiguillage", "pas d'etage question a un seul choix"
    assert tops[0]["sources"], "la feuille garde sa provenance"


def test_symptome_a_deux_causes_garde_letage_question():
    pivot, _ = build_pivot_from_atoms(
        [_atom(cause="a", constat_cause="signe A"), _atom(cause="b", constat_cause="signe B")],
        document_id=12,
        document_title="Notice",
    )

    tops = [c for c in pivot["cas"] if c["parents"] == []]
    assert len(tops) == 1
    assert tops[0]["type"] == "aiguillage"
    assert len(_children_of(pivot, tops[0]["id"])) == 3, "2 causes + la sortie SAV"


def test_au_dela_de_six_causes_un_etage_famille_apparait():
    """Sous un symptome, la famille n'est pas un theme (le document n'en donne pas a ce
    niveau) : c'est le COUT de la verification, que les atomes portent vraiment."""
    atoms = [
        _atom(cause="c1", constat_cause="pas de courant", geste="mesurer le 220 V", qui="sav"),
        _atom(cause="c2", constat_cause="cable sectionne", geste="deposer le carter",
              effort="demontage"),
        _atom(cause="c3", constat_cause="borne desserree", geste="resserrer la borne",
              effort="outil_simple"),
        _atom(cause="c4", constat_cause="telecommande muette", geste="changer la pile"),
        _atom(cause="c5", constat_cause="pas d association", geste="refaire l association radio"),
        _atom(cause="c6", constat_cause="pene bloque", geste="ajuster la gache",
              effort="outil_simple"),
        _atom(cause="c7", constat_cause="aimant decale", geste="repositionner l aimant"),
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice")

    familles = [c for c in pivot["cas"] if "_f" in c["id"] and c["type"] == "aiguillage"]
    assert len(familles) >= 2, "la profondeur doit naitre de la largeur"

    # le symptome ne montre plus que les familles (+ la sortie SAV) ...
    symptome = next(c for c in pivot["cas"] if c["parents"] == [])
    enfants = _children_of(pivot, symptome["id"])
    assert {c["id"] for c in enfants} == {f["id"] for f in familles} | {"sav_global"}
    # ... et aucune cause n'est perdue en route
    causes = [c for c in pivot["cas"] if "_c" in c["id"]]
    assert len(causes) == 7
    assert all(c["parents"][0] in {f["id"] for f in familles} for c in causes)


def test_letage_famille_ne_sinvente_pas_si_tout_se_verifie_pareil():
    """Sept causes qui coutent toutes la meme chose : un etage a une seule famille ne
    range rien, il ajoute juste un clic. La liste reste plate."""
    atoms = [
        _atom(cause=f"c{i}", constat_cause=f"signe {i}", geste=f"geste {i}")
        for i in range(7)
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice")

    assert not [c for c in pivot["cas"] if "_f" in c["id"]]
    symptome = next(c for c in pivot["cas"] if c["parents"] == [])
    assert len(_children_of(pivot, symptome["id"])) == 8, "7 causes + la sortie SAV"


def test_plus_de_huit_symptomes_un_etage_famille_apparait_a_la_racine():
    """La famille vient du temps B (le vocabulaire du document), jamais d'une taxonomie
    codee en dur : elle doit donc survivre a la fusion des libelles parenthetes."""
    familles = ["Manoeuvre du volet", "Etancheite", "Serrure et verrouillage"]
    atoms = [
        _atom(symptome=f"probleme {i}", cause=f"c{i}", constat_cause=f"signe {i}")
        for i in range(9)
    ]
    groups = [
        {"label": f"probleme {i}", "famille": familles[i % 3], "atomes": [i]}
        for i in range(9)
    ]

    pivot, _ = build_pivot_from_atoms(
        atoms, document_id=12, document_title="Notice", groups=groups
    )

    racines = [c for c in pivot["cas"] if c["parents"] == [] and c["id"] != "sav_global"]
    assert {c["nom"] for c in racines} == set(familles)
    assert all(c["type"] == "aiguillage" for c in racines)
    # les neuf symptomes sont ranges dessous, aucun perdu en route
    ranges = [c for c in pivot["cas"] if c["parents"] and c["parents"][0].startswith("fam")]
    assert len(ranges) == 9


def test_la_famille_survit_a_la_fusion_des_parentheses():
    fusionnes = svc.merge_parenthetical_groups(
        [
            {"label": "ca ne se verrouille pas (mode jour)", "famille": "Serrure", "atomes": [0]},
            {"label": "ca ne se verrouille pas (aimant decale)", "famille": "", "atomes": [1]},
        ]
    )

    assert len(fusionnes) == 1
    assert fusionnes[0]["atomes"] == [0, 1]
    assert fusionnes[0]["famille"] == "Serrure"


def test_une_seule_sortie_sav_partagee():
    """17 noeuds identiques sur le premier arbre reel — le pivot gere le multi-parent."""
    atoms = [
        _atom(symptome="s A", cause="a1", constat_cause="signe 1"),
        _atom(symptome="s A", cause="a2", constat_cause="signe 2"),
        _atom(symptome="s B", cause="b1", constat_cause="signe 3"),
        _atom(symptome="s B", cause="b2", constat_cause="signe 4"),
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice")

    savs = [c for c in pivot["cas"] if c["id"] == "sav_global"]
    assert len(savs) == 1
    assert len(savs[0]["parents"]) == 2, "rattachee aux deux symptomes"


def test_sortie_sav_remonte_au_premier_niveau_si_tout_sest_effondre():
    pivot, _ = build_pivot_from_atoms(
        [_atom(constat_cause="un seul signe")], document_id=12, document_title="Notice"
    )

    savs = [c for c in pivot["cas"] if c["type"] == "sav"]
    assert len(savs) == 1
    assert savs[0]["parents"] == [], "le client garde une porte de sortie"


# ---------------------------------------------------------------------------
# L2b — B5 : une citation qui n'est pas litterale n'est pas une citation
# ---------------------------------------------------------------------------


def test_citation_litterale_est_acceptee():
    page = "Appuyez sur le bouton Reset de l'unite interieure (env. 3 secondes)."
    assert citation_is_verbatim("Appuyez sur le bouton Reset de l'unite interieure", page)


def test_citation_tolere_la_typographie_du_pdf():
    page = "Le fil jaune doit rester non attribue, sinon il pontera l’interrupteur."
    assert citation_is_verbatim(
        "Le fil jaune doit rester non attribue, sinon il pontera l'interrupteur", page
    )


def test_citation_reformulee_est_refusee():
    """Cas reel : « pontera l'interrupteur » devenu « active par erreur » — le sens change."""
    page = "Le fil jaune doit rester non attribue, sinon il pontera l'interrupteur de l'ouvrant."
    reformulee = (
        "Le fil jaune doit rester non attribue pour eviter que l'interrupteur ne soit "
        "active par erreur."
    )
    assert not citation_is_verbatim(reformulee, page)


def test_citation_non_verbatim_part_en_note_interne():
    atoms = [
        _atom(constat_cause="signe", citation="Une phrase totalement inventee par le modele.")
    ]
    pivot, _ = build_pivot_from_atoms(
        atoms,
        document_id=12,
        document_title="Notice",
        pages_text={34: "Le texte reel de la page ne contient pas cette phrase du tout."},
    )

    node = [c for c in pivot["cas"] if c["parents"] == [] and c["id"] != "sav_global"][0]
    assert "note_interne" in node
    assert "vérifier" in node["note_interne"].lower()
    assert "precision" not in node["sources"][0], "pas presentee comme une citation"


def test_citation_verbatim_reste_une_citation():
    phrase = "Retirer le corps etranger de la gache sans forcer."
    atoms = [_atom(constat_cause="signe", citation=phrase)]
    pivot, _ = build_pivot_from_atoms(
        atoms,
        document_id=12,
        document_title="Notice",
        pages_text={34: "Solution : " + phrase},
    )

    node = [c for c in pivot["cas"] if c["parents"] == [] and c["id"] != "sav_global"][0]
    assert node["sources"][0]["precision"] == phrase
    assert "note_interne" not in node


# ---------------------------------------------------------------------------
# L2b — B6 : le titre client vient du produit, pas du nom de fichier
# ---------------------------------------------------------------------------


def test_titre_vient_de_la_designation_produit():
    atoms = [SavAtom.model_validate(_atom(produit="Eneo CC")) for _ in range(3)]
    assert (
        derive_tree_title(atoms, "Proferm - Eneo CC - Notice simplifiee v2 (2022) (1)")
        == "Eneo CC"
    )


def test_titre_nettoie_le_nom_de_fichier_a_defaut_de_produit():
    atoms = [SavAtom.model_validate(_atom(produit=""))]
    titre = derive_tree_title(atoms, "Proferm - Eneo CC - Notice simplifiee v2 (2022) (1).pdf")

    for bruit in ("(1)", "v2", "2022", ".pdf", "Notice"):
        assert bruit not in titre, f"{bruit} ne doit pas remonter au picker client"
    assert "Eneo CC" in titre


def test_titre_explicite_est_respecte():
    pivot, _ = build_pivot_from_atoms(
        [_atom()], document_id=12, document_title="Notice", tree_title="Serrure Eneo CC"
    )
    assert pivot["titre"] == "Serrure Eneo CC"
    assert pivot["symptome"] == "Serrure Eneo CC"


# ---------------------------------------------------------------------------
# Bout en bout : le pivot assemble passe le lint
# ---------------------------------------------------------------------------


class _FakeDocumentSession:
    def get(self, _model, doc_id):
        return object() if doc_id == 12 else None


def test_pivot_assemble_passe_le_lint_sans_blocage():
    from app.services.guided_json_import_service import convert_pivot
    from app.services.guided_lint_service import has_blocking_issues, lint_tree

    atoms = [
        _atom(
            symptome="ne se verrouille pas",
            cause="porte ouverte",
            constat_cause="la porte n est pas jointive",
            page=9,
        ),
        _atom(
            symptome="ne se verrouille pas",
            cause="mode jour",
            constat_cause="le voyant est vert",
            page=9,
        ),
        _atom(symptome="ne reagit pas", cause="pas de 220 V", constat_cause="", qui="sav", page=9),
        _atom(symptome="ne reagit pas", cause="polarite inversee", constat_cause="", page=9),
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice Eneo CC")

    converted = convert_pivot(_FakeDocumentSession(), pivot)
    issues = lint_tree(converted["payload"])

    assert has_blocking_issues(issues) is False, [i.model_dump() for i in issues]


def test_aucun_atome_rend_un_pivot_vide():
    pivot, _ = build_pivot_from_atoms([], document_id=12, document_title="Notice")
    assert pivot["cas"] == []


def test_atome_invalide_est_ignore_avec_avertissement():
    pivot, warnings = build_pivot_from_atoms(
        [{"symptome": ""}], document_id=12, document_title="Notice"
    )
    assert pivot["cas"] == []
    assert warnings


# ---------------------------------------------------------------------------
# L2c — corrections tirees du 2e arbre reel (Roto Safe E | Eneo CC)
# ---------------------------------------------------------------------------


def test_designations_imbriquees_sont_le_meme_materiel():
    """« Eneo CC » et « Roto Safe E | Eneo CC » : le modele ecrit tantot le produit,
    tantot l'en-tete du document. Un faux etage produit en decoulait."""
    atoms = [
        SavAtom.model_validate(_atom(produit="Eneo CC")),
        SavAtom.model_validate(_atom(produit="Roto Safe E | Eneo CC")),
        SavAtom.model_validate(_atom(produit="Eneo CC")),
    ]
    canon_of, labels = svc.canonical_produits(atoms)

    assert len(labels) == 1, "un seul materiel"
    assert len(set(canon_of.values())) == 1


def test_pas_detage_produit_pour_un_seul_materiel_nomme_de_deux_facons():
    atoms = [
        _atom(produit="Eneo CC", symptome="s1", constat_cause="signe 1"),
        _atom(produit="Roto Safe E | Eneo CC", symptome="s2", constat_cause="signe 2"),
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice")

    tops = [c for c in pivot["cas"] if c["parents"] == [] and c["id"] != "sav_global"]
    noms = {c["nom"] for c in tops}
    assert noms == {"s1", "s2"}, "les symptomes doivent etre au premier niveau"
    assert "Eneo CC" not in noms


def test_deux_materiels_reellement_distincts_gardent_letage_produit():
    atoms = [
        _atom(produit="Oximo 40 WF RTS", symptome="s1", constat_cause="signe 1"),
        _atom(produit="RS100 iO", symptome="s2", constat_cause="signe 2"),
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice")

    tops = [c for c in pivot["cas"] if c["parents"] == [] and c["id"] != "sav_global"]
    assert {c["nom"] for c in tops} == {"Oximo 40 WF RTS", "RS100 iO"}


def test_meme_signe_observable_donne_un_seul_bouton():
    """« l'Eneo bipe 3 fois » vaut pour un corps etranger ET un defaut d'alignement.
    Deux boutons « bipe 3 fois » et « bipe 3 fois (2) » n'aident personne."""
    atoms = [
        _atom(cause="corps etranger dans la gache", constat_cause="il bipe 3 fois"),
        _atom(cause="porte et gaches mal alignees", constat_cause="il bipe 3 fois"),
        _atom(cause="contact reed non ferme", constat_cause="il bipe 2 fois"),
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice")

    boutons = [c["nom"] for c in pivot["cas"] if "_c" in c["id"]]
    assert sorted(boutons) == ["il bipe 2 fois", "il bipe 3 fois"]
    assert not any("(2)" in n for n in boutons), "plus de suffixe artificiel"

    # les deux causes derriere « bipe 3 fois » restent lisibles
    trois = next(c for c in pivot["cas"] if c["nom"] == "il bipe 3 fois")
    assert "corps etranger dans la gache" in trois["description"]
    assert "porte et gaches mal alignees" in trois["description"]


def test_citation_est_nettoyee_du_markdown_de_lextracteur():
    """Les tableaux de pannes reviennent en markdown du transcripteur : la citation doit
    rester fidele mais lisible."""
    phrase = "La porte n'est pas completement fermee : **Solution** : Fermer la porte."
    atoms = [_atom(constat_cause="signe", citation=phrase)]
    pivot, _ = build_pivot_from_atoms(
        atoms, document_id=12, document_title="Notice", pages_text={34: phrase}
    )

    node = [c for c in pivot["cas"] if c["parents"] == [] and c["id"] != "sav_global"][0]
    precision = node["sources"][0]["precision"]
    assert "**" not in precision
    assert "La porte n'est pas completement fermee" in precision


def test_racine_porte_une_question_de_depart():
    """Le noeud racine arrivait avec un message vide : LIA n'avait aucune matiere."""
    pivot, _ = build_pivot_from_atoms(
        [_atom(constat_cause="signe")], document_id=12, document_title="Notice"
    )
    assert pivot["question_depart"], "la racine doit poser une question"


def test_question_de_depart_parle_du_materiel_quand_il_y_a_un_etage_produit():
    atoms = [
        _atom(produit="Oximo 40 WF RTS", symptome="s1", constat_cause="signe 1"),
        _atom(produit="RS100 iO", symptome="s2", constat_cause="signe 2"),
    ]
    pivot, _ = build_pivot_from_atoms(atoms, document_id=12, document_title="Notice")
    assert "matériel" in pivot["question_depart"]
