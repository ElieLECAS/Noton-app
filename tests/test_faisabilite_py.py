"""Le calculateur du SOLEAL PY : dimensions maximales du vantail, abaques de paumelles, pivot, vitrage, liste de coupe.

Les attendus sont ceux des exemples écrits dans le wiki (l'oracle du débit de la porte 1 vantail, les deux exemples
imprimés des abaques de paumelles, les limites du tableau des dimensions maximales) ou recalculés à la main depuis ses
tableaux : si une valeur ou un en-tête change dans le wiki, un test le dit — c'est voulu."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services import wiki_famille
from app.services import faisabilite_py as py
from app.services import faisabilite_systemes as reg
from app.services.faisabilite import DonneeIntrouvable, ETUDE, HORS, INFO, OK, _corps
from app.services.faisabilite_ga import FormuleIllisible, evaluer
from app.services.wiki_service import load_snapshot, wiki_root
from app.services.wiki_tables import tables_toutes


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


def _ctl(r, id_):
    return next(c for c in r["controles"] if c["id"] == id_)


def _ids(r):
    return [c["id"] for c in r["controles"]]


def _lignes(f, ref):
    return [(l["qte"], l["longueur"]) for l in f["lignes"] if l["ref"] == ref]


SA1 = {"configuration": "sa1_pmr", "largeur_mm": 900, "hauteur_mm": 2200}


# ---- la grammaire, sur toutes les planches d'application ---------------------------------------------


@pytest.mark.parametrize("formule, variables, attendu", [
    ("H-50.5", {"H": 2200}, 2149.5),
    ("L/2-26.5", {"L": 1000}, 473.5),
    ("4H+3L", {"H": 2200, "L": 900}, 11500),
    ("2(L/2-74)", {"L": 1000}, 852),
    ("Hv1-186.5", {"Hv1": 600}, 413.5),
    ("4H+2L+2L1", {"H": 2200, "L": 1000, "L1": 900}, 12600),
])
def test_la_grammaire_lit_les_formules_de_la_porte(formule, variables, attendu):
    assert evaluer(py.normaliser_formule(formule), variables).valeur == pytest.approx(attendu)


def test_toutes_les_cellules_de_debit_se_lisent(snap):
    """Chaque débit de profilé, de joint et chaque remplissage des 28 applications est lu par la grammaire : une
    formule nouvelle que la grammaire refuse fait échouer le test, avec l'application et la raison. Les quantités écrites
    « *** » ou « 3/ml » (VER-572, VER-574) ne sont pas des formules : elles restent hors de ce contrôle."""
    d = py.donnees(snap)
    vues, refusees = 0, []
    for k, a in d.applications.items():
        formules = [l.debit for l in a.app.profils + a.app.joints if l.debit]
        formules += [r[c] for r in a.app.remplissages for c in ("hauteur", "largeur")]
        for f in formules:
            noms = set(py._identifiants(f))
            try:
                evaluer(f, {n: 1000.0 for n in noms})
                vues += 1
            except FormuleIllisible as exc:
                refusees.append((k, f, str(exc)))
    assert refusees == []
    assert vues > 500


def test_les_applications_du_registre_sont_celles_des_pages(snap):
    """Chaque page d'application porte autant de tableaux de profilés que le registre y compte d'applications : une
    application ajoutée au wiki fait échouer ce test, qu'il faut alors compléter."""
    d = py.donnees(snap)
    assert len(d.applications) == 28
    for page in py.PAGES_APPLICATIONS:
        inscrites = [a for a in d.applications.values() if a.defn.page == page]
        profils = [t for t in tables_toutes(_corps(snap, page)) if py._genre(t) == "profils"]
        assert len(profils) == len(inscrites), page
        titres = {a.defn.section for a in inscrites} - {None}
        assert all(any(c in titres for c in t.chemin[-2:]) for t in profils) or titres == set(), page
    for a in d.applications.values():
        assert a.app.profils and a.app.joints and a.app.accessoires and a.app.remplissages, a.defn.id


# ---- la lecture des tableaux : le contrat avec le wiki -----------------------------------------------


def test_les_tableaux_se_lisent(snap):
    d = py.donnees(snap)
    assert len(d.limites) == 16
    sa1 = next(l for l in d.limites if l.typologie == "Porte simple action 1 vantail" and "T910002" in l.refs)
    assert (sa1.hv, sa1.largeur, sa1.largeur_texte, sa1.poids) == (2400, 1200, "Lv", 130)
    sa1_150 = next(l for l in d.limites if l.typologie == "Porte simple action 1 vantail" and "TPY3615" in l.refs)
    assert sa1_150.refs == ["T910003", "TPY3615"] and sa1_150.poids == 150
    tube1 = next(l for l in d.limites if l.typologie == "Porte tube APD 1 vantail")
    assert tube1.largeur_texte == "colonne titrée Lv1+Lv2" and tube1.refs == ["TPY3711"]     # INC-681
    assert d.lames == {"T910002": "2", "T910003": "3", "TPY3615": "3"}
    assert [(c.cas, c.ferme_porte, c.sens, c.seuil, c.nombre) for c in d.cas] == [
        ("A", False, "<", 2200, 2), ("B", False, ">", 2200, 3), ("C", True, "<", 2200, 3), ("D", True, ">", 2200, 4)]
    assert d.precision_kg == 5
    # les quatre abaques : la ligne 2200 mm, colonne 1200, de la porte non renforcée 3 lames : 125 kg
    assert d.abaques[("3", False)].grille[2200][1200] == 125 and d.abaques[("2", False)].grille[2400][1200] == 120
    assert sorted(d.abaques[("3", True)].grille) == list(range(2000, 2601, 50))
    assert [(p.force, p.h, p.l) for p in d.pivots] == [(2, 2200, 850), (3, 2300, 900), (4, 2400, 1200)]
    assert d.densite_verre == 2.5
    t591005 = [p for p in d.parcloses if p.parclose == "T591005"]
    assert {(p.joint, p.epaisseur_joint, p.remplissage) for p in t591005} == {
        ("TAS0018", 8, 23), ("TAS0017", 7, 24), ("TAS0016", 6, 25), ("TAS0015", 5, 26), ("T710004", 4, 27), ("T710002", 2, 29)}
    assert next(p for p in t591005 if p.joint == "TAS0017").zone == (24, 26)
    # la T591001 ne se monte pas avec les joints jaune et marron : deux cellules vides, donc quatre combinaisons
    assert len([p for p in d.parcloses if p.parclose == "T591001"]) == 4


def test_une_planche_se_lit_en_profils_joints_accessoires_vis_remplissage(snap):
    a = py.donnees(snap).applications["sa1_pmr"]
    assert [(l.ref, l.qte, l.debit) for l in a.app.profils][:3] == [("T225114", "2", "H-50.5"), ("T225114", "1", "L-70"),
                                                                    ("T525051", "1", "L-220")]
    assert a.app.profils[0].coupe == "\\|\\\\" and a.app.profils[2].coupe == "\\|\\|"
    assert [(l.ref, l.debit) for l in a.app.joints] == [("T410010", "2H+2L"), ("T710041", "4H+3L"), ("TAS0017", "2H+2L"),
                                                        ("TPY5001", "L")]
    assert next(l for l in a.app.accessoires if l.ref == "T910002").qte == "*"      # « *** », les ** du gras retirés
    assert next(l for l in a.app.vis if l.ref == "T770022").qte == "3/m"
    assert a.app.remplissages == [{"libelle": "Porte 1 vantail, plinthe et seuil PMR bâtiment", "hauteur": "H-233",
                                   "largeur": "L-232"}]
    assert {"VER-571", "VER-572", "VER-574"} <= set(a.anomalies) and a.saisies == [] and a.indefinies == []
    # la F.I.T : références sans T, débit avant la colonne des dessins, joints intitulés « Quantité »
    f = py.donnees(snap).applications["py1100_sa1_ext"]
    assert [(l.ref, l.qte, l.debit, l.coupe) for l in f.app.profils][:2] == [("225011", "2", "H-6", "45°/90°"),
                                                                             ("225011", "1", "L", "45°/45°")]
    assert [(l.ref, l.debit) for l in f.app.joints][-1] == ("PY5000", "L-95")
    assert f.app.remplissages == [{"libelle": "", "hauteur": "H-214.5", "largeur": "L-232"}]
    assert f.images["225011"].endswith("profile-225011.png")


def test_un_en_tete_change_dans_le_wiki_casse_la_lecture(snap, monkeypatch):
    corps = _corps(snap, py.PAGE_LIMITES)
    assert "| Typologie | Organe de rotation | Référence | Hv maxi (mm) | Largeur maxi (mm) | Poids maxi par vantail (kg) |" in corps
    d = py.APPLICATIONS["sa1_pmr"]
    page = _corps(snap, d.page)
    assert "| Référence | Désignation | Qté | Coupe | Débit (mm) |" in page
    # le même chargement, sur une page à l'en-tête renommé : la lecture échoue au lieu de se taire
    original = wiki_famille.corps_famille
    monkeypatch.setattr(wiki_famille, "corps_famille", lambda s, c: original(s, c).replace(
        "| Référence | Désignation | Qté | Coupe | Débit (mm) |", "| Référence | Désignation | Qté | Coupe | Longueur (mm) |")
        if c == d.page else original(s, c))
    with pytest.raises(DonneeIntrouvable):
        py._lire_application(snap, d)


# ---- les dimensions maximales du vantail ------------------------------------------------------------


def test_dimensions_dans_la_limite(snap):
    r = py.verifier(snap, dict(SA1, organe="T910002"))
    c = _ctl(r, "dimensions")
    assert c["statut"] == OK and c["valeurs"]["hv_max"] == 2400 and c["valeurs"]["largeur_max"] == 1200
    assert c["valeurs"]["typologie"] == "Porte simple action 1 vantail" and "hypothèse" in c["detail"]


def test_dimensions_hors_tout_majorant_sur_etude_vantail_saisi_decide(snap):
    """T910002, Hv 2400 / Lv 1200 / 130 kg : H = 2500 hors tout majore le vantail, donc « sur étude » ; Hv = 2500 saisi
    est « hors domaine »."""
    sa = dict(SA1, organe="T910002", hauteur_mm=2500)
    c = _ctl(py.verifier(snap, sa), "dimensions")
    assert c["statut"] == ETUDE and "saisir Hv et Lv" in c["detail"]
    saisi = _ctl(py.verifier(snap, dict(sa, hauteur_vantail_mm=2500, largeur_vantail_mm=900)), "dimensions")
    assert saisi["statut"] == HORS and "dépassée" in saisi["detail"]
    ok = _ctl(py.verifier(snap, dict(sa, hauteur_vantail_mm=2400, largeur_vantail_mm=1200)), "dimensions")
    assert ok["statut"] == OK
    # une hauteur saisie décide de la hauteur, même sans la largeur
    assert _ctl(py.verifier(snap, dict(sa, hauteur_vantail_mm=2500)), "dimensions")["statut"] == HORS
    assert _ctl(py.verifier(snap, dict(sa, hauteur_vantail_mm=2400, largeur_vantail_mm=1201)), "dimensions")["statut"] == HORS


def test_porte_renforcee_deux_vantaux_et_organes_absents(snap):
    base = {"configuration": "sa2_pmr_fpi", "largeur_mm": 2400, "hauteur_mm": 2600, "organe": "T910003"}
    assert _ctl(py.verifier(snap, base), "dimensions")["statut"] == ETUDE         # SA 2 vantaux : 2400 maxi en hauteur
    c = _ctl(py.verifier(snap, dict(base, renforcee=True)), "dimensions")
    assert c["statut"] == OK and c["valeurs"]["typologie"] == "Porte simple action renforcée 2 vantaux"
    assert c["valeurs"]["hv_max"] == 2600 and c["valeurs"]["largeur_max"] == 2400
    # un organe que la typologie ne liste pas : sur étude, avec la raison
    pivot = _ctl(py.verifier(snap, dict(base, organe="TBN1370", renforcee=True)), "dimensions")
    assert pivot["statut"] == ETUDE and "n'est pas listé" in pivot["detail"]
    # la renforcée n'existe qu'en simple action
    vv = _ctl(py.verifier(snap, {"configuration": "vv1", "largeur_mm": 900, "hauteur_mm": 2200, "renforcee": True}), "dimensions")
    assert vv["statut"] == OK and vv["valeurs"]["typologie"] == "Porte va-et-vient 1 vantail"   # champ non montré : ignoré
    tube = _ctl(py.verifier(snap, {"configuration": "tube1_sa", "largeur_mm": 1200, "hauteur_mm": 2500}), "dimensions")
    assert tube["statut"] == OK and "INC-681" in tube["anomalies"] and tube["valeurs"]["organe"] == "TPY3711"


# ---- le poids du vantail, les paumelles et le pivot -----------------------------------------------


def test_abaque_exemples_imprimes_du_wiki(snap):
    """Paumelles 3 lames non renforcée : H = 2200, L = 1000 → 150 kg ; H = 2300, L = 1100 → 145 kg (VER-600 : lecture
    ± 5 kg). Saisis comme Hv et Lv, ils décident ; à la graduation, aucune interpolation."""
    base = dict(SA1, organe="T910003", hauteur_vantail_mm=2200, largeur_vantail_mm=1000)
    c = _ctl(py.verifier(snap, base), "organe")
    assert c["valeurs"]["abaque_kg"] == 150 and c["valeurs"]["limite_nominale_kg"] == 150 and "VER-600" in c["anomalies"]
    c = _ctl(py.verifier(snap, dict(base, hauteur_vantail_mm=2300, largeur_vantail_mm=1100)), "organe")
    assert c["valeurs"]["abaque_kg"] == 145
    # l'abaque 2 lames : 130 kg, et 125 kg à 2300 × 1100
    c = _ctl(py.verifier(snap, dict(base, organe="T910002", hauteur_vantail_mm=2300, largeur_vantail_mm=1100)), "organe")
    assert c["valeurs"]["abaque_kg"] == 125 and c["valeurs"]["limite_nominale_kg"] == 130


def test_abaque_entre_deux_graduations_la_plus_restrictive(snap):
    """H = 2275, L = 1075 : les quatre croisements 2250/2300 × 1050/1100 valent 145, 140, 150, 145 (3 lames) ; le plus bas
    est lu, sans interpolation."""
    c = _ctl(py.verifier(snap, dict(SA1, organe="T910003", hauteur_vantail_mm=2275, largeur_vantail_mm=1075)), "organe")
    assert c["valeurs"]["abaque_kg"] == 140 and "sans interpolation" in c["detail"]


def test_abaque_porte_renforcee_et_hors_graduations(snap):
    # la porte renforcée va jusqu'à 2600 × 1300 : à 2600 × 1300 la case vaut 145
    c = _ctl(py.verifier(snap, dict(SA1, organe="T910003", renforcee=True, hauteur_vantail_mm=2600, largeur_vantail_mm=1300)), "organe")
    assert c["valeurs"]["abaque_kg"] == 145
    # la porte non renforcée ne va pas au-delà de 2400 mm : hors des graduations, sur étude
    hors = _ctl(py.verifier(snap, dict(SA1, organe="T910003", hauteur_vantail_mm=2500, largeur_vantail_mm=1000)), "organe")
    assert hors["statut"] == ETUDE and "hors des graduations" in hors["detail"]


def test_poids_du_verre_depasse_la_limite(snap):
    """40 mm de verre sur 1967 × 668 mm : 131 kg, plus que le poids de l'abaque à 2200 × 900 (130 kg) quand le vantail est
    saisi ; majorant seulement, « sur étude »."""
    lourd = dict(SA1, organe="T910002", verre_mm=40)
    r = py.verifier(snap, dict(lourd, hauteur_vantail_mm=2200, largeur_vantail_mm=900))
    c = _ctl(r, "organe")
    assert r["poids_verre_kg"] == pytest.approx(131.4, abs=0.1)
    assert c["statut"] == HORS and "dépassé par le verre seul" in c["detail"] and r["verdict"] == HORS
    maj = _ctl(py.verifier(snap, lourd), "organe")
    assert maj["statut"] in (ETUDE, HORS)
    leger = _ctl(py.verifier(snap, dict(SA1, organe="T910002")), "organe")
    assert leger["statut"] == INFO and "ne prouve pas" in leger["detail"] and "VER-600" in leger["anomalies"]


def test_pivot_force_selon_h_et_l(snap):
    """TBN1370 : force 2 jusqu'à 2200 × 850, force 3 jusqu'à 2300 × 900, force 4 jusqu'à 2400 × 1200."""
    def pivot(h, l, **k):
        return _ctl(py.verifier(snap, {"configuration": "vv1", "largeur_mm": l, "hauteur_mm": h, "organe": "TBN1370", **k}),
                    "organe")
    assert pivot(2200, 850)["valeurs"]["force_pivot"] == 2
    assert pivot(2300, 900)["valeurs"]["force_pivot"] == 3
    assert pivot(2400, 1200)["valeurs"]["force_pivot"] == 4
    hors = pivot(2400, 1300, hauteur_vantail_mm=2400, largeur_vantail_mm=1300)
    assert hors["statut"] == HORS and "aucune force" in hors["detail"]
    assert pivot(2400, 1300)["statut"] == ETUDE                      # majorant : sur étude
    crapaudine = _ctl(py.verifier(snap, {"configuration": "vv1", "largeur_mm": 900, "hauteur_mm": 2200}), "organe")
    assert crapaudine["statut"] == INFO and "pas d'abaque de poids" in crapaudine["detail"]


def test_nombre_de_paumelles_cas_a_d(snap):
    """A : 2 (H < 2200) ; B : 3 (> 2200) ; C : 3 avec ferme-porte ; D : 4. Un vantail saisi à 2200 mm exactement n'est dans
    aucun cas (INC-698) ; H = 2200 mm hors tout, le vantail étant plus petit, est dans le cas A ou C."""
    def cas(**k):
        return _ctl(py.verifier(snap, dict(SA1, organe="T910002", **k)), "paumelles")
    assert cas(hauteur_mm=2150)["valeurs"] == {"cas": "A", "nombre_par_vantail": 2}
    assert cas(hauteur_mm=2300)["valeurs"]["cas"] == "B"
    assert cas(hauteur_mm=2150, ferme_porte=True)["valeurs"] == {"cas": "C", "nombre_par_vantail": 3}
    assert cas(hauteur_mm=2300, ferme_porte=True)["valeurs"] == {"cas": "D", "nombre_par_vantail": 4}
    assert cas(hauteur_mm=2200)["valeurs"]["cas"] == "A" and cas(hauteur_mm=2200, ferme_porte=True)["valeurs"]["cas"] == "C"
    c = cas(hauteur_vantail_mm=2200, largeur_vantail_mm=900)
    assert c["statut"] == ETUDE and "INC-698" in c["anomalies"] and "aucun cas" in c["detail"]
    # la planche FPI est une porte avec ferme-porte : C
    fpi = _ctl(py.verifier(snap, {"configuration": "sa1_pmr_fpi", "largeur_mm": 900, "hauteur_mm": 2150}), "paumelles")
    assert fpi["valeurs"]["cas"] == "C" and "VER-572" in fpi["anomalies"] and "« *** »" in fpi["detail"]
    # la planche retard à l'effraction pose 3 TPY3615 par vantail : pas moins que le cas A
    rc3 = _ctl(py.verifier(snap, {"configuration": "sa1_rc3_ext", "largeur_mm": 900, "hauteur_mm": 2150, "hauteur_vantail_mm": 2100,
                                  "largeur_vantail_mm": 900, "hv1_mm": 700, "hv2_mm": 1300}), "paumelles")
    assert rc3["statut"] == INFO and "3 TPY3615" in rc3["detail"]


# ---- le vitrage ------------------------------------------------------------------------------------


def test_vitrage_la_planche_donne_24_mm(snap):
    """T591005 avec le joint TAS0017 (B = 7 mm) : 24 mm, dans la zone recommandée 24 à 26 mm."""
    c = _ctl(py.verifier(snap, SA1), "vitrage")
    assert c["statut"] == OK and "24 mm" in c["detail"] and "zone recommandée 24 à 26" in c["detail"]
    assert c["valeurs"]["joint"] == "TAS0017" and "VER-566" in c["anomalies"]


def test_vitrage_autre_epaisseur_meme_parclose_autre_joint(snap):
    """26 mm : la même parclose T591005 prend le joint bleu TAS0015 (B = 5 mm)."""
    c = _ctl(py.verifier(snap, dict(SA1, epaisseur_vitrage_mm="26")), "vitrage")
    assert c["statut"] == OK and c["valeurs"]["joint"] == "TAS0015" and "même longueur" in c["detail"]
    # 36 mm : T591005 n'a pas de joint qui convienne, la matrice le donne avec T591001 : la parclose changerait
    c = _ctl(py.verifier(snap, dict(SA1, epaisseur_vitrage_mm="36")), "vitrage")
    assert c["statut"] == ETUDE and "T591001" in c["detail"] and "change la parclose" in c["detail"]


def test_vitrage_aucune_parclose(snap):
    d = py.donnees(snap)
    assert {int(p.remplissage) for p in d.parcloses} >= {4, 39}      # les parcloses à pelle ; la pose de face va à 42 mm
    # 3 mm n'existe pas dans le champ proposé : la saisie est refusée
    with pytest.raises(ValueError, match="Épaisseur du remplissage"):
        py.verifier(snap, dict(SA1, epaisseur_vitrage_mm="3"))


# ---- les applications « sur étude » --------------------------------------------------------------------


def test_ensemble_menuise_h1_h2_non_definis(snap):
    """H1 et H2 « se partagent la hauteur » sans dire laquelle est laquelle : l'application reste sur étude."""
    a = py.donnees(snap).applications["sa1_ensemble"]
    assert set(a.indefinies) == {"H1", "H2"} and a.saisies == []
    r = py.verifier(snap, {"configuration": "sa1_ensemble", "largeur_mm": 1500, "hauteur_mm": 2200})
    c = _ctl(r, "application")
    assert r["verdict"] in (ETUDE, HORS) and c["statut"] == ETUDE and "H1 et H2" in c["detail"]
    assert _ctl(r, "organe")["statut"] == ETUDE and "Poids du verre non calculé" in _ctl(r, "organe")["detail"]
    assert "H1" in py.fiche_debit(snap, {"configuration": "sa1_ensemble", "largeur_mm": 1500, "hauteur_mm": 2200})["indisponible"]


def test_porte_2_vantaux_sans_fpi_l2_ecrit_pour_l_sur_2(snap):
    """INC-684 : la planche sans FPI écrit L2 pour L/2 dans le rejet d'eau ; L2 n'est défini que par la planche de l'ensemble
    menuisé. La correction n'est pas appliquée : l'application sans FPI est sur étude, celle avec FPI se calcule."""
    sans = py.donnees(snap).applications["sa2_pmr"]
    assert sans.indefinies == ["L2"] and "INC-684" in sans.anomalies
    f = py.fiche_debit(snap, {"configuration": "sa2_pmr", "largeur_mm": 1800, "hauteur_mm": 2200})
    assert "INC-684" in f["indisponible"] and "décision du service technique" in f["indisponible"]
    avec = py.fiche_debit(snap, {"configuration": "sa2_pmr_fpi", "largeur_mm": 1800, "hauteur_mm": 2200})
    assert avec["lignes"] and "indisponible" not in avec


def test_cotes_definies_par_la_page_sont_saisies(snap):
    """Hv1 et Hv2 (retard à l'effraction), L1 et L2 (fixe latéral) : cotes lues sur les coupes, définies par la page."""
    d = py.donnees(snap)
    assert d.applications["sa1_rc3_ext"].saisies == ["Hv1", "Hv2"] and d.applications["sa1_rc3_peri"].saisies == ["Hv1", "Hv2"]
    assert d.applications["sa1_fixe"].saisies == ["L1", "L2"] and d.applications["tube1_da_fixe"].saisies == ["L1", "L2"]
    assert d.applications["py1100_fixe_ext"].saisies == ["L1", "L2"]
    ids = {c["id"] for c in py.options(snap)["champs"]}
    assert {"hv1_mm", "hv2_mm", "l1_mm", "l2_mm"} <= ids
    champs = {c["id"]: c for c in py.options(snap)["champs"]}
    assert "sa1_rc3_ext" in champs["hv1_mm"]["si"]["configuration"] and "sa1_ensemble" not in champs["l1_mm"]["si"]["configuration"]


def test_autre_application(snap):
    r = py.verifier(snap, dict(SA1, configuration="autre"))
    assert r["verdict"] == ETUDE and _ids(r) == ["application"]
    assert "pas de liste de coupe" in py.fiche_debit(snap, dict(SA1, configuration="autre"))["indisponible"]


# ---- la fiche de débit : l'oracle de la porte 1 vantail --------------------------------------------------


def test_fiche_oracle_sa_1_vantail_plinthe_seuil_pmr_batiment(snap):
    """H = 2200, L = 900 : ouvrants 2 × 2149,5 et 830 ; T525060 780 ; T525051 680 ; T525052 828 ; T591005 2 × 1935,5 et
    2 × 680 ; T700047 680 ; TPY1102 2 × 2200 et 900 ; vitrage 1967 × 668 ; T710041 4H+3L = 11 500."""
    f = py.fiche_debit(snap, SA1)
    assert _lignes(f, "T225114") == [(2, 2149.5), (1, 830)]
    assert _lignes(f, "T525060") == [(1, 780)] and _lignes(f, "T525051") == [(1, 680)] and _lignes(f, "T525052") == [(1, 828)]
    assert _lignes(f, "T591005") == [(2, 1935.5), (2, 680)] and _lignes(f, "T700047") == [(1, 680)]
    assert _lignes(f, "TPY1102") == [(1, 2200), (1, 2200), (1, 900)]
    assert _lignes(f, "T710041") == [(1, 11500)] and _lignes(f, "T410010") == [(1, 6200)]
    (vit,) = [l for l in f["lignes"] if l["famille"] == "Remplissage"]
    assert vit["calcul"] == "H-233 × L-232 = 1967 × 668 mm" and vit["qte"] == 1
    # le dormant TPY1102 : une ligne par pièce de la planche, donc 3 lignes
    n = next(n for n in f["nomenclature"] if n["ref"] == "TPY1102")
    assert n["qte"] == 3 and n["ml"] == pytest.approx(5.3)


def test_fiche_forme_pictogrammes_sources_et_paumelles(snap):
    f = py.fiche_debit(snap, dict(SA1, hauteur_mm=2150))
    assert {"verdict", "libelle", "lignes", "nomenclature", "accessoires", "notes", "brouillons"} <= set(f)
    for l in f["lignes"]:
        assert {"rep", "famille", "piece", "ref", "qte", "longueur", "calcul", "coupe", "source", "note", "image"} <= set(l)
        assert l["source"]["page"] == py.P_SA1
    assert [l["rep"] for l in f["lignes"]] == list(range(1, len(f["lignes"]) + 1))
    ouvrant = next(l for l in f["lignes"] if l["ref"] == "T225114")
    assert ouvrant["coupe"] == "pictogramme « \\|\\\\ » (non légendé)"
    # « *** » : la répartition à 2150 mm sans ferme-porte est le cas A, 2 paumelles
    assert _lignes(f, "T910002") == [(2, None)]
    assert "cas A, 2 par vantail, 2 au total" in " ".join(f["notes"])
    # un vantail saisi à 2200 mm exactement : aucun cas, la quantité n'est pas lue, la ligne est signalée
    f = py.fiche_debit(snap, dict(SA1, hauteur_vantail_mm=2200, largeur_vantail_mm=900))
    assert _lignes(f, "T910002") == [] and "INC-698" in " ".join(f["notes"])
    f = py.fiche_debit(snap, SA1)
    # des quantités en formules sans légende ne sont pas calculées, et la fiche le dit
    assert "Lignes non calculées" in " ".join(f["notes"]) and "T1110" in " ".join(f["notes"])
    assert "VER-574" in " ".join(f["notes"])


def test_fiche_deux_vantaux_et_vitrage_par_vantail(snap):
    f = py.fiche_debit(snap, {"configuration": "sa2_pmr_fpi", "largeur_mm": 1800, "hauteur_mm": 2150})
    (vit,) = [l for l in f["lignes"] if l["famille"] == "Remplissage"]
    assert vit["calcul"].startswith("H-281 × L/2-189") and vit["qte"] == 2 and "un par vantail" in vit["note"]
    assert _lignes(f, "T910002") == [(6, None)]       # cas C (FPI), 3 par vantail, 2 vantaux


def test_fiche_retard_a_l_effraction_deux_hauteurs_de_vitrage(snap):
    """RC3, 1 vantail : Hv1-155 × L-278 et Hv2-170 × L-278, cotes Hv1 et Hv2 saisies."""
    saisie = {"configuration": "sa1_rc3_ext", "largeur_mm": 900, "hauteur_mm": 2150, "hv1_mm": 700, "hv2_mm": 1300}
    f = py.fiche_debit(snap, saisie)
    vit = [l["calcul"] for l in f["lignes"] if l["famille"] == "Remplissage"]
    assert vit == ["Hv1-155 × L-278 = 545 × 622 mm", "Hv2-170 × L-278 = 1130 × 622 mm"]
    f2 = py.fiche_debit(snap, dict(saisie, configuration="sa2_rc3_ext", largeur_mm=1800))
    qtes = [l["qte"] for l in f2["lignes"] if l["famille"] == "Remplissage"]
    assert qtes == [2, 2] and "hypothèse" in next(l["note"] for l in f2["lignes"] if l["famille"] == "Remplissage")
    assert "INC-687" in " ".join(f2["notes"])


def test_fiche_fixe_lateral_remplissages_du_vantail_et_du_fixe(snap):
    saisie = {"configuration": "sa1_fixe", "largeur_mm": 1300, "hauteur_mm": 2200, "l1_mm": 900, "l2_mm": 400}
    f = py.fiche_debit(snap, saisie)
    vit = [l["calcul"] for l in f["lignes"] if l["famille"] == "Remplissage"]
    assert sorted(vit) == sorted(["H-66 × L2-66 = 2134 × 334 mm", "H-233 × L1-232 = 1967 × 668 mm"])
    r = py.verifier(snap, saisie)
    # le verre du fixe latéral n'est pas celui du vantail : 8 mm × 2,5 × 1,967 × 0,668
    assert r["poids_verre_kg"] == pytest.approx(26.3, abs=0.1)


def test_fiche_F_I_T_2012_est_etiquetee_et_jamais_fusionnee(snap):
    """Le seuil PY1100 de la F.I.T : références sans T, angles écrits ; la génération est affichée, l'écart renvoyé aux CTR."""
    saisie = {"configuration": "py1100_sa1_int", "largeur_mm": 900, "hauteur_mm": 2200}
    f = py.fiche_debit(snap, saisie)
    assert f["document"] == py.FIT and f["application"].endswith("(F.I.T 2012)")
    assert _lignes(f, "225104") == [(2, 2147.5), (2, 830)] and _lignes(f, "PY2102") == [(1, 780)]
    assert next(l for l in f["lignes"] if l["ref"] == "225104")["coupe"] == "angles de coupe écrits : 45°/45°"
    (vit,) = [l for l in f["lignes"] if l["famille"] == "Remplissage"]
    assert vit["calcul"] == "H-214.5 × L-232 = 1985,5 × 668 mm"
    r = py.verifier(snap, saisie)
    doc = _ctl(r, "document")
    assert doc["statut"] == INFO and doc["anomalies"].count("CTR-243") == 1 and "jamais fusionnés" in doc["detail"]
    # l'écart est affiché, avec la planche de 2018 que CTR-243 lui oppose : H-214.5 contre H-215
    assert "remplissage H-214.5 × L-232 dans la F.I.T contre H-215 × L-232 dans le catalogue de 2018" in doc["detail"]
    assert "VER-152" in doc["detail"] and "VER-151" in " ".join(f["notes"])
    # le catalogue de 2018 n'a pas ce contrôle de génération
    assert "document" not in _ids(py.verifier(snap, SA1))
    # la paumelle 910001 (90 kg) n'est pas au tableau des dimensions maximales : l'organe est à choisir
    assert _ctl(r, "dimensions")["statut"] == ETUDE and "Organe de rotation inconnu" in _ctl(r, "dimensions")["detail"]
    choisi = _ctl(py.verifier(snap, dict(saisie, organe="T910002")), "dimensions")
    assert choisi["statut"] == OK


def test_fiche_joint_de_parclose_a_une_autre_epaisseur(snap):
    f = py.fiche_debit(snap, dict(SA1, hauteur_mm=2150, epaisseur_vitrage_mm="26"))
    assert _lignes(f, "TAS0017") == [] and _lignes(f, "TAS0015") == [(1, 2 * 2150 + 2 * 900)]
    assert "même longueur reprise" in " ".join(f["notes"])
    # 36 mm : la parclose de la planche ne convient pas, la fiche reste celle de la planche et la faisabilité le dit
    r = py.verifier(snap, dict(SA1, hauteur_mm=2150, epaisseur_vitrage_mm="36"))
    assert _ctl(r, "vitrage")["statut"] == ETUDE and r["verdict"] == ETUDE


# ---- la forme du résultat et les anomalies --------------------------------------------------------------


def test_la_forme_du_resultat_et_les_anomalies(snap):
    r = py.verifier(snap, dict(SA1, hauteur_mm=2150, hauteur_vantail_mm=2100, largeur_vantail_mm=850, organe="T910002"))
    assert r["verdict"] == OK, [(c["id"], c["statut"], c["detail"]) for c in r["controles"]]
    assert _ids(r) == ["dimensions", "organe", "paumelles", "vitrage", "goupillage"]
    assert {"verdict", "libelle", "contrainte", "controles", "brouillons"} <= set(r)
    registres = "".join(p.read_text(encoding="utf-8") for p in (Path(snap.root) / "wiki" / "anomalies").glob("*.md"))
    for k in py.APPLICATIONS:
        for c in py.verifier(snap, {"configuration": k, "largeur_mm": 1200, "hauteur_mm": 2200, "organe": "T910002"})["controles"]:
            assert set(c) == {"id", "titre", "statut", "detail", "sources", "anomalies", "valeurs"} and c["detail"]
            assert c["statut"] in (OK, HORS, ETUDE, INFO)
            for s in c["sources"]:
                assert set(s) == {"page", "section", "localisation"} and s["page"] in snap.pages
            for a in c["anomalies"]:
                assert re.search(rf"\|\s*{a}\s*\|", registres), f"{a} n'existe pas dans les registres ({k})"
    # des pages en brouillon : la planche, les limites, les abaques ; la page des parcloses est stable
    assert py.P_SA1 in r["brouillons"] and py.PAGE_LIMITES in r["brouillons"] and py.PAGE_ABAQUES["2"] in r["brouillons"]
    assert py.PAGE_PARCLOSES not in r["brouillons"]
    assert r["application"] == APPLICATION_SA1 and r["document"] == py.CONCEPTION and r["organe"] == "T910002"


APPLICATION_SA1 = "SA 1 vantail, ouverture extérieure, plinthe et seuil PMR bâtiment"


def test_les_anomalies_du_contrat_sont_affichees(snap):
    d = py.donnees(snap)
    assert "CTR-258" in d.applications["sa1_plat"].anomalies
    assert {"CTR-128", "CTR-129"} <= set(d.applications["py1100_sa2_ext"].anomalies)
    assert {"INC-687"} <= set(d.applications["sa2_rc3_ext"].anomalies) and "INC-688" in d.applications["sa2_rc3_peri"].anomalies
    assert "INC-684" in d.applications["sa2_pmr"].anomalies and "VER-573" in d.applications["sa2_pmr_fpi"].anomalies
    assert {"INC-689"} <= set(d.applications["tube2_sa"].anomalies) and "INC-690" in d.applications["vv2"].anomalies
    assert "INC-686" in d.applications["sa1_rpt"].anomalies
    f = py.fiche_debit(snap, {"configuration": "sa1_plat", "largeur_mm": 900, "hauteur_mm": 2150})
    assert "CTR-258" in " ".join(f["notes"])


# ---- par le registre ----------------------------------------------------------------------------------------


def test_le_registre_propose_le_soleal_py(snap):
    s = next(x for x in reg.systemes()["systemes"] if x["id"] == "soleal-py")
    assert s["libelle"] == "SOLEAL PY (porte LUMINE)" and s["gammes"] == ["LUMINE"] and s["legacy"] is False
    assert s["portee"] == py.PORTEE
    o = reg.options(snap, "soleal-py")
    assert {"champs", "configurations", "pages", "brouillons", "avertissements"} <= set(o)
    assert [c["id"] for c in o["champs"]] == ["configuration", "largeur_mm", "hauteur_mm", "epaisseur_vitrage_mm", "verre_mm",
                                              "organe", "renforcee", "ferme_porte", "hauteur_vantail_mm",
                                              "largeur_vantail_mm", "hv1_mm", "hv2_mm", "l1_mm", "l2_mm"]
    assert len(o["configurations"]) == 29 and "autre" in o["configurations"]
    assert all(p in snap.pages for p in o["pages"]) and py.P_SA1 in o["brouillons"] and py.PAGE_PARCLOSES not in o["brouillons"]
    assert all(a for a in o["avertissements"])
    r = reg.verifier(snap, "soleal-py", SA1)
    assert r["application"] == APPLICATION_SA1
    f = reg.fiche_debit(snap, "soleal-py", dict(SA1, hauteur_mm=2150))
    assert f["lignes"] and f["nomenclature"]


def test_toutes_les_applications_passent_par_le_registre(snap):
    """Les 28 applications donnent un verdict et, hors celles qui restent sur étude, une fiche."""
    sur_etude = {k for k, a in py.donnees(snap).applications.items() if a.indefinies}
    assert sur_etude == {"sa2_pmr", "sa1_ensemble"}
    for k in py.APPLICATIONS:
        saisie = {"configuration": k, "largeur_mm": 1200, "hauteur_mm": 2200}
        r = reg.verifier(snap, "soleal-py", saisie)
        assert r["verdict"] in (OK, ETUDE, HORS) and r["controles"], k
        f = reg.fiche_debit(snap, "soleal-py", saisie)
        if k in sur_etude:
            assert "indisponible" in f, k
        else:
            assert f["lignes"] and f["nomenclature"], k
            assert all(l["longueur"] is None or l["longueur"] > 0 for l in f["lignes"]), k


def test_la_saisie_est_verifiee_contre_les_champs(snap):
    with pytest.raises(ValueError, match="Application"):
        py.verifier(snap, dict(SA1, configuration="coulissant"))
    with pytest.raises(ValueError, match="Largeur hors tout L : au moins"):
        py.verifier(snap, dict(SA1, largeur_mm=10))
    with pytest.raises(ValueError, match="Organe de rotation"):
        py.verifier(snap, dict(SA1, organe="T999999"))
    # un champ non montré (application « autre ») n'est pas vérifié
    assert py.verifier(snap, dict(SA1, configuration="autre", organe="zzz"))["verdict"] == ETUDE
    # la cote d'une planche qui l'utilise est une saisie obligatoire de sa plage
    with pytest.raises(ValueError, match="Cote L1 : au moins"):
        py.verifier(snap, {"configuration": "sa1_fixe", "largeur_mm": 1300, "hauteur_mm": 2200, "l1_mm": 10})
