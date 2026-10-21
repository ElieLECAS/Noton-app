"""Le calculateur ASKEY Coulissant 65 NV : abaques, combinaisons, zone bleue, charge de roulette.

Les attendus sont les calculs d'exemple écrits dans le wiki et les cases de ses tableaux : si une valeur ou un
en-tête change dans le wiki, le test le dit — c'est voulu, un outil qui calcule ne doit pas dériver en silence."""
from __future__ import annotations

import pytest

from app.services import faisabilite_askey_nv as nv
from app.services import faisabilite_systemes as fs
from app.services.faisabilite import DonneeIntrouvable
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


def _saisie(**k):
    base = dict(configuration="chicanes", module="32", pression="800", fleche="150", largeur_mm=1300, hauteur_mm=2200,
                profil1_chicanes_32="W3041250", profil2_chicanes_32="W3041251")
    base.update(k)
    return base


def _chic(p1, p2, **k):
    return _saisie(profil1_chicanes_32=p1, profil2_chicanes_32=p2, **k)


def _ctl(r, id_):
    return next(c for c in r["controles"] if c["id"] == id_)


# ---- la lecture des tableaux : le contrat avec le wiki ------------------------------------------

def test_les_tableaux_se_lisent(snap):
    d = nv.donnees(snap)
    assert set(d.abaques) == {(p, f) for p in (800, 1200) for f in (150, 200, 300)}
    ab = d.abaques[(800, 150)]
    assert len(ab.hauteurs) == 16 and len(ab.largeurs) == 16 and len(ab.cases) == 256
    assert (ab.hauteurs[0], ab.hauteurs[-1], ab.largeurs[0], ab.largeurs[-1]) == (1.0, 2.5, 0.5, 2.0)
    assert ab.cases[(2.2, 1.3)] == 26.7 and ab.cases[(2.3, 1.3)] == 31.6 and ab.cases[(2.4, 1.3)] == 37.9
    # la zone bleue : 1,30 m à 800 Pa L/150, 1,50 m partout ailleurs, 2,20 m de haut
    assert (ab.h_bleu, ab.l_bleu) == (2.2, 1.3)
    assert all((a.h_bleu, a.l_bleu) == (2.2, 1.5) for k, a in d.abaques.items() if k != (800, 150))
    assert d.abaques[(1200, 150)].cases[(2.2, 1.3)] == 40.1
    # les combinaisons : paire non ordonnée, entiers
    assert d.combinaisons[("chicanes", "32")].paires[("W3041250", "W3041251")] == 39
    assert d.combinaisons[("montants", "24")].paires[("W1041261", "W1041261")] == 55
    assert d.combinaisons[("montants", "32")].paires[("Y1740040", "Y1740041")] == 81
    # l'inertie des montants par longueur
    assert d.statiques["32"].ix["W1041270"][1200] == 6.19 and d.statiques["24"].ix["W1041260"][3000] == 8.14
    assert d.statiques["32"].graduations == [1200, 1600, 2000, 2250, 2500, 3000]
    # charges, rail, facteur de poids
    assert {r: v["par_vantail"] for r, v in d.roulettes.items()} == {
        "W6050151": 80, "W6050153": 160, "W6050155": 220, "W6000025": 250}
    assert d.rail_seuil_mm == 2250 and d.rails == ("W3041224", "W4000158") and d.kg_par_m2_mm == 2.5


def test_les_abaques_ne_decroissent_jamais(snap):
    """Le cran supérieur n'est du côté sûr que si l'inertie requise ne décroît ni avec la hauteur ni avec la largeur."""
    for (p, f), ab in nv.donnees(snap).abaques.items():
        for l in ab.largeurs:
            col = [ab.cases[(h, l)] for h in ab.hauteurs]
            assert col == sorted(col), (p, f, l)
        for h in ab.hauteurs:
            ligne = [ab.cases[(h, l)] for l in ab.largeurs]
            assert ligne == sorted(ligne), (p, f, h)


def test_l_abaque_1200_l200_est_celui_de_800_l300(snap):
    """VER-728 : les 256 cases sont identiques une à une ; le jour où le wiki corrige l'une des deux, ce test le dit."""
    d = nv.donnees(snap)
    assert d.abaques[(1200, 200)].cases == d.abaques[(800, 300)].cases


def test_un_en_tete_change_casse_le_calcul(snap):
    """Un tableau dont l'en-tête a changé n'est plus lu : le calculateur s'arrête au lieu de se tromper."""
    for page, avant, apres in [
        (nv.PAGE_COMBINAISONS, "Ix de la combinaison (cm⁴)", "Inertie (cm⁴)"),
        (nv.PAGE_QUINCAILLERIE, "Charge maxi par vantail (kg)", "Charge (kg)"),
        (nv.PAGES_ABAQUES[0], "Abaque 800 Pa, flèche admissible L/200", "Abaque 800 Pa, flèche L/200"),
        (nv.PAGE_STATIQUES, "Ix à 1 200 mm (cm⁴)", "Ix à 1200 mm"),
    ]:
        p = snap.pages[page]
        original = p.body
        assert avant in original
        p.body = original.replace(avant, apres)
        try:
            with pytest.raises(DonneeIntrouvable):
                nv._charge(snap)
        finally:
            p.body = original


# ---- les oracles du catalogue ------------------------------------------------------------------

@pytest.mark.parametrize("p1, p2, ix, verdict", [
    ("W3041250", "W3041250", 12, "hors"),      # 800 Pa, L/150, H 2,20 × L 1,30 : 26,7 cm⁴ requis
    ("W3041250", "Y3740020", 25, "hors"),
    ("W3041250", "W3041251", 39, "ok"),
])
def test_800_pa_l150_case_2_20_par_1_30(snap, p1, p2, ix, verdict):
    r = nv.verifier(snap, _chic(p1, p2))
    assert _ctl(r, "abaque")["valeurs"]["requis"] == 26.7
    assert _ctl(r, "combinaison")["valeurs"]["ix"] == ix
    assert _ctl(r, "inertie")["statut"] == verdict
    assert r["verdict"] == verdict


def test_1200_pa_l150_meme_case(snap):
    r = nv.verifier(snap, _chic("W3041250", "W3041251", pression="1200"))
    assert _ctl(r, "abaque")["valeurs"]["requis"] == 40.1
    assert _ctl(r, "inertie")["statut"] == "hors" and _ctl(r, "inertie")["valeurs"]["marge"] == -1.1   # 39 contre 40,1
    r = nv.verifier(snap, _chic("W3041251", "Y3740020", pression="1200"))                              # 52
    assert _ctl(r, "inertie")["statut"] == "ok" and r["verdict"] == "ok"


def test_les_autres_fleches(snap):
    r = nv.verifier(snap, _chic("W3041250", "W3041251", fleche="200"))
    assert _ctl(r, "abaque")["valeurs"]["requis"] == 35.6 and r["verdict"] == "ok"
    r = nv.verifier(snap, _chic("W3041251", "Y3740020", fleche="300"))
    assert _ctl(r, "abaque")["valeurs"]["requis"] == 53.5 and r["verdict"] == "hors"       # 52 < 53,5
    r = nv.verifier(snap, _chic("W3041250", "W3041255", fleche="300"))
    assert r["verdict"] == "ok"                                                              # 66


def test_les_trois_fleches_sont_affichees(snap):
    """Le wiki ne dit pas quelle flèche imposer : la saisie décide, les trois sont données avec leur verdict."""
    r = nv.verifier(snap, _chic("W3041250", "W3041251"))
    etapes = _ctl(r, "inertie")["valeurs"]["etapes"]
    assert [e["etape"] for e in etapes] == ["Flèche L/150 (retenue)", "Flèche L/200", "Flèche L/300"]
    assert [e["resultat"] for e in etapes] == ["suffisante", "suffisante", "insuffisante"]   # 26,7 ; 35,6 ; 53,5 contre 39


def test_cran_superieur_jamais_d_interpolation(snap):
    """H 2,15 × L 1,25 lit la case de H 2,20 × L 1,30 : le cran supérieur, jamais la graduation du dessous."""
    r = nv.verifier(snap, _chic("W3041250", "W3041251", largeur_mm=1250, hauteur_mm=2150))
    v = _ctl(r, "abaque")["valeurs"]
    assert (v["hauteur_lue"], v["largeur_lue"], v["requis"]) == (2.2, 1.3, 26.7)
    assert "arrondi au cran supérieur" in _ctl(r, "abaque")["detail"]
    # une graduation exacte ne monte pas d'un cran
    v = _ctl(nv.verifier(snap, _chic("W3041250", "W3041251")), "abaque")["valeurs"]
    assert (v["hauteur_lue"], v["largeur_lue"]) == (2.2, 1.3)


def test_exemple_de_la_page_hors_zone_bleue(snap):
    """L 1,30 × H 2,25 : ligne 2,30, 31,6 cm⁴ (le texte de la page dit 37,9 : INC-741). Hors zone bleue : sur étude,
    même si 39 cm⁴ passent les deux valeurs."""
    r = nv.verifier(snap, _chic("W3041250", "W3041251", hauteur_mm=2250))
    assert _ctl(r, "abaque")["valeurs"]["requis"] == 31.6
    assert _ctl(r, "zone_bleue")["statut"] == "etude"
    assert _ctl(r, "inertie")["statut"] == "etude" and r["verdict"] == "etude"
    assert 39 >= 37.9 and 39 >= 31.6
    assert "INC-741" in _ctl(r, "abaque")["anomalies"]


def test_hors_zone_bleue_ne_dit_jamais_non(snap):
    """Une paire insuffisante hors zone bleue reste « sur étude » : l'abaque sort du DTA, il ne condamne pas."""
    r = nv.verifier(snap, _chic("W3041250", "W3041250", hauteur_mm=2250))
    assert _ctl(r, "inertie")["statut"] == "etude" and r["verdict"] == "etude"


@pytest.mark.parametrize("fleche, largeur, zone", [
    ("150", 1300, "ok"), ("150", 1400, "etude"),     # 1,30 m à 800 Pa L/150
    ("200", 1500, "ok"), ("200", 1600, "etude"),     # 1,50 m ailleurs
])
def test_zone_bleue_en_largeur(snap, fleche, largeur, zone):
    r = nv.verifier(snap, _chic("W3041250", "W3041251", fleche=fleche, largeur_mm=largeur))
    assert _ctl(r, "zone_bleue")["statut"] == zone
    assert "VER-729" in _ctl(r, "zone_bleue")["anomalies"]


def test_au_dela_de_la_grille_sur_etude(snap):
    for k in ({"hauteur_mm": 2600}, {"largeur_mm": 2100}):
        r = nv.verifier(snap, _chic("W3041250", "W3041251", **k))
        assert _ctl(r, "abaque")["statut"] == "etude" and r["verdict"] == "etude"
        assert not any(c["id"] == "inertie" for c in r["controles"])


@pytest.mark.parametrize("hauteur, largeur, p1, p2", [
    (2200, 500, "W3041250", "W3041250"),    # 11,6 requis, 12 : marge 0,4
    (2200, 1200, "W3041250", "Y3740020"),   # 25,2 requis, 25 : marge −0,2
])
def test_marge_de_moins_de_1_cm4_sur_etude(snap, hauteur, largeur, p1, p2):
    """Les inerties des combinaisons sont imprimées en entiers (VER-727) : à moins de 1 cm⁴, on ne tranche pas."""
    r = nv.verifier(snap, _chic(p1, p2, hauteur_mm=hauteur, largeur_mm=largeur))
    assert _ctl(r, "inertie")["statut"] == "etude" and r["verdict"] == "etude"
    assert "VER-727" in _ctl(r, "inertie")["anomalies"]


def test_identite_des_abaques_affichee(snap):
    """Sur les deux abaques identiques, l'anomalie VER-728 accompagne la lecture et le verdict."""
    for pression, fleche in (("1200", "200"), ("800", "300")):
        r = nv.verifier(snap, _chic("W3041250", "W3041251", pression=pression, fleche=fleche))
        assert "VER-728" in _ctl(r, "abaque")["anomalies"] and "VER-728" in _ctl(r, "inertie")["anomalies"]
    r = nv.verifier(snap, _chic("W3041250", "W3041251"))
    assert "VER-728" not in _ctl(r, "abaque")["anomalies"]


# ---- les montants : la valeur vaut pour 2 000 mm ---------------------------------------------------

def _montants(p1, p2, **k):
    return _saisie(configuration="montants", profil1_montants_32=p1, profil2_montants_32=p2, **k)


def test_montants_a_2000_mm_valeur_imprimee(snap):
    r = nv.verifier(snap, _montants("W1041271", "W1041271", hauteur_mm=2000))
    assert _ctl(r, "combinaison")["valeurs"]["ix"] == 58 and r["verdict"] == "ok"      # 800 Pa, L 1,30, H 2,00 : 19,5


def test_montants_autre_hauteur_graduation_inferieure(snap):
    """H 1 800 mm : graduation 1 600 mm (jamais celle du dessus), somme des deux Ix du profilé."""
    r = nv.verifier(snap, _montants("W1041271", "W1041271", hauteur_mm=1800))
    assert _ctl(r, "combinaison")["valeurs"]["ix"] == pytest.approx(2 * 25.09)
    assert "à 1600 mm, graduation immédiatement inférieure à 1800 mm" in _ctl(r, "combinaison")["detail"]
    # H 2 200 mm : graduation 2 000 mm, non 2 250
    r = nv.verifier(snap, _montants("W1041271", "W1041271", hauteur_mm=2200))
    assert _ctl(r, "combinaison")["valeurs"]["ix"] == pytest.approx(2 * 29.49)
    # deux profilés différents : somme de leurs Ix
    r = nv.verifier(snap, _montants("Y1740040", "Y1740041", hauteur_mm=1600))
    assert _ctl(r, "combinaison")["valeurs"]["ix"] == pytest.approx(29.05 + 38.63)


def test_montants_sous_1200_mm_sur_etude(snap):
    r = nv.verifier(snap, _montants("W1041271", "W1041271", hauteur_mm=1100))
    assert _ctl(r, "combinaison")["statut"] == "etude" and r["verdict"] == "etude"


def test_paire_absente_sur_etude_meme_si_la_somme_se_calcule(snap):
    """W3041241 + W3041242 : les deux chicanes 24 mm existent (Ixx' 29,22 et 29,00), la paire n'est pas au catalogue
    des combinaisons (VER-727 : règle d'arrondi inconnue, la somme n'est pas retenue)."""
    r = nv.verifier(snap, _saisie(module="24", profil1_chicanes_24="W3041241", profil2_chicanes_24="W3041242"))
    assert _ctl(r, "combinaison")["statut"] == "etude" and "n'est pas dans le tableau" in _ctl(r, "combinaison")["detail"]
    assert r["verdict"] == "etude"
    # la paire non ordonnée est reconnue dans les deux sens
    r = nv.verifier(snap, _chic("W3041251", "W3041250"))
    assert _ctl(r, "combinaison")["valeurs"]["ix"] == 39


# ---- le poids et le rail ----------------------------------------------------------------------------

def test_poids_sans_verre_saisi_informe(snap):
    r = nv.verifier(snap, _chic("W3041250", "W3041251", roulette="W6050153"))
    c = _ctl(r, "poids")
    assert c["statut"] == "info" and c["valeurs"]["charge_kg"] == 160 and r["verdict"] == "ok"


def test_poids_du_verre_contre_la_charge(snap):
    # 8 mm × 2,5 kg/m²/mm × 1,30 × 2,20 m = 57,2 kg au plus, contre 80 kg
    r = nv.verifier(snap, _chic("W3041250", "W3041251", roulette="W6050151", verre_mm=8))
    c = _ctl(r, "poids")
    assert c["statut"] == "ok" and c["valeurs"]["poids_kg"] == 57.2 and c["valeurs"]["charge_kg"] == 80
    assert "hypothèse" in c["detail"] and c["valeurs"]["hypothese"].startswith("hypothèse")
    # 14 mm : 100,1 kg de majorant contre 80 kg : le verre réel est plus petit, on ne condamne pas
    r = nv.verifier(snap, _chic("W3041250", "W3041251", roulette="W6050151", verre_mm=14))
    assert _ctl(r, "poids")["statut"] == "etude" and r["verdict"] == "etude"


def test_roulette_w6050155_et_inc_749(snap):
    r = nv.verifier(snap, _chic("W3041250", "W3041251", roulette="W6050155"))
    assert _ctl(r, "poids")["valeurs"]["charge_kg"] == 220 and "INC-749" in _ctl(r, "poids")["anomalies"]
    r = nv.verifier(snap, _chic("W3041250", "W3041251", roulette="W6050151"))
    assert "INC-749" not in _ctl(r, "poids")["anomalies"]


def test_rail_au_dela_de_2250_mm(snap):
    r = nv.verifier(snap, _chic("W3041250", "W3041251", hauteur_mm=2300))
    c = _ctl(r, "rail")
    assert c["statut"] == "info" and "W3041224" in c["detail"] and "W4000158" in c["detail"]
    assert not any(c["id"] == "rail" for c in nv.verifier(snap, _chic("W3041250", "W3041251"))["controles"])


# ---- la forme du résultat et le registre ---------------------------------------------------------------

def test_forme_du_resultat(snap):
    r = nv.verifier(snap, _chic("W3041250", "W3041251"))
    assert {"verdict", "libelle", "contrainte", "controles", "brouillons", "notes", "vantail"} <= set(r)
    assert r["verdict"] == "ok" and r["libelle"] == "Réalisable" and r["contrainte"] is None
    for c in r["controles"]:
        assert set(c) == {"id", "titre", "statut", "detail", "sources", "anomalies", "valeurs"}
        assert c["statut"] in ("ok", "hors", "etude", "info") and c["sources"]
        for s in c["sources"]:
            assert s["page"] in snap.pages
    assert [c["id"] for c in r["controles"]] == ["zone_bleue", "abaque", "combinaison", "inertie", "poids"]
    assert set(r["brouillons"]) >= {nv.PAGES_ABAQUES[0], nv.PAGE_COMBINAISONS, nv.PAGE_QUINCAILLERIE}   # tout ASKEY est en brouillon
    assert "VER-727" in _ctl(r, "combinaison")["anomalies"]
    assert any("VER-729" in c["anomalies"] for c in r["controles"])
    # la contrainte est le contrôle qui décide
    r = nv.verifier(snap, _chic("W3041250", "W3041250"))
    assert r["verdict"] == "hors" and r["contrainte"]["id"] == "inertie"


def test_les_sources_pointent_la_bonne_section(snap):
    r = nv.verifier(snap, _chic("W3041250", "W3041251", pression="1200", fleche="300"))
    assert _ctl(r, "abaque")["sources"][0] == {
        "page": nv.PAGES_ABAQUES[1], "section": "Abaque 1 200 Pa, flèche admissible L/300",
        "localisation": "schéma: raw/askey1.pdf, p. 31"}
    assert _ctl(r, "combinaison")["sources"][0]["section"] == "Chicanes du vantail 32 mm"


# ---- la porte commune des calculateurs ---------------------------------------------------------------------

def test_par_le_registre(snap):
    liste = fs.systemes()
    sysnv = next(s for s in liste["systemes"] if s["id"] == "askey-nv")
    assert sysnv["libelle"] == "ASKEY Coulissant 65 NV" and sysnv["gammes"] == ["Coulissant 65 NV"] and sysnv["legacy"] is False
    assert sysnv["portee"]
    o = fs.options(snap, "askey-nv")
    assert {"champs", "configurations", "pages", "brouillons", "avertissements"} <= set(o)
    ids = [c["id"] for c in o["champs"]]
    assert {"configuration", "largeur_mm", "hauteur_mm", "pression", "fleche", "module", "roulette", "verre_mm"} <= set(ids)
    assert len(ids) == len(set(ids))
    # une saisie par défaut du formulaire (champs masqués à None) passe par la même porte
    brut = {c["id"]: c.get("defaut") for c in o["champs"]}
    for c in o["champs"]:
        if c.get("si") and any(brut[k] not in v for k, v in c["si"].items()):
            brut[c["id"]] = None
    r = fs.verifier(snap, "askey-nv", brut)
    assert r["verdict"] in ("ok", "hors", "etude") and r["controles"]
    f = fs.fiche_debit(snap, "askey-nv", brut)
    assert "indisponible" in f and "Coulissant 65 NV" in f["indisponible"]


def test_saisie_invalide(snap):
    with pytest.raises(ValueError):
        nv.verifier(snap, _saisie(pression="900"))
    with pytest.raises(ValueError):
        nv.verifier(snap, _saisie(profil1_chicanes_32="W9999999"))
    with pytest.raises(ValueError):
        nv.verifier(snap, _saisie(largeur_mm="beaucoup"))
