"""Le calculateur du SOLEAL GY (coulissant SOLÉAL55 de la gamme LUMINE), 21/10/2026 — la 2 vantaux sur 2 rails.

Même forme que ``faisabilite_ga`` (dont il réutilise l'évaluateur de formules, la lecture des courbes et des
nomenclatures, les contrôles de charge et de roulettes) : toutes les valeurs sont lues dans les tableaux du wiki.

Le wiki ne permet l'enveloppe que pour la **2 vantaux 2 rails** ; toute autre typologie (3, 4, 6, 8 vantaux,
galandage, angle, composée) est « sur étude » :

- **le DTA 6/15-2261** borne la baie (fenêtre 1,65 × 1,85 m, porte-fenêtre 2,50 × 3,00 m) et nomme les montants
  centraux **à employer selon l'épaisseur du vitrage** ; une porte-fenêtre doit recevoir des montants latéraux
  renforcés. La planche d'application pose pourtant TGY1207 + TGY1208, une paire absente du DTA : le choix des
  montants est donc une saisie, et une paire qui n'est pas celle du DTA est « sur étude » ;
- **les courbes d'utilisation A2 / A3** donnent la hauteur maximale selon L pour chaque paire de montants centraux
  et de montants latéraux. Le wiki ne dit pas de quel côté de la courbe est le domaine admis (VER-455) : hypothèse
  « H sous la courbe », affichée. Jamais d'interpolation, ±25 mm de précision de lecture ;
- **les limites imprimées** (200 kg par ouvrant, demi-périmètre de 5 m), le vitrage, les roulettes (unité du poids
  non imprimée, VER-203) et la cale TGA3817 (conditions sans connecteur, VER-204) sont traités comme pour le
  LUMEAL GA ;
- **la liste de coupe** de la porte-fenêtre 2 vantaux (24 à 28 mm) : la fenêtre 2 vantaux n'a pas de nomenclature
  dans le wiki, et le seuil PMR a ses débits sur une autre page que la planche.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from app.services import wiki_famille
from app.services.faisabilite import (DonneeIntrouvable, ETUDE, HORS, INFO, OK, _controle, _corps, _fmt, _nombre, _source,
                                      _une, tables)
from app.services.faisabilite_ga import (Application, Ligne, construire_fiche, controle_roulettes, controles_limites,
                                         densite_du_verre, dimensions, graphe_du_controle, images_par_reference, lire_colonnes, lire_nomenclature,
                                         lire_roulettes, lire_sur_courbe, ligne_de, limites_sous_les_courbes,
                                         paire_de, poids_du_verre)
from app.services.faisabilite_systemes import coercer, synthese
from app.services.wiki_service import WikiSnapshot
from app.services.wiki_tables import tables_toutes

ID = "soleal-gy"
LIBELLE = "SOLEAL GY (coulissant LUMINE)"
GAMMES = ["LUMINE"]
PORTEE = ("Fenêtre et porte-fenêtre 2 vantaux sur 2 rails : DTA 6/15-2261 (dimensions maximales, montants centraux "
          "selon l'épaisseur du vitrage, montants latéraux renforcés), courbes d'utilisation au vent A2 et A3 des "
          "montants centraux et latéraux, limites de charge et de demi-périmètre, vitrage, roulettes. Liste de coupe de "
          "la porte-fenêtre 2 vantaux (vitrage de 24 à 28 mm). Toutes les autres typologies (3, 4, 6, 8 vantaux, "
          "galandage, angle, composée) : « sur étude ».")

PAGE_DTA = "/certifications/dta-6-15-2261-soleal-gy.md"
PAGE_DTA_DOSSIER = "/certifications/dta-6-15-2261-soleal-gy-dossier-technique.md"
PAGE_COURBES = "/profiles/soleal-gy-courbes-d-utilisation.md"
PAGE_APPLICATION = "/profiles/soleal-gy-application-2-vantaux.md"
PAGE_DEBIT = "/profiles/soleal-gy-cotes-de-debit.md"
PAGE_VOLUME = "/profiles/soleal-gy-prises-de-volume-et-joints-de-vitrage.md"
PAGE_ROULETTES = "/procedures/pose-soleal-gy-choix-des-roulettes.md"
PAGE_VERRE = "/profiles/systeme-70-abaques-dimensionnels.md"
PAGES = (PAGE_DTA, PAGE_DTA_DOSSIER, PAGE_COURBES, PAGE_APPLICATION, PAGE_DEBIT, PAGE_VOLUME, PAGE_ROULETTES, PAGE_VERRE)

# L'intitulé de chaque menuiserie dans le tableau du DTA.
MENUISERIES = {
    "fenetre2": "Fenêtre 2 vantaux",
    "pf2": "Porte-fenêtre 2 vantaux",
    "pf2_pmr": "Porte-fenêtre 2 vantaux + Seuil PMR",
}
AUTRE = "autre"
LIBELLE_AUTRE = "Autre typologie (3, 4, 6, 8 vantaux, galandage, angle, composée)"
SECTIONS = {"profils": "Profilés", "joints": "Joints", "accessoires": "Accessoires", "vis": "Vis",
            "remplissage": "Remplissage"}
HYPOTHESE_COURBE = "hypothèse : le domaine admis est sous la courbe, H inférieur à la valeur lue (VER-455)"
HYPOTHESE_UNITE = "hypothèse : le poids maxi du tableau est en kg par vantail (unité non imprimée, VER-203)"
HYPOTHESE_BAIE = "hypothèse : la baie est prise égale aux dimensions hors tout saisies"


# ---- les données lues dans le wiki ------------------------------------------------------------


@dataclass
class LigneDta:
    menuiserie: str
    paires: Dict[Tuple[int, int], Optional[Tuple[str, str]]]    # (épaisseur mini, maxi) → paire ; ``None`` : illisible
    imprime: Dict[Tuple[int, int], str]                          # la cellule telle que le DTA l'écrit
    h_mm: float
    l_mm: float


@dataclass
class Donnees:
    statuts: Dict[str, str]
    dta: Dict[str, LigneDta]
    dta_localisation: str
    dta_validite: str
    dta_vitrage: Tuple[float, float]
    lateraux_porte_fenetre: List[str]                            # « montants latéraux renforcés réf. TGY1303 ou TGY1304 »
    courbes: Dict[str, Dict[str, Dict[str, Any]]]                # classe → « centraux » / « lateraux » → courbes
    limites: Dict[str, float]
    application: Application
    plages: Dict[str, Tuple[float, float]]                       # traverse → épaisseur de vitrage
    joints_volume: Dict[int, Dict[str, str]]
    loc_volume: str
    roulettes: Dict[str, Dict[str, Any]]
    loc_roulettes: str
    cale: List[List[str]]
    densite_verre: Optional[float]
    images: Dict[str, str]


_CACHE: Dict[str, Any] = {"cle": None, "donnees": None}


def donnees(snap: WikiSnapshot) -> Donnees:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        _CACHE["donnees"] = _charge(snap)
        _CACHE["cle"] = cle
    return _CACHE["donnees"]


def _paire_dta(cellule: str) -> Optional[Tuple[str, str]]:
    """« TGY1207 + TGY1209 » → la paire ; « TGY12013 + TGY1215 » (INC-382) → ``None`` : une référence qui n'existe pas."""
    refs = re.findall(r"TGY\d+", cellule)
    if len(refs) != 2 or any(not re.fullmatch(r"TGY\d{4}", r) for r in refs):
        return None
    return tuple(sorted(refs))  # type: ignore[return-value]


def _charge(snap: WikiSnapshot) -> Donnees:
    # -- le DTA
    corps = _corps(snap, PAGE_DTA)
    t = _une(tables(corps), "Édition du DTA | Menuiserie | Montants centraux, vitrages de", PAGE_DTA)
    if t.entete[4:] != ["H (m)", "L (m)"]:
        raise DonneeIntrouvable(f"{PAGE_DTA} : en-têtes du tableau des dimensions maximales changés : {t.entete}")
    tranches = []
    for h in t.entete[2:4]:
        m = re.search(r"vitrages de (\d+) à (\d+) mm", h)
        if not m:
            raise DonneeIntrouvable(f"{PAGE_DTA} : épaisseur de vitrage illisible dans « {h} »")
        tranches.append((int(m.group(1)), int(m.group(2))))
    dta: Dict[str, LigneDta] = {}
    for r in t.lignes:
        dta.setdefault(r[1], LigneDta(r[1], {tranches[0]: _paire_dta(r[2]), tranches[1]: _paire_dta(r[3])},
                                      {tranches[0]: r[2], tranches[1]: r[3]},
                                      round(_nombre(r[4]) * 1000), round(_nombre(r[5]) * 1000)))
    ml = re.search(r"montants latéraux renforcés réf\. (TGY\d+) ou (TGY\d+)", corps)
    lateraux = [ml.group(1), ml.group(2)] if ml else []
    mval = re.search(r"valide du (\d{2}/\d{2}/\d{4}) au (\d{2}/\d{2}/\d{4})", snap.pages[PAGE_DTA].raw_text)
    mv = re.search(r"Isolant double de (\d+) à (\d+) mm d'épaisseur", _corps(snap, PAGE_DTA_DOSSIER))
    if not mv:
        raise DonneeIntrouvable(f"{PAGE_DTA_DOSSIER} : épaisseur du vitrage isolant introuvable")

    # -- les courbes : centraux et latéraux, par classe de vent
    corps_c = _corps(snap, PAGE_COURBES)
    limites = limites_sous_les_courbes(corps_c, PAGE_COURBES)
    courbes: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tb in tables_toutes(corps_c):
        m = re.match(r"Montants (centraux|latéraux) .*classe (A\d), vent (\d+) Pa", tb.section)
        if not (m and tb.entete and tb.entete[0].startswith("L (mm)")):
            continue
        colonnes, entetes = {}, {}
        for entete, courbe in lire_colonnes(tb):
            paire = paire_de(entete)
            if paire:
                colonnes[paire] = courbe
                entetes[paire] = entete
        courbes.setdefault(m.group(2), {})["centraux" if m.group(1) == "centraux" else "lateraux"] = {
            "section": tb.section, "localisation": tb.localisation, "pression": int(m.group(3)),
            "colonnes": colonnes, "entetes": entetes}
    if not courbes or any(set(v) != {"centraux", "lateraux"} for v in courbes.values()):
        raise DonneeIntrouvable(f"{PAGE_COURBES} : tableaux de montants centraux et latéraux, classes A2 et A3, introuvables")

    # -- la nomenclature de la porte-fenêtre 2 vantaux
    application = lire_nomenclature(_corps(snap, PAGE_APPLICATION), None, PAGE_APPLICATION, "Remplissage",
                                    id_="pf2", libelle=MENUISERIES["pf2"])

    # -- traverses, joints, roulettes, cale
    plages = {m.group(1): (float(m.group(2)), float(m.group(3)))
              for m in re.finditer(r"(T\d{6}) \(vitrage de (\d+) à (\d+) mm\)", _corps(snap, PAGE_DEBIT))}
    tv = _une(tables(_corps(snap, PAGE_VOLUME)), "Référence | Joint pour vitrage de (mm)", PAGE_VOLUME)
    joints = {int(_nombre(r[1])): {"ref": r[0], "couleur": r[2], "ouvrant": r[3]} for r in tv.lignes
              if _nombre(r[1]) is not None}
    tr = tables(_corps(snap, PAGE_ROULETTES))
    t_rou = _une(tr, "Référence | Groupe imprimé | Type imprimé | Poids maxi par vantail", PAGE_ROULETTES)
    t_cale = _une(tr, "Prise de volume | Profil de réf. | Condition de rajout de la cale TGA3817", PAGE_ROULETTES)
    return Donnees(
        statuts={p: (snap.pages[p].status or "") for p in PAGES if p in snap.pages},
        dta=dta, dta_localisation=t.localisation, dta_validite=f"{mval.group(1)} au {mval.group(2)}" if mval else "",
        dta_vitrage=(float(mv.group(1)), float(mv.group(2))), lateraux_porte_fenetre=lateraux, courbes=courbes,
        limites=limites, application=application, plages=plages, joints_volume=joints, loc_volume=tv.localisation,
        roulettes=lire_roulettes(t_rou), loc_roulettes=t_rou.localisation, cale=t_cale.lignes,
        densite_verre=densite_du_verre(snap), images=images_par_reference(snap, "soleal-gy"))


# ---- le formulaire ------------------------------------------------------------------------------


def _paires_centraux(d: Donnees) -> List[Tuple[str, str]]:
    ps = {p for c in d.courbes.values() for p in c["centraux"]["colonnes"]}
    ps |= {p for l in d.dta.values() for p in l.paires.values() if p}
    return sorted(ps)


def _lateraux(d: Donnees) -> List[str]:
    refs = {p[0] for c in d.courbes.values() for p in c["lateraux"]["colonnes"] if p[0] == p[1]}
    return sorted(refs | set(d.lateraux_porte_fenetre))


def paire_application(app: Application) -> Optional[Tuple[str, str]]:
    """La paire de montants centraux posée par la planche d'application."""
    refs = [l.ref for l in app.profils for _ in range(int(l.qte))
            if l.designation.lower().startswith("montant cent")]
    return tuple(sorted(refs)) if len(refs) == 2 else None  # type: ignore[return-value]


def lateral_application(app: Application) -> Optional[str]:
    l = ligne_de(app, "Montant latéral")
    return l.ref if l else None


def _lib(p: Tuple[str, str]) -> str:
    return " + ".join(p)


def options(snap: WikiSnapshot) -> Dict[str, Any]:
    d = donnees(snap)
    app_paire = paire_application(d.application)
    configurations = dict(MENUISERIES)
    configurations[AUTRE] = LIBELLE_AUTRE
    suivantes = list(MENUISERIES)
    montants = {"dta": "Ceux du DTA pour cette épaisseur de vitrage"}
    for p in _paires_centraux(d):
        montants[p[0] + "+" + p[1]] = _lib(p) + (" (planche d'application)" if p == app_paire else "")
    lat_planche = lateral_application(d.application)
    lateraux = {"dta": "Ceux du DTA (porte-fenêtre : renforcés), sinon celui de la planche d'application"}
    lateraux.update({r: r + (" (planche d'application)" if r == lat_planche else "") for r in _lateraux(d)})
    vents = {c: f"{c} : vent {v['centraux']['pression']} Pa" for c, v in sorted(d.courbes.items())}
    epaisseurs = {str(e): f"{e} mm" for e in sorted(d.joints_volume)}
    champs = [
        {"id": "configuration", "libelle": "Typologie", "type": "select", "options": configurations,
         "defaut": "pf2"},
        {"id": "largeur_mm", "libelle": "Largeur hors tout L", "type": "nombre", "min": 500, "max": 10000,
         "defaut": 2400, "unite": "mm"},
        {"id": "hauteur_mm", "libelle": "Hauteur hors tout H", "type": "nombre", "min": 500, "max": 10000,
         "defaut": 2150, "unite": "mm"},
        {"id": "epaisseur_vitrage_mm", "libelle": "Épaisseur du double vitrage", "type": "select",
         "options": epaisseurs, "defaut": "24", "si": {"configuration": suivantes}},
        {"id": "verre_mm", "libelle": "Verre seul (somme des verres, sans lame)", "type": "nombre", "min": 0, "max": 60,
         "defaut": 8, "unite": "mm", "si": {"configuration": suivantes},
         "aide": "4-16-4 : 8 mm de verre. Sert au poids du vitrage."},
        {"id": "montants", "libelle": "Montants centraux d'ouvrant", "type": "select", "options": montants,
         "defaut": "dta", "si": {"configuration": suivantes},
         "aide": "Le DTA nomme la paire à employer selon l'épaisseur du vitrage ; la planche d'application en pose une autre."},
        {"id": "lateral", "libelle": "Montants latéraux d'ouvrant", "type": "select", "options": lateraux,
         "defaut": "dta", "si": {"configuration": suivantes},
         "aide": "Le DTA impose des latéraux renforcés TGY1303 ou TGY1304 sur une porte-fenêtre ; la planche pose TGY1202."},
        {"id": "vent", "libelle": "Classe de vent", "type": "select", "options": vents, "defaut": "A2",
         "si": {"configuration": suivantes}},
    ]
    return {
        "champs": champs, "configurations": configurations, "pages": list(PAGES),
        "brouillons": sorted(p for p, s in d.statuts.items() if s == "draft"),
        "avertissements": [a for a in (
            f"La planche d'application pose {_lib(app_paire)}, une paire absente du tableau du DTA." if app_paire else "",
            "Le wiki ne dit pas de quel côté de la courbe est le domaine admis : le calcul suppose « H sous la courbe » "
            "(VER-455).",
            "Le wiki ne donne ni le poids des profilés ni la largeur du cadre ouvrant : les contrôles de poids et de "
            "demi-périmètre ne peuvent que constater un dépassement.",
            f"Validité du DTA : du {d.dta_validite}." if d.dta_validite else "") if a],
    }


# ---- la faisabilité ------------------------------------------------------------------------------


def _tranche(d: Donnees, n: int) -> Optional[Tuple[int, int]]:
    ligne = next(iter(d.dta.values()))
    return next((t for t in ligne.paires if t[0] <= n <= t[1]), None)


def _choix(d: Donnees, s: Dict[str, Any]) -> Dict[str, Any]:
    """Les montants que l'utilisateur emploie : la paire de montants centraux et la référence du latéral."""
    n = int(s["epaisseur_vitrage_mm"])
    tranche = _tranche(d, n)
    ligne = d.dta[MENUISERIES[s["configuration"]]]
    requise = ligne.paires.get(tranche) if tranche else None
    if s["montants"] == "dta":
        paire, comment = requise, "ceux du DTA pour cette épaisseur"
    else:
        a, b = s["montants"].split("+")
        paire, comment = (a, b), "ceux choisis"
    if s["lateral"] == "dta":
        # le DTA ne parle de latéraux que pour une porte-fenêtre ; pour une fenêtre il se tait : celui de la planche
        lateral = (d.lateraux_porte_fenetre[0] if ligne.menuiserie.lower().startswith("porte-fenêtre")
                   and d.lateraux_porte_fenetre else lateral_application(d.application))
    else:
        lateral = s["lateral"]
    return {"tranche": tranche, "requise": requise, "imprime": ligne.imprime.get(tranche, ""), "paire": paire,
            "comment": comment, "lateral": lateral}


def _pire(*statuts: str) -> str:
    return max(statuts, key=[OK, ETUDE, HORS].index)


def _controle_dta(d: Donnees, s: Dict[str, Any], choix: Dict[str, Any]) -> Dict[str, Any]:
    titre = "Dimensions maximales de baie et montants (DTA)"
    cle = s["configuration"]
    ligne = d.dta[MENUISERIES[cle]]
    src = [_source(PAGE_DTA, "Dimensions maximales", d.dta_localisation)]
    anomalies = ["CTR-139", "VER-178"] + (["INC-382"] if cle == "pf2_pmr" else [])
    H, L = s["hauteur_mm"], s["largeur_mm"]
    parties, statuts = [], []
    # la baie
    dans = H <= ligne.h_mm and L <= ligne.l_mm
    statuts.append(OK if dans else HORS)
    parties.append(f"Baie {_fmt(H / 1000, 2)} × {_fmt(L / 1000, 2)} m (H × L), limite {_fmt(ligne.h_mm / 1000, 2)} × "
                   f"{_fmt(ligne.l_mm / 1000, 2)} m pour une {ligne.menuiserie.lower()} ({HYPOTHESE_BAIE}) : "
                   + ("dans la limite." if dans else "dépassée ; au-delà, seul le Certificat de Qualification du "
                                                     "menuisier peut l'admettre."))
    # les montants centraux selon l'épaisseur du vitrage
    n = int(s["epaisseur_vitrage_mm"])
    if choix["requise"] is None:
        statuts.append(ETUDE)
        parties.append(f"Montants centraux : le DTA écrit « {choix['imprime']} » pour {n} mm, une référence qui n'existe "
                       "pas (INC-382) ; la paire à employer n'est pas lisible.")
    elif choix["paire"] == choix["requise"]:
        statuts.append(OK)
        parties.append(f"Montants centraux {_lib(choix['paire'])} ({choix['comment']}) : ceux que le DTA nomme pour "
                       f"{n} mm.")
    else:
        statuts.append(ETUDE)
        parties.append(f"Montants centraux {_lib(choix['paire'] or ('?', '?'))} ({choix['comment']}) : le DTA nomme "
                       f"{_lib(choix['requise'])} pour {n} mm ; cette paire est absente du tableau du DTA.")
    # les latéraux renforcés d'une porte-fenêtre
    if ligne.menuiserie.lower().startswith("porte-fenêtre") and d.lateraux_porte_fenetre:
        ok_lat = choix["lateral"] in d.lateraux_porte_fenetre
        statuts.append(OK if ok_lat else ETUDE)
        refs = " ou ".join(d.lateraux_porte_fenetre)
        parties.append(f"Montants latéraux {choix['lateral']} : " + (
            f"un des latéraux renforcés ({refs}) avec lesquels le DTA fabrique les portes-fenêtres." if ok_lat else
            f"le DTA fabrique les portes-fenêtres avec les latéraux renforcés {refs} ; ce montant n'y est pas nommé."))
    detail = " ".join(parties) + (f" DTA valide du {d.dta_validite}." if d.dta_validite else "")
    return _controle("dta", titre, _pire(*statuts), detail, src, anomalies,
                     {"limite": {"h": ligne.h_mm, "l": ligne.l_mm}, "paire_requise": list(choix["requise"] or []),
                      "paire": list(choix["paire"] or []), "lateral": choix["lateral"], "hypothese": HYPOTHESE_BAIE})


def _courbe(d: Donnees, s: Dict[str, Any], role: str, paire: Optional[Tuple[str, str]], titre: str, id_: str,
            anomalies: List[str]) -> Dict[str, Any]:
    cb = d.courbes[s["vent"]][role]
    src = [_source(PAGE_COURBES, cb["section"], cb["localisation"])]
    if paire is None or paire not in cb["colonnes"]:
        tracees = " ; ".join(_lib(p) for p in sorted(cb["colonnes"]))
        return _controle(id_, titre, ETUDE,
                         f"{_lib(paire) if paire else 'Paire non lisible'} : pas de courbe sur cet abaque (tracées : "
                         f"{tracees}).", src, ["VER-455"] + anomalies)
    lecture = lire_sur_courbe(cb["colonnes"][paire], s["largeur_mm"], s["hauteur_mm"], d.limites["cadre"],
                              d.limites["precision"])
    detail = (f"Classe {s['vent']} ({cb['pression']} Pa), {cb['entetes'][paire]} : hauteur "
              f"{_fmt(s['hauteur_mm'])} mm pour L = {_fmt(s['largeur_mm'])} mm. "
              f"{lecture.detail[0].upper() + lecture.detail[1:]} ({HYPOTHESE_COURBE}).")
    return _controle(id_, titre, lecture.statut, detail, src, ["VER-455"] + anomalies,
                     {"h_max": None if lecture.h_max in (None, float("inf")) else lecture.h_max,
                      "paire": list(paire), "classe": s["vent"]})


def _controles_courbes(d: Donnees, s: Dict[str, Any], choix: Dict[str, Any]) -> List[Dict[str, Any]]:
    paire = choix["paire"]
    inerties = ["VER-457"] if paire and "TGY1208" in paire else []
    lat = choix["lateral"]
    return [_courbe(d, s, "centraux", paire, "Résistance au vent, montants centraux", "courbe", inerties),
            _courbe(d, s, "lateraux", (lat, lat) if lat else None, "Résistance au vent, montants latéraux",
                    "courbe_lateraux", [])]


def _controle_vitrage(d: Donnees, s: Dict[str, Any]) -> Dict[str, Any]:
    titre = "Vitrage et joint"
    n = int(s["epaisseur_vitrage_mm"])
    j = d.joints_volume.get(n)
    src = [_source(PAGE_VOLUME, "Joints de vitrage", d.loc_volume), _source(PAGE_DTA_DOSSIER, "Vitrage")]
    if j is None:
        return _controle("vitrage", titre, HORS, f"Aucune prise de volume de {n} mm.", src)
    base = f"Double vitrage de {n} mm : joint {j['ref']} (repère {j['couleur'].lower()}, ouvrant {j['ouvrant']})."
    valeurs = {"joint": j["ref"], "couleur": j["couleur"], "epaisseur": n}
    verre = s["verre_mm"] or 0
    lo, hi = d.dta_vitrage
    if verre >= n:
        return _controle("vitrage", titre, ETUDE, base + f" Le verre seul ({_fmt(verre, 1)} mm) n'est pas inférieur à "
                                                         "l'épaisseur du vitrage.", src, [], valeurs)
    if not lo <= n <= hi:
        return _controle("vitrage", titre, HORS, base + f" Le DTA admet de {_fmt(lo)} à {_fmt(hi)} mm.", src, [], valeurs)
    traverse = ligne_de(d.application, "Traverse basse")
    plage = d.plages.get(traverse.ref) if traverse else None
    suite = ""
    if plage and not plage[0] <= n <= plage[1]:
        suite = (f" La planche d'application porte la traverse basse {traverse.ref} ({_fmt(plage[0])} à "
                 f"{_fmt(plage[1])} mm) : la liste de coupe n'est pas établie à cette épaisseur.")
    return _controle("vitrage", titre, OK, base + f" Épaisseur admise par le DTA ({_fmt(lo)} à {_fmt(hi)} mm)." + suite,
                     src, [], valeurs)


def _controle_cale(d: Donnees) -> List[Dict[str, Any]]:
    """La cale TGA3817 : deux conditions imprimées sur deux tirets, sans « et » ni « ou » (VER-204)."""
    traverse = ligne_de(d.application, "Traverse basse")
    lignes = [r for r in d.cale if traverse and r[1].strip() == traverse.ref]
    if not lignes:
        return []
    conditions = " ; ".join(r[2] for r in lignes)
    return [_controle("cale", "Cale de vitrage TGA3817", INFO,
                      f"Traverse basse {traverse.ref} (prise de volume {lignes[0][0]}) : {conditions}. Les deux conditions "
                      "sont écrites sans connecteur : le calcul ne décide pas si l'une suffit ou si les deux sont requises.",
                      [_source(PAGE_ROULETTES, "Cale TGA3817")], ["VER-204", "VER-480"], {"traverse": traverse.ref})]


def _brouillons(d: Donnees, pages: List[str]) -> List[str]:
    return [p for p in dict.fromkeys(pages) if d.statuts.get(p) == "draft"]


def verifier(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = coercer(options(snap)["champs"], saisie)
    if s["configuration"] == AUTRE:
        return wiki_famille.relocaliser(snap, synthese(
            [_controle("typologie", "Typologie", ETUDE,
                       "Le wiki ne donne l'enveloppe (DTA et courbes d'utilisation) que pour la fenêtre et la "
                       "porte-fenêtre à 2 vantaux sur 2 rails : 3, 4, 6 ou 8 vantaux, galandage, angle et châssis composé "
                       "n'ont ni dimension maximale ni courbe.", [_source(PAGE_DTA, "Domaine d'emploi accepté")], [])],
            _brouillons(d, [PAGE_DTA])))
    app = d.application
    variables = {"H": s["hauteur_mm"], "L": s["largeur_mm"]}
    choix = _choix(d, s)
    dim = dimensions(app, variables)
    ctl = [_controle_dta(d, s, choix)]
    ctl += _controles_courbes(d, s, choix)
    ctl += controles_limites(d.densite_verre, d.limites, s["verre_mm"], dim,
                             [_source(PAGE_COURBES, "Courbes d'utilisation")])
    ctl.append(_controle_vitrage(d, s))
    poids, raison = poids_du_verre(d.densite_verre, s["verre_mm"], dim)
    ctl.append(controle_roulettes(d.roulettes, [_source(PAGE_ROULETTES, "Choix des roulettes", d.loc_roulettes)], app,
                                  poids, raison, ["VER-203"], HYPOTHESE_UNITE))
    ctl += _controle_cale(d)
    n = int(s["epaisseur_vitrage_mm"])
    return wiki_famille.relocaliser(snap, synthese(
        ctl, _brouillons(d, [PAGE_DTA, PAGE_COURBES, PAGE_VOLUME, PAGE_ROULETTES, PAGE_APPLICATION]),
        application=MENUISERIES[s["configuration"]],
        vitrage={"epaisseur_mm": n, "verre_mm": s["verre_mm"], "joint": d.joints_volume[n]["ref"]},
        remplissages=[{"libelle": r["libelle"], "hauteur": r["h"], "largeur": r["l"]} for r in dim["remplissages"]],
        montants={"centraux": list(choix["paire"] or []), "lateral": choix["lateral"]},
        graphe_courbes=graphe_du_controle(ctl, lambda classe: d.courbes[classe]["centraux"], s["largeur_mm"], s["hauteur_mm"]),
        poids_verre_kg=None if poids is None else round(poids, 1)))


# ---- la fiche de débit ---------------------------------------------------------------------------


def _avec_montants(app: Application, choix: Dict[str, Any], notes: List[str]) -> Application:
    """La nomenclature de la planche, avec les montants centraux et latéraux choisis : ils se débitent tous à H-77
    (page des règles de débit) ; la planche n'en dessine que quelques-uns, et le dit-on."""
    profils = list(app.profils)

    def remplace(prefixe: str, designation: str, refs: List[str], note: str) -> None:
        idx = [i for i, l in enumerate(profils) if l.designation.lower().startswith(prefixe)]
        if not idx:
            return
        modele = profils[idx[0]]
        nouvelles = [Ligne(r, designation, str(refs.count(r)), modele.debit, modele.coupe) for r in dict.fromkeys(refs)]
        for i in reversed(idx[1:]):
            del profils[i]
        profils[idx[0]:idx[0] + 1] = nouvelles
        notes.append(note)

    if choix["paire"] and choix["paire"] != paire_application(app):
        remplace("montant cent", "Montant central", list(choix["paire"]),
                 f"Montants centraux {_lib(choix['paire'])} : la planche pose {_lib(paire_application(app) or ('?', '?'))} ; "
                 "le débit H-77 est celui que la page des règles de débit donne à tous les montants centraux.")
    lat = choix["lateral"]
    if lat and lat != lateral_application(app):
        remplace("montant lat", "Montant latéral", [lat] * 2,
                 f"Montant latéral {lat} : la planche pose {lateral_application(app)} ; le débit H-77 est attribué à ce "
                 "montant par la page des règles de débit, sans planche d'application qui le confirme.")
    return replace(app, profils=profils)


def fiche_debit(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = coercer(options(snap)["champs"], saisie)
    if s["configuration"] == AUTRE:
        return {"indisponible": "Aucune nomenclature dans ce calcul pour cette typologie : seules la fenêtre et la "
                                "porte-fenêtre 2 vantaux sont traitées."}
    if s["configuration"] == "fenetre2":
        return {"indisponible": "Le wiki n'a pas de nomenclature de la fenêtre 2 vantaux : la planche d'application du "
                                "catalogue est celle de la porte-fenêtre 2 vantaux (fermeture 3 points, roulettes doubles)."}
    if s["configuration"] == "pf2_pmr":
        return {"indisponible": "La planche d'application de la porte-fenêtre 2 vantaux ne porte pas le seuil PMR : ses "
                                "débits (TGY2100, TGY3608, TGY2531) sont sur la page du seuil PMR, que ce calcul ne "
                                "compose pas (VER-467, VER-468)."}
    app = d.application
    n = int(s["epaisseur_vitrage_mm"])
    traverse = ligne_de(app, "Traverse basse")
    plage = d.plages.get(traverse.ref) if traverse else None
    if plage and not plage[0] <= n <= plage[1]:
        return {"indisponible": f"La planche d'application porte la traverse basse {traverse.ref} ({_fmt(plage[0])} à "
                                f"{_fmt(plage[1])} mm) : le wiki n'a pas les débits de la traverse à {n} mm pour cette "
                                "typologie."}
    ver = verifier(snap, saisie)
    notes: List[str] = []
    app_choisie = _avec_montants(app, _choix(d, s), notes)
    fiche = construire_fiche(app_choisie, {"H": s["hauteur_mm"], "L": s["largeur_mm"]}, n, d.joints_volume[n], d.images,
                             SECTIONS, 2, "un par vantail (porte-fenêtre 2 vantaux ; la planche n'écrit pas de quantité)")
    fiche["notes"] += notes
    return wiki_famille.relocaliser(snap, {
        "verdict": ver["verdict"], "libelle": ver["libelle"], "lignes": fiche["lignes"],
        "nomenclature": fiche["nomenclature"], "accessoires": [], "notes": fiche["notes"],
        "brouillons": _brouillons(d, [PAGE_APPLICATION, PAGE_VOLUME]), "application": MENUISERIES[s["configuration"]]})
