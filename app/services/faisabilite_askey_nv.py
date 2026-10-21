"""Le calculateur ASKEY Coulissant 65 NV : l'inertie du vantail contre l'inertie que le vent exige (21/10/2026).

Le catalogue de conception du coulissant 65 NV ne porte ni DTA écrit ni limite de baie : le dimensionnement
d'un vantail tient dans les **abaques** (une inertie requise par pression de vent, par flèche admissible, par
hauteur et largeur de vantail), la **zone bleue** que ces abaques appellent « dimensions maximales (DTA) », les
**combinaisons de profilés** qui disent quelle inertie donne une paire de montants ou de chicanes, et la
**charge** que la roulette tient. Le calcul suit ces étapes, dans l'ordre du catalogue :

1. la **zone bleue** : au-delà, les cases de l'abaque existent mais sortent du DTA → « sur étude » ;
2. l'**inertie requise**, lue dans la case de l'abaque, hauteur et largeur arrondies au **cran supérieur**
   (l'inertie requise ne décroît jamais avec la hauteur ni la largeur : le cran supérieur est du côté
   sûr, et le wiki l'écrit : « aucune valeur n'est interpolée entre deux graduations ») ;
3. l'**inertie de la paire de profilés** : valeur imprimée de la combinaison ; pour des montants, elle ne vaut
   que pour 2 000 mm de hauteur, une autre hauteur se lit dans les inerties du profilé à la graduation
   immédiatement inférieure (l'inertie croît avec la longueur) ;
4. le **verdict** : inertie de la paire au moins égale à l'inertie requise, à 1 cm⁴ près (les inerties des
   combinaisons sont imprimées en nombres entiers, sans règle d'arrondi : VER-727) ;
5. la **charge** de la roulette contre le poids du verre, et le rail à ajouter au-dessus de 2 250 mm.

Les trois flèches (L/150, L/200, L/300) sont affichées : le wiki ne dit pas laquelle imposer, la flèche saisie
décide du verdict. Aucune valeur n'est recopiée : tout vient des tableaux du wiki, relus à chaque rechargement
de l'instantané.

Ce que le calculateur ne fait pas : le **vantail** se saisit (largeur et hauteur du vantail), parce que le
wiki ne le définit pas à partir du châssis ; le **débit** des onze applications n'est pas calculé.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services import faisabilite as fa
from app.services import faisabilite_systemes as fs
from app.services import wiki_famille
from app.services.faisabilite import DonneeIntrouvable, _controle, _fmt, _nombre, _source, _texte_section
from app.services.wiki_service import WikiSnapshot
from app.services.wiki_tables import tables_toutes

ID = "askey-nv"
LIBELLE = "ASKEY Coulissant 65 NV"
GAMMES = ["Coulissant 65 NV"]
PORTEE = ("Inertie requise des abaques (800 et 1 200 Pa, trois flèches) contre l'inertie de la paire de profilés du "
          "vantail, zone bleue du DTA, charge de la roulette. Vantail saisi directement ; débit des applications "
          "non calculé ; hors zone bleue ou paire absente du catalogue : sur étude.")

PAGES_ABAQUES = ("/profiles/askey-coulissant-65-nv-abaques-800-pa.md", "/profiles/askey-coulissant-65-nv-abaques-1200-pa.md")
PAGE_COMBINAISONS = "/profiles/askey-coulissant-65-nv-combinaisons-de-profiles-de-vantaux.md"
PAGE_STATIQUES = "/profiles/askey-coulissant-65-nv-valeurs-statiques-ouvrants-et-chicanes.md"
PAGE_QUINCAILLERIE = "/quincaillerie/askey-coulissant-65-nv-quincaillerie.md"
PAGES = (*PAGES_ABAQUES, PAGE_COMBINAISONS, PAGE_STATIQUES, PAGE_QUINCAILLERIE, fa.PAGE_ROTO)

OK, HORS, ETUDE, INFO = fs.OK, fs.HORS, fs.ETUDE, fs.INFO

# Les inerties des combinaisons sont imprimées en nombres entiers (« Ix = 39 cm4 ») : à moins de 1 cm⁴ de
# l'inertie requise, on ne tranche pas.
PRECISION_IX_CM4 = 1.0
HAUTEUR_MONTANT_COMBINAISON_MM = 2000   # « Montants … hauteur 2000mm » : la hauteur des planches de combinaison
CONFIGURATIONS = {"chicanes": "Chicanes (montant central)", "montants": "Montants latéraux"}
MODULES = {"24": "Module 24 mm", "32": "Module 28/32 mm"}
HYPOTHESE_VERRE = ("hypothèse : poids du vantail = verre seul, sur la surface du vantail entier (majorant) ; "
                   "le wiki ne donne pas le poids des profilés")

ENTETE_ABAQUE = "Hauteur de vantail (m)"
ENTETE_COMBINAISON = ["Profilé 1", "Profilé 2", "Ix de la combinaison (cm⁴)"]
ENTETE_CHARGES = ["Type de roulette (libellé imprimé)", "Référence", "Charge maxi par roulette (kg)",
                  "Charge maxi par vantail (kg)"]


# ---- les données, telles que le wiki les écrit ----------------------------------------------

@dataclass
class Abaque:
    pression: int
    fleche: int
    titre: str
    localisation: str
    page: str
    hauteurs: List[float]                      # m, croissantes
    largeurs: List[float]                      # m, croissantes
    cases: Dict[Tuple[float, float], float]    # (hauteur, largeur) -> inertie requise (cm⁴)
    h_bleu: float                              # hauteur maxi de la zone bleue (m)
    l_bleu: float                              # largeur maxi de la zone bleue (m)


@dataclass
class Combinaisons:
    titre: str
    localisation: str
    paires: Dict[Tuple[str, str], float]       # paire non ordonnée (triée) -> Ix (cm⁴)
    refs: List[str]                            # dans l'ordre du tableau


@dataclass
class Statique:
    titre: str
    localisation: str
    graduations: List[int]                     # mm, croissantes
    ix: Dict[str, Dict[int, float]]            # référence -> longueur -> Ix (cm⁴)


@dataclass
class Donnees:
    abaques: Dict[Tuple[int, int], Abaque]
    combinaisons: Dict[Tuple[str, str], Combinaisons]   # (« chicanes » | « montants », module) -> paires
    statiques: Dict[str, Statique]                      # module -> Ix par longueur des montants
    roulettes: Dict[str, Dict[str, Any]]                # référence -> libellé, charges
    localisation_charges: str
    rail_seuil_mm: Optional[int]
    rails: Tuple[str, ...]
    kg_par_m2_mm: Optional[float]
    statuts: Dict[str, str] = field(default_factory=dict)


_CACHE: Dict[str, Any] = {"cle": None, "donnees": None}


def _norm(texte: str) -> str:
    return re.sub(r"[\s  ]+", " ", texte).strip()


def _metres(c: str) -> float:
    n = _nombre(c)
    if n is None:
        raise DonneeIntrouvable(f"valeur illisible dans un abaque du coulissant 65 NV : « {c} »")
    return n


def _corps(snap: WikiSnapshot, page: str) -> str:
    return fa._corps(snap, page)


def _charge(snap: WikiSnapshot) -> Donnees:
    # -- les six abaques : trois flèches par pression, une page par pression
    abaques: Dict[Tuple[int, int], Abaque] = {}
    for page in PAGES_ABAQUES:
        corps = _corps(snap, page)
        for t in tables_toutes(corps):
            if not t.entete or not t.entete[0].startswith(ENTETE_ABAQUE):
                continue
            m = re.fullmatch(r"Abaque ([\d ]+) Pa, flèche admissible L/(\d+).*", _norm(t.section))
            if not m:
                raise DonneeIntrouvable(f"{page} : abaque « {t.section} » illisible")
            pression, fleche = int(re.sub(r"\D", "", m.group(1))), int(m.group(2))
            largeurs = [_metres(c) for c in t.entete[1:]]
            hauteurs, cases = [], {}
            for r in t.lignes:
                h = _metres(r[0])
                hauteurs.append(h)
                for l, c in zip(largeurs, r[1:]):
                    cases[(h, l)] = _metres(c)
            # la zone bleue est écrite sous le titre de l'abaque ; sur la page 1 200 Pa, une fois pour les trois
            zb = re.compile(r"zone bleue(?: des trois abaques)? couvre les hauteurs de ([\d,]+) m à ([\d,]+) m et les largeurs de "
                            r"([\d,]+) m à ([\d,]+) m")
            mz = zb.search(_texte_section(corps, t.section)) or zb.search(corps)
            if not mz:
                raise DonneeIntrouvable(f"{page} : zone bleue de « {t.section} » introuvable")
            ha, hb, la, lb = (_metres(g) for g in mz.groups())
            abaques[(pression, fleche)] = Abaque(pression, fleche, t.section, t.localisation, page, sorted(hauteurs),
                                                 sorted(largeurs), cases, max(ha, hb), max(la, lb))
    attendues = {(p, f) for p in (800, 1200) for f in (150, 200, 300)}
    if set(abaques) != attendues:
        raise DonneeIntrouvable(f"abaques du coulissant 65 NV : {sorted(attendues - set(abaques))} manquants, "
                                f"{sorted(set(abaques) - attendues)} inattendus")

    # -- les combinaisons de profilés : quatre sections
    combinaisons: Dict[Tuple[str, str], Combinaisons] = {}
    for t in tables_toutes(_corps(snap, PAGE_COMBINAISONS)):
        if t.entete[:3] != ENTETE_COMBINAISON:
            continue
        m = re.match(r"(Chicanes|Montants) du vantail (\d+) mm", t.section)
        if not m:
            raise DonneeIntrouvable(f"{PAGE_COMBINAISONS} : section « {t.section} » illisible")
        paires, refs = {}, []
        for r in t.lignes:
            ix = _nombre(r[2])
            if ix is None:
                raise DonneeIntrouvable(f"{PAGE_COMBINAISONS} : Ix illisible pour {r[0]} / {r[1]}")
            paires[tuple(sorted((r[0], r[1])))] = ix
            refs += [x for x in (r[0], r[1]) if x not in refs]
        combinaisons[(m.group(1).lower(), m.group(2))] = Combinaisons(t.section, t.localisation, paires, refs)
    manquantes = {(t, m) for t in CONFIGURATIONS for m in MODULES} - set(combinaisons)
    if manquantes:
        raise DonneeIntrouvable(f"{PAGE_COMBINAISONS} : sections manquantes {sorted(manquantes)}")

    # -- les inerties des montants, par longueur : « ouvrants 24 mm » et « ouvrants 28-32 mm »
    statiques: Dict[str, Statique] = {}
    for t in tables_toutes(_corps(snap, PAGE_STATIQUES)):
        m = re.fullmatch(r"Valeurs statiques des ouvrants (24|28-32) mm", t.section)
        if not m or t.entete[:1] != ["Référence"]:
            continue
        cols = [(k, re.fullmatch(r"Ix à ([\d ]+) mm \(cm⁴\)", _norm(h))) for k, h in enumerate(t.entete) if h.startswith("Ix à")]
        if not cols or any(c is None for _, c in cols):
            raise DonneeIntrouvable(f"{PAGE_STATIQUES} : en-têtes « Ix à … mm » de « {t.section} » illisibles")
        grad = [int(re.sub(r"\D", "", c.group(1))) for _, c in cols]
        ix = {r[0]: {g: _nombre(r[k]) for (k, _), g in zip(cols, grad) if _nombre(r[k]) is not None} for r in t.lignes}
        statiques["24" if m.group(1) == "24" else "32"] = Statique(t.section, t.localisation, sorted(grad), ix)
    if set(statiques) != set(MODULES):
        raise DonneeIntrouvable(f"{PAGE_STATIQUES} : inerties des ouvrants 24 et 28-32 mm introuvables")

    # -- les charges de roulette, le rail au-dessus de 2 250 mm
    quincaillerie = _corps(snap, PAGE_QUINCAILLERIE)
    roulettes, loc = {}, ""
    for t in tables_toutes(quincaillerie):
        if t.entete[:4] == ENTETE_CHARGES:
            loc = t.localisation
            roulettes = {r[1]: {"libelle": r[0], "par_roulette": _nombre(r[2]), "par_vantail": _nombre(r[3])}
                         for r in t.lignes}
    if not roulettes:
        raise DonneeIntrouvable(f"{PAGE_QUINCAILLERIE} : tableau des charges de roulette introuvable")
    mr = re.search(r"Au dessus de (\d+) mm de haut.*?réf\s*:\s*(\w+).*?inox\s+(\w+)", quincaillerie, re.S)
    kg = re.search(r"1 mm/m² d'épaisseur de vitre ≙ ([\d,]+) kg", _corps(snap, fa.PAGE_ROTO))
    return Donnees(abaques, combinaisons, statiques, roulettes, loc,
                   int(mr.group(1)) if mr else None, (mr.group(2), mr.group(3)) if mr else (),
                   _nombre(kg.group(1)) if kg else None,
                   {p: (snap.pages[p].status or "") for p in PAGES})


def donnees(snap: WikiSnapshot) -> Donnees:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        _CACHE["donnees"] = _charge(snap)
        _CACHE["cle"] = cle
    return _CACHE["donnees"]


# ---- le formulaire ----------------------------------------------------------------------------

def _champ_profil(rang: int, type_: str, module: str, c: Combinaisons) -> Dict[str, Any]:
    return {"id": f"profil{rang}_{type_}_{module}",
            "libelle": f"Profilé {rang} ({CONFIGURATIONS[type_].split(' (')[0].lower()}, module {module} mm)",
            "type": "select", "options": {r: r for r in c.refs}, "defaut": c.refs[0],
            "si": {"configuration": [type_], "module": [module]},
            **({"aide": "Une paire absente du catalogue des combinaisons reste sur étude."} if rang == 2 else {})}


def options(snap: WikiSnapshot) -> Dict[str, Any]:
    d = donnees(snap)
    champs: List[Dict[str, Any]] = [
        {"id": "configuration", "libelle": "Profilés verticaux du vantail", "type": "select",
         "options": CONFIGURATIONS, "defaut": "chicanes"},
        {"id": "module", "libelle": "Module du vantail", "type": "select", "options": MODULES, "defaut": "32"},
        {"id": "pression", "libelle": "Pression de vent", "type": "select",
         "options": {str(p): f"{p:,} Pa".replace(",", " ") for p in sorted({k[0] for k in d.abaques})}, "defaut": "800"},
        {"id": "fleche", "libelle": "Flèche admissible", "type": "select",
         "options": {"150": "L/150 (max. 15 mm)", "200": "L/200", "300": "L/300"}, "defaut": "150",
         "aide": "Le catalogue ne dit pas laquelle imposer : les trois sont affichées, celle-ci décide du verdict."},
        {"id": "largeur_mm", "libelle": "Largeur du vantail", "type": "nombre", "unite": "mm", "min": 100, "max": 6000,
         "pas": 10, "defaut": 1300},
        {"id": "hauteur_mm", "libelle": "Hauteur du vantail", "type": "nombre", "unite": "mm", "min": 100, "max": 6000,
         "pas": 10, "defaut": 2200},
    ]
    for (type_, module), c in sorted(d.combinaisons.items(), key=lambda kv: (list(CONFIGURATIONS).index(kv[0][0]), kv[0][1])):
        champs += [_champ_profil(1, type_, module, c), _champ_profil(2, type_, module, c)]
    champs += [
        {"id": "roulette", "libelle": "Roulette", "type": "select", "defaut": next(iter(d.roulettes)),
         "options": {ref: f"{r['libelle']} ({ref}) — {_fmt(r['par_vantail'])} kg par vantail" for ref, r in d.roulettes.items()}},
        {"id": "verre_mm", "libelle": "Épaisseur de verre cumulée (facultatif)", "type": "nombre", "unite": "mm",
         "min": 0, "max": 100, "pas": 0.5, "defaut": None, "requis": False,
         "aide": "Somme des verres, sans les lames d'air : sert au contrôle de la charge de la roulette."},
    ]
    return {
        "champs": champs, "configurations": CONFIGURATIONS, "pages": list(PAGES),
        "brouillons": sorted(p for p, s in d.statuts.items() if s == "draft"),
        "avertissements": [
            "Le wiki ne définit pas le vantail à partir du châssis : saisir la largeur et la hauteur du VANTAIL.",
            "Hauteur et largeur sont arrondies au cran supérieur de l'abaque (0,10 m) ; aucune interpolation.",
            "Les inerties des combinaisons sont imprimées en nombres entiers : à moins de 1 cm⁴ de l'inertie requise, "
            "le verdict reste sur étude.",
        ],
    }


# ---- le calcul ---------------------------------------------------------------------------------

def _m(x: float) -> str:
    return _fmt(x, 2)


def _cran_superieur(grille: List[float], valeur: float) -> Optional[float]:
    return next((g for g in grille if g >= valeur - 1e-9), None)


def inertie_requise(ab: Abaque, hauteur_m: float, largeur_m: float) -> Dict[str, Any]:
    """La case de l'abaque au cran supérieur de la hauteur et de la largeur ; ``None`` au-delà de la grille."""
    h, l = _cran_superieur(ab.hauteurs, hauteur_m), _cran_superieur(ab.largeurs, largeur_m)
    return {"hauteur_lue": h, "largeur_lue": l, "requis": ab.cases[(h, l)] if h is not None and l is not None else None}


def _ix_paire(d: Donnees, type_: str, module: str, p1: str, p2: str, hauteur_mm: float) -> Dict[str, Any]:
    """L'inertie de la paire : valeur imprimée de la combinaison, ou somme des Ix des montants à la longueur
    de graduation immédiatement inférieure à la hauteur (jamais d'interpolation, jamais la graduation au-dessus)."""
    comb = d.combinaisons[(type_, module)]
    src = [_source(PAGE_COMBINAISONS, comb.titre, comb.localisation)]
    ix = comb.paires.get(tuple(sorted((p1, p2))))
    if ix is None:
        return {"ix": None, "raison": f"la paire {p1} / {p2} n'est pas dans le tableau « {comb.titre} » "
                                      "(une somme calculable n'est pas retenue : règle d'arrondi inconnue, VER-727)",
                "sources": src}
    if type_ == "chicanes" or abs(hauteur_mm - HAUTEUR_MONTANT_COMBINAISON_MM) < 1e-6:
        return {"ix": ix, "texte": f"{_fmt(ix)} cm⁴ (valeur imprimée de la combinaison"
                                   + (")" if type_ == "chicanes" else f" à {_fmt(HAUTEUR_MONTANT_COMBINAISON_MM)} mm)"),
                "sources": src}
    st = d.statiques[module]
    src.append(_source(PAGE_STATIQUES, st.titre, st.localisation))
    grad = max((g for g in st.graduations if g <= hauteur_mm), default=None)
    if grad is None:
        return {"ix": None, "raison": f"montants de {_fmt(hauteur_mm)} mm : sous la plus courte graduation d'inertie "
                                      f"({_fmt(st.graduations[0])} mm)", "sources": src}
    valeurs = [st.ix.get(p, {}).get(grad) for p in (p1, p2)]
    if None in valeurs:
        absent = next(p for p, v in zip((p1, p2), valeurs) if v is None)
        return {"ix": None, "raison": f"{absent} n'a pas d'inertie à {_fmt(grad)} mm dans « {st.titre} »", "sources": src}
    return {"ix": round(sum(valeurs), 4),
            "texte": f"{_fmt(sum(valeurs), 2)} cm⁴ (somme des Ix de {p1} et {p2} à {_fmt(grad)} mm, graduation "
                     f"immédiatement inférieure à {_fmt(hauteur_mm)} mm : {_fmt(valeurs[0], 2)} + {_fmt(valeurs[1], 2)})",
            "sources": src}


def _verdict_ix(ix: float, requis: float) -> str:
    marge = round(ix - requis, 3)
    if marge >= PRECISION_IX_CM4:
        return OK
    return HORS if marge <= -PRECISION_IX_CM4 else ETUDE


def _flecheur(d: Donnees, pression: int, fleche: int, h_m: float, l_m: float, ix: Optional[float]) -> Dict[str, Any]:
    """Le résultat d'une flèche : zone bleue, inertie requise, verdict."""
    ab = d.abaques[(pression, fleche)]
    lu = inertie_requise(ab, h_m, l_m)
    bleu = h_m <= ab.h_bleu + 1e-9 and l_m <= ab.l_bleu + 1e-9
    verdict = None
    if ix is not None and lu["requis"] is not None:
        verdict = _verdict_ix(ix, lu["requis"])
        if not bleu and verdict != ETUDE:
            verdict = ETUDE      # hors zone bleue, l'abaque sort du DTA : ni « oui » ni « non » sans étude
    return dict(lu, fleche=fleche, abaque=ab, zone_bleue=bleu, verdict=verdict)


def verifier(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    d = donnees(snap)
    s = fs.coercer(options(snap)["champs"], saisie)
    type_, module, pression, fleche = s["configuration"], s["module"], int(s["pression"]), int(s["fleche"])
    L, H = s["largeur_mm"], s["hauteur_mm"]
    l_m, h_m = L / 1000, H / 1000
    p1, p2 = s[f"profil1_{type_}_{module}"], s[f"profil2_{type_}_{module}"]
    ctl: List[Dict[str, Any]] = []

    paire = _ix_paire(d, type_, module, p1, p2, H)
    choisie = _flecheur(d, pression, fleche, h_m, l_m, paire["ix"])
    ab: Abaque = choisie["abaque"]
    src_ab = _source(ab.page, ab.titre, ab.localisation)
    identite = ["VER-728"] if (pression, fleche) in ((1200, 200), (800, 300)) else []

    # 1. la zone bleue (le DTA des abaques)
    ctl.append(_controle(
        "zone_bleue", "Zone bleue des abaques (dimensions maximales du DTA)", OK if choisie["zone_bleue"] else ETUDE,
        (f"Vantail {_m(l_m)} × {_m(h_m)} m (L × H) dans la zone bleue de l'abaque {pression} Pa, L/{fleche} "
         f"(H ≤ {_m(ab.h_bleu)} m, L ≤ {_m(ab.l_bleu)} m)." if choisie["zone_bleue"] else
         f"Vantail {_m(l_m)} × {_m(h_m)} m (L × H) hors de la zone bleue de l'abaque {pression} Pa, L/{fleche} "
         f"(H ≤ {_m(ab.h_bleu)} m, L ≤ {_m(ab.l_bleu)} m) : les cases existent mais l'abaque sort du DTA, "
         "la demande se traite sur étude."),
        [src_ab], ["VER-729"], {"h_max": ab.h_bleu, "l_max": ab.l_bleu}))

    # 2. l'inertie requise
    if choisie["requis"] is None:
        ctl.append(_controle(
            "abaque", "Inertie requise (abaque)", ETUDE,
            f"Vantail {_m(l_m)} × {_m(h_m)} m : au-delà de la grille de l'abaque (hauteur jusqu'à {_m(ab.hauteurs[-1])} m, "
            f"largeur jusqu'à {_m(ab.largeurs[-1])} m).", [src_ab], identite))
    else:
        rond = "" if (choisie["hauteur_lue"], choisie["largeur_lue"]) == (round(h_m, 6), round(l_m, 6)) else \
            f", arrondi au cran supérieur {_m(choisie['largeur_lue'])} × {_m(choisie['hauteur_lue'])} m"
        ctl.append(_controle(
            "abaque", "Inertie requise (abaque)", INFO,
            f"Abaque {pression} Pa, L/{fleche} : {_fmt(choisie['requis'], 1)} cm⁴ requis pour un vantail "
            f"{_m(l_m)} × {_m(h_m)} m{rond}.", [src_ab],
            (["INC-741"] if (pression, fleche) == (800, 150) else []) + identite,
            {"requis": choisie["requis"], "hauteur_lue": choisie["hauteur_lue"], "largeur_lue": choisie["largeur_lue"]}))

    # 3. l'inertie de la paire
    if paire["ix"] is None:
        ctl.append(_controle("combinaison", "Inertie de la paire de profilés", ETUDE,
                             paire["raison"][0].upper() + paire["raison"][1:] + ".", paire["sources"], ["VER-727"]))
    else:
        ctl.append(_controle("combinaison", "Inertie de la paire de profilés", INFO,
                             f"{p1} + {p2} : {paire['texte']}.", paire["sources"], ["VER-727"], {"ix": paire["ix"]}))

    # 4. le verdict, les trois flèches affichées
    if paire["ix"] is not None and choisie["requis"] is not None:
        etapes = []
        for f in (150, 200, 300):
            r = choisie if f == fleche else _flecheur(d, pression, f, h_m, l_m, paire["ix"])
            if r["requis"] is None:
                etapes.append({"etape": f"Flèche L/{f}", "calcul": "au-delà de la grille de l'abaque", "resultat": "sur étude"})
                continue
            marge = paire["ix"] - r["requis"]
            etapes.append({
                "etape": f"Flèche L/{f}" + (" (retenue)" if f == fleche else ""),
                "calcul": f"requis {_fmt(r['requis'], 1)} cm⁴, paire {_fmt(paire['ix'], 1)} cm⁴, marge {_fmt(marge, 1)}"
                          + ("" if r["zone_bleue"] else ", hors zone bleue"),
                "resultat": {OK: "suffisante", HORS: "insuffisante", ETUDE: "sur étude"}[r["verdict"]]})
        v = choisie["verdict"]
        marge = paire["ix"] - choisie["requis"]
        detail = {
            OK: f"{_fmt(paire['ix'], 1)} cm⁴ pour {_fmt(choisie['requis'], 1)} cm⁴ requis (marge {_fmt(marge, 1)} cm⁴) : "
                f"la paire {p1} + {p2} suffit à {pression} Pa, L/{fleche}.",
            HORS: f"{_fmt(paire['ix'], 1)} cm⁴ pour {_fmt(choisie['requis'], 1)} cm⁴ requis (manque {_fmt(-marge, 1)} cm⁴) : "
                  f"la paire {p1} + {p2} ne suffit pas à {pression} Pa, L/{fleche}.",
            ETUDE: f"{_fmt(paire['ix'], 1)} cm⁴ pour {_fmt(choisie['requis'], 1)} cm⁴ requis : "
                   + ("vantail hors de la zone bleue, l'abaque sort du DTA." if not choisie["zone_bleue"] else
                      f"marge de moins de {_fmt(PRECISION_IX_CM4)} cm⁴, dans l'arrondi des inerties imprimées."),
        }[v]
        ctl.append(_controle("inertie", "Inertie du vantail contre inertie requise", v,
                             detail + " Le catalogue ne dit pas quelle flèche imposer : les trois sont données.",
                             [src_ab] + paire["sources"], ["VER-727"] + identite,
                             {"requis": choisie["requis"], "ix": paire["ix"], "marge": round(marge, 3), "etapes": etapes}))

    # 5. la charge de la roulette et le rail
    ctl.extend(_charge_roulette(d, s, l_m, h_m))
    if d.rail_seuil_mm and H > d.rail_seuil_mm:
        ctl.append(_controle(
            "rail", "Rail en traverse haute", INFO,
            f"Vantail de plus de {_fmt(d.rail_seuil_mm)} mm de haut : rajouter le rail aluminium {d.rails[0]} ou le rail inox "
            f"{d.rails[1]} en traverse haute. La gamme écrit « hauteur de châssis », la quincaillerie « hauteur de "
            "vantail » : à confirmer.", [_source(PAGE_QUINCAILLERIE, "Charge maximale du vantail", d.localisation_charges)]))

    brouillons = [p for p, st in d.statuts.items() if st == "draft"]
    return wiki_famille.relocaliser(snap, fs.synthese(
        ctl, brouillons,
        notes=["Le vantail est saisi : le wiki ne le définit pas à partir du châssis et les formules de débit "
               "s'écrivent en L et H du châssis (aucune liste de coupe pour cette gamme)."],
        vantail={"largeur_mm": L, "hauteur_mm": H, "paire": [p1, p2]}))


def _charge_roulette(d: Donnees, s: Dict[str, Any], l_m: float, h_m: float) -> List[Dict[str, Any]]:
    rou = d.roulettes[s["roulette"]]
    src = [_source(PAGE_QUINCAILLERIE, "Charge maximale du vantail", d.localisation_charges)]
    anomalies = ["INC-749"] if s["roulette"] == "W6050155" else []
    nom = f"{rou['libelle']} ({s['roulette']}) : {_fmt(rou['par_vantail'])} kg par vantail"
    verre = s.get("verre_mm")
    if not verre:
        return [_controle("poids", "Charge de la roulette", INFO,
                          f"{nom}. Épaisseur de verre non saisie : le poids n'est pas contrôlé.", src, anomalies,
                          {"charge_kg": rou["par_vantail"]})]
    if d.kg_par_m2_mm is None:
        return [_controle("poids", "Charge de la roulette", ETUDE,
                          f"{nom}. Le facteur kg par mm et par m² de verre n'est plus lisible dans le wiki.", src, anomalies)]
    poids = verre * d.kg_par_m2_mm * l_m * h_m
    src.append(_source(fa.PAGE_ROTO, "Convertir une épaisseur de vitrage en poids de vantail"))
    detail = (f"{_fmt(poids, 1)} kg de verre au plus ({_fmt(verre, 1)} mm × {_fmt(d.kg_par_m2_mm, 1)} kg/m²/mm × "
              f"{_m(l_m)} × {_m(h_m)} m, vantail entier) pour {nom} ({HYPOTHESE_VERRE}).")
    if poids <= rou["par_vantail"]:
        return [_controle("poids", "Charge de la roulette", OK, detail, src, anomalies,
                          {"poids_kg": round(poids, 1), "charge_kg": rou["par_vantail"], "hypothese": HYPOTHESE_VERRE})]
    # le majorant dépasse : le vitrage réel est plus petit que le vantail, on ne sait pas de combien
    return [_controle("poids", "Charge de la roulette", ETUDE,
                      detail + " Le majorant dépasse la charge : le verre réel, plus petit, peut passer ; à calculer sur "
                               "les dimensions de remplissage.", src, anomalies,
                      {"poids_kg": round(poids, 1), "charge_kg": rou["par_vantail"], "hypothese": HYPOTHESE_VERRE})]


def fiche_debit(snap: WikiSnapshot, saisie: Dict[str, Any]) -> Dict[str, Any]:
    donnees(snap)
    return {"indisponible": "Liste de coupe non calculée pour le Coulissant 65 NV : ses onze applications (formules en L et H "
                            "du châssis, formats irréguliers) ne sont pas encore lues. Ce calculateur contrôle l'inertie "
                            "des vantaux, la zone bleue du DTA et la charge de la roulette."}
