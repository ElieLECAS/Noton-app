"""La fiche de débit PERFORM76 (audit du 22/09, § 9.2) — premier jet.

La même saisie que le vérificateur de faisabilité, une autre sortie : **ce que l'atelier coupe et
ce qu'il commande**. Tout vient des pages du wiki :

- les **cotes à déduire** de `systeme-76-cotes-de-debit.md`, une colonne par pièce (renfort,
  meneau, parclose, vitrage…), appliquées **une fois par coupe** comme le manuel le dit ;
- ce que la faisabilité a décidé : la **zone de renforcement** (quels profilés renforcer), le
  **renfort d'ouvrant** de l'abaque, la **parclose** à l'épaisseur du vitrage, la **tapée** et
  l'**appui** de l'isolant, le nombre de **paumelles** ;
- les **accessoires** de chaque profilé employé, avec leur dessin
  (`systeme-76-accessoires-par-profile.md`), sans quantité quand la page n'en donne pas.

Ce que le wiki ne donne pas n'est pas inventé : la **surlongueur de soudure** du système 76
(les barres soudées sortent en cote finie, sauf si l'atelier saisit sa surcote), le choix entre
plusieurs renforts de dormant (« au choix selon la statique »), les longueurs de tapée et d'appui.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from app.services import faisabilite as fa
from app.services.faisabilite import (PAGE_DEBIT, PAGE_DORMANTS, PAGE_OUVRANTS, PAGE_PARCLOSES, PAGE_TAPEES,
                                      Saisie, _corps, _fmt, _nombre, _ref, _une, calcul_debit, donnees, tables,
                                      verifier)
from app.services.wiki_service import WikiSnapshot

PAGE_ACCESSOIRES = "/profiles/systeme-76-accessoires-par-profile.md"
PAGE_MENEAUX = "/profiles/perform76-meneaux.md"
PAGES_IMAGES = (PAGE_DORMANTS, PAGE_OUVRANTS, PAGE_MENEAUX, PAGE_PARCLOSES, PAGE_TAPEES, PAGE_ACCESSOIRES)

_CACHE: Dict[str, Any] = {"cle": None, "d": None}


def _deduction(c: str) -> Tuple[Optional[float], Optional[float]]:
    """« 45 » → (45, 45) ; « 75 en haut, 45 vers le bas » → (75, 45) : haut, bas."""
    n = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:,\d+)?", c)]
    if not n:
        return None, None
    return (n[0], n[1]) if len(n) >= 2 else (n[0], n[0])


def _charge(snap: WikiSnapshot) -> Dict[str, Any]:
    td = tables(_corps(snap, PAGE_DEBIT))
    t = _une(td, "Dormant | DEO (mm) | DFO (mm)", PAGE_DEBIT)
    col = {h: k for k, h in enumerate(t.entete)}
    dormants = {r[0]: {"renfort": _deduction(r[col["Renfort de dormant (mm)"]]),
                       "meneau": _nombre(r[col["Meneau/traverse (mm)"]]),
                       "renfort_meneau": _nombre(r[col["Renfort de meneau/traverse (mm)"]]),
                       "vitrage_fixe": _nombre(r[col["Vitrage fixe (mm)"]]),
                       "parclose": _nombre(r[col["Parclose (mm)"]]),
                       "section": t.section, "localisation": t.localisation} for r in t.lignes}
    t = _une(td, "Ouvrant | DFO (mm) | Vitrage (mm)", PAGE_DEBIT)
    col = {h: k for k, h in enumerate(t.entete)}
    ouvrants = {r[0]: {"renfort": _nombre(r[col["Renfort d'ouvrant (mm)"]]),
                       "parclose": _nombre(r[col["Parclose (mm)"]]),
                       "section": t.section, "localisation": t.localisation} for r in t.lignes}
    t = _une(td, "Battement | Débit du battement | Renfort associé", PAGE_DEBIT)
    battements = {}
    for r in t.lignes:
        m = re.search(r"DEO\s*[−-]\s*2\s*×\s*(\d+)", r[1])
        mr = re.search(r"DEO\s*[−-]\s*2\s*×\s*(\d+)", r[3])
        ref = _ref(r[0])
        if m and ref not in battements:  # la première ligne : « avec embouts » pour le 76473
            battements[ref] = {"debit": float(m.group(1)), "renfort": None if r[2] in ("—", "-") else r[2],
                               "debit_renfort": float(mr.group(1)) if mr else None, "section": t.section}
    t = _une(td, "Dormant | Cas a (mm)", PAGE_DEBIT)
    renfort_vertical_seuil = {r[0]: _nombre(r[1]) for r in t.lignes}
    t = _une(td, "Cas | Seuils concernés | DEO (mm)", PAGE_DEBIT)
    dormant_sur_seuil = next((_nombre(r[4]) for r in t.lignes if r[0] == "a"), None)

    # accessoires : une section par profilé, une ligne par référence
    acc: Dict[str, List[Dict[str, Any]]] = {}
    for tb in tables(_corps(snap, PAGE_ACCESSOIRES)):
        profils = re.findall(r"\b\d{5}\b", tb.section)
        if not profils or tb.entete[-1] != "Dessin":
            continue
        a_colonne = tb.entete[0] in ("Ouvrant", "Battement")
        for r, b in zip(tb.lignes, tb.brut):
            if tb.entete[0] == "Dormant":
                continue  # sets d'assemblage sur seuil : dépendent du seuil, tabulés ailleurs
            img = re.search(r"\((/assets/[^)]+)\)", b[-1])
            cibles = [r[0]] if a_colonne else profils
            k = 1 if a_colonne else 0
            if tb.entete[k] != "Référence":
                continue
            for p in cibles:
                acc.setdefault(p, []).append({"ref": r[k], "designation": r[k + 1],
                                              "image": img.group(1) if img else None, "section": tb.section})

    # une coupe par référence : l'image dont le texte alternatif est « Famille RÉF »
    images: Dict[str, str] = {}
    for page in PAGES_IMAGES:
        if page in snap.pages:
            for alt, src in re.findall(r"!\[([^\]]+)\]\((/assets/[^)]+)\)", snap.pages[page].body):
                m = re.fullmatch(r"[A-Za-zÀ-ÿ']+ ([A-Z]?\d[\w.]*(?: R/L)?)", alt.strip())
                if m:
                    images.setdefault(m.group(1), src)
    return {"dormants": dormants, "ouvrants": ouvrants, "battements": battements,
            "renfort_vertical_seuil": renfort_vertical_seuil, "dormant_sur_seuil": dormant_sur_seuil,
            "accessoires": acc, "images": images}


def _donnees(snap: WikiSnapshot) -> Dict[str, Any]:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        _CACHE["d"] = _charge(snap)
        _CACHE["cle"] = cle
    return _CACHE["d"]


def _l(x: float) -> str:
    return _fmt(x, 1 if x % 1 else 0)


def fiche(snap: WikiSnapshot, s: Saisie, surcote_soudure_mm: Optional[float] = None) -> Dict[str, Any]:
    d = fa.donnees(snap)
    x = _donnees(snap)
    ver = verifier(snap, s)
    deb = calcul_debit(d, s)
    ctl = {c["id"]: c for c in ver["controles"]}
    dor = x["dormants"].get(s.dormant)
    if dor is None:
        raise ValueError(f"dormant {s.dormant} absent des cotes de débit")
    L, H = s.largeur_mm, s.hauteur_mm
    lignes: List[Dict[str, Any]] = []
    notes: List[str] = []
    src_dor = {"page": PAGE_DEBIT, "section": dor["section"], "localisation": dor["localisation"]}

    def ligne(famille, piece, ref, qte, longueur, calcul, coupe, source, note="", image_ref=None):
        lignes.append({"rep": len(lignes) + 1, "famille": famille, "piece": piece, "ref": ref, "qte": qte,
                       "longueur": longueur, "calcul": calcul, "coupe": coupe, "source": source, "note": note,
                       "image": x["images"].get(image_ref or ref or "")})

    def soudee(cote: float, base: str) -> Tuple[float, str]:
        if surcote_soudure_mm:
            return cote + 2 * surcote_soudure_mm, f"{base} + 2 × {_l(surcote_soudure_mm)}"
        return cote, base

    # ---- le dormant
    seuil = s.bas == "seuil" and s.configuration != "fixe"
    lg, cl = soudee(L, _l(L))
    ligne("Dormant", "Traverse haute" + ("" if seuil else " et basse"), s.dormant, 1 if seuil else 2, lg,
          cl, "45° soudé", src_dor, "" if surcote_soudure_mm else "cote finie")
    if seuil and x["dormant_sur_seuil"] is not None:
        h_m = H - x["dormant_sur_seuil"]
        ligne("Dormant", "Montant sur seuil", s.dormant, 2, h_m, f"{_l(H)} − {_l(x['dormant_sur_seuil'])}",
              "posé sur le seuil", {"page": PAGE_DEBIT, "section": "Cotes de débit des seuils aluminium"},
              "cas a ; pièce d'assemblage sur seuil selon le dormant", s.dormant)
        ligne("Seuil", "Seuil aluminium", "A076", 1, L, _l(L), "droite",
              {"page": PAGE_DEBIT, "section": "Cotes de débit des seuils aluminium"},
              "longueur prise à la largeur hors tout (hypothèse)", "A076")
    else:
        hg, ch = soudee(H, _l(H))
        ligne("Dormant", "Montant", s.dormant, 2, hg, ch, "45° soudé", src_dor,
              "" if surcote_soudure_mm else "cote finie")

    renforts_dormant = [a["ref"] for a in x["accessoires"].get(s.dormant, []) if a["designation"].startswith("Renfort")]
    choix = " / ".join(renforts_dormant) if renforts_dormant else "—"
    r_haut, r_bas = dor["renfort"]
    if r_haut is not None:
        note = "au choix selon la statique" if len(renforts_dormant) > 1 else ""
        ligne("Renfort", "Renfort de dormant, traverses", choix, 1 if seuil else 2, L - 2 * r_haut,
              f"{_l(L)} − 2 × {_l(r_haut)}", "droite", src_dor, note, renforts_dormant[0] if renforts_dormant else None)
        if seuil and s.dormant in x["renfort_vertical_seuil"]:
            b = x["renfort_vertical_seuil"][s.dormant]
            ligne("Renfort", "Renfort de dormant, montants sur seuil", choix, 2, H - r_haut - b,
                  f"{_l(H)} − {_l(r_haut)} − {_l(b)}", "droite",
                  {"page": PAGE_DEBIT, "section": "Cotes de débit des seuils aluminium"},
                  (note + " ; " if note else "") + "déduction basse du cas a", renforts_dormant[0] if renforts_dormant else None)
        else:
            ligne("Renfort", "Renfort de dormant, montants", choix, 2, H - r_haut - r_bas,
                  f"{_l(H)} − {_l(r_haut)} − {_l(r_bas)}" if r_haut != r_bas else f"{_l(H)} − 2 × {_l(r_haut)}",
                  "droite", src_dor, note + (" ; 75 mm en haut, 45 vers le bas" if r_haut != r_bas else ""),
                  renforts_dormant[0] if renforts_dormant else None)

    # ---- le vitrage et les parcloses
    parc = (ctl.get("parclose") or {}).get("valeurs", {}).get("parcloses") or []
    ref_parc = " / ".join(p["ref"] for p in parc) or "à choisir"
    if s.configuration == "fixe":
        vf, pd = dor["vitrage_fixe"], dor["parclose"]
        ligne("Vitrage", "Vitrage fixe", s.vitrage, 1, None, f"{_l(L - 2 * vf)} × {_l(H - 2 * vf)} (DHT − 2 × {_l(vf)})",
              "", src_dor)
        ligne("Parclose", "Parclose de dormant, horizontale", ref_parc, 2, L - 2 * pd, f"{_l(L)} − 2 × {_l(pd)}", "45°",
              src_dor, image_ref=parc[0]["ref"] if parc else None)
        ligne("Parclose", "Parclose de dormant, verticale", ref_parc, 2, H - 2 * pd, f"{_l(H)} − 2 × {_l(pd)}", "45°",
              src_dor, image_ref=parc[0]["ref"] if parc else None)

    # ---- meneau
    if s.configuration == "2v_meneau":
        ligne("Meneau", "Meneau de dormant", s.meneau, 1, H - 2 * dor["meneau"], f"{_l(H)} − 2 × {_l(dor['meneau'])}",
              "droite, assemblé mécaniquement", src_dor, "", s.meneau)
        ligne("Renfort", "Renfort de meneau", "—", 1, H - 2 * dor["renfort_meneau"],
              f"{_l(H)} − 2 × {_l(dor['renfort_meneau'])}", "droite", src_dor, "référence non donnée par la planche du meneau")

    # ---- les ouvrants
    n_v = deb.get("vantaux") or 0
    if n_v and deb.get("deo"):
        ouv = x["ouvrants"][s.ouvrant]
        src_ouv = {"page": PAGE_DEBIT, "section": ouv["section"], "localisation": ouv["localisation"]}
        wl, wh = deb["deo"]
        a, b = soudee(wl, _l(wl))
        ligne("Ouvrant", "Traverses", s.ouvrant, 2 * n_v, a, b, "45° soudé", src_ouv,
              "DEO" + ("" if surcote_soudure_mm else " ; cote finie"))
        a, b = soudee(wh, _l(wh))
        ligne("Ouvrant", "Montants", s.ouvrant, 2 * n_v, a, b, "45° soudé", src_ouv,
              "DEO" + ("" if surcote_soudure_mm else " ; cote finie"))
        # renfort d'ouvrant : la zone de l'abaque dit lesquels
        rf = ctl.get("renfort")
        ref_r = (rf or {}).get("valeurs", {}).get("renfort") or "—"
        zone = (rf or {}).get("valeurs", {}).get("zone")
        systematique = rf is not None and not rf["detail"].startswith("Zone")
        horizontal = systematique or zone in ("B", "D")
        vertical = systematique or zone in ("C", "D")
        ro = ouv["renfort"]
        src_ab = {"page": fa.PAGE_ABAQUES, "section": "Zones de renforcement"}
        if rf is None:
            notes.append("Renfort d'ouvrant non déterminé : l'abaque de cet ouvrant n'est pas exploitable.")
        if horizontal:
            ligne("Renfort", "Renfort d'ouvrant, traverses", ref_r, 2 * n_v, wl - 2 * ro, f"{_l(wl)} − 2 × {_l(ro)}",
                  "droite", src_ouv, f"zone {zone}" if zone and not systematique else "renforcement systématique")
        if vertical:
            ligne("Renfort", "Renfort d'ouvrant, montants", ref_r, 2 * n_v, wh - 2 * ro, f"{_l(wh)} − 2 × {_l(ro)}",
                  "droite", src_ouv, f"zone {zone}" if zone and not systematique else "renforcement systématique")
        if rf is not None and not horizontal and not vertical:
            notes.append(f"Zone {zone} de l'abaque : ouvrant sans renfort.")
        po = ouv["parclose"]
        vl, vh = deb["vitrage"]
        c_vit = d.debit_ouvrants[s.ouvrant]["vitrage"]
        ligne("Vitrage", "Vitrage d'ouvrant", s.vitrage, n_v, None, f"{_l(vl)} × {_l(vh)} (DEO − 2 × {_l(c_vit)})",
              "", src_ouv)
        ligne("Parclose", "Parclose d'ouvrant, horizontale", ref_parc, 2 * n_v, wl - 2 * po, f"{_l(wl)} − 2 × {_l(po)}",
              "45°", src_ouv, image_ref=parc[0]["ref"] if parc else None)
        ligne("Parclose", "Parclose d'ouvrant, verticale", ref_parc, 2 * n_v, wh - 2 * po, f"{_l(wh)} − 2 × {_l(po)}",
              "45°", src_ouv, image_ref=parc[0]["ref"] if parc else None)

        # battement et son renfort, sur l'ouvrant qui le porte
        if s.configuration == "2v_battement" and s.battement in x["battements"]:
            bt = x["battements"][s.battement]
            src_b = {"page": PAGE_DEBIT, "section": bt["section"]}
            ligne("Battement", "Battement", s.battement, 1, wh - 2 * bt["debit"], f"{_l(wh)} − 2 × {_l(bt['debit'])}",
                  "droite", src_b, "avec embouts M462" if s.battement == "76473" else "", s.battement)
            if bt["renfort"] and bt["debit_renfort"]:
                ligne("Renfort", "Renfort de battement", bt["renfort"], 1, wh - 2 * bt["debit_renfort"],
                      f"{_l(wh)} − 2 × {_l(bt['debit_renfort'])}", "droite", src_b)

        pm = ctl.get("paumelles")
        if pm:
            n = int(re.match(r"(\d+)", pm["detail"]).group(1))
            ligne("Quincaillerie", "Paumelles", "—", n * n_v, None, f"{n} par vantail", "",
                  pm["sources"][0], "référence selon la ferrure")

    # ---- tapée et appui
    iso = ctl.get("isolant")
    if iso and iso["statut"] == "ok":
        m = re.search(r"tapée (\S+)", iso["detail"])
        if m:
            ligne("Tapée", "Tapée de doublage", m.group(1).rstrip(","), 1, None, "sur le périmètre à doubler", "",
                  iso["sources"][0], "longueur selon la pose, non documentée", m.group(1).rstrip(","))
        ma = re.search(r"(?:avec l'appui|Appui) (\S+?)[.,]", iso["detail"])
        if ma:
            ligne("Appui", "Pièce d'appui", ma.group(1), 1, None, "sous le dormant bas", "", iso["sources"][-1],
                  "longueur non documentée", ma.group(1))

    # ---- les repères se suivent famille par famille, dans l'ordre où elles apparaissent
    ordre = list(OrderedDict.fromkeys(l["famille"] for l in lignes))
    lignes.sort(key=lambda l: ordre.index(l["famille"]))
    for i, l in enumerate(lignes, 1):
        l["rep"] = i

    # ---- la nomenclature : regroupée par référence et par pièce
    nomen: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
    for l in lignes:
        sans_ref = l["ref"] in ("—", "", None)
        k = (l["piece"] if sans_ref else l["ref"], l["famille"])
        n = nomen.setdefault(k, {"ref": "—" if sans_ref else l["ref"], "designation": l["piece"],
                                 "famille": l["famille"], "qte": 0, "ml": 0.0, "image": l["image"]})
        if l["piece"] not in n["designation"].split(" + "):
            n["designation"] += " + " + l["piece"]
        n["qte"] += l["qte"]
        if l["longueur"] is not None:
            n["ml"] += l["qte"] * l["longueur"] / 1000
    nomenclature = [dict(v, ml=round(v["ml"], 2) if v["ml"] else None) for v in nomen.values()]

    # ---- les accessoires des profilés employés
    employes = [s.dormant] + ([s.ouvrant] if n_v else []) + ([s.battement] if s.configuration == "2v_battement" else [])
    deja = {l["ref"] for l in lignes}
    accessoires = []
    for p in employes:
        for a in x["accessoires"].get(p, []):
            if a["designation"].startswith("Renfort") or a["ref"] in deja:
                continue
            if a["designation"].lower().startswith("capot") and s.couleur != "capot_alu":
                continue
            accessoires.append(dict(a, profil=p))

    if not surcote_soudure_mm:
        notes.append("Les barres soudées (dormant, ouvrant) sont en cote finie : le wiki ne donne pas la "
                     "surlongueur de soudure du système 76. Saisir la surcote de l'atelier pour obtenir les longueurs de coupe.")
    if "VER-21" in (ctl.get("debit") or {}).get("anomalies", []):
        notes.append("Le dormant 76185 n'a pas de cote de débit avec battement (VER-21) : les ouvrants ne sont pas débités.")
    return {"verdict": ver["verdict"], "libelle": ver["libelle"], "lignes": lignes, "nomenclature": nomenclature,
            "accessoires": accessoires, "notes": notes, "debit": ver["debit"], "surcote": surcote_soudure_mm,
            "brouillons": sorted(set(ver["brouillons"]) | ({PAGE_ACCESSOIRES}
                                  if snap.pages.get(PAGE_ACCESSOIRES) and snap.pages[PAGE_ACCESSOIRES].status == "draft" else set()))}
