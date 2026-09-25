"""Le calculateur de parcloses et de joints : pour un vitrage de X mm, quoi monter, gamme par gamme.

Le wiki porte ces règles sous des formes différentes, et chacune est lue telle quelle :

- **système 76 Advanced** (profine, PVC) : une parclose tient une épaisseur A avec un joint de
  4 mm et B = A + 2 avec un joint de 2 mm, dans une **tolérance** (+1,0 / −0,5 mm) ; les joints
  changent avec le contexte (ouvrant, capot alu et EPDM, dormant avec compensateur 76570) ;
- **PERFORM76** (cahier technique PROFERM) et le **poster des complémentaires** du 76 : une
  épaisseur par parclose, sans tolérance écrite ;
- **SOLEAL FY 55** (Technal, alu) : une matrice parclose × joint intérieur, où chaque cellule est
  l'épaisseur obtenue, avec la plage de prise de volume recommandée ;
- **ASKEY**, **LUMEAL GA**, **SOLEAL GY** : c'est le **profilé d'ouvrant** qui se choisit par
  l'épaisseur (une valeur ou une plage), et parfois la parclose avec lui.

Une épaisseur sans tolérance écrite est acceptée à ±0,5 mm et signalée « à ±1,5 mm » au-delà :
jamais une valeur voisine présentée comme exacte.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.services.faisabilite import Table, _corps, _nombre, lire_vitrage, tables
from app.services.wiki_service import WikiSnapshot

P76_VITRAGE = "/profiles/systeme-76-tableau-de-vitrage.md"
P76_COMPL = "/profiles/systeme-76-profiles-complementaires.md"
P76_PERFORM = "/profiles/perform76-parcloses.md"
P_FY = "/profiles/soleal-fy-parcloses-et-vitrage.md"
P_ASKEY_OC = "/profiles/askey-frappe-65-oc-dormants-et-ouvrants.md"
P_ASKEY_NV = "/profiles/askey-coulissant-65-nv-dormants-et-ouvrants.md"
P_LUMEAL = "/profiles/lumeal-ga-dormants-et-ouvrants.md"
P_GY = "/profiles/soleal-gy-dormants-et-rails.md"
P_ASKEY_OV = "/profiles/askey-frappe-65-ov-dormants-et-ouvrants.md"
P_S70 = "/profiles/systeme-70-profiles-et-renforts.md"

# Ce que le wiki ne permet pas de calculer, et pourquoi : affiché, pour qu'une absence ne passe pas
# pour un « rien ne convient ».
NON_CALCULABLES = [
    {"systeme": "ASKEY Frappe 65 Ouvrant Visible", "materiau": "Aluminium", "page": P_ASKEY_OV,
     "raison": "les six parcloses clipées sont listées sans l'épaisseur de vitrage qu'elles tiennent"},
    {"systeme": "Système 70 (KBE e.MOTION)", "materiau": "PVC", "page": P_S70,
     "raison": "le plan des complémentaires ne cote pas l'épaisseur de vitrage ; les parcloses partagées avec le 76 "
               "ne sont pas confirmées être les mêmes pièces"},
]

# Le filtre de la page : une gamme telle qu'on la nomme à PROFERM, et les systèmes qui la documentent.
GAMMES = {
    "PERFORM76": ("Système 76 Advanced", "PERFORM76 — cahier technique PROFERM"),
    "SOLEAL FY": ("SOLEAL FY 55", "SOLEAL FY 65"),
    "LUMEAL GA": ("LUMEAL GA",),
    "SOLEAL GY": ("SOLEAL GY 55",),
    "ASKEY Frappe 65": ("ASKEY Frappe 65 Ouvrant Caché",),
    "ASKEY Coulissant 65": ("ASKEY Coulissant 65 NV",),
}

SANS_TOLERANCE = 0.5     # une épaisseur unique, sans tolérance écrite : acceptée à ±0,5 mm
VOISINE = 1.5            # au-delà, jusqu'à ±1,5 mm : proposée « proche », jamais « exacte »

_CACHE: Dict[str, Any] = {"cle": None, "sol": None}


def _plage(c: str):
    m = re.search(r"(\d+(?:,\d+)?)\s*à\s*(\d+(?:,\d+)?)", c)
    return (_nombre(m.group(1)), _nombre(m.group(2))) if m else None


def _mm(c: str) -> Optional[float]:
    return _nombre(re.sub(r"\s*mm$", "", c.strip()))


def _img(cell: str) -> Optional[str]:
    m = re.search(r"\((/assets/[^)]+)\)", cell)
    return m.group(1) if m else None


def _sol(**k) -> Dict[str, Any]:
    """Une solution. `elements` dit ce qu'on monte, un rôle par ligne : la parclose, le joint, le
    support de cale… — c'est ce que l'écran affiche, rôle en toutes lettres."""
    base = {"elements": [], "recommande": None, "image": None, "anomalies": [], "note": ""}
    base.update(k)
    return base


def _el(role: str, valeur: Optional[str], detail: str = "") -> Optional[Dict[str, str]]:
    return {"role": role, "valeur": valeur, "detail": detail} if valeur else None


def _els(*items) -> List[Dict[str, str]]:
    return [i for i in items if i]


def _refs_joint(c: str) -> str:
    """« PCE (« Joint PCE »), G049.T « Joint thermosoudable », G047 … » → « PCE, G049.T ou G047 »."""
    refs = re.findall(r"\b(PCE|[A-Z]\d{3}(?:\.[A-Z])?)\b", c)
    refs = list(dict.fromkeys(refs))
    return (", ".join(refs[:-1]) + " ou " + refs[-1]) if len(refs) > 1 else (refs[0] if refs else c)


def _t(ts: List[Table], debut: str, section: Optional[str] = None) -> Optional[Table]:
    for t in ts:
        if " | ".join(t.entete).startswith(debut) and (section is None or t.section.startswith(section)):
            return t
    return None


# ---- système 76 Advanced ---------------------------------------------------------------------

def _systeme76(snap: WikiSnapshot) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ts = tables(_corps(snap, P76_VITRAGE))
    tj = _t(ts, "Tableau | Joint A - 4 mm | Joint B - 2 mm")
    joints = {r[0]: {"A": r[1], "B": r[2], "support": r[3]} for r in tj.lignes} if tj else {}
    base = {"systeme": "Système 76 Advanced", "fournisseur": "profine", "materiau": "PVC",
            "gammes": ["PERFORM76", "et toute gamme sur le 76"]}
    contextes = [
        ("ouvrant", "Ouvrant"),
        ("ouvrant avec capot alu et joint de vitrage EPDM", "Ouvrant, capot alu et joint EPDM"),
        ("dormant et meneau avec compensateur de parclose", "Dormant et meneau avec compensateur 76570"),
        ("dormant et meneau avec compensateur de parclose, capot alu et joint EPDM",
         "Dormant et meneau avec compensateur 76570, capot alu et joint EPDM"),
    ]
    tol_re = re.compile(r"\+?(\d+(?:,\d+)?)\s*/\s*[−-](\d+(?:,\d+)?)|±\s*(\d+(?:,\d+)?)")

    def tolerance(c: str):
        m = tol_re.search(c)
        if not m:
            return None
        if m.group(3):
            v = _nombre(m.group(3))
            return v, v
        return _nombre(m.group(1)), _nombre(m.group(2))   # (plus, moins)

    for tb, minces in ((_t(ts, "Parclose | Épaisseur de vitrage A", "Parcloses d'ouvrant"), False),
                       (_t(ts, "Parclose | Épaisseur de vitrage A", "Parcloses d'ouvrant pour vitrage mince"), True)):
        if tb is None:
            continue
        src = {"page": P76_VITRAGE, "section": tb.section, "localisation": tb.localisation}
        for r, b in zip(tb.lignes, tb.brut):
            tol = tolerance(r[3])
            for cle_ctx, lib_ctx in contextes:
                if minces and not cle_ctx == "ouvrant" and not cle_ctx.startswith("ouvrant avec"):
                    continue
                j = joints.get(cle_ctx, {})
                for fam, col in (("A", 1), ("B", 2)):
                    if minces and fam == "A":
                        continue  # la page : seule la famille « Joint B - 2 mm » est remplie
                    e = _nombre(r[col])
                    if e is None or tol is None:
                        continue
                    out.append(_sol(**base, contexte=lib_ctx, piece="Parclose", ref=r[0],
                                    elements=_els(_el("Parclose", r[0], "pour panneau mince" if minces else ""),
                                                  _el("Joint de vitrage", _refs_joint(j.get(fam, "")),
                                                      f"famille {fam}, joint de {'4' if fam == 'A' else '2'} mm"),
                                                  _el("Support de cale", j.get("support")),
                                                  _el("Compensateur", "76570", "clippé en fond de feuillure")
                                                  if "compensateur" in cle_ctx else None),
                                    cible=e, min=e - tol[1], max=e + tol[0],
                                    tolerance=f"+{tol[0]:g} / −{tol[1]:g}".replace(".", ","),
                                    image=_img(b[-1]), source=src))
    # AluClip Zero : une épaisseur par parclose
    tz = _t(ts, "Parclose | Épaisseur de vitrage AluClip Zero")
    if tz:
        src = {"page": P76_VITRAGE, "section": tz.section, "localisation": tz.localisation}
        for r in tz.lignes:
            e = _nombre(r[1])
            out.append(_sol(**base, contexte="AluClip Zero (ouvrant 76282, capot A195)", piece="Parclose", ref=r[0],
                            elements=_els(_el("Parclose", r[0])), cible=e, min=None, max=None, tolerance=None, source=src))
    # dormant et meneau sans compensateur : la page du tableau ne l'a pas généré, le poster le donne
    tp = next((t for t in tables(_corps(snap, P76_COMPL)) if t.section == "Seconde ligne"
               and t.entete[:3] == ["Parclose", "Largeur (mm)", "Épaisseur du remplissage (mm)"]), None)
    if tp:
        src = {"page": P76_COMPL, "section": "Parcloses — seconde ligne", "localisation": tp.localisation}
        for r in tp.lignes:
            out.append(_sol(**base, contexte="Dormant et meneau sans compensateur", piece="Parclose", ref=r[0],
                            cible=_nombre(r[2]), min=None, max=None, tolerance=None, source=src,
                            elements=_els(_el("Parclose", r[0]),
                                          _el("Joint de vitrage", "post-extrudé ou d'épaisseur équivalente", "feuillure de 62 mm")),
                            note="tableau absent de la page du tableau de vitrage : lu sur le poster des complémentaires",
                            anomalies=["INC-52"]))
    return out


def _perform76(snap: WikiSnapshot) -> List[Dict[str, Any]]:
    out = []
    for tb in tables(_corps(snap, P76_PERFORM)):
        if tb.entete[:2] != ["Parclose", "Épaisseur de vitrage (mm)"]:
            continue
        ctx = "Ouvrant" if "ouvrant" in tb.section else "Dormant"
        src = {"page": P76_PERFORM, "section": tb.section, "localisation": tb.localisation}
        for r, b in zip(tb.lignes, tb.brut):
            out.append(_sol(systeme="PERFORM76 — cahier technique PROFERM", fournisseur="PROFERM", materiau="PVC",
                            gammes=["PERFORM76"], contexte=ctx, piece="Parclose", ref=r[0], cible=_nombre(r[1]),
                            elements=_els(_el("Parclose", r[0], f"épaisseur de parclose {r[2]} mm"),
                                          _el("Joint de vitrage", "non précisé par le cahier", "voir le système 76 Advanced")),
                            min=None, max=None, tolerance=None, image=_img(b[-1]), source=src,
                            anomalies=["CTR-19"] if r[0] in ("76508", "2454", "2433", "2638") else []))
    return out


# ---- SOLEAL FY 55 ----------------------------------------------------------------------------

def _html_lignes(html: str) -> List[List[str]]:
    return [[re.sub(r"<.*?>", "", c).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
            for tr in re.findall(r"<tr.*?</tr>", html, re.S)]


def _soleal_fy(snap: WikiSnapshot) -> List[Dict[str, Any]]:
    corps = _corps(snap, P_FY)
    out = []
    base = {"systeme": "SOLEAL FY 55", "fournisseur": "Technal", "materiau": "Aluminium", "gammes": ["LUMINE55"]}
    m = re.search(r"<table>.*?</table>", corps, re.S)
    if m:
        rows = _html_lignes(m.group(0))
        noms, bs = rows[1], rows[2]
        ext = re.search(r"joint extérieur est toujours le \*\*(\w+)\*\*", corps)
        loc = re.search(r"</table>\s*\n\s*\(schéma: ([^)]+)\)", corps[m.start():])
        src = {"page": P_FY, "section": "Matrice des parcloses droites et arrondies", "localisation": loc.group(1) if loc else ""}
        # « Vert TAS0017 » : la couleur sert de clé au tableau de l'élargisseur, qui ne donne qu'elle
        couleur_ref = {n.split()[0]: n.split()[-1] for n in noms}
        for r in rows[3:]:
            ref, forme, c = r[0], r[1], r[2]
            reco = _plage(r[-1])
            for k, cell in enumerate(r[3:3 + len(noms)]):
                e = _nombre(cell)
                if e is None:
                    continue
                couleur, jref = noms[k].split()[0], noms[k].split()[-1]
                out.append(_sol(**base, contexte="Ouvrant apparent et fixe", piece=f"Parclose {forme}", ref=ref,
                                elements=_els(_el("Parclose", ref, f"{forme}, hauteur C = {c} mm"),
                                              _el("Joint intérieur", f"{jref} · {couleur.lower()}", bs[k]),
                                              _el("Joint extérieur", ext.group(1) if ext else None)),
                                cible=e, min=None, max=None, tolerance=None, source=src,
                                recommande=bool(reco and reco[0] <= e <= reco[1]),
                                note=f"prise de volume recommandée pour cette parclose : {r[-1]} mm"))
    ts = tables(corps)
    te = _t(ts, "Réf. Parclose | Hauteur $C$ (mm) | Joint Blanc")
    if te:
        src = {"page": P_FY, "section": "Prises de volume avec élargisseur de feuillure T530031", "localisation": te.localisation}
        for r in te.lignes:
            for k, cell in enumerate(r[2:]):
                e = _mm(cell)
                if e is None:
                    continue
                mc = re.match(r"Joint (\w+) \((\d+) mm\)", te.entete[k + 2])
                couleur = mc.group(1) if mc else te.entete[k + 2]
                jref = couleur_ref.get(couleur) if m else None
                out.append(_sol(**base, contexte="Fixe avec élargisseur de feuillure T530031", piece="Parclose", ref=r[0],
                                elements=_els(_el("Parclose", r[0], f"hauteur C = {r[1]} mm"),
                                              _el("Joint intérieur", f"{jref} · {couleur.lower()}" if jref else couleur.lower(),
                                                  f"B = {mc.group(2)} mm" if mc else ""),
                                              _el("Élargisseur", "T530031", "28 mm, vis T770020")),
                                cible=e, min=None, max=None, tolerance=None, source=src))
    tf = _t(ts, "Réf. Parclose | Type de parclose | Hauteur $C$")
    if tf:
        src = {"page": P_FY, "section": "Parcloses pose de face (avec clip T770070)", "localisation": tf.localisation}
        for r in tf.lignes:
            for k, ctx in ((3, "Pose de face, sans élargisseur"), (4, "Pose de face, avec élargisseur T530031")):
                p = _plage(r[k])
                if p:
                    out.append(_sol(**base, contexte=ctx, piece=f"Parclose {r[1].lower()}", ref=r[0],
                                    elements=_els(_el("Parclose", r[0], f"{r[1].lower()}, hauteur C = {r[2]} mm"),
                                                  _el("Clip de verrouillage", "T770070", "5 par mètre, à 150 mm des angles"),
                                                  _el("Élargisseur", "T530031") if k == 4 else None),
                                    cible=None, min=p[0], max=p[1],
                                    tolerance="plage", source=src,
                                    note="vitrage 4 mm plus petit qu'en parclose standard ; interdit en exigence NF EN 13049"))
    tm = _t(ts, "Épaisseur vitrage (mm) | Profilé d'ouvrant masqué")
    if tm:
        src = {"page": P_FY, "section": "Vitrage de l'Ouvrant Minimal (FYm / OM)", "localisation": tm.localisation}
        for r in tm.lignes:
            e = _mm(re.sub(r"\(.*?\)", "", r[0]))
            fy65 = "FY 65" in r[0]
            out.append(_sol(systeme="SOLEAL FY 65" if fy65 else "SOLEAL FY 55", fournisseur="Technal", materiau="Aluminium",
                            gammes=["LUMINE65"] if fy65 else ["LUMINE55"], contexte="Ouvrant minimal (masqué)",
                            piece="Profilé d'ouvrant masqué", ref=r[1],
                            elements=_els(_el("Profilé d'ouvrant", r[1]), _el("Parclose extérieure TPE", r[2]),
                                          _el("Joint intérieur", re.sub(r"\s*\(.*?\)", "", r[3]),
                                              (re.search(r"\((.*?)\)", r[3]) or [None, ""])[1]),
                                          _el("Support de cale", r[4])),
                            cible=e, min=None, max=None, tolerance=None, source=src))
    return out


# ---- profilés d'ouvrant choisis par l'épaisseur ------------------------------------------------

def _profils(snap: WikiSnapshot) -> List[Dict[str, Any]]:
    out = []
    # ASKEY Frappe 65 Ouvrant Caché : prise de vitrage par profilé, parcloses TPE par prise de volume
    corps = _corps(snap, P_ASKEY_OC)
    tpe: Dict[float, str] = {}
    m = re.search(r"prise de volume 24 mm et 28 mm sont exclusivement équipés de la parclose coextrudée \*\*(\w+)\*\* "
                  r"\(noir\) ou \*\*(\w+)\*\* \(gris\)\. Les ouvrants en prise de volume 36 mm reçoivent la parclose TPE \*\*(\w+)\*\*", corps)
    if m:
        tpe = {24: f"{m.group(1)} (noir) ou {m.group(2)} (gris)", 28: f"{m.group(1)} (noir) ou {m.group(2)} (gris)", 36: m.group(3)}
    for debut, col, sec in (("Référence | Rôle du profilé | Prise de vitrage", 2, "Profilés ouvrants cachés"),
                            ("Référence | Rôle | Épaisseur vitrage", 2, "Traverses d'ouvrants et montants serrures")):
        tb = _t(tables(corps), debut)
        if tb is None:
            continue
        src = {"page": P_ASKEY_OC, "section": sec, "localisation": tb.localisation}
        for r in tb.lignes:
            e = _mm(r[col])
            if e is None:
                continue
            out.append(_sol(systeme="ASKEY Frappe 65 Ouvrant Caché", fournisseur="ASKEY", materiau="Aluminium",
                            gammes=[], contexte=sec, piece=r[1], ref=r[0], cible=e, min=None, max=None, tolerance=None,
                            elements=_els(_el("Profilé d'ouvrant", r[0], r[1]),
                                          _el("Parclose TPE", tpe.get(e)) if sec == "Profilés ouvrants cachés" else None),
                            source=src))
    # ASKEY Coulissant 65 NV : un tableau par module d'épaisseur
    for tb in tables(_corps(snap, P_ASKEY_NV)):
        mm_ = re.search(r"module (\d+)(?:/(\d+))? mm", tb.section)
        if not mm_ or tb.entete[:2] != ["Référence", "Rôle du profilé"]:
            continue
        valeurs = [float(v) for v in mm_.groups() if v]
        src = {"page": P_ASKEY_NV, "section": tb.section, "localisation": tb.localisation}
        for r in tb.lignes:
            for e in valeurs:
                out.append(_sol(systeme="ASKEY Coulissant 65 NV", fournisseur="ASKEY", materiau="Aluminium", gammes=[],
                                contexte=tb.section, piece=r[1], ref=r[0], cible=e, min=None, max=None, tolerance=None,
                                elements=_els(_el("Profilé d'ouvrant", r[0], r[1])), source=src))
    # LUMEAL GA et SOLEAL GY : une plage par profilé ou par famille
    tb = _t(tables(_corps(snap, P_LUMEAL)), "Référence | Rôle et cinématique | Face vue")
    if tb:
        src = {"page": P_LUMEAL, "section": tb.section, "localisation": tb.localisation}
        for r in tb.lignes:
            p = _plage(r[4])
            if p:
                out.append(_sol(systeme="LUMEAL GA", fournisseur="Technal", materiau="Aluminium", gammes=["LUMÉAL55"],
                                contexte="Profilés ouvrants", piece=r[1], ref=r[0], cible=None, min=p[0], max=p[1],
                                elements=_els(_el("Profilé d'ouvrant", r[0], r[1])),
                                tolerance="plage", source=src))
    tb = _t(tables(_corps(snap, P_GY)), "Famille | Épaisseur vitrage")
    if tb:
        src = {"page": P_GY, "section": "Profilés ouvrants", "localisation": tb.localisation}
        for r in tb.lignes:
            p = _plage(r[1])
            if p:
                sans_cotes = [re.sub(r"\s*\(.*?\)", "", c) for c in r]
                out.append(_sol(systeme="SOLEAL GY 55", fournisseur="Technal", materiau="Aluminium", gammes=["GALANDAGE55"],
                                contexte="Ouvrants", piece=r[0], ref=r[0],
                                elements=_els(*[_el(tb.entete[k], sans_cotes[k]) for k in range(2, len(r))]),
                                cible=None, min=p[0], max=p[1],
                                tolerance="plage", source=src))
    return out


def solutions(snap: WikiSnapshot) -> List[Dict[str, Any]]:
    cle = (id(snap), snap.loaded_at)
    if _CACHE["cle"] != cle:
        sol: List[Dict[str, Any]] = []
        for f in (_systeme76, _perform76, _soleal_fy, _profils):
            try:
                sol += f(snap)
            except Exception as exc:  # une page réécrite ne doit pas éteindre les autres systèmes
                sol.append({"erreur": f"{f.__name__} : {exc}"})
        _CACHE["sol"] = sol
        _CACHE["cle"] = cle
    return _CACHE["sol"]


def _statut(s: Dict[str, Any], e: float) -> Optional[str]:
    if s.get("min") is not None and s.get("max") is not None:
        # une tolérance ou une plage écrite par la source est la règle : dedans ou pas
        return "ok" if s["min"] - 1e-9 <= e <= s["max"] + 1e-9 else None
    if s.get("cible") is None:
        return None
    d = abs(e - s["cible"])
    return "ok" if d <= SANS_TOLERANCE else ("proche" if d <= VOISINE else None)


def chercher(snap: WikiSnapshot, epaisseur: Optional[float] = None, vitrage: Optional[str] = None,
             gamme: Optional[str] = None) -> Dict[str, Any]:
    vit = None
    if vitrage:
        vit = lire_vitrage(vitrage)
        epaisseur = vit.total_mm
    if epaisseur is None or epaisseur <= 0:
        raise ValueError("donner une épaisseur de vitrage ou une composition")
    if gamme and gamme not in GAMMES:
        raise ValueError(f"gamme inconnue : {gamme}")
    retenus = set(GAMMES[gamme]) if gamme else None
    trouves = []
    erreurs = []
    for s in solutions(snap):
        if "erreur" in s:
            erreurs.append(s["erreur"])
            continue
        if retenus is not None and s["systeme"] not in retenus:
            continue
        st = _statut(s, epaisseur)
        if st:
            trouves.append(dict(s, statut=st, ecart=None if s.get("cible") is None else round(epaisseur - s["cible"], 2)))
    # groupes : système › contexte, les correspondances franches d'abord
    ordre = {"ok": 0, "proche": 1}
    trouves.sort(key=lambda s: (s["materiau"], s["systeme"], s["contexte"], ordre[s["statut"]],
                                0 if s.get("recommande") else 1, s["ref"]))
    groupes: Dict[str, Dict[str, Any]] = {}
    for s in trouves:
        g = groupes.setdefault(s["systeme"], {"systeme": s["systeme"], "fournisseur": s["fournisseur"],
                                              "materiau": s["materiau"], "gammes": s["gammes"], "contextes": {}})
        g["contextes"].setdefault(s["contexte"], []).append(s)
    couverts = {s["systeme"] for s in solutions(snap) if "erreur" not in s and (retenus is None or s["systeme"] in retenus)}
    sans = sorted(couverts - set(groupes))
    return {"epaisseur": epaisseur, "vitrage": vit.__dict__ if vit else None,
            "groupes": [dict(g, contextes=[{"contexte": k, "solutions": v} for k, v in g["contextes"].items()])
                        for g in groupes.values()],
            "sans_solution": sans, "erreurs": erreurs,
            "systemes": sorted(couverts), "gammes": list(GAMMES),
            "non_calculables": [] if gamme else NON_CALCULABLES}
