"""Sonde — la cote mal lue est-elle un problème de PIXELS ou de modèle ?

Mesuré le 14/09 : Mistral Large 3 réduit toute image à **1540 px de grand côté** et la
découpe en patches de 28 px (FAQ vision officielle). Une A4 portrait (595 × 842 pt) arrive
donc au modèle en 1089 × 1540, soit 132 dpi — quel que soit le DPI de rendu qu'on envoie.
Sur la planche des parcloses, un chiffre de cote y fait 12 à 14 px : un demi-patch. C'est
l'explication candidate des erreurs stables « valeur voisine » (76503 → 22 au lieu de 36).

Le facteur d'échelle vaut ``1540 / max(largeur, hauteur)``. Pour l'augmenter, il faut
réduire la plus GRANDE dimension de ce qu'on envoie — d'où les découpes testées ici :

    page entière   595 × 842 pt   →  ×1,83   chiffre ≈ 13 px
    2 bandes       595 × 421 pt   →  ×2,59   chiffre ≈ 18 px
    4 quarts       297 × 421 pt   →  ×3,66   chiffre ≈ 26 px   (un patch entier)
    6 tuiles       297 × 280 pt   →  ×5,18   chiffre ≈ 37 px

Un découpage en colonnes (gauche/droite) ne gagne RIEN sur une page portrait : la hauteur
reste la plus grande dimension, donc l'échelle ne bouge pas. Seul compte le grand côté.

Ce n'est pas un mécanisme de lecture : aucune règle n'est ajoutée, le modèle ne choisit
aucun recadrage, il n'y a ni vote ni filtrage de couleur. C'est la même page, livrée en
morceaux, chacun recevant tout le budget de pixels de l'API.

    docker compose exec web python -m app.scripts.probe_tuilage_planche
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Grand côté servi par l'API pour les modèles Mistral 3 (FAQ vision).
MAX_SIDE = 1540

SYSTEM = (
    "Tu lis des images d'une page de document technique. Tout ce que tu écris vient de ces "
    "images.\n"
    "Sur une planche cotée, chaque schéma porte plusieurs nombres et une légende dit ce que "
    "chaque cotation représente. Lis la légende, puis rattache la valeur au repère dont elle "
    "est la cote : jamais celle d'un repère voisin.\n"
    "Si le repère demandé n'apparaît sur aucune image, réponds valeur vide.\n"
    'Réponds en JSON strict : {"repere": str, "valeur": str, "autres_valeurs_du_repere": [str]}'
)

# (page, question, valeur attendue, valeur piège)
CAS: List[Tuple[int, str, str, str]] = [
    (8, "Quelle est l'épaisseur de vitrage de la parclose 2636 ?", "30", "27"),
    (8, "Quelle est l'épaisseur de vitrage de la parclose 76503 ?", "36", "22"),
    (8, "Quelle est l'épaisseur de vitrage de la parclose 2454 ?", "31", "26.5"),
    (8, "Quelle est l'épaisseur de vitrage de la parclose 2433 ?", "33", "23.5"),
    (8, "Quelle est l'épaisseur de vitrage de la parclose 76527 ?", "26", "31.5"),
    (8, "Quelle est l'épaisseur de vitrage de la parclose 2452 ?", "16", "41.5"),
    (8, "Quelle est l'épaisseur de vitrage de la parclose 2638 ?", "31", "26.5"),
    (8, "Quelle est l'épaisseur de vitrage de la parclose 76579 ?", "48", "10.8"),
    (17, "Quelle tapée utiliser pour une isolation de 120 mm ?", "6140", ""),
    (6, "Pour un ouvrant de 1700 mm de hauteur, quelle hauteur d'axe de poignée "
        "en fond de feuillure quincaillerie ?", "763", "1000"),
    (6, "Pour un ouvrant de 1500 mm de hauteur, quelle hauteur d'axe de poignée "
        "en fond de feuillure quincaillerie ?", "563", "1000"),
]


def _tiles(rect, cols: int, rows: int, overlap: float = 0.06) -> List[Tuple[str, Any]]:
    """Découpe un rectangle en cols × rows tuiles qui se CHEVAUCHENT.

    Le chevauchement est indispensable : sans lui, un schéma à cheval sur la coupe perd sa
    cote ou son étiquette, et la tuile ment par omission.
    """
    import fitz

    w, h = rect.width, rect.height
    dx, dy = w / cols, h / rows
    ox, oy = dx * overlap, dy * overlap
    noms_col = ["gauche", "centre", "droite"] if cols == 3 else ["gauche", "droite"]
    noms_lig = ["haut", "milieu", "bas"] if rows == 3 else (["haut", "bas"] if rows == 2 else [""])
    out: List[Tuple[str, Any]] = []
    for j in range(rows):
        for i in range(cols):
            clip = fitz.Rect(
                max(rect.x0, rect.x0 + i * dx - ox),
                max(rect.y0, rect.y0 + j * dy - oy),
                min(rect.x1, rect.x0 + (i + 1) * dx + ox),
                min(rect.y1, rect.y0 + (j + 1) * dy + oy),
            )
            libelle = " ".join(x for x in (noms_lig[j] if rows > 1 else "", noms_col[i] if cols > 1 else "") if x)
            out.append((libelle or "page entière", clip))
    return out


def rendre(pdf_path: str, page_no: int, cols: int, rows: int) -> List[Tuple[str, str]]:
    """(libellé, PNG base64) — chaque morceau rendu à MAX_SIDE sur son grand côté."""
    import fitz

    out: List[Tuple[str, str]] = []
    with fitz.open(pdf_path) as pdf:
        page = pdf[page_no - 1]
        for libelle, clip in _tiles(page.rect, cols, rows):
            zoom = MAX_SIDE / max(clip.width, clip.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
            out.append((libelle, base64.b64encode(pix.tobytes("png")).decode("utf-8")))
    return out


_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _norm(v: str) -> str:
    m = _NUM.search(str(v or ""))
    return m.group(0).replace(",", ".").rstrip("0").rstrip(".") if m else ""


async def interroger(
    images: List[Tuple[str, str]], question: str, page_no: int, model: str
) -> Dict[str, Any]:
    from app.services.mistral_service import chat

    if len(images) == 1:
        legende = f"Image 1 = page {page_no} (page entière)."
    else:
        legende = "\n".join(
            f"Image {k} = page {page_no}, partie {lib}" for k, (lib, _) in enumerate(images, 1)
        )
    contexte = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"{legende}\n\n{question}",
            "images": [b64 for _, b64 in images],
        },
    ]
    t0 = time.perf_counter()
    try:
        res = await chat(
            message="", model=model, context=contexte, temperature=0.0,
            max_tokens=600, response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        return {"erreur": str(exc)[:200], "ms": int((time.perf_counter() - t0) * 1000)}
    contenu = ((res.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if isinstance(contenu, list):
        contenu = " ".join(p.get("text", "") for p in contenu if isinstance(p, dict))
    try:
        data = json.loads(contenu)
    except (TypeError, ValueError):
        m = re.search(r"\{.*\}", str(contenu), re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else {}
        except (TypeError, ValueError):
            data = {}
    data["ms"] = int((time.perf_counter() - t0) * 1000)
    return data


VARIANTES: Dict[str, Tuple[int, int]] = {
    "page entière": (1, 1),
    "2 bandes": (1, 2),
    "4 quarts": (2, 2),
    "6 tuiles": (2, 3),
}


# ---------------------------------------------------------------------------
# Mode « relevé » — le modèle VOIT-IL la couleur, ou la déduit-il ?
# ---------------------------------------------------------------------------
#
# Première mesure (14/09) : la tuile ne change rien, et 22 lectures sur 24 rendent le PLUS
# PETIT des deux nombres du repère, quelle que soit la résolution. Ce n'est donc pas un
# défaut de perception mais un A PRIORI : le modèle décide qu'une épaisseur de vitrage est
# le petit nombre, tout en écrivant « cote en bleu ». Reste à savoir s'il sait lire la
# couleur quand on ne lui demande QUE ça — la réponse décide du correctif.

SYSTEM_RELEVE = (
    "Tu relèves des cotes sur une planche technique fournie en image. Tu ne réponds à "
    "aucune question et tu ne choisis aucune valeur : tu RELÈVES ce qui est imprimé.\n"
    "Pour le repère demandé, liste TOUS les nombres cotés qui lui sont attachés, dans "
    "l'ordre où ils apparaissent de GAUCHE à DROITE, en donnant pour chacun la couleur "
    "EXACTE de l'encre telle qu'elle est imprimée (bleu, noir, rouge, vert…).\n"
    "N'omets aucun nombre, n'en invente aucun, ne les réordonne pas, ne déduis aucune "
    "couleur de ce qu'un nombre te semble représenter.\n"
    'Réponds en JSON strict : {"repere": str, "nombres": [{"valeur": str, "couleur": str}]}'
)

# (page, repère, valeur bleue attendue, valeur noire attendue)
CAS_RELEVE: List[Tuple[int, str, str, str]] = [
    (8, "Parclose 2636", "30", "27"),
    (8, "Parclose 76503", "36", "22"),
    (8, "Parclose 2454", "31", "26.5"),
    (8, "Parclose 2433", "33", "23.5"),
    (8, "Parclose 76527", "26", "31.5"),
    (8, "Parclose 2452", "16", "41.5"),
    (8, "Parclose 2638", "31", "26.5"),
    (8, "Parclose 76579", "48", "10.8"),
]


async def relever(
    images: List[Tuple[str, str]], repere: str, page_no: int, model: str
) -> Dict[str, Any]:
    from app.services.mistral_service import chat

    if len(images) == 1:
        legende = f"Image 1 = page {page_no} (page entière)."
    else:
        legende = "\n".join(
            f"Image {k} = page {page_no}, partie {lib}" for k, (lib, _) in enumerate(images, 1)
        )
    contexte = [
        {"role": "system", "content": SYSTEM_RELEVE},
        {
            "role": "user",
            "content": f"{legende}\n\nRepère à relever : « {repere} ».",
            "images": [b64 for _, b64 in images],
        },
    ]
    try:
        res = await chat(
            message="", model=model, context=contexte, temperature=0.0,
            max_tokens=600, response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        return {"erreur": str(exc)[:200]}
    contenu = ((res.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if isinstance(contenu, list):
        contenu = " ".join(p.get("text", "") for p in contenu if isinstance(p, dict))
    try:
        return json.loads(contenu)
    except (TypeError, ValueError):
        m = re.search(r"\{.*\}", str(contenu), re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {}
        except (TypeError, ValueError):
            return {}


# ---------------------------------------------------------------------------
# Mode « prompts » — la décision, pas la perception
# ---------------------------------------------------------------------------
#
# Ce que les deux premières mesures établissent (14/09) :
#   * découper la page ne corrige rien (5/11 entière, 2/11 en quarts) ;
#   * 22 lectures sur 24 rendent le PLUS PETIT des deux nombres du repère ;
#   * mais interrogé en RELEVÉ, le même modèle sur la même image entière rend la paire
#     complète 8/8, dans l'ordre gauche-droite 8/8, et reconnaît le bleu 7/8.
# Il voit donc ce qu'il faut voir et choisit mal. Restent deux correctifs possibles, tous
# deux sans rien ajouter à l'image ni appel supplémentaire :
#   A. nommer l'a priori pour l'interdire (« un nombre plus petit n'est pas plus
#      vraisemblable ») ;
#   B. imposer l'ORDRE de production : relever d'abord, conclure ensuite — le relevé
#      conditionne alors la réponse au lieu de la justifier après coup.

SYSTEM_ANTI_PRIORI = (
    "Tu lis des images d'une page de document technique. Tout ce que tu écris vient de ces "
    "images.\n"
    "Sur une planche cotée, un même repère porte PLUSIEURS nombres, et une légende dit ce "
    "que chaque cotation représente. Lis la légende, puis retiens la valeur qu'elle "
    "désigne.\n"
    "INTERDIT : choisir un nombre parce qu'il te paraît plausible. Un nombre plus PETIT "
    "n'est pas plus vraisemblable qu'un grand, et l'ordre de grandeur que tu attends d'une "
    "cote n'est pas une preuve. Seules la légende et la position de la cote décident.\n"
    "Si le repère demandé n'apparaît sur aucune image, réponds valeur vide.\n"
    'Réponds en JSON strict : {"repere": str, "valeur": str, "autres_valeurs_du_repere": [str]}'
)

SYSTEM_DEUX_TEMPS = (
    "Tu lis des images d'une page de document technique. Tout ce que tu écris vient de ces "
    "images.\n"
    "Tu procèdes en DEUX TEMPS, dans cet ordre, sans jamais l'inverser :\n"
    "  1. RELEVER — recopie la légende de cotation si la page en porte une, puis liste TOUS "
    "les nombres attachés au repère demandé, dans l'ordre où ils apparaissent de GAUCHE à "
    "DROITE, avec la couleur EXACTE de l'encre de chacun. Tu relèves, tu ne choisis pas.\n"
    "  2. CONCLURE — applique la légende au relevé pour désigner la valeur demandée.\n"
    "Ne choisis jamais un nombre parce qu'il te paraît plausible : un nombre plus petit "
    "n'est pas plus vraisemblable qu'un grand.\n"
    "Si le repère n'apparaît sur aucune image, « valeur » vide.\n"
    'Réponds en JSON strict, champs dans cet ordre : {"repere": str, "legende": str, '
    '"nombres": [{"valeur": str, "couleur": str}], "valeur": str}'
)

PROMPTS: Dict[str, str] = {
    "direct": SYSTEM,
    "anti-a-priori": SYSTEM_ANTI_PRIORI,
    "relevé puis conclusion": SYSTEM_DEUX_TEMPS,
}


async def mode_prompts(args: argparse.Namespace, pdf_path: str) -> int:
    """Même image (page entière), mêmes questions, trois contrats de sortie."""
    noms = [n.strip() for n in args.variantes.split(",") if n.strip()] or list(PROMPTS)
    cas = [c for c in CAS if not args.page or c[0] == args.page]
    rendus = {page: rendre(pdf_path, page, 1, 1) for page in sorted({c[0] for c in cas})}
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def _un(page: int, question: str, attendu: str, piege: str, nom: str) -> Dict[str, Any]:
        global SYSTEM
        async with semaphore:
            ancien, SYSTEM = SYSTEM, PROMPTS[nom]
            try:
                data = await interroger(rendus[page], question, page, args.model)
            finally:
                SYSTEM = ancien
        lu = _norm(data.get("valeur"))
        return {
            "page": page, "question": question, "attendu": attendu, "piege": piege,
            "prompt": nom, "lu": lu, "legende": data.get("legende"),
            "nombres": data.get("nombres"),
            "juste": bool(lu) and lu == _norm(attendu),
            "piege_lu": bool(piege) and lu == _norm(piege),
            "erreur": data.get("erreur"), "ms": data.get("ms"),
        }

    taches = [_un(p, q, a, pg, nom) for nom in noms for (p, q, a, pg) in cas]
    resultats = [await c for c in asyncio.as_completed(taches)]

    print(f"{'cas':52s} " + "  ".join(f"{n:>22s}" for n in noms))
    for (page, question, attendu, piege) in cas:
        cells = []
        for nom in noms:
            r = next((x for x in resultats if x["question"] == question and x["prompt"] == nom), None)
            if r is None or r.get("erreur"):
                cells.append(f"{'ERREUR':>22s}")
            else:
                marque = "OK " if r["juste"] else ("!! " if r["piege_lu"] else "   ")
                cells.append(f"{marque + (r['lu'] or '—'):>22s}")
        libelle = re.sub(r"\s+", " ", question)[:47]
        print(f"p.{page:<3d} {libelle:47s} " + "  ".join(cells) + f"   [{attendu}]")
    print()
    for nom in noms:
        rs = [x for x in resultats if x["prompt"] == nom and not x.get("erreur")]
        if not rs:
            continue
        ms = sorted(x["ms"] or 0 for x in rs)
        print(
            f"  {nom:24s} : {sum(1 for x in rs if x['juste'])}/{len(rs)} justes · "
            f"{sum(1 for x in rs if x['piege_lu'])} valeur(s) piège · "
            f"médiane {ms[len(ms) // 2] / 1000:.1f}s"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"mode": "prompts", "resultats": resultats}, fh, ensure_ascii=False, indent=1)
        print(f"\nrapport : {args.out}")
    return 0


async def mode_releve(args: argparse.Namespace, pdf_path: str) -> int:
    variantes = [v.strip() for v in args.variantes.split(",") if v.strip()] or ["page entière", "4 quarts"]
    rendus = {
        nom: rendre(pdf_path, 8, *VARIANTES[nom]) for nom in variantes
    }
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def _un(page: int, repere: str, bleu: str, noir: str, nom: str) -> Dict[str, Any]:
        async with semaphore:
            data = await relever(rendus[nom], repere, page, args.model)
        nombres = [
            {"valeur": _norm(n.get("valeur")), "couleur": str(n.get("couleur") or "").lower()}
            for n in (data.get("nombres") or [])
            if isinstance(n, dict) and _norm(n.get("valeur"))
        ]
        vals = [n["valeur"] for n in nombres]
        couleur_du_bleu = next(
            (n["couleur"] for n in nombres if n["valeur"] == _norm(bleu)), ""
        )
        couleur_du_noir = next(
            (n["couleur"] for n in nombres if n["valeur"] == _norm(noir)), ""
        )
        return {
            "repere": repere, "variante": nom, "valeurs_lues": vals,
            "paire_complete": _norm(bleu) in vals and _norm(noir) in vals,
            "ordre_ok": vals[:2] == [_norm(bleu), _norm(noir)],
            "couleur_du_bleu": couleur_du_bleu, "couleur_du_noir": couleur_du_noir,
            "couleurs_ok": "bleu" in couleur_du_bleu and "bleu" not in couleur_du_noir,
            "erreur": data.get("erreur"),
        }

    taches = [
        _un(p, r, b, n, nom) for nom in variantes for (p, r, b, n) in CAS_RELEVE
    ]
    resultats = [await c for c in asyncio.as_completed(taches)]

    print(f"{'repère':18s} {'variante':14s} {'attendu':16s} {'relevé':26s} couleurs")
    for (p, r, b, n) in CAS_RELEVE:
        for nom in variantes:
            x = next((y for y in resultats if y["repere"] == r and y["variante"] == nom), None)
            if x is None or x.get("erreur"):
                print(f"{r:18s} {nom:14s} ERREUR")
                continue
            marque = "OK" if x["couleurs_ok"] else "!!"
            print(
                f"{r:18s} {nom:14s} {b + ' bleu / ' + n:16s} "
                f"{str(x['valeurs_lues'])[:25]:26s} "
                f"{marque} {b}→{x['couleur_du_bleu'] or '?'} · {n}→{x['couleur_du_noir'] or '?'}"
            )
    print()
    for nom in variantes:
        rs = [x for x in resultats if x["variante"] == nom and not x.get("erreur")]
        if not rs:
            continue
        print(
            f"  {nom:14s} : paire complète {sum(1 for x in rs if x['paire_complete'])}/{len(rs)} · "
            f"ordre gauche-droite {sum(1 for x in rs if x['ordre_ok'])}/{len(rs)} · "
            f"couleur du bleu reconnue {sum(1 for x in rs if x['couleurs_ok'])}/{len(rs)}"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"mode": "releve", "resultats": resultats}, fh, ensure_ascii=False, indent=1)
        print(f"\nrapport : {args.out}")
    return 0


async def main_async(args: argparse.Namespace) -> int:
    from sqlmodel import Session

    from app.database import engine
    from app.models.document import Document

    with Session(engine) as session:
        doc = session.get(Document, args.document)
        if doc is None or not doc.source_file_path:
            print(f"Document {args.document} introuvable ou sans fichier source.")
            return 1
        pdf_path, titre = doc.source_file_path, doc.title

    if args.mode == "prompts":
        print(f"[sonde] contrats de sortie — doc {args.document} « {titre} » — modèle {args.model}\n")
        return await mode_prompts(args, pdf_path)

    if args.mode == "releve":
        print(f"[sonde] relevé couleur — doc {args.document} « {titre} » — modèle {args.model}\n")
        return await mode_releve(args, pdf_path)

    variantes = [v.strip() for v in args.variantes.split(",") if v.strip()] or list(VARIANTES)
    cas = [c for c in CAS if not args.page or c[0] == args.page]

    print(f"[sonde] doc {args.document} « {titre} » — modèle {args.model}")
    print(f"[sonde] {len(cas)} cas × {len(variantes)} variante(s), grand côté {MAX_SIDE} px\n")

    # Rendu une seule fois par (page, variante) : le coût est dans l'appel, pas dans le PNG.
    rendus: Dict[Tuple[int, str], List[Tuple[str, str]]] = {}
    for page in sorted({c[0] for c in cas}):
        for nom in variantes:
            cols, rows = VARIANTES[nom]
            rendus[(page, nom)] = rendre(pdf_path, page, cols, rows)

    resultats: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def _un(page: int, question: str, attendu: str, piege: str, nom: str) -> Dict[str, Any]:
        async with semaphore:
            data = await interroger(rendus[(page, nom)], question, page, args.model)
        lu = _norm(data.get("valeur"))
        autres = [_norm(x) for x in (data.get("autres_valeurs_du_repere") or [])]
        return {
            "page": page, "question": question, "attendu": attendu, "piege": piege,
            "variante": nom, "images": len(rendus[(page, nom)]),
            "lu": lu, "autres": autres, "repere": data.get("repere"),
            "juste": bool(lu) and lu == _norm(attendu),
            "piege_lu": bool(piege) and lu == _norm(piege),
            "erreur": data.get("erreur"), "ms": data.get("ms"),
        }

    taches = [
        _un(page, question, attendu, piege, nom)
        for nom in variantes
        for (page, question, attendu, piege) in cas
    ]
    for coro in asyncio.as_completed(taches):
        resultats.append(await coro)

    # ——— Rapport ———
    print(f"{'cas':52s} " + "  ".join(f"{v:>14s}" for v in variantes))
    for (page, question, attendu, piege) in cas:
        cells = []
        for nom in variantes:
            r = next(
                (x for x in resultats if x["question"] == question and x["variante"] == nom), None
            )
            if r is None or r.get("erreur"):
                cells.append(f"{'ERREUR':>14s}")
            else:
                marque = "OK " if r["juste"] else ("!! " if r["piege_lu"] else "   ")
                cells.append(f"{marque + (r['lu'] or '—'):>14s}")
        libelle = re.sub(r"\s+", " ", question)[:44]
        print(f"p.{page:<3d} {libelle:47s} " + "  ".join(cells) + f"   [attendu {attendu}]")

    print()
    for nom in variantes:
        rs = [x for x in resultats if x["variante"] == nom and not x.get("erreur")]
        if not rs:
            print(f"  {nom:14s} : aucun résultat")
            continue
        justes = sum(1 for x in rs if x["juste"])
        pieges = sum(1 for x in rs if x["piege_lu"])
        ms = sorted(x["ms"] or 0 for x in rs)
        print(
            f"  {nom:14s} : {justes}/{len(rs)} justes · {pieges} valeur(s) piège · "
            f"{rs[0]['images']} image(s)/appel · médiane {ms[len(ms) // 2] / 1000:.1f}s"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"document": args.document, "model": args.model, "resultats": resultats}, fh,
                      ensure_ascii=False, indent=1)
        print(f"\nrapport : {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=int, default=438)
    parser.add_argument("--mode", default="valeur", choices=["valeur", "releve", "prompts"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--variantes", default="")
    parser.add_argument("--page", type=int, default=0, help="Ne garder que les cas de cette page.")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if not args.model:
        from app.config import settings

        args.model = settings.MODEL_FAST
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
