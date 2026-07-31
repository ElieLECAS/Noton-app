"""Import d'un arbre SAV depuis un JSON « pivot » rédigé par un LLM (Gemini, ChatGPT…).

Le format interne (`node_key` + `choices[].next_node_key`) est trop verbeux et trop
fragile pour être produit à la main ou par un LLM : les arêtes y sont portées par le
parent, donc rattacher un cas à plusieurs parents oblige à éditer N nœuds.

Le format pivot inverse la responsabilité : **chaque cas déclare ses parents**. C'est
exactement le geste de l'UI (« ce cas répond à quel(s) cas du dessus ? ») et ça rend le
multi-parent trivial. Le convertisseur ci-dessous reconstruit les `choices`, la racine,
les feuilles, et résout les documents de la bibliothèque par titre.

Schéma attendu (clés françaises, alias anglais tolérés) :

    {
      "titre": "Serrure motorisée ROTO",
      "symptome": {"nom": "Serrure motorisée", "synonymes": ["serrure qui bipe"]},
      "description": "Quand utiliser cet arbre.",
      "question_depart": "De quel produit s'agit-il ?",
      "perimetre": {"fournisseur": ["Roto"], "materiaux": [], "familles": [], "gammes": []},
      "cas": [
        {"id": "eneo", "nom": "Serrure Eneo CC", "description": "…", "parents": [],
         "type": "aiguillage", "outils": "", "photo": false,
         "sources": [{"document": "Eneo CC — Notice simplifiée", "pages": "7-9",
                      "precision": "tableau des erreurs"}]}
      ]
    }

`parents: []` (ou absent) = cas de premier étage, rattaché à la racine.
`type` : aiguillage (par défaut) | solution (feuille résolue) | sav (feuille escalade).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.document import Document
from app.models.document_category import DocumentCategory
from app.models.guided_entry import GuidedSymptomAlias

logger = logging.getLogger(__name__)

ROOT_KEY = "root"
MAX_CASES = 400

# Types de cas → (is_terminal, termination_type, step_type)
CASE_TYPES: Dict[str, Tuple[bool, Optional[str], str]] = {
    "aiguillage": (False, None, "question"),
    "solution": (True, "resolution", "resolution"),
    "sav": (True, "escalation", "escalation"),
}
TYPE_ALIASES = {
    "": "aiguillage",
    "question": "aiguillage",
    "branch": "aiguillage",
    "etape": "aiguillage",
    "étape": "aiguillage",
    "resolution": "solution",
    "résolution": "solution",
    "reparation": "solution",
    "réparation": "solution",
    "fix": "solution",
    "escalade": "sav",
    "escalation": "sav",
    "technicien": "sav",
    "pro": "sav",
}

# perimetre : libellés lisibles → axes réels de Document / GuidedTree.perimeter
PERIMETER_KEYS = {
    "materiaux": "materials",
    "matériaux": "materials",
    "materials": "materials",
    "familles": "product_types",
    "famille": "product_types",
    "product_types": "product_types",
    "fournisseur": "source",
    "fournisseurs": "source",
    "source": "source",
    "gammes": "proferm_gammes",
    "gamme": "proferm_gammes",
    "proferm_gammes": "proferm_gammes",
}


def _norm(value: str) -> str:
    v = unicodedata.normalize("NFD", (value or "").lower())
    v = "".join(c for c in v if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", v).strip()


def _key(value: str) -> str:
    k = re.sub(r"[^a-z0-9]+", "_", _norm(value)).strip("_")
    return (k or "cas")[:110]


# Marqueurs de citation que les LLM laissent dans leur texte : « [cite: 5] »,
# « [cite: 11, 13] », « 【4:2†source】 ». Ils ne doivent jamais atteindre le client.
_CITE_MARKERS = re.compile(r"\[\s*cite[^\]]*\]|【[^】]*】|\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", re.I)


def _clean(value: Any) -> str:
    """Nettoie un texte destiné au client ou au rédacteur : marqueurs de citation
    retirés, espaces avant ponctuation recollés."""
    text = _CITE_MARKERS.sub("", str(value or ""))
    text = re.sub(r"\(\s*\)|\[\s*\]", "", text)
    # Le français garde une espace avant ? ! : ; — on la normalise sans la supprimer,
    # mais la virgule et le point se recollent.
    text = re.sub(r"[ \t]+([,.])", r"\1", text)
    text = re.sub(r"[ \t]+([;:!?])", r" \1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _first(data: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for n in names:
        if n in data and data[n] not in (None, ""):
            return data[n]
    return default


# ---------------------------------------------------------------------------
# Lecture tolérante du texte collé
# ---------------------------------------------------------------------------


def parse_json_text(raw: str) -> Dict[str, Any]:
    """Accepte le JSON brut, entouré de ```json … ``` ou précédé d'un préambule."""
    text = (raw or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Rien à importer : le JSON est vide.")
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise HTTPException(
                status_code=422,
                detail="Le texte collé n'est pas du JSON valide (aucun objet { … } trouvé).",
            )
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"JSON invalide ligne {exc.lineno}, colonne {exc.colno} : {exc.msg}.",
            )
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Le JSON doit être un objet, pas une liste.")
    return data


def is_native_payload(data: Dict[str, Any]) -> bool:
    """Format interne (export de l'app) plutôt que pivot LLM."""
    return isinstance(data.get("nodes"), list) and "meta" in data


# ---------------------------------------------------------------------------
# Résolution des documents de la bibliothèque
# ---------------------------------------------------------------------------


def _resolve_document(session: Session, label: str, cache: Dict[str, Optional[int]]) -> Optional[int]:
    # Un LLM cite volontiers le NOM DE FICHIER joint (« guide.pdf ») là où la
    # bibliothèque porte un titre nettoyé : l'extension ne doit pas peser.
    wanted = _norm(re.sub(r"\.(pdf|docx?|xlsx?|pptx?)$", "", (label or "").strip(), flags=re.I))
    if not wanted:
        return None
    if wanted in cache:
        return cache[wanted]
    docs = session.exec(select(Document.id, Document.title)).all()
    best: Optional[int] = None
    best_score = 0.0
    for doc_id, title in docs:
        t = _norm(title or "")
        if not t:
            continue
        if t == wanted:
            best, best_score = doc_id, 10.0
            break
        if wanted in t or t in wanted:
            score = min(len(wanted), len(t)) / max(len(wanted), len(t))
            if score > best_score:
                best, best_score = doc_id, score
            continue
        # Recouvrement de mots significatifs (≥ 4 lettres) — tolère les titres remaniés.
        w1 = {m for m in re.findall(r"[a-z0-9]{4,}", wanted)}
        w2 = {m for m in re.findall(r"[a-z0-9]{4,}", t)}
        if w1 and w2:
            score = len(w1 & w2) / len(w1)
            if score >= 0.6 and score > best_score:
                best, best_score = doc_id, score
    cache[wanted] = best
    return best


def _int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _span(raw: Any) -> Tuple[Optional[int], Optional[int]]:
    """« 7-9 », « 7 à 9 », « 24, 25 », 8 → (7, 9) / (24, 25) / (8, 8)."""
    found = re.findall(r"\d+", str(raw or ""))
    if not found:
        return None, None
    start = _int(found[0])
    end = _int(found[-1]) if len(found) > 1 else start
    if start and end and end < start:
        start, end = end, start
    return start, end


def _pages(source: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], str]:
    """Retourne (début, fin, mention) en pages RÉELLES du fichier.

    Piège rencontré deux fois : un guide fabricant est souvent un extrait dont les pages
    imprimées ne commencent pas à 1 (un PDF de 16 pages numérotées 110→125). Or le texte
    est indexé par rang dans le fichier. `page_fichier` fait donc foi quand elle est
    fournie, et le numéro imprimé part en légende pour rester citable au client.
    """
    printed = _first(source, "pages", "page_start", "page_debut", "page_début", "page", default="")
    p_end = _first(source, "page_end", "page_fin", default=None)
    printed_start, printed_end = _span(printed)
    if p_end is not None and _int(p_end):
        printed_end = _int(p_end)

    real_start, real_end = _span(
        _first(source, "page_fichier", "pages_fichier", "page_pdf", "page_physique", default="")
    )
    if real_start:
        mention = f"page imprimée {printed_start}" if printed_start and printed_start != real_start else ""
        return real_start, (real_end or real_start), mention
    return printed_start, (printed_end or printed_start), ""


# ---------------------------------------------------------------------------
# Conversion pivot → format interne
# ---------------------------------------------------------------------------


def convert_pivot(session: Session, data: Dict[str, Any]) -> Dict[str, Any]:
    """Valide le JSON pivot et retourne `{payload, symptom, report}`.

    `payload` est directement consommable par save_tree_draft / import_tree.
    Les erreurs bloquantes lèvent un 422 avec la liste complète (pas la première),
    pour que l'auteur corrige son JSON en une passe.
    """
    errors: List[str] = []
    warnings: List[str] = []

    title = _clean(_first(data, "titre", "title", default=""))
    if not title:
        errors.append("Le champ « titre » est obligatoire (nom de l'arbre).")

    cases = _first(data, "cas", "nodes", "noeuds", "nœuds", "steps", default=None)
    if not isinstance(cases, list) or not cases:
        errors.append("Le champ « cas » doit être une liste non vide.")
        cases = []
    if len(cases) > MAX_CASES:
        errors.append(f"Trop de cas ({len(cases)}) — maximum {MAX_CASES}.")
        cases = cases[:MAX_CASES]

    # --- Passe 1 : identités ------------------------------------------------
    entries: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    alias_to_key: Dict[str, str] = {}  # id d'origine ET nom normalisés → clé finale
    for i, raw in enumerate(cases, start=1):
        if not isinstance(raw, dict):
            errors.append(f"Cas n°{i} : ce n'est pas un objet JSON.")
            continue
        name = _clean(_first(raw, "nom", "name", "titre", "title", "label", default=""))
        raw_id = str(_first(raw, "id", "cle", "clé", "key", "node_key", default="")).strip()
        if not name:
            errors.append(f"Cas n°{i} ({raw_id or 'sans id'}) : « nom » est obligatoire.")
            continue
        key = _key(raw_id or name)
        if key == ROOT_KEY:
            key = f"{key}_1"
        if key in by_key:
            errors.append(f"Cas « {name} » : l'identifiant « {key} » est utilisé deux fois.")
            continue
        entry = {"key": key, "name": name, "raw": raw}
        entries.append(entry)
        by_key[key] = entry
        for alias in {_norm(raw_id), _norm(name)}:
            if alias:
                alias_to_key.setdefault(alias, key)

    # --- Passe 2 : parents --------------------------------------------------
    for entry in entries:
        parents_raw = _first(entry["raw"], "parents", "parent", "parents_ids", "depuis", default=[])
        if isinstance(parents_raw, str):
            parents_raw = [parents_raw]
        resolved: List[str] = []
        for p in parents_raw or []:
            token = _norm(str(p))
            if not token or token in ("root", "racine", "depart", "départ"):
                continue
            target = alias_to_key.get(token)
            if target is None:
                errors.append(
                    f"Cas « {entry['name']} » : parent « {p} » inconnu "
                    "(aucun cas ne porte cet id ni ce nom)."
                )
                continue
            if target == entry["key"]:
                errors.append(f"Cas « {entry['name']} » : il ne peut pas être son propre parent.")
                continue
            if target not in resolved:
                resolved.append(target)
        entry["parents"] = resolved

    if errors:
        raise HTTPException(status_code=422, detail={"message": "JSON à corriger", "errors": errors})

    # --- Cycles : un DAG est obligatoire (le runtime descend) ---------------
    children: Dict[str, List[str]] = {e["key"]: [] for e in entries}
    for e in entries:
        for p in e["parents"]:
            children[p].append(e["key"])
    state: Dict[str, int] = {}

    def walk(key: str, trail: List[str]) -> None:
        state[key] = 1
        for child in children.get(key, []):
            if state.get(child) == 1:
                cycle = " → ".join(by_key[k]["name"] for k in trail[trail.index(child) :] + [child])
                errors.append(f"Boucle interdite dans l'arbre : {cycle}.")
                continue
            if state.get(child) is None:
                walk(child, trail + [child])
        state[key] = 2

    for e in entries:
        if state.get(e["key"]) is None:
            walk(e["key"], [e["key"]])
    if errors:
        raise HTTPException(status_code=422, detail={"message": "JSON à corriger", "errors": errors})

    # --- Passe 3 : nœuds internes ------------------------------------------
    doc_cache: Dict[str, Optional[int]] = {}
    nodes: List[Dict[str, Any]] = []
    roots = [e for e in entries if not e["parents"]]
    if not roots:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "JSON à corriger",
                "errors": [
                    "Aucun cas de premier étage : au moins un cas doit avoir "
                    "« parents »: [] pour être proposé en premier."
                ],
            },
        )

    for entry in entries:
        raw = entry["raw"]
        declared = TYPE_ALIASES.get(_norm(str(_first(raw, "type", "kind", default=""))), None)
        if declared is None:
            declared = _norm(str(_first(raw, "type", "kind", default=""))) or "aiguillage"
        if declared not in CASE_TYPES:
            warnings.append(
                f"Cas « {entry['name']} » : type « {declared} » inconnu, traité comme aiguillage."
            )
            declared = "aiguillage"
        kids = children.get(entry["key"], [])
        if kids and declared != "aiguillage":
            warnings.append(
                f"Cas « {entry['name']} » : marqué « {declared} » mais d'autres cas en dépendent "
                "— traité comme aiguillage."
            )
            declared = "aiguillage"
        if not kids and declared == "aiguillage":
            warnings.append(
                f"Cas « {entry['name']} » : aucun cas en dessous et aucune issue déclarée "
                "— traité comme une solution."
            )
            declared = "solution"
        is_terminal, termination, step_type = CASE_TYPES[declared]

        attachments: List[Dict[str, Any]] = []
        sources = _first(raw, "sources", "source_documents", "documents", "notices", default=[])
        if isinstance(sources, dict):
            sources = [sources]
        for src in sources or []:
            if isinstance(src, str):
                src = {"document": src}
            if not isinstance(src, dict):
                continue
            label = str(_first(src, "document", "titre", "title", "nom", default="")).strip()
            if not label:
                continue
            doc_id = _resolve_document(session, label, doc_cache)
            if doc_id is None:
                warnings.append(
                    f"Cas « {entry['name']} » : document « {label} » absent de la bibliothèque "
                    "— notice non rattachée."
                )
                continue
            p1, p2, mention = _pages(src)
            caption = _clean(_first(src, "precision", "précision", "caption", "note", default=""))
            if mention:
                caption = f"{mention} — {caption}" if caption else mention
            attachments.append(
                {
                    "document_id": doc_id,
                    "page_start": p1,
                    "page_end": p2,
                    "caption": caption[:300],
                    "kind": "notice",
                }
            )

        nodes.append(
            {
                "node_key": entry["key"],
                "step_type": step_type,
                "title": entry["name"][:200],
                "message": _clean(_first(raw, "description", "message", "detail", "texte", default="")),
                "internal_note": _clean(_first(raw, "note_interne", "internal_note", default="")),
                "is_terminal": is_terminal,
                "termination_type": termination,
                "ask_photo": bool(_first(raw, "photo", "ask_photo", "demander_photo", default=False)),
                "allow_free_text": True,
                "tools_hint": _clean(_first(raw, "outils", "tools", "tools_hint", default=""))[:200],
                "choices": [],
                "attachments": attachments,
            }
        )

    node_by_key = {n["node_key"]: n for n in nodes}

    def choice_for(child_key: str) -> Dict[str, Any]:
        child = node_by_key[child_key]
        return {
            "label": child["title"],
            "value": f"v_{child_key}"[:120],
            "hint": "",
            "next_node_key": child_key,
        }

    for entry in entries:
        for parent in entry["parents"]:
            node_by_key[parent]["choices"].append(choice_for(entry["key"]))

    # Racine : porte la question de départ et pointe vers les cas du premier étage.
    root_question = _clean(
        _first(data, "question_depart", "question_départ", "question", default="")
    )
    nodes.insert(
        0,
        {
            "node_key": ROOT_KEY,
            "step_type": "question",
            "title": title[:200],
            "message": root_question,
            "internal_note": "",
            "is_terminal": False,
            "termination_type": None,
            "ask_photo": False,
            "allow_free_text": True,
            "tools_hint": "",
            "choices": [choice_for(e["key"]) for e in roots],
            "attachments": [],
        },
    )

    # --- Symptôme d'entrée --------------------------------------------------
    symptom_raw = _first(data, "symptome", "symptôme", "symptom", default=None)
    symptom: Optional[Dict[str, Any]] = None
    if isinstance(symptom_raw, str) and symptom_raw.strip():
        symptom = {"label": symptom_raw.strip(), "aliases": []}
    elif isinstance(symptom_raw, dict):
        label = str(_first(symptom_raw, "nom", "label", "titre", "slug", default="")).strip()
        if label:
            syns = _first(symptom_raw, "synonymes", "aliases", "alias", default=[]) or []
            symptom = {
                "label": label,
                "aliases": [str(s).strip() for s in syns if str(s).strip()],
                "description": str(_first(symptom_raw, "description", default="")).strip(),
            }

    # --- Périmètre ----------------------------------------------------------
    perimeter: Dict[str, List[str]] = {}
    for k, v in (_first(data, "perimetre", "périmètre", "perimeter", default={}) or {}).items():
        axis = PERIMETER_KEYS.get(_norm(str(k)))
        if not axis:
            warnings.append(f"Périmètre : critère « {k} » ignoré (inconnu).")
            continue
        values = [str(x).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
        if values:
            perimeter.setdefault(axis, []).extend(values)

    payload = {
        "meta": {
            "title": title,
            "description": _clean(_first(data, "description", default="")),
            "entry_symptom": _key(symptom["label"]) if symptom else None,
            "perimeter": perimeter or None,
            "root_node_key": ROOT_KEY,
            "layout": {},
        },
        "nodes": nodes,
    }
    report = {
        "cases": len(nodes),
        "first_level": len(roots),
        "solutions": sum(1 for n in nodes if n["termination_type"] == "resolution"),
        "sav": sum(1 for n in nodes if n["termination_type"] == "escalation"),
        "attachments": sum(len(n["attachments"]) for n in nodes),
        "shared": sum(1 for e in entries if len(e["parents"]) > 1),
        "warnings": warnings,
    }
    return {"payload": payload, "symptom": symptom, "report": report}


def ensure_symptom(session: Session, symptom: Dict[str, Any]) -> str:
    """Crée le symptôme (DocumentCategory axis=symptom) et ses synonymes si besoin."""
    slug = _key(symptom["label"])
    existing = session.exec(select(DocumentCategory).where(DocumentCategory.slug == slug)).first()
    if existing is None:
        session.add(
            DocumentCategory(
                slug=slug,
                label=symptom["label"][:200],
                axis="symptom",
                description=str(symptom.get("description") or "")[:500],
            )
        )
    for alias in symptom.get("aliases") or []:
        found = session.exec(
            select(GuidedSymptomAlias).where(
                GuidedSymptomAlias.symptom_slug == slug, GuidedSymptomAlias.alias == alias
            )
        ).first()
        if found is None:
            session.add(GuidedSymptomAlias(symptom_slug=slug, alias=alias[:200]))
    session.commit()
    return slug


def import_from_json_text(
    session: Session, raw: str, *, space_id: Optional[int], user_id: int
) -> Dict[str, Any]:
    """Point d'entrée : texte collé → arbre en brouillon + rapport d'import."""
    from app.services.guided_authoring_service import get_tree_draft, import_tree

    data = parse_json_text(raw)
    if is_native_payload(data):
        tree = import_tree(session, data, space_id=space_id, user_id=user_id)
        draft = get_tree_draft(session, tree.id)
        draft["import_report"] = {
            "cases": len(draft["nodes"]),
            "warnings": ["Format interne détecté (export de l'application) — importé tel quel."],
        }
        return draft

    converted = convert_pivot(session, data)
    if converted["symptom"]:
        ensure_symptom(session, converted["symptom"])
    tree = import_tree(session, converted["payload"], space_id=space_id, user_id=user_id)
    draft = get_tree_draft(session, tree.id)
    draft["import_report"] = converted["report"]
    logger.info(
        "[guided_json_import] arbre « %s » importé : %d cas, %d notices, %d avertissements",
        draft["meta"]["title"],
        converted["report"]["cases"],
        converted["report"]["attachments"],
        len(converted["report"]["warnings"]),
    )
    return draft
