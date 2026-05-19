"""
Service d'extraction d'entités KAG (Knowledge Augmented Generation).

Extrait les entités techniques des chunks via LLM configurable (OpenAI, Mistral ou Ollama)
pour enrichir le graphe de connaissances et améliorer le RAG.
"""

import json
import re
import unicodedata
import logging
import asyncio
from typing import List, Dict, Optional
from app.config import settings
from app.tracing import trace_run

logger = logging.getLogger(__name__)

# Nouvelle taxonomie métier des entités KAG.
# Chaque type a :
# - id        : valeur technique utilisée dans le champ "entity_type" (DB) et "type" (JSON LLM)
# - label     : label métier lisible (reprend le tableau fourni)
# - description : portée de ce que l'IA doit capturer
# - examples  : quelques exemples clés pour guider le LLM
ENTITY_TYPES_CONFIG: List[Dict[str, object]] = [
    {
        "id": "gamme_systeme",
        "label": "GAMME_SYSTEME",
        "description": "Noms commerciaux et systèmes de profils profine/Kömmerling.",
        "examples": [
            "76 Advanced",
            "InnoSlide",
            "Trocal",
            "KBE",
            "Hybride",
            "Luméal",
        ],
    },
    {
        "id": "profil_code",
        "label": "PROFIL_CODE",
        "description": "Pièces de structure, traverses et leurs numéros techniques.",
        "examples": [
            "Dormant 76171",
            "Ouvrant 76274",
            "Meneau",
            "AluClip",
            "Parclose",
        ],
    },
    {
        "id": "concept_technique",
        "label": "CONCEPT_TECHNIQUE",
        "description": "Procédés spécifiques de fabrication et d'assemblage en usine.",
        "examples": [
            "Soudure grain d'ange",
            "Ébavurage",
            "Sertissage",
            "Décompression",
        ],
    },
    {
        "id": "regle_pose",
        "label": "REGLE_POSE",
        "description": "Instructions d'installation et méthodes de mise en œuvre.",
        "examples": [
            "Applique",
            "Tunnel",
            "Calfeutrement",
            "Fond de joint",
            "Fixation",
        ],
    },
    {
        "id": "performance_test",
        "label": "PERFORMANCE_TEST",
        "description": "Valeurs certifiées, unités de mesure et résultats de tests.",
        "examples": [
            "Uw 1.2",
            "A*4 E*9A V*4 3",
            "dB (acoustique)",
            "Essai de rupture",
        ],
    },
    {
        "id": "pathologie_desordre",
        "label": "PATHOLOGIE_DESORDRE",
        "description": "Problèmes techniques, sinistres et défauts (CSTB/AQC).",
        "examples": [
            "Condensation",
            "Défaut d'étanchéité",
            "Déformation",
            "Corrosion",
        ],
    },
    {
        "id": "composant_acc",
        "label": "COMPOSANT_ACC",
        "description": "Quincaillerie, joints, renforts acier et petits accessoires.",
        "examples": [
            "Roto",
            "SoftClose",
            "Joint central",
            "Renfort 194",
            "Crémone",
        ],
    },
    {
        "id": "norme_doc",
        "label": "NORME_DOC",
        "description": "Références réglementaires, lois et labels de qualité.",
        "examples": [
            "DTU 36.5",
            "DTA 6/16-2334",
            "NF",
            "CEKAL",
            "Avis Technique",
        ],
    },
    {
        "id": "materiau_finition",
        "label": "MATERIAU_FINITION",
        "description": "Nature des profils (PVC/Alu) et leur aspect visuel.",
        "examples": [
            "PVC Greenline",
            "Alu bas carbone",
            "Plaxé Gris T016",
            "Laqué",
        ],
    },
    {
        "id": "garantie_duree",
        "label": "GARANTIE_DUREE",
        "description": "Temps de couverture spécifique par type de composant.",
        "examples": [
            "15 ans (structure)",
            "5 ans (plaxage)",
            "7 ans (laquage)",
        ],
    },
]

SUPPORTED_ENTITY_TYPE_IDS = [t["id"] for t in ENTITY_TYPES_CONFIG]

CRITICAL_ENTITY_TYPES = {
    "garantie_duree",
    "performance_test",
    "norme_doc",
}

ENTITY_TYPES_PROMPT_BLOCK = "\n".join(
    f"- {t['id']}: {t['description']} Exemples: {', '.join(t['examples'])}."
    for t in ENTITY_TYPES_CONFIG
)


PARENT_SUMMARY_PROMPT_TEMPLATE = """Analyse ce passage de document technique et retourne UNIQUEMENT un JSON valide sans markdown.

Format attendu:
{{"summary": "résumé en 1-2 phrases", "generated_questions": ["question 1", "question 2", "question 3"]}}

Règles:
- Le résumé doit capturer l'intention métier de la section (ce que cette section apporte, pas comment elle est structurée)
- Les 3 questions doivent simuler des questions réelles d'un technicien ou d'un technico-commercial face au document
- Questions courtes et précises (interrogatives, sans réponse)
- Retourne UNIQUEMENT le JSON, sans texte avant ni après

Passage:
{content}

JSON:"""


# Relations typées strictes (co_occurs = uniquement mécanique via refresh_entity_entity_relations)
CANONICAL_RELATION_TYPE_IDS = frozenset(
    {
        "appartient_a",
        "compatible_avec",
        "remplace",
        "contrainte",
        "reference",
        "cause",
    }
)

# Compatibilité lecture (anciennes arêtes en base)
LEGACY_RELATION_TYPE_IDS = frozenset({"depend_de", "co_occurs"})

TYPED_RELATION_TYPE_IDS = CANONICAL_RELATION_TYPE_IDS | LEGACY_RELATION_TYPE_IDS

# Types vagues interdits (LLM ne doit pas inventer de libellés génériques)
VAGUE_RELATION_BLOCKLIST = frozenset(
    {
        "co_occurs",
        "est_lie_a",
        "lie_a",
        "lie",
        "associe_a",
        "associe",
        "relie_a",
        "relie",
        "est_associe_a",
        "connexe",
        "connexe_a",
        "en_lien_avec",
        "linked_to",
        "related_to",
    }
)

RELATION_TYPE_ALIASES = {
    "depend_de": "appartient_a",
    "fait_partie_de": "appartient_a",
    "appartient_a": "appartient_a",
    "compatible_avec": "compatible_avec",
    "remplace": "remplace",
    "contrainte": "contrainte",
    "reference": "reference",
    "cause": "cause",
}

_ARTICLE_PREFIX_RE = re.compile(
    r"^(?:la|le|les|l|un|une|des|du|de la|d|ce|cette|cet|ces|this|the)\s+",
    re.IGNORECASE,
)

_COREFERENCE_MENTION_RE = re.compile(
    r"^(?:la|le|les|l|ce|cette|cet|ces|celle|celui|celle-ci|celui-ci|"
    r"cette série|ce système|cette gamme|ce profil)\b",
    re.IGNORECASE,
)

RELATION_EXTRACTION_PROMPT_TEMPLATE = """Tu analyses un passage technique et les entités déjà identifiées dans ce passage.

Entités extraites (noms EXACTS à recopier pour entity_a / entity_b) :
{entity_names}

Types de relation AUTORISÉS UNIQUEMENT (champ "relation_type") :
- appartient_a : hiérarchie, appartenance, inclusion (gamme > produit, système > composant)
- compatible_avec : compatibilité technique, association permise, fonctionnement conjoint
- remplace : substitution, évolution, remplacement d'une référence par une autre
- contrainte : incompatibilité, interdiction, limite réglementaire ou technique
- reference : renvoi normatif, citation documentaire, renvoi sans causalité directe
- cause : causalité explicite (A provoque / entraîne B)

INTERDIT : relations vagues ("est lié à", "associé à", "en rapport avec") ou co_occurs (géré ailleurs).
Si le lien n'est pas clair ou pas typable → n'écris PAS la paire.

Règles :
- Retourne UNIQUEMENT un JSON valide (tableau), sans markdown
- entity_a et entity_b : orthographe IDENTIQUE à la liste ci-dessus
- confidence entre 0.0 et 1.0 (>= 0.6 si relation explicite dans le texte)

Format :
[{{"entity_a": "...", "entity_b": "...", "relation_type": "appartient_a", "confidence": 0.82}}]

Texte :
{chunk_content}

JSON :"""


EXTRACTION_PROMPT_TEMPLATE = """Extrais les entités techniques de ce texte.

Types possibles pour le champ "type" du JSON (liste exhaustive) :
{entity_types}

{context_section}

Règles:
- Retourne UNIQUEMENT un JSON valide, sans markdown ni commentaires
- Maximum 10 entités par chunk
- Importance entre 0.0 et 1.0 (1.0 = très important)
- Noms courts et précis (pas de phrases entières)
- Utilise EXACTEMENT l'une des valeurs suivantes pour le champ "type" : {entity_types}

Résolution des coréférences (CRITIQUE) :
- Si le texte dit « la gamme », « cette série », « ce système » en renvoyant à une entité déjà nommée,
  utilise le champ "canonical_name" avec le nom complet déjà introduit (ex. "Gamme Alpha").
- Le champ "name" peut reprendre la forme courte telle qu'écrite ; "canonical_name" pointe vers l'entité mère.
- Ne crée PAS deux entités pour « Gamme Alpha » et « la gamme » si c'est la même chose.
- Les acronymes ou abréviations évidentes (ex. « 76 » pour « 76 Advanced ») : canonical_name = forme complète.

Format attendu:
[{{"name": "nom tel qu'écrit", "canonical_name": "nom canonique ou omis", "type": "type", "importance": 0.8}}]

Texte:
{chunk_content}

JSON:"""


def normalize_entity_name(name: str) -> str:
    """
    Normalise un nom d'entité pour la déduplication.
    
    - Lowercase
    - Suppression des accents
    - Remplacement des tirets/underscores par espaces
    - Suppression des caractères spéciaux
    - Trim des espaces
    """
    if not name:
        return ""
    normalized = name.lower().strip()
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    normalized = re.sub(r"[-_]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_entity_core(name: str) -> str:
    """Clé de regroupement sans articles définis en tête (coréférences)."""
    core = normalize_entity_name(name)
    while core:
        m = _ARTICLE_PREFIX_RE.match(core)
        if not m:
            break
        core = core[m.end() :].strip()
    return core


def is_probably_coreference_mention(name: str) -> bool:
    """Détecte les mentions anaphoriques courtes (la gamme, cette série, …)."""
    if not name or len(name.strip()) < 3:
        return False
    raw = name.strip()
    if _COREFERENCE_MENTION_RE.match(raw):
        return True
    lowered = raw.lower()
    if lowered.startswith(
        ("la ", "le ", "les ", "l'", "une ", "un ", "des ", "du ", "de la ", "ce ", "cette ", "cet ", "ces ")
    ):
        return len(lowered.split()) <= 4
    return False


def normalize_relation_type(relation_type: str) -> Optional[str]:
    """
    Normalise un type de relation LLM vers le schéma strict.
    Retourne None si vague ou non autorisé.
    """
    if not relation_type:
        return None
    rt = str(relation_type).strip().lower().replace(" ", "_").replace("-", "_")
    if rt in VAGUE_RELATION_BLOCKLIST:
        return None
    mapped = RELATION_TYPE_ALIASES.get(rt, rt)
    if mapped in CANONICAL_RELATION_TYPE_IDS:
        return mapped
    return None


def resolve_entities_coreference_in_chunk(entities: List[Dict]) -> List[Dict]:
    """
    Fusionne les variantes d'un même chunk avant persistance :
    - canonical_name explicite (LLM)
    - déduplication par nom normalisé
    - rapprochement embedding des mentions anaphoriques vers ancres du chunk
    """
    if not entities:
        return []

    if not getattr(settings, "KAG_COREFERENCE_ENABLED", True):
        return _dedupe_entities_by_normalized_name(entities)

    working: List[Dict] = []
    for e in entities:
        name = (e.get("name") or "").strip()
        if not name or len(name) < 2:
            continue
        canonical = (e.get("canonical_name") or "").strip()
        aliases: List[str] = list(e.get("aliases") or [])
        final_name = canonical if canonical and len(canonical) >= 2 else name
        if canonical and normalize_entity_name(canonical) != normalize_entity_name(name):
            aliases.append(name)
        working.append(
            {
                "name": final_name,
                "type": (e.get("type") or "concept_technique").strip(),
                "importance": float(e.get("importance", 1.0) or 1.0),
                "aliases": aliases,
            }
        )

    working = _dedupe_entities_by_normalized_name(working)

    anchors = [e for e in working if not is_probably_coreference_mention(e["name"])]
    mentions = [e for e in working if is_probably_coreference_mention(e["name"])]
    if not mentions or not anchors:
        return working

    try:
        from app.services.embedding_service import generate_embeddings_batch

        anchor_texts = [a["name"] for a in anchors]
        mention_texts = [m["name"] for m in mentions]
        vectors = generate_embeddings_batch(anchor_texts + mention_texts, batch_size=16)
        if not vectors or len(vectors) != len(anchor_texts) + len(mention_texts):
            return working

        import numpy as np

        threshold = float(getattr(settings, "KAG_ENTITY_MERGE_SIMILARITY", 0.88) or 0.88)
        resolved: List[Dict] = list(anchors)
        n_a = len(anchors)

        for mi, mention in enumerate(mentions):
            vec = vectors[n_a + mi]
            if not vec:
                resolved.append(mention)
                continue
            v = np.array(vec, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            best_idx = -1
            best_sim = -1.0
            for ai, anchor in enumerate(anchors):
                if anchor.get("type") != mention.get("type"):
                    continue
                av = vectors[ai]
                if not av:
                    continue
                a = np.array(av, dtype=np.float32)
                a = a / (np.linalg.norm(a) + 1e-9)
                sim = float(np.dot(v, a))
                if sim > best_sim:
                    best_sim = sim
                    best_idx = ai
            if best_idx >= 0 and best_sim >= threshold:
                anchor = anchors[best_idx]
                anchor["importance"] = max(
                    float(anchor.get("importance", 0)),
                    float(mention.get("importance", 0)),
                )
                nn = normalize_entity_name(mention["name"])
                if nn and nn not in [normalize_entity_name(a) for a in anchor.get("aliases", [])]:
                    anchor.setdefault("aliases", []).append(mention["name"])
                logger.debug(
                    "Coréférence chunk fusionnée: '%s' → '%s' (sim=%.3f)",
                    mention["name"],
                    anchor["name"],
                    best_sim,
                )
            else:
                resolved.append(mention)

        return _dedupe_entities_by_normalized_name(resolved)
    except Exception as err:
        logger.warning("Résolution coréférence chunk ignorée: %s", err)
        return working


def _dedupe_entities_by_normalized_name(entities: List[Dict]) -> List[Dict]:
    """Déduplique par nom normalisé en conservant la plus forte importance."""
    merged: Dict[str, Dict] = {}
    for e in entities:
        key = normalize_entity_name(e.get("name", ""))
        if not key:
            continue
        imp = float(e.get("importance", 1.0) or 1.0)
        aliases = list(e.get("aliases") or [])
        if key not in merged or imp > float(merged[key].get("importance", 0)):
            merged[key] = {
                "name": e["name"],
                "type": e.get("type", "concept_technique"),
                "importance": imp,
                "aliases": aliases,
            }
        else:
            merged[key]["aliases"].extend(aliases)
    return list(merged.values())


def _repair_truncated_json_array(text: str) -> str:
    """Tente de réparer un tableau JSON tronqué en supprimant le dernier élément incomplet."""
    text = text.strip()
    if not text.startswith("["):
        return text
    if text.endswith("]"):
        return text

    # Chercher la dernière accolade fermante d'un objet complet
    last_brace = text.rfind("}")
    if last_brace != -1:
        # On coupe juste après le dernier objet complet et on ferme proprement le tableau
        repaired = text[:last_brace + 1] + "\n]"
        return repaired
    return text


def _parse_typed_relations_response(response_text: str) -> List[Dict]:
    """Parse la réponse LLM en liste de relations typées."""
    if not response_text:
        return []
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        text = "\n".join(json_lines)
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        text = json_match.group()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.info("JSON des relations typées potentiellement tronqué, tentative de réparation...")
        repaired = _repair_truncated_json_array(text)
        try:
            data = json.loads(repaired)
            logger.info("JSON des relations typées réparé avec succès !")
        except json.JSONDecodeError:
            logger.warning("Erreur parsing JSON relations typées: %s - %s", e, text[:200])
            return []

    if not isinstance(data, list):
        return []
    out: List[Dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ea = str(item.get("entity_a", "")).strip()
        eb = str(item.get("entity_b", "")).strip()
        rt = normalize_relation_type(str(item.get("relation_type", "")))
        conf = item.get("confidence", 0.7)
        if not ea or not eb or ea.lower() == eb.lower():
            continue
        if not rt:
            continue
        if not isinstance(conf, (int, float)):
            conf = 0.7
        conf = max(0.0, min(1.0, float(conf)))
        out.append(
            {
                "entity_a": ea,
                "entity_b": eb,
                "relation_type": rt,
                "confidence": conf,
            }
        )
    return out


def _parse_llm_response(response_text: str) -> List[Dict]:
    """
    Parse la réponse LLM en liste d'entités.
    Gère les cas où le LLM ajoute du markdown ou du texte autour du JSON.
    """
    if not response_text:
        return []
    
    text = response_text.strip()
    
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        text = "\n".join(json_lines)
    
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        text = json_match.group()
    
    try:
        entities = json.loads(text)
    except json.JSONDecodeError as e:
        logger.info("JSON des entités potentiellement tronqué, tentative de réparation...")
        repaired = _repair_truncated_json_array(text)
        try:
            entities = json.loads(repaired)
            logger.info("JSON des entités réparé avec succès !")
        except json.JSONDecodeError:
            logger.warning("Erreur parsing JSON LLM: %s - Réponse: %s", e, text[:200])
            return []
        
    if not isinstance(entities, list):
        logger.warning("Réponse LLM n'est pas une liste: %s", type(entities))
        return []
    
    valid_entities = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        raw_type = str(e.get("type", "")).strip().lower()
        importance = e.get("importance", 1.0)
        
        if not name or len(name) < 2:
            continue
        if not raw_type or raw_type not in SUPPORTED_ENTITY_TYPE_IDS:
            # On ignore les entités dont le type n'est pas dans la nouvelle taxonomie
            continue
        if not isinstance(importance, (int, float)):
            importance = 1.0
        importance = max(0.0, min(1.0, float(importance)))
        if raw_type in CRITICAL_ENTITY_TYPES:
            importance = 1.0
        
        canonical_name = str(e.get("canonical_name", "")).strip()
        entry: Dict = {
            "name": name,
            "type": raw_type,
            "importance": importance,
        }
        if canonical_name and len(canonical_name) >= 2:
            entry["canonical_name"] = canonical_name
        valid_entities.append(entry)

    resolved = resolve_entities_coreference_in_chunk(valid_entities)
    return resolved[:10]


def _parse_summary_questions_response(response_text: str) -> Optional[Dict]:
    """
    Parse la réponse LLM pour le résumé + questions d'un chunk parent.
    Retourne un dict {"summary": str, "generated_questions": [str, str, str]} ou None.
    """
    if not response_text:
        return None

    text = response_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        text = "\n".join(json_lines)

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group()

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None

        summary = data.get("summary", "").strip()
        questions = data.get("generated_questions", [])

        if not summary:
            return None
        if not isinstance(questions, list):
            questions = []
        questions = [str(q).strip() for q in questions if q and str(q).strip()][:3]

        return {"summary": summary, "generated_questions": questions}
    except json.JSONDecodeError as e:
        logger.warning("Erreur parsing JSON résumé parent: %s - Réponse: %s", e, text[:200])
        return None


async def generate_parent_summary_questions(content: str) -> Optional[Dict]:
    """
    Génère un résumé + 3 questions pour un chunk parent (section) via LLM.

    Args:
        content: Contenu textuel du chunk parent

    Returns:
        Dict {"summary": str, "generated_questions": [str, str, str]} ou None en cas d'échec
    """
    if not content or len(content.strip()) < 30:
        return None

    content_truncated = content[:3000]
    prompt = PARENT_SUMMARY_PROMPT_TEMPLATE.format(content=content_truncated)

    provider = settings.KAG_EXTRACTION_PROVIDER.lower()
    model = settings.KAG_EXTRACTION_MODEL

    try:
        if provider == "openai":
            from app.services import openai_service
            response = await openai_service.chat(
                message=prompt,
                model=model,
                context=[{"role": "user", "content": prompt}],
            )
            raw = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif provider == "mistral":
            from app.services import mistral_service
            response = await mistral_service.chat(
                message=prompt,
                model=model,
                context=[{"role": "user", "content": prompt}],
            )
            raw = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif provider == "ollama":
            from app.services import ollama_service
            response = await ollama_service.chat(
                message=prompt,
                model=model,
                context=[{"role": "user", "content": prompt}],
            )
            raw = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            logger.error("Provider KAG inconnu pour enrichissement parent: %s", provider)
            return None

        result = _parse_summary_questions_response(raw)
        if result:
            logger.debug(
                "Enrichissement parent OK (provider=%s): summary=%d chars, questions=%d",
                provider,
                len(result["summary"]),
                len(result["generated_questions"]),
            )
        return result

    except Exception as e:
        logger.error("Erreur génération résumé parent: %s", e, exc_info=True)
        return None


def generate_parent_summary_questions_sync(content: str) -> Optional[Dict]:
    """
    Version synchrone de generate_parent_summary_questions.
    Utilisée dans les workers de background (même pattern que extract_entities_sync).
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    generate_parent_summary_questions(content),
                )
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(generate_parent_summary_questions(content))
    except RuntimeError:
        return asyncio.run(generate_parent_summary_questions(content))
    except Exception as e:
        logger.warning("Génération résumé parent échouée: %s", e)
        return None


async def extract_entities_from_chunk(chunk_content: str, context_hint: Optional[str] = None) -> List[Dict]:
    """
    Extrait les entités d'un chunk via LLM.
    
    Args:
        chunk_content: Contenu textuel du chunk
        context_hint: Contexte additionnel (ex: titre du doc, section)
        
    Returns:
        Liste de dicts {"name": str, "type": str, "importance": float}
    """
    if not chunk_content or len(chunk_content.strip()) < 20:
        return []
    
    content_truncated = chunk_content[:2000]
    context_section = f"Contexte du document : {context_hint}" if context_hint else ""
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        entity_types=", ".join(SUPPORTED_ENTITY_TYPE_IDS),
        context_section=context_section,
        chunk_content=content_truncated,
    )
    
    provider = settings.KAG_EXTRACTION_PROVIDER.lower()
    model = settings.KAG_EXTRACTION_MODEL

    with trace_run(
        "kag_entity_extraction",
        run_type="llm",
        inputs={
            "provider": provider,
            "model": model,
            "content_preview": chunk_content[:200],
            "content_len": len(chunk_content),
        },
        tags=["kag", "extraction", "llm", provider],
    ) as kag_run:
        try:
            if provider == "openai":
                from app.services import openai_service
                response = await openai_service.chat(
                    message=prompt,
                    model=model,
                    context=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            elif provider == "mistral":
                from app.services import mistral_service
                response = await mistral_service.chat(
                    message=prompt,
                    model=model,
                    context=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif provider == "ollama":
                from app.services import ollama_service
                response = await ollama_service.chat(
                    message=prompt,
                    model=model,
                    context=[{"role": "user", "content": prompt}],
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            else:
                logger.error("Provider KAG inconnu: %s", provider)
                kag_run.end(error=f"Provider KAG inconnu: {provider}")
                return []

            entities = _parse_llm_response(content)
            entity_types_found = list({e.get("type", "") for e in entities if e.get("type")})
            kag_run.end(outputs={
                "nb_entities": len(entities),
                "entity_types_found": entity_types_found,
                "entities_preview": [
                    {"name": e.get("name"), "type": e.get("type"), "importance": e.get("importance")}
                    for e in entities[:10]
                ],
            })
            logger.debug(
                "Extraction KAG: %d entités extraites (provider=%s, model=%s)",
                len(entities),
                provider,
                model,
            )
            return entities

        except Exception as e:
            logger.error("Erreur extraction KAG: %s", e, exc_info=True)
            kag_run.end(error=str(e))
            return []


async def extract_typed_relations_from_chunk(
    chunk_content: str,
    entities: List[Dict],
) -> List[Dict]:
    """
    Extrait des relations typées entre entités déjà listées, pour un même passage.

    Args:
        chunk_content: Texte du chunk
        entities: Liste d'entités canoniques {name, type, importance}

    Returns:
        Liste de dicts entity_a, entity_b, relation_type, confidence
    """
    if not chunk_content or len(chunk_content.strip()) < 20:
        return []
    if not entities or len(entities) < 2:
        return []

    names = []
    for e in entities:
        n = (e.get("name") or "").strip()
        if n and len(n) >= 2:
            names.append(n)
    if len(names) < 2:
        return []

    content_truncated = chunk_content[:3500]
    entity_names_block = "\n".join(f"- {n}" for n in names)
    prompt = RELATION_EXTRACTION_PROMPT_TEMPLATE.format(
        entity_names=entity_names_block,
        chunk_content=content_truncated,
    )

    provider = settings.KAG_EXTRACTION_PROVIDER.lower()
    model = settings.KAG_EXTRACTION_MODEL

    with trace_run(
        "kag_typed_relation_extraction",
        run_type="llm",
        inputs={
            "provider": provider,
            "model": model,
            "nb_entities": len(names),
            "content_len": len(chunk_content),
        },
        tags=["kag", "relations", "llm", provider],
    ) as rel_run:
        try:
            if provider == "openai":
                from app.services import openai_service
                response = await openai_service.chat(
                    message=prompt,
                    model=model,
                    context=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif provider == "mistral":
                from app.services import mistral_service
                response = await mistral_service.chat(
                    message=prompt,
                    model=model,
                    context=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif provider == "ollama":
                from app.services import ollama_service
                response = await ollama_service.chat(
                    message=prompt,
                    model=model,
                    context=[{"role": "user", "content": prompt}],
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.error("Provider KAG inconnu (relations typées): %s", provider)
                rel_run.end(error=f"Provider inconnu: {provider}")
                return []

            parsed = _parse_typed_relations_response(content)
            rel_run.end(
                outputs={
                    "nb_relations": len(parsed),
                    "preview": parsed[:8],
                }
            )
            return parsed
        except Exception as e:
            logger.error("Erreur extraction relations typées KAG: %s", e, exc_info=True)
            rel_run.end(error=str(e))
            return []


def extract_typed_relations_sync(chunk_content: str, entities: List[Dict]) -> List[Dict]:
    """Version synchrone pour workers / pipeline document."""
    if not chunk_content or not entities or len(entities) < 2:
        return []
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    extract_typed_relations_from_chunk(chunk_content, entities),
                )
                return future.result(timeout=90)
        else:
            return loop.run_until_complete(
                extract_typed_relations_from_chunk(chunk_content, entities)
            )
    except RuntimeError:
        return asyncio.run(extract_typed_relations_from_chunk(chunk_content, entities))
    except Exception as e:
        logger.warning("extract_typed_relations_sync échoué: %s", e)
        return []


async def extract_entities_from_query(query_text: str) -> List[str]:
    """
    Extrait les entités mentionnées dans une requête utilisateur.
    Version simplifiée pour le retrieval (juste les noms).
    
    Args:
        query_text: Texte de la requête
        
    Returns:
        Liste des noms d'entités normalisés
    """
    entities = await extract_entities_from_chunk(query_text)
    return [normalize_entity_name(e["name"]) for e in entities if e.get("name")]


def extract_entities_sync(chunk_content: str, context_hint: Optional[str] = None) -> List[Dict]:
    """
    Version synchrone de extract_entities_from_chunk.
    Utile pour les workers de background.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    extract_entities_from_chunk(chunk_content, context_hint)
                )
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(extract_entities_from_chunk(chunk_content, context_hint))
    except RuntimeError:
        return asyncio.run(extract_entities_from_chunk(chunk_content, context_hint))


def extract_entities_from_query_sync(query_text: str) -> List[str]:
    """
    Version synchrone de extract_entities_from_query.
    Retourne les noms d'entités normalisés de la requête (pour pivot KAG).
    """
    if not query_text or not query_text.strip():
        return []
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    extract_entities_from_query(query_text),
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(extract_entities_from_query(query_text))
    except RuntimeError:
        return asyncio.run(extract_entities_from_query(query_text))
    except Exception as e:
        logger.warning("Extraction entités requête échouée: %s", e)
        return []
