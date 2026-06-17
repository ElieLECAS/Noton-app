# Slot filling Espace — Menuiserie

Documentation du pré-contrôle de requête avant retrieval RAG dans le chat Espace.

## Objectif

Réduire le bruit sémantique et les réponses hors-sujet en imposant des critères métier minimaux avant de lancer la recherche documentaire.

## Flux

```
Message utilisateur
    → Routing direct / RAG
    → Extraction slots (heuristique + LLM optionnel)
    → Merge avec slots conversation précédents
    → Validation obligatoire / facultatif
    → Si incomplet : need_clarification (pas de RAG)
    → Si complet : canonical_query → retrieval → génération
```

## Schéma de slots

| Slot | Valeurs | Obligatoire |
|------|---------|-------------|
| `intent_type` | `diagnostic_probleme`, `recherche_reference`, `norme_procedure`, `comparatif_produits` | implicite |
| `type` | `fenetre`, `porte`, `coulissant` | toujours (sauf routing direct) |
| `galandage` | `oui`, `non`, `inconnu` | si `type=coulissant` |
| `material` | `pvc`, `alu`, `mixte`, `bois`, `inconnu` | selon intention (voir ci-dessous) |
| `problem_symptom` | texte libre | si `diagnostic_probleme` |
| `range_or_model` | texte libre | si `recherche_reference` (ou `supplier_brand`) |
| `supplier_brand` | texte libre | alternative à `range_or_model` |
| `context_usage` | `applique_exterieure`, `applique_interieure`, `monomur`, `tableau`, `ite`, `renovation` | si question de pose |
| `opening_type`, `component_part` | facultatif | enrichissement retrieval |

## Règles par intention

### `diagnostic_probleme`
Obligatoire : `type`, `material`, `problem_symptom`  
+ `galandage` si `type=coulissant`

### `recherche_reference`
Obligatoire : `type`, `material`, et (`range_or_model` ou `supplier_brand`)  
+ `galandage` si `type=coulissant`

### `norme_procedure`
Obligatoire : `type`, `material`  
+ `context_usage` si signaux de pose (membrane, applique, monomur, calfeutrement…)  
+ `galandage` si `type=coulissant`  
Exception ferrure : `type=porte` + gamme Roto/Eneo → `material` non bloquant

### `comparatif_produits`
Obligatoire : `type`, `material` (sauf comparatif explicite PVC vs alu dans le message)

## Requête canonique

Les slots validés sont transformés en `canonical_query` (ex. `fenêtre pvc Perform 70 diagnostic ferme mal côté poignée`).

La recherche utilise l'historique utilisateur complet :

```
retrieval_query = build_retrieval_query_from_conversation(
    canonical_query, conversation_context, message_courant, slot_state
)
```

Les slots `context_usage` / `component_part` enrichissent la requête via un suffixe retrieval générique (`schéma coupe pose calfeutrement`).

## Mémoire conversationnelle

Lors d’une clarification, l’état slots est persisté dans `Message.sources` :

```json
{
  "need_clarification": true,
  "slot_state": { "type": "fenetre", ... },
  "missing_required_slots": ["material"],
  "suggested_options": { "material": ["pvc", "alu", "mixte", "bois"] }
}
```

Au tour suivant, les nouveaux slots sont fusionnés avec cet état.

## Événement SSE `need_clarification`

```json
{
  "need_clarification": true,
  "missing_required_slots": ["material"],
  "suggested_options": { "material": ["pvc", "alu", "mixte", "bois"] },
  "clarification_questions": ["Quel est le matériau ? (PVC / alu / mixte / bois)"],
  "slot_state": { ... }
}
```

## Fichiers

- [`app/catalog/slot_config.py`](../app/catalog/slot_config.py) — taxonomie slots, validation, détection et enrichment retrieval
- [`app/services/slot_filling_service.py`](../app/services/slot_filling_service.py) — extraction, validation, canonical query
- [`app/routers/chat.py`](../app/routers/chat.py) — intégration Pass 0.5 avant retrieval
- [`tests/test_slot_filling_service.py`](../tests/test_slot_filling_service.py) — tests unitaires et intégration

## Exemples

| Entrée | Résultat |
|--------|----------|
| `J'ai un problème sur ma fenêtre qui ferme pas` | Clarification (matériau manquant) |
| `Mon coulissant ferme mal` | Clarification (galandage + matériau) |
| `retombée membrane monomur applique extérieure` | Clarification (type, matériau, contexte pose) |
| `Fenêtre PVC, gamme Perform 70, ferme mal côté poignée` | Retrieval lancé |
