# Plan P2 — mistral-small partout, vérification post-génération, réponses concises

_2026-07-20 · branche `fix/retriever`. Décisions actées par l'utilisateur :
100 % `mistral-small-latest` en génération (droit réservé de repasser sur
`medium` pour un futur chemin reasoning), fix vision sur small, étape de
vérification anti-erreur, réponses plus courtes, température 0.2 confirmée._

---

## 0. Réponse à la question chunks L2

**Oui, les chunks L2 (`contextual_enrichment`) sont bien donnés au LLM dans le
contexte CAG** — vérifié dans `_load_leaf_records`
([context_packer_service.py:98-104](app/services/context_packer_service.py:98)) :

```python
select(DocumentChunk).where(
    DocumentChunk.document_id == document_id,
    DocumentChunk.is_leaf == True,
)
```

Cette requête charge **tous** les chunks `is_leaf=True` d'un document, **sans
filtrer par `content_type`**. Or les chunks L2 sont créés avec `is_leaf=True`
(`contextual_enrichment_service.py:703`). Donc quand le CAG packe un document,
il inclut L1 (feuilles sémantiques) **et** L2 (synthèses générées par LLM) mêlés
dans le même bloc de texte, dans l'ordre de lecture (page, chunk_index). Le
docstring de la fonction dit « L1 » mais c'est trompeur — le filtre réel est
« is_leaf », qui capture les deux niveaux. Aucun changement nécessaire ici sauf
si tu veux les distinguer visuellement dans le prompt (hors périmètre de ce
plan sauf demande explicite).

---

## 1. Modèle : 100 % mistral-small-latest, option medium réservée

### Constat

- `config.py:58` : `MODEL_FAST` défaut = déjà `mistral-small-latest`.
- `.env` : **`MODEL_FAST=mistral-large-latest`** — c'est le `.env` qui gagne en
  prod, donc la bascule est purement une ligne à changer.
- **Bug bloquant identifié le 20/07** : `is_vision_model()`
  ([rag_generation_service.py:27-36](app/services/rag_generation_service.py:27))
  ne reconnaît pas `"small"` :
  ```python
  return ("pixtral" in name_lower or "vision" in name_lower
          or "large-latest" in name_lower or "ministral" in name_lower
          or "gpt-4o" in name_lower)
  ```
  Basculer `MODEL_FAST` sur small SANS ce fix **coupe silencieusement les PNG
  de pages envoyées au modèle** — régression directe sur la capacité visuelle
  qu'on vient de renforcer (ColPali toujours actif + 12 PNG). C'est le
  correctif n°1, obligatoire avant le changement d'env.

### Décisions

1. **Fix** : ajouter `"small"` à `is_vision_model()`. Small 4 est multimodal
   (confirmé par le commentaire existant `config.py:125` sur
   `PAGE_EXTRACTION_MODEL`).
2. **`.env` + `.env_exemple`** : `MODEL_FAST=mistral-small-latest`.
3. **Droit réservé « medium pour le reasoning »** : je n'implémente PAS de
   bascule automatique de modèle dans ce lot — le plan reasoning-first
   (`plan_impl_reasoning_query_generation_2026-07-17.md`, chantier C2/C3)
   reste la voie prévue pour introduire `reasoning_effort` avec repli
   `mistral-medium-3-5` sur l'appel de rédaction. Ici, je m'assure juste que
   **rien dans ce lot ne verrouille durablement sur small** : `MODEL_FAST`
   reste une simple variable d'env, aucun modèle n'est codé en dur dans le
   nouveau service de vérification (paramétrable, défaut = `MODEL_FAST`).

---

## 2. Étape de vérification post-génération

### Problème concret (preuve du 20/07)

Sur la question « comment définir la hauteur de poignée fixe ? », la réponse
a **inventé 3 références de normes** (NF P20-501, NF P96-104, NF P20-302 —
aucune présente dans le document packé), une **valeur standard fabriquée**
(« 1,00 m »), et **deux calculs arithmétiques en direct** (« 2,15/2 = 1,075 m »,
« 980 mm ») présentés comme des cotes du document. Le prompt système interdit
déjà explicitement ça (« N'invente rien », « Ne combine JAMAIS des
références… ») — la consigne seule ne suffit pas, même à température 0.2.

### Conception retenue

Deux contrôles, complémentaires, exécutés **après** la génération complète
(le texte a déjà streamé — voir §2.3 sur la contrainte UX) :

**(a) Contrôle programmatique — zéro LLM, rapide, déterministe.**
Extrait de la réponse générée :
- références de normes (`NF\s?[A-Z]?\s?\d[\d\-\.]*`, `DTU\s?\d+`, `EN\s?\d+`),
- cotes numériques avec unité (`\d+([.,]\d+)?\s?(mm|cm|m|kg|N)\b`),
- références produit (réutilise `ILLUSTRATION_REFERENCE_PATTERN`, déjà
  corrigé au P0).

Vérifie la présence **littérale** (normalisée espaces/virgules-points) de
chaque occurrence dans le texte du contexte CAG réellement packé
(`space_context_draft["content"]`). Toute occurrence absente → `unsupported_claims`.
Ce contrôle a détecté 100 % des fabrications du cas réel du 20/07 sans appel
LLM.

**(b) Contrôle LLM — pertinence et complétude.**
Un contrôle programmatique ne peut pas juger « est-ce que cette réponse
répond vraiment à la question posée ? » (hors-sujet confiant, réponse
incomplète). Un appel LLM court (mistral-small, température 0.0, sortie JSON
stricte) reçoit : la question, la réponse générée, un extrait du contexte
(tronqué si besoin). Il rend :
```json
{"answers_question": true|false, "grounded": true|false, "issues": ["..."]}
```
Inspiré du pattern déjà existant dans `chat_critique_service.py` (JSON
decision + sanitisation), mais un service **nouveau**
(`response_verification_service.py`) car `chat_critique_service` est
spécifiquement câblé pour la comparaison à des FAQ correctives — périmètre
différent, pas réutilisable tel quel.

### 2.3 Contrainte UX : le texte a déjà streamé

Le pipeline actuel streame les tokens au fil de l'eau (SSE) ; au moment où la
génération complète est disponible (`complete_response`), le texte est déjà
affiché côté client. Une vérification **bloquante avant affichage**
(buffeuriser toute la réponse, vérifier, puis streamer) changerait l'UX
(latence perçue, plus de streaming token-par-token) — **décision hors
périmètre de ce lot**, à ne pas faire à la légère.

**Choix retenu pour ce P2** : vérification **non bloquante**, exécutée juste
avant la persistance (point d'accroche existant : juste avant
`_persist_reply_with_retry`, [chat.py:1919](app/routers/chat.py:1919), au même
endroit que l'extraction d'illustration qui tourne déjà après le texte
complet). Le résultat est :
1. **Toujours loggé** (observabilité, base pour un futur jeu golden
   answer-level « réponse correcte / hors-sujet / hallucinée »).
2. **Persisté dans `Message.metadata_json`** (champ déjà existant, JSON libre)
   sous la clé `verification` — disponible immédiatement via l'API sans
   migration, sans travail frontend obligatoire pour ce lot.
3. Le frontend n'est **pas modifié** dans ce lot — l'affichage d'un badge
   « information non vérifiée » est une itération UI séparée, volontairement
   hors périmètre (je ne invente pas de composant UI sans validation de ta
   part).

C'est un compromis assumé : on n'empêche pas encore l'utilisateur de voir une
réponse fautive en temps réel, mais on la **détecte et la trace
systématiquement** — pré-requis nécessaire avant de décider si la correction
doit être bloquante (streaming différé) ou corrective a posteriori (message
édité). Sujet à trancher avec toi une fois qu'on a des données réelles sur la
fréquence des détections.

---

## 3. Réponses plus courtes

Constat : la réponse « hauteur de poignée » fait ~3600 caractères avec
sections numérotées, sous-listes, schéma ASCII — pour une question qui
appelle une réponse de quelques phrases. Le prompt système dit déjà
« concis » (§1 POLITIQUE DE RÉPONSE, `chat.py:227`) mais ce n'est pas assez
contraignant.

**Changements au prompt système** (`SPACE_CHAT_SYSTEM_PROMPT`,
`chat.py:212-251`) et au prompt fiche technique (`FICHE_SYSTEM_PROMPT`,
`fiche_technique_service.py:186-208`) :
- Ajouter une consigne de longueur explicite : réponse par défaut en
  **quelques phrases ou un court paragraphe** ; les tableaux/listes structurées
  réservés aux cas où les documents fournissent eux-mêmes un tableau de
  données (dimensions, compatibilités) — pas pour reformater une explication
  en plan à 5 sections.
- Interdire explicitement les schémas ASCII improvisés (comme le schéma
  « Sol fini / axe poignée » halluciné dans l'exemple) — un schéma texte n'est
  légitime que s'il retranscrit un schéma RÉELLEMENT présent dans le document,
  jamais une invention pour illustrer un raisonnement.
- Recentrer : répondre à la question posée, pas à toutes les variantes
  possibles de la question (le hors-sujet du « hauteur de poignée » vient en
  partie du modèle qui a voulu être exhaustif — cas standard + PMR + oscillo-
  coulissant + méthode de réglage — alors que rien n'indiquait laquelle de ces
  configurations concernait l'utilisateur).

---

## 4. Température

Déjà à 0.2 depuis le P0 (`.env: SPACE_CHAT_TEMPERATURE=0.2`,
`config.py:72` défaut 0.2, `docker-compose.yaml` à aligner si ce n'est pas
déjà fait). Ce lot ne change pas la valeur — je vérifie juste qu'aucune
régression ne la remonte, et que le nouveau contrôle LLM (§2) tourne à
température 0.0 (jugement, pas de génération créative).

---

## 5. Ce que ce lot NE fait PAS (explicitement hors périmètre)

- Pas de `reasoning_effort` natif (reste le plan C0-C5 du 17/07, non
  commencé).
- Pas de blocage du streaming pour la vérification (voir §2.3).
- Pas de nouveau composant UI pour afficher les alertes de vérification.
- Pas de retry automatique de génération en cas de détection (juste
  détection + trace pour ce lot ; le retry est une décision de produit à
  prendre une fois qu'on a mesuré la fréquence réelle des problèmes détectés).
