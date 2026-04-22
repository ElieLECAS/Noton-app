# Plan — Résumés de pages & agentic RAG (pensée)

Document de suivi pour reprendre le travail plus tard. Synthèse des idées discutées pour améliorer le RAG sur documents peu structurés (ex. marketing visuel), **sans traitement des images** (texte + tableaux uniquement).

---

## 1. Contexte actuel (rappel)

- Pipeline existant : Docling → chunking hiérarchique → embeddings BGE-m3 → pgvector → retrieval hybride (vectoriel + lexical + KAG) → reranking → Mistral.
- Endpoints admin bibliothèque déjà présents (`reindex`, stop/skip, etc.) — modèle à réutiliser pour un bouton « traiter les résumés ».

---

## 2. Résumés de pages (priorité produit)

### 2.1 Objectif

- Enrichir le retrieval sans **tout retraiter** le document : cibler uniquement le contenu par **page** (ou bloc majeur) à partir du texte/tableaux déjà extraits.
- Ne **pas** traiter les images : rester sur le signal texte + tableaux issu de Docling.

### 2.2 Déclenchement

- **Manuel admin** d’abord : bouton sur la tuile document dans la bibliothèque (« Traiter les résumés »), optionnellement action globale « Tout traiter ».
- Éviter de mélanger avec le pipeline d’upload : pas d’exécution automatique au début (contrôle coût + qualité).
- Plus tard : auto-déclenchement possible si confiance sur la détection « document marketing ».

### 2.3 Contenu des résumés (clair, dense, technique)

Structurer chaque entrée par page (ex. JSON logique) :

| Champ | Rôle |
|--------|------|
| `page` / identifiant page | Ancrage |
| `dense_summary` | 5–8 phrases factuelles, sans blabla |
| `key_facts` | Puces atomiques (chiffres, promesses, contraintes, dates) |
| `entities` | Marque, produit, offre, segment, canal, KPI |
| `qa_pairs` | 5–10 paires Q/R courtes (variantes lexicales : prix/tarif, etc.) |
| `keywords` | Mots-clés normalisés |
| `confidence` | (optionnel) score qualité extraction |

### 2.4 Type de chunk : ni `leaf` ni `parent` classique

- **`page_summary`** (et éventuellement **`page_qa`**) : artefacts **dérivés**, pas des feuilles sources.
- Ne pas les confondre avec les leaves (extraits bruts) ni les parents (regroupement hiérarchique parsing).
- Embeddings **dans la même base pgvector** que les chunks existants, avec **discrimination par type** pour pondérer au retrieval.

### 2.5 Embeddings & BDD

- Oui : les résumés (et Q/R) sont **embeddés** et stockés vectoriellement (même « endroit » technique : pgvector), avec métadonnées explicites (`chunk_type`, `document_id`, `page`, version de génération).
- Option schéma : **même table** `documentchunk` + `chunk_type` dédié *ou* table séparée `document_page_summary` si séparation claire souhaitée.

### 2.6 Priorité au résumé — pourquoi

- Signal **plus sémantique** qu’un leaf isolé sur docs marketing.
- **Meilleur rappel** sur questions métier.
- Moins de bruit que des punchlines fragmentées.
- Combiner avec des **leaves** de la même page pour preuves / citations : résumé pour trouver la zone, leaves pour le détail.

### 2.7 Contexte final injecté au LLM (éviter la redondance)

- Le retrieval peut ramener à la fois résumés, parents et leaves ; **dédupliquer** avant prompt.
- Règle simple : par groupe page/section, si `page_summary` fort → **ne pas** aussi injecter le `parent` du même groupe ; garder 1–2 `leaf` max + budget tokens par document.

---

## 3. Améliorations liées (rappel court)

- **Chunking** : unité « page complète » pour certains types marketing (détection de type de document).
- **Métadonnées** : titre document, n° de page, section sur chaque chunk injecté.
- **HyDE (optionnel)** : questions hypothétiques en ingestion pour renforcer l’alignement requête↔contenu.
- **Prompt génération** : grounding strict (« si l’info n’est pas dans les passages, le dire »).

*(Vision / description d’images volontairement **hors scope** ici.)*

---

## 4. Agentic RAG & « pensée » (phase suivante)

### 4.1 Principe

- Passer d’un seul aller-retour retrieval → génération à un **raisonnement en étapes** : reformuler, décomposer, réévaluer, relancer le retrieval.

### 4.2 Idées de patterns à intégrer (alignées avec l’existant `query_reasoning_service`)

| Pattern | Idée |
|---------|------|
| **Boucle réflexion** | Après 1ʳᵉ réponse, évaluer si les passages suffisent ; si non, reformuler la requête et re-retrieve (max N tours). |
| **Step-back** | Récupérer une requête plus générale + requête originale, fusionner les résultats (RRF / union). |
| **Query decomposition** | Sous-questions + retrieval par sous-question + agrégation. |
| **Contextual compression** | Avant prompt, compacter chaque passage pour ne garder que l’utile (modèle rapide). |
| **Chain-of-thought contrôlé** | Scratchpad interne (liste faits des passages → synthèse) pour réduire les erreurs de raisonnement. |

### 4.3 Garde-fous

- Plafonner les tours (latence, coût).
- Tracer les étapes (observabilité : déjà un socle `tracing` / LangSmith côté config).
- Conserver le même socle KAG + hybride : l’agentic RAG s’**ajoute** au-dessus, ne remplace pas tout.

### 4.4 Ordre de mise en œuvre suggéré

1. Résumés de page + type `page_summary` / `page_qa` + déduplication contexte.
2. Step-back ou double requête (moins lourd qu’une boucle complète).
3. Boucle réflexion / décomposition sur les espaces ou types de questions les plus difficiles.

### 4.5 Positionnement « robustesse niveau CAG »

- Ce plan rapproche déjà le système d'un niveau de robustesse **CAG-like** même avant une boucle agentique complète.
- Le gain immédiat vient du trio : `page_summary`/`page_qa` + priorisation retrieval + déduplication du contexte final.
- Cela stabilise le comportement sur les documents difficiles (marketing peu structurés) et réduit les hallucinations dues à des leaves trop fragmentés.
- Le vrai palier « CAG mature » sera atteint en ajoutant une boucle de réflexion courte (2-3 tours max) + décomposition des requêtes complexes + garde-fous de confiance.

### 4.6 Boucle de réflexion agentique — design détaillé

#### Objectif opérationnel

- Ajouter une étape de **vérification de suffisance des preuves** avant de finaliser la réponse.
- Relancer un retrieval ciblé uniquement quand le contexte est insuffisant ou contradictoire.
- Garder un coût/latence maîtrisés (2-3 tours max).

#### Pipeline cible (vue haut niveau)

1. `Plan` : classer la requête (simple vs complexe) et décider la stratégie (`single`, `step_back`, `decompose`).
2. `Retrieve` : lancer retrieval hybride/KAG (avec priorisation `page_summary`/`page_qa`).
3. `Critique` : évaluer automatiquement la qualité des passages (couverture, cohérence, ambiguïtés, conflit de sources).
4. `Refine` : si qualité insuffisante, reformuler la requête (ou sous-requêtes) puis re-lancer `Retrieve`.
5. `Answer` : générer la réponse finale avec citations, en indiquant explicitement les limites d'information.

#### État de boucle (state machine)

- `iteration`: entier (départ 1, max `MAX_AGENTIC_TURNS`).
- `query_current`: requête active à ce tour.
- `strategy`: `single | step_back | decompose`.
- `retrieval_snapshot`: passages, scores, coverage par document/page, types de chunk présents.
- `evidence_grade`: `strong | medium | weak`.
- `stop_reason`: `enough_evidence | max_turns | no_new_evidence | error`.

#### Critères d'arrêt recommandés

- Stop immédiat si `evidence_grade = strong`.
- Stop si gain marginal faible entre deux tours (`delta_recall < seuil`).
- Stop si aucun nouveau passage utile au tour N.
- Hard stop à 2 tours (3 max uniquement pour questions complexes/multi-étapes).

#### Heuristique de suffisance (rubrique Critique)

Score composite `evidence_score` (0-1), par exemple :

- `coverage_score` : la question est couverte par au moins 2 passages pertinents.
- `specificity_score` : présence de faits concrets (chiffres, références, contraintes).
- `consistency_score` : absence de contradiction majeure entre passages.
- `grounding_score` : possibilité de citer des extraits précis.

Règle simple :

- `>= 0.75` : suffisant → répondre.
- `0.45 - 0.75` : 1 tour de raffinement.
- `< 0.45` : reformulation forte / décomposition obligatoire.

#### Stratégies de raffinement de requête

- `step_back`: générer une version plus générale + garder la requête originale, puis fusion RRF.
- `disambiguation`: lever les ambiguïtés explicites (gamme/version/fournisseur).
- `decompose`: découper en sous-questions indépendantes, retrieval par sous-question, puis agrégation.
- `keyword_anchor`: injecter des entités/faits extraits du tour précédent pour guider la 2e recherche.

#### Contrat de sortie intermédiaire (JSON interne conseillé)

Chaque tour produit une structure normalisée, par exemple :

- `need_more_retrieval`: bool
- `confidence`: 0-1
- `missing_points`: liste courte des informations manquantes
- `next_query`: requête reformulée (si nécessaire)
- `selected_evidence_ids`: ids des passages jugés les plus solides

Cela facilite le debug et l'observabilité.

#### Règles de génération finale (Answer)

- Utiliser prioritairement les passages `page_summary`/`page_qa`, puis compléter avec `leaf`.
- Ne jamais répondre sans au moins 1-2 preuves citables.
- Si preuves insuffisantes : répondre explicitement « la documentation fournie ne permet pas de conclure ».
- Préserver la concision métier (3-6 lignes par défaut) + citations.

#### Observabilité et tracing (indispensable)

- Logger par tour : stratégie, requête, nb passages, score de suffisance, raison d'arrêt.
- Tracer le coût : tokens in/out, latence par tour, latence totale.
- Ajouter des tags de run (`agentic`, `turn_1`, `turn_2`, `step_back`, `decompose`) pour filtrer dans les traces.

#### Garde-fous produit

- Feature flag : `AGENTIC_RAG_ENABLED` (off par défaut en prod initiale).
- Scope progressif : activer d'abord sur `space_chat` uniquement.
- Circuit breaker : fallback immédiat au pipeline RAG actuel si erreur dans un tour.
- Limites strictes : timeout global et plafonds tokens pour éviter les dérives de coût.

#### Plan de rollout conseillé

1. **V1** : `step_back` + 1 tour max de retry (faible risque, fort gain).
2. **V2** : ajout du module `Critique` et score de suffisance.
3. **V3** : décomposition multi-sous-questions pour requêtes complexes.
4. **V4** : optimisation fine (seuils, pondérations, politiques de stop).

#### KPIs de succès (avant/après)

- Taux de réponses « non prouvées » (doit baisser).
- Taux de corrections manuelles utilisateur (doit baisser).
- Precision@k perçue sur corpus marketing.
- Latence p95 de réponse (ne pas dépasser budget UX).
- Coût moyen par requête (surveiller augmentation acceptable).

---

## 5. Checklist de reprise (à cocher)

- [ ] Spécification API admin : lancer génération résumés (document + option globale).
- [ ] Schéma BDD : `chunk_type` + champs page / version.
- [ ] Worker async (Celery) : lecture texte+tableaux par page, pas reparse lourd inutile.
- [ ] Embeddings + index pgvector cohérents avec dimension actuelle.
- [ ] `space_search_service` : pondération + dédup résumé vs parent.
- [ ] UI bibliothèque : bouton admin + indicateur d’état (`summary_status`).
- [ ] (Phase 2) Agentic : step-back → boucle → décomposition.
- [ ] Tests : régression RAG + cas document marketing pauvre en texte linéaire.

---

*Dernière mise à jour : avril 2026.*
