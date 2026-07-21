# Refonte extraction & traitement KAG — pipeline-native + UI fluides

> Date : 2026-07-21 · Branche : fix/retriever
> Principe acté : **AUCUN script one-shot.** Toute la logique vit dans le pipeline de traitement.
> Retraiter un document (mode « KAG seul ») produit l'état propre ; retraiter l'espace fait converger tout le graphe.

## ✅ RÉALISÉ — Volet 1 (R1→R6) livré et testé le 21/07

Tous les changements vivent dans le pipeline `extract_kag_for_document` / `_persist_page_kag` / `_upsert_entity`.
Migration `add_ref_code_to_knowledge_entity` appliquée (colonne `ref_code` + index). Validé en 2 temps :
test déterministe sans LLM (R1→R5) + **vrai retraitement KAG_ONLY doc 406** (0/18 → 246 liens, 9/18 orphelins,
162 entités en codes canoniques, **0 co_occurs**, 1402 liens lexicaux).

- **R1** identité par code : `_resolve_ref_code` (LLM `code`/`code_kind` déclarés + fallback regex), lookup
  collision-safe (code → nom-code → nom-brut) dans `_upsert_entity`, nom canonique = code nu, RAL disjoints (`RAL:7016`).
- **R2** linking lexical intégré (`_lexical_link_codes`, appelé avant chaque commit) : relie chaque code à
  TOUS les chunks du space le contenant (word-boundary), RAL numériques nus (9016…) exigent le contexte « RAL ».
- **R3** `chunk_indexes` par entité → liens `subject` ; fallback `mention` page-globale si absent.
- **R4** le pipeline n'écrit plus `co_occurs` ; endpoints de relation reliés à l'ancre de page seulement.
- **R5** `prune_kag_entities_after_chunk_removal` basé sur la VRAIE condition d'orphelin (plus aucun lien chunk),
  fini la sur-décrémentation de `mention_count` (bug préexistant aggravé par R2).
- **R6** endpoint `POST /api/spaces/{id}/kag/reindex` + bouton « Retraiter KAG » dans la modale Fiches produit
  (docs 0 % d'abord). Mode `kag_only` par-document déjà dans la modale de la bibliothèque.

### 🔑 Cause racine découverte : loader L1 trop restrictif

Les ~17 docs à 0 couverture KAG (space 28) ont des chunks `content_type = "page_raw_enriched"`, PAS
`"semantic_leaf"`. Le loader `_load_l1_chunks_by_page` ne prenait QUE `semantic_leaf` → leur extraction KAG
ne voyait aucun chunk → 0 entité. **Corrigé** : le loader accepte désormais `semantic_leaf` ET
`page_raw_enriched` (exclut `contextual_enrichment`, dérivé). Aucun doc n'a les deux types (pas de doublon).

### Reste à faire (Volets 2-3)
G1 chunk pinning · G2 retrieval exact/pondéré · U1 fiches légères · U2 graphe fluide.
Et : cliquer « Retraiter KAG (espace) » sur space 28 pour converger les 16 docs 0 % restants.

---

## Philosophie

Le graphe KAG n'est **pas une source de vérité**, c'est un **index qui dit où la vérité est écrite**.
- La valeur exacte (« 6111 = 155 mm ») reste dans le texte des chunks ; le graphe route vers le bon chunk.
- On n'affirme jamais un fait depuis le graphe : on **cite le chunk source verbatim**.
- Abandonné : dimensions→attributs du nœud (ré-extraction lossy d'une info déjà parfaite dans le texte).

## Mécanisme de convergence (pourquoi zéro script suffit)

`cleanup_kag_for_document` (appelé avant tout retraitement KAG) supprime les liens du doc **et**
`prune_kag_entities_after_chunk_removal` purge les entités qui n'ont plus aucun lien.
→ Si le pipeline devient propre : retraiter un doc = ses vieux doublons perdent leurs liens → GC ;
les nouvelles extractions se résolvent sur les nœuds canoniques. Retraiter tous les docs = graphe 100 % propre.
Les doublons multi-docs survivent seulement tant que TOUS leurs docs n'ont pas été retraités — d'où R6.

## État des lieux (21/07)

87 % de chunks orphelins (4 173/4 784) · ~25 docs à 0 % de couverture · 6111 éclaté en 5 nœuds ·
codes jusqu'à 17 variantes (7016 = RAL mélangé aux refs) · compatible_avec : 9 · co_occurs : 237 (bruit).

---

# Volet 1 — Refonte du pipeline d'extraction/traitement

## R1 — Résolution d'identité par code, à l'upsert

Dans `_upsert_entity` (le cœur de la refonte) :

1. **Le LLM déclare le code** : le prompt batch demande par entité deux champs de plus :
   `code` (la référence exacte si l'entité en porte une, sinon null) et
   `code_kind` ∈ `ref_produit | couleur_ral | norme | aucun`.
   Le modèle voit la page : il SAIT si « 7016 » est un RAL ou une ref profil. La regex
   `_extract_ref_code` devient le **fallback** quand le LLM n'a rien déclaré.
2. **Nouvelle clé d'identité** : colonne `ref_code` (nullable, indexée, migration idempotente
   — règle create_all au startup). Lookup dans cet ordre :
   a. `(space_id, ref_code)` si code présent → nœud canonique ;
   b. sinon `(space_id, name_normalized)` (comportement actuel, **sans** entity_type dans la clé).
   Le lookup (b) sert aussi d'**adoption** : si une vieille entité « 6111 » (ref_code NULL)
   matche par nom, on la réutilise et on lui tamponne son ref_code.
3. **Nom canonique** : pour une entité à code, `name` = le code nu (« 6111 ») ;
   les formes descriptives (« Profil 6111 », « Dormant 6111 ») deviennent des **alias** automatiques.
4. **RAL jamais mergé avec une ref** : `code_kind=couleur_ral` → ref_code préfixé (`RAL:7016`),
   entity_type `couleur` (nouveau type UI). Identité disjointe des refs produit par construction.
5. mention_count/confidence/description : agrégation inchangée.

## R2 — Linking lexical intégré au traitement (les 2 directions)

Nouvelle étape du pipeline, après le persist des entités d'un document :

- **(a) doc → entités du space** : pour chaque chunk du doc, lier toutes les entités du space
  dont le `ref_code` apparaît en word-boundary (`~ '\mCODE\M'`) dans le contenu.
  → retraiter un doc guérit TOUS ses chunks, y compris ceux que le LLM n'a pas revus (ex-47744).
- **(b) nouvelle entité → corpus** : à la création d'une entité à code, une requête unique lie
  tous les chunks existants du space contenant ce code.
  → les vieux docs se font guérir dès qu'un code apparaît ailleurs.
- Liens `role="mention"`, `relevance_score=0.9`, `context_snippet` ±120 c. Idempotent (skip existants).
- Garde-fous : code purement numérique → ≥ 4 chiffres ; millésimes 1990–2035 exclus ;
  les codes `RAL:` matchent « RAL 7016 » mais pas « 7016 » nu.

C'est l'ex-« B1 » mais **vivant dans le pipeline** : chaque traitement l'exécute, pas de rattrapage externe.

## R3 — Attribution par chunk dans le prompt

Les entités gagnent `chunk_indexes` (même mécanique que `chunk_categories`, déjà en place) :
- chunks listés par le LLM → lien `role="subject"` (le chunk PARLE de l'entité) ;
- autres chunks de la page → plus de lien automatique page-globale (fin du bruit) ;
- la couverture large est assurée par R2 (mention lexicale), la précision par R3 (subject LLM).
C'est ce qui rend le chunk-pinning (G1) fiable : « subject » ≫ « mention ».

## R4 — Relations : qualité > quantité

- **Le pipeline n'écrit plus `co_occurs`** : le linking lexical R2 capture mieux la co-présence.
  Les co_occurs existants meurent par GC au fil des retraitements.
- Prompt : relation UNIQUEMENT si le lien est explicite dans le texte/l'image ; sinon rien.
- **Tables de compatibilité** : heuristique « page dense en codes » → le batch reçoit une
  instruction renforcée « cette page est probablement un tableau de compatibilité : extrais les
  paires compatible_avec ». Chaque paire garde `source_chunk_id` = la preuve citée en génération.
- Objectif : compatible_avec 9 → centaines, toutes sourcées ; co_occurs → 0.

## R5 — GC complet au cleanup

Vérifier/compléter `prune_kag_entities_after_chunk_removal` : purge aussi les `entityalias`
et `entityentityrelation` des entités supprimées (sinon lignes fantômes).

## R6 — Bouton « Retraiter KAG (espace) »

Feature produit (pas script) : dans l'UI espace, un bouton qui met en file le mode
« KAG seul » pour tous les documents de l'espace (worker existant, séquentiel, progression visible).
C'est LE geste de convergence : un clic → tout l'espace repasse dans le pipeline propre.
Priorité d'affichage : les ~25 docs à 0 % de couverture d'abord.

---

# Volet 2 — Retrieval & génération (exploitent le pipeline propre)

## G1 — Chunk pinning à la génération (ex-C2)

- La question contient un code exact (regex) ou matche une entité product/reference ≥ 0.75
  → `select_authority_chunks(space, ref_code, limit=2)` :
  chunks `role='subject'` d'abord, puis `mention` ; bonus si code + unité (mm/kg) dans le texte ;
  cap ~1 200 c/chunk.
- Injection AVANT les chunks retriever, dans chat + fiche technique :
  `### EXTRAITS DE RÉFÉRENCE — 6111 (source : DTA …, p.14)` + texte **verbatim**.
- Les relations (`compatible_avec`…) servent à aller chercher le chunk du produit voisin,
  jamais à affirmer le fait.

## G2 — Retrieval graphe intelligent

- **Codes = match exact** dans `_query_entity_candidates` : token-code → égalité stricte sur
  `ref_code`/alias (score 1.0), jamais de trigram (fin de la confusion 6110/6111).
  Trigram conservé pour les tokens non-codes.
- **Traversal pondéré par type** dans `_neighbor_entities` : `eid → poids`
  (est_compose_de/compatible_avec 1.0 · requiert_piece 0.9 · utilise/installe_sur 0.7 ·
  conforme_a 0.6 · mesure/reference 0.5) ; hop_factor 0.65 × poids.

---

# Volet 3 — UI fluides

## U1 — Fiches produit (léger et réactif)

Backend 2 niveaux :
- `GET /kag/references` → liste **compacte** seule (code, variant_count, types, mention_total,
  page_count, relation_count, compatible_count) — < 50 Ko ; cache mémoire par space (TTL 5 min,
  invalidé par indexation) ;
- `GET /kag/references/{code}` → détail lazy (variants, pages, relations) à l'expansion de la carte.

Frontend :
- rendu **incrémental par lots de 50** (IntersectionObserver, sentinelle bas de liste) ;
- recherche **debounce 250 ms**, filtrage local sur la liste compacte ;
- détail mémorisé après premier fetch (pas de re-fetch).

## U2 — Graphe de connaissances (supprimer le lag)

Causes mesurées : fetch bloquant au page-load (`loadKagGraph()` ligne ~3386) ; destroy/recreate
cytoscape + layout cose 1000 itérations ANIMÉ à chaque ouverture de modale ; bézier + labels
outline sans seuil de zoom ; aucun flag perf ; hairball 300 nœuds dont 27 % other/dimension.

1. **Lazy** : plus de fetch au page-load ; bouton affiché via compteur méta léger ;
   fetch complet à la première ouverture.
2. **Instance persistante** : cytoscape créé une fois ; ré-ouverture = show + `cy.resize()` ;
   positions conservées (jamais destroy/recreate, jamais de re-simulation).
3. **Layout** : `numIter 300`, `animate:false`, fit direct ; ensuite positions mémorisées (preset).
4. **Flags perf** : `pixelRatio:1`, `textureOnViewport:true`, `hideEdgesOnViewport:true`,
   `motionBlur:false` ; arêtes `straight` (bézier réservé à la sélection) ;
   labels `min-zoomed-font-size:8` (masqués au dézoom).
5. **Vue par défaut allégée** : 150 nœuds top-mentions ; endpoint avec `include_types`
   (exclut `dimension`/`other` par défaut) ; chips de légende = filtres cliquables.
6. **Exploration progressive** : double-clic nœud → `GET /kag/entity/{id}/neighbors` →
   `cy.add()` incrémental + layout local du voisinage. Le graphe complet n'est jamais chargé.
7. Après R1, le graphe n'affiche que des nœuds canoniques → moins de nœuds, plus lisible.

---

# Ordre d'exécution

```
1. R1 + R2 + R5   refonte upsert/linking/GC        (cœur — un seul PR cohérent)
2. R3 + R4        prompt (chunk_indexes, code/code_kind, relations strictes, stop co_occurs)
3. R6             bouton « Retraiter KAG (espace) »
4. ► RETRAITEMENT de l'espace via R6 → le graphe converge (mesurer avant/après)
5. G2 puis G1     retrieval exact + chunk pinning   (exploitent le graphe propre)
6. U1 + U2        UI fiches + graphe                (indépendants, parallélisables à tout moment)
```

# Métriques de succès

- Orphelins : 87 % → < 15 % après retraitement espace complet.
- 1 nœud par code ; RAL séparés (`RAL:7016` ≠ ref 7016) ; co_occurs → 0.
- Golden space 28 : canal KAG ≥ neutre en doc-recall ; « longueur 6111 » = 155 mm
  en reasoning none ET high (grâce au pinning verbatim).
- Modale graphe : ré-ouverture instantanée (< 100 ms), zéro re-simulation.
- Fiches : liste < 200 ms perçu ; frappe sans jank ; détail < 300 ms par carte.
