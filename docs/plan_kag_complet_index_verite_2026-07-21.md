# Plan complet KAG — « Le graphe pointe, le chunk affirme » + UI fluides

> Date : 2026-07-21 · Branche : fix/retriever
> Remplace les volets KAG des plans précédents (plan_kag_graphe_produit, plan_reference_exacte_retrieval).

## Philosophie (décision actée)

Le graphe KAG n'est **pas une source de vérité**, c'est un **index qui dit où la vérité est écrite**.
- La valeur exacte (« 6111 = 155 mm », compatibilités, normes) reste dans le **texte des chunks**.
- Le graphe sert à **retrouver le bon chunk** et ses voisins pertinents.
- On n'extrait plus de « faits » du graphe pour les affirmer : on **cite le chunk source verbatim**.
- Conséquence : l'ancienne idée « dimensions → attributs du nœud » (ex-C3) est **abandonnée**.

## État des lieux (mesuré le 21/07)

| Métrique | Valeur |
|---|---|
| Entités | 2 140 (545 product, 446 reference, 323 other, 256 dimension…) |
| Relations entité↔entité | 1 144, dont co_occurs 237 (bruit) et compatible_avec **9** |
| Liens chunk↔entité | 18 611 |
| **Chunks orphelins** | **4 173 / 4 784 = 87 %** |
| Docs à 0 % couverture | ~25 (Roto NX 671 chunks, TROCAL, DTA, catalogues SOLEAL/LUMEAL) |
| Produit 6111 | 5 nœuds doublons (`6111`, `Dormant 6111`, `Profil 6111`, `Profilé 6111`, `Référence 6111`) |
| Codes multi-entités | jusqu'à 17 variantes (7016 = RAL couleur mélangé avec refs) |

Bug emblématique : le chunk 47744 (« 6111 : longueur totale de 155 mm ») a **zéro lien** → le graphe route vers des pages où co_occurs contamine (6110 → 135 mm).

---

## Phase 1 — B1 : Linking lexical universel (fondation)

**Objectif** : toute entité à code est reliée à **tous** les chunks du corpus contenant ce code. Répare les 87 % d'orphelins pour les entités à code, sans LLM.

- Script `app/scripts/kag_lexical_linking.py` avec `--dry-run` (défaut) / `--apply`.
- Pour chaque entité dont `_extract_ref_code(name)` ou un alias donne un code :
  - match **word-boundary** dans `documentchunk.content` du même space (`~ '\mCODE\M'` Postgres) ;
  - garde-fous : code ≥ 4 caractères si purement numérique (éviter « 155 »), écarter millésimes 1990–2035 (déjà dans la regex) ;
  - INSERT `ChunkEntityRelation(role="mention", relevance_score=0.9, context_snippet=±120 chars autour du match)` ;
  - idempotent (skip liens existants), commit par lots de 500.
- Sortie dry-run : liens à créer par doc, top codes, nouveau % d'orphelins projeté.
- **Métrique de succès** : orphelins < 30 % ; chunk 47744 relié au 6111.

## Phase 2 — A : Résolution d'identité par code

**Objectif** : un produit = un nœud. Les doublons deviennent des alias du nœud canonique.

1. **Migration** : colonne `ref_code` (nullable, indexée) sur `knowledgeentity` — idempotente (cf. règle create_all au startup). Backfill via `_extract_ref_code`.
2. **Groupes de merge** : réutiliser `build_kag_reference_index` (les groupes par code existent déjà).
3. **Tri automatique vs arbitrage LLM** :
   - merge **auto** : même code, types ⊂ {product, reference}, pas de motif RAL ;
   - **arbitrage LLM 256K** (mistral-small, UN appel) pour les ambigus : groupes + descriptions + snippets → verdict JSON `{code, merge: bool, sous_type: ref_profil|couleur_ral|norme|quincaillerie|autre}`. Les RAL (7016, 9016, 9005, 8019…) sont typés `couleur_ral` et **jamais** mergés avec des refs profil.
4. **Mécanique de merge** (script `--dry-run`/`--apply`) : survivant = max mentions ; re-pointer `entityalias`, `chunkentityrelation` (dédup sur (chunk, entity, role)), `entityentityrelation` (fusion des poids si la paire existe déjà) ; noms des perdants → alias ; suppression des perdants ; `mention_count` sommé.
- **Métrique** : 1 nœud par code produit (≈ 324 codes) ; zéro perte de lien (somme des liens conservée à dédup près).

## Phase 3 — C2 : Chunk pinning à la génération

**Objectif** : quand une question cible un produit, injecter en tête de contexte le(s) chunk(s) **faisant autorité**, cités verbatim avec leur source.

- `select_authority_chunks(session, space_id, ref_code, limit=2)` :
  1. chunks liés au nœud du code avec `relation_role='subject'` d'abord, puis `mention` ;
  2. bonus si le chunk contient le code **et** une unité (mm, cm, kg…) — densité « spec » ;
  3. tri par `relevance_score`, cap ~1 200 chars/chunk.
- Gating : la question contient un code exact (regex) **ou** une entité product/reference matche ≥ 0.75.
- Injection dans l'assemblage du contexte (chat + fiche technique) :
  ```
  ### EXTRAITS DE RÉFÉRENCE — 6111 (source : DTA 6/16-2335_V5, p.14)
  « …texte verbatim du chunk… »
  ```
  placé AVANT les chunks retriever ; budget ≤ 2 chunks épinglés.
- Les relations (`compatible_avec`…) servent à **aller chercher le chunk du voisin**, jamais à affirmer — le fait est toujours cité depuis son chunk.
- **Métrique** : golden Q « longueur 6111 » → 155 mm systématique, y compris reasoning=none.

## Phase 4 — Retrieval graphe intelligent

1. **Codes = match exact** dans `_query_entity_candidates` : les tokens-codes (regex) matchent `ref_code`/alias en **égalité stricte** (score 1.0, source `code_exact`), jamais en trigram → fin des confusions 6110/6111. Trigram conservé pour les tokens non-codes.
2. **Traversal pondéré par type** dans `_neighbor_entities` : retourner `eid → poids` avec
   `est_compose_de/compatible_avec 1.0 · requiert_piece 0.9 · utilise/installe_sur 0.7 · conforme_a 0.6 · mesure/reference 0.5 · co_occurs 0.15` ; hop_factor 0.65 × poids ; LIMIT par type plutôt que 100 global.
- **Métrique** : golden set — le canal KAG passe de net-négatif à neutre/positif sur doc-recall.

## Phase 5 — B4 : Moisson `compatible_avec` corpus-level

- Détecter les pages « tables de compatibilité » (heuristique : densité de codes ≥ seuil/page).
- Batch 256K (plusieurs pages entières + images) avec prompt dédié → paires `compatible_avec` **avec `source_chunk_id`** (le chunk de la table = la preuve citée par C2).
- Objectif : 9 → plusieurs centaines de paires, toutes sourcées.

---

## Phase 6 — UI Fiches produit : fluide et légère

Problèmes actuels : `limit=600` cartes en un seul `innerHTML` ; re-render complet à chaque frappe ; backend qui recharge toutes les entités + 2 gros JOINs à chaque appel.

1. **Backend en 2 niveaux** :
   - `GET /kag/references` → **liste compacte** uniquement (`code, variant_count, types, mention_total, page_count, relation_count, compatible_count`) — sans variants/pages/relations (~60 o/fiche, réponse < 50 Ko) ;
   - `GET /kag/references/{code}` → **détail lazy** (variants, pages, relations) chargé à l'expansion d'une carte ;
   - cache serveur en mémoire par space (TTL 5 min, invalidé par indexation/merge).
2. **Frontend** :
   - rendu **incrémental par lots de 50** via IntersectionObserver (sentinelle en bas de liste) ;
   - recherche **debounce 250 ms**, filtrage local sur la liste compacte (tout est déjà côté client), re-render du seul conteneur ;
   - expansion de carte → fetch détail + squelette de chargement ; détail mémorisé (pas de re-fetch).

## Phase 7 — UI Graphe de connaissances : supprimer le lag

Causes identifiées : fetch bloquant au page-load ; destroy/recreate cytoscape + layout cose 1000 itérations **animé** à chaque ouverture ; bézier + labels outline sans seuil de zoom ; aucun flag perf ; hairball de 300 nœuds dont 27 % `other`/`dimension`.

1. **Lazy load** : supprimer `loadKagGraph()` du page-load ; le bouton s'affiche via un compteur déjà connu (ou endpoint méta ultraléger) ; fetch complet à la **première ouverture** seulement.
2. **Instance persistante** : créer cytoscape une fois ; ré-ouverture = `show + cy.resize()` (jamais destroy/recreate) ; positions conservées.
3. **Layout** : `numIter 1000 → 300`, `animate:false` (fit direct après calcul) ; positions mémorisées ensuite (`preset`).
4. **Flags perf cytoscape** : `pixelRatio:1`, `textureOnViewport:true`, `hideEdgesOnViewport:true`, `motionBlur:false` ; arêtes `curve-style:"straight"` (bézier seulement sur sélection) ; labels avec `min-zoomed-font-size:8` (masqués au dézoom = gros gain).
5. **Vue par défaut allégée** : `max_nodes 300 → 150` top-mentions ; exclure `dimension` et `other` par défaut côté endpoint (`include_types` paramétrable) ; chips de la légende = **filtres cliquables** par type.
6. **Exploration progressive** : double-clic sur un nœud → `GET /kag/entity/{id}/neighbors` → `cy.add()` incrémental + petit layout local sur le voisinage. Le graphe complet n'est jamais chargé d'un coup.
7. Bonus cohérence : après Phase 2, le graphe affiche les **nœuds canoniques** (1 par code) → moins de nœuds, plus lisible, plus rapide.

---

## Ordre d'exécution & dépendances

```
P1 B1 lexical (SQL, dry-run→apply)        ← fondation, tout en dépend
P2 A  résolution par code (+ ref_code)    ← dépend de rien, mieux après P1 pour stats
P7 UI graphe (indépendant, quick win perf)
P6 UI fiches (indépendant, quick win perf)
P4 retrieval intelligent                   ← dépend de P2 (ref_code)
P3 C2 chunk pinning                        ← dépend de P1 + P2
P5 moisson compatible_avec                 ← dépend de P2 ; alimente P3
+ re-lancer « KAG seul » sur les ~25 docs à 0 % (UI prête)
```

## Métriques de succès globales

- Orphelins : 87 % → < 30 % (P1) puis < 15 % (après KAG seul sur docs 0 %).
- 1 nœud par code ; RAL séparés des refs.
- Golden set space 28 : doc-recall canal KAG ≥ neutre ; « longueur 6111 » = 155 mm en reasoning none et high.
- Modale graphe : ouverture < 300 ms après premier rendu ; aucune re-simulation à la ré-ouverture.
- Fiches : première liste < 200 ms perçu ; frappe recherche sans jank.
