# Plan — assainir et enrichir le graphe produit (KAG)

_2026-07-21 · branche `fix/retriever`. Analyse conjointe données (espace 28) +
code d'extraction. Objectif : un graphe entités+relations fiable pour répondre
aux questions métier réelles (« la 4 points de TGY3702 ? », « seuils compatibles
6101 ? », « montants compatibles 6127 ? »)._

---

## 1. État réel (mesuré)

**Entités : 2004, massivement dupliquées, de façon structurée.**
- **6115** → 6 entités (`6115`, `Profil 6115`, `Profilé 6115`, `Référence 6115`,
  `Élément 6115`, `Ouvrant 6115`). **6127** → 8. **7016** (couleur) → 17.
- Pattern : le nom canonique inclut le **préfixe descriptif** → `name_normalized`
  différent → entité distincte.

**Relations : 1023, faibles là où le métier en a besoin.**
```
co_occurs 222 (bruit)   est_compose_de 229   utilise 140 …   compatible_avec 9
```
- **`compatible_avec` = 9 dans tout l'espace.** Les tables de compatibilité
  (seuils/montants/poignées — le cœur des questions SAV) ne sont PAS modélisées.
- La fragmentation éparpille les relations sur les doublons : « Traverse 6127
  co_occurs Profilé 6127 » = une relation entre deux doublons du MÊME 6127.

**Conclusion : les questions « réf → réf liée » (classe 2, les plus fréquentes)
sont non répondables depuis ce graphe.**

---

## 2. Causes racines (confirmées dans le code)

| # | Cause | Preuve (code) |
|---|---|---|
| C1 | **Aucune canonicalisation par code.** Le canonique = ce que le LLM a dit. | `normalize_and_expand_entity:261-284` ; gazetteer `:246-258` ne couvre que matériaux/gammes, **rien pour les références** |
| C2 | **`entity_type` dans la clé unique.** `6111` en `product` ET `reference` = 2 entités. | `uq_knowledgeentity_space_name_type (space_id, name_normalized, entity_type)` `knowledge_entity.py:31-39` |
| C3 | **Entités fantômes issues des relations.** Une relation nommant une entité non listée crée un `_upsert_entity(type="other")`. | `kag_extraction_service.py:1042-1055` |
| C4 | **`compatible_avec` sous-extrait, `co_occurs` domine.** Le prompt le suggère mais le LLM produit du co-occurrence ; les TABLES ne sont pas parsées en relations. | prompt `:190-196` ; données (9 vs 222) |
| C5 | **Linking 100% piloté LLM, granularité page, sans secours lexical.** Une entité n'est liée qu'aux chunks de la page **où le LLM l'a listée**. | `_link_entity_to_chunks:921-952` ; c'est pourquoi `6111` n'est PAS lié au chunk 47744 (fait « 155 mm ») |
| C6 | **Direction des relations perdue** (`sorted(a_id,b_id)`). | `_upsert_entity_relation:970` |
| C7 | **Aucune passe de dédup/merge offline.** `scripts/` vide ; seul du GC destructif existe. | `prune_kag_entities_after_chunk_removal:1426` |

Principe unificateur du plan : **une référence produit a pour identité son CODE**
(6111, TGY3702, 9F67). Tout le descriptif (« Profil », « Traverse », « Dormant »)
est un **rôle/alias**, pas une entité. Et la **compatibilité est une relation de
1ʳᵉ classe**, récoltée des tables.

---

## 3. Plan — deux temps

### B) 2ᵉ PASSE OFFLINE (sur l'existant, rejouable, SANS re-ingérer) — à faire EN PREMIER

Le plus rentable : les IDs de chunks sont stables → on rejoue sur les données
actuelles, gain immédiat, pas de coût LLM d'ingestion.

**B1 — Merge des entités par code (le cœur).**
- Extraire le code de chaque entité (`\b([A-Z]{0,4}\d{3,6}[A-Za-z]?)\b`), pour
  les types `product`/`reference`/`other` porteurs d'un code.
- Regrouper par code → 1 entité canonique (nom = code nu), les variantes
  descriptives deviennent des **alias**.
- Réaffecter `chunkentityrelation`, `entityentityrelation`, `entityalias` vers le
  canonique ; re-sommer `mention_count` ; supprimer les doublons.
- Réutiliser la logique destructive de `prune_kag_entities_after_chunk_removal`
  comme base de nettoyage post-merge.
- Nuance : couleurs/matériaux (7016 granité vs satiné) = merge par **code+finition**,
  ou laissés (moins critique que les références produit).
- Effet attendu : ~2004 → probablement &lt;1000 entités, chaque référence
  consolidée avec TOUTES ses relations et TOUS ses liens chunks.

**B2 — Récolte `compatible_avec` depuis les TABLES (la vraie valeur métier).**
- Les chunks-tables (« 6101 | 9F67 | 9F65 ou 9F71 », « 6100 | M/S/SP | M | M »)
  sont propres et structurés. Passe de parsing table-aware → relations typées
  vérifiées : `6101 compatible_avec 9F65`, `6101 assemblage 9F67`, etc.
- Ces relations pointent vers les entités canoniques (post-B1).
- Purge/reclasse les `co_occurs` (bruit) — au minimum ne pas les utiliser au
  retrieval.

**B3 — Reliaison lexicale entité↔chunk.**
- Pour chaque entité canonique (code), lier à TOUS les chunks contenant
  littéralement le code (`\mCODE\M`). Corrige C5 : `6111` sera enfin lié au
  chunk 47744. C'est la MÊME infra que le retriever exact P-A du plan
  `plan_reference_exacte_retrieval_2026-07-21.md`.

**B4 — Vérification.** Une relation/attribut n'est gardé que si co-localisé dans
un énoncé où la référence est sujet (gate déjà défini pour le référentiel).

### A) INGESTION (empêcher la re-création des doublons)

**A1 — Canonicalisation par code** dans `normalize_and_expand_entity` : si le nom
contient un code de référence (types product/reference), canonique = le code,
préfixe descriptif → alias. Convergence garantie à l'écriture.

**A2 — Sortir `entity_type` de la clé de dédup** pour les codes (ou fusionner
`product`/`reference` en un type « référence »). Tue la duplication structurelle.

**A3 — Extraction table-aware + prompt** : renforcer `_KAG_BATCH_SYSTEM_PROMPT`
pour privilégier `compatible_avec`/`assemblage` sur `co_occurs`, et brancher le
parsing des tables de compatibilité à l'ingestion (même logique que B2).

**A4 — Linking lexical de secours** après extraction LLM (même scan que B3) :
lier chaque entité aux chunks contenant son code, indépendamment de ce que le LLM
a listé sur la page. Corrige C5 durablement.

**A5 — Préserver la direction** des relations orientées (`remplace`,
`installe_sur`, `symptome_cause`) : ne pas trier a_id/b_id, ou flag `directed`.

---

## 4. Ordre, mesure, garde-fous

1. **B1 + B3** (merge + reliaison) — gain immédiat sur le graphe existant,
   rejouable, mesurable (chute du nb d'entités, `6111` lié à sa page 155).
2. **B2** (compatible_avec depuis les tables) — débloque la classe 2.
3. **A1-A5** (ingestion) — pérennise pour les futurs documents.
4. **B4 + vérif** — qualité.

**Mesure** : nb entités avant/après ; nb `compatible_avec` avant/après (9 → ?) ;
et les cas nommés — « seuils compatibles 6101 » doit sortir 9F67/9F68/Z043 depuis
le graphe, « 6127 » doit avoir une entité unique avec toutes ses relations.

**Garde-fous merge** (destructif) : tourner en DRY-RUN d'abord (rapport des
groupes de merge proposés pour revue humaine), sauvegarde DB, et NE PAS
merger deux codes réellement distincts (ex. ne pas confondre `6101` et `6101A`
si ce dernier existe — le regex doit capturer la lettre suffixe).

## 5. Écarté

- Re-ingérer tout le corpus (coûteux, inutile : IDs stables → rejouer offline).
- Faire confiance au LLM pour dédupliquer seul (prouvé : il produit les variantes).
