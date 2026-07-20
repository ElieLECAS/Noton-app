# Refonte de Qualité RAG pour l'évaluation CAG (document / set de pages)

_2026-07-20 · branche `fix/retriever`. Basé sur le run réel du golden espace 28
(20 questions, retrieval seul, export `evaluation_retriever_2026-07-20.json`)._

---

## 1. Le constat qui déclenche la refonte

L'évaluation actuelle est **page-stricte** : un hit = la page exacte attendue est
dans le top-k final. Or le retriever ne rend plus des chunks isolés, il alimente
un **CAG qui packe des documents entiers (ou des fenêtres)**. La bonne métrique
n'est donc plus « la bonne page est-elle dans le top-k » mais **« le bon document
est-il packé, et la bonne page est-elle dans le contexte final »**.

Preuve par deux cas réels que la page-stricte classe tous deux en **échec
(recall=0)** — mais que le CAG traite différemment :

| Cas | Page-strict | Réalité CAG (vérifié) |
|---|---|---|
| **g_415** (câbles Roto Safe, doc 415 p.32) | ❌ recall=0 | ✅ **Rattrapé** : le reranker a bien retiré la p.32 du top-k, MAIS d'autres pages du doc 415 ont classé le document en tête → CAG packe le doc **415 entier (41 pages)** → **la p.32 EST dans le contexte** |
| **g_412** (murs pose ITE, doc 412 p.4) | ❌ recall=0 | ❌ **Vrai échec, plus profond** : CAG packe 425/426/431/430 (dépliants Hybride/général), **le doc 412 n'est même pas packé** |

**Donc la page-stricte a un faux négatif (g_415) ET masque la vraie nature de
l'échec (g_412).** C'est précisément pourquoi il faut une éval document/set-de-pages.

---

## 2. Ce que révèle g_412 (l'exemple de l'utilisateur) — analyse pipeline

Contrairement à l'intuition « il a eu la page 10 donc le CAG donne la page 4 »,
la vérification montre un **double échec du reranker** :

1. **Retrieval : parfait.** pgvector, BM25 et le triple-RRF ont tous trouvé la
   **p.4** (la bonne). `post_rrf` (avant rerank) contient bien doc 412 p.4.
2. **Reranker (étape 1) : dégrade la page.** Après rerank, le doc 412 n'a plus
   que les pages **[10, 20]** — la p.4 correcte a été **écartée** au profit de
   pages moins pertinentes du même document.
3. **Reranker (étape 2) : dégrade le document.** Pire, le reranker classe des
   pages de **dépliants d'autres gammes** (Perform Hybride, DEPLIANT-HYBRIDE,
   Innoslide) **au-dessus** du doc 412. Résultat : `aggregate_documents` (CAG)
   élit 425/426/431/430 et **jette le doc 412**. Le bon document n'est pas packé.
4. **KAG n'a pas aidé** ici : `kag_only` n'a pas trouvé la p.4, et l'ajout du
   canal KAG a fait **baisser** la précision de 9 points sur cette question.

Cause racine probable : le cross-encoder **camembert-L2-mmarcoFR (2 couches
seulement)**, fraîchement activé et **jamais recalibré** (les seuils datent de
ms-marco), favorise la similarité de surface des dépliants marketing (« PVC »,
« isolation », « pose ») sur la réponse technique dense de la DTA. C'est un
problème de **qualité de discrimination du reranker**, pas de retrieval.

---

## 3. Ce que disent les métriques agrégées (20 questions, page-strict)

| Étape | Recall | Précision | MRR |
|---|---|---|---|
| **pgvector seul** | **1.00** | 0.67 | **0.72** |
| triple-RRF (sans KAG) | **1.00** | 0.635 | 0.69 |
| + KAG (post_rrf) | 0.90 | 0.52 | 0.58 |
| + reranker (**FINAL**) | 0.90 | 0.47 | 0.50 |
| bm25 seul | 0.75 | 0.32 | 0.32 |
| colpali seul | 0.65 | 0.51 | 0.55 |

Lectures clés (page-strict, donc précision à relativiser pour le CAG) :
- **pgvector seul a un recall PARFAIT (1.00)** : le bon contenu est toujours
  trouvé. Le retrieval de base n'est pas le problème.
- **KAG est net-négatif** : `kag_impact` = précision **−0.11**, recall **−0.10**,
  MRR **−0.11**. Il déplace des bonnes pages hors du top-k et ajoute du bruit.
- **Le reranker est net-négatif** en l'état : post_rrf → final, précision
  0.52→0.47, MRR 0.58→0.50, **aucun gain de recall**. Il **pousse la bonne page
  vers le bas**. Le final (0.90/0.47) est **pire que pgvector seul** (1.00/0.67).

⚠ La précision page-stricte pénalise injustement le reranker (rendre la p.10 au
lieu de la p.4 du même doc n'est pas une faute pour le CAG). Mais les **pertes de
recall** (2 questions) et l'**échec de sélection de document de g_412** sont
réels, eux, et pas des artefacts de métrique.

---

## 4. La refonte : métriques CAG-aware

### Hiérarchie de métriques cible

**PRIMAIRE (ce qui compte pour le CAG) :**
1. **Document Recall** — le bon document (ou l'un des `acceptable_document_ids`)
   est-il dans le contexte packé par `build_cag_context` ? (par étape ET final)
2. **Page-in-context** — la page attendue est-elle dans le set de pages
   réellement packé (`cag_documents[].pages`) ? C'est la VRAIE condition de
   succès : elle distingue « doc entier packé → page incluse » (g_415) de
   « fenêtre packée sans la page » ou « doc absent » (g_412).

**SECONDAIRE (diagnostic, à garder) :**
3. **Document Precision** — parmi les docs packés, combien sont pertinents (dans
   l'acceptable set) ? Mesure le bruit au niveau document (g_412 = 0/4 → 0 %).
4. **Recall/MRR page par étape** (colpali/pgvector/bm25/kag, avant/après rerank) —
   conservé comme diagnostic : c'est ce qui montre que le reranker dégrade le
   rang et que KAG ajoute du bruit. Mais ce n'est plus la métrique **titre**.

### Ce qu'il faut changer dans le code

Le golden a **déjà** le champ `acceptable_document_ids` (les docs valides par
gamme). Il reste à :

1. **`retriever_evaluator.evaluate_retriever_dataset`** : ajouter un matching
   **niveau document** (par `document_id` ∈ acceptable_set) en parallèle du
   matching page existant. Fonction déjà écrite : `evaluate_cag_document_hit`
   (ajoutée ce jour) — à brancher.
2. **Étape CAG dans l'évaluateur** : après le retrieval, appeler
   `build_cag_context(passages)` et calculer doc-hit + page-in-context sur
   `cag_documents`. C'est LA nouveauté (le CAG n'était pas évalué).
3. **`admin.html` (Qualité RAG)** : 
   - Métrique **titre** = Document Recall + Page-in-context (au lieu de la
     précision/recall page).
   - Nouveau panneau **CAG** par question : documents packés, bon doc présent ?,
     page attendue dans le contexte ?, full_document vs fenêtre.
   - Garder les panneaux par-retriever et avant/après rerank en **diagnostic
     repliable** (ils restent précieux pour voir POURQUOI ça rate).
4. **Golden** : `acceptable_document_ids` déjà là ; on peut y ajouter
   `reponse_attendue` plus tard SEULEMENT quand on voudra activer le juge de
   génération (pas maintenant — pas de crédits LLM de génération dépensés).

---

## 5. Ce que la refonte révélera (et actions au-delà de l'éval)

Avec les métriques CAG-aware, le tableau se recompose :
- **g_415 : miss → hit** (page dans le contexte). Faux négatif corrigé.
- **g_412 : reste un vrai miss**, mais l'éval montrera la cause exacte
  (« bon document non packé »), pas un flou « page non trouvée ».

Et surtout, ça isole **deux chantiers réels** que l'éval page masquait :
1. **Reranker** : net-négatif en l'état (dégrade rang + sélection de document).
   → recalibrer les seuils pour le modèle FR (jamais fait depuis P0), et
   **tester un cross-encoder plus profond** que le L2 (2 couches) — ou, vu que
   pgvector seul fait 1.00 de recall, **tester le pipeline reranker OFF** au
   niveau document et comparer.
2. **KAG** : net-négatif sur ce golden (−0.10 recall). → soit corriger son
   scoring/fusion, soit le sortir de la fusion de ranking et ne le garder que
   comme signal d'appoint (il reste utile pour les pages sans texte, mais pas
   comme canal RRF à poids égal ici).

**Décision méthodo importante** : on continue en **retrieval-only** (pas de
génération) tant que le document-recall CAG n'est pas bon. Inutile de payer des
crédits de génération sur une pipeline qui packe le mauvais document (g_412).
