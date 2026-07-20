# Plan — resynchronisation ColPali/LanceDB ↔ Postgres

_2026-07-20 · branche `fix/retriever`. Mesuré sur le corpus prod réel (dump
Postgres + scan complet de la table `colpali_patches` LanceDB, 1 703 424
lignes)._

---

## 1. Diagnostic — ampleur réelle (mesurée, pas estimée)

| Métrique | Valeur |
|---|---|
| Documents en base | 57 |
| Documents avec des patches ColPali (LanceDB) | 38 |
| Documents **sains** (0 % orphelin) | **32** |
| Documents **100 % orphelins** (index ColPali mort) | **6** |
| Documents **partiellement** orphelins | **0** |
| Total patches LanceDB | 1 703 424 |
| Total patches orphelins | 250 245 (14,7 %) |

Le partage est **strictement binaire** (aucun document partiellement touché) :
soit un document est parfaitement synchronisé, soit son index ColPali est
**intégralement mort** (100 % des patches pointent vers des `chunk_id` qui
n'existent plus). C'est la signature d'un événement précis, pas d'une dérive
progressive.

### Les 6 documents concernés

| doc_id | Titre | Pages (L0 actuels) | Patches LanceDB (100 % orphelins) |
|---|---|---|---|
| 389 | Mise en oeuvre 9708 - Fer cintrés | 1 | 747 |
| 391 | Montage Roto NX KSR PVC IMO_180_NX_FR_v2 | 124 | 92 628 |
| 411 | DTD-DBV-6-16-2335_V5 | 53 | 39 591 |
| 412 | 6-16-2334_V5_TROCAL_76_ADVANCED_PROFINE (2) | 56 | 41 832 |
| 413 | DTD-DBV-25-06-16-2334_V5_TROCAL_76_ADVANCED (1) | 60 | 44 820 |
| 415 | Roto Safe E _ Jonction de câble_SUG_28_FR_v3 | 41 | 30 627 |

Les 6 fichiers PDF sources existent bien sur disque (vérifié). Les 6
documents ont des ancres L0 `page_anchor` actuelles valides en base (1 à 124
selon le document) — **c'est bien le côté LanceDB qui est mort, pas le côté
Postgres.**

**Correctif de mon diagnostic précédent** : j'avais initialement soupçonné le
document 422 (le DTA testé sur « profil 6101 ») — c'était une erreur de
lecture d'un warning agrégé sur une requête multi-documents. Le scan complet
montre que le **422 est parfaitement sain (0 % orphelin)**. Les orphelins de
ce warning venaient de 411/412/413, présents dans la même requête.

---

## 2. Cause racine (déduite du code + de la donnée)

Vérifié dans `document_indexing_service.py` :

- **`IndexingMode.TEXT_ONLY`** (`_extract_and_persist_chunks(preserve_page_anchors=True)`,
  lignes 397-448) : pour chaque page, si une ancre L0 existante est trouvée
  (par `page_no`), elle est **mise à jour en place** (`session.add(existing)`)
  — le `chunk.id` Postgres, qui sert de clé (`chunk_id`) dans LanceDB, **ne
  change jamais**. Par conception, TEXT_ONLY seul ne peut PAS désynchroniser
  ColPali. Ton scénario « colpali puis texte only » est donc **sûr par
  construction**, et les 32 documents sains le confirment empiriquement.

- **`IndexingMode.FULL`** (`_delete_all_chunks`, lignes 275-295) : supprime
  **toutes** les ancres L0 et recrée tout de zéro avec de **nouveaux**
  `chunk.id` (pas de logique de préservation dans ce chemin). Si un document a
  subi un passage en mode FULL **après** son premier sync ColPali, sans
  qu'un nouveau sync ColPali ne soit relancé ensuite, 100 % de ses patches
  LanceDB (qui référencent les anciens `chunk_id`) deviennent orphelins d'un
  coup — exactement le motif observé (tout ou rien, jamais partiel).

**Hypothèse la plus probable** : ces 6 documents ont été repassés en FULL au
moins une fois après leur ingestion ColPali initiale (reprocessing, correction
de contenu, etc.), sans resync ColPali derrière. Sans journal d'audit détaillé
des opérations passées, on ne peut pas remonter à l'événement exact — mais
l'explication est cohérente avec le code, avec la donnée (tout-ou-rien), et
suffisante pour agir : peu importe la cause exacte, le correctif est le même.

---

## 3. Le correctif sûr : `IndexingMode.COLPALI_ONLY`, ciblé sur les 6 documents

### Pourquoi c'est sûr

- **Ne touche ni texte, ni KAG, ni embeddings, ni catégories.** Dans
  `process_document_indexing`, le bloc extraction/KAG/enrichissement/embedding
  est gardé par `if mode in (IndexingMode.FULL, IndexingMode.TEXT_ONLY):`
  (ligne 124) — **COLPALI_ONLY ne rentre jamais dans ce bloc.** Il saute
  directement à la synchronisation ColPali (ligne 199).
- **Aucune suppression de chunk.** Le mode COLPALI_ONLY ne supprime aucun
  chunk Postgres — les ancres L0 actuelles (déjà valides) sont utilisées
  telles quelles.
- **Le nettoyage LanceDB est automatique et correct.**
  `insert_colpali_patches_batch_lancedb` fait `table.delete(f"document_id =
  {document_id}")` **avant** d'insérer ([lancedb_service.py:93-94](app/services/lancedb_service.py:93))
  — supprime TOUS les patches existants pour ce document (y compris les
  orphelins), puis réinsère des patches propres pointant vers les `chunk_id`
  **actuellement** valides. Pas de risque de doublons ni de résidus.
- **Zéro coût API LLM.** `_sync_colpali_for_pages` appelle
  `embed_pdf_pages_colpali` → ColQwen2 local (le même modèle déjà préchargé au
  démarrage). Aucun appel Mistral, donc pas de coût facturé — juste du temps
  de calcul local (CPU/GPU) proportionnel au nombre de pages.
- **Zéro risque pour les 32 documents sains** : le plan ne les touche pas.

### Ce qui manque pour l'exécuter

Aujourd'hui, **aucun endpoint ni script n'appelle `IndexingMode.COLPALI_ONLY`**
— le mode existe dans l'enum et est géré par `process_document_indexing`, mais
il est inatteignable (tous les appelants existants forcent `mode=FULL`). Il
faut un petit script (pas une modification d'endpoint, pas de risque sur le
chemin d'ingestion normal) qui appelle directement :

```python
process_document_indexing(
    document_id=doc_id,
    file_path=doc.source_file_path,
    user_id=<un user id valide, ex. l'admin>,
    mode=IndexingMode.COLPALI_ONLY,
)
```

pour chacun des 6 `doc_id` : `389, 391, 411, 412, 413, 415`.

---

## 4. Étapes d'exécution

1. **Écrire le script de resync ciblé** (jetable, `app/scripts/` — pas un
   endpoint permanent, sauf si tu veux garder cette capacité pour plus tard).
   Boucle sur les 6 `doc_id`, appelle `process_document_indexing(...,
   mode=IndexingMode.COLPALI_ONLY)`, logge le nombre de patches insérés par
   document.
2. **Exécuter dans le conteneur `web`** (ou `worker`, peu importe — pas besoin
   de passer par Celery, c'est un one-shot).
3. **Vérifier** : relancer le scan complet du §1 (mesure déjà écrite et
   validée dans cette session) → attendu : les 6 documents passent à 0 %
   orphelin, les 32 autres restent inchangés à 0 %.
4. **Vérifier fonctionnellement** : relancer une requête ColPali sur un de ces
   documents (ex. doc 411, requête visuelle) et confirmer l'absence du warning
   `[retrieve_colpali_pages] ... orphelins`.
5. **Optionnel, séparé** : 19 documents sur 57 n'ont **aucun** patch ColPali
   du tout (ni sain ni orphelin — absents de la table). Ce n'est pas le même
   problème (pas une désynchronisation, potentiellement des documents ingérés
   avant l'activation de ColPali, ou non-PDF). À vérifier séparément si tu
   veux une couverture ColPali complète du corpus — hors périmètre de ce
   correctif de resync.

---

## 5. Ce que je NE recommande PAS

- **Ne pas repasser ces 6 documents en FULL** pour « repartir propre » — ce
  serait payer à nouveau l'extraction vision + KAG + enrichissement (coût API
  réel) pour un problème qui ne concerne que ColPali. COLPALI_ONLY est
  strictement suffisant et gratuit.
- **Ne pas toucher aux 32 documents sains** — le scan confirme qu'ils n'ont
  besoin de rien.
- **Ne pas généraliser TEXT_ONLY comme suspect** — le code et la donnée
  convergent : TEXT_ONLY seul est sûr par conception. Le risque vient
  spécifiquement d'un passage FULL sans resync ColPali derrière ; à garder en
  tête pour l'avenir (toute future FULL réindexation doit être suivie d'un
  sync ColPali, ce qui est déjà le comportement du mode FULL lui-même — le
  risque n'existe que si FULL et COLPALI_ONLY/sync sont dissociés dans le
  temps par erreur opérationnelle).
