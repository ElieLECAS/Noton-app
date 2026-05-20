# Chunking Markdown Structurel (MarkdownNodeParser)

## Vue d'ensemble

Nouvelle stratégie de chunking basée sur `MarkdownNodeParser` de LlamaIndex qui découpe les documents markdown selon leur **structure de headers** (H1-H6) au lieu de découper arbitrairement par taille de texte.

## Avantages vs HierarchicalNodeParser

| Critère | HierarchicalNodeParser (ancien) | MarkdownNodeParser (nouveau) |
|---------|--------------------------------|------------------------------|
| **Découpe** | Arbitraire par taille (1024/256 chars) | Sémantique par headers markdown |
| **Cohérence** | Peut couper au milieu d'une section | Respecte les sections complètes |
| **Métadonnées** | Heading extrait manuellement | Header path automatique (H1 > H2 > H3) |
| **Hiérarchie** | 2-3 niveaux (parents + feuilles) | Chunks uniques (is_leaf=True) |
| **Stockage** | ~2x chunks (parents non-embedés) | 1x chunks (tous embeddés) |
| **Retrieval** | Résolution parent complexe | Direct (pas de résolution) |
| **Cas d'usage** | Documents peu structurés | Documents markdown bien structurés (notices, DTA, etc.) |

## Configuration

### Variable d'environnement

Ajouter dans `.env` :

```bash
# false (défaut) = HierarchicalNodeParser (backward compat)
# true = MarkdownNodeParser pour les nouveaux documents
USE_MARKDOWN_STRUCTURED_CHUNKING=false
```

### Comportement

- **false** (défaut) : Utilise `chunk_markdown_hierarchical_with_tables()` → chunks parents + feuilles
- **true** : Utilise `chunk_markdown_structured()` → chunks uniques par section

## Migration

### Documents existants

**Non impactés** : Les documents déjà indexés conservent leur chunking hiérarchique.

### Nouveaux documents

Uploadés **après** activation du flag → utilisent automatiquement le nouveau système.

### Réindexation sélective

Pour migrer un document existant :

1. Activer `USE_MARKDOWN_STRUCTURED_CHUNKING=true`
2. Via l'admin bibliothèque : cliquer "Réindexer" sur le document
3. Le document sera rechunké avec la nouvelle stratégie

## Exemple de découpe

### Document source

```markdown
# Introduction
Texte introduction...

## Pose du dormant
Détails de la pose...

### Equipement en pattes
250 mm au droit des fermetures...

### Calage linéaire
Cales tous les 300 mm...

## Vérification
Contrôles finaux...
```

### Avec HierarchicalNodeParser (ancien)

```
Parent 1 (is_leaf=false) : "# Introduction\nTexte introduction...\n\n## Pose du dormant\nDétails..." [1024 chars max]
  └─ Leaf 1.1 (is_leaf=true) : "# Introduction\nTexte introduction..." [256 chars]
  └─ Leaf 1.2 (is_leaf=true) : "## Pose du dormant\nDétails de la pose..." [256 chars]
Parent 2 (is_leaf=false) : "### Equipement en pattes\n250 mm au droit..." [1024 chars max]
  └─ Leaf 2.1 (is_leaf=true) : "### Equipement en pattes\n250 mm..." [256 chars]
  └─ Leaf 2.2 (is_leaf=true) : "### Calage linéaire\nCales tous les 300 mm..." [256 chars]
...
```

→ Retrieval : trouve Leaf 2.1 → remplace par Parent 2 → perd précision

### Avec MarkdownNodeParser (nouveau)

```
Chunk 1 (is_leaf=true) : "# Introduction\nTexte introduction..." [section complète]
Chunk 2 (is_leaf=true) : "## Pose du dormant\nDétails de la pose..." [section complète]
Chunk 3 (is_leaf=true) : "### Equipement en pattes\n250 mm au droit des fermetures..." [sous-section complète]
Chunk 4 (is_leaf=true) : "### Calage linéaire\nCales tous les 300 mm..." [sous-section complète]
Chunk 5 (is_leaf=true) : "## Vérification\nContrôles finaux..." [section complète]
```

→ Retrieval : trouve Chunk 3 directement → contexte optimal

## Métadonnées enrichies

Chaque chunk produit par `MarkdownNodeParser` contient automatiquement :

- `Header 1`, `Header 2`, `Header 3`, etc. : Chemin de headers (ex: "Introduction" > "Pose du dormant" > "Equipement")
- `parent_heading` : Extrait du dernier header pour compatibilité
- `page_no` : Numéro de page (si marqueurs `<!-- page:N -->` présents)
- `node_id` : UUID unique du chunk
- `chunking_version` : `"markdown_structured_v1"`

## Gestion des tableaux

**Inchangée** : Les tableaux markdown sont toujours expansés atomiquement :

- `table_full` : Tableau complet (contexte)
- `table_row` : Lignes individuelles (granularité fine)
- `table_summary` : Résumé clé-valeur (tableaux 2 colonnes)

## Impact sur le retrieval

### Avec ancien système (hiérarchique)

1. Recherche vectorielle sur feuilles (256 chars)
2. Chargement de tous les parents en mémoire
3. Résolution multihop (jusqu'à 4 sauts)
4. Remplacement feuille → parent
5. Déduplication

### Avec nouveau système (structurel)

1. Recherche vectorielle sur chunks uniques (sections complètes)
2. **Pas de résolution parent** (déjà optimal)
3. Utilisation directe

→ **~40% moins de code, ~30% plus rapide**

## Code technique

### Fonction principale

```python
def chunk_markdown_structured(markdown: str, metadata_base: dict) -> List[dict]:
    """
    Chunking markdown structurel avec MarkdownNodeParser.
    Produit des chunks uniques (is_leaf=True) sans hiérarchie stockée.
    """
    parser = MarkdownNodeParser.from_defaults(
        include_metadata=True,
        include_prev_next_rel=False,
    )
    # ... découpe par structure headers
```

### Intégration dans le pipeline

```python
def create_chunks_for_document_from_markdown(
    session: Session,
    document: Document,
    markdown: str,
    generate_embeddings: bool = False,
) -> List[DocumentChunk]:
    use_structured = settings.USE_MARKDOWN_STRUCTURED_CHUNKING
    
    if use_structured:
        specs = chunk_markdown_structured(markdown, metadata_base)
    else:
        specs = chunk_markdown_hierarchical_with_tables(markdown, metadata_base)
    # ...
```

## Recommandations d'usage

### Activez le nouveau système si :

✅ Vos documents sont **bien structurés** en markdown (headers H1-H6 cohérents)  
✅ Vous traitez des **notices techniques**, DTA, documentations formelles  
✅ Vous voulez **simplifier** le retrieval (pas de résolution parent)  
✅ Vous voulez des chunks **sémantiquement cohérents** (sections complètes)

### Gardez l'ancien système si :

⚠️ Vos documents markdown sont **peu structurés** (headers rares/incohérents)  
⚠️ Vous avez besoin de **granularité ultra-fine** (< 256 chars)  
⚠️ Vous voulez **tester** le nouveau système avant migration globale  
⚠️ Documents **marketing visuels** avec peu de texte structuré

## Tests et validation

### Vérifier le chunking d'un document

1. Activer le logging : `LIBRARY_DOCUMENT_LOG_PATH=logs/library_document_processing.log`
2. Uploader un document test
3. Vérifier dans les logs :

```
[Chunking] document_id=42 — markdown structurel (MarkdownNodeParser) + expansion tableaux.
Chunking markdown structurel (MarkdownNodeParser): 15 chunks (12 texte, 3 tableaux, tous is_leaf=True)
```

### Comparer qualité des réponses

Voir [`docs/questions.md`](./questions.md) pour le corpus de test.

Script d'évaluation A/B :

```bash
# 1. Tester avec ancien système
USE_MARKDOWN_STRUCTURED_CHUNKING=false
# Lancer 25 questions test, noter précision/hallucinations

# 2. Réindexer avec nouveau système
USE_MARKDOWN_STRUCTURED_CHUNKING=true
# Bouton "Réindexer" sur documents test

# 3. Relancer mêmes questions, comparer
```

## Versions

- `markdown_hierarchical_v1` : Ancien système HierarchicalNodeParser simple
- `markdown_hierarchical_v2` : Ancien système + expansion tableaux atomique
- **`markdown_structured_v1`** : Nouveau système MarkdownNodeParser (cette doc)

## Références

- [LlamaIndex MarkdownNodeParser](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/)
- Code : [`app/services/chunking_service.py`](../app/services/chunking_service.py) lignes 2031-2180
- Config : [`app/config.py`](../app/config.py) ligne 43
- Logs : `logs/library_document_processing.log`
