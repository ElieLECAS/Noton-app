# Migration vers le système Docling RAG

Ce document décrit la procédure complète de migration vers le nouveau système de RAG avec Docling, qui permet l'import de documents de tous types (PDF, Excel, Word, images, etc.).

## Vue d'ensemble

### Nouveautés

- ✅ **Import de documents** : PDF, DOCX, XLSX, PPTX, CSV, images, JSON
- ✅ **Parsing intelligent** : Extraction de structure avec Docling (tableaux, titres, paragraphes)
- ✅ **Chunking unifié** : Traitement identique pour notes manuelles et documents
- ✅ **Métadonnées enrichies** : Type de chunk, numéro de page, section, etc.
- ✅ **Recherche vectorielle** : Embeddings sur chunks avec pgvector + HNSW
- ✅ **Interface améliorée** : Modal d'upload avec drag & drop, badges de type de fichier

### Architecture

**Avant** : Note → embedding direct (384 dimensions)

**Après** : Note → DocumentChunk[] → embeddings (384 dimensions)
- Chaque note est découpée en chunks intelligents
- Chaque chunk a son propre embedding
- Recherche sémantique au niveau des chunks

## Étapes de migration

### 1. Prérequis

Assurez-vous que Docker est en cours d'exécution et que vos services sont à jour :

```bash
docker-compose build
docker-compose up -d
```

### 2. Migration du schéma de base de données

Appliquer la migration Alembic pour créer la table `document_chunk` et ajouter les champs nécessaires :

```bash
docker-compose exec app alembic upgrade head
```

**Résultat attendu** :
- ✅ Table `document_chunk` créée
- ✅ Index vectoriels HNSW créés
- ✅ Champs `source_file_path` et `source_file_type` ajoutés à `note`

### 3. Migration des données existantes

Convertir toutes les notes existantes vers le système de chunks :

```bash
docker-compose exec app python -m app.scripts.migrate_to_docling --migrate
```

**Que fait cette commande ?**
- Parcourt toutes les notes existantes
- Crée des chunks pour chaque note
- Génère les embeddings pour chaque chunk
- Sauvegarde en base de données

**Durée estimée** : ~2-5 secondes par note (selon la longueur)

### 4. Vérification

Vérifier que la migration s'est bien déroulée :

```bash
docker-compose exec app python -m app.scripts.migrate_to_docling --verify
```

**Résultat attendu** :
```
📊 Statistiques :
  - Notes totales: X
  - Chunks totaux: Y
  - Chunks avec embeddings: Y/Y (100.0%)
  - Notes sans chunks: 0
✅ Toutes les notes ont des chunks
```

### 5. (Optionnel) Nettoyage des anciens embeddings

Une fois que vous avez vérifié que le nouveau système fonctionne correctement, vous pouvez supprimer les anciens embeddings au niveau des notes (champ deprecated) :

```bash
docker-compose exec app python -m app.scripts.migrate_to_docling --cleanup
```

**Note** : Cette étape est optionnelle. Les anciens embeddings sont conservés pour compatibilité mais ne sont plus utilisés.

## Utilisation

### Créer une note manuelle

Fonctionne exactement comme avant :
1. Cliquer sur "Nouvelle note"
2. Saisir titre et contenu
3. Les chunks et embeddings sont générés automatiquement

### Importer un document

Nouvelle fonctionnalité :
1. Cliquer sur "Importer un document"
2. Glisser-déposer ou sélectionner un fichier
3. Le document est :
   - Parsé avec Docling (extraction de structure)
   - Découpé en chunks intelligents
   - Indexé pour la recherche sémantique

**Formats supportés** :
- 📄 PDF
- 📝 Word (.docx)
- 📊 Excel (.xlsx)
- 🎨 PowerPoint (.pptx)
- 📋 CSV
- 🖼️ Images PNG, JPEG (avec OCR)
- 📄 Texte, Markdown, JSON

**Taille maximale** : 50 MB

### Recherche sémantique

Le chat RAG utilise maintenant les chunks :
- Recherche plus précise (chunks de ~500 caractères)
- Métadonnées enrichies dans les résultats (page, section, type)
- Support multi-sources (notes + documents)

## Résolution de problèmes

### La migration échoue sur une note

Si une note particulière pose problème, vous pouvez :
1. Noter l'ID de la note
2. La corriger ou la supprimer
3. Relancer la migration (elle skip automatiquement les notes déjà migrées)

### Pas d'embeddings générés

Vérifier que le modèle sentence-transformers est bien chargé :
```bash
docker-compose logs app | grep "embedding"
```

Si le modèle ne se charge pas, redémarrer le conteneur :
```bash
docker-compose restart app
```

### Docling ne parse pas un document

Vérifier les logs :
```bash
docker-compose logs app | grep "Docling"
```

Certains formats peuvent nécessiter des dépendances supplémentaires (déjà incluses dans requirements.txt).

## Rollback (en cas de problème)

Si vous devez revenir à l'ancien système :

```bash
# 1. Rollback du schéma
docker-compose exec app alembic downgrade -1

# 2. Redémarrer l'application
docker-compose restart app
```

**Attention** : Cela supprime tous les chunks et les documents importés.

## Performance

### Comparaison avant/après

**Avant** :
- 1 embedding par note
- Recherche sur notes complètes
- Contexte RAG parfois trop large

**Après** :
- ~2-10 embeddings par note (selon la longueur)
- Recherche sur chunks précis
- Contexte RAG optimisé

### Index HNSW

L'index HNSW (Hierarchical Navigable Small World) permet des recherches vectorielles ultra-rapides :
- Temps de recherche : O(log n) au lieu de O(n)
- Précision : 95%+ de rappel
- Paramètres : m=16, ef_construction=64 (optimisés)

## Support

Pour toute question ou problème :
1. Consulter les logs : `docker-compose logs -f app`
2. Vérifier la base de données : `docker-compose exec db psql -U noton_user -d noton_db`
3. Ouvrir une issue sur le dépôt GitHub

## Fichiers modifiés/créés

### Nouveaux fichiers
- `app/models/document_chunk.py` - Modèle DocumentChunk
- `app/services/docling_service.py` - Parsing avec Docling
- `app/services/document_chunking_service.py` - Chunking intelligent
- `app/services/file_storage_service.py` - Gestion des uploads
- `app/routers/documents.py` - API d'upload
- `app/scripts/migrate_to_docling.py` - Script de migration
- `alembic/versions/add_document_chunk_table.py` - Migration BDD

### Fichiers modifiés
- `app/models/note.py` - Ajout champs source_file_*
- `app/services/note_service.py` - Intégration chunking
- `app/services/semantic_search_service.py` - Recherche sur chunks
- `app/templates/project_detail.html` - UI d'upload
- `app/main.py` - Router documents
- `app/requirements.txt` - Dépendances Docling

## Prochaines étapes

Une fois la migration effectuée, vous pouvez :
1. ✅ Tester l'import de différents types de documents
2. ✅ Vérifier que la recherche sémantique fonctionne bien
3. ✅ Tester le chat RAG avec des documents importés
4. 🔄 Monitorer les performances et ajuster si nécessaire
5. 📊 Analyser les métriques d'utilisation

