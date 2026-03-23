# Résumé de l'implémentation de la refonte bibliothèque centralisée

## État d'avancement

### ✅ Phase 1 : COMPLÉTÉE
- Nouveaux modèles créés : Library, Folder, Space, Document, DocumentSpace, DocumentChunk
- Modèles KAG modifiés : `project_id` → `space_id` dans KnowledgeEntity, ChunkEntityRelation, Conversation
- Migration Alembic créée : `add_library_architecture.py`

### ✅ Phase 2 : COMPLÉTÉE  
- Services de base créés :
  - `library_service.py` - Gestion bibliothèque
  - `folder_service.py` - Arborescence de dossiers
  - `space_service.py` - Gestion espaces
  - `document_space_service.py` - Associations documents-espaces
- `document_service_new.py` créé avec :
  - Logique Docling conservée
  - Support multi-espaces
  - Upload centralisé
  - Workers asynchrones adaptés

### ⚠️ Phase 3 : À COMPLÉTER MANUELLEMENT

Les fichiers suivants nécessitent des modifications substantielles pour remplacer `note` par `document` et gérer le KAG multi-espaces :

#### 1. `app/services/chunk_service.py`

**Modifications requises :**

```python
# Remplacements à effectuer dans tout le fichier:
# - Note → Document
# - note → document
# - note_id → document_id
# - NoteChunk → DocumentChunk
# - notechunk → documentchunk
# - project_id → library_id (pour les workers)

# Fonction clé à adapter : _process_kag_extraction_for_note
# Nouvelle logique :
def _process_kag_extraction_for_document(session: Session, document_id: int, space_id: int):
    """
    Extrait les entités KAG pour UN espace spécifique.
    Appelé pour chaque espace ayant accès au document.
    """
    from app.services.kag_graph_service import save_entities_for_chunk, delete_entities_for_document
    from app.services.kag_extraction_service import extract_entities_sync
    
    # Supprimer anciennes entités pour ce document dans cet espace
    delete_entities_for_document(session, document_id, space_id)
    
    # Récupérer chunks du document
    chunks = get_chunks_by_document(session, document_id)
    leaf_chunks = [c for c in chunks if c.is_leaf]
    
    # Extraire entités pour chaque chunk
    for chunk in leaf_chunks:
        entities = extract_entities_sync(chunk.content)
        if entities:
            save_entities_for_chunk(session, chunk.id, entities, space_id)

# Nouvelle fonction pour traiter tous les espaces:
def process_kag_for_all_document_spaces(session: Session, document_id: int):
    """Traite le KAG pour tous les espaces ayant accès au document."""
    from app.models.document_space import DocumentSpace
    from sqlmodel import select
    
    doc_spaces = session.exec(
        select(DocumentSpace).where(DocumentSpace.document_id == document_id)
    ).all()
    
    for ds in doc_spaces:
        _process_kag_extraction_for_document(session, document_id, ds.space_id)
```

#### 2. `app/services/kag_graph_service.py`

**Modifications requises :**

```python
# Remplacements globaux:
# - project_id → space_id partout
# - Project → Space

# Fonctions à adapter:

def get_or_create_entity(session: Session, name: str, entity_type: str, space_id: int) -> KnowledgeEntity:
    """Clé logique: (space_id, name_normalized)"""
    name_normalized = normalize_entity_name(name)
    
    statement = select(KnowledgeEntity).where(
        KnowledgeEntity.space_id == space_id,
        KnowledgeEntity.name_normalized == name_normalized
    )
    entity = session.exec(statement).first()
    
    if not entity:
        entity = KnowledgeEntity(
            name=name,
            name_normalized=name_normalized,
            entity_type=entity_type,
            space_id=space_id
        )
        session.add(entity)
        session.commit()
        session.refresh(entity)
    
    return entity

def save_entities_for_chunk(
    session: Session,
    chunk_id: int,
    entities: List[dict],
    space_id: int  # NOUVEAU PARAMÈTRE
) -> List[KnowledgeEntity]:
    """Sauvegarde entités pour un chunk dans un espace spécifique."""
    saved_entities = []
    
    for entity_data in entities:
        entity = get_or_create_entity(
            session,
            entity_data["name"],
            entity_data["type"],
            space_id  # Passer space_id
        )
        
        # Créer relation chunk-entité
        relation = ChunkEntityRelation(
            chunk_id=chunk_id,
            entity_id=entity.id,
            relevance_score=entity_data.get("importance", 1.0),
            space_id=space_id  # Ajouter space_id
        )
        session.add(relation)
        saved_entities.append(entity)
    
    session.commit()
    return saved_entities

def delete_entities_for_document(session: Session, document_id: int, space_id: int):
    """Supprime entités KAG d'un document pour un espace spécifique."""
    from app.models.document_chunk import DocumentChunk
    from sqlmodel import select
    
    # Récupérer tous les chunks du document
    chunks = session.exec(
        select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    ).all()
    
    chunk_ids = [c.id for c in chunks]
    
    # Supprimer relations pour cet espace
    relations = session.exec(
        select(ChunkEntityRelation).where(
            ChunkEntityRelation.chunk_id.in_(chunk_ids),
            ChunkEntityRelation.space_id == space_id
        )
    ).all()
    
    for rel in relations:
        session.delete(rel)
    
    session.commit()

def delete_entities_for_space(session: Session, space_id: int):
    """Supprime toutes les entités et relations d'un espace."""
    entities = session.exec(
        select(KnowledgeEntity).where(KnowledgeEntity.space_id == space_id)
    ).all()
    
    for entity in entities:
        session.delete(entity)
    
    relations = session.exec(
        select(ChunkEntityRelation).where(ChunkEntityRelation.space_id == space_id)
    ).all()
    
    for rel in relations:
        session.delete(rel)
    
    session.commit()

def rebuild_kag_for_space(session: Session, space_id: int):
    """Reconstruit le KAG pour tous les documents d'un espace."""
    from app.services.document_space_service import get_documents_for_space
    from app.services.chunk_service import _process_kag_extraction_for_document
    
    delete_entities_for_space(session, space_id)
    
    documents = get_documents_for_space(session, space_id, user_id=None)  # À ajuster
    
    for document in documents:
        if document.processing_status == "completed":
            _process_kag_extraction_for_document(session, document.id, space_id)

# Fonction utilitaire pour process_kag_for_document_space (appelée depuis document_service)
def process_kag_for_document_space(session: Session, document_id: int, space_id: int):
    """Point d'entrée pour traiter le KAG d'un document dans un espace."""
    from app.services.chunk_service import _process_kag_extraction_for_document
    _process_kag_extraction_for_document(session, document_id, space_id)
```

#### 3. `app/services/semantic_search_service.py`

**Modifications requises :**

```python
# Remplacements:
# - project_id → space_id
# - Project → Space

def search_relevant_passages(
    session: Session,
    query: str,
    space_id: int,  # CHANGÉ de project_id
    user_id: int,
    top_k: int = 10,
    similarity_threshold: float = 0.7
) -> List[dict]:
    """Recherche sémantique scopée à un espace."""
    from app.models.document_chunk import DocumentChunk
    from app.models.document import Document
    from app.models.document_space import DocumentSpace
    from app.services.embedding_service import generate_query_embedding
    
    # Générer embedding de la query
    query_embedding = generate_query_embedding(query)
    
    # Requête avec filtrage par espace via DocumentSpace
    statement = select(DocumentChunk, Document).join(
        Document, DocumentChunk.document_id == Document.id
    ).join(
        DocumentSpace, Document.id == DocumentSpace.document_id
    ).where(
        DocumentSpace.space_id == space_id,
        DocumentChunk.is_leaf == True,
        DocumentChunk.embedding.isnot(None)
    ).order_by(
        DocumentChunk.embedding.cosine_distance(query_embedding)
    ).limit(top_k)
    
    results = session.exec(statement).all()
    
    passages = []
    for chunk, document in results:
        distance = chunk.embedding.cosine_distance(query_embedding)
        similarity = 1 - distance
        
        if similarity >= similarity_threshold:
            passages.append({
                "chunk_id": chunk.id,
                "document_id": document.id,
                "document_title": document.title,
                "content": chunk.content,
                "similarity": similarity,
                "metadata": chunk.metadata_json or {}
            })
    
    # Si KAG activé, fusionner avec résultats graphe
    if getattr(settings, "KAG_ENABLED", False):
        from app.services.kag_graph_service import get_chunks_by_entity_names
        from app.services.kag_extraction_service import extract_entities_from_query_sync
        
        entities = extract_entities_from_query_sync(query)
        entity_names = [e["name"] for e in entities]
        
        if entity_names:
            graph_chunks = get_chunks_by_entity_names(session, entity_names, space_id)
            # Fusionner et dédupliquer
            # ... (logique de fusion à conserver)
    
    return passages
```

### ⚠️ Phase 4 : À COMPLÉTER MANUELLEMENT

#### Créer les routers

**1. `app/routers/library.py`**

Points clés :
- `GET /api/library` - Récupérer bibliothèque user
- `GET /api/library/folders` - Dossiers racine
- `POST /api/library/folders` - Créer dossier
- `GET /api/library/documents` - Tous documents
- **`POST /api/library/upload`** - Upload unifié avec `space_ids: List[int]`
- `POST /api/library/documents/{document_id}/spaces` - Gérer espaces d'un document

**2. `app/routers/spaces.py`**

Structure similaire à l'ancien `projects.py` mais pour les espaces.

**3. Adapter routers existants**

- `conversations.py` : `project_id` → `space_id`
- `chat.py` : `/api/projects/{project_id}/chat/stream` → `/api/spaces/{space_id}/chat/stream`
- `kag.py` : `/api/kag/projects/{project_id}/*` → `/api/kag/spaces/{space_id}/*`

### ⚠️ Phase 5 : Nettoyage

- Supprimer ou renommer anciens services/routers
- Mettre à jour `app/main.py` pour inclure nouveaux routers
- Tester la migration Alembic

## Structure finale des fichiers

### Nouveaux fichiers créés
✅ `app/models/library.py`
✅ `app/models/folder.py`
✅ `app/models/space.py`
✅ `app/models/document.py`
✅ `app/models/document_chunk.py`
✅ `app/models/document_space.py`
✅ `app/services/library_service.py`
✅ `app/services/folder_service.py`
✅ `app/services/space_service.py`
✅ `app/services/document_space_service.py`
✅ `app/services/document_service_new.py` (renommer en `document_service.py`)
✅ `app/alembic/versions/add_library_architecture.py`

### Fichiers à adapter (TODO manuel)
⚠️ `app/services/chunk_service.py`
⚠️ `app/services/kag_graph_service.py`
⚠️ `app/services/kag_extraction_service.py` (si nécessaire)
⚠️ `app/services/semantic_search_service.py`
⚠️ `app/routers/conversations.py`
⚠️ `app/routers/chat.py`
⚠️ `app/routers/kag.py`

### Fichiers à créer (TODO manuel)
⚠️ `app/routers/library.py`
⚠️ `app/routers/spaces.py`

### Fichiers à supprimer/archiver
- `app/services/project_service.py`
- `app/services/note_service.py`
- `app/routers/projects.py`
- `app/routers/notes.py`

## Instructions pour finaliser l'implémentation

1. **Renommer le nouveau document_service**
   ```bash
   mv app/services/document_service.py app/services/document_service_old.py
   mv app/services/document_service_new.py app/services/document_service.py
   ```

2. **Adapter chunk_service.py**
   - Utiliser rechercher/remplacer pour les renommages de base
   - Adapter manuellement les fonctions KAG selon les exemples ci-dessus

3. **Adapter kag_graph_service.py**
   - Remplacer `project_id` par `space_id` partout
   - Ajouter paramètre `space_id` aux fonctions de sauvegarde d'entités

4. **Adapter semantic_search_service.py**
   - Modifier requêtes pour joindre DocumentSpace
   - Filtrer par `space_id`

5. **Créer les routers**
   - Copier la structure de `projects.py` pour `spaces.py`
   - Créer `library.py` avec endpoints d'upload

6. **Exécuter la migration**
   ```bash
   cd app && alembic upgrade head
   ```

7. **Tester**
   - Créer un espace
   - Uploader un document dans plusieurs espaces
   - Vérifier que le KAG est isolé par espace

## Architecture finale

```
User
  └── Library
        ├── Folders (arborescence)
        │     └── Documents
        └── Documents (racine)

Document (traité 1x)
  ├── DocumentChunks (embeddings partagés)
  └── DocumentSpace (associations)
        └── Spaces
              ├── KnowledgeEntity (isolé par space)
              ├── ChunkEntityRelation (isolé par space)
              └── Conversations (scopées par space)
```

## Principes clés de l'architecture

1. **Un document, un traitement** : Docling, chunking, embeddings ne sont exécutés qu'une seule fois
2. **KAG multi-espaces** : Les entités sont extraites pour chaque espace ayant accès au document
3. **Embeddings partagés** : Les chunks vectoriels sont réutilisés par tous les espaces
4. **Isolation stricte** : Les conversations ne voient que les documents de leur espace
5. **Ajout/retrait dynamique** : Cocher/décocher un espace régénère/supprime uniquement son KAG
