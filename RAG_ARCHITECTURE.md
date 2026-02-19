# Architecture RAG - Noton App

Ce document explique le fonctionnement complet du système RAG (Retrieval-Augmented Generation) dans l'application Noton, de la lecture des documents jusqu'à la recherche sémantique.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Lecture des documents](#1-lecture-des-documents)
3. [Découpage en chunks](#2-découpage-en-chunks)
4. [Génération des embeddings](#3-génération-des-embeddings)
5. [Stockage](#4-stockage)
6. [Recherche sémantique](#5-recherche-sémantique)
7. [Architecture asynchrone](#6-architecture-asynchrone)
8. [Utilisation dans le chat](#7-utilisation-dans-le-chat)

---

## Vue d'ensemble

Le système RAG permet de :
- **Extraire** le contenu de documents (PDF, DOCX, images, etc.)
- **Découper** le contenu en chunks optimisés pour la recherche
- **Générer** des embeddings vectoriels pour chaque chunk
- **Stocker** les embeddings dans PostgreSQL avec pgvector
- **Rechercher** les passages les plus pertinents pour enrichir les réponses du chatbot

```
Document Upload → Extraction → Chunking → Embedding → Stockage → Recherche
```

---

## 1. Lecture des documents

**Fichier principal** : `app/services/document_service.py`

### Processus d'upload

Lorsqu'un utilisateur upload un document via l'endpoint `/api/projects/{project_id}/documents` :

1. **Création immédiate d'une note** avec :
   - Statut : `pending`
   - Contenu : "⏳ Traitement en cours..."
   - Type : `document`
   - Chemin du fichier sauvegardé

2. **Sauvegarde du fichier** dans `media/documents/` avec un nom unique (UUID)

3. **Ajout à la file d'attente** par projet (traitement séquentiel par projet pour éviter la surcharge)

### Stratégie de traitement

La fonction `process_document()` utilise une stratégie optimisée en deux étapes :

#### Stratégie 1 : PDF avec texte natif (ultra-rapide)
- **Outil** : PyMuPDF (fitz)
- **Méthode** : Extraction directe du texte natif
- **Performance** : Quasi-instantané (quelques millisecondes)
- **Cas d'usage** : PDFs avec texte sélectionnable

#### Stratégie 2 : Documents complexes (Docling)
- **Outil** : Docling DocumentConverter
- **Méthode** : Conversion structurée en markdown
- **Performance** : Plus lent mais nécessaire pour :
  - PDFs scannés (avec OCR)
  - DOCX/XLSX/PPTX (conversion structurée)
  - Images (OCR)

### Résultat

Le contenu extrait est stocké dans `note.content` au format **markdown**, prêt pour le découpage.

---

## 2. Découpage en chunks

**Fichier principal** : `app/services/chunking_service.py`

### Configuration

- **Taille des chunks** : 1000 caractères maximum (≈250 tokens)
- **Overlap** : 100 caractères (10% de chevauchement entre chunks)
- **Algorithme** : Adaptatif qui respecte la structure markdown

### Algorithme de chunking

La fonction `chunk_note()` utilise un algorithme intelligent :

1. **Combinaison** : Titre + contenu de la note
2. **Séparation en paragraphes** : Découpe selon `\n\n` (paragraphes markdown)
3. **Regroupement intelligent** :
   - Ne coupe jamais au milieu d'un paragraphe
   - Regroupe les paragraphes jusqu'à atteindre ~1000 caractères
   - Pour les très longs paragraphes : coupe aux fins de phrases

4. **Overlap** : Les chunks se chevauchent de 100 caractères pour maintenir le contexte

### Métadonnées des chunks

Chaque chunk contient :
- `chunk_index` : Position dans la note (0, 1, 2...)
- `start_char` : Position de début dans le texte original
- `end_char` : Position de fin dans le texte original
- `content` : Texte du chunk

### Stockage initial

Les chunks sont créés **sans embeddings** (rapide) et stockés dans la table `notechunk`. Les embeddings seront générés en arrière-plan.

---

## 3. Génération des embeddings

**Fichier principal** : `app/services/embedding_service.py`

### Modèle utilisé

- **Modèle** : Ollama `nomic-embed-text`
- **Dimensions** : 768 (configuré dans `app/embedding_config.py`)
- **API** : `POST {OLLAMA_BASE_URL}/api/embeddings`

### Processus de génération

#### Traitement par batch

La fonction `generate_embeddings_batch()` optimise les performances :

1. **Regroupement** : Traite jusqu'à 32 chunks à la fois
2. **Appels API** : Un appel par texte (Ollama API)
3. **Connexion réutilisable** : Utilise `httpx.Client` pour optimiser les requêtes HTTP

#### Insertion optimisée

La fonction `_process_embeddings_for_note()` utilise une méthode ultra-rapide :

1. **Table temporaire** : Crée une table temporaire PostgreSQL
2. **COPY** : Utilise `copy_expert` pour insérer tous les embeddings en une seule opération
3. **UPDATE batch** : Met à jour tous les chunks en une seule requête SQL

```python
# Exemple de la requête batch
UPDATE notechunk
SET embedding = temp_chunk_embeddings.embedding_vector::vector
FROM temp_chunk_embeddings
WHERE notechunk.id = temp_chunk_embeddings.chunk_id
```

### Résultat

Chaque chunk reçoit un **embedding vectoriel de 768 dimensions** stocké dans la colonne `embedding` de type `Vector(768)` (pgvector).

---

## 4. Stockage

**Fichier principal** : `app/models/note_chunk.py`

### Structure de la table `NoteChunk`

```python
class NoteChunk(SQLModel, table=True):
    id: Optional[int]                    # ID unique du chunk
    note_id: int                         # Référence à la note parente
    chunk_index: int                     # Position dans la note (0, 1, 2...)
    content: str                         # Texte du chunk
    embedding: Optional[List[float]]     # Vector(768) - embedding pgvector
    start_char: int                      # Position de début dans la note originale
    end_char: int                        # Position de fin dans la note originale
```

### Base de données

- **SGBD** : PostgreSQL
- **Extension** : pgvector (pour le stockage et la recherche vectorielle)
- **Type de colonne** : `Vector(768)` pour les embeddings

### Avantages de pgvector

- **Recherche native** : Opérateurs SQL pour la similarité vectorielle
- **Performance** : Index HNSW pour des recherches rapides
- **Intégration** : Fonctionne directement avec SQLModel/SQLAlchemy

---

## 5. Recherche sémantique

**Fichier principal** : `app/services/semantic_search_service.py`

### Deux fonctions de recherche

#### a) `search_relevant_notes()` - Recherche par note

**Objectif** : Retourner les notes les plus pertinentes

**Processus** :
1. Génère l'embedding de la requête utilisateur
2. Recherche les chunks les plus pertinents avec pgvector
3. Groupe par note (`DISTINCT ON (nc.note_id)`)
4. Retourne les k notes les plus pertinentes avec leur score

**Utilisation** : Pour obtenir une vue d'ensemble des notes pertinentes

#### b) `search_relevant_passages()` - Recherche par chunk ⭐

**Objectif** : Retourner les passages (chunks) les plus pertinents

**Processus** :
1. Génère l'embedding de la requête utilisateur
2. Recherche directement les k chunks les plus pertinents
3. Retourne les passages avec :
   - Le contenu du chunk
   - Le titre de la note source
   - Le score de similarité
   - Les métadonnées (chunk_id, chunk_index, etc.)

**Utilisation** : Pour enrichir le contexte du chatbot avec des passages précis

### Requête SQL utilisée

```sql
SELECT 
    nc.id as chunk_id,
    nc.content as chunk_content,
    nc.chunk_index,
    nc.start_char,
    nc.end_char,
    n.id as note_id,
    n.title as note_title,
    n.note_type,
    n.project_id,
    n.user_id,
    1 - (nc.embedding <=> '{query_embedding}'::vector) as similarity_score
FROM notechunk nc
INNER JOIN note n ON nc.note_id = n.id
WHERE n.project_id = {project_id}
    AND n.user_id = {user_id}
    AND nc.embedding IS NOT NULL
ORDER BY nc.embedding <=> '{query_embedding}'::vector
LIMIT k
```

### Opérateur pgvector

- **`<=>`** : Opérateur de distance cosinus
- **`1 - distance`** : Conversion en score de similarité (0 à 1)
- **Tri** : Les chunks les plus similaires sont retournés en premier

### Sécurité

- Vérification que le projet appartient à l'utilisateur
- Filtrage par `project_id` et `user_id`
- Seuls les chunks avec embeddings sont recherchés

---

## 6. Architecture asynchrone

### Workers de documents

**Fichier** : `app/services/document_service.py`

- **File d'attente** : `project_queues[project_id]` (une queue par projet)
- **Workers** : Threads daemon qui traitent les documents séquentiellement par projet
- **Configuration** : `MAX_CONCURRENT_DOCUMENTS` workers en parallèle
- **Isolation** : Traitement isolé pour ne pas bloquer l'application

**Flux** :
```
Upload → File d'attente par projet → Worker → Extraction → Chunking → File d'embeddings
```

### Workers d'embeddings

**Fichier** : `app/services/chunk_service.py`

- **File d'attente** : `embedding_queue` (globale)
- **Workers** : 1 worker à la fois (`MAX_CONCURRENT_EMBEDDINGS = 1`)
- **Raison** : Éviter la surcharge CPU (génération d'embeddings intensive)

**Flux** :
```
Chunks créés → File d'embeddings → Worker → Génération batch → Stockage optimisé
```

### Avantages

- **Non-bloquant** : L'upload retourne immédiatement
- **Scalable** : Traitement en arrière-plan
- **Robuste** : Gestion d'erreurs et retry automatique
- **Performant** : Traitement par batch et insertion optimisée

---

## 7. Utilisation dans le chat

**Fichier principal** : `app/routers/chat.py`

### Intégration RAG

Lorsqu'un utilisateur pose une question dans le chat :

1. **Recherche sémantique** : `search_relevant_passages()` trouve les k chunks les plus pertinents
2. **Construction du contexte** : Les passages sont formatés avec le titre de la note
3. **Enrichissement du prompt** : Les passages sont ajoutés au contexte du LLM
4. **Génération de la réponse** : Le LLM génère une réponse enrichie par le contexte RAG

### Format du contexte

```markdown
**Titre de la note**
Contenu du chunk pertinent...

**Autre note**
Autre chunk pertinent...
```

### Avantages

- **Réponses précises** : Basées sur le contenu réel des documents
- **Contexte local** : Utilise uniquement les notes du projet
- **Transparence** : Les sources sont identifiables (titre de la note)

---

## Schéma complet du flux

```
┌─────────────────┐
│  Upload Document │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  File d'attente  │ (par projet)
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Worker Document │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Extraction     │ (Docling/PyMuPDF)
│  → Markdown     │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Chunking       │ (1000 chars, 10% overlap)
│  → NoteChunk    │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  File Embeddings │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Worker Embedding│
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Génération     │ (Ollama nomic-embed-text)
│  → Vector(768)  │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Stockage       │ (PostgreSQL + pgvector)
│  → NoteChunk    │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Recherche      │ (pgvector <=>)
│  → Passages     │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Chatbot        │ (Contexte enrichi)
└─────────────────┘
```

---

## Configuration

### Variables d'environnement

- `EMBEDDING_MODEL` : Modèle Ollama pour les embeddings (défaut: `nomic-embed-text`)
- `EMBEDDING_DIMENSION` : Dimension des embeddings (défaut: `768`)
- `OLLAMA_BASE_URL` : URL de base d'Ollama
- `MAX_CONCURRENT_DOCUMENTS` : Nombre de workers de documents
- `TORCH_NUM_THREADS` : Nombre de threads PyTorch pour Docling
- `DOCLING_USE_GPU` : Activer/désactiver le GPU pour Docling

### Fichiers de configuration

- `app/embedding_config.py` : Configuration centralisée des embeddings
- `app/config.py` : Configuration générale de l'application

---

## Performance et optimisations

### Optimisations implémentées

1. **Extraction rapide** : PyMuPDF pour PDFs avec texte natif
2. **Chunking adaptatif** : Respecte la structure markdown
3. **Batch processing** : Génération d'embeddings par batch (32 chunks)
4. **Insertion optimisée** : COPY PostgreSQL pour insertion rapide
5. **Recherche vectorielle native** : pgvector avec index HNSW
6. **Traitement asynchrone** : Workers en arrière-plan

### Métriques

- **Upload** : Retour immédiat (< 1 seconde)
- **Extraction PDF natif** : < 100ms
- **Extraction PDF scanné** : 5-30 secondes (selon la taille)
- **Génération embeddings** : ~100-500ms par batch de 32 chunks
- **Recherche** : < 50ms pour trouver les k chunks les plus pertinents

---

## Conclusion

Le système RAG de Noton App est conçu pour être :
- **Performant** : Traitement asynchrone et optimisations multiples
- **Scalable** : Architecture modulaire avec workers
- **Précis** : Recherche sémantique basée sur les chunks
- **Robuste** : Gestion d'erreurs et fallbacks

Le système permet d'enrichir les réponses du chatbot avec le contenu réel des documents uploadés par l'utilisateur, offrant une expérience de chat contextuelle et précise.
