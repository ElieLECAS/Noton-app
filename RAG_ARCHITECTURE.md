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
- **Méthode** : Conversion structurée en markdown + **JSON** pour le chunking (voir ci-dessous)
- **Performance** : Plus lent mais nécessaire pour :
  - PDFs scannés (avec OCR)
  - DOCX/XLSX/PPTX (conversion structurée)
  - Images (OCR)

#### OCR (schémas techniques, cotes)
- **Activation** : Configurable via `DOCLING_OCR_ENABLED` (défaut : `true`) et `DOCLING_OCR_LANG` (ex. `fr,en` ou `fra+eng`).
- **Moteurs** : EasyOCR ou Tesseract selon les options Docling (`PdfPipelineOptions`, `EasyOcrOptions` / `TesseractOcrOptions`). L’OCR permet de capturer le texte dans les images et schémas (cotes, légendes).
- **Format de sortie** : Le DocumentConverter produit en une seule passe le **markdown** (pour `note.content`) et le **JSON** du `DoclingDocument` (pour le DoclingNodeParser). Le JSON conserve les coordonnées (bbox) et la hiérarchie (tableaux, pictures, légendes).

### Résultat

Le contenu extrait est stocké dans `note.content` au format **markdown**, prêt pour l’affichage. Le chunking sémantique utilise exclusivement le **JSON** Docling (jamais le Markdown) pour préserver la structure.

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

### Documents importés (Docling) : chunking sémantique et JSON

Pour les documents traités par **Docling** (`document_service.py`), le découpage ne repose pas sur le markdown mais sur la **structure JSON** du document. Le `DocumentConverter` produit un `DoclingDocument` sérialisé en JSON via `model_dump_json()`, transmis au **DoclingNodeParser** (`chunking_service.py`). Ce format préserve la structure hiérarchique des tableaux (colonnes/lignes) et limite les confusions de valeurs numériques. Ne pas remplacer ce flux par du Markdown pour le parser. Voir `app/services/document_service.py` (création des `llama_docs`) et `app/services/chunking_service.py` (`chunk_note_from_docling_docs`).

#### Hiérarchie et étiquetage par section (parent_heading)
- Chaque chunk est **étiqueté** par le titre de sa section : `parent_heading` est le libellé complet construit à partir de tous les niveaux de headings (ex. « 1.3.1 Montage », « 2 Drainage »).
- Le regroupement pour les nœuds parents se fait par ce même libellé : tous les blocs (paragraphes, tableaux, schémas) d’une même section sont regroupés sous un parent commun, ce qui garde **texte et schéma ensemble**.

#### Tableaux et schémas comme une seule unité
- Le **HierarchicalChunker** Docling produit un chunk par élément (paragraphe, tableau, picture). Les tableaux et schémas ne sont pas recoupés ; ils restent une seule unité.

#### Légendes (fusion et injection)
- Pour les blocs **picture** ou **table**, la légende (ex. « Fig. 4 : Détail du perçage ») est **fusionnée** au texte du chunk.
- La légende est aussi **injectée** dans les métadonnées de tous les chunks de la même section (`image_anchor`, `figure_title`), afin que le contexte de la section soit disponible pour la recherche et le LLM.

#### Métadonnées stockées (metadata_json)
- **parent_heading** : titre de section (sujet).
- **page_no** : numéro de page (fourni par Docling).
- **figure_title** / **image_anchor** : légende(s) de la figure ou du tableau (section ou chunk).
- **contains_image** : présent et à `true` lorsque le chunk ou la section contient une image/table avec légende.

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
2. Recherche vectorielle SQL (pgvector) sur les leaves → `k * RERANKER_CANDIDATE_MULTIPLIER` candidats
3. Filtrage pré-reranking (similarité minimale), puis reranking (BGE-reranker-v2-m3) sur au plus `MAX_RERANK_CANDIDATES` candidats
4. Retourne les **k** passages les plus pertinents (contexte enrichi via résolution des parents)
5. Passages avec : contenu, titre de note, score, métadonnées (chunk_id, chunk_index, etc.)

**Reranker et titres descriptifs** : Avant le reranking, le texte de chaque candidat est enrichi avec `parent_heading` et `figure_title` (section et légende). Les chunks dont le titre de section ou la légende sont très descriptifs sont ainsi mieux notés par le reranker. Le passage final envoyé au LLM inclut aussi cette en-tête (section + légende) pour un contexte explicite.

**Paramètres** (voir `semantic_search_service.py` et `app/routers/chat.py`) :
- **k** (nombre de passages envoyés au LLM) : configurable via `RAG_TOP_K` (défaut : 10)
- **MAX_RERANK_CANDIDATES** : nombre max de candidats rerankés (défaut : 30)

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
- `DOCLING_OCR_ENABLED` : Activer l’OCR pour les schémas et PDF scannés (défaut : true)
- `DOCLING_OCR_LANG` : Langues OCR (ex. `fr,en` ou `fra+eng`, optionnel)
- `RAG_TOP_K` : Nombre de passages RAG envoyés au LLM (défaut : 10)
- `MAX_RERANK_CANDIDATES` : Nombre max de candidats rerankés avant sélection des k passages (défaut : 30)

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
