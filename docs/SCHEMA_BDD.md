# Schéma de la base de données — Noton-app

_Dernière mise à jour : 2026-03-17_

Documentation table par table, colonne par colonne.  
Base PostgreSQL avec extension **pgvector** pour les embeddings (dimension par défaut : **1024**, BGE-m3).  
Source de vérité : modèles SQLModel et migrations Alembic du projet.

---

## 1. `user`

Utilisateurs de l'application.

| Colonne        | Type           | Contraintes              | Description                    |
|----------------|----------------|---------------------------|--------------------------------|
| `id`           | INTEGER        | PK, auto                  | Identifiant unique             |
| `username`     | VARCHAR(150)   | NOT NULL, UNIQUE, index   | Nom d'utilisateur              |
| `email`        | VARCHAR(255)   | NOT NULL, UNIQUE, index   | Email                          |
| `password_hash`| VARCHAR        | NOT NULL                  | Hash du mot de passe           |
| `created_at`   | TIMESTAMP      | NOT NULL                  | Date de création               |
| `updated_at`   | TIMESTAMP      | NOT NULL                  | Dernière mise à jour           |

---

## 2. `project`

Projets (dossiers de notes) appartenant à un utilisateur.

| Colonne       | Type          | Contraintes    | Description                    |
|---------------|---------------|----------------|--------------------------------|
| `id`          | INTEGER       | PK, auto       | Identifiant unique             |
| `title`       | VARCHAR(200)  | NOT NULL       | Titre du projet                |
| `description` | TEXT          | NULL           | Description optionnelle       |
| `user_id`     | INTEGER       | FK → user.id   | Propriétaire du projet        |
| `created_at`  | TIMESTAMP     | NOT NULL       | Date de création               |
| `updated_at`  | TIMESTAMP     | NOT NULL       | Dernière mise à jour           |

---

## 3. `note`

Notes (écrites, vocales ou issues de documents) dans un projet.

| Colonne               | Type               | Contraintes    | Description |
|-----------------------|--------------------|----------------|-------------|
| `id`                  | INTEGER            | PK, auto       | Identifiant unique |
| `title`               | VARCHAR(200)       | NOT NULL       | Titre de la note |
| `content`             | TEXT               | NULL           | Contenu texte (optionnel pour documents) |
| `note_type`           | VARCHAR            | NOT NULL, défaut `'written'` | `written`, `voice` ou `document` |
| `source_file_path`    | VARCHAR            | NULL           | Chemin du fichier source si document uploadé |
| `processing_status`   | VARCHAR            | NOT NULL, défaut `'completed'` | `pending`, `processing`, `completed`, `failed` |
| `processing_progress` | INTEGER            | NULL, défaut 100 | Progression du traitement (0–100) |
| `project_id`          | INTEGER            | FK → project.id | Projet parent |
| `user_id`             | INTEGER            | FK → user.id    | Créateur |
| `created_at`          | TIMESTAMP          | NOT NULL       | Date de création |
| `updated_at`          | TIMESTAMP          | NOT NULL       | Dernière mise à jour |
| `embedding`           | VECTOR(1024)      | NULL           | Embedding note globale (hérité/optionnel) |

---

## 4. `notechunk`

Segments (chunks) d’une note pour le RAG et la recherche sémantique.

| Colonne         | Type           | Contraintes    | Description |
|-----------------|----------------|----------------|-------------|
| `id`            | INTEGER        | PK, auto       | Identifiant unique |
| `note_id`       | INTEGER        | FK → note.id, index | Note parente |
| `chunk_index`   | INTEGER        | NOT NULL, défaut 0 | Position dans la note (0, 1, 2…) |
| `content`       | TEXT           | NOT NULL       | Texte du chunk |
| `text`          | TEXT           | NULL           | Alias textuel (compatibilité vector stores) |
| `embedding`     | VECTOR(1024)   | NULL           | Embedding pgvector |
| `start_char`    | INTEGER        | NOT NULL, défaut 0 | Début dans le texte original |
| `end_char`      | INTEGER        | NOT NULL, défaut 0 | Fin dans le texte original |
| `node_id`       | VARCHAR(255)   | NULL, index    | ID nœud (hiérarchie LlamaIndex) |
| `parent_node_id`| VARCHAR(255)   | NULL, index    | Parent dans la hiérarchie |
| `is_leaf`       | BOOLEAN        | NOT NULL, défaut true, index | Nœud feuille ou non |
| `hierarchy_level` | INTEGER      | NOT NULL, défaut 0, index | Niveau dans l’arbre |
| `metadata_json` | JSONB          | NULL           | Métadonnées (headings, page_no, etc.) |
| `metadata_`     | JSONB          | NULL           | Métadonnées alternatives |

---

## 5. `conversation`

Conversations de chat (historique par utilisateur / projet).

| Colonne       | Type          | Contraintes       | Description                    |
|---------------|---------------|-------------------|--------------------------------|
| `id`          | INTEGER       | PK, auto          | Identifiant unique             |
| `title`       | VARCHAR(200)  | NOT NULL, défaut `'Nouvelle conversation'` | Titre de la conversation |
| `user_id`     | INTEGER       | FK → user.id      | Utilisateur                     |
| `project_id`  | INTEGER       | FK → project.id, NULL | Projet lié (optionnel)     |
| `created_at`  | TIMESTAMP     | NOT NULL          | Date de création                |
| `updated_at`  | TIMESTAMP     | NOT NULL          | Dernière mise à jour            |

---

## 6. `message`

Messages d’une conversation (tour par tour user / assistant).

| Colonne           | Type          | Contraintes              | Description                    |
|-------------------|---------------|---------------------------|--------------------------------|
| `id`              | INTEGER       | PK, auto                  | Identifiant unique             |
| `conversation_id` | INTEGER       | FK → conversation.id, CASCADE | Conversation parente       |
| `role`            | VARCHAR(50)   | NOT NULL                  | `user`, `assistant` ou `system` |
| `content`         | TEXT          | NOT NULL                  | Contenu du message             |
| `model`           | VARCHAR(100)  | NULL                      | Modèle utilisé (réponses assistant) |
| `provider`        | VARCHAR(50)   | NULL                      | Fournisseur (mistral, openai…) |
| `created_at`      | TIMESTAMP     | NOT NULL                  | Date d’envoi                   |

---

## 7. `agent`

Agents IA personnalisés (system prompt / personnalité) par utilisateur.

| Colonne       | Type          | Contraintes    | Description                    |
|---------------|---------------|----------------|--------------------------------|
| `id`          | INTEGER       | PK, auto       | Identifiant unique             |
| `user_id`     | INTEGER       | FK → user.id   | Propriétaire                   |
| `name`        | VARCHAR(200)  | NOT NULL       | Nom de l’agent                 |
| `personality` | TEXT          | NOT NULL       | System prompt / personnalité   |
| `model_preset`| VARCHAR(50)   | NULL           | Legacy (utiliser `fast`) |
| `created_at`  | TIMESTAMP     | NOT NULL       | Date de création               |
| `updated_at`  | TIMESTAMP     | NOT NULL       | Dernière mise à jour           |

---

## 8. `agenttask`

Tâches associées à un agent (objectif / instruction).

| Colonne       | Type     | Contraintes              | Description                    |
|---------------|----------|---------------------------|--------------------------------|
| `id`          | INTEGER  | PK, auto                  | Identifiant unique             |
| `agent_id`    | INTEGER  | FK → agent.id, CASCADE    | Agent parent                   |
| `name`        | VARCHAR(200) | NOT NULL              | Nom de la tâche                |
| `instruction`| TEXT     | NOT NULL                  | Prompt / objectif de la tâche  |
| `created_at`  | TIMESTAMP| NOT NULL                  | Date de création               |

---

## 9. `scheduledjob`

Tâches planifiées (scheduler) : exécution d’un agent sur des tâches à horaires définis.

| Colonne          | Type       | Contraintes    | Description                    |
|------------------|------------|----------------|--------------------------------|
| `id`             | INTEGER    | PK, auto       | Identifiant unique             |
| `user_id`        | INTEGER    | FK → user.id   | Propriétaire                   |
| `agent_id`       | INTEGER    | FK → agent.id  | Agent à exécuter               |
| `task_ids`       | JSON       | NOT NULL       | Liste d’IDs de tâches (agenttask) |
| `cron_expression`| VARCHAR(100) | NOT NULL    | Expression cron (ex. `0 18 * * *`) |
| `schedule_hour`  | INTEGER    | NOT NULL       | Heure (0–23)                   |
| `schedule_minute`| INTEGER    | NOT NULL       | Minute (0–59)                  |
| `schedule_days`  | JSON       | NULL           | Jours (0–6, 0 = Lundi)        |
| `enabled`        | BOOLEAN    | NOT NULL, défaut true | Job actif ou non        |
| `last_run_at`    | TIMESTAMP  | NULL           | Dernière exécution             |
| `created_at`     | TIMESTAMP  | NOT NULL       | Date de création               |
| `updated_at`     | TIMESTAMP  | NOT NULL       | Dernière mise à jour           |

---

## 10. `taskrunlog`

Historique d’exécution des tâches planifiées.

| Colonne           | Type        | Contraintes              | Description                    |
|-------------------|-------------|---------------------------|--------------------------------|
| `id`              | INTEGER     | PK, auto                  | Identifiant unique             |
| `scheduled_job_id`| INTEGER     | FK → scheduledjob.id      | Job qui a été exécuté          |
| `agent_task_id`   | INTEGER     | FK → agenttask.id         | Tâche exécutée                 |
| `task_name`       | VARCHAR(200)| NOT NULL                  | Nom dénormalisé (affichage)    |
| `output`          | TEXT        | NULL                      | Sortie de l’exécution         |
| `error`           | TEXT        | NULL                      | Message d’erreur éventuel     |
| `run_at`          | TIMESTAMP   | NOT NULL                  | Date/heure d’exécution        |
| `user_id`         | INTEGER     | FK → user.id              | Utilisateur concerné           |

---

## 11. `knowledgeentity`

Entités de connaissance (KAG) : concepts, équipements, procédures, paramètres, etc., extraits des chunks.

| Colonne         | Type          | Contraintes    | Description                    |
|-----------------|---------------|----------------|--------------------------------|
| `id`            | INTEGER       | PK, auto       | Identifiant unique             |
| `name`          | VARCHAR(500)  | NOT NULL, index| Nom (ex. « Pompe Centrifuge »)  |
| `name_normalized` | VARCHAR(500) | NOT NULL, index | Version normalisée (déduplication) |
| `entity_type`   | VARCHAR(100)  | NOT NULL, index| Type : equipement, procedure, parametre… |
| `project_id`    | INTEGER       | FK → project.id, CASCADE, index | Projet (isolation)   |
| `mention_count` | INTEGER       | NOT NULL, défaut 1 | Nombre d’occurrences      |
| `embedding`     | VECTOR(1024)  | NULL           | Embedding pgvector             |
| `created_at`    | TIMESTAMP     | NOT NULL       | Date de création               |
| `updated_at`    | TIMESTAMP     | NOT NULL       | Dernière mise à jour           |

Index unique : `(project_id, name_normalized)`.

---

## 12. `chunkentityrelation`

Relation entre un chunk et une entité de connaissance (KAG) : « ce chunk mentionne cette entité ».

| Colonne          | Type     | Contraintes                    | Description                    |
|------------------|----------|--------------------------------|--------------------------------|
| `id`             | INTEGER  | PK, auto                       | Identifiant unique             |
| `chunk_id`       | INTEGER  | FK → notechunk.id, CASCADE, index | Chunk concerné             |
| `entity_id`      | INTEGER  | FK → knowledgeentity.id, CASCADE, index | Entité concernée        |
| `relevance_score`| FLOAT    | NOT NULL, défaut 1.0           | Pertinence (0.0–1.0, ex. LLM) |
| `project_id`     | INTEGER  | NOT NULL, index                | Dénormalisé pour les requêtes  |
| `created_at`     | TIMESTAMP| NOT NULL                       | Date de création               |

Contrainte unique : `(chunk_id, entity_id)`.

---

## Résumé des relations

- **user** → project, note, conversation, agent, scheduledjob, taskrunlog  
- **project** → note, conversation, knowledgeentity  
- **note** → notechunk  
- **notechunk** → chunkentityrelation  
- **conversation** → message  
- **agent** → agenttask, scheduledjob  
- **agenttask** → taskrunlog  
- **knowledgeentity** → chunkentityrelation  

Les vecteurs (`embedding`) utilisent **pgvector** avec une dimension de **1024** (configurable via `EMBEDDING_DIMENSION`).

Note : la recherche sémantique RAG s’appuie principalement sur `notechunk.embedding` (chunks), pas sur `note.embedding`.
