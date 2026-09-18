# Plan — LIA sur le wiki : fin du retriever, chat CAG avec Small 4

Date : 2026-09-18 · Branche : `feature/wiki` · Statut : **L0 à L5 livrés le 18/09 (non commité)**, L6 déploiement à faire.

> **Résultat L5 (golden 62 questions Perform76, runner `app/scripts/eval_golden_generation.py`, 1 essai)** : **83,9 %** (49 justes + 3 abstentions correctes, 8 faux, 2 ambigus, 0 erreur) — le même chiffre que la référence de l'ancien système (83,9 % image-only du 14/09), avec un premier token médian de **3,5 s** (p90 5,1 s) contre ~25 s, un tour complet médian de 4,2 s, 185 507 tokens de prompt dont **96,8 % en cache**. Document de preuve cité dans 72,6 % des tours (le golden compte des pages PDF ; le wiki cite des pages markdown issues du PDF). Une seule citation vers une page inexistante sur 62. Les échecs se concentrent sur le tableau des hauteurs de poignée (6 questions, 33 %) et sur deux cotes de parclose ; plusieurs sont des artefacts de notation (« 2,0 mm » compté faux pour « 2 ») ou renvoient à ce que le wiki écrit : ce sont des **corrections du wiki** à faire (page poignée et pivot, parclose 76503), pas des mécanismes.

> **Décisions d'Elie (18/09)** : PDF de `raw/` hors git, `.md` du wiki versionnés ; espaces supprimés (réutilisation éventuelle plus tard via les métadonnées des pages) ; arbres SAV supprimés ; une page chat + une page wiki.
>
> **Résultat L0 (mesuré sur trois appels réels, `mistral-small-latest`, `reasoning_effort: high`)** : prompt de 542 491 caractères → **185 508 tokens** (2,924 car./token, le ratio du prototype se confirme) ; cache Mistral à **0 %** aux deux premiers appels puis **99,98 %** au troisième (185 472 / 185 504) ; latence 15,5 s → 5,9 s en cache ; réponse juste (parclose 76507 pour 44 mm en ouvrant). Verdict : `sources/` reste dans le prompt (décision 6), il reste ~70 k tokens de fenêtre.

> **Verdict en quatre lignes.** Le dossier `wiki_llm/` contient déjà les trois pièces du système cible : un wiki OKF de 73 pages écrit et relu à la main, un prototype de chat CAG (`chat/server.py`, 260 lignes) qui met **tout** le wiki dans le prompt système avec `prompt_cache_key`, et un visualiseur de graphe (`graph/index.html`). La refonte consiste à porter ces trois pièces dans FastAPI (auth, conversations, feedback, RBAC conservés) et à supprimer tout ce que le wiki remplace : bibliothèque, espaces, ingestion, ColPali, BM25, KAG, élection, packing, arbres SAV — soit ~45 000 des 57 000 lignes de `app/`. Un point change le plan avant d'écrire une ligne : le wiki a grossi depuis le README de `wiki_llm/` (« ~132 500 tokens ») ; il pèse aujourd'hui **551 000 caractères hors journal, soit 157 000 à 188 000 tokens** selon le ratio retenu — la première tâche est de **mesurer** sur un appel réel.

---

## 1. Constat

### 1.1 Ce que contient `wiki_llm/`

| Pièce | Contenu | Réutilisation |
|---|---|---|
| `wiki/` | 75 fichiers `.md` : 73 pages concept + `index.md` + `log.md`. 12 types (21 `Document source`, 10 `Quincaillerie`, 10 `Profilé`, 8 `Gamme`, 5 `Procédure`, 5 `Fournisseur`, 4 `Porte d'entrée`, 3 `Anomalie`, 2 `Équipement`, 2 `Vitrage`, 2 `Certification`, 1 `Garantie`). 40 `stable`, 33 `draft`. Frontmatter homogène (`type`, `title`, `description`, `tags`, `status`, `sources`, `generated`, `stale_after` sur 69 pages). | **La source unique de LIA**, telle quelle. |
| `raw/` | 30 PDF, 282 Mo, le plus gros 35 Mo. Cités par les pages sous la forme `(schéma: raw/x.pdf, p. 5)`. | Servis par l'app pour ouvrir la planche citée. |
| `chat/server.py` | Prototype CAG : consignes (12 règles) + wiki entier dans le prompt système, `mistral-small-latest`, `reasoning_effort: high`, `temperature 0.2`, `max_tokens 4096`, `prompt_cache_key = sha256(prompt entier)`, SSE. | La **logique** est portée dans un service FastAPI ; le serveur `http.server` disparaît. |
| `graph/build_graph.py` | Parse le frontmatter, extrait les liens `[..](/x.md)`, crée les nœuds fantômes (liens vers pages non écrites), calcule degrés, orphelines, périmées → `graph.json` (658 Ko, corps inclus). | Porté en Python dans le service wiki ; plus de fichier `graph.json`. |
| `graph/index.html` | Visualiseur d3 à la Obsidian : lobes par type, recherche Ctrl+K, arbre des pages, panneau de lecture (markdown rendu, sommaire, rétroliens, historique), thème clair/sombre. 1 230 lignes autonomes. | Devient un template de l'app, données servies par l'API. |
| `CLAUDE.md` | Conventions d'écriture du wiki (OKF, tableaux de cotes, registres d'anomalies, citations, ingestion par Claude Code). | Inchangé : c'est le protocole d'alimentation du wiki, hors app. |
| `a_faire/` | 3 PDF en attente d'ingestion. | File d'attente humaine, inchangée. |

### 1.2 Le budget de contexte — le chiffre qui conditionne tout

| Périmètre | Caractères | Tokens à 3,5 car./tok | Tokens à 2,93 car./tok (ratio mesuré par le prototype) |
|---|---|---|---|
| Wiki entier | 577 939 | 165 000 | 197 000 |
| **Hors `log.md`** (journal, 27 214 car., aucune valeur de réponse) | **550 725** | **157 000** | **188 000** |
| Hors `log.md` et hors `sources/` (21 pages, 128 188 car.) | 422 537 | 121 000 | 144 000 |

Fenêtre Small : 256 000 tokens. Avec le wiki hors journal il reste **68 000 à 99 000 tokens** pour l'historique, le raisonnement et la réponse — tenable, mais le README du prototype (« 52 % de la fenêtre ») est périmé : l'ingestion du tome 2 profine le 18/09 a ajouté trois pages de cotes et d'abaques. Le plafond de « ~30 documents sources » annoncé par le prototype est plus proche de **25**. Aucun mécanisme n'est proposé ici pour le dépasser ; c'est une décision à prendre quand la mesure le dira (§ 7).

### 1.3 Ce que l'app fait aujourd'hui et qui devient inutile

`stream_space_chat_message` (`app/routers/chat.py`, 2 881 lignes) enchaîne : compréhension de la question (appel LLM), périmètre, arbre SAV, ancre conversationnelle, retrieval ColPali + BM25, RRF, boost catégorie, reranker, élection de documents, packing CAG de pages PNG + markdown, génération, contrôle d'ancrage, recalage des sources, illustrations. Tout l'amont de la génération disparaît ; la génération elle-même est réécrite (le pack devient le wiki, les sources deviennent des pages).

Ce qui est **indépendant du retrieval** et reste : authentification et cookies, RBAC (`config.manage_users`, `config.manage_roles`), conversations et messages, feedback 👍/👎 avec catégorie, journal d'audit admin, client Mistral (`mistral_service.chat_stream` accepte déjà `**kwargs` → `prompt_cache_key` et `reasoning_effort` passent sans modification ; il gère déjà les blocs `thinking`), notification Discord, mise en page `base.html`.

---

## 2. Cible

### 2.1 Principes

- **Le wiki est la seule source.** Aucun texte indexé, aucun PNG de page, aucun retriever : le prompt système = consignes + wiki, la question arrive après. C'est la règle « page + question, point » appliquée au wiki entier.
- **Décision par le code, pas par un juge.** Le seul contrôle de sortie est déterministe : les chemins de pages cités existent-ils ? (le prototype a mesuré des « citations fabriquées »).
- **Pas de drapeau.** Le nouveau tour de chat remplace l'ancien ; le code remplacé est supprimé dans le même lot, pas désactivé.
- **Le wiki s'écrit hors de l'app.** Ingestion et corrections passent par Claude Code + git (protocole `wiki_llm/CLAUDE.md`). L'app **lit** le dossier et se recharge quand il change ; elle n'écrit jamais dedans.

### 2.2 Architecture

```
wiki_llm/wiki/*.md ──► WikiSnapshot (chargé au démarrage, rechargé si un mtime change)
                          ├─ pages (frontmatter + corps + liens sortants), fantômes, degrés, orphelines, périmées
                          ├─ prompt système = consignes + index.md + registres d'anomalies + pages (hors log.md)
                          └─ clé de cache = sha256(prompt système)

POST /api/chat/stream ──► messages = [system: prompt] + historique (≤ 10 msgs, ≤ 24 000 car.) + question
                          └─► mistral_service.chat_stream(model=MODEL_FAST, reasoning_effort=high,
                                temperature=0.2, max_tokens=4096, prompt_cache_key=clé)
                                → SSE thinking / message → citations extraites par regex → sources → done

GET /api/wiki/graph, /api/wiki/pages/{chemin}, /api/wiki/raw/{fichier}, /api/wiki/stats
```

### 2.3 Pages de l'application

| URL | Contenu |
|---|---|
| `/` | **Le chat** : liste des conversations à gauche (comme aujourd'hui), fil au centre, **panneau lecteur** à droite qui s'ouvre au clic sur une source et affiche la page du wiki rendue (marked + DOMPurify, déjà chargés). Les liens `/x.md` dans la page naviguent dans le panneau ; les `(schéma: raw/x.pdf, p. N)` ouvrent le PDF à la page N dans la modale pdf.js déjà présente. Bouton « Voir dans le graphe ». |
| `/wiki` | **Le graphe**, port de `graph/index.html` : mêmes lobes, recherche, arbre, panneau de lecture, thème. Les données viennent de `/api/wiki/graph` (sans les corps : ~60 Ko au lieu de 658) et les corps sont chargés à l'ouverture d'une page. Le hash d'URL `#/profiles/x.md` ouvre la page directement (lien profond depuis le chat). Barre de navigation de l'app ajoutée (Chat · Wiki · Admin · Déconnexion). |
| `/admin` | Utilisateurs, rôles, permissions, feedbacks, conversations (existant, épuré) + **carte « Wiki »** : pages, liens, orphelines, fantômes, périmées, brouillons, caractères, tokens estimés, **tokens du dernier appel réel** et part en cache, clé de cache, date du dernier rechargement. |
| `/spaces/{id}`, `/library`, `/admin/sav-trees` | Redirection vers `/`, puis suppression des routes au lot L4. |

---

## 3. Le tour de chat en détail

### 3.1 Construction du prompt

1. **Consignes** : le texte `CONSIGNES` du prototype, déplacé dans `app/prompts/wiki_consignes.md` (ainsi modifier une consigne change le hash, donc la clé de cache — exactement le comportement voulu par le prototype).
2. **`index.md`** en premier (table des matières, 15 700 car.).
3. **Les trois registres d'anomalies** ensuite : la consigne n° 2 impose de les consulter avant de répondre ; les placer en tête les rend faciles à retrouver.
4. **Toutes les autres pages**, triées par chemin, chacune précédée de `===== PAGE /dossier/page.md =====` (format du prototype, c'est le chemin que le modèle doit citer).
5. **`log.md` exclu** : journal d'opérations, aucune valeur de réponse, et il change à chaque édition.
6. Fin : `===== FIN DU WIKI =====`.

Le préfixe mis en cache est le prompt système seul ; l'historique de conversation vient **après** et ne casse jamais le cache. Un garde-fou au chargement : si les caractères / 2,93 dépassent 200 000 tokens estimés, un avertissement est journalisé et affiché dans la carte admin — pas de troncature silencieuse.

### 3.2 Appel et flux

Paramètres repris du prototype (les seuls réglages conservés dans `config.py` pour la génération) : `MODEL_FAST = mistral-small-latest`, `GENERATION_REASONING_EFFORT = high`, `SPACE_CHAT_TEMPERATURE = 0.2` (renommé `CHAT_TEMPERATURE`), `max_tokens 4096`.

`mistral_service.chat_stream` reçoit une seule évolution : émettre `{"usage": {...}}` en fin de flux (Mistral l'envoie dans le dernier chunk ; le service le lit déjà pour le journal mais ne le transmet pas). Ce compteur — `prompt_tokens`, `completion_tokens`, tokens en cache — est persisté dans `message.metadata_json` et affiché dans la trace du message et la carte admin. Sans lui, on ne sait ni si le cache fonctionne ni où en est le budget.

### 3.3 Événements SSE

Le front actuel sait déjà lire ces événements ; on garde le vocabulaire et on retire le reste.

| Événement | Sort |
|---|---|
| `stage` | Conservé, une seule étape : « Lecture du wiki ». |
| `thinking` | Conservé (raisonnement Small, déjà géré par `chat_stream`). |
| `message` | Conservé, streamé au fil de l'eau, **sans filtre de balises** (`<sources>` / `<evidence>` n'existent plus : la citation est le chemin dans le texte). |
| `sources` | Conservé, nouveau contenu : `[{path, title, type, status, exists}]`. |
| `done` | Conservé : `{message_id, trace}` avec `trace = {model, prompt_tokens, cached_tokens, completion_tokens, first_token_ms, duration_ms, wiki_hash, unknown_citations}`. |
| `error` | Conservé. |
| `slot_prompt`, `step`, `scope_proposal`, `sav_suggestion`, `status: cropping` | Supprimés avec leurs fonctions. |

### 3.4 Sources et contrôle par le code

- Après la fin du flux, une regex extrait les chemins `/dossier/page.md` du texte. Chaque chemin est résolu dans le snapshot : trouvé → chip cliquable (titre, type) ; **introuvable → chip rouge « page inexistante »**, chemin consigné dans `unknown_citations`. C'est la réponse déterministe à la défaillance « citations fabriquées » mesurée par le prototype : constater et montrer, sans réécrire la réponse.
- Les identifiants d'anomalie (`INC-07`, `CTR-12`, `VER-35`) présents dans la réponse sont transformés en liens vers le registre correspondant (préfixe → page), la ligne étant surlignée à l'ouverture. Zéro appel, une regex.
- `message.sources` (colonne JSON existante) reçoit la liste des pages ; `message.metadata_json` reçoit la trace. Les anciens messages gardent leurs sources documentaires, qui s'affichent en texte inerte (voir décision 4).

### 3.5 Ce qui n'existe plus dans le tour

Compréhension de la question par LLM, périmètre, ancre conversationnelle, retrieval, élection, packing, PNG de pages, contrôle d'ancrage par corpus de preuve, recalage des pages, illustrations recadrées, FAQ correctives, arbres SAV, repli eco/minimal sur erreur 400 (le prompt a une taille connue et fixe : s'il ne passe pas, c'est le budget du § 1.2 qu'il faut traiter, pas un repli).

---

## 4. Le wiki dans l'app

### 4.1 `app/services/wiki_service.py` (port de `build_graph.py` + `build_context()`)

- `WikiPage` : `path` (identité OKF, ex. `/profiles/perform76-parcloses.md`), `title`, `type`, `description`, `status`, `tags`, `stale_after`, `stale`, `sources` (titres), `folder`, `body` (markdown sans frontmatter), `out_links`, `in_degree`, `out_degree`, `reserved` (index/log), `missing` (fantôme).
- `WikiSnapshot` : pages, liens (`source, target, count`), `system_prompt`, `cache_key`, `char_count`, `estimated_tokens`, `loaded_at`, `lint` (orphelines, fantômes, périmées, brouillons, frontmatter illisible, `type` vide, pages absentes de `index.md`).
- Rechargement : au démarrage, puis à chaque requête on compare le max des mtime des `.md` (75 `stat`, négligeable) ; s'il a changé, on reconstruit. Pas de bouton, pas de tâche planifiée.
- Dépendance ajoutée : `PyYAML` (déjà utilisé par le prototype).

### 4.2 API (`app/routers/wiki.py`, authentifié comme le reste)

| Route | Réponse |
|---|---|
| `GET /api/wiki/graph` | `{generated, nodes: [sans body], links}` — le contrat de `graph.json` moins les corps. |
| `GET /api/wiki/pages/{chemin}` | `{page (frontmatter), body_markdown, in_links, out_links}`. Le chemin est cherché **dans le dictionnaire du snapshot**, jamais résolu sur disque : aucune traversée de répertoire possible. |
| `GET /api/wiki/index` | `index.md` brut (sommaire du panneau lecteur). |
| `GET /api/wiki/raw/{fichier}` | Le PDF, en flux, si `fichier` figure dans la liste des fichiers de `raw/` (même règle : liste blanche, pas de chemin). `#page=N` côté client. |
| `GET /api/wiki/stats` | La carte admin du § 2.3 (réservée `config.manage_users`). |

### 4.3 Port du graphe

Le CSS et la logique d3 de `graph/index.html` sont repris tels quels dans `app/templates/wiki_graph.html` ; trois changements : `d3.json("graph.json?v=…")` → `fetch("/api/wiki/graph")` ; `renderPeek` charge le corps via `/api/wiki/pages/…` au lieu de `n.body` ; `open(id)` écrit `location.hash` et le hash est lu au chargement. Le thème sombre du graphe suit la classe `dark` de `base.html` au lieu de son propre `localStorage`.

---

## 5. Ce qui est supprimé

| Couche | Supprimé | Conservé |
|---|---|---|
| Routers | `library.py` (2 314 l.), `spaces.py` (461), `guided_trees.py` (738), `chat.py` réécrit (2 881 → ~250), `admin.py` épuré (config RAG, files d'attente, documents bloqués, éval retriever/RAG/génération, catégories, gammes : ~700 l. retirées) | `auth.py`, `conversations.py` (moins `space_id`) |
| Services | 70 des 85 fichiers : tout `chunk*`, `colpali*`, `lancedb*`, `bm25*`, `lexical_search*`, `page_retrieval*`, `page_reranker*`, `reranker*`, `retrieval_boost*`, `document_*` (7), `kag_*` (3), `context_packer*`, `document_election*`, `rag_generation*`, `answer_generation*`, `stream_source_filter*`, `response_verification*`, `lightweight_query_understanding`, `query_*schemas`, `slot_catalog`, `scope_resolver*`, `fiche_technique*`, `reference_*` (2), `illustration*`, `multimodal_page*`, `vision_page_extraction*`, `text_page_extraction*`, `page_markdown*`, `pdf_extraction*`, `mistral_ocr*`, `ocr_preprocessing`, `file_conversion`, `embedding_service`, `guided_*` (7), `authored_tree_*` (2), `sav_extraction*`, `space_*` (3), `folder_service`, `library_service`, `category_catalog`, `theme_tree_catalog`, `gamme_*` (2), `feedback_knowledge*`, `coverage_service`, `conversation_state*`, `chat_critique*`, `chat_tools`, `brave_search*`, `ollama_service`, `openai_service`, `task_dispatch`, `celery_queue_health`, `indexing_health*`, `retriever_evaluator`, `generation_eval*`, `eval_datasets*`, `document_processing_snapshot`, `document_run`, `bm25_thesaurus` | `auth_service`, `authorization_service`, `rbac_seed_service` (permissions bibliothèque/espaces/FAQ retirées), `admin_audit_service`, `mistral_service` (moins l'import de `chat_tools`, plus l'événement `usage`), `discord_service` ; **nouveaux** : `wiki_service`, `wiki_chat_service` |
| Modèles | `Document`, `DocumentChunk`, `DocumentSpace`, `Library`, `Folder`, `Space`, `KnowledgeEntity` + `ChunkEntityRelation` + `EntityAlias` + `EntityEntityRelation`, `DocumentCategory`, `ChunkCategoryRelation`, `SpaceThemeSynthesis`, `CategoryCandidate`, `GuidedSession`, `GuidedTree`, `GuidedTreeNode`, `GuidedTreeVersion`, `GuidedNodeAttachment`, `GuidedEntryIndex`, `GuidedSymptomAlias`, `GuidedGap`, `GammeCommerciale` | `User`, `Role`, `Permission`, `UserRole`, `RolePermission`, `Conversation` (sans `space_id`), `Message`, `MessageFeedback` (sans `space_id`, `chunk_ids`, `auto_faq_*`), `AdminAuditLog` |
| Base | Une migration `wiki_refonte_drop_retrieval` : `DROP TABLE IF EXISTS … CASCADE` sur les ~25 tables ci-dessus, `ALTER TABLE conversation DROP COLUMN IF EXISTS space_id`, idem `message_feedback` (`space_id`, `chunk_ids`, `auto_faq_generated`, `auto_faq_content`), `DROP EXTENSION IF EXISTS vector`. Idempotente (règle de la maison : `create_all` tourne aussi au démarrage). La chaîne Alembic existante est conservée telle quelle. | — |
| Tâches | `tasks/`, `celery_app.py`, `embedding_config.py`, `library_document_logging.py`, `tracing.py` (LangSmith), `catalog/`, `brain/`, `scripts/` (sondes, seeds, éval — voir L5 pour le runner golden) | `logging_config.py` |
| Config | 202 réglages → ~20 : base, secret, cookies, CORS, Mistral (clé, URL, retries), `MODEL_FAST`, `GENERATION_REASONING_EFFORT`, `CHAT_TEMPERATURE`, `CHAT_MAX_TOKENS`, `CHAT_HISTORY_MAX_MESSAGES`, `CHAT_HISTORY_MAX_CHARS`, `WIKI_DIR` (défaut `wiki_llm`), `ADMIN_EMAIL`, `DISCORD_WEBHOOK_URL`. `feature_summary()` et `coherence_warnings()` disparaissent. | |
| Docker | Services `worker` et `redis` ; dans l'image : LibreOffice, poppler, torch, `colpali-engine`, `transformers`, `sentence-transformers`, `llama-index`, `lancedb`, `pyarrow`, `pgvector`, `pymupdf4llm`, `pdf2image`, `mistralai`, `celery`, `redis`, `langgraph`, `langchain-core`, `langsmith`, le pré-téléchargement du reranker. Image de plusieurs Go → ~250 Mo, build de ~20 min → ~1 min. Bloc `x-app-env` ramené aux ~20 variables. Volume ajouté : `./wiki_llm:/app/wiki_llm:ro`. | `db` (image `pgvector/pgvector:pg15` conservée pour ne pas toucher au volume ; l'extension est simplement retirée par la migration), `web` |
| Templates | `library.html` (3 460 l.), `space_detail.html` (4 209, remplacé par `chat.html`), `home_spaces.html`, `admin_sav_trees.html`, `project_detail.html`, `note_edit.html`, `index.html` (ces trois derniers ne sont plus référencés par aucune route) ; `admin.html` épuré | `base.html`, `login.html`, `register.html`, `feedbacks.html`, `admin.html` ; **nouveaux** : `chat.html`, `wiki_graph.html` |
| Statique | `page-viewer.js` (visionneuse PDF + texte extrait ; la partie pdf.js est réutilisée pour `raw/`) | |
| Tests | ~82 des 90 fichiers (tout ce qui importe retrieval, chunks, KAG, guidé, bibliothèque, espaces, extraction, éval) | `test_auth_redirects`, `test_conversations_chat` (adapté), `test_rbac_library_spaces` → `test_rbac` (cas espaces/bibliothèque retirés), `test_discord_notification`, `test_mistral_tool_protocol` (`_clean_messages`) ; **nouveaux** : § 6, lot L1–L2 |
| Données locales | `data/` (LanceDB, 6,4 Go) et `media/` (299 Mo) n'ont plus de consommateur : suppression après sauvegarde, hors git de toute façon. Le `.env` perd ~110 variables mortes. | |
| `wiki_llm/` | `chat/`, `graph/`, `requirements.txt`, `.claude/launch.json` (les deux serveurs autonomes) ; `README.md` réécrit (le wiki est servi par LIA) | `wiki/`, `raw/`, `a_faire/`, `CLAUDE.md` |

Ordre de grandeur après refonte : `app/` passe de ~57 000 à ~12 000 lignes, dont ~7 000 pour `admin.html` + `base.html` + le graphe.

---

## 6. Lots, dans l'ordre

Chaque lot se termine sur la suite de tests verte (`docker compose exec web pytest`) ; rien n'est commité tant que l'app ne répond pas de bout en bout.

**L0 — Mesurer avant de construire (½ journée).** Un script jetable construit le prompt système exactement comme au § 3.1 et fait **deux appels réels** identiques sur une question du golden. On lit `usage.prompt_tokens` (le vrai budget, qui tranche entre 157 k et 188 k), les tokens en cache au second appel (le cache fonctionne-t-il ?), le délai du premier token avec `reasoning_effort: high`. Trois issues : sous 190 k → wiki entier hors journal, comme prévu ; entre 190 et 230 k → `sources/` sort du prompt (décision 6) ; au-delà → on s'arrête et on décide (§ 7, risque 1). Résultat consigné en tête de ce document.

**L1 — Module wiki (1 jour).** `wiki_service.py` + `routers/wiki.py` + tests : frontmatter et corps parsés sur les 75 fichiers réels, liens et fantômes identiques à ce que produit `build_graph.py` aujourd'hui (comparaison avec `graph/graph.json` comme oracle, une fois), orphelines/périmées/brouillons, clé de cache stable entre deux chargements et différente après modification d'un caractère, `log.md` absent du prompt, `index.md` et registres en tête, chemin inconnu → 404, `../` → 404, PDF hors liste → 404, rechargement sur mtime. Aucun changement visible dans l'app à ce stade.

**L2 — Le tour de chat (1 jour).** `wiki_chat_service.py` (messages, appel, extraction des citations, trace) + `POST /api/chat/stream` + événement `usage` dans `mistral_service` + persistance. Tests avec un `chat_stream` simulé : ordre des événements, `sources` avec `exists` vrai/faux, `unknown_citations` dans la trace, historique borné, message utilisateur et réponse persistés avec leurs métadonnées, erreur Mistral → événement `error` et rien de persisté.

**L3 — Le front (1,5 jour).** `chat.html` dérivé de `space_detail.html` (on retire espaces, SAV, périmètre, slots, illustrations, trace de retrieval ; on ajoute le panneau lecteur, les chips de pages, les liens d'anomalies, la modale PDF sur `raw/`) ; `wiki_graph.html` (§ 4.3) ; `/` sert le chat ; carte « Wiki » dans `admin.html`. Vérification dans le navigateur intégré : question → réponse streamée → clic sur une source → page dans le panneau → clic sur un schéma → PDF à la bonne page → « Voir dans le graphe » → nœud centré.

**L4 — Suppression (1 jour).** Tout le § 5 : fichiers, modèles, migration, config, compose, Dockerfile, requirements, tests, `.env.example`, README racine, CLAUDE.md racine (absent aujourd'hui : en créer un court qui renvoie à `wiki_llm/CLAUDE.md` pour l'alimentation du wiki). Reconstruction de l'image, `alembic upgrade head` sur une copie de la base de prod, suite de tests verte, smoke test des pages.

**L5 — Mesurer après (½ journée).** Le golden `tests/fixtures/golden/space29_perform76_generation.json` (62 questions, valeurs attendues typées) est l'actif d'évaluation à conserver. Le runner `eval_golden_generation.py` est porté : il vise `POST /api/chat/stream`, garde ses fonctions pures de notation (`value_present`, verdicts `juste / ambigu / faux / abstention_ok / abstention_ko`) et remplace `page_citee` par « une page du wiki dont le frontmatter `sources` référence le PDF de preuve est citée » (les identifiants de documents du golden sont mis en correspondance avec les fichiers de `raw/` par une table de 3 lignes : cahier technique Perform76, DTA 6/16-2334, mise en œuvre profine). Référence à battre : **83,9 % sur 62** (image-only sans outils, 14/09) et 94,3 % sur 35 (17/09 matin). Le prototype n'a été évalué que sur 10 questions (5 justes, 4 partielles, 1 fausse) : c'est ce lot qui dit si le wiki tient la promesse.

**L6 — Déploiement (½ journée).** Sur le VPS : `git pull`, `docker compose build`, `docker compose up -d` (le `worker` et `redis` s'arrêtent), migration appliquée par la commande du conteneur `web`. Nginx : les délais relevés pour l'export/import ne servent plus, on garde ceux du SSE. Déployer une modification du wiki = `git pull` (l'app recharge au prochain appel ; le premier appel après un changement paie le prompt plein tarif, c'est attendu).

Total : **6 jours** de travail, L0 en premier et bloquant.

---

## 7. Décisions en attente

| # | Question | Recommandation |
|---|---|---|
| 1 | Où vit le wiki dans le dépôt ? | **Garder `wiki_llm/` comme racine de connaissance** (`wiki/`, `raw/`, `a_faire/`, `CLAUDE.md`), lue via `WIKI_DIR`. Zéro lien cassé, `CLAUDE.md` reste valable tel quel pour les sessions d'ingestion. Un renommage (`connaissance/`) est possible mais n'apporte rien. |
| 2 | Les 282 Mo de PDF de `raw/` : dans git ? | **Oui, en git ordinaire** : immuables, le plus gros fait 35 Mo (limite GitHub 100 Mo par fichier), et `git pull` déploie alors tout d'un coup. Alternative si le poids du dépôt gêne : Git LFS. À trancher avant le premier commit de `wiki_llm/`. |
| 3 | Arbres SAV (diagnostic guidé) | **Supprimer.** Ils s'appuient sur des pages de documents de la bibliothèque (pièces jointes, vignettes) qui disparaissent ; le wiki a un dossier `procedures/` qui couvre le même besoin par la conversation. Les remonter sur le wiki serait un projet à part. |
| 4 | Conversations et feedbacks existants | **Conserver.** Les colonnes `space_id` sont retirées, les messages restent ; leurs anciennes sources documentaires s'affichent en texte inerte. Une purge est possible plus tard si l'affichage gêne. |
| 5 | Espaces | **Supprimer entièrement** : un seul corpus, un seul chat. L'écran d'accueil devient le chat. |
| 6 | Pages `sources/` dans le prompt (128 000 car., 23 %) | **Garder** : elles portent la carte de chaque PDF (pagination, registres, ce qui est exploitable) et fondent la règle « le document produit prime ». Tranché par la mesure L0. |
| 7 | Identifiant de modèle | Démarrer avec `mistral-small-latest` (comme le prototype et le `.env` actuel), puis **épingler l'identifiant daté de Small 4** une fois vérifié dans la réponse `model` de l'API : un alias qui bouge change le comportement et le cache sans prévenir. |
| 8 | LangSmith | **Supprimer** : il traçait le pipeline de retrieval. La trace utile (tokens, cache, citations) est dans `metadata_json`. |
| 9 | Feedback négatif → FAQ corrective automatique | **Supprimer** le mécanisme (il créait des documents dans la bibliothèque). Le feedback reste enregistré et listé dans l'admin ; le traitement devient « corriger le wiki » via le protocole d'ingestion, ce qui est plus juste : la correction profite à toutes les questions suivantes. |

---

## 8. Risques et ce qui les surveille

1. **Le plafond de la fenêtre.** Le wiki grossit à chaque source (≈ 20 000 à 30 000 car. par PDF technique, moins pour une brochure). À 256 k, la marge est de ~25 sources contre 18 aujourd'hui. Surveillance : la carte admin et l'avertissement à 200 000 tokens estimés. Le jour où c'est atteint, le prototype suggère lui-même « noyau permanent (index + registres) + navigation vers le reste » — c'est un autre plan, à écrire à ce moment-là et pas avant.
2. **Latence.** ~180 k tokens de prompt + raisonnement `high` : même en cache, le premier token peut prendre plusieurs secondes. L0 mesure ; si c'est trop lent pour l'usage, le réglage à toucher est `reasoning_effort` (mesuré à comparer sur le golden), pas un retriever.
3. **Durée de vie du cache Mistral.** Non documentée dans le prototype ; L0 mesure le taux de hit à quelques minutes d'intervalle. Un cache froid coûte le plein tarif d'un appel, jamais une erreur.
4. **Qualité de Small 4 sur les contradictions.** Le prototype a vu les registres oubliés et des interpolations de tableau (consignes n° 2 et 4 déjà durcies en réponse). L5 mesure sur 62 questions ; les échecs deviennent des **corrections du wiki** (une page plus explicite) avant de devenir des mécanismes.
5. **Sécurité des nouveaux endpoints.** `pages/{chemin}` et `raw/{fichier}` résolvent contre des listes fermées ; tests de traversée au lot L1.
6. **Migration destructive.** Une seule migration supprime 25 tables : elle s'exécute d'abord sur une copie de la base de prod (L4), et la sauvegarde `pg_dump` précède le déploiement (L6).
