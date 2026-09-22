# LIA — l'assistant documentaire PROFERM, adossé au wiki

LIA répond aux questions métier de PROFERM Multitechniques (gammes PVC, aluminium, hybride,
profilés, quincaillerie, garanties, certifications) à partir d'un **wiki interne** écrit et relu
à la main. Pas de moteur de recherche : à chaque question, **le wiki entier** est placé dans le
contexte du modèle (Mistral Small, CAG — *Cache-Augmented Generation*), qui répond en citant
les pages dont il tire sa réponse. Chaque page citée s'ouvre dans l'interface, et le PDF source
s'ouvre à la bonne planche.

> Le nom d'image Docker du produit est `lia` ; le dépôt historique s'appelle `Noton-app`.

## Ce que fait l'application

- **Chat** (`/`) : conversations, réponse streamée avec raisonnement, pages du wiki citées en
  chips cliquables, identifiants d'anomalie (`INC-`, `CTR-`, `VER-`) liés aux registres,
  lecteur de page à droite, visionneuse PDF des documents sources, retours 👍/👎.
- **Vocal** (`/vocal`) : on parle, LIA transcrit (Voxtral Mini, biais de vocabulaire tiré du
  wiki), lit les pages avec le même tour d'outils, et répond de vive voix (Voxtral TTS, voix
  Marie) en prose courte, phrase par phrase, dès la première phrase complète. Fin de parole
  détectée dans le navigateur, mains libres, transcription surlignée au fil de la voix, pages
  citées en pastilles, retours 👍/👎. Les conversations vocales sont persistées à part.
- **Wiki** (`/wiki`) : le graphe des pages (d3, à la Obsidian) avec recherche, arbre par type,
  filtres et lecteur ; lien profond `/wiki#/dossier/page.md`.
- **Administration** (`/admin`) : utilisateurs, rôles, permissions, retours utilisateurs,
  conversations, et la carte **Wiki** (pages, liens, orphelines, périmées, brouillons, budget de
  contexte, tokens et cache du dernier appel réel).

## Le wiki : la seule source

Le dossier `wiki_llm/` est la racine de connaissance :

```
wiki_llm/wiki/       le wiki OKF v0.2 : un .md par concept, frontmatter YAML, liens /dossier/page.md
wiki_llm/raw/        les PDF sources (HORS git : copiés sur le serveur, montés en volume)
wiki_llm/a_faire/    les PDF en attente d'ingestion (hors git)
wiki_llm/CLAUDE.md   le protocole d'écriture du wiki (format des pages, tableaux de cotes,
                     registres d'anomalies, citations, ingestion)
```

**Le wiki s'écrit hors de l'application** : on dépose un PDF dans `raw/`, on demande
l'ingestion à Claude Code (protocole `wiki_llm/CLAUDE.md`), on relit, on commite les `.md`.
L'application **lit** le dossier et se recharge dès qu'un fichier change (aucun bouton, aucune
tâche). Déployer une mise à jour du wiki = `git pull` sur le serveur.

### Le prompt système

Consignes (`app/prompts/wiki_consignes.md`) + `index.md` + les trois registres d'anomalies +
toutes les pages, `log.md` exclu. La clé de cache Mistral (`prompt_cache_key`) est le sha256 du
prompt entier : toute modification du wiki ou des consignes invalide proprement le cache.

Mesuré le 18/09/2026 sur 73 pages : **542 000 caractères → 185 500 tokens** (2,92 car./token),
cache à 99,98 % dès le troisième appel, premier appel 15 s puis 6 s en cache. Fenêtre de Small :
256 k. La carte admin affiche le budget ; un avertissement est journalisé au-delà de 200 k
tokens estimés.

## Stack

- **Backend** : FastAPI (Python 3.11), SQLModel, PostgreSQL 15
- **IA** : Mistral (`mistral-small-latest`, `reasoning_effort: high`, température 0,2) ;
  Voxtral Mini (transcription) et Voxtral TTS (synthèse, voix Marie) pour l'assistant vocal
- **Front** : templates Jinja2, Tailwind, marked + DOMPurify, d3 (graphe), pdf.js (sources)
- **Déploiement** : Docker Compose (`db` + `web`)

## Installation

Prérequis : Docker + Docker Compose, une clé API Mistral, un fichier `.env` à la racine.

```bash
docker compose up -d          # db + web
# Application : http://localhost:8001
```

Les migrations Alembic sont exécutées par la commande du conteneur `web` avant uvicorn.

Variables du `.env` (défauts dans `app/config.py`) :

| Variable | Rôle |
| --- | --- |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `SECRET_KEY` | base et session |
| `MISTRAL_API_KEY`, `MODEL_FAST` | le modèle du chat |
| `VOCAL_MODELE_TRANSCRIPTION` (`voxtral-mini-latest`), `VOCAL_MODELE_SYNTHESE` (`voxtral-mini-tts-latest`), `VOCAL_VOIX` (`fr_marie_excited`) | l'assistant vocal |
| `GENERATION_REASONING_EFFORT` (`high`), `CHAT_TEMPERATURE` (0.2), `CHAT_MAX_TOKENS` (4096) | la génération |
| `CHAT_HISTORY_MAX_MESSAGES` (10), `CHAT_HISTORY_MAX_CHARS` (24000) | l'historique renvoyé au modèle |
| `WIKI_DIR` | la racine de connaissance (`wiki_llm` par défaut, `/app/wiki_llm` dans le conteneur) |
| `ADMIN_EMAIL` | l'utilisateur qui reçoit le rôle admin à la connexion |
| `AUTH_COOKIE_SECURE` | `true` en prod HTTPS ; `false` en dev sur `http://` |

## Principales routes API

- **Auth** : `POST /api/auth/register|login|logout`, `GET /api/auth/me`
- **Chat** : `POST /api/chat/stream` `{message, conversation_id}` → SSE
  (`stage`, `thinking`, `message`, `sources`, `done` | `error`)
- **Vocal** : `POST /api/vocal/tour?conversation_id=` corps `audio/wav` (16 kHz mono) ou JSON
  `{texte}` → SSE (`transcription`, `etape`, `message`, `phrase`, `audio`, `sources`, `done` |
  `error`) ; `GET /api/conversations?mode=vocal`
- **Conversations** : `GET/POST/PATCH/DELETE /api/conversations…`, retours
  `POST /api/conversations/messages/{id}/feedback`
- **Wiki** : `GET /api/wiki/graph`, `GET /api/wiki/pages/{chemin}`, `GET /api/wiki/raw/{fichier}`,
  `GET /api/wiki/stats` (admin)
- **Admin** : utilisateurs, rôles, permissions, retours, conversations, messages

## Développement

```bash
# Tests (dépendances Docker-only)
docker compose exec web pytest

# Logs
docker compose logs -f web
```

Le plan de la refonte et ses décisions : `docs/plan_refonte_wiki_cag_2026-09-18.md`.
