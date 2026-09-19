# LIA — repères pour Claude Code

Assistant documentaire PROFERM. **Une seule source : le wiki** (`wiki_llm/wiki/`), placé en
entier dans le prompt système à chaque question (CAG, Mistral Small). Pas de retriever, pas
d'index, pas d'ingestion dans l'application.

**19/09/2026 — l'application est en stand-by.** Le wiki passe en transcription exhaustive des
PDF (2 382 pages) et dépassera largement la fenêtre de 256 k : le CAG intégral ne tiendra plus,
un outil de navigation par métadonnées OKF le remplacera. D'ici là on ne touche pas à `app/`, et
on ne raccourcit jamais le wiki pour le faire rentrer dans le prompt. Le protocole d'écriture
fait foi : `wiki_llm/CLAUDE.md`.

## Où sont les choses

- `app/services/wiki_service.py` — charge le wiki, construit le prompt système et sa clé de
  cache, le graphe et le lint ; rechargé dès qu'un fichier change.
- `app/services/wiki_chat_service.py` — le tour de chat : messages, flux SSE, citations
  vérifiées par le code (`/dossier/page.md` → existe ou pas), identifiants d'anomalie.
- `app/prompts/wiki_consignes.md` — les consignes du modèle (modifier = nouvelle clé de cache).
- `app/routers/chat.py`, `wiki.py`, `conversations.py`, `admin.py`, `auth.py`.
- `app/templates/chat.html` (chat + lecteur + PDF), `wiki.html` (graphe), `admin.html`.
- `wiki_llm/CLAUDE.md` — le protocole d'écriture du wiki : c'est LUI qui fait foi pour toute
  ingestion ou correction de page. L'application ne modifie jamais le wiki.

## Règles de la maison

- Pas de drapeau de fonctionnalité : comportement en dur, code remplacé supprimé.
- Pas de mécanisme ajouté au tour de chat (retriever, juge, zoom, vote) : une réponse fausse se
  corrige dans le wiki. Mesurer avant de proposer, proposer avant de coder.
- Tests : `docker compose exec web pytest` (dépendances Docker-only).
- Migrations Alembic idempotentes (`IF EXISTS`) ; `create_all` tourne aussi au démarrage.
- Les PDF de `wiki_llm/raw/` ne sont pas versionnés ; les `.md` le sont.

Plan et décisions de la refonte : `docs/plan_refonte_wiki_cag_2026-09-18.md`.
