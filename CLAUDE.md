# LIA — repères pour Claude Code

Assistant documentaire PROFERM. **Une seule source : le wiki** (`wiki_llm/wiki/`). Il ne tient
plus dans aucune fenêtre de contexte (198 pages) : le modèle le **navigue par outils** au lieu de
le recevoir en entier. Mistral Small, trois outils, rien d'autre.

**22/09/2026 — le CAG est remplacé par la navigation outillée.** Le prompt permanent ne porte que
les consignes, un **vocabulaire** (types, tags, gammes, systèmes) et l'**index des anomalies**
(identifiant + sujet) : ~4 600 tokens au lieu de 185 000. Les pages arrivent par `chercher`, qui
livre directement le contenu **entier** des trois premières trouvées. On ne raccourcit jamais le
wiki pour le faire rentrer quelque part. **La lecture des PDF est multimodale native** : chaque
page est visualisée directement, sans script d'extraction ni PyMuPDF, pour garantir zéro perte
d'information technique. Le protocole d'écriture fait foi : `wiki_llm/CLAUDE.md`.

## Où sont les choses

- `app/services/wiki_index.py` — l'index de navigation : recherche lexicale BM25 sur le texte
  intégral + facettes, vocabulaire, registres d'anomalies, mise en forme des résultats.
- `app/services/wiki_service.py` — charge le wiki, construit l'index, le prompt permanent et sa
  clé de cache, le graphe et le lint ; rechargé dès qu'un fichier change.
- `app/services/wiki_chat_service.py` — le tour de chat : boucle d'outils, injection serveur des
  anomalies, filtre des coupes, flux SSE, citations vérifiées par le code (`/dossier/page.md` →
  existe ou pas), identifiants d'anomalie.
- `app/prompts/wiki_consignes.md` — les consignes du modèle (modifier = nouvelle clé de cache).
- `app/routers/chat.py`, `wiki.py`, `conversations.py`, `admin.py`, `auth.py`.
- `app/templates/chat.html` (chat + étapes de lecture + lecteur + PDF), `wiki.html` (graphe),
  `admin.html`.
- `wiki_llm/CLAUDE.md` — le protocole d'écriture du wiki : c'est LUI qui fait foi pour toute
  ingestion ou correction de page. L'application ne modifie jamais le wiki.

## Le tour de chat en trois outils

`chercher(mots_cles, type, tags, gamme, systeme, limite)` — `lire_page(chemin)` —
`lire_anomalie(identifiant)`. Six allers-retours au maximum, puis le tour s'arrête sur un message
clair.

Deux garde-fous ne dépendent pas de la discipline du modèle :

- **les anomalies sont injectées par le serveur**, rapprochées de la question et des pages
  chargées : la règle 2 (donner la valeur *et* signaler la contradiction) est trop importante ;
- **les coupes sont vérifiées sur disque**, et rattachées à la référence demandée par le couple
  référence/image relevé dans le tableau de la page : un chemin inventé est retiré, une image
  prise sur la ligne voisine est remplacée.

Et une reprise : si le modèle répond sans avoir chargé la moindre page, il est renvoyé lire **une
fois**, et ce qu'il avait commencé à écrire est effacé de l'écran.

## Règles de la maison

- Pas de drapeau de fonctionnalité : comportement en dur, code remplacé supprimé.
- **Rien de plus que les trois outils** au tour de chat : pas de juge, pas de reranker, pas de
  vote, pas d'index vectoriel. La recherche est lexicale par choix — deux pages qui se
  contredisent doivent toutes les deux remonter, un top-k sémantique les met en concurrence. Une
  réponse fausse se corrige d'abord dans le wiki, ensuite dans les consignes. Mesurer avant de
  proposer, proposer avant de coder.
- Une facette **remonte** une page, elle ne l'exclut pas : une facette mal choisie cachait la
  bonne page.
- Tests : `docker compose exec web pytest` (dépendances Docker-only).
- Migrations Alembic idempotentes (`IF EXISTS`) ; `create_all` tourne aussi au démarrage.
- Les PDF de `wiki_llm/raw/` ne sont pas versionnés ; les `.md` le sont.

Plan et décisions de la refonte CAG : `docs/plan_refonte_wiki_cag_2026-09-18.md`.
