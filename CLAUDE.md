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
- `app/prompts/vocal_consignes.md` — la forme parlée, placée DEVANT les consignes générales pour
  le tour vocal : prose, pas de liste ni de tableau, les citations restent, pas d'image. Les
  règles de vérité ne sont écrites qu'une fois, dans `wiki_consignes.md`.
- `app/services/vocal_service.py` — l'assistant vocal (`/vocal`) : transcription Voxtral Mini
  par lots avec biais de vocabulaire tiré du wiki, le MÊME `WikiAnswer` (prompt vocal, coupes
  désactivées), synthèse Voxtral TTS en flux phrase par phrase (voix Marie), `texte_parle`
  (ce qui se dit ≠ ce qui s'affiche), phrases d'attente préchauffées.
- `app/routers/chat.py`, `vocal.py`, `wiki.py`, `conversations.py`, `admin.py`, `auth.py`.
- `app/templates/chat.html` (chat + étapes de lecture + lecteur + PDF), `vocal.html` (orbe,
  micro, détection de fin de parole côté navigateur, transcription surlignée au fil de la
  voix), `wiki.html` (graphe), `admin.html`.
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

## Le tour vocal (`/vocal`, 22/09/2026)

Le même tour, dit à voix haute : `POST /api/vocal/tour?conversation_id=` reçoit l'enregistrement
(`audio/wav` 16 kHz produit par le navigateur) ou une question écrite (JSON), rend un flux SSE où
les événements du tour (`etape`, `message`, `sources`) sont entrelacés avec `phrase` et `audio`
(float32 24 kHz base64, relayé tel quel). Décisions mesurées le 22/09 :

- **transcription par lots, pas temps réel** : Voxtral Mini transcrit dix secondes en 0,5 s et
  accepte un **biais de vocabulaire** (cent termes : PROFERM, parclose, gammes, systèmes, tags)
  que le modèle temps réel refuse — sans lui, « parclose » devient « part close » ;
- **on ne parle qu'une fois une page chargée** : tant que `pages_lues` est vide, le texte peut
  être effacé par la relance ; il est mis en attente, jamais dit. Un `reset` jette ce qui
  attendait et coupe la synthèse en cours ;
- **première phrase seule, puis groupes de ~220 caractères** (`Phraseur`) ; une phrase d'attente
  est dite au premier appel d'outil ;
- **ce qui se dit ≠ ce qui s'affiche** : `texte_parle` retire les chemins cités, lit les
  identifiants d'anomalie en clair, déplie les unités ; l'écran garde les pastilles.
- Les conversations vocales ont `mode = "vocal"` et ne sont pas listées dans le chat.

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
- Migrations Alembic idempotentes (`IF EXISTS`) ; `create_all` tourne aussi au démarrage. Deux
  révisions : `lia_wiki_schema` (le schéma) puis `lia_vocal_mode` (le mode d'une conversation).
- Voxtral : `voxtral-mini-latest` (transcription), `voxtral-mini-tts-latest` (synthèse), voix
  par slug (`fr_marie_excited`). Le `pcm` de la synthèse est du float32 24 kHz. La liste des voix
  (`GET /v1/audio/voices`) est paginée bizarrement ; `GET /v1/audio/voices/{slug}` répond.
- Les PDF de `wiki_llm/raw/` ne sont pas versionnés ; les `.md` le sont.

Plan et décisions de la refonte CAG : `docs/plan_refonte_wiki_cag_2026-09-18.md`.
