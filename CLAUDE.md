# LIA — repères pour Claude Code

Assistant documentaire PROFERM. **Une seule source : le wiki** (`wiki_llm/wiki/`). Il ne tient
plus dans aucune fenêtre de contexte (246 pages) : le modèle le **navigue par outils** au lieu de
le recevoir en entier. Mistral Small, trois outils, rien d'autre.

**22/09/2026 — le CAG est remplacé par la navigation outillée.** Le prompt permanent ne porte que
les consignes et un **vocabulaire** (types, tags, gammes, systèmes) : ~3 200 tokens au lieu de
185 000. **Aucune anomalie dans le prompt** (30/09) : on pose une question, il cherche dans le
wiki. L'index des 479 entrées (35 000 car.) lui faisait perdre la référence demandée — audit
`docs/audit_recherche_2026-09-30.md`. **01/10/2026 — la recherche se fait par SECTIONS et se
livre par page ou par sections** (`docs/plan_indexation_sections_2026-10-01.md`) : le wiki fait
maintenant ~2 M tokens et 23 pages de plus de 50 000 car., alors que « trois pages entières »
avait été réglé le 22/09 sur 193 pages sans page > 50 k. `chercher` classe des sections (découpe
faite à l'indexation, jamais dans le wiki), livre la page **entière si elle fait moins de 15 000
car.**, sinon sa **fiche**, son **sommaire** (§n, lignes) et les **sections qui répondent** ; 6 pages
et 60 000 car. au plus par recherche, rien n'est livré deux fois dans un tour. **`lire_page(chemin,
section)`** lit une section du sommaire (« §13 », un mot du titre, « sommaire »), sans `section` la
page complète. On ne raccourcit jamais le wiki pour le faire rentrer quelque part ; le budget d'un
tour est de 200 000 car. **La lecture des PDF est multimodale native** : chaque page est visualisée
directement, sans script d'extraction ni PyMuPDF, pour garantir zéro perte d'information
technique. Le protocole d'écriture fait foi : `wiki_llm/CLAUDE.md`.

## Où sont les choses

- `app/services/wiki_index.py` — l'index de navigation : BM25 sur le texte intégral (pages) et
  par **sections** (`decouper_sections`, `classer`, `livrer`, `lire`), facettes, vocabulaire,
  registres d'anomalies.
- `app/services/wiki_service.py` — charge le wiki, construit l'index, le prompt permanent et sa
  clé de cache, le graphe et le lint ; rechargé dès qu'un fichier change.
- `app/services/wiki_chat_service.py` — le tour de chat : boucle d'outils, livraison page/sections
  et budget du tour (`Livraison`), injection serveur des anomalies, filtre des coupes, flux SSE,
  citations vérifiées par le code (`/dossier/page.md` → existe ou pas), identifiants d'anomalie.
- `app/scripts/mesurer_recuperation.py` — la mesure hors ligne, sans modèle : rejoue la première
  recherche sur le golden (`tests/fixtures/golden/golden_40_questions.json`) et sur le banc du
  30/09 et dit si la preuve arrive au modèle, à quel volume. À relancer après toute modification de
  l'index (`--grille` balaie seuil, nombre de pages, budget).
- `app/prompts/wiki_consignes.md` — les consignes du modèle (modifier = nouvelle clé de cache).
- `app/prompts/vocal_consignes.md` — la forme parlée, placée DEVANT les consignes générales pour
  le tour vocal : prose, pas de liste ni de tableau, les citations restent, pas d'image. Les
  règles de vérité ne sont écrites qu'une fois, dans `wiki_consignes.md`.
- `app/services/vocal_service.py` — l'assistant vocal (`/vocal`) : transcription Voxtral Mini
  par lots avec biais de vocabulaire tiré du wiki, le MÊME `WikiAnswer` (prompt vocal, coupes
  désactivées), synthèse Voxtral TTS en flux phrase par phrase (voix Marie), `texte_parle`
  (ce qui se dit ≠ ce qui s'affiche), phrases d'attente préchauffées.
- `app/services/wiki_depot.py` — le dépôt : glisser `wiki_llm/` dans l'administration met le
  wiki à jour et joint les PDF, sans pull ni rebuild (voir plus bas).
- `app/routers/chat.py`, `vocal.py`, `wiki.py`, `conversations.py`, `admin.py`, `auth.py`.
- `app/templates/chat.html` (chat + étapes de lecture + lecteur + PDF), `vocal.html` (orbe,
  micro, détection de fin de parole côté navigateur, transcription surlignée au fil de la
  voix), `wiki.html` (accueil par produit et métier, lecteur), `carte.html` (carte mentale,
  entrée à part dans la navigation), `admin.html`.
- `app/services/wiki_carte.py` — l'arbre de la carte mentale (`GET /api/wiki/carte`) : la
  taxonomie de la frontmatter, niveaux inutiles sautés. Le choix des nœuds se teste ici ; le
  navigateur ne fait que la mise en page. Plus de vue graphe (illisible, retirée le 25/09/2026) ;
  `/api/wiki/graph` reste, c'est la liste des pages du wiki, du chat et du vocal.
- `app/services/faisabilite.py` + `faisabilite.html` (`/faisabilite`, 25/09/2026, premier jet) —
  **réservé au rôle admin** (30/09 : API posée sur le routeur, page renvoyée au chat, lien du
  menu caché ; débit compris) — le vérificateur PERFORM76 de l'audit § 9.1 : débit → DTA → abaque de dormant → abaque
  d'ouvrant (couleur, 25 %, courbe de verre, J079, zones) → parclose → ferrure Roto NX →
  pivot bas → isolant et tapée. Aucun modèle, toutes les
  valeurs lues dans les tableaux du wiki (un en-tête changé casse `tests/test_faisabilite.py`).
  Jamais d'interpolation (la plus restrictive des deux graduations) ; à moins de la précision
  de lecture d'une limite, « sur étude ». Hypothèse affichée : LFF / HFF Roto = DFO profine ;
  poids = verre seul (le wiki n'a pas le poids des profilés) ; CTR-18 → bornes les plus basses.
- `app/services/debit_atelier.py` (onglet « Débit et nomenclature » de `/faisabilite`, § 9.2) —
  même saisie, liste de coupe : cotes à déduire de `systeme-76-cotes-de-debit.md` une fois par
  coupe, renforts d'ouvrant selon la zone de l'abaque, parclose, tapée, appui, paumelles ; puis
  nomenclature (qté, ml) et accessoires par profilé avec dessins. Barres soudées en cote finie
  sauf surcote saisie (le 76 n'en documente pas). Impression A4 et CSV côté navigateur.
- `app/services/parcloses.py` + `parcloses.html` (`/parcloses`) — pour un vitrage de X mm, les
  parcloses et joints de chaque gamme, chacune lue dans sa forme : 76 Advanced (A joint 4 mm /
  B joint 2 mm, tolérance +1 / −0,5), PERFORM76 (cahier), SOLEAL FY 55 (matrice parclose ×
  joint intérieur, plage recommandée, élargisseur, pose de face, ouvrant minimal), ASKEY /
  LUMEAL GA / SOLEAL GY (profilé d'ouvrant par épaisseur). Tolérance écrite = règle ; sans
  tolérance : ±0,5 mm « correspond », ±1,5 mm « proche ». Les gammes non calculables sont listées.
- `wiki_llm/CLAUDE.md` — le protocole d'écriture du wiki : c'est LUI qui fait foi pour toute
  ingestion ou correction de page. L'application ne corrige jamais une page : elle remplace le
  wiki EN BLOC par ce qu'on lui dépose.

## Le tour de chat en trois outils

`chercher(mots_cles, type, tags, gamme, systeme, limite)` — `lire_page(chemin, section)` —
`lire_anomalie(identifiant)`. Huit allers-retours au maximum, puis le tour s'arrête sur un message
clair.

Deux garde-fous ne dépendent pas de la discipline du modèle :

- **les anomalies sont injectées par le serveur**, rapprochées des pages chargées, après chaque
  recherche et jamais avant : la règle 2 (donner la valeur *et* signaler la contradiction) est
  trop importante ;
- **les coupes sont vérifiées sur disque**, et rattachées à la référence demandée par le couple
  référence/image relevé dans le tableau de la page : un chemin inventé est retiré, une image
  prise sur la ligne voisine est remplacée.

Et une reprise : si le modèle répond sans avoir chargé la moindre page, il est renvoyé lire **une
fois**, et ce qu'il avait commencé à écrire est effacé de l'écran.

La recherche aussi est tenue par le serveur (30/09/2026, mesuré sur les recherches réelles de
Mistral) :

- **les références de la question sont cherchées À PART** (TGY3702 : le modèle ne les gardait
  que 3 fois sur 8) et ajoutées après les résultats : remises dans sa requête, elles
  l'écrasaient (section du DTA du rang 2 au rang 29, 01/10) ;
- **une cote de la question n'est pas une référence** (« 1 800 mm », « 1 200 de large ») : elle
  ne reçoit pas le poids ×3 qui ramenait les grands tableaux de ferrures ;
- **le filtre `type` ne classe plus** : Mistral le devine mal (« Profilé » pour une limite que
  fixe une page de gamme) ; bonne page dans les trois premières 68 % → 85 % sans lui ;
- **une référence qu'aucune page ne porte est annoncée** (« Absent de tout le wiki ») au lieu de
  six recherches pour une absence ;
- les graphies se rejoignent : œ, « 487 206 », « LUMINE 65 », pluriel en -s/-x (pas plus) ;
- la page d'une **coupe servie rejoint les sources**, que le modèle la cite ou non.

Mesuré le 01/10 (`mesurer_recuperation`, sans modèle) : la preuve arrive au modèle dans 95 % des
questions du golden et 91 % du banc pour ~52 000 car., contre 95 % et 82 % pour ~69 000 avec trois
pages entières. Réglages fins (taille de section, poids des titres) : ±3 points, sans effet. Un
routeur à règles « intention → pages » n'a pas fait mieux que le classement lexical à volume égal.
Ne livre pas d'une page lue en partie une conclusion d'absence : le sommaire dit ce qui n'a pas
été reçu (consigne 0).

La règle 14 des consignes interdit un « oui » de faisabilité sans avoir lu la limite (3 → 8 sur 9
justes sur les trois questions de faisabilité du golden).

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

## Le dépôt du wiki (`/admin`, onglet Wiki, 23/09/2026)

Le wiki s'écrit hors de l'application et arrivait sur le serveur en deux morceaux : `git pull`
pour les `.md`, `scp` pour les PDF (hors git, 800 Mo). Il se dépose maintenant depuis
l'administration — on glisse `wiki_llm/` (ou `wiki/`, ou `raw/`), sans pull ni rebuild.

Trois temps, parce qu'on ne lance pas 800 Mo à l'aveugle : `POST /api/wiki/depot` annonce ce
qu'on a et reçoit le **plan** (pages nouvelles, pages qui vont disparaître, PDF manquants) ;
`PUT /api/wiki/depot/{id}/fichier?chemin=` envoie un fichier par requête, corps brut, en flux,
trois en parallèle ; `POST …/valider` bascule et recharge l'instantané.

Deux dossiers, deux contrats — ils n'ont pas la même nature :

- `wiki/` est un **miroir** : ce qui est déposé DEVIENT le wiki, une page absente disparaît —
  sans quoi une page retirée de la rédaction continuerait de répondre. Tout monte dans
  `.depot/<id>/wiki`, et la bascule est un couple de renommages sous le verrou de l'instantané
  (`wiki_service.reload_lock`) : un envoi interrompu ne laisse pas un wiki à trous.
- `raw/` s'**accumule** : seuls les PDF manquants ou de taille différente montent, et rien n'est
  jamais supprimé. Sans ce différentiel, corriger une page coûterait 800 Mo.

Un dépôt sans aucune page ne touche pas au wiki : glisser `raw/` seul ajoute des PDF, point.
C'est ce qui rend la suppression en miroir sûre. Aucun état en mémoire : le manifeste est sur
disque, un dépôt abandonné est ramassé au suivant.

`wiki_llm/` est donc monté **en écriture** (`docker-compose.yaml`). Derrière nginx :
`client_max_body_size` ≥ le plus gros PDF (256 Mo couvre large), `proxy_request_buffering off`.

## Règles de la maison

- Pas de drapeau de fonctionnalité : comportement en dur, code remplacé supprimé.
- **Rien de plus que les trois outils** au tour de chat : pas de juge, pas de reranker, pas de
  vote, pas d'index vectoriel. La recherche est lexicale par choix — deux pages qui se
  contredisent doivent toutes les deux remonter, un top-k sémantique les met en concurrence. Une
  réponse fausse se corrige d'abord dans le wiki, ensuite dans les consignes. Mesurer avant de
  proposer, proposer avant de coder.
- Une facette **remonte** une page, elle ne l'exclut pas : une facette mal choisie cachait la
  bonne page. Le `type` ne classe pas du tout ; il ne sert qu'à lister sans mot-clé.
- Tests : `docker compose exec web pytest` (dépendances Docker-only).
- Migrations Alembic idempotentes (`IF EXISTS`) ; `create_all` tourne aussi au démarrage. Deux
  révisions : `lia_wiki_schema` (le schéma) puis `lia_vocal_mode` (le mode d'une conversation).
- Voxtral : `voxtral-mini-latest` (transcription), `voxtral-mini-tts-latest` (synthèse), voix
  par slug (`fr_marie_excited`). Le `pcm` de la synthèse est du float32 24 kHz. La liste des voix
  (`GET /v1/audio/voices`) est paginée bizarrement ; `GET /v1/audio/voices/{slug}` répond.
- Les PDF de `wiki_llm/raw/` ne sont pas versionnés ; les `.md` le sont.

Plan et décisions de la refonte CAG : `docs/plan_refonte_wiki_cag_2026-09-18.md`.
