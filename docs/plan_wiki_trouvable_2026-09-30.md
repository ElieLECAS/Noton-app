# Plan — un wiki fait pour être retrouvé (30/09/2026)

**Avancement (30/09)** : L0 fait (`docs/benchmark_navigation_2026-09-30.md`). **L1 fait** : section *Writing to be found* dans `wiki_llm/CLAUDE.md` (6 règles, plus le lien anomalie → page et les axes de frontmatter, complétés là où ils vivent ; groupe *Findability* dans le lint). L2 : plan dans `docs/plan_l2_decoupage_2026-09-30.md`, en attente de validation. Le fichier de tags `wiki_llm/tags.md` est créé par L4.

**PROPOSITION, à valider.** Phase 1 : le wiki seul. La boucle de LIA ne bouge pas (hors le retrait
des anomalies du prompt, déjà fait). Le juge de la phase 1, c'est moi : je navigue le wiki avec
**exactement les outils de LIA** (le vrai index `chercher`, trois pages livrées entières,
`lire_page`), sans grep ni PDF, et je dois retrouver chaque réponse. Phase 2, ensuite : copier
dans LIA la façon dont je navigue.

Pourquoi le wiki d'abord : le contenu est juste (TGY3704 était dans la page), c'est la forme qui
empêche de le retrouver — voir `docs/audit_recherche_2026-09-30.md`.

## État mesuré le 30/09

| Critère | Mesure |
| --- | --- |
| Pages | 297 ; médiane 11,9 k car. |
| Pages géantes | 25 > 50 k car., 9 > 100 k (Roto NX crémones 141 k, Système 70 plans de combinaison 129 k…) |
| Volume livré par une recherche | médiane 64 k car., p90 137 k (golden pris comme requêtes) |
| Tags | 736 distincts, 385 employés une seule fois, 24 couples singulier / pluriel, 27 quasi-doublons à une lettre (`cremona`) |
| Anomalies reliées à leur page | 150 / 479 (31 %) — les 329 autres ne remontent qu'aux mots de la question |
| Titres | exemple : la page qui donne les crémones Technal s'intitule « Roulements, fermetures et manœuvres Technal SOLEAL GY 55 » |

## L0 — Le banc d'essai, avant toute modification

**Le jeu « navigation » (≈ 60 questions)**, au format du golden existant pour que le runner le
rejoue ensuite sur LIA (`tests/fixtures/golden/navigation.json`) :

- ~30 questions **réelles** tirées de la base (373 questions distinctes, 79 depuis le wiki,
  76 avec une référence), choisies pour couvrir fournisseurs et gammes : Technal (SOLEAL FY / GY
  / PY, LUMÉAL), ASKEY, Roto NX, profine, Kömmerling (PERFORM, PERFORM76, Système 70),
  INNOSLIDE, garanties, procédures, AEV et régions climatiques ;
- ~15 à **référence seule**, sans gamme ni fournisseur (le cas TGY3702) ;
- ~10 **relances** (« et chez Technal ? ») ;
- ~5 **sans réponse** : le wiki ne l'a pas, il faut le dire.

Chaque question porte la réponse attendue, la page et la ligne de tableau qui la donnent. Je
les rédige depuis le wiki et les PDF ; **tu valides les réponses attendues** — le juste métier,
c'est toi.

**Deux mesures par question :**

1. **Trouvabilité**, mécanique, sans modèle : rang de la page attendue pour la question telle
   quelle et pour trois formulations naturelles (avec / sans la référence, mot du métier,
   synonyme). Automatique, rejouable en quelques secondes après chaque chantier.
2. **Ma navigation** : je réponds comme LIA, six appels au plus. Je journalise chaque requête,
   ce qui revient, où je trouve ou échoue, et je classe chaque échec : page introuvable, page
   trop longue, titre ou tags, donnée absente, tableau ambigu, anomalie non reliée.

Plus, pour la phase 2 : LIA sur le même jeu (base de comparaison, rien à changer chez elle).

**Livrable :** la liste des défauts par page. C'est la liste de travail de L2 à L6.

## L1 — Les règles, écrites dans le protocole

Une section « Écrire pour être retrouvé » dans `wiki_llm/CLAUDE.md`, qui fait foi :

1. **Taille.** Une page ≤ ~20 000 caractères. Au-delà, on découpe le long d'un axe de la source
   (famille, usage, tableau), jamais arbitrairement, et sans créer de page hub : la navigation
   humaine passe par les facettes.
2. **Titre.** Il nomme les produits dont la page donne les références, avec les mots du
   menuisier, et le fournisseur ou la gamme : « Crémones, gâches et chariots Technal SOLEAL
   GY 55 », pas « Roulements, fermetures et manœuvres ».
3. **Description.** Elle dit à quelles questions la page répond (quelles familles, quelles
   références, quelles valeurs).
4. **Tags.** Liste fermée, au singulier, tenue dans le protocole ; un tag nouveau entre dans la
   liste avant d'être employé.
5. **Noms du métier.** Quand un objet a plusieurs noms (crémone / fermeture multipoint,
   appui / seuil, fiche / paumelle), la page emploie le nom de la source **et** le nom courant,
   dans le texte.
6. **Références** écrites en entier, comme la source (`TGY3702`), une ligne par référence
   (règle existante).
7. **Frontmatter complet** : type, gamme, système, fournisseur, usage, famille — tout ce qui sert
   de facette.
8. **Anomalies reliées** : chaque entrée d'un registre porte le lien de la ou des pages qu'elle
   concerne.

## L2 à L6 — Les chantiers, dans l'ordre du rendement

| Lot | Chantier | Périmètre | Contrôle |
| --- | --- | --- | --- |
| L2 | Découpe des pages géantes | Roto NX (5 pages de 90 à 141 k), Système 70 (6 pages de 80 à 129 k), puis les 14 autres > 50 k | Inventaire avant / après des références et des lignes de tableau : **zéro perte**, comme la refonte du 19/09 |
| L3 | Titres et descriptions | Les 297 pages, en commençant par celles où L0 a échoué | Trouvabilité rejouée |
| L4 | Tags | 736 → liste fermée ; couples singulier / pluriel fusionnés, fautes corrigées, tags uniques fusionnés ou retirés | Plus aucun tag hors liste |
| L5 | Frontmatter | Facettes complètes sur chaque page | Aucun champ facette vide sur une page Profilé / Quincaillerie |
| L6 | Anomalies | Les 329 entrées sans lien reçoivent celui de leur page | 100 % d'entrées reliées |

Après chaque lot : trouvabilité rejouée (automatique), et ma navigation refaite sur les questions
qui échouaient plus un échantillon de celles qui passaient, pour détecter une régression.

**Critère de sortie de la phase 1 :**
- page attendue dans les trois premières pour ≥ 95 % des formulations ;
- ma navigation donne la bonne réponse sur ≥ 95 % du jeu, en trois appels ou moins en médiane ;
- aucune page au-delà du budget, aucun tag hors liste, toutes les anomalies reliées.

## Phase 2 — ensuite, pas maintenant

Le journal de ma navigation dit ce que je fais et que LIA ne fait pas : garder la référence,
lire la ligne du tableau plutôt que le paragraphe, reformuler avec le nom du métier, n'employer
une facette qu'après avoir vu les résultats, s'arrêter quand la page est trouvée. On le copie
dans LIA, mesure par mesure, sur le même banc (R1, R2… de l'audit).

## Ce que j'attends de toi

1. Ton accord sur ce plan, ou tes corrections.
2. Ton accord sur les huit règles de L1 avant que je les écrive dans le protocole.
3. La validation des réponses attendues du banc quand je te les soumets.
