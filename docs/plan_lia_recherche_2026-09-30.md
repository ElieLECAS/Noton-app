# Plan — rapprocher la recherche de LIA de celle de Claude (30/09/2026)

**PROPOSITION, à valider étape par étape.** Appuyé sur `docs/benchmark_navigation_2026-09-30.md`
et `docs/audit_recherche_2026-09-30.md`.

**Principe.** Tout ce qui est mécanique passe dans le code ; le modèle ne garde que la lecture
et la rédaction. Les règles de la maison tiennent : trois outils, recherche lexicale, ni juge,
ni reranker, ni index vectoriel, ni drapeau. Chaque étape est mesurée avant d'être gardée, et
gardée seulement si elle améliore sans rien casser.

## Ce que chaque étape copie de Claude

| Ce que fait Claude | Aujourd'hui chez LIA | Étape |
| --- | --- | --- |
| Garde la référence de la question | Perdue par la reformulation du modèle (0 à 3 sur 8) | 1 |
| Sait qu'un mot n'existe nulle part | L'outil ne le dit pas ; réponse inventée le 30/09 | 1 |
| Ne réinjecte pas un appel raté | Relance qui le prolonge, réponse sans page affichée | 2 |
| Essaie les variantes d'écriture | `œ`, `487 206`, `LUMINE 65`, pluriels non reconnus | 3 |
| Lit la ligne et sa section | Trois pages entières, 116 000 car. en médiane | 4 |
| Référence seule d'abord, pas de facette au premier essai, une réponse par gamme, cherche les jumelles | Dépend du modèle, non mesuré hors PERFORM76 | 5 |

## Étape 0 — Mesurer LIA elle-même (préalable)

Sans cette mesure, on ne sait pas quelle étape rapporte.

1. **Tu valides les 11 réponses attendues** signalées dans le benchmark (R03, R05, R09, R10, R18,
   R20, R22, R31, S01, S07, G10).
2. **Le banc est converti au format du runner** (`tests/fixtures/golden/navigation.json`) : pour
   chaque question, les valeurs qui doivent figurer et celles qui ne doivent pas (R33 : `TGY3704`,
   interdit `771923`), ou une abstention pour les 11 questions où le wiki est muet.
3. **Le runner apprend les relances** : il joue d'abord les messages qui précèdent la question,
   dans la même conversation.
4. **Passage de LIA réelle** sur les 57 questions, une à la fois (à trois en parallèle, Mistral
   répond 429). Environ 8 millions de tokens d'entrée par passage. Plus le golden PERFORM76
   (93,5 % aujourd'hui), qui sert de garde-fou contre les régressions.

Résultat : le score de départ, par catégorie, et la liste des échecs, classés par cause.

## Étape 1 — Ne plus perdre la question (corrige TGY3702)

- **La référence de la question entre dans chaque recherche.** Le serveur relève les références
  de la question (mots qui portent un chiffre et que l'index connaît) et les ajoute à tout appel
  de `chercher` qui les omet. Pour une relance sans référence (« chez Technal ? »), celles du
  message précédent de l'utilisateur. Mesuré hors ligne : ajoutée à la requête réellement écrite
  par le modèle, `TGY3702` fait passer la page Technal du rang 8 (ou d'absente) au rang 1.
- **L'absence est dite.** En tête du résultat de `chercher`, une ligne : « Absents de tout le
  wiki : 3702 » pour tout mot ou référence de la requête qu'aucune page ne contient. L'index le
  sait déjà (fréquence nulle).

Mesure : référence gardée 8 fois sur 8 (déterministe) ; R33, S01, S03 justes ; R09, R26, S02,
S05 en abstention juste ; golden inchangé.

## Étape 2 — Une relance et une sortie propres

- Un appel d'outil écrit en texte (`chercher(`, `"mots_cles"`) est reconnu comme raté : on relance
  **sans** remettre ce texte dans le contexte.
- Si le tour se termine sans qu'aucune page ait été lue alors que la question porte sur le
  métier, l'écran affiche un message clair (« je n'ai pas pu consulter le wiki, reformule ») au
  lieu d'une réponse sans source.

Mesure : le tour du 30/09 (« chez la crémone chez technal ») rejoué 8 fois, 0 texte parasite,
0 page inventée.

## Étape 3 — L'index reconnaît les écritures

Dans `wiki_index.tokenise`, appliqué au wiki comme aux requêtes :

- `œ` → `oe`, `æ` → `ae` (« œil de bœuf » est aujourd'hui découpé en `il / b / uf`) ;
- un nombre écrit par groupes (`487 206`) est aussi indexé d'un seul tenant (`487206`) ;
- une gamme ou un système suivi d'un nombre (`LUMINE 65`, `Perform 76`) est aussi indexé collé
  (`lumine65`, `perform76`) ;
- pluriel simple : `parcloses` → `parclose`.

En parallèle, côté wiki : les ~30 références mal écrites (15 groupées par une espace, 15
abrégées en `/`) et les noms PROFERM ↔ Technal sur une dizaine de pages (LUMINE55 = SOLEAL FY,
SOLÉAL55 = SOLEAL GY, LUMÉAL55 = LUMEAL GA), selon les règles écrites dans le protocole (L1).

Mesure : trouvabilité mécanique (en quelques secondes, sans modèle) : 84 % des formulations en
top 3 aujourd'hui, 12 absentes du top 15 ; puis golden et banc, parce que le classement bouge
partout.

## Étape 4 — Livrer ce qui sert (le plus gros gain, le plus délicat)

`chercher` livre la **première page entière** ; pour les deux suivantes, **les sections qui
contiennent les mots de la requête**, chacune avec sa chaîne de titres et l'introduction du
tableau (« comment lire »), et une ligne « le reste : `lire_page` ». La suite des résultats reste
en métadonnées.

C'est ce que Claude fait quand il a le choix : il ouvre la ligne et son en-tête, pas trois
manuels. Cela change une décision inscrite dans `CLAUDE.md` (« trois pages entières ») : à
valider par toi avant de coder.

Mesure : volume lu par question (médiane 116 000 car. aujourd'hui, cible ≤ 40 000), taux de
bonnes réponses inchangé ou meilleur, nombre de `lire_page` en plus. Risque à surveiller : une
page 2 dont la section utile ne contient pas le mot de la requête ; le banc le montrera.

## Étape 5 — La conduite, dans les consignes

Ajouter à `wiki_consignes.md` ce que Claude fait et que le code ne peut pas faire à sa place :

- chercher d'abord la référence seule, puis les mots du métier ;
- pas de facette au premier appel (la facette `gamme` fait remonter les pages coloris et
  commerciales : R03, R09, R10) ;
- gamme non précisée : une réponse par gamme, ou la question posée en retour ;
- chercher les jumelles (même référence sur deux dormants, même numéro sur deux systèmes) ;
- conclure « absent » après deux recherches concordantes, dont la référence seule ;
- lire la section « comment lire » d'un tableau avant d'en tirer une valeur.

Et deux retouches de l'outil : afficher `status: draft` dans les résultats de `chercher`
(aujourd'hui visible seulement par `lire_page`) ; si le banc montre encore le biais de la
facette `gamme`, le corriger dans l'index.

Mesure : banc et golden ; toute consigne modifiée change la clé de cache (premier tour plus lent).

## Objectif : « proche de Claude »

| Critère | Claude (banc) | Cible LIA |
| --- | --- | --- |
| Réponses équivalentes à la référence | 57 / 57 | ≥ 52 / 57 |
| Abstentions justes (wiki muet) | 11 / 11 | 11 / 11 |
| Pages inventées | 0 | 0 |
| Volume lu par question (médiane) | ~116 000 car. avec les outils de LIA | ≤ 40 000 |
| Temps de réponse | 1 à 3 min | ≤ 20 s |
| Golden PERFORM76 | — | ≥ 93,5 % |

## Hors plan, à décider si l'écart persiste

- **Un modèle plus grand.** Le projet est bâti sur Mistral Small ; si, après les étapes 1 à 5,
  les échecs restants sont des erreurs de lecture et non de recherche, le banc permettra de
  mesurer un autre modèle en une passe. Ta décision.
- **La découpe L2** des pages qui resteraient lourdes après l'étape 4.

## Ordre et coût

Une étape = un changement, un passage du banc et du golden (≈ 1 h de mesure chacun, une question
à la fois), un commit. Étapes 1 et 2 d'abord : petites, déterministes, et elles corrigent les
deux bugs vus en prod.
