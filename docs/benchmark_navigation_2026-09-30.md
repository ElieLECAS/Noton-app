# Benchmark « Claude imitant LIA » contre « Claude normal » — 30/09/2026

Banc L0 du plan `docs/plan_wiki_trouvable_2026-09-30.md`. Aucun appel à Mistral : les deux bras sont
Claude.

- **Claude imitant LIA** : le prompt permanent de LIA et ses trois outils exécutés par le code du
  chat lui-même (`app/scripts/naviguer_wiki.py` → `WikiAnswer._executer`) : mêmes trois pages
  livrées entières, mêmes anomalies injectées après lecture, 5 appels au plus, aucun accès aux
  fichiers du wiki. 14 agents à contexte neuf, 4 à 5 questions chacun.
- **Claude normal** : accès libre aux fichiers du wiki (grep, lecture), le wiki pour seule source.
  Il fournit la réponse attendue, la ligne qui la prouve, trois formulations naturelles de la
  question et les défauts du wiki rencontrés.

Données brutes : `logs/bench/2026-09-30/` (questions, `normal/`, `lia/` avec chaque résultat
d'outil, `trouvabilite.json`, `navigation.json`, script `depouiller.py`).

## Le jeu : 57 questions

| Catégorie | n | Origine |
| --- | --- | --- |
| Réelles | 36 | Questions d'utilisateurs en base (Technal, Roto NX, profine 70, PERFORM76, LUMINE, garanties, SAV), dont TGY3702 |
| Relances | 7 | Relances réelles, avec les messages qui les précèdent |
| Référence seule | 10 | Fabriquées depuis une ligne de tableau, réponse connue par construction, sans gamme ni fournisseur |
| Sans réponse | 4 | Plausibles, absentes du wiki (vérifié par grep) |

## Résultats

| | Claude imitant LIA | Claude normal |
| --- | --- | --- |
| Réponse équivalente à la référence | **57 / 57** | référence (43 réelles), **14 / 14** sur les réponses connues |
| Abstention juste (wiki muet) | 11 / 11 | 11 / 11 |
| Page attendue lue | 56 / 56 | — |
| Appels / outils par question (médiane) | **2** (1 en référence seule) | 4 |
| Caractères lus par question (médiane) | **116 000** (max 475 000) | quelques lignes ciblées |
| Caractères lus au total | **9,1 millions** | — |

**Trouvabilité mécanique** (rang de la page attendue, sans modèle) : question brute dans le
top 3 pour 48 / 56 ; les 224 formulations naturelles, 189 / 224 (84 %) ; 12 formulations ne la
font pas remonter dans les 15 premières.

## Ce que ça dit

1. **Le wiki d'aujourd'hui est navigable avec les outils de LIA.** Un bon navigateur, limité à
   ces trois outils, trouve tout ce que trouve un accès libre, et sait dire quand le wiki est muet.
   Le cas TGY3702 : un appel, `chercher "TGY3702 …"`, page Technal en tête, TGY3704.
2. **Ce qui manque à LIA n'est donc pas d'abord la trouvabilité, c'est la conduite de la
   recherche** (phase 2) — et **le volume** : 116 000 caractères en médiane (≈ 40 000 tokens) par
   question, jusqu'à 475 000, dont l'essentiel est du bruit. Claude le lit ; un petit modèle s'y
   noie.
3. **Le bruit a une cause précise** : les pages géantes (Roto NX aperçu côté P 124 k, crémones
   141 k, compas 97 k ; Système 70 profilés complémentaires 117 k, plans 129 k) arrivent parmi les
   trois pages entières dès que la requête contient un mot courant (charge, largeur, LFF,
   parclose, crémone, DIN, coffre). En référence seule, la bonne page est **première 10 fois sur
   10**, mais deux pages voisines sans rapport sont livrées avec, 100 000 à 200 000 caractères.

## Comment Claude-LIA navigue (matière de la phase 2)

- **La référence seule d'abord** (`TGY3731`, `SP350`, `76576`) : c'est la requête la plus sûre, et
  la plus rapide pour établir qu'une donnée manque (toutes les pages qui la citent remontent).
- **Pas de facette au premier essai.** La facette `gamme` fait remonter les pages coloris et
  commerciales (R09, R10, R03) ; la recherche sans facette, avec les mots du métier, a mieux
  marché à chaque fois.
- **Toutes les pages livrées sont lues**, ce qui fait voir les jumelles : 76506 / 76507,
  NT1947 sur 76171 (155 mm) ou 76180 (140 mm), 10/14/4 contre 4/14/10.
- **Gamme non précisée → une réponse par gamme** (R01, R30, R31), ou la question posée en retour.
- **Un lien ou un résultat en métadonnées → `lire_page`** ; une anomalie citée dans une page mais
  non injectée → `lire_anomalie` (VER-72).
- **« Absent » seulement après deux ou trois recherches concordantes**, dont la référence seule.

## Les défauts du wiki relevés — et le lot qui les traite

| Défaut | Exemples | Lot |
| --- | --- | --- |
| Pages géantes livrées en bruit | Roto NX aperçu / crémones / compas / accessoires ; Système 70 complémentaires / plans / profilés et renforts | **L2** |
| Référence écrite autrement | `487 206` (deux tokens) contre `487206` ; `TGY3702/03` (la TGY3703 introuvable sur cette ligne) ; `10 / 14 / 4` | **L1** règle 6 : référence entière, sans espace ni abréviation |
| Nom de gamme ou d'objet écrit autrement | `LUMINE65` contre « LUMINE 65 », `PERFORM76` contre « Perform 76 », SHANGHAI contre « Shangai » | **L1** règle 5, **L3** |
| Noms PROFERM absents des pages fournisseur | LUMINE55 = SOLEAL FY, SOLÉAL55 = SOLEAL GY, LUMÉAL55 = LUMEAL GA : les pages Technal ne portent le nom PROFERM qu'en frontmatter (`gamme: LUMINE`) | **L3 / L5** |
| Synonymes du métier | arche / cintre, ralentir / freiner (SoftClose), isolant / isolation, carbone (texture) / bas carbone | **L1** règle 5 |
| Pages source non transcrites alors que la fiche dit « zéro à faire » | SOLEAL GY conception p. 10-21 et 82-85, fabrication p. 108-122 ; SOLEAL FY p. 1-12 (acoustique) ; ASKEY talons et embouts | **nouveau L7** : trous de transcription + fiches sources cohérentes |
| Contradictions non enregistrées | fraisage de la crémone Roto KSR (65 / 28 contre 30 / 20), embout du 2502, AIP36 343 ou 360 mm, W4070495 / 496, VER-32 périmé | **L6** |
| Valeur sans source | « typiquement 24 h à 20 °C » (page ASKEY) | **L6** + lint |
| Anomalie citée dans une page mais non reliée | VER-72 (A491) | **L6** |

Côté code, pour la phase 2 (rien n'est touché maintenant) : `œ` n'est pas converti par l'index
(« œil de bœuf » devient `il / b / uf`, introuvable) ; le statut `draft` n'apparaît qu'à
`lire_page`, pas dans `chercher`.

## Réponses attendues à valider par Elie

Le bras normal a signalé ces cas comme demandant un arbitrage métier (fichiers
`logs/bench/2026-09-30/normal/<id>.json`) : **R03** (garantie structure LUMINE65 : 20 ou 15 ans,
CTR-03), **R05** (aucun diamètre maxi d'œil-de-bœuf), **R09** (pas de parclose 3702), **R10**
(« chicane LUMINE 55 » lu comme les coulissants 55), **R18** (pièce de percussion centrale non
identifiée), **R20** (le « piège » du report de charge), **R22** (fonction domotique inexistante),
**R31** (2452 : 16 mm en 76, 8 / 24 mm en 70), **S01** et **S07** (lecture des relances), **G10**
(343 mm contredit par la fiche source).

## Suite proposée

1. Valider les réponses attendues ci-dessus : le banc devient la référence de mesure.
2. Lancer L1 à L7 dans l'ordre révisé : **L2 (découpe) d'abord** — c'est le volume, puis L1
   (règles), L3 / L5 (noms et facettes), L7 (transcriptions manquantes), L4 (tags), L6 (anomalies).
3. Après chaque lot : trouvabilité rejouée (script, secondes) et Claude-LIA rejoué sur les
   questions touchées ; critère : même taux de réponse, volume lu divisé.
4. Phase 2 : troisième bras, **LIA réelle (Mistral)** sur le même banc, pour mesurer l'écart avec
   Claude-LIA ; puis copier la conduite de recherche décrite plus haut.
