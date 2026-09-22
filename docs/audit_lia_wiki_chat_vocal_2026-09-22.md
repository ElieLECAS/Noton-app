# Audit LIA du 22/09/2026 : wiki, recherche, chat et vocal

Périmètre : `wiki_llm/wiki/` (198 fichiers `.md`, 27 coupes PNG), `wiki_llm/CLAUDE.md`, l'index
`app/services/wiki_index.py`, le tour de chat, le tour vocal, l'application et le dépôt.
Méthode : lecture du code et des pages, lint applicatif, 86 tests (tous verts), une batterie de
40 requêtes sur l'index réel, une évaluation hors ligne de la recherche sur les 62 questions du
golden Perform76 (page attendue fixée à la main), et les 27 tours réellement joués en base.
**Rien n'a été modifié dans le wiki ni dans le code.** Les PDF de `raw/` ne sont pas sur cette
machine (0 fichier) : la fidélité des pages aux documents n'a pas pu être contrôlée.

---

## 0. Verdict

1. **Le fond est solide et l'application fait ce qu'elle promet.** 196 pages OKF, 1 714 liens,
   0 fantôme, 0 page hors index, 3 registres d'anomalies à identifiants stables, un tour de chat
   qui livre les pages entières, vérifie les citations sur disque et injecte les anomalies sans
   dépendre du modèle. Les 20 questions du compte rendu du 22/09 sont justes. C'est rare.
2. **La recherche laisse la bonne page hors du top 3 dans 11 questions sur 62.** Trois causes
   mesurées, toutes dans l'index et pas dans le wiki : « perform 76 » ne trouve pas
   « perform76 », le pluriel ne trouve pas le singulier, et un nombre quelconque (1 300 mm)
   est primé comme une référence. Avec ces trois corrections et une pondération des fiches
   `sources/`, la bonne page est livrée dans **59 cas sur 62** au lieu de 51, sans toucher au
   wiki ni au modèle. Prototype mesuré, prêt à coder.
3. **Les métadonnées de navigation ne sont pas remplies là où elles comptent.** `gamme` manque
   sur 140 pages sur 194, `systeme` sur 139. Aucune des 9 pages PERFORM76 ne porte
   `gamme` ni `systeme` ; les trois pages `systeme-76-*` n'ont pas `systeme: 76`. La facette
   `systeme=76` renvoie donc quatre pages de gamme et aucune page de profilé. Le modèle emploie
   les facettes dans 14 recherches sur 38 : elles ne lui servent à rien aujourd'hui.
4. **Le wiki a été écrit par quatre générateurs en cinq jours et cela se voit.** 1 939 formules
   LaTeX dans 85 fichiers, que ni le chat ni la voix ne rendent ; deux frontmatters illisibles ;
   8 fiches source d'un type `Source` qui n'existe pas dans le protocole, sans registre de
   couverture ; 81 pages dont les ressources pointent vers `a_faire/` ou `raw/moustiquaires/`,
   chemins que le lecteur PDF ne résout pas ; 442 tags dont 221 n'apparaissent que sur une page.
5. **La couverture déclarée n'est pas crédible telle quelle.** 532 plages « transcrit », zéro
   « à faire », mais 85 à 272 caractères de wiki par page de PDF sur les quatre gros manuels
   (395, 424, 371 et 451 pages). Le protocole prévoit la mesure qui tranche (références nommées
   par le PDF et absentes du wiki) ; elle n'a jamais été lancée. Elle doit tourner sur le serveur
   qui a les PDF.
6. **Oui au découpage PERFORM70 / PERFORM76**, non aux autres scissions de gamme. Et trois
   pages à scinder pour la taille, trois familles de contenu à déplacer de `sources/` vers les
   pages concept.

7. **Le wiki est une base de données lue comme de la prose.** Les cotes de débit sont des
   formules, les abaques des bornes, l'AEV par site une matrice à trois entrées, et rien de tout
   cela n'est calculé : c'est récité. Les fonctionnalités qui dépassent le chat et le wiki
   consultable sont au § 9, avec ce qu'elles supposent.

Le plan ordonné est en § 8, les fonctionnalités au § 9. Les corrections d'index (§ 3) sont les
plus rentables : une demi-journée, +8 questions sur 62, aucune écriture de wiki.

---

## 1. Mesures

| Indicateur | Valeur | Lecture |
| --- | --- | --- |
| Pages concept / fiches source / registres | 131 / 62 / 3 | `sources/` = 31 % des pages |
| Liens, fantômes, orphelines | 1 714 / 0 / 18 | les 18 orphelines sont toutes des fiches source |
| Prompt permanent | 13 582 car., ~4 600 tokens | consignes + vocabulaire + index de 77 anomalies |
| Frontmatter : `gamme` / `systeme` / `famille` absents | 140 / 139 / 172 sur 194 | les axes de navigation du protocole |
| Tags distincts / employés une seule fois / montrés au modèle | 442 / 221 / 60 | 17 grappes singulier‑pluriel‑accent |
| `source_pages` absent sur une page concept | 49 | champ « obligatoire » du protocole |
| `verified` renseigné | 8 pages, toutes PERFORM76 | la seconde passe n'a pas commencé ailleurs |
| Frontmatter illisible | 2 | apparaissent comme « Sans type » dans le prompt |
| Descriptions `index.md` ≠ page | 21 | 19 fiches source, 2 pages cassées |
| Formules LaTeX `$…$` | 1 939 dans 85 fichiers | non rendues par le chat, lues telles quelles par la voix |
| Fiches source sans « Registre de couverture » | 11 | 8 Soprofen (`type: Source`), 2 posters, plans e.VOLUTION |
| États des registres | 532 transcrit, 51 sans contenu propre, 0 à faire | aucun manque déclaré sur 32 documents |
| Densité de transcription, gros manuels | 85 à 272 car. par page PDF | e.VOLUTION 395 p., 76 Advanced 424 p., 70 Plateforme 371 p., Roto NX 451 p. |
| Ressources non résolvables par le lecteur PDF | 215 entrées, 81 pages | `a_faire/…`, `wiki_llm/a_faire/…`, `raw/moustiquaires/…`, `.zip` |
| Golden 62 questions : bonne page livrée entière (top 3) | **51 / 62** | rang médian 1, p90 = 10 |
| Idem avec l'index corrigé (§ 3.3) | **59 / 62** | p90 = 3 ; fiches source dans le top 10 : 2,0 → 0,6 par question |
| Tours réels en base | 27 (7 chat, 9 vocal) | 38 `chercher`, 4 `lire_page` |
| Tours avec une recherche répétée à l'identique | 4 / 27 | « collection lumière » cherché 4 fois dans un tour |
| Tokens d'entrée par tour, p50 / p90 / max | 27 900 / 57 200 / 124 900 | dominés par les pages livrées |
| Durée d'un tour, p50 / p90 | 5,3 s / 9,3 s | |
| Tests | 86 verts | aucun test de régression de la recherche |

---

## 2. Le wiki : ce qui est bon

- **Les pages de profilés PERFORM76 et les pages système 76** : une ligne par référence, unité
  en en‑tête, contexte en colonne, coupes vérifiées, familles jumelles séparées (parclose
  d'ouvrant / de dormant), matrice vitrage‑parclose‑joint‑cale du registre 2.3.2. C'est le
  modèle à suivre, et c'est là que les 8 `verified` sont.
- **Les registres d'anomalies** : 77 entrées, verbatim, correction probable, impact. Cités
  spontanément dans 6 réponses sur 20 le 22/09. La règle 2 tient parce que le serveur la
  porte.
- **La table de correspondance produit PROFERM ↔ système TECHNAL** de `fournisseurs/technal.md`
  (LUMINE55 = SOLEAL FY 55, GALANDAGE55 = SOLEAL GY 55, LUMÉAL55 = LUMEAL GA, portes =
  SOLEAL PY 55). C'est la clé de voûte de tout le côté aluminium. Elle n'est écrite qu'à cet
  endroit (§ 4.2).
- **Le registre d'écriture a été redressé** depuis l'audit du 18/09 : 18 tournures bannies
  restantes sur les pages assertives (166 il y a quatre jours), 6 % de recouvrement médian entre
  une fiche source et les pages concept (le « rapport de lecture » a disparu).
- **Le glossaire** et les tables de référence (régions climatiques, AEV par site, résistance au
  vent) : c'est ce qui permet de répondre à « Lille » ou au « 59 ».

---

## 3. La recherche : le plus rentable à corriger

### 3.1 Trois défauts mesurés dans `wiki_index.py`

**a. « perform 76 » ≠ « perform76 ».** Le tokeniseur garde « perform76 » entier. Une question
qui écrit « Perform 76 » (c'est ainsi que les 62 questions du golden l'écrivent, et le vocal
transcrit « performe 76 ») ne touche ni le titre ni les tags des 9 pages PERFORM76.

| Requête | Aujourd'hui : rang de la bonne page | Après correction |
| --- | --- | --- |
| `dormant perform 76` | **absente du top 10** (livrées : gammes/perform, innoslide, kommerling) | 1 |
| `meneaux perform 76` | 5 | 1 |
| `hauteur poignée perform 76` | 3 | 1 |
| `Hauteur de poignée pour un ouvrant de 2000 mm sur Perform 76` (golden 016) | absente du top 10, 3 pages Roto NX livrées | livrée (top 3) |

« perform76 » figure dans 20 titres ou descriptions, et une trentaine d'autres composés du même
type les accompagnent (lumine65, py1100, rc3, soléal55, galandage55, luméal55). Correction : émettre, en plus du mot entier, ses deux parties
(« perform », « 76 ») à l'indexation et à la requête.

**b. Pas de repli singulier‑pluriel.** « meneaux » ne trouve pas « meneau », « parcloses »
descend `soleal-fy-parcloses-et-vitrage` au rang 4 derrière une fiche source. Correction :
replier le `s`/`x` final des mots de plus de quatre lettres, des deux côtés (index et requête).
Les collisions sont neutres puisque le repli est symétrique.

**c. Toute suite de chiffres est primée ×3 comme une référence.** « 1300 » (une hauteur),
« 2000 », « 450 » reçoivent le même bonus que « 76507 ». C'est ce qui fait remonter deux pages
Roto NX KSR sur « crémone coulissant clé 1300 » (le bruit noté au § 4.4 du compte rendu du 22/09)
et qui envoie `roto-nx-ksr-cintre-trapezoidal` en tête sur « hauteur de poignée pour un ouvrant
de 700 mm ». Une règle simple suffit : lettres + chiffres (A076, TGY3704, NT1947) = référence ;
5 chiffres ou plus (76507) = référence ; 4 chiffres ne finissant pas par 0 (2452, 6137, 8355) =
référence ; le reste (1300, 2000, 140, 44) = mot ordinaire. Mesuré : c'est cette règle qui
apporte le plus (+3 questions à elle seule, p90 de rang 5 → 3).

### 3.2 Les fiches `sources/` saturent le top 10

En moyenne **2 fiches source par top 10** sur les 62 questions ; jusqu'à 6 sur
« garantie structure », 3 sur « seuil PMR », 3 sur « volet roulant soprofen ». `trier_resultats`
les fait passer après les pages concept, mais elles ont déjà pris des places dans les dix
candidats, et la bonne page concept peut être 11ᵉ. Une fiche source répond rarement à un
menuisier ; quand c'est le cas (« que couvre le DTA ? »), la page `certifications/` répond
mieux. Correction : score ×0,5 pour `/sources/` et score nul pour `/anomalies/` **avant** la
coupe à `limite`. Mesuré : fiches source dans le top 10 de 2,0 à 0,6 par question, sans perte.

### 3.3 Résultat du prototype (hors ligne, index réel, 62 questions du golden)

| Variante | Bonne page livrée entière (top 3) | Rang p90 | Sources / top 10 |
| --- | --- | --- | --- |
| V0 aujourd'hui | 51 / 62 | 10 | 2,0 |
| + découpage des mots composés | 55 / 62 | 5 | 2,3 |
| + repli du pluriel | 56 / 62 | 5 | 2,7 |
| + règle des références | **59 / 62** | 3 | 2,6 |
| + sources ×0,5, anomalies exclues | **59 / 62** | 3 | **0,6** |
| sources ×0,5 seul (sans le reste) | 51 / 62 | 8 | 0,5 |

Livrer 4 ou 5 pages entières au lieu de 3 n'apporte rien une fois l'index corrigé (59 dans les
trois cas) : `PAGES_COMPLETES = 3` est le bon réglage. Les trois questions qui restent en échec
sont dures par nature : « poignée pour un petit ouvrant de 450 mm » sans nommer la gamme,
« Uw de la gamme Lumine » que `coulissants-aluminium` capte, « parclose pour 52 mm » dont la
valeur n'existe sur aucune page. Sur mes 18 sondes libres : 16 → 17 livrées, et 7 de plus passent au
rang 1.

Le script d'évaluation (variantes, golden, sondes) est écrit ; il a vocation à devenir
`app/scripts/eval_recherche.py` et un test de régression `tests/test_wiki_recherche_golden.py`
avec le seuil « ≥ 58 / 62 ». Aujourd'hui rien ne protège la recherche d'une régression.

### 3.4 Autres corrections d'outil, petites et sûres

- **Livrer chaque page avec son en‑tête.** `chercher` renvoie le corps sans frontmatter, et
  173 pages sur 194 ne commencent pas par leur titre. Le modèle lit donc « # Comment choisir
  une parclose » sans savoir que la page est « Parcloses PERFORM76, type Profilé ». Une ligne
  `titre · type · gamme · système · description` en tête de chaque page livrée coûte trente
  tokens et lève l'ambiguïté des familles jumelles. `lire_page`, lui, renvoie le frontmatter
  YAML entier (sources, generated…) : unifier les deux rendus.
- **Ne pas rejouer une recherche identique.** 4 tours sur 27 répètent la même requête, jusqu'à
  quatre fois « collection lumière » : chaque répétition relivre 30 000 caractères. Renvoyer
  « recherche identique déjà rendue ci‑dessus » ; même chose pour `lire_page` sur une page
  déjà livrée entière.
- **L'index des anomalies montre la source au lieu du sujet pour INC et VER.** Le chargeur
  prend la deuxième colonne ; or `incoherences-internes.md` a deux tables (`ID | Source | Page |
  Constat…` et `ID | Retirée le | …`) et `informations-a-verifier.md` en a trois. Le prompt
  affiche « INC-03 | Cahier technique PERFORM76 » quatre fois de suite et « INC-01 |
  2026-09-18 ». Le rapprochement serveur n'en souffre pas (il lit la ligne entière), mais le
  modèle, lui, ne peut pas juger la pertinence d'une entrée à partir de cet index. Deux
  corrections complémentaires : le chargeur choisit la colonne par son en‑tête (« Sujet », sinon
  « Constat » tronqué), et le protocole impose une colonne « Sujet » aux trois registres.
- **La relance ne devrait pas s'appliquer à « merci ».** Elle a forcé une recherche du mot
  « merci ». Exempter les messages de moins de trois mots sans chiffre.
- **Le vocabulaire du prompt** liste « Sans type (2) » (les deux frontmatters cassés),
  « Profilés complémentaires » comme gamme, et « Frappe 65 » à côté de « Frappe 65 Ouvrant
  Caché ». Il se corrige par le wiki (§ 4.2), pas par le code.

---

## 4. Le wiki : structure, découpage, métadonnées

### 4.1 Scinder PERFORM en trois pages : oui

`gammes/perform.md` porte les deux déclinaisons avec `systeme: [70, 76]`. Or le corpus est
asymétrique : la PERFORM76 a 7 pages de profilés, 3 pages système, une pose, une quincaillerie,
27 coupes ; la PERFORM70 n'a rien au‑delà de l'épaisseur, du Uw, de l'AEV, du cintrage et du
laquage, et son rattachement au système 70 Plateforme n'est pas établi (VER‑28). Un lecteur qui
arrive sur la page pour la 70 lit une page à 80 % sur la 76. Le modèle s'en est sorti le 22/09
(question 15, « Non spécifiée » + VER‑03), mais c'est une réponse juste malgré la page, pas
grâce à elle.

Découpage proposé, sans dupliquer une donnée :

| Page | Contenu | Rôle |
| --- | --- | --- |
| `gammes/perform.md` | GREENLINE, dix équipements de série, accessoires et poignées, coloris, croisillons, garanties, restrictions communes | le tronc commun, déjà là |
| `gammes/perform70.md` (nouvelle) | 70 mm, Uw 1,3, A\*4/E\*9A/V\*A3, cintrage autorisé, laquage sous conditions, chambres et joints **non spécifiés** (VER‑03), système fournisseur **non établi** (VER‑28) avec le lien vers les pages système 70 assorti de cette réserve | dire clairement ce qu'on ne sait pas |
| `gammes/perform76.md` (nouvelle) | 76 mm, 6 chambres, 3 joints, Uw 0,8, RC2 CERIBOIS, FFCP, dimensions limites du DTA, **cintrage et laquage interdits**, et la carte des 12 pages techniques (profilés, système 76, pose, poignée) | le carrefour que « perform76 » doit trouver |

Bénéfices mesurables : la facette `gamme` prend les valeurs PERFORM70 et PERFORM76 ; une
question « cintrage perform76 » livre une page qui dit « impossible » au lieu d'une page qui
dit « autorisé sur 70, impossible sur 76 » à interpréter ; les 9 pages PERFORM76 gagnent un
parent naturel.

**Ne pas scinder HYBRIDE ni LUMINE.** Leurs déclinaisons tiennent dans un tableau et le corpus
est mince des deux côtés. En revanche, **la table produit PROFERM ↔ système TECHNAL doit vivre
sur `gammes/lumine.md`** (et être pointée depuis `coulissants-aluminium.md`), pas seulement sur
la fiche fournisseur : c'est elle qui permet de passer de « parclose LUMINE65 » à
`soleal-fy-parcloses-et-vitrage`. Aujourd'hui `gamme: LUMINE` posé sur les 33 pages Technal
joue ce rôle en creux.

### 4.2 Le modèle de métadonnées à trois axes, et le remplir

Le protocole dit « une page est atteinte par son `type`, son `systeme` et sa `gamme` ». Dans les
faits, `systeme` est un nombre (55, 65, 70, 76, 100) qui confond LUMINE65, ASKEY 65 et
SOLEAL FY 65, et vaut 100 pour LUMEAL GA (son module) ; `gamme` mélange des noms PROFERM
(PERFORM, LUMINE) et des noms de systèmes fournisseur (« Coulissant 65 NV », « Frappe 65 Ouvrant
Caché », « Profilés complémentaires »). Proposition, à écrire dans `CLAUDE.md` et à vérifier par
le lint :

| Axe | Sens | Valeurs |
| --- | --- | --- |
| `gamme` | le nom commercial PROFERM, tel que le catalogue le vend | PERFORM70, PERFORM76, PERFORM+, HYBRIDE70, HYBRIDE76, HYBRIDE+, TEXTURAL, INNOSLIDE, LUMINE55, LUMINE65, SOLÉAL55, GALANDAGE55, LUMÉAL55, porte 97 / 118 / SOLEAL 100, collections de portes |
| `systeme` | le système du fournisseur, en clair | 76 Advanced, 70 Plateforme, SOLEAL FY 55, SOLEAL FY 65 QC, SOLEAL GY 55, LUMEAL GA, SOLEAL PY 55, ASKEY Coulissant 65 NV, ASKEY Frappe 65 OC, ASKEY Frappe 65 OV, Roto NX, Roto Safe E, Patio Inowa, Chrono One, Bloc LX… |
| `famille` | la famille de pièces ou de contenu | dormants, ouvrants, parcloses, meneaux, tapées, appuis, renforts, cotes‑de‑débit, abaques, crémones, compas, gâches, pose, fabrication, entretien… |

Une page ASKEY n'a pas de `gamme` (aucun document PROFERM ne dit quel produit l'emploie) ; elle
a un `systeme`. Le remplissage concerne les 131 pages concept ; les 14 pages PERFORM76 et
système 70/76 d'abord, parce que ce sont celles que les questions visent. La facette
`systeme=76` passe alors de 4 pages à 16, et « une facette remonte, elle n'exclut pas » devient
un levier réel.

### 4.3 Les tags : 442, dont la moitié sur une seule page

Le modèle n'en voit que 60. Dix‑sept grappes ne diffèrent que par un `s` ou un accent
(`profile` / `profilé` / `profilés`, `renfort` / `renforts`, `poignee` / `poignees` / `poignées`,
`etancheite` / `étanchéité`, `procedure` / `procédure`). 221 tags n'existent que sur une page :
`ozroll`, `chenilles`, `cyclone`, `iz`, `blanc`… ce sont des mots du corps, que BM25 trouve
déjà. Proposition : un vocabulaire contrôlé d'environ 80 tags, au singulier, sans accent, écrit
dans `CLAUDE.md`, refusé par le lint au‑delà ; les 80 tiennent en entier dans le prompt. Les
familles jumelles y ont leur place (`parclose-ouvrant`, `parclose-dormant`, `meneau-ouvrant`,
`meneau-dormant`), c'est la seule chose qu'un tag fait mieux que le texte intégral.

### 4.4 `sources/` : un tiers du wiki, trois problèmes

- **Deux types pour la même chose.** 53 pages `Document source`, 8 pages `Source` (les fiches
  Soprofen du 21/09). Le protocole ne connaît que la première. Les 8 n'ont ni « Identité » ni
  « Registre de couverture » ; elles sont écrites en « Registre d'analyse page par page », avec
  le contenu technique dedans : le guide Bloc LX (20 000 caractères, 135 formules LaTeX) porte
  des cotes de fabrication que la page `equipements/volet-roulant-bloc-lx-demi-linteau.md`
  (13 800 caractères) ne porte pas toutes. C'est du contenu de concept enfermé dans une fiche
  source, exactement ce que l'audit du 18/09 relevait pour les deux DTD, qui sont toujours à
  24 000 et 22 500 caractères avec leurs sections « Données techniques de référence » (renforts
  du 76, parcloses, tapées, dormants larges du 70). Destination : `equipements/`,
  `certifications/`, `profiles/systeme-76-renforts`, `perform76-parcloses`,
  `systeme-70-profiles-et-renforts`. La fiche garde l'identité et le registre.
- **215 ressources que le lecteur ne résout pas.** `raw_path` n'accepte que `raw/<fichier>.pdf`
  à plat ; 135 entrées pointent vers `a_faire/…` ou `wiki_llm/a_faire/…`, 80 vers
  `raw/moustiquaires/…`, certaines vers des `.zip`. 81 pages concernées, et au moins 24 locators
  `(schéma: …)` sur 404. Sur ces pages, le clic « ouvrir le PDF à la planche » ne peut pas
  marcher. Deux options : déplacer les PDF à plat dans `raw/` et corriger les chemins (le
  protocole le demande), ou accepter des sous‑dossiers dans `raw_path`. La première est la
  bonne : `a_faire/` n'est pas une provenance.
- **La couverture déclarée est un drapeau rouge, pas une preuve.** 0 « à faire » sur 32
  documents, et pourtant 85 caractères de wiki par page pour les 395 planches e.VOLUTION,
  194 pour les 371 pages du 70 Plateforme, 245 pour les 424 pages du 76 Advanced, 272 pour les
  451 pages du catalogue Roto NX. Un catalogue de commande porte un tableau par page : 250
  caractères, c'est trois lignes. Je ne peux pas trancher sans les PDF, mais le protocole a
  déjà la mesure qui tranche (« pour chaque page de chaque PDF, les références qu'elle nomme et
  qui n'apparaissent nulle part dans le wiki »). Elle n'a jamais tourné. À lancer sur le
  serveur, document par document, en commençant par ces quatre‑là ; le résultat décide si les
  plages repassent en « en cours ».

### 4.5 Scinder pour la taille, fusionner pour le sens

À scinder, le long des axes de la source (une page livrée entière pèse jusqu'à 8 000 tokens) :

| Page | Taille | Découpage |
| --- | --- | --- |
| `quincaillerie/roto-nx-cremones.md` | 24 900 car., 259 lignes de tableau | crémones OB (fixe, variable, EasyMix) · crémones à sortie de tringle et semi‑fixe · serrures H100 de porte‑fenêtre |
| `profiles/systeme-70-profiles-et-renforts.md` | 21 000 car., `draft` | dormants et capots · ouvrants, battements, meneaux · complémentaires et références partagées, sur le modèle des pages PERFORM76 |
| `quincaillerie/roto-nx-ksr-oscillo-battant.md` | 18 800 car. | un vantail · deux vantaux · positionnement des gâches |

À fusionner : `quincaillerie/serrure-motorisee.md` (3 300 car., la vue catalogue) dans
`controle-acces-eneo-cc.md` (la notice) ; ce sont deux vues du même Safe E Eneo CC, la jonction
de câble reste à part. À alléger : `fournisseurs/kommerling.md` reprend la table des deux
systèmes profine qui appartient à `profine.md` ; garder KÖMMERLING comme porte d'entrée (c'est
le nom du catalogue), en une page courte. `equipements/volets-roulants-tradi-non-premonte.md`
et `tradi-tunnel-vtr.md` partagent 10 lignes de tableau (motorisations et lames) : une seule
page les porte, l'autre renvoie.

### 4.6 Conformité au protocole

- **Squelettes.** Quincaillerie : « Champs d'application » sur 4 pages sur 29, « Compatibilités »
  sur 2. Procédure : « Étapes » sur 10 sur 21, « Conditions et interdictions » sur 8. Profilé :
  « Cotes » sur 20 sur 29, « Compatibilités » sur 14. Les exclusions (« ne se monte pas sur »)
  sont la ligne qui évite une mauvaise commande ; elles manquent sur la moitié des pages qui en
  auraient besoin.
- **LaTeX.** 1 939 formules dans 85 fichiers, concentrées sur les pages écrites du 19 au 21/09
  (`soleal-fy-cotes-de-debit` 162, `soleal-gy-cotes-de-debit` 126, `soleal-py-cotes-de-debit`
  84, `lumeal-ga-cotes-de-debit` 82, les fiches Soprofen). Le chat les affiche brutes
  (`$4,5 \times 30\text{ mm}$`, défaut n° 2 du compte rendu du 22/09 ; le modèle les recopie
  depuis la page), l'index les découpe en tokens `times`, `text`, `pm`, la voix les lit. Une
  passe de conversion (`\text{ mm}` → « mm », `\times` → « × », `\varnothing` → « Ø », `\pm` →
  « ± », `\ge` → « ≥ », `_{\min}` → « min ») plus une règle « pas de LaTeX » dans `CLAUDE.md`
  et un contrôle du lint. 100 `<br>` à traiter de même.
- **Frontmatter.** 2 YAML illisibles (`moustiquaires-battantes-coulissantes-fixes.md` et sa
  fiche source : un `:` dans une valeur non citée). 49 pages concept sans `source_pages`.
  3 `sources[]` sans `last_modified`. 21 descriptions qui ne correspondent plus à `index.md`.
  `generated.by` : 89 claude‑code, 55 gemini‑coder, 40 multimodal‑direct, 7 multimodal‑okf.
- **Ce que le lint applicatif ne voit pas** et devrait voir : la dérive `index.md` ↔
  `description`, `source_pages` absent, un `type` hors vocabulaire, un tag hors vocabulaire,
  `gamme`/`systeme` absents sur une page `Profilé` ou `Quincaillerie`, une ressource hors
  `raw/<fichier>.pdf`, un `$…$`, un registre de couverture absent ou lacunaire.

### 4.7 Ce que les usages réels demandent et que le wiki n'a pas

- **Les prix.** « LUMINE prix tarif », « fremi prix » : `tarifs/` est prévu par le protocole et
  vide. Soit on verse les tarifs, soit une page dit une fois pour toutes que le wiki n'en porte
  pas et vers qui se tourner.
- **Les normes.** NF EN 13659 (19 mentions), NF DTU 36.5 (21), EN 356 (11), NF EN 14501 (9),
  NF EN 14024 (9), EN 1627‑30 (9), NF EN 1670 (6), DIN EN 13126 (5)… citées sur des dizaines de
  pages, sans page cible. `reference/` en couvre trois (EN 12211, AEV par site, régions
  climatiques). Le dossier `normes/` du protocole est toujours vide.
- **Les coupes.** 27 PNG, toutes des parcloses PERFORM76. Les dormants, ouvrants, meneaux,
  tapées et appuis ont leurs planches au cahier technique ; le filtre serveur des coupes est déjà
  écrit et n'attend que les fichiers.
- **PERFORM70** : voir § 4.1. La page qui dit « rien n'est documenté au‑delà de… » vaut mieux
  que l'absence.

---

## 5. Le chat

- **Le tour est bien construit** : trois outils, six allers‑retours, pages livrées entières,
  anomalies injectées côté serveur, coupes vérifiées sur disque, citations résolues, relance
  unique quand rien n'a été lu. 0 citation inconnue sur les 27 tours en base, 1 tour sans page
  lue (la relance a servi). Les consignes ont absorbé la leçon du 22/09 (« ne raconte pas ta
  procédure »).
- **Le LaTeX n'est pas rendu** (§ 4.6) : à traiter à la source dans le wiki ; en attendant,
  `parseMarkdown` peut convertir les `$…$` en texte.
- **Deux moteurs de recherche.** `/api/wiki/search` (l'accueil de `/wiki`) fait un ET de
  sous‑chaînes ; `chercher` fait du BM25. Une référence tapée à l'accueil ne se classe pas comme
  dans le chat. Exposer `index.search` à l'accueil unifie les deux et fait profiter l'humain des
  corrections du § 3.
- **`/assets` est monté sans authentification** : les 27 coupes sont servies à quiconque connaît
  l'URL. Peu grave, mais incohérent avec `/api/wiki/pages` qui exige la session.
- **Les retours 👍/👎 n'alimentent rien.** Ils sont stockés, comptés dans l'admin, jamais
  rapprochés des pages citées. Un export « 👎 + question + pages livrées + anomalies » est la
  liste de travail du wiki la plus directe qui soit ; les traces le permettent déjà.
- **Les traces sont riches et inexploitées** : `steps`, `pages_lues`, `cited_pages`,
  `unknown_citations`, tokens, durée sur chaque message. Le tableau du § 1 en sort en dix
  lignes ; une carte admin « recherche » (pages les plus livrées, recherches répétées, tours
  sans citation) coûte peu.

## 6. Le vocal

- **L'architecture tient** : transcription par lots avec biais, même `WikiAnswer`, synthèse
  phrase par phrase, rien dit avant qu'une page soit chargée, `texte_parle` distinct du texte
  affiché. 9 conversations vocales en base, le tour se termine avec ses sources.
- **Le biais de vocabulaire est rempli aux deux tiers par des tags génériques.** Sur 100 termes :
  « source », « reference », « fabrication », « fournisseur », « conception », « certification »,
  « porte », « profile », « profilés », « 55mm », « atelier »… Aucun homophone dangereux n'y
  gagne. Manquent LUMÉAL, SOLÉAL, GALANDAGE, GREENLINE, Kömmerling (présent en minuscules),
  TROCAL, KBE, Designo, Patio Inowa, Safe E, Eneo, Chrono, Bloc LX, GoodNight, battement,
  feuillure, clameau, élargisseur, appui, soufflet, imposte, galbé. Remplacer « tags les plus
  fréquents » par une liste curée dans `vocal_service.py` : noms de produits et de systèmes,
  familles de pièces, unités.
- **`texte_parle` ne traite pas le LaTeX** : une page qui en contient peut faire dire « dollar
  backslash text » à Marie. La purge du wiki règle le cas ; un filet `\$[^$]+\$` → texte dans
  `texte_parle` coûte une ligne.
- Le micro réel n'a toujours pas été essayé de bout en bout (mémoire de projet) : c'est la
  seule vérification qui manque au vocal.

## 7. L'application et le dépôt

- 86 tests verts, 13 s. Couverture correcte du tour, des coupes, du vocal, du RBAC. **Aucun
  test ne protège la recherche** ; plusieurs tests d'index dépendent du contenu réel du wiki
  (« 76507 → parcloses »), ce qui est voulu mais fragile à une réécriture.
- `.env` porte une cinquantaine de clés mortes de l'ancien système (OPENAI, KAG, EMBEDDING,
  RERANKER, DOCLING, CELERY, LANGSMITH…) que `config.py` ignore. `.env_exemple` est propre ;
  aligner `.env` évite de croire qu'un réglage agit.
- Fichiers suivis par git sans rôle : `metadata.json` (export de chunks d'un document
  Soprofen de l'ancien système, 29 Ko), `push.txt`, `powershell.cmd`. `media/` (racine, root)
  n'est pas suivi.
- `@app.on_event("startup")` déprécié ; `fastapi==0.104.1` (fin 2023), `sqlmodel==0.0.14`,
  `uvicorn 0.24` : à remonter avant la production, avec la suite de tests comme filet.
- Docker : image minimale, wiki monté en volume, migrations idempotentes : bien.

---

## 8. Plan d'action

**P0 : cette semaine, sans réécrire le wiki**

1. `wiki_index.py` : découpage des mots composés, repli du pluriel, règle des références,
   sources ×0,5, anomalies hors candidats, en‑tête sur chaque page livrée, refus des recherches
   et lectures répétées, colonne « Sujet » par en‑tête. Mesuré : 51 → 59 / 62. Ajouter
   `eval_recherche.py` et le test de régression à 58 / 62.
2. Wiki, dix minutes : réparer les 2 frontmatters, passer les 8 `Source` en `Document source`,
   poser `gamme` et `systeme` sur les 14 pages PERFORM76 et système 70 / 76.
3. Consignes : « pas de LaTeX, écris 4,5 × 30 mm », « ne relance pas une recherche identique ».
   Chat : convertir les `$…$` au rendu ; relance exemptée des messages de courtoisie.

**P1 : le wiki, deux ou trois sessions**

4. Purger le LaTeX et les `<br>` des 85 fichiers, avec relecture des tableaux touchés.
5. Scinder PERFORM en `perform.md`, `perform70.md`, `perform76.md` ; porter la table
   produit ↔ système sur `lumine.md`.
6. Sortir le contenu technique des fiches Soprofen et des deux DTD vers les pages concept ;
   donner aux 11 fiches sans registre un registre conforme.
7. Déplacer les PDF de `a_faire/` et `raw/moustiquaires/` à plat dans `raw/`, corriger les 215
   ressources et les locators concernés.
8. Écrire dans `CLAUDE.md` le modèle `gamme` / `systeme` / `famille` et le vocabulaire de 80
   tags ; les remplir sur les 131 pages concept ; étendre le lint (§ 4.6).

**P2 : ensuite**

9. Scinder `roto-nx-cremones`, `roto-nx-ksr-oscillo-battant`, `systeme-70-profiles-et-renforts` ;
   fusionner `serrure-motorisee` dans `controle-acces-eneo-cc`.
10. Sur le serveur qui a les PDF : la mesure de couverture par références du protocole sur les
    quatre gros manuels, puis la relecture inverse (`verified`) au‑delà des 8 pages PERFORM76.
11. `normes/` (une page par norme citée), une réponse aux tarifs, les coupes des dormants,
    ouvrants, meneaux, tapées et appuis.
12. Export des 👎 vers une liste de corrections ; biais vocal curé ; recherche unifiée à
    l'accueil du wiki ; `.env` nettoyé ; dépendances remontées.

Ce plan remet le chat au niveau que les pages permettent. Il ne crée aucune fonctionnalité : ce
qui va au‑delà du chat et du wiki consultable est au § 9, et deux de ces entrées y trouvent leur
prolongement — la veille des validités (§ 9.5) est le débouché du `stale_after` posé en P1, et
l'export des 👎 (P2, entrée 12) est la première brique de la boucle usage vers wiki.

## 9. Au-delà du chat : ce que les tableaux permettent de construire

Le wiki n'est pas un corpus de texte, c'est une base de données lue comme de la prose. Le
protocole impose une ligne par référence et l'unité en en‑tête : les cotes de débit sont des
formules, les abaques sont des bornes, l'AEV par site est une matrice à trois entrées. Aujourd'hui
le modèle lit ces tableaux et les récite. Les fonctionnalités qui suivent les **calculent**. C'est
la différence entre un assistant qui répond et un outil qui décide, et c'est exactement là où un
petit modèle est le plus faible : croiser quatre tableaux sans se tromper de ligne.

### 9.1 Le vérificateur de faisabilité

**Ce qu'il fait.** On saisit une configuration — gamme, dimensions hors tout, type d'ouverture,
épaisseur de vitrage, couleur du profilé, épaisseur d'isolant — et il répond **réalisable**,
**hors domaine d'emploi** ou **sur étude**, en nommant la contrainte qui mord et la page qui la
porte. Pas une réponse rédigée : un verdict, ses bornes, ses citations.

**Ce qui existe déjà**, et qu'il suffit de croiser :

| Contrainte | Où elle est écrite |
| --- | --- |
| Dimensions maximales de baie par configuration d'ouverture | `/certifications/dta-6-16-2334.md`, reprises sur `/gammes/perform.md` |
| Cotes d'ouvrant maximales par renfort | `/profiles/systeme-76-abaques-dimensionnels.md` |
| Renforcement total obligatoire au‑delà de 12 mm de verre ; courbes 12, 16, 20, 24, 28 mm, arrondi à la courbe supérieure | idem |
| Renfort systématique en couleur et en IR‑Reflex, préconisé en blanc | idem |
| Largeur, hauteur et poids de vantail admissibles par type d'ouverture et classe de sécurité, côtés P et Designo II, plus la conversion épaisseur de vitrage → poids | `/quincaillerie/roto-nx-champs-application.md` |
| Épaisseur de vitrage plafonnée à 50 mm par le DTA, parclose correspondante | `/profiles/perform76-parcloses.md` |
| Charge du pivot bas, 100 kg (CTR‑01) | `/quincaillerie/perform76-poignee-et-pivot.md` |
| Isolant admissible par couple dormant et tapée | `/profiles/perform76-tapees-et-isolation.md` |

**Ce qui manque.** Les tableaux sous forme de données (§ 9.6) et un moteur de règles déterministe.
Le modèle n'intervient que pour remplir le formulaire depuis une phrase et pour rédiger le verdict.
**Aucune valeur ne doit sortir du modèle.** Une case que le wiki laisse à `-` reste indécidable :
le verdict est alors « sur étude », jamais une interpolation.

**Pourquoi en premier.** C'est la question qui coûte de l'argent quand la réponse est fausse, et
c'est celle où le tour de chat est structurellement le plus exposé. Les échecs du golden du 18/09
étaient déjà des croisements de tableaux.

### 9.2 Le calcul de débit et la nomenclature

`/profiles/systeme-76-cotes-de-debit.md` écrit la méthode en toutes lettres : ce sont des **cotes
à déduire, pour une seule coupe**, avec l'exemple chiffré du manuel (DHT 2 000 × 1 200, dormant
76171, meneau 76372 : DEO = 1 000 − (38 + 13) = 949, vitrage = 949 − 2 × 60 = 829) et la table de
déduction des sept dormants, colonne par colonne — DEO, DFO, vitrage fixe, renfort, meneau,
renfort de meneau, parclose. Le système 70, l'ASKEY, le SOLEAL FY, le GY, le PY et le LUMEAL GA
ont chacun la leur.

Saisir la dimension hors tout et la configuration rend les cotes de coupe et la liste des
références à commander. Deux pièges que le calculateur doit hériter du wiki plutôt que les
lisser : le renfort du 76173 ne se débite pas symétriquement (75 mm en haut, 45 vers le bas), et
trois des sept dormants du système ne sont pas proposés par PROFERM. C'est l'outil que l'atelier
ouvrira tous les jours, et c'est aussi le meilleur test de fidélité du wiki : une cote fausse s'y
voit en une journée.

### 9.3 La prescription par chantier

Trois tables déjà présentes se composent sans aucun modèle :

1. `/reference/regions-climatiques-par-departement.md` : département, et découpage cantonal des
   départements partagés, vers la région 1 à 4.
2. `/reference/classification-aev-par-site.md` : région, catégorie de terrain (0, II, IIIa, IIIb,
   IV) et hauteur du bâtiment en cinq tranches, vers la classe A\*E\*V minimale, avec la réduction
   pour ouvrage protégé.
3. Les classements des gammes (`/gammes/perform.md` en A\*4/E\*9A/V\*A3,
   `/certifications/labels-et-certifications.md`, les pages LUMINE et coulissants) : qui atteint
   la classe exigée.

Le même enchaînement existe pour les fermetures avec
`/reference/resistance-au-vent-volets-soprofen.md` et le DTU 34‑2. Entrée : un code postal, un
terrain, une hauteur. Sortie : la classe exigée et les produits qui la tiennent. C'est une réponse
que le commerce donne aujourd'hui au jugé.

### 9.4 La fiche d'identité d'une référence

Taper `76758` et obtenir sa **famille** (un appui, pas une tapée), sa page, ses cotes, ses
compatibilités, ses exclusions et sa coupe quand elle existe. L'index construit pour cet audit
recense **1 189 références** en première colonne de tableau dans tout le wiki ; la table est déjà
calculée à chaque chargement. C'est le moins cher de la liste, et c'est ce qui referme
définitivement la confusion tapée / appui / patte de pose que la règle 9 des consignes essaie de
tenir par le texte. À exposer dans les deux sens : la fiche pour l'humain, et un outil
`fiche_reference(ref)` pour le modèle, qui lui évite de charger 30 000 caractères pour une ligne.

### 9.5 Rendre le wiki vivant plutôt que consultable

- **Veille réglementaire.** Le DTA SOLEAL FY est valide **jusqu'au 31 octobre 2026**, soit cinq
  semaines, et il couvre toute la LUMINE55 battante. Rien ne le signalera : `stale_after` n'est
  posé que sur cinq pages et pas sur celle‑là, et le lint des pages périmées renvoie une liste
  vide. Les autres échéances sont l'Avis Technique des coffres Chrono au 31/01/2027, le DTA 76
  Advanced au 31/07/2028, les DTA LUMEAL GA et SOLEAL GY au 31/07/2029. Un tableau de bord des
  validités et une alerte à six mois relèvent du risque contractuel, pas du confort. Prérequis :
  poser `stale_after` sur chaque page qui porte une validité réelle, ce que le protocole demande
  déjà.
- **Le registre d'anomalies comme flux de travail.** 77 entrées avec leur impact et la personne à
  interroger, figées dans du markdown en lecture seule. Leur donner un responsable, un état, une
  date et la réponse obtenue en fait la liste de dettes techniques de l'entreprise. Les classer
  par le nombre de fois où elles sont réellement citées en réponse donne l'ordre de traitement :
  sur 27 tours enregistrés, CTR‑03 est sortie cinq fois.
- **La boucle usage vers wiki.** Les pouces baissés, les 8 tours sur 27 qui n'ont cité aucune
  page, les « le wiki ne couvre pas », les recherches répétées : tout est déjà dans les traces des
  messages et exploité nulle part. C'est la file d'ingestion, dérivée de l'usage réel.
- **Couverture des PDF, mesurée.** `source_pages` est le réciproque du registre de couverture. Un
  tableau de bord par document — pages couvertes, par quelle page du wiki, ce qui reste — remplace
  une affirmation (§ 4.4) par une mesure.
- **Mode SAV.** Le wiki conserve volontairement les valeurs historiques : l'HYBRIDE en 72 mm
  jusqu'en 2025, l'évolution des garanties depuis 2023, le laquage à 25, 10 ou 7 ans selon la
  gamme et le coloris, la garantie bord de mer à 5 ou 7 ans selon la distance au littoral. Un
  parcours qui demande l'année de fabrication et lit la bonne ligne répond sur la pièce de
  rechange et sur la garantie applicable, avec CTR‑03 et CTR‑09 en réserve. Cet actif existe déjà
  et personne ne s'en sert.
- **Mode atelier.** Un contexte de session (« je suis sur la PERFORM76 ») pour que les questions
  suivantes n'aient plus à nommer la gamme, et une fiche imprimable d'une page par procédure, avec
  ses coupes. Le vocal mains libres est déjà construit ; ce qui lui manque est ce contexte.

### 9.6 Le prérequis commun : promouvoir les tableaux en données

Les quatre premières fonctionnalités supposent la même chose, et c'est la seule décision
d'architecture de cette section : **les tableaux doivent devenir des lignes typées.** Deux voies.

| Voie | Coût | Risque |
| --- | --- | --- |
| Extraction au chargement, comme l'index de recherche | moyen, une fois | une table non conforme n'est pas extraite |
| Bloc `data:` en frontmatter, saisi à la main | élevé, à chaque page | la donnée et le tableau divergent |

Je recommande l'extraction. Elle ne demande rien aux rédacteurs, elle s'appuie sur des règles que
le protocole impose déjà (première colonne = la référence, unité en en‑tête, une ligne par
référence, un contexte par ligne), et **le contrôle d'extraction devient le contrôle de conformité
des tableaux** : une table qui ne s'extrait pas est une table à corriger. C'est le même mouvement
que le lint étendu du § 4.6, avec une sortie utilisable en plus.

### 9.7 Ce que je ne construirais pas

- **Un moteur sémantique sur les PDF.** C'est ce que fait la solution externe, à 2 réponses
  exploitables sur 31.
- **Un chiffrage ou un devis**, tant que `tarifs/` est vide. Le wiki ne porte aucun prix.
- **Un configurateur commercial complet.** Le vérificateur du § 9.1 dit ce qui est réalisable ;
  vendre, c'est l'ERP.

### 9.8 Ordre proposé

| Fonctionnalité | Donnée présente | Effort | Ce que ça évite |
| --- | --- | --- | --- |
| Fiche d'identité d'une référence | oui, déjà indexée | faible | la mauvaise famille commandée |
| Veille des validités | oui, dans les corps de page | faible | poser sous un avis expiré |
| Vérificateur de faisabilité | oui, cinq tableaux à croiser | moyen | une menuiserie refaite |
| Anomalies en flux de travail | oui, 77 entrées | moyen | une contradiction qui traîne deux ans |
| Prescription par chantier | oui, trois tables | moyen | une classe AEV donnée au jugé |
| Calcul de débit | oui, formules et déductions | élevé | une barre coupée trop court |
| Mode SAV | oui, valeurs historiques | élevé | une garantie mal engagée |

Si une seule est retenue, c'est le **vérificateur de faisabilité** : la question qui coûte quand
elle est fausse, celle où le modèle seul est le plus faible, et la donnée est là. La fiche de
référence se fait en parallèle pour presque rien.

## 10. Ce que cet audit n'a pas pu faire

- Confronter une seule cellule du wiki à son PDF : `raw/` est vide ici. Les registres de
  couverture et les 8 `verified` sont pris sur parole ; la densité du § 4.4 est un indice, pas
  une preuve.
- Rejouer le golden de génération sur l'API réelle (clé et coût) : les 59 / 62 du § 3 mesurent
  la page livrée, pas la réponse du modèle. Le lien entre les deux est fort (les 8 échecs du
  golden du 18/09 venaient de la page ou de la lecture, pas du raisonnement) mais il reste à
  confirmer d'un seul passage réel après la correction de l'index.
