# Plan L2 — découper les pages trop longues, sans orpheline (30/09/2026)

**PROPOSITION, à valider.** Suite de `docs/plan_wiki_trouvable_2026-09-30.md` (L1 écrit :
*Writing to be found* dans `wiki_llm/CLAUDE.md`). Mesures : `docs/benchmark_navigation_2026-09-30.md`.
Aucune page n'est modifiée par ce document ; le proposeur de découpe est
`app/scripts/plan_decoupe.py` (lecture seule, sortie `logs/l2/plan_decoupe.json`).

## 1. Jugement d'architecture — le wiki est-il fait pour un LLM ?

**Verdict : la charpente est bonne, le grain est faux.** On ne réorganise pas, on redécoupe.

### Ce qui est bon, et qu'on garde

| Atout | Pourquoi il compte pour un LLM |
| --- | --- |
| Axes de frontmatter (`gamme`, `systeme`, `fournisseur`, `usage`, `famille`) | La navigation humaine est calculée, aucune page hub : rien ne concurrence les vraies pages dans la recherche. |
| Une donnée = une page ; registres d'anomalies ; cartes `sources/` avec registre de couverture | Un fait a une adresse ; l'absence est démontrable (« zéro à faire » — sous réserve de L7). |
| Chemin = identifiant permanent, préfixé par système (`roto-nx-…`, `systeme-70-…`, `perform76-…`) | Le tableau « Autres résultats » montre le chemin : il dit déjà de quoi on parle. |
| **0 page assertive orpheline** sur 232 | Toutes sont citées par une autre page. Les 22 pages sans lien entrant sont des fiches `sources/`. |
| Pas de doublon à traiter en chantier séparé | Les pages « aperçu de ferrure » Roto NX ne recopient pas la page des crémones : elles portent des cotes de positionnement (HFF, HP, G1…) et aucun numéro d'article. Ce n'est pas une affaire de redite, c'est de la taille. |

### Ce qui coince (mesuré le 30/09)

| # | Défaut | Mesure |
| --- | --- | --- |
| 1 | **Une page n'est pas une famille de questions.** `systeme-70-profiles-et-renforts` (100 k car.) répond à cinq questions différentes : dormants, ouvrants, battements, meneaux, renforts. | 30 pages = 40 % du volume ; 80 pages assertives > 20 000 car. ; 22 > 50 000 |
| 2 | **Le grain ne colle pas à l'outil.** `chercher` livre trois pages entières : le coût d'une recherche est donc celui de trois pages, pas d'une ligne de tableau. | Médiane 96 000 car. de pages livrées par question du banc, p90 265 000 |
| 3 | **Deux éditions, deux pages, aucun titre pour les distinguer** : `traitement-du-battement-systeme-70` et `…-evo2008`, `accouplement`, `assemblage-meneau-traverse`… | Les deux doivent remonter (contradictions) ; le titre doit dire l'édition |
| 4 | **3 pages sans type**, donc hors OKF et hors tableaux de bord | `systeme-70-abaques-evo2008` (82 k), `choix-des-fenetres-exposition-au-vent-2008` (50 k), `sources/roto-nx-bras-report-de-charge` |
| 5 | **Fiches `sources/` non atteignables** depuis les pages qu'elles ont produites | 22 fiches sans lien entrant, 17 orphelines pour le lint |
| 6 | **Découper coûte des liens** | 172 liens à ancre (`page.md#…`) ; 118 pages renvoient au glossaire (280 liens, dont 2 à ancre), 50 pages à `profiles-et-renforts`, 40 à `fournisseurs/roto` ; 18 à 34 lignes de registres de couverture par grosse page |
| 7 | **Des références sur plusieurs pages** | 87 numéros à 6 chiffres sur ≥ 3 pages (29 sur ≥ 5) : à contrôler pendant la découpe, pas à part |

### Ce qu'on ne change pas — et pourquoi

- **Les dossiers par type.** La recherche n'indexe pas le chemin, le tableau de bord vient du
  frontmatter : des dossiers thématiques ne feraient gagner aucun rang, et coûteraient chaque lien,
  chaque registre, chaque test et les chemins codés en dur des trois outils. **Aucun fichier n'est
  déplacé** ; les nouvelles pages se nomment `<système>-<thème>[-<variante>]`.
- **`index.md` ligne par page** (voir décision 3).

### Ce que L2 ne fera pas seul

Le banc (57 questions), rejoué comme si les pages de chaque vague étaient coupées en parties de
18 000 caractères au plus — estimation **optimiste** : elle suppose que la recherche ramène une
seule partie par page —, donne :

| Découpe | Pages en plus | Médiane lue / question | p90 | Total banc |
| --- | --- | --- | --- | --- |
| aucune (aujourd'hui) | 0 | 95 700 | 265 000 | 100 % |
| vague A (22 pages > 50 k) | +106 | 69 600 | 129 000 | **61 %** |
| **vagues A + B** (54 pages > 30 k) | +162 | 60 200 | 106 000 | **51 %** |
| A + B + C (80 pages > 20 k) | +188 | 54 700 | 103 000 | 49 % |

Même en découpant tout, on **divise par deux**, pas par dix : trois pages de 18 000 caractères
arrivent toujours entières. La seconde moitié du gain est dans la livraison (phase 2 : une page
entière et, pour les autres, la section ou les lignes qui ont fait remonter la page). Or ce
levier a une condition que L2 crée : **des sections nettes, titrées, autonomes**. Un wiki bien
découpé rend la phase 2 possible ; une phase 2 sur des pages de 140 000 caractères n'a rien à
extraire proprement.

## 2. Principe

**Une page = une famille de questions du métier, entre 5 000 et 18 000 caractères.** Un thème est
ce que le menuisier demande d'un trait : « quelle crémone pour cette hauteur de poignée »,
« quel compas pour cette largeur de feuillure », « quel embout pour ce seuil ». On coupe le long de
l'axe de la source (famille de pièces, usage, système, tableau), jamais à la ligne. Une partie se
lit seule, porte les axes du frontmatter de la page mère, un titre et une description qui
nomment son contenu, et ne s'appuie sur ses voisines que par un lien dans la phrase concernée
(protocole, règle 1 de *Writing to be found*). **Aucune page de liaison.**

## 3. Vagues

`plan_decoupe` coupe chaque page à ses propres titres (`#`, puis `##` si une section dépasse
seule). Ce n'est qu'un point de départ : le regroupement par thème se fait à la main, page par
page.

| Vague | Pages | Parties (budget 20 000, cible 18 000) | Familles | Protégées par un outil | Questions du banc concernées |
| --- | --- | --- | --- | --- | --- |
| **A** : > 50 000 car. | 22 | 128 (+106) | Roto 6 · Système 70 : 10 · Système 76 : 4 · autres 2 | 4 | 28 / 57 |
| **B** : 30 000 à 50 000 | 32 | 88 (+56) | Système 70 : 12 · Système 76 : 9 · Roto 7 · autres 4 | 2 | 27 / 57 |
| **C** : 20 000 à 30 000 | 26 | 52 (+26) | Système 76 : 7 · Système 70 : 6 · Roto 6 · portes 5 · autres 2 | 2 | 19 / 57 |

Les gains décroissent vite : la vague A seule descend à 61 % du volume lu, A + B à 51 %, et la
vague C n'ajoute que 2 points pour 26 pages de plus. **La vague C ne se lance que si la mesure
après B montre ses pages dans le bruit.** Critère d'arrêt de chaque vague : médiane lue par
question du banc, et taux de bonnes réponses inchangé.

**Ordre dans une vague : par famille, pas par taille** — les pages d'une même famille se
découpent ensemble pour que leurs parties se nomment de façon cohérente et se renvoient les unes
aux autres :

1. Roto NX (catalogue, 6 pages en A) — cœur du problème constaté
2. Système 70 (10 pages en A) — plans, profilés, accessoires, éditions 2008
3. Système 76 / PERFORM76 (4 en A, dont 2 protégées)
4. Glossaire (cas à part, voir § 6)

### Vague A en détail

| Page | Taille | Parties | Entrants | Note |
| --- | --- | --- | --- | --- |
| `quincaillerie/roto-nx-cremones` | 141 k | 9 | 21 | **pilote** |
| `profiles/systeme-70-plans-de-combinaison` | 129 k | 8 | 11 | 282 images ; sections 43 k et 75 k à recouper |
| `quincaillerie/roto-nx-apercu-ferrures-cote-p` | 124 k | 9 | 13 | 1 185 lignes de tableau |
| `profiles/systeme-70-profiles-complementaires` | 113 k | 7 | 21 | **protégée** (`parcloses.py`) |
| `quincaillerie/roto-nx-apercu-ferrures-designo` | 109 k | 9 | 10 | tableaux HTML : ne jamais couper dans un `<table>` |
| `quincaillerie/roto-nx-accessoires-et-gabarits` | 104 k | 8 | 11 | 10 anomalies liées |
| `profiles/systeme-70-profiles-et-renforts` | 100 k | 7 | **50** | le plus lié : réécriture de liens en masse |
| `quincaillerie/roto-nx-compas-et-paliers` | 91 k | 6 | 18 | |
| `procedures/directives-generales-systeme-70-evo2008` | 84 k | 6 | 5 | édition 2008 : le titre le dit |
| `profiles/systeme-70-abaques-evo2008` | 82 k | 5 | 7 | **sans type** : à typer en même temps |
| `profiles/systeme-70-accessoires-par-profile` | 78 k | 5 | 6 | |
| `procedures/mise-en-oeuvre-systeme-70-evo2008` | 75 k | 6 | 6 | 9 anomalies liées |
| `reference/glossaire` | 71 k | 5 | **118** | cas à part |
| `procedures/mise-en-oeuvre-seuil-systeme-76` | 70 k | 6 | 10 | variantes 1, 2, 3 = un axe net |
| `quincaillerie/roto-nx-champs-application` | 61 k | 4 | 27 | **protégée** (`faisabilite.py`) |
| `profiles/systeme-76-profiles-complementaires` | 59 k | 4 | 20 | **protégée** |
| `procedures/accouplement-elements-systeme-76` | 59 k | 4 | 5 | |
| `profiles/systeme-70-statique-et-inerties` | 57 k | 4 | 28 | |
| `certifications/dta-6-16-2334` | 54 k | 4 | 22 | **protégée**, 47 k de « Prescriptions » : une prescription par clause numérotée |
| `procedures/capotage-aluclip-systeme-76` | 53 k | 4 | 2 | 49 k d'« Étapes » : à découper par étape ou variante |
| `procedures/mise-en-oeuvre-renovation-systeme-70-evo2008` | 50 k | 4 | 4 | trois poses = trois axes |
| `profiles/systeme-70-tableau-de-vitrage` | 50 k | 4 | 3 | |

La liste des parties proposées, page par page : `logs/l2/plan_decoupe.json`.

## 4. Le geste, pour une page

Chaque page suit les mêmes étapes, dans l'ordre. Rien n'est mécanique au sens de « sans relire » :
le script coupe et compte, l'écriture des parties est à la main.

1. **Inventaire avant.** Références, lignes de tableau, images, identifiants d'anomalie, liens
   sortants, liens entrants (avec ancres), lignes de registres de couverture qui pointent vers la
   page. Un fichier, avant de toucher à quoi que ce soit.
2. **Plan de découpe.** Parties, regroupées par thème (pas seulement dans l'ordre de la source),
   nom de fichier `<système>-<thème>`, titre, description, tags de la liste fermée. La partie qui
   garde le nom de la famille garde l'ancien chemin.
3. **Coupe.** Le corps est déplacé tel quel, aux frontières de titres du plan : jamais au milieu
   d'un tableau ni d'un `<table>`. Aucune reformulation à ce stade.
4. **Frontmatter de chaque partie.** Axes hérités de la page mère ; `sources` et `source_pages`
   réduits aux pages PDF que la partie porte réellement ; `title`, `description`, `tags` écrits ;
   `verified` **retiré** (la relecture inverse se refait sur la partie) ; `status` hérité.
5. **Ouverture et queue.** Phrase de définition de la partie (elle nomme le produit PROFERM et le
   système : règle 5), `# Ce que la source ne donne pas` et `# Citations` limités à la partie,
   `# Voir aussi` vers les parties sœurs.
6. **Liens entrants.** Chaque lien vers l'ancienne page (y compris `#ancre`) est réaffecté à la
   partie qui porte la valeur dont parle la phrase du lien. Le script propose (la partie qui
   contient les références de la phrase) et sort la liste des cas ambigus, à trancher à la main.
7. **Registres de couverture** des fiches `sources/` : la colonne « Page du wiki » de chaque plage
   de pages PDF pointe vers la partie qui la porte ; une plage à cheval sur deux parties est
   scindée en deux lignes.
8. **Anomalies** : la colonne `Pages du wiki` de chaque entrée liée pointe vers la ou les parties
   qui portent la valeur.
9. **Index et journal.** `index.md` : une ligne par partie, description **identique** au
   frontmatter ; `log.md` : une entrée ; pages gamme, système, fournisseur : les liens vers la page
   qui a été coupée sont réécrits vers les parties (étape 6, même mécanisme).
10. **Vérification (script), puis tests, puis banc.** Voir § 5. Une page n'est « faite » que si tout
    passe ; sinon elle est rouverte, pas déclarée finie.

## 5. Zéro orpheline, zéro perte — les invariants vérifiés après chaque page

Le script `verifier_decoupe` (à écrire, lecture seule) échoue si l'un d'eux tombe.

| # | Invariant | Comment |
| --- | --- | --- |
| I1 | **Rien ne se perd.** Chaque référence, chaque ligne de tableau, chaque image, chaque identifiant d'anomalie de l'inventaire avant se retrouve dans exactement une partie | comparaison d'inventaires ; les lignes retirées volontairement sont listées à part |
| I2 | **Chaque partie est atteignable par trois chemins** : une ligne d'`index.md`, ses axes de frontmatter (tableaux de bord), et au moins un lien entrant depuis une page concept | graphe de liens recalculé |
| I3 | **Aucun lien cassé, aucune ancre morte** | lint de liens du snapshot ; les 172 ancres |
| I4 | **Chaque partie tient dans le budget** (20 000) ou est signalée comme indivisible | mesure du corps |
| I5 | **Les descriptions d'`index.md` sont celles du frontmatter**, mot pour mot | lint existant |
| I6 | **Chaque ligne de registre de couverture pointe vers une partie qui porte cette plage de pages** | lecture de `source_pages` des parties |
| I7 | **Chaque entrée d'anomalie liée à l'ancienne page a un lien vers la bonne partie** | comparaison avant / après |
| I8 | **Les tests passent** ; pour une page protégée, l'outil qui la lit est modifié dans le même changement | `docker compose exec web pytest` |
| I9 | **Aucune page assertive sans `type`, sans `title`, ni sans `gamme` ou `systeme`** | lint de frontmatter |

Après chaque vague : `docs/benchmark_navigation` rejoué sur les questions touchées (trouvabilité
mécanique, puis Claude imitant LIA), **même taux de bonnes réponses**, volume lu en baisse.

## 6. Cas particuliers

- **8 pages protégées** (lues par `parcloses.py`, `faisabilite.py`, `debit_atelier.py`) : 4 en A
  (`systeme-70-profiles-complementaires`, `roto-nx-champs-application`,
  `systeme-76-profiles-complementaires`, `dta-6-16-2334`), 2 en B, 2 en C. Elles se coupent **avec
  l'outil**, en gardant intacts les titres de section et les en-têtes de colonnes que l'outil
  cherche ; l'outil reçoit la liste des pages qui portent désormais ses tableaux. Elles passent en
  fin de leur famille, une fois la méthode rodée. Un test qui casse est rapporté, on ne tord pas la
  page.
- **Glossaire** (71 k, 118 pages y renvoient par 280 liens, dont 2 seulement avec une ancre) :
  5 parties proposées, dont une section de 26 k (« Pièces et gestes de la menuiserie ») sans
  sous-titre. Ces liens disent « voir le glossaire » sans viser un terme : la partie qui garde le
  chemin `glossaire.md` reçoit ceux-là ; les autres se nomment `glossaire-<thème>`, et les deux
  ancres sont réaffectées à la main. À valider par toi avant d'y toucher.
- **Pages sans type** : typées et dotées de leurs axes pendant la découpe (I9).
- **Éditions** : quand deux pages traitent du même sujet à deux éditions, le titre des deux dit
  l'édition (« … (manuel 2023) », « … (classeur e.VOLUTION 2008) »). Les deux pages restent.
- **Tableaux HTML** (matrices à deux niveaux d'en-têtes) : jamais coupés ; un `<table>` qui
  dépasse le budget reste entier.
- **Images** : les chemins `assets/…` ne bougent pas.

## 7. Le pilote : `quincaillerie/roto-nx-cremones`

141 434 car., 685 lignes de tableau, 69 images, 21 pages qui la citent, 18 lignes de registres,
4 anomalies liées, non protégée, structure nette (une section `#` par famille de crémone). Proposition
de départ : 9 parties de 12 à 18 k. Regroupement thématique visé : OB KSR poignée fixe · OB poignée
centrée / variable · solutions spéciales (adaptée, Confort) · verrou · raccords · semi-fixe
standard · semi-fixe Plus · levier séparé et verrou d'arête · crémones à sortie de tringle / H100.

**Les deux pages « aperçu de ferrure » (côté P : 124 k, Designo : 109 k) se découpent en
miroir de celle-ci**, avec les mêmes familles et des noms alignés (`roto-nx-cremones-<famille>`,
`roto-nx-apercu-cote-p-<famille>`, `roto-nx-apercu-designo-<famille>`) et un lien de l'une à l'autre :
la question « quelle crémone, et où poser sa gâche pour telle hauteur » traverse les deux.
Ces pages ne contiennent aucun numéro d'article, seulement des cotes de positionnement : elles se
retrouvent par le titre de leur section, qui devra donc nommer la famille.

Critère de réussite : I1 à I9 verts ; questions du banc qui l'ont lue rejouées — **R23** et
**S05** — même réponse ; volume lu en baisse sur ces deux ; et **tu relis les 9 titres et
descriptions** avant qu'on passe aux autres pages.

## 8. Outils à écrire (scripts de maintenance, pas de code applicatif)

| Script | Rôle |
| --- | --- |
| `plan_decoupe` (fait) | propose les parties, lecture seule |
| `decouper_page` | inventaire avant ; coupe aux frontières du plan ; écrit les parties (frontmatter hérité) ; rapport des liens entrants à réaffecter |
| `verifier_decoupe` | les invariants I1 à I9, lecture seule |

Ils manipulent du markdown déjà écrit et jamais un PDF : la règle « jamais de couche texte »
du protocole ne les concerne pas.

## 9. Décisions à prendre

1. **Périmètre : vagues A + B (54 pages, +162 pages, volume lu à ~51 %), C seulement si la
   mesure la justifie.** Ou A seule (+106 pages, 61 %) pour mesurer avant d'aller plus loin.
2. **Validation :** tu relis le pilote (9 titres / descriptions), les 4 pages protégées de la
   vague A et le glossaire ; le reste est tenu par les invariants et le banc. D'accord ?
3. **`index.md`** compte aujourd'hui 98 000 caractères (une ligne par page) et passera à ~170 000
   après A + B. Il sert à l'accueil de l'application et au lint, et de point d'entrée à qui
   maintient le wiki. On le garde ligne par page, en notant qu'il faudra un jour le
   découper si l'on veut le relire d'un trait — c'est ma recommandation ; l'autre voie serait de
   cesser d'y lister chaque partie, ce qui casse le lint et l'accueil.
4. **Outils protégés :** accord pour modifier `parcloses.py`, `faisabilite.py`, `debit_atelier.py`
   dans les mêmes changements que les pages qu'ils lisent ?
5. **Commit par famille** (Roto, Système 70, Système 76), pour pouvoir revenir en arrière sans
   défaire le reste.
