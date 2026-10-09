# Retraitement des 15 documents SOPROFEN — compte rendu avant / après (09/10/2026)

Périmètre : les 15 PDF SOPROFEN (275 pages) de `raw/moustiquaires/`, retraités selon le protocole
de `wiki_llm/CLAUDE.md` : rendu PNG page par page, lecture en image seulement (jamais la couche
texte), un PDF à la fois, découpe de chaque schéma à côté de ce qu'il illustre. Technal et Askey
ne sont pas touchés (demandé : plus tard). Rien n'est commité.

## 1. En chiffres

| | Avant | Après |
| --- | --- | --- |
| Pages de PDF qualifiées `transcrit` ou `sans contenu propre` | 0 sur 275 (les 15 registres étaient remis à `à faire` le 28/09) | 275 sur 275 : 268 `transcrit`, 7 `sans contenu propre` (pages « Notes » et blancs), 0 `à faire` |
| Pages de contenu (hors `sources/`, `anomalies/`) tirées de ces 15 documents | 16 pages, ≈ 108 000 caractères | 67 pages, ≈ 608 000 caractères (51 créées, 16 réécrites) ; trois pages de liaison complétées en plus (`volets-roulants`, `fournisseurs/soprofen`, `glossaire`) |
| Plus grosse page de contenu créée | — | moins de 20 000 caractères de corps (seul le glossaire, 74 000 avant mon passage, reste au-dessus) |
| Images découpées | 0 (les pages renvoyaient au PDF) | 763 dans `assets/` |
| Anomalies | — | **51 INC** (INC-307 à 357), **6 CTR** (CTR-122 à 127), **24 VER** (VER-127 à 150), toutes ouvertes, toutes liées aux pages qui les portent |
| Tags | 658 (30/09) | 704 dans `tags.md` ; chaque page entre 4 et 10 tags de la liste fermée |

Les registres de couverture des 15 cartes `sources/` couvrent 1 à N sans trou ni recouvrement
(contrôlé par le vérificateur à chaque carte). Les pages réécrites et créées passent le vérificateur
(frontmatter, titres ≤ 90 caractères, descriptions = lignes de `index.md`, tags de la liste,
phrases interdites sur pages assertives, citations `[1 p. N]`) : 0 problème sur les 70 pages
de contenu et les 15 cartes, hors le glossaire (défaut antérieur : 74 972 caractères au dernier
commit, citations [7], [28], [50]). Le vérificateur relève aussi 15 liens à ancre (`#section`) dans
les registres d'anomalies `contradictions-entre-sources` et `incoherences-internes` : ils pointent
vers des sections des systèmes 70 et 76 déplacées par la découpe du 30/09 et existaient déjà au
dernier commit ; je n'y ai pas touché.

**Tests de l'application** (`pytest` dans le conteneur `web`, clé Mistral factice, flux simulés, aucun
appel de modèle) : **208 passés, 0 échec**, en 22 min 26 s — y compris `test_faisabilite`,
`test_debit_atelier`, `test_parcloses` (pages lues par l'application) et
`test_recherche_par_reference` (« 76507 »), qui échouait après le retraitement du 28/09.

## 2. Document par document

### 2.1 Fiche moustiquaire enroulable verticale (2 p.)
- **Avant** (page de 6,2 k caractères) : inventé ou attribué à tort à la fiche — « pose en tableau
  ou applique extérieure avec fixations invisibles », « frein régulateur de remontée », « ressort
  avec barre de charge et crochets », « tringle oscillante », 230 V (qui vient du catalogue),
  « tous coloris RAL au même prix » (idem). Absent : garantie 2 ans, blocage manuel, destination
  des coffres, largeur 3,50 m, pictogrammes de manœuvre, images, coupes de mise en œuvre.
- **Après** (9,9 k) : cotes et surfaces limites par modèle, garantie, crochets de verrouillage,
  voltages 230 / 230 / 12 V, toile noire en option ; 5 modèles et 2 coupes de mise en œuvre en
  images ; page `coloris-moustiquaires` créée (6 pastilles RAL, toiles, accessoires).
- Anomalies : INC-309, INC-310, CTR-125, CTR-126.

### 2.2 Fiche moustiquaire plissée (2 p.)
- **Avant** (5,8 k) : **cotes de pose fausses** — 2VO « 47 / 84 / 100 » (réel : 47-100-47 en
  tableau, 75-100-75 en applique), 2VJ applique « 75 / 84 / 98 » (réel 75-98-75) ; « seuil
  rétractable », « profilés articulés », « haute ténacité », « cornières de 75 mm » non sourcés ;
  « tous les coloris RAL au même prix » attribué à la fiche. Absent : le 5ᵉ modèle MOPF 29 1R, la vue
  « toile fermée » (47, 75, 84, 98, 100 mm, profondeur 27), la garantie, 4 dessins de pose.
- **Après** (11,0 k) : 7 lignes de cotes de pose (entre tableaux / applique) relues sur dessins à
  450 dpi ; 5 modèles ; 4 dessins ; accessoires noirs.
- Anomalies : CTR-122, CTR-124.

### 2.3 Fiche moustiquaires battante, coulissante, cadre fixe (2 p.)
- **Avant** (6,4 k) : « dormant de 45 mm de face », « montant ouvrant de 40 mm », « battement
  central de 45 mm » (lecture erronée des cotes de pose) ; coulissante « entre tableaux ou en
  applique » (la fiche : « entre tableaux uniquement ») ; « renfort obligatoire » (la fiche : « en
  option ») ; « MOFI CLIP par clips » non sourcé.
- **Après** (11,7 k) : cotes de pose relevées (45/45/40 ; 30/65/60/16/N/L ; 12 mm, H, L), 5 rendus,
  3 dessins, garantie, accessoires, barrière « la plupart des animaux ».
- Anomalies : INC-307, INC-308, CTR-123.

### 2.4 Catalogue moustiquaires (13 planches)
- **Avant** : carte de 8,9 k caractères, registre « à faire », données reprises dans les trois
  fiches ; pas de guide de choix, pas de page de présentation du fabricant.
- **Après** : `moustiquaires-guide-de-choix` (usages × familles), `commercial/moustiquaires-soprofen`
  créées ; MOBA / MOCO / MOFI en images ; couleurs (toile noire, RAL au même prix sauf MOHO 48) ;
  655 collaborateurs, huit sites ; erreur « MOFI CLIP fixation par clips » retirée.
- Anomalies : INC-311, VER-127 ; CTR-122 à 126 étendues.

### 2.5 Fiche screen extérieur zip SCREENAMAX (2 p.)
- **Avant** (6,9 k) : non sourcé — « filtre jusqu'à 97 % avant le vitrage », « moustiquaire
  hermétique intégrale », « compatible TaHoma », « protocole radio propre », descriptions de toiles
  (Mermet, Copaco, Sattler, Serge Ferrari), « système breveté ». Garantie bord de mer : zones sans
  durées. 13 pastilles sur 15.
- **Après** (9,1 k) : garantie 5 ans (0-10 km) / 7 ans (> 10 km) ; panneau solaire 7 mm, 45 jours à
  un cycle par jour, −20 / +70 °C ; 15 pastilles dans `coloris-screenamax` ; 13 images ; faute
  « Sergi Ferrari » consignée.
- Anomalies : INC-312, VER-128.

### 2.6 à 2.8 Fiches volet TRADI : non prémonté, prémonté, coffre tunnel et VTR (6 p.)
- **Avant** (3 pages de 6,3 à 7,2 k, quasi identiques entre elles) : « 24 teintes LA 37/50 »
  (25 sur la fiche), « 14 teintes LA 546 » (17), « lame 546 = 54,6 mm renforcé » (coupe cotée 55),
  « certifiée », « laquage réflecteur d'infrarouges », « projection à l'italienne » (la fiche :
  « Projection »), « NF EN 13659 » (la fiche n'imprime que NF EN 13241), motorisations avec
  attributs inventés, « Gris sablé 2900 » classé en LA 37/50, extrapolation du sigle VTR. Absent :
  coupes cotées des lames, IO Hybrid, garantie bord de mer 5 / 7 ans, score de réparabilité,
  84 pastilles, lames bicolores LATH, encadrements niveaux 1 et 2.
- **Après** : 3 pages produit (4,7 + 4,9 + 5,6 k) + 2 pages communes (technologies, manœuvres et
  garanties) + 1 page de coloris (84 pastilles) — chaque bloc commun sur UNE page ; ≈ 30
  affirmations non sourcées retirées ou corrigées ; 20 images.
- Anomalies : INC-313 (4 500 mm contre 3,5 m), INC-314, VER-129 à 133.

### 2.9 Notice de perçage des coulisses pour lame à serrure (2 p.)
- **Avant** (4,4 k) : outillage inventé (mètre, réglet, crayon de menuisier, foret ∅ 10, lime
  plate), « remonter le tablier d'environ 500 mm », étape 5 « contrôle fonctionnel » entière
  inventée, regroupement en 5 étapes. Exact : cotes −10 / +15, ∅ 10, hotline.
- **Après** (4,4 k, même volume, contenu corrigé) : les 7 étapes de la notice avec leur
  numérotation, 4 images (2 dessins cotés, 2 pictogrammes de cylindre), horaires exacts, centre de
  formation ; coordonnées bancaires volontairement non reprises. Aucune anomalie.

### 2.10 Notice de pose Chrono One en atelier (4 p.)
- **Avant** (6,1 k) : référence « NO-BB-T-CD-00 » (la notice : **CO**) ; « interdiction stricte de
  stockage en position verticale sur la tranche » (le pictogramme, sans texte, montre des blocs
  dressés sous le soleil et la pluie) ; **cotes 50 / 400 attribuées à l'adaptateur** alors qu'elles
  sont portées sur les vis clip des montants ; « doigt d'indexage dans la rainure supérieure » (la
  notice : sous-face) ; « vis fournies » non dit.
- **Après** (11,2 k) : 15 schémas découpés ; cote 21 mm et 3 coupes ; consignes de sécurité aux
  installateurs ; texte « Choix des vis / chevilles » ; avis de non-contractualité.
- Anomalies : VER-134 (« (Moteur) »), VER-135 (pictogramme de stockage).

### 2.11 Notice de pose Chrono One sur dormant monobloc (3 p.)
- **Avant** (6,0 k) : « isolation diélectrique obligatoire », « 3 positions d'inclinaison »,
  « ergots de guidage inférieurs », tableau « condition d'ancrage mécanique », même fausse
  interdiction de stockage.
- **Après** (9,9 k) : 13 images, légendes du dessin, cartouche Moteur, exemples L1 commande / L2 ;
  ordre des opérations conservé.

### 2.12 Guide de pose Coffre Demi-Linteau Bloc LX (3 p.)
- **Avant** (5,9 k) : « commande par treuil ou motorisation (filaire, radio, solaire) », « entraxe
  **maximal** de 600 mm » (la notice : « tout les 600 mm **mini** », INC-315), DL-R10 « 49,7 × 32 ×
  2,5 perforations oblongues » et DL-R12 « renforcé » (la notice ne nomme pas les deux profils
  dessinés), « RE2020 / perméabilité à l'air », motif inventé de la vis de 9,5 mm.
- **Après** (10,1 k) : encadré rouge de l'étape 4 (verticalité dans les deux sens, même longueur,
  largeur L dos à dos constante), NFC 15-100, cotes 12,6 / 5 / 10, astuce des tapées creuses ;
  15 images.
- Anomalies : INC-315, VER-136, VER-137.

### 2.13 Guide technique Bloc LX (16 p.)
- **Avant** : une page de 12,5 k + une carte de 19,4 k au registre « toutes pages à faire ».
  Erreurs : lame finale LF30 « Blanc, Beige 1015, Gris 7016, Brun » (réel : Blanc, Gris métal, Brun,
  RAL) ; « LATH 50 : 2,8 kg/m² » (case vide) ; « L 37 épaisseur 8 » (pas de dessin) ; « mousse
  polyuréthane » (non sourcé) ; « tiroir équipé en standard du bouclier DL52 » (le guide : « sans
  isolant DL52 », donc option) ; « Hybrid 4 × 0,75 » (3 × 0,75 individuelle, 4 × 0,75 inverseur à
  clé) ; « tige oscillante code 9 » (confusion code longueur / code coloris).
- **Après** : 5 pages (14,9 + 17,5 + 8,5 + 7,0 + 5,7 k) + coloris + carte à 16 lignes ; 47 images.
  Ajouts : bon de commande en 17 rubriques, sigles de la grille, étude préalable, Smoove (11 lignes
  × 8 cadres), émetteurs par défaut et codes (68, SM, QM…), déport BTD, tiges oscillantes, supports
  de cardan, conditions de la projection, pattes VX26, trappe VX03/18, renforts (inerties),
  mortaise VMC, câbles de réglage, plans d'intégration, codes de coloris, nuancier de 21 teintes avec
  RAL approchantes.
- Anomalies : INC-316 à 321, VER-137 à 140.

### 2.14 Avis Technique CSTB 6/16-2339_V2 des coffres Chrono (72 p.)
- **Avant** : une page de 9,5 k (p. 1-7) + carte au registre `toutes → à faire`. Erreurs :
  « RT globale » (la source : « RT existant ») ; Uc 230 tunnel donné seulement avec isolant de joue
  (12 lignes sur 30 reprises) ; ITE 140 : 1 ligne sur 4 ; « coffre en aucun cas porteur » non
  retrouvé. Les références des **planches p. 21-72** (profilés, coulisses, adaptateurs, embouts, joues,
  renforts, garnitures, isolants, flasques, coupes, détails de pose : MXB20-H, CRX31, JCX52, CX40/18,
  CV23/46, MX59/6…) n'étaient nulle part dans le wiki.
- **Après** : 8 pages `certifications/` (domaine d'emploi, 30 lignes de coefficients Uc / Up
  Chrono ONE, dossier technique, références des matériaux, fabrication et mise en œuvre,
  composition et coulisses — tableaux 3 et 4, coupes cotées CX22 à MX20, détails de pose p. 48-72)
  et 11 pages `profiles/chrono-*` (une par famille de pièces), soit 19 pages et ≈ 167 000 caractères ;
  **347 images** (une par référence, dans la ligne de son tableau) ; carte à 72 pages.
- Anomalies : INC-322 à 332, VER-141 à 145, **CTR-127** (largeurs de coulisse de l'AT contre le
  guide 132).

### 2.15 Guide technique Blocs-baies 2022 (148 p.)
- **Avant** : deux pages de produit (Chrono PSE² 6,9 k ; Mono VI 6,1 k) et la page du vent (4,6 k),
  citant un `.zip` que l'application ne sert pas ; carte `toutes → à faire`. Le Chrono ONE, les
  CX14 / CRX, les huit bons de commande, les conditions générales de vente, les services et
  contacts n'étaient nulle part. Dans les pages existantes : formules en LaTeX illisibles hors
  rendu, « performances thermiques **certifiées** » et « surpassant les exigences des bâtiments
  passifs et de la RE2020 » (non retrouvés dans le guide), « affaiblissement acoustique certifié »,
  « idéal en rénovation » ; page du vent : « Non admissible » pour des cases à tiret, cote CTA13
  « 54,7 × 29 » non portée par le guide ; page fournisseur : « Mertzwiller site historique ».
- **Après** : 24 pages, ≈ 246 500 caractères, 149 images. Chrono ONE en 5 pages (cotes et coloris ;
  coulisses et adaptateurs ; lame finale, mortaises VMC et isolation ; manœuvres et motorisations ;
  options, renforts, croquage) ; CX14 / CRX14 / CRX18 en 5 pages + une page de coupes (dont les
  valeurs thermiques Uc de 800 à 3 000 mm) ; Chrono PSE² en 5 pages (dix coupes d'intégration, BSO) ;
  Mono VI ; huit bons de commande ; conditions générales de vente (16 articles) ; services, atouts
  et contacts ; coloris ; résistance au vent avec sa carte de zones et son tableau de
  classification ; page `soprofen` complétée (huit sites, effectifs, certificats NF, société
  titulaire de l'AT).
- Anomalies : INC-333 à 357 (25), VER-146 à 150. Parmi elles : EM « coffre de 300 », MX10-35
  « réalisable / non », BSO 15 contre 13 teintes, déport R, bleu / gris 7023, Hybrid câble 2,5 / 3 m,
  inverseurs 1 / 2 inversés, sigles LATH / LAHD jamais définis.

## 3. Ce que ces 15 documents apportent maintenant et qui manquait

Les informations qui n'existaient pas du tout avant dans le wiki (et dont l'absence faisait
répondre « sur étude » ou à côté) :

1. **Tout le Chrono ONE, CX14 / CRX et le Bloc LX de 2022** : cotes de commande, tabliers, poids,
   coloris de coffre, coulisses, adaptateurs, mortaises VMC, côté de manœuvre, renforts d'inertie,
   brides, croquage, surlongueur de trappe, accessoires et conseils de mise en œuvre.
2. **Les motorisations et leurs codes** : largeurs L mini des moteurs, multi-tabliers, radio RTS /
   iO / M-SOFT 2, solaire, Hybrid, filaires, manœuvre de secours, émetteurs par défaut et leurs
   codes, Smoove.
3. **Les huit bons de commande** et les codes de chaque rubrique.
4. **Les valeurs thermiques Uc** du CX14 / CRX par largeur de coffre, et les 30 lignes Chrono ONE
   (230 / 200) de l'Avis Technique.
5. **Les coupes d'intégration** : dix pour le Chrono PSE², cinq pour le BSO, coupes cotées CX22 à
   MX20, détails de pose de l'AT (drainage, consoles, croquage, dormants larges, rénovation,
   applique, tunnel, reprise structurelle).
6. **Le catalogue de profilés Chrono** (AT p. 21-39) : coffre, coulisses PVC et aluminium,
   adaptateurs, embouts, joues, flasques, renforts avec leurs inerties, garnitures, isolants,
   pièces de guidage — 11 pages, 269 images.
7. **Garanties et conditions** : conditions générales de vente professionnels (16 articles),
   garantie bord de mer 5 / 7 ans, garanties des moustiquaires et du screen, score de réparabilité.
8. **Coloris** : 6 pages (moustiquaires, SCREENAMAX, TRADI — 84 pastilles —, Bloc LX — 21 teintes avec
   RAL approchantes —, blocs-baies 2022).
9. **Services, contacts, sites, chiffres** du fabricant, société titulaire de l'Avis Technique,
   certificats NF.
10. **Les deux notices Chrono One, la notice de perçage et le guide de pose Bloc LX** refaits au plus
    près des étapes et des cotes de chaque notice.

## 4. Ce qui était faux ou inventé : le bilan

Une quarantaine d'affirmations de l'ancienne rédaction étaient absentes des documents ou en
contradiction avec eux, dont : trois jeux de cotes de pose de moustiquaires, deux interdictions
inventées de stockage, des cotes attribuées au mauvais organe (adaptateur au lieu des vis clip), une
étape de procédure entière, des listes d'outillage, des mentions « certifié » / « RE2020 »,
des motorisations dotées de protocoles ou de compatibilités qui n'existent pas dans les fiches, des
nombres de teintes (24 pour 25, 14 pour 17), une cote de coulisse (29 pour 25,2), « RT globale » pour
« RT existant ». Chacune a été retirée ou corrigée sur la page, avec la valeur de la source ; le
détail avant → après est dans [`log.md`](../wiki_llm/wiki/log.md) (entrée du 2026-10-09).

## 5. Effet sur la recherche du chat (mesure hors ligne, sans modèle)

`app.scripts.mesurer_recuperation`, rejoué sur le wiki du dernier commit (323 pages) puis sur le
wiki actuel (372 pages) — aucun appel Mistral, clé factice :

| | Wiki au dernier commit | Wiki actuel |
| --- | --- | --- |
| Golden (40 q.) : section de la preuve désignée | 100 % | 100 % |
| Golden : toutes les pages attendues dans la carte (20 q. avec preuve) | 55 % | 60 % |
| Banc du 30/09 (56 q.) : section de la preuve désignée | 91,1 % | 85,7 % |
| Banc : toutes les pages attendues dans la carte | 69,6 % | 67,9 % |

Les trois questions du banc qui perdent leur « section désignée » (G10 « Coffre AIP36 : quelle
profondeur de maçonnerie ? », N01 porte coupe-feu EI30, N02 classe de grêle des lames LA 37) citent
comme preuve un extrait pris sur une page SOPROFEN **réécrite aujourd'hui** : l'extrait de
l'ancienne rédaction n'existe plus, la mesure ne peut plus le retrouver. Les pages restent dans la
carte (G10 : la page Chrono PSE² en tête et la page des codes de coffre juste derrière ; N02 : la
page du vent passe du 3ᵉ au 1ᵉʳ rang). Le banc du 30/09 est à régénérer sur les nouvelles pages.

Un déplacement réel : pour la question golden Q7 « Technal nous fournit nos volets roulants ? »
(pages attendues `equipements/volets-roulants.md` et `fournisseurs/soprofen.md`), la carte de douze
pages place maintenant en tête la page `commercial/soprofen-services-et-atouts-produits.md` et ne
liste plus ces deux pages. Rien n'a été changé pour le corriger : à voir avec vous si la page doit
porter dans sa description le mot « volets roulants » ou si le classement doit rester tel quel.

## 6. Limites et reste à arbitrer

- **Aucune page n'est `verified`** : il n'y a pas eu de relecture inverse. C'est une étape
  séparée, à faire dans une autre session.
- **Beaucoup de pages sont en `status: draft`** parce qu'elles portent une anomalie ouverte. Les
  57 anomalies (51 INC + 6 CTR) et 24 VER attendent l'arbitrage du service technique ou du
  fournisseur : aucune n'a été tranchée ni fermée.
- **Les PDF SOPROFEN sont dans `raw/moustiquaires/exports/NNN.pdf`** (132 à 138) : l'application ne
  sert que les PDF de `raw/` à la racine, donc la citation d'un de ces documents ne s'ouvre pas
  depuis le lecteur. À décider : déplacer à la racine, ou étendre le service.
- **Valeur `systeme: Chrono VX2`** sur les pages CX14 / CRX : elle n'est pas dans le vocabulaire
  documenté du protocole (Chrono One, Chrono PSE², Mono VI, Bloc LX, TRADI). À valider ou à
  remplacer.
- **Les bons de commande du guide 132** sont conservés en images pleine page (huit formulaires) en
  plus des tableaux de codes.
- **Le glossaire dépasse 20 000 caractères** (il les dépassait déjà) ; je l'ai étendu d'une section
  « Volets roulants et blocs-baies » sans le couper.
- **Le Bloc LX** est écrit depuis le guide technique (133) et comparé tableau par tableau au guide
  2022 (p. 72-85) : identique sauf le déport A à E du bon de commande et les valeurs R (INC-354).
- Les `generated.at` des pages touchées portent l'heure de ce passage (09/10/2026) et non les
  horodatages progressifs que j'avais posés au fil de la session.
- Technal et Askey restent à retraiter, comme convenu.
