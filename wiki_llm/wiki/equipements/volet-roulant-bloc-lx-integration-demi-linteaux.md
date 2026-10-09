---
type: Équipement
title: "Intégration du Bloc LX SOPROFEN : demi-linteaux Terreal, Genova, Prefatec, Imerys, Stradal"
description: Coupes et cotes d'intégration du Bloc LX SOPROFEN, treuil et moteur, isolation de 120 mm, dans les demi-linteaux Terreal, Genova, Prefatec, Imerys et Stradal (Sd, DC, Ti, Te, ØC, A, B), H demi-linteau 292,5 mm, 216 mm mini sans renfort ; Stradal moteur seulement.
resource: https://www.soprofen.com
tags: [volet-roulant, bloc-lx, demi-linteau, bloc-baie, isolation, soprofen]
systeme: Bloc LX
fournisseur: SOPROFEN
usage: chiffrage
famille: bloc-lx
status: draft
sources:
  - resource: raw/moustiquaires/exports/133.pdf
    id: soprofen-guide-technique-bloc-lx
    title: Guide technique Bloc LX, extrait du catalogue SOPROFEN (pages imprimées 110 à 125)
    last_modified: 2024-01-01
  - resource: raw/moustiquaires/exports/132.pdf
    id: soprofen-guide-technique-blocs-baies-2022
    title: Guide technique Blocs-baies 2022, SOPROFEN, réf. DOC86505 (09/2022)
    last_modified: 2022-09-01
source_pages:
  - resource: raw/moustiquaires/exports/133.pdf
    pages: 15-16
  - resource: raw/moustiquaires/exports/132.pdf
    pages: 73, 84-85
generated:
  by: process:claude-code
  at: 2026-10-09T02:30:00Z
---

Les deux dernières pages du guide technique du volet roulant
[Bloc LX](/equipements/volet-roulant-bloc-lx-demi-linteau.md) de [SOPROFEN](/fournisseurs/soprofen.md)
sont des plans d'intégration : « Bloc LX TREUIL, isolation de 120 mm » et « Bloc LX MOTEUR, isolation
de 120 mm ». Chaque plan montre la coupe verticale du bloc-baie (la fenêtre et son volet) sous un
demi-linteau, puis une coupe horizontale, avec un tableau de cotes par fabricant de demi-linteau
(TERREAL, GENOVA, PREFATEC, IMERYS ; STRADAL pour le moteur seul). **Une étude préalable est
indispensable** avant toute intégration [1 p. 2, 15, 16].

# Cotes d'intégration

Cotes d'intégration, en millimètres, relevées sur les tableaux des deux plans, pour une isolation de
120 mm. Une ligne par fabricant de demi-linteau et par manœuvre. Les colonnes sont les cotes
imprimées sur les dessins : Sd, DC, Ti, Te, ØC, A et B. Le guide n'en donne pas
de définition écrite ; DC est la distance de dos de la coulisse à l'axe de la tige oscillante
(rubrique 10, voir [Manœuvres et motorisations](/equipements/volet-roulant-bloc-lx-manoeuvres-et-motorisations.md)).
Sd et DC ne sont donnés que pour le treuil [1 p. 15, 16].

| Fabricant du demi-linteau | Manœuvre | Sd | DC | Ti | Te | ØC | A | B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TERREAL | treuil | 148,5 | 15 | 50 | 150 | 190 | 135 | 70 |
| GENOVA | treuil | 148,5 | 15 | 50 | 150 | 190 | 135 | 70 |
| PREFATEC | treuil | 148,5 | 15 | 70 | 210 | 180 | 135 | 70 |
| IMERYS | treuil | 148,5 | 15 | 55 | 80 | 180 | 135 | 70 |
| TERREAL | moteur | - | - | 55 | 80 | 180 | 135 | 70 |
| GENOVA | moteur | - | - | 50 | 150 | 190 | 135 | 70 |
| PREFATEC | moteur | - | - | 70 | 210 | 180 | 135 | 70 |
| IMERYS | moteur | - | - | 50 | 55 | 180 | 135 | 70 |
| STRADAL | moteur | - | - | 100 | 100 | 165 | 135 | 70 |

(schéma: raw/moustiquaires/exports/133.pdf, p. 15 pour le treuil, p. 16 pour le moteur)

STRADAL n'a pas de ligne « treuil » : la manœuvre par treuil est incompatible avec les
demi-linteaux sans espace latéral (réservation) pour le déport, tel que le STRADAL [1 p. 5, 11]. Les
hauteurs H maximales de ce demi-linteau sont à la ligne B du tableau des lames, celles des autres
fabricants à la ligne A ([Tabliers et lames](/equipements/volet-roulant-bloc-lx-demi-linteau.md)).

# Plans

Cotes imprimées sur les coupes verticales des deux plans : « H demi-linteau = 292,5 mm » ; « 216 mini
sans renfort » ; « 45 mini » ; « 75 » ; « Hauteur châssis » ; « H = hauteur » ; « SD » (sur le plan
du treuil) ; l'« alignement dormant / retombée de linteau », la « tige oscillante » (plan du treuil),
la « côte intérieure » et la « côte extérieure ». Sur les coupes horizontales : « 327 », « 42 »,
« L, largeur dos coulisses », « Ti », « Te », « DC » (plan du treuil), « A » et « B » [1 p. 15, 16].

![Bloc LX treuil, isolation de 120 mm : coupe verticale](/assets/equipements/volets-roulants/bloc-lx/bloc-lx-treuil-isolation-120-mm-coupe-verticale.png)

![Bloc LX treuil, isolation de 120 mm : coupe horizontale et tableau des cotes](/assets/equipements/volets-roulants/bloc-lx/bloc-lx-treuil-isolation-120-mm-coupe-horizontale-et-cotes.png)

![Bloc LX moteur, isolation de 120 mm : coupe verticale](/assets/equipements/volets-roulants/bloc-lx/bloc-lx-moteur-isolation-120-mm-coupe-verticale.png)

![Bloc LX moteur, isolation de 120 mm : coupe horizontale et tableau des cotes](/assets/equipements/volets-roulants/bloc-lx/bloc-lx-moteur-isolation-120-mm-coupe-horizontale-et-cotes.png)

(schéma: raw/moustiquaires/exports/133.pdf, p. 15 et 16)

# Ce que la source ne donne pas

Le guide ne donne ces plans que pour une isolation de 120 mm ; pour les doublages de 100, 140 et
160 mm, il ne donne que les coupes du renfort d'inertie
([Options](/equipements/volet-roulant-bloc-lx-options.md)). Il ne donne aucun plan pour le
demi-linteau EVENO, dont il cite pourtant la hauteur H maximale (ligne A).

# Citations

[1] Guide technique Bloc LX, SOPROFEN, extrait du catalogue — `raw/moustiquaires/exports/133.pdf`,
p. 2, 5, 11, 15 et 16 (numérotation du PDF ; pages imprimées 111, 114, 120, 124 et 125)

# Voir aussi

- [Volet roulant Bloc LX : cotes, tabliers, lames, coulisses et lames finales](/equipements/volet-roulant-bloc-lx-demi-linteau.md)
- [Manœuvres et motorisations du Bloc LX](/equipements/volet-roulant-bloc-lx-manoeuvres-et-motorisations.md)
- [Options du Bloc LX](/equipements/volet-roulant-bloc-lx-options.md)
- [Pose du coffre demi-linteau Bloc LX](/procedures/pose-bloc-lx-demi-linteau-soprofen.md)
- [SOPROFEN](/fournisseurs/soprofen.md)
- [Guide technique volet roulant demi-linteau Bloc LX SOPROFEN](/sources/soprofen-guide-technique-bloc-lx.md)
