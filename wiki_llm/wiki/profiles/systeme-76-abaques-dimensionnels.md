---
type: Profilé
title: Abaques dimensionnels du système 76
description: Les limites de dimension d'ouvrant du système 76 Advanced selon le renfort, la couleur et l'épaisseur de vitrage, et pourquoi un profilé de couleur change la fabrication.
tags: [systeme-76-advanced, abaque, dimension, renfort, couleur, vitrage, paumelle, poids-ouvrant]
status: stable
sources:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
generated:
  by: process:claude-code
  at: 2026-09-18T09:00:00Z
stale_after: 2024-12-31
---

# Ce qu'est un abaque dimensionnel d'ouvrant

Les abaques du registre 2.3.3 du manuel
[Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md) donnent, pour
chaque renfort, **les cotes extérieures d'ouvrant maximales réalisables**. Quatre facteurs les
déterminent (registre 2.3.3, p. 2, version janvier 2016) :

1. les valeurs statiques **IW et IG du renfort** — voir
   [Renforts du système 76](/profiles/systeme-76-renforts.md)
2. la **conception du châssis** — un vantail seul, deux vantaux, avec ou sans battement
3. la **couleur** du profilé
4. l'**épaisseur du vitrage** et la ferrure retenue

**Ce sont des cotes d'ouvrant, pas des cotes de fenêtre.** Pour obtenir la dimension de l'élément
il faut ajouter les profilés adjacents sur les quatre côtés, selon la règle de calcul des
[Directives générales profine](/sources/profine-directives-generales.md). Les dimensions
maximales de baie relèvent, elles, du
[DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md) : les deux jeux de limites sont
indépendants et s'appliquent tous les deux.

Les calculs d'inertie sont faits pour une **flèche de 1/150ᵉ**, sous des charges de vent de
**0,8 kN/m² (800 Pa)** et **1,2 kN/m² (1 200 Pa)** (registre 2.3.3, p. 3, version janvier 2022).

# Cotes maximales d'ouvrant par renfort

Bornes des abaques d'ouvrant simple — châssis à la française et oscillo-battant — en cm, relevées
sur le registre 2.3.3 (p. 6 à 11 et 19). Un châssis basculant se lit sur le même abaque en
inversant hauteur et largeur (registre 2.3.3, p. 1).

| Renfort | Ouvrants concernés | Largeur maxi (cm) | Hauteur maxi (cm) |
| --- | --- | --- | --- |
| V266.Z | 76275, 76281 | 130 | 235 |
| V306.Z | 76271 | 150 | 235 |
| V307.Z | 76271 | 150 | 250 |
| V308 | 76271 | 150 | 250 |
| V314.Z | 76272, 76279 | 150 | 250 |
| V326.Z | 76272, 76279 | 150 | 250 |
| capot alu A072, AluClip Pro, sans renfort acier | 76271 | 150 | 250 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.3, p. 6 à 11 et 19)

**Ces bornes sont les coins de l'abaque, pas un rectangle admissible.** Un ouvrant de 150 × 250 cm
n'est réalisable que si le point tombe sous la courbe de sa couleur et de son épaisseur de
vitrage. Les courbes elles-mêmes ne sont pas transcrites dans ce wiki : elles n'existent que sous
forme de tracés sur les planches. Pour un cas limite, lire l'abaque.

Sur les abaques à **deux vantaux** — ouvrant plus battement, registre 2.3.3, p. 12 à 18 — les
bornes retombent à **130 cm de largeur et 235 cm de hauteur** par vantail, quelle que soit la
combinaison de renforts d'ouvrant et de battement.

# La règle des 25 %

**La largeur d'ouvrant ne peut dépasser la hauteur d'ouvrant de plus de 25 %** (registre 2.3.3,
p. 6 à 11). Exemple donné par le manuel : pour une largeur de 150 cm, la hauteur doit être d'au
moins 120 cm ; pour une largeur de 130 cm, au moins 103 cm.

C'est une contrainte de **quincaillerie et de poids**, indépendante du renfort : elle interdit
l'ouvrant large et bas même quand l'abaque l'autoriserait par ailleurs.

# L'épaisseur de vitrage

L'épaisseur de vitrage se calcule en **additionnant les verres sans les intercalaires** : un
vitrage 4-12-4-12-4 compte pour 4 + 4 + 4 = **12 mm** (registre 2.3.3, p. 6).

| Règle | Valeur |
| --- | --- |
| Seuil au-delà duquel le renforcement total devient obligatoire | 12 mm |
| Épaisseurs portées par les courbes des abaques | 12, 16, 20, 24 et 28 mm |
| Épaisseur intermédiaire | arrondir à la courbe supérieure : 13 mm se lit sur la courbe 16 mm |

**Au-delà de 12 mm de verre, le renforcement total est obligatoire**, et les limites
dimensionnelles correspondantes s'appliquent. La restriction vaut aussi bien pour le blanc que
pour la couleur, et **elle s'applique également aux vantaux recevant le battement**.

L'**équerre de feuillure J079**, posée dans les quatre angles de l'ouvrant, augmente
considérablement la stabilité de l'ouvrant au poids du vitrage et repousse les limites de vitrage
(registre 2.3.3, p. 2). Sur la variante **AluClip Pro**, elle devient obligatoire dès **40 kg de
poids d'ouvrant** (registre 2.6.3, p. 1).

Le **renfort de feuillure V280** (IW 3 cm⁴, 2 mm) posé sur les montants permet aux couleurs
standard d'atteindre les dimensions des couleurs IR-Reflex (registre 2.3.3, p. 6).

# Les trois catégories de couleur

C'est le classement qui commande l'abaque à lire (registre 2.3.3, p. 3 et 4, version janvier 2022).

| Catégorie | Périmètre | Renforcement |
| --- | --- | --- |
| 1. Blanc | blanc, blanc crème et ton pierre (cccc), quel que soit le traitement de surface — couleur de base, plaxé | renfort préconisé |
| 2. Couleurs IR-Reflex | 16 couleurs listées, capables de réfléchir le rayonnement thermique du soleil | **systématique** |
| 3. Couleurs standard | tous les autres produits couleur du programme de livraison | **systématique** |

Trois cas particuliers s'attachent à la catégorie 1 :

- les **éléments à vitrage collé de couleur blanc** se font **sans renfort acier**
- les **capots aluminium** se réalisent **aux dimensions du blanc**, mais **leur fabrication suit
  les règles de la couleur** : aération des préchambres et vissage du renfort
- avec des **paumelles**, le renforcement côté paumelle n'est pas nécessaire en blanc et ton pierre

**Un profilé de couleur est donc toujours renforcé et toujours plus limité en dimension qu'un
profilé blanc.** Les couleurs IR-Reflex occupent la position intermédiaire : elles réfléchissent
le rayonnement solaire, donc elles admettent des cotes supérieures à celles des couleurs
standard, sans atteindre celles du blanc.

## Les 16 couleurs IR-Reflex

Couleurs admises dans la catégorie 2, avec leur code profine et la référence du fournisseur de
film, relevées sur le registre 2.3.3 (p. 4, version janvier 2022).

| Désignation | Code profine | Référence fournisseur |
| --- | --- | --- |
| Chêne doré | 32 | 9.2178.301 - 116700 |
| Noyer | 52 | 9.2178.307 - 116700 |
| Acajou | 24 | 9.2178.321 - 116700 |
| Anthracite veiné | 16 | 02.20.71.000001 - 116700 |
| Gris argent | 21 | 02.20.71.000007 - 116700 |
| Gris signal | UJ | 02.20.71.000009 - 504700 |
| Chêne irlandais | IO | 9.3211.305 - 114800 |
| Métal brossé silver | AR | F436-1002 |
| Métal brossé platine | MP | F436-1004 |
| Alux DB 703 | AL | F436-1014 |
| Anthracite mat | AU | F436-6003 |
| AnTeak | AN | 9.3241.302 - 119500 |
| Rouge foncé | 46 | 02.11.31.000013 - 116700 |
| Vert mousse | 53 | 02.20.61.000002 - 116700 |
| Bleu acier | 88 | 02.20.51.000001 - 116700 |
| Brun noir | 80 | 02.20.81.000010 - 116700 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.3, p. 4)

Le manuel précise que cette liste est celle **en vigueur à l'édition de la page** et qu'elle
évolue avec l'offre : le tarif en vigueur fait foi.

## Les 6 couleurs standard

| Désignation | Code profine | Référence fournisseur |
| --- | --- | --- |
| Brun chocolat | 27 | 02.20.81.000018 - 116700 |
| Honey Oak super mat | HB | 3.0078.007 - 102201 |
| Siena PR | SC | 4.0131.006 - 114800 |
| Métal brossé quartz | MQ | F436-1005 |
| Gris basalte mat | BQ | F436 6048 |
| Brun sepia ulti mat | PF | 02.20.81.000030 - 504700 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.3, p. 4)

**Deux anthracites différents coexistent dans le système** : l'**anthracite veiné 16** et
l'**anthracite mat AU** sont tous deux IR-Reflex, tandis que le **gris basalte mat BQ**, très
proche visuellement, est une couleur standard donc plus limitée en dimension. La nuance choisie
change l'abaque à lire.

# Pourquoi la couleur change la fabrication

Le manuel donne une raison, et une seule, pour la ventilation :

> « Pour éviter toute accumulation de chaleur, les préchambres extérieures des profilés de
> couleur, laqués ou filmés 2 faces ou 1 face extérieure doivent impérativement être ventilées »
> (registre 2.4.2, p. 2 et 13, version janvier 2016).

Cette ventilation s'ajoute au drainage, elle ne le remplace pas, et elle vaut aussi pour les
profilés **capotés aluminium**, traités comme des profilés de couleur.

**Le manuel n'énonce aucune raison physique au renforcement systématique des profilés de
couleur** : il pose la règle et donne les abaques réduits qui vont avec. Attribuer ce renforcement
à la dilatation thermique est une déduction plausible, que le document ne fait pas.

## Le critère objectif est réglementaire, pas commercial

Le [DTD n° DBV-25-6/16-2334_V5](/sources/dtd-6-16-2334.md) ne raisonne pas en « couleurs » mais en
**clarté colorimétrique L\***, avec un seuil unique : **L\* inférieur à 82**.

| Prescription du DTD | Ce qu'elle impose |
| --- | --- |
| Profilé PVC revêtu d'un film ou d'une laque à L\* < 82 | renfort obligatoire |
| Profilé PVC revêtu d'un capotage aluminium à L\* < 82 | renfort obligatoire, en dormant **et** en ouvrant |
| Chambres des profilés à L\* < 82 communiquant avec l'extérieur | décompression par orifices de **Ø 5 mm minimum** |
| Habillage monoparoi à L\* < 82 ou non défini | **interdit en traverse basse**, quelle que soit la technologie de coloration |

**C'est la réponse que le manuel de fabrication ne donne pas** : le renforcement d'un profilé
anthracite est une prescription du Groupe Spécialisé du CSTB, déclenchée par une mesure de clarté,
et non une préférence d'extrudeur. Les catégories de couleur ci-dessus sont la traduction
commerciale de ce seuil par profine.

Deux points d'usinage à respecter au drainage (registre 2.4.2, p. 1 et 2) :

- la chambre signalée sur chaque coupe **ne doit pas être endommagée par le perçage du trou de
  drainage**
- pour une profondeur de perçage supérieure à **50 mm**, les trous oblongs peuvent être remplacés
  par **trois trous de Ø 6 mm**

# Paumelles et points de verrouillage

Le nombre de paumelles va de **2 à 5 selon la hauteur d'ouvrant**, sur une échelle de 50 à 250 cm
portée en pied de chaque abaque (registre 2.3.3, p. 3 et 6 à 19). **Le détail de cette
correspondance n'est pas transcrit** : il est porté sous forme de barres graphiques, sans tableau
de valeurs.

Le manuel donne **deux distances maximales différentes entre points de verrouillage** :

| Page | Formulation du manuel | Valeur |
| --- | --- | --- |
| Registre 2.3.3, p. 3 | « les points de verrouillage (paumelle) ne doivent pas être distants de plus de 70 cm » | 70 cm |
| Registre 2.3.3, p. 5 | « les points de verrouillage (galets etc.) ne doivent pas être distants de plus de 80 cm » | 80 cm |

Les deux phrases emploient les mêmes mots pour des organes différents — paumelles d'un côté,
galets de l'autre. **Retenir 70 cm tant que l'arbitrage n'est pas fait** est le choix sûr, mais le
manuel ne dit pas que c'est la bonne lecture. Entrée **INC-10** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

# Poids d'ouvrant admissible selon la ferrure

Poids d'ouvrant admissibles déterminés à l'**ift Rosenheim** selon la directive **TBDK**, relevés
sur le registre 2.3.3 (p. 5, version janvier 2016). Une ligne par configuration de fixation.

| Ferrure | Acier dans le dormant ou le meneau | Vis dans l'acier | Vis dans le PVC | Poids d'ouvrant admissible (kg) | Rapport d'essais |
| --- | --- | --- | --- | --- | --- |
| 100 kg | sans | 0 | 4 | 80 | 12-002529-PR01 |
| 100 kg | court, 55 mm | 1 | 3 | 90 | 12-002529-PR02 |
| 100 kg | long | 3 | 1 | 100 | 12-002529-PR03 |
| 130 kg | sans | 0 | 6 | 100 | 12-002529-PR04 |
| 130 kg | court, 55 mm | 2 | 4 | 110 | 12-002529-PR05 |
| 130 kg | long | 5 | 1 | 130 | 12-002529-PR06 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.3, p. 5)

**Une ferrure annoncée pour 100 kg ne porte 100 kg que si l'acier long est présent dans le
dormant et que trois vis sur quatre le traversent.** Sans acier, la même ferrure tombe à 80 kg.
C'est la fixation qui commande, pas la ferrure.

La longueur de l'acier dit « long » est illisible sur la planche — le texte porte « long (5 mm) »,
ce qui ne peut pas être une longueur de vis en regard des 55 mm de l'acier court. Entrée
**VER-25** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

Le manuel insiste : ces valeurs ne valent **que pour les composants des rapports d'essais** —
compas, vis, acier — et sont **indicatives**. Le contrôle et la garantie du poids d'ouvrant
incombent au fabricant de fenêtres, dans le cadre de son contrôle de production en usine (WPK).
Voir [Roto NX](/quincaillerie/roto-nx.md).

# Jeux de dilatation des éléments couplés

Pour les éléments couplés, les dimensions maximales de dormant dépendent de la performance de
l'accouplement. Des jeux de dilatation deviennent nécessaires à partir de (registre 2.3.3, p. 1) :

| Finition | Largeur à partir de laquelle un jeu de dilatation est nécessaire (m) |
| --- | --- |
| Blanc | 3,50 |
| Capot aluminium | 3,50 |
| Film IR-Reflex | 3,50 |
| Couleur | 2,50 |

**Un mètre d'écart entre le blanc et la couleur** : c'est la conséquence dimensionnelle la plus
visible du classement par couleur.

# Citations

[1] Mise en œuvre Système 76 Advanced, profine —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, registres 2.3.3, 2.4.2 et 2.6.3

# Voir aussi

- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
