---
type: Profilé
title: Cotes de débit du système 76
description: Les cotes à déduire de la dimension hors tout pour débiter dormants, meneaux, ouvrants, battements et seuils du système 76 Advanced à joint central, et les limites des châssis cintrés et trapézoïdaux.
tags: [systeme-76-advanced, cote-de-debit, cintrage, trapeze, dormant, ouvrant, meneau, battement, seuil, atelier]
systeme: 76
fournisseur: KÖMMERLING
usage: atelier
status: stable
sources:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
source_pages:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    pages: 81-88, 302-303
generated:
  by: process:claude-code
  at: 2026-09-18T09:00:00Z
---

# Ce que donnent ces tableaux

Les cotes de débit du **système 76 Advanced à joint central** de
[profine](/fournisseurs/profine.md) sont des **cotes à déduire**, pas des longueurs à couper.
On part de la dimension hors tout de l'élément et on retranche, coupe par coupe, la valeur du
profilé concerné. « Pour déterminer les cotes à déduire, il faut se reporter aux valeurs indiquées
dans les tableaux des pages suivantes. Ces valeurs sont à prendre sur les différentes coupes
représentées pour chaque cas. » [1 p. 81]

Sur chaque planche, la coupe cotée porte des **repères numérotés ①, ②, ③…** sous le dessin, et
chaque ligne du tableau commence par le repère de la cote qu'elle donne ; une ligne sans repère
(les renforts, le meneau d'ouvrant) n'a pas de cote dessinée. Les planches sont à l'échelle 1:1,
les vignettes des tableaux « non à l'échelle ».

Quatre dimensions se déduisent en cascade, avec les sigles fixés par les
[Directives générales profine](/sources/profine-directives-generales.md) :

| Sigle | Dimension |
| --- | --- |
| DHT | dimension hors tout, c'est-à-dire la dimension extérieure du dormant |
| DEO | dimension extérieure d'ouvrant |
| DFO | dimension de feuillure d'ouvrant |
| — | dimension de vitrage |

**Les valeurs de chaque tableau valent pour une seule coupe.** Un dormant a deux montants : une
largeur hors tout perd la valeur du montant gauche *et* celle du montant droit.

# L'exemple du manuel

Fenêtre à deux vantaux avec meneau, « Dimension extérieur dormant = dimension hors tout DHT =
2000 x 1200 mm (L x H) », dormant 76171, meneau 76372, ouvrant 76271 [1 p. 81] :

![Fenêtre deux vantaux de l'exemple, 2 000 × 1 200 mm](/assets/procedures/moe-76-advanced/debit-exemple-fenetre-deux-vantaux.png)

La largeur hors tout de 2 000 mm est partagée à l'axe du meneau en X et Y. La coupe horizontale
ci-dessous montre, de gauche à droite, le dormant, l'ouvrant gauche, le meneau, l'ouvrant droit et
le dormant : sous la coupe, `a` est la cote du dormant et `b` celle du meneau, retranchées de X et
de Y ; l'ouvrant se lit de l'extérieur vers le vitrage, **20 mm** de la dimension extérieure
d'ouvrant à la dimension de feuillure, puis **40 mm** de la feuillure au vitrage, de chaque côté.

![Coupe de l'exemple : dormant, ouvrants et meneau](/assets/procedures/moe-76-advanced/debit-coupe-exemple-deux-vantaux-meneau.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 81, registre 2.3.1, p. 1)

Les formules du manuel : **DEO = X / Y – (a + b)** ; **Dimension vitrage = DEO – 2 x c**.

| Étape | Calcul | Résultat (mm) |
| --- | --- | --- |
| Demi-largeur au meneau | X = 2 000 / 2 | 1 000 |
| Dimension extérieure d'ouvrant | DEO = X − (a + b) = 1 000 − (38 + 13) | 949 |
| Dimension du vitrage | 949 − 2 × 60 | 829 |

`a` = 38 est la cote du dormant 76171 (« Page 2 (tableau) »), `b` = 13 celle du meneau 76372
(« Page 3 (tableau) »), `c` = 60 celle du vitrage de l'ouvrant 76271 (« Page 4 (tableau) ») —
les trois se lisent dans les tableaux ci-dessous.

![Repères a, b et c de l'exemple](/assets/procedures/moe-76-advanced/debit-exemple-reperes-a-b-c.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 81)

# Cotes de débit des dormants

Cotes à déduire de la **dimension hors tout (DHT)**, en mm, pour une seule coupe, relevées sur le
registre 2.3.1 (p. 2, version octobre 2021). Les deux pictogrammes de la planche montrent la
coupe prise sur un châssis fixe à traverse et sur une fenêtre oscillo-battante, la hauteur DHT
et la largeur DHT cotées.

![Coupes cotées des dormants, repères ① à ④](/assets/procedures/moe-76-advanced/debit-dormants-reperes.png)

Sur la coupe de droite, le dormant avec l'ouvrant : **①** DEO et **②** DFO, prises depuis le bord
extérieur du dormant, la DFO se trouvant **20 mm** plus loin que la DEO ; un jeu de **12+1 mm**
est coté entre le dormant et l'ouvrant. Sur la coupe de gauche, le dormant seul : **④** meneau /
traverse et **③** vitrage fixe. Les repères ③ et ④ du dessin sont tracés dans cet ordre ; le
tableau numérote Vitrage (fixe) ③, Meneau/traverse ④ et Parclose ⑤, repère que le dessin ne porte
pas.

| Dormant | DEO (mm) | DFO (mm) | Vitrage fixe (mm) | Renfort de dormant (mm) | Meneau/traverse (mm) | Renfort de meneau/traverse (mm) | Parclose (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 76171 | 38 | 58 | 59 | 45 | 40 | 71 | 46 |
| 76172 | 56 | 76 | 77 | 61 | 58 | 89 | 64 |
| 76173 | 68 | 88 | 89 | 75 en haut, 45 vers le bas | 70 | 101 | 76 |
| 76177 | 15 | 35 | 36 | 24 | 17 | 50 | 25 |
| 76178 | 15 | 35 | 36 | 24 | 17 | 50 | 25 |
| 76180 | 38 | 58 | 59 | 45 | 40 | 71 | 46 |
| 76185 | 15 | 35 | 36 | 24 | 17 | 50 | 25 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 2)

Le renfort du **76173** porte deux valeurs, « 75\*/45\*\* », avec le renvoi « \*haut,
\*\*vers le bas » : 75 mm en haut, 45 mm vers le bas.

Les dormants rénovation **76177, 76178 et 76185** ont les mêmes cotes de débit ; les dormants
**76171 et 76180** aussi.

Trois de ces sept dormants ne sont pas proposés par PROFERM : **76173** et **76178** n'ont jamais
figuré au cahier technique PERFORM76. Voir
[Dormants PERFORM76](/profiles/perform76-dormants.md).

# Cotes de débit des meneaux de dormant

Cotes à déduire de la **dimension hors tout (DHT)**, en mm, pour une seule coupe, relevées sur le
registre 2.3.1 (p. 3, version octobre 2021). Ces valeurs se prennent **à l'axe du meneau** : le
pictogramme cote la « Dimension à l'axe », de l'axe du meneau au bord extérieur du dormant.

![Coupe cotée du meneau, repères ① à ⑤](/assets/procedures/moe-76-advanced/debit-meneaux-reperes.png)

Toutes les cotes partent de l'axe du meneau : vers l'ouvrant, **①** DEO et **②** DFO, avec un jeu
de **12+1 mm** ; vers le côté fixe, **④** meneau dans meneau, **⑤** parclose et **③** vitrage fixe.

| Meneau | DEO (mm) | DFO (mm) | Vitrage fixe (mm) | Meneau dans meneau (mm) | Renfort de meneau/traverse (mm) | Parclose (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| 76372 | 13 | 33 | 32 | 19 | 46 | 21 |
| 76373 | 26 | 46 | 45 | 19 | 46 | 34 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 3)

Le **débit du renfort de meneau est le même pour les deux meneaux** — 46 mm — alors que tout le
reste change. Voir [Meneaux PERFORM76](/profiles/perform76-meneaux.md).

# Cotes de débit des ouvrants

Cotes à déduire de la **dimension extérieure d'ouvrant (DEO)**, et non de la DHT, en mm, pour une
seule coupe, relevées sur le registre 2.3.1 (p. 4, version octobre 2021).

![Coupe cotée de l'ouvrant, repères ① à ④](/assets/procedures/moe-76-advanced/debit-ouvrants-reperes.png)

Toutes les cotes partent du bord extérieur de l'ouvrant : **①** DFO, puis **③** renfort, **④**
parclose et **②** vitrage, dans cet ordre de longueur sur le dessin.

| Ouvrant | DFO (mm) | Vitrage (mm) | Renfort d'ouvrant (mm) | Parclose (mm) | Meneau (mm) | Renfort de meneau (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| 76271 | 20 | 60 | 55 | 57 | 51 | 82 |
| 76272 | 20 | 92 | 87 | 89 | 83 | 116 |
| 76275 | 20 | 52 | 47 | 49 | 43 | 74 |
| 76279 | 20 | 92 | 87 | 89 | 83 | 116 |
| 76281 | 20 | 52 | 47 | 49 | 43 | 74 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 4)

La feuillure se déduit de 20 mm sur les cinq ouvrants. Les 76275 et 76281 ont les mêmes cotes, les
76272 et 76279 aussi.

L'ouvrant **76271 n'est pas au cahier technique PERFORM76**, qui n'en documente que quatre. Voir
[Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md).

# Cotes de débit d'un dormant recevant un battement

Ces cotes ne se déduisent pas de la DHT mais de **X, la distance de l'axe du châssis au bord
extérieur du dormant**. Elles changent avec le battement utilisé : une ligne par couple dormant et
battement (registre 2.3.1, p. 5, 6 et 7, version octobre 2021). Sur les trois coupes, **①** DEO et
**②** DFO partent de l'axe, la DFO **20 mm** plus loin ; le jeu entre ouvrants est de **12+1 mm**.
Sous le battement, l'écart entre l'axe et l'ouvrant qui porte le battement est coté **4 mm**
(76473), **6 + 6 mm** (76471) et **6 mm** (76472).

![Coupe cotée du battement 76473](/assets/procedures/moe-76-advanced/debit-battement-76473-reperes.png)

![Coupe cotée du battement 76471](/assets/procedures/moe-76-advanced/debit-battement-76471-reperes.png)

![Coupe cotée du battement 76472](/assets/procedures/moe-76-advanced/debit-battement-76472-reperes.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 85 à 87)

| Dormant | Battement | DEO (mm) | DFO (mm) |
| --- | --- | --- | --- |
| 76171 | 76471 | X − 32 | X − 72 |
| 76171 | 76472 | X − 41 | X − 81 |
| 76171 | 76473 | X − 24 | X − 64 |
| 76172 | 76471 | X − 50 | X − 90 |
| 76172 | 76472 | X − 59 | X − 99 |
| 76172 | 76473 | X − 42 | X − 82 |
| 76173 | 76471 | X − 62 | X − 102 |
| 76173 | 76472 | X − 71 | X − 111 |
| 76173 | 76473 | X − 54 | X − 94 |
| 76177 | 76471 | X − 9 | X − 49 |
| 76177 | 76472 | X − 18 | X − 58 |
| 76177 | 76473 | X − 1 | X − 41 |
| 76178 | 76471 | X − 9 | X − 49 |
| 76178 | 76472 | X − 18 | X − 58 |
| 76178 | 76473 | X − 1 | X − 41 |
| 76180 | 76471 | X − 32 | X − 72 |
| 76180 | 76472 | X − 41 | X − 81 |
| 76180 | 76473 | X − 24 | X − 64 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 5, 6 et 7)

**Le dormant 76185 n'apparaît sur aucune des trois planches de battement**, alors qu'il figure sur
la planche générale des dormants. Ni exclusion ni oubli ne sont énoncés —
entrée **VER-21** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

# Débit du battement lui-même et de son renfort

Formules relevées sur le registre 2.3.1 (p. 5, 6 et 7, version octobre 2021). `DEO` est la
dimension extérieure d'ouvrant obtenue au tableau précédent.

| Battement | Débit du battement | Renfort associé | Débit du renfort |
| --- | --- | --- | --- |
| 76471 | DEO − 2 × 47 mm | V316 | DEO − 2 × 59 mm |
| 76472 | DEO − 2 × 47 mm | V317 | DEO − 2 × 59 mm |
| 76473 avec embouts M462 | DEO − 2 × 47 mm | — | — |
| 76473 sans embout bas M462 | DEO − (47 mm + 33 mm) | — | — |

Les trois battements se débitent à DEO − 2 × 47 mm ; le 76473 se débite à DEO − (47 mm + 33 mm)
sans embout bas M462. La planche du 76473 ne porte pas de ligne de renfort. Voir
[Renforts du système 76](/profiles/systeme-76-renforts.md).

# Cotes de débit des seuils aluminium

Cotes à déduire de la **dimension hors tout (DHT)**, en mm, relevées sur le registre 2.3.1
(p. 8, version octobre 2021). Deux coupes verticales sont dessinées, « Seuil alu A075 » et « Seuil
alu A076 / A077 », avec un jeu de **10+1 mm** entre l'ouvrant et le seuil.

![Coupes cotées des seuils A075 et A076 / A077](/assets/procedures/moe-76-advanced/debit-seuils-reperes.png)

Toutes les cotes partent du dessous du seuil : **①** DEO, **②** DFO et **③** dormant. Sur la coupe
A076 / A077, le repère ③ est dessiné deux fois, **③a** jusqu'au dessous du dormant posé sur le
seuil et **③b**, plus court ; sur la coupe A075, un seul **③c**.

| Cas | Seuils concernés | DEO (mm) | DFO (mm) | Dormant (mm) |
| --- | --- | --- | --- | --- |
| a | A076, A077-A343 | 10 | 30 | 20 |
| b, usiné | A076, A077-A343 | 10 | 30 | 10,6 |
| c, dormant usiné | A075 | 10 | 30 | 12,1 |

Débit du renfort vertical de dormant, en mm, dans les trois mêmes cas :

| Dormant | Cas a (mm) | Cas b, usiné (mm) | Cas c, dormant usiné (mm) |
| --- | --- | --- | --- |
| 76171 | 100 | 91 | 92 |
| 76172 | 100 | 91 | 92 |
| 76173 | 100 | 91 | 92 |
| 76177 | 86 | 75 | 75 |
| 76178 | 86 | 75 | 75 |
| 76180 | 76 | 74 | 74 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 8)

L'en-tête du tableau se lit « **a** | **b** usiné | **c** dormant usiné », les seuils A076 et
A077-A343 étant portés sous a et sous b, l'A075 sous c. Les colonnes correspondent aux repères
③a, ③b et ③c du dessin ; la planche ne dit pas quelle pièce est usinée dans le cas b — entrée
**VER-22** du registre [Informations à vérifier](/anomalies/informations-a-verifier.md).

Le seuil **A076** est celui que PROFERM met en œuvre, avec son rejet d'eau **A062**. Voir
[Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md).

# Châssis cintrés et trapézoïdaux

## Diamètre minimum de cintrage

Le **cintrage** est la mise en forme courbe d'un profilé PVC, pour une menuiserie cintrée (un
châssis en plein cintre, par exemple ; voir le [glossaire](/reference/glossaire.md)). Les
diamètres minimum de cintrage des profilés du système 76 Advanced ont été déterminés en
coopération avec les fabricants de machines de cintrage : ce sont les diamètres mini qui étaient
jugés qualitativement acceptables. Les conditions préalables sont que les inserts prévus par le
constructeur de la machine soient utilisés et que leurs recommandations soient suivies. Le
**diamètre de cintrage minimum est égal à la largeur du profilé × 10** [1 p. 302].

**Attention** — conditions pour un bon cintrage : **les profilés ne doivent pas être cintrés avec
leurs joints** ; les joints d'étanchéité doivent être enfilés manuellement sur le profilé cintré
[1 p. 302].

Diamètre de cintrage minimum de chaque profilé, en mètres, tel que la planche le liste :

| Profilé | Famille | Diamètre de cintrage minimum Ø (m) |
| --- | --- | --- |
| 76171 | dormant | 0,74 |
| 76172 | dormant | 0,92 |
| 76173 | dormant | 1,04 |
| 76177 | dormant rénovation | 0,91 |
| 76178 | dormant rénovation | 1,21 |
| 76180 | dormant | 0,94 |
| 76281 | ouvrant | 0,70 |
| 76271 | ouvrant | 0,78 |
| 76272 | ouvrant | 1,10 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 302, registre 2.6.1, p. 1, version juillet 2017, dessin non à l'échelle)

Le dormant rénovation 76185, les meneaux et les battements ne figurent pas dans la liste.

La planche montre à gauche une fenêtre en plein cintre vue de face, le rayon **R** du cintre
tracé depuis le centre de l'arc, et le sommet de l'arc entouré d'un cercle tireté ; à droite, ce
détail agrandi : un tronçon d'arc de dormant et d'ouvrant cintrés, avec, en tireté, la coupe des
deux profilés dans l'épaisseur de l'arc.

![Diamètre minimum de cintrage des profilés du système 76](/assets/procedures/moe-76-advanced/cintrage-diametre-minimum.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 302)

## Fenêtre oblique : trapèze, triangle

Une fenêtre **oblique** (en trapèze ou en triangle) a au moins un côté incliné, qui forme avec le
montant un angle aigu appelé **angle de pointe**. Le plus petit angle de pointe réalisable sans
usinage supplémentaire de l'ouvrant, tout en respectant le **jeu de feuillure de 12 + 1 mm**
(l'espace entre le dormant et l'ouvrant) et la **distance de l'axe de rotation de la
quincaillerie = 10,5 / 20 mm, en partant du fond de feuillure dormant**, est de **29,5°** entre
l'ouvrant incliné et le montant [1 p. 303].

La planche montre, en vignette, une fenêtre trapézoïdale vue de face, l'angle de pointe entouré
d'un cercle tireté ; puis ce coin agrandi. On y lit, de gauche à droite, le dormant et l'ouvrant
du montant vertical (coupes en tireté) et, en haut à droite, leur coupe le long du côté incliné.
Cotes portées : **12** mm de jeu de feuillure entre dormant et ouvrant ; l'axe de rotation (le
petit cercle) placé à **10,5** mm en hauteur et à **20** mm du fond de feuillure dormant, annoté
« Axe de rotation dépendant de la quincaillerie » ; l'angle de **29,5°** entre le montant de
l'ouvrant et son côté incliné ; un second angle, de **59°**, coté en bas à droite de la planche
sans légende.

![Angle de pointe minimum d'une fenêtre oblique du système 76](/assets/procedures/moe-76-advanced/trapeze-angle-de-pointe-minimum.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 303, registre 2.6.1, p. 2, version juillet 2017, dessin non à l'échelle)

# Citations

[1] Mise en œuvre Système 76 Advanced, profine —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, registre 2.3.1 « Cotes de débit — Coupes »,
p. 81 à 88 du PDF (pages imprimées 1 à 8), version d'octobre 2021 ; registre 2.6.1 « Châssis cintrés et
trapézoïdaux », p. 302 et 303 du PDF (pages imprimées 1 et 2), version juillet 2017

# Voir aussi

- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md)
- [Meneaux PERFORM76](/profiles/perform76-meneaux.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
