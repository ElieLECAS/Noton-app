---
type: Profilé
title: Cotes de débit du système 76
description: Les cotes à déduire de la dimension hors tout pour débiter dormants, meneaux, ouvrants, battements et seuils du système 76 Advanced à joint central, celles des capots aluminium AluClip, et les limites des châssis cintrés et trapézoïdaux.
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
    pages: 81-88, 302-303, 345-356, 404-405, 418-419
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

# Cotes de débit des capots aluminium AluClip

Le **capot aluminium** de la variante AluClip (voir le [glossaire](/reference/glossaire.md)) se
débite, comme les profilés, par des **cotes à déduire** de la dimension hors tout : **DHT**
(dimension hors tout du dormant) pour les capots de dormant, de meneau et de l'ouvrant 76283,
**DEO** (dimension extérieure de l'ouvrant) pour les capots d'ouvrant et de traverse d'ouvrant.

## Règles de débit

**Remarque concernant le débit des capots aluminium !** Pour déterminer les cotes de débit, il
faut se reporter aux valeurs indiquées dans les tableaux ci-dessous. Ces valeurs sont à prendre
sur les différentes coupes représentées pour chaque cas [1 p. 345].

**Les dimensions exactes des capots alu doivent être relevées directement sur les cadres soudés,
ébavurés.** Pour des raisons de coefficients de dilatations longitudinales divergeants entre la
matière PVC et aluminium, il est recommandé de couper les capots aluminium plus court de
**0,5 mm à chaque extrémité** [1 p. 345].

Les capots alu peuvent être débités de deux façons :

1. **Débit capots alu à onglet** — **dimensions mini ouvrant 850 × 850 mm**.
2. **Débit capot alu coupe droite 90°**. Pour la coupe droite, la liaison entre les capots peut se
   faire suivant deux variantes : 1. coupe droite à 90° ; 2. coupe droite à 90° + fraisage de
   contour. Pour ce second type de montage, il convient de **rajouter 2,3 mm par côté** aux valeurs
   indiquées respectivement dans les tableaux de débit [1 p. 345].

Les deux variantes de coupe droite : le capot vertical (en haut) s'arrête sur le capot horizontal
avec un jeu de **0,5** mm ; dans la variante 2, son bout est fraisé à **75°** au contour du capot
horizontal, qui dépasse alors de **2,3** mm.

![Coupe droite à 90° et coupe droite + fraisage de contour des capots AluClip](/assets/procedures/moe-76-advanced/aluclip-debit-coupe-droite-90-variantes.png)

Le schéma d'ensemble nomme les capots d'un châssis : à gauche, le dormant, avec ses **capots
dormant horizontaux** en haut et en bas et ses **capots dormant verticaux** sur les côtés ; à droite,
l'ouvrant, avec ses **capots ouvrant horizontaux** et **verticaux**.

![Capots de dormant et capots d'ouvrant, horizontaux et verticaux](/assets/procedures/moe-76-advanced/aluclip-debit-capots-dormant-ouvrant.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 345, registre 2.6.2, p. 41, version décembre 2016)

## Capots de dormant et de traverse

Le **capot alu dormant horizontal** se débite à **DHT + 2,4** mm, quel que soit le dormant
(76171, 76172, 76173, 76178, 76177 / 76185, 76180) : il dépasse de **1,2** mm de chaque côté la
largeur hors tout ; jeu de feuillure **12 + 1** mm [1 p. 346].

![Capot de dormant horizontal, dépassement de 1,2 mm](/assets/procedures/moe-76-advanced/aluclip-debit-capot-dormant-horizontal.png)

Le **capot alu dormant vertical** (repère ①) et le **capot alu traverse / meneau** (repère ②) se
débitent à DHT moins la cote du tableau ; le capot vertical s'arrête à **1,2** mm sous l'extrémité
du dormant et à **75°** contre le capot horizontal ; **différence entre dormant et traverse
0,5 mm** [1 p. 347].

| Dormant | ① Capot alu dormant vertical, à déduire de DHT (mm) | ② Capot alu traverse / meneau, à déduire de DHT (mm) |
| --- | --- | --- |
| 76171 | 72,2 | 72,4 |
| 76172 | 90,2 | 90,4 |
| 76173 | 102,2 | 102,4 |
| 76178 | 49 | 50 |
| 76177 / 76185 | 49 | 50 |
| 76180 | 72 | 73 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 347, registre 2.6.2, p. 43, version mars 2021)

Les valeurs des 76178, 76177 / 76185 et 76180 sont écrites en gras sur la planche, et la
différence entre les deux lignes y est de 1 mm au lieu des 0,2 mm des 76171 à 76173.

![Capot de dormant vertical et capot de traverse](/assets/procedures/moe-76-advanced/aluclip-debit-capot-dormant-vertical-traverse.png)

## Capots verticaux sur seuil aluminium

Sur une porte-fenêtre à seuil, le capot vertical du dormant et celui du meneau descendent sur le
seuil. En **variante 1**, la DHT est cotée jusqu'au pied du seuil et la cote à déduire est
**0,0** ; en **variante 2**, le capot s'arrête sur le seuil, à la hauteur repérée ② (A076, A077,
A343) ou ① (A075). Le bout du capot est coupé à **84°** (seuils A076, A077, A343) ou **80°** (seuil
A075), avec un jeu de **0,5** mm ; la variante 1 porte en plus un rayon **R1,5** et **4,4** mm entre
le capot et le seuil, et, sur le seuil A076, une hauteur de **12,6** mm [1 p. 348-349].

| Seuil | Variante | À déduire de DHT, capot alu dormant vertical (mm) | À déduire de DHT, capot alu meneau (mm) | Page PDF |
| --- | --- | --- | --- | --- |
| A076 | 1 | 0,0 | 0,0 | 348 |
| A076 / A077 / A343 | 2 | 10,5 | 10,5 | 348 |
| A075 | 1 | 0,0 | 0,0 | 349 |
| A075 | 2 | 23,2 | 23,2 | 349 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 348 et 349, registre 2.6.2, p. 44 (mars 2021) et 45 (décembre 2016))

![Capot vertical sur seuils A076, A077 et A343, variantes 1 et 2](/assets/procedures/moe-76-advanced/aluclip-debit-seuil-a076-a077-a343.png)

![Capot vertical sur seuil A075, variantes 1 et 2](/assets/procedures/moe-76-advanced/aluclip-debit-seuil-a075.png)

Les hauteurs repérées ① et ② ne portent pas de valeur écrite.

## Capot de meneau

Le **capot alu meneau** (repère ①, du bord du capot à l'axe du meneau) se déduit de l'**axe** ;
son bout est coupé à **75°** [1 p. 350].

| Meneau | À déduire, capot alu meneau (mm) |
| --- | --- |
| 76372 | 47,5 |
| 76373 | 60,5 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 350, registre 2.6.2, p. 46, version mars 2021)

![Capot de meneau, débit depuis l'axe](/assets/procedures/moe-76-advanced/aluclip-debit-capot-meneau.png)

## Capots d'ouvrant

Le **capot alu horizontal** d'ouvrant (repère ①) se déduit de la DEO ; le **capot alu vertical**
(repère ②) de la DHT, selon les en-têtes des tableaux ; les deux sont coupés à **75°** [1 p. 351-352].

| Ouvrant | ① Capot alu horizontal, à déduire de DEO (mm) | ② Capot alu vertical, à déduire (mm) |
| --- | --- | --- |
| 76281 | 29,8 | 69,2 |
| 76271 | 29,8 | 77,2 |
| 76272 | 29,8 | 109,2 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 351 et 352, registre 2.6.2, p. 47 et 48, version décembre 2016)

Le tableau du capot vertical est intitulé « partant de la dimension hors tout = DHT », mais la
coupe cote le repère ② sous la **DEO** — entrée **INC-76** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

![Capot d'ouvrant horizontal](/assets/procedures/moe-76-advanced/aluclip-debit-capot-ouvrant-horizontal.png)

![Capot d'ouvrant vertical](/assets/procedures/moe-76-advanced/aluclip-debit-capot-ouvrant-vertical.png)

L'ouvrant à ouverture extérieure **76283** (capot **A039**) a ses propres cotes, à déduire de la
DHT : **1,2** mm pour le capot alu horizontal (repère ①) et **109,2** mm pour le capot alu vertical
(repère ②). **Coupe à 45° : appliquer les cotes de débit ① pour le capot A039 horizontal et
vertical** [1 p. 353].

![Ouvrant 76283 et capot A039](/assets/procedures/moe-76-advanced/aluclip-debit-ouvrant-76283-a039.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 353, registre 2.6.2, p. 49, version décembre 2016)

La hauteur du pictogramme est écrite « DNHT Hauteur ».

## Capots de traverse d'ouvrant et de battement

Le **capot alu traverse d'ouvrant** (repère ①, du bord du capot à l'axe) se déduit de la DEO,
bouts coupés à 75° [1 p. 354] :

| Traverse d'ouvrant | À déduire de DEO, capot alu traverse d'ouvrant (mm) |
| --- | --- |
| 76300 | 33,4 |
| 76301 | 41,4 |
| 76303 | 54,4 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 354, registre 2.6.2, p. 50, version décembre 2016)

![Capot de traverse d'ouvrant](/assets/procedures/moe-76-advanced/aluclip-debit-capot-traverse-ouvrant.png)

**Capot alu battement = longueur battement** : le capot du battement se débite à la longueur du
battement lui-même ; la planche dessine les battements 76471 (en haut) et 76472 (en bas), jeu
**12 + 1** mm [1 p. 355]. Le débit du battement est dans *Débit du battement lui-même et de son
renfort* ci-dessus.

![Capot de battement, débit à la longueur du battement](/assets/procedures/moe-76-advanced/aluclip-debit-capot-battement.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 355, registre 2.6.2, p. 51, version décembre 2016)

## Capot A073 du dormant 76172

Le dormant **76172** reçoit aussi le capot **A073** (capot alu dormant ouverture extérieure ou
élargisseur, voir [Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md)) ;
la coupe cote **1,3** mm entre le bout du capot vertical et l'extrémité du dormant [1 p. 356].

| Capot A073 sur dormant 76172 | À déduire de DHT (mm) |
| --- | --- |
| capot alu A073 dormant vertical (du haut) | 64 |
| capot alu A073 dormant vertical (du bas) | 64 |
| capot alu A073 pour seuil (du bas) | 0,0 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 356, registre 2.6.2, p. 52, version décembre 2016)

**Pour le capot alu A073 horizontal : DHT + 2,6 mm** [1 p. 356].

![Capot A073 sur dormant 76172](/assets/procedures/moe-76-advanced/aluclip-debit-capot-a073-76172.png)

## Capot A072 de l'AluClip Pro

Le capot **A072** de l'ouvrant 76271 en variante **AluClip Pro** se débite selon la variante de
fabrication retenue ; la prise de cote s'effectue **toujours sur le cadre soudé** [1 p. 404-405].

| Variante | Capot | Cote de débit (mm) |
| --- | --- | --- |
| 1, coupe à 45° | capot A072, horizontal et vertical | DEO − 59,6 (2 × 29,8) |
| 2, imbriquée (débit à 90°) | capot alu horizontal | DEO − 59,6 (2 × 29,8) |
| 2, imbriquée (débit à 90°) | capot alu vertical | DEO − 94 (2 × 47) |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 404 et 405, registre 2.6.3, p. 10 et 11, version décembre 2016)

La mise en œuvre du capot A072 est dans
[Capotage AluClip du système 76 Advanced](/procedures/capotage-aluclip-systeme-76.md#mise-en-œuvre-du-capot-a072-aluclip-pro).

## Ouvrant 76282 et capot A195 de l'AluClip Zero

Cotes de débit de l'ouvrant **76282** de la variante **AluClip Zero**, à prendre sur la coupe, par
dormant : ① **DEO** (dimension extérieure de l'ouvrant), ② **FFM** (dimension fond de feuillure
ouvrant), ③ **vitrage**, en mm à déduire. La coupe cote aussi **20** mm entre ① et ②, **5** mm entre
le fond de feuillure et le vitrage et **7,8** mm en tête [1 p. 418].

| Dormant | ① DEO (mm) | ② FFM (mm) | ③ Vitrage (mm) |
| --- | --- | --- | --- |
| 76171 | 38 | 58 | 96 |
| 76172 | 56 | 76 | 114 |
| 76173 | 68 | 88 | 126 |
| 76177 | 23 | 43 | 81 |
| 76178 | 23 | 43 | 81 |
| 76180 | 38 | 58 | 96 |
| 76185 | 15 | 35 | 71 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 418, registre 2.6.5, p. 10, version janvier 2019)

![Cotes de débit de l'ouvrant AluClip Zero](/assets/procedures/moe-76-advanced/aluclip-zero-debit-ouvrant.png)

Le **capot A195** se débite à **90°, coupe droite** : capots alu verticaux (coupe A-A) à **DEO − 62,6
(2 × 31,3) mm** ; capots alu horizontaux (coupe B-B, **73** mm d'A195) à **DEO − 146 (2 × 73) mm**.
**À noter : les dimensions exactes des capots alu doivent être relevées directement sur les cadres
soudés, ébavurés.** Pour des raisons de coefficients de dilatation longitudinale divergents entre la
matière PVC et l'aluminium, il est recommandé de couper les capots aluminium plus court de **0,5 mm à
chaque extrémité** [1 p. 419].

![Cotes de débit du capot A195, coupes A-A et B-B](/assets/procedures/moe-76-advanced/aluclip-zero-debit-capot-a195.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 419, registre 2.6.5, p. 11, version janvier 2019)

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
trapézoïdaux », p. 302 et 303 du PDF (pages imprimées 1 et 2), version juillet 2017 ; registre 2.6.2 « AluClip,
cotes de débit », p. 345 à 356 du PDF (pages imprimées 41 à 52), versions décembre 2016 et mars 2021 ; registre 2.6.3
« AluClip Pro, Mise en oeuvre », p. 404 et 405 du PDF (pages imprimées 10 et 11), version décembre 2016 ;
registre 2.6.5 « AluClip Zero, Mise en oeuvre », p. 418 et 419 du PDF (pages imprimées 10 et 11),
version janvier 2019

# Voir aussi

- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md)
- [Meneaux PERFORM76](/profiles/perform76-meneaux.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
