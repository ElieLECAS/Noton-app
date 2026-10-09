---
type: Certification
title: "Coefficients Uc et Up et compositions vinyliques des coffres Chrono (AT 6/16-2339_V2)"
description: "Expressions du coefficient surfacique moyen Uc du coffre et Up de la paroi en fonction de la longueur Lc pour les coffres Chrono ONE 230 et 200 (tunnel, ITI 100 à 160 mm, ITE 140 mm, avec ou sans renfort et isolant de joue) et compositions vinyliques certifiées Inoutic, Alphacan, Benvic."
tags: [certification, avis-technique, cstb, chrono-one, thermique, coffre, abaque, soprofen]
systeme: Chrono One
fournisseur: SOPROFEN
usage: chiffrage
famille: coffres-chrono
status: stable
sources:
  - resource: raw/moustiquaires/exports/135.pdf
    id: soprofen-at-cstb-coffres-chrono
    title: Avis Technique n° 6/16-2339_V2, procédé Chrono C, Chrono VX2, Chrono VRX2, Chrono ONE
    last_modified: 2022-02-22
source_pages:
  - resource: raw/moustiquaires/exports/135.pdf
    pages: 8-9
generated:
  by: process:claude-code
  at: 2026-10-09T02:30:00Z
stale_after: 2027-01-31
---

# Uc et Up des coffres Chrono ONE

Le coefficient surfacique moyen du coffre Uc (en W/m².K) mesure les déperditions thermiques du
coffre seul ; le coefficient surfacique moyen de la paroi intégrant le coffre Up (en W/m².K)
mesure celles du mur qui le porte, coffre compris (voir le [glossaire](/reference/glossaire.md)). Le
tableau 1 de l'Avis Technique
[6/16-2339_V2](/certifications/at-cstb-6-16-2339-coffres-chrono.md) de [SOPROFEN](/fournisseurs/soprofen.md)
les donne pour le Chrono ONE sous la forme a + b / Lc, où **Lc est la longueur du coffre en
mètre**, la surface de référence étant celle de la projection du coffre sur un plan vertical. Chaque
valeur est donc à calculer pour la longueur du coffre posé [1 p. 8].

Pose : « Tunnel » est la pose du coffre en tableau, dans l'épaisseur de la baie (la valeur est un
Uc) ; « ITI 100 mm » à « ITI 160 mm » est la pose en applique intérieure derrière un doublage
(isolation thermique par l'intérieur) de 100 à 160 mm ; « ITE 140 mm » est la pose en tableau avec
isolation thermique par l'extérieur de 140 mm (la valeur est un Up) [1 p. 8].

Les notes du tableau [1 p. 8] :

- (1) Les calculs « avec renfort » sont faits avec le renfort de sous-face réf. MX20-2.
- (2) Les calculs sont faits avec la sous-face MX01 (pose tunnel, doublage de 100 et 120 mm) et avec
  la sous-face MX01R (doublage de 140 et 160 mm).
- (3) Les isolants linéaires considérés sont : l'isolant thermique, coquille réf. MX50-23,
  PSE 17 kg/m³, λutile = 0,031 W/(m.K) ; l'isolant thermique\*, coquille réf. M50-20 (écrite ainsi :
  **INC-322**), PSE 17 kg/m³, λutile = 0,031 W/(m.K) ; l'isolant thermoacoustique, coquille
  réf. MX50-23, PSE 17 kg/m³, plus plaque polymère réf. MX58/18 de 10 kg/m², λutile = 0,25 W/(m.K).
- (4) Les calculs avec isolants de joue sont faits avec les références MXJT50-23 et MXJT50-20,
  PSE 17 kg/m³, λutile = 0,031 W/(m.K).

## Chrono ONE 230

Uc (pose tunnel) et Up (autres poses) du coffre Chrono ONE 230, en W/(m².K), avec Lc en mètre. Une
ligne par combinaison. L'adaptateur est le profilé qui lie le coffre au dormant. Les deux dernières
lignes de la pose tunnel (adaptateur « / », isolant joue « Non ») n'ont pas de valeur : la case est
grisée sur le tableau [1 p. 8].

| Pose | Renfort | Adaptateur | Isolant linéaire | Isolant joue | Coefficient (W/m².K) |
| --- | --- | --- | --- | --- | --- |
| Tunnel | sans | aluminium MX10/35 | thermique | non | Uc = 1,16 + 0,554 / Lc |
| Tunnel | sans | aluminium MX10/35 | thermique | oui | Uc = 1,16 + 0,192 / Lc |
| Tunnel | avec | aluminium MX10/35 | thermique | non | Uc = 1,17 + 0,554 / Lc |
| Tunnel | avec | aluminium MX10/35 | thermique | oui | Uc = 1,17 + 0,192 / Lc |
| Tunnel | sans | aluminium MX10/35 | thermo-acoustique | non | Uc = 1,15 + 0,554 / Lc |
| Tunnel | sans | aluminium MX10/35 | thermo-acoustique | oui | Uc = 1,15 + 0,192 / Lc |
| Tunnel | avec | aluminium MX10/35 | thermo-acoustique | non | Uc = 1,16 + 0,554 / Lc |
| Tunnel | avec | aluminium MX10/35 | thermo-acoustique | oui | Uc = 1,16 + 0,192 / Lc |
| Tunnel | sans | / | thermique | non | - |
| Tunnel | avec | / | thermique | non | - |
| ITI 100 mm | sans | aluminium MX10/46 | thermique | oui | Up = 1,08 + 0,166 / Lc |
| ITI 100 mm | avec | aluminium MX10/46 | thermique | oui | Up = 1,09 + 0,166 / Lc |
| ITI 120 mm | sans | aluminium MX10/66 | thermique | oui | Up = 1,01 + 0,122 / Lc |
| ITI 120 mm | avec | aluminium MX10/66 | thermique | oui | Up = 1,03 + 0,122 / Lc |
| ITI 140 mm | sans | aluminium MX10/46 | thermique | oui | Up = 0,936 + 0,080 / Lc |
| ITI 140 mm | avec | aluminium MX10/46 | thermique | oui | Up = 0,955 + 0,080 / Lc |
| ITI 160 mm | sans | aluminium MX10/66 | thermique | oui | Up = 0,920 + 0,042 / Lc |
| ITI 160 mm | avec | aluminium MX10/66 | thermique | oui | Up = 0,928 + 0,042 / Lc |
| ITE 140 mm | sans | aluminium MX10/35 | thermique | non | Up = 0,958 + 0,554 / Lc |
| ITE 140 mm | sans | aluminium MX10/35 | thermique | oui | Up = 0,958 + 0,192 / Lc |
| ITE 140 mm | avec | aluminium MX10/35 | thermique | non | Up = 0,986 + 0,554 / Lc |
| ITE 140 mm | avec | aluminium MX10/35 | thermique | oui | Up = 0,986 + 0,192 / Lc |

(schéma: raw/moustiquaires/exports/135.pdf, p. 8)

## Chrono ONE 200

Même présentation pour le coffre Chrono ONE 200. Toutes les lignes sont avec isolant de joue « oui » et
isolant linéaire « thermique\* » (coquille M50-20) [1 p. 8].

| Pose | Renfort | Adaptateur | Coefficient (W/m².K) |
| --- | --- | --- | --- |
| ITI 100 mm | sans | aluminium MX10/40 | Up = 0,931 + 0,140 / Lc |
| ITI 120 mm | sans | aluminium MX10/55F | Up = 0,835 + 0,107 / Lc |
| ITI 140 mm | sans | aluminium MX10/40 | Up = 0,736 + 0,027 / Lc |
| ITI 160 mm | sans | aluminium MX10/55F | Up = 0,661 + 0,027 / Lc |
| Tunnel | sans | aluminium MX10/40 | Uc = 0,942 + 0,140 / Lc |
| ITI 120 mm | avec | aluminium MX10/55F | Up = 0,851 + 0,107 / Lc |
| ITI 140 mm | avec | aluminium MX10/40 | Up = 0,743 + 0,027 / Lc |
| ITI 160 mm | avec | aluminium MX10/55F | Up = 0,667 + 0,027 / Lc |

Le tableau ne donne pas de ligne « ITI 100 mm avec renfort » ni « tunnel avec renfort » pour le
Chrono ONE 200 ; il ne donne aucun coefficient pour les coffres Chrono C, Chrono VX2 et Chrono VX2
réno, ni pour le Chrono ONE 200 en ITE.

(schéma: raw/moustiquaires/exports/135.pdf, p. 8)

## Exigence en rénovation

Dans les bâtiments existants relevant de la RT existant, le coefficient Uc des coffres doit être
inférieur ou égal à 3 W/(m².K) ; dans ceux relevant de la RT élément par élément, inférieur ou
égal à 2,5 W/(m².K) [1 p. 8].

# Compositions vinyliques

Le tableau 2 liste les compositions vinyliques certifiées dont sont extrudés les profilés de coffre.
Une colonne par composition ; le code CSTB est celui de la composition. « Composition devant être
plaxée » signifie que la composition doit être recouverte d'un film décoratif [1 p. 9].

| Fabricant | Composition | Code CSTB | Coloris | Composition devant être plaxée |
| --- | --- | --- | --- | --- |
| INOUTIC | 2113/57 | 263 | blanc | - |
| INOUTIC | 2113.35 | 296 | beige | - |
| INOUTIC | 2113/33 (L2) | 294 | gris | - |
| ALPHACAN | 674 | 222 | blanc | - |
| ALPHACAN | 716 | 366.02 | blanc | - |
| ALPHACAN | 674/1015 | 308 | beige | - |
| ALPHACAN | 716/1015 | 427 | beige | - |
| ALPHACAN | 674/7035 | 339 | gris | - |
| ALPHACAN | 716/7035 | 426 | gris | - |
| ALPHACAN | 780 Marron G | 49px | marron | oui |
| ALPHACAN | 780 Marron | 52px | marron | oui |
| ALPHACAN | PCN216 | 4006 | / | / |
| BENVIC | ER 019/0900 | / | noir | / |

(schéma: raw/moustiquaires/exports/135.pdf, p. 9)

Les cases « Composition devant être plaxée » des quatre premiers groupes ne sont pas imprimées ; seules
les deux compositions « 780 Marron » portent « oui ».

# Citations

[1] Avis Technique n° 6/16-2339_V2, SOPROFEN — `raw/moustiquaires/exports/135.pdf`, p. 8 et 9
(le numéro de page imprimé est celui du PDF)

# Voir aussi

- [Avis Technique CSTB 6/16-2339_V2 des coffres Chrono : domaine d'emploi et prescriptions](/certifications/at-cstb-6-16-2339-coffres-chrono.md)
- [Dimensions et rigidité des coffres Chrono](/certifications/at-cstb-6-16-2339-dossier-technique.md)
- [SOPROFEN](/fournisseurs/soprofen.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
- [Avis Technique CSTB n° 6/16-2339_V2, coffres Chrono SOPROFEN (2022)](/sources/soprofen-at-cstb-coffres-chrono.md)
