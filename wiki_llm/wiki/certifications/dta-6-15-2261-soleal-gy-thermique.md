---
type: Certification
title: "Tableaux Ufi, Ψg, Uw, Sw1, Sw2, Sws et TLw du coulissant SOLEAL GY (DTA 6/15-2261, V1)"
description: "Tableaux de l'édition V1 du DTA 6/15-2261 du coulissant SOLEAL GY : Ufi des montants et traverses, Ψg (aluminium, TGI Spacer, Swisspacer V, WE EN 10077) par Ug, exemple de Uw, facteurs solaires Sw1, Sw2, Sws et facteurs lumineux TLw."
tags: [dta, cstb, soleal-gy, coulissant, technal, thermique, coefficient, aluminium, rupture-pont-thermique, vitrage]
gamme: LUMINE
systeme: SOLEAL GY
fournisseur: TECHNAL
usage: chiffrage
famille: thermique
status: draft
sources:
  - resource: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf
    id: technal-dta-soleal-gy-6-15-2261-v1
    title: DTA CSTB n° 6/15-2261_V1 SOLEAL GY
    last_modified: 2019-07-12
source_pages:
  - resource: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf
    pages: 6-7
generated:
  by: process:claude-code
  at: 2026-10-09T07:52:56Z
stale_after: 2024-07-31
---
# Tableaux thermiques du coulissant SOLEAL GY, édition V1

Le coulissant SOLÉAL55 de PROFERM est fabriqué sur le système TECHNAL SOLEAL GY (voir [TECHNAL](/fournisseurs/technal.md)). Les tableaux de cette section sont ceux de l'édition V1 du DTA 6/15-2261, publiée le 12 juillet 2019 (voir [DTA CSTB 6/15-2261_V1 SOLEAL GY, édition de 2019 : avis, domaine d'emploi, prescriptions](/certifications/dta-6-15-2261-soleal-gy-v1-avis.md)) ; l'édition V3 n'en reprend pas les tableaux et renvoie au rapport d'étude thermique RE CSTB n° DBV-24-31295 (voir [Dossier technique du DTA 6/15-2261 SOLEAL GY : conception, pose, fabrication, essais](/certifications/dta-6-15-2261-soleal-gy-dossier-technique.md)). Les coefficients sont définis au [glossaire](/reference/glossaire.md) : Uw (coefficient de transmission thermique de la fenêtre, en W/m²K), Sw (facteur solaire de la fenêtre), TLw (transmission lumineuse de la fenêtre). Les formules sont sur [Formules de calcul thermique du coulissant SOLEAL GY (DTA 6/15-2261, V1)](/certifications/dta-6-15-2261-soleal-gy-thermique-formules.md). Les cas « non prévus par le système » sont des cases grisées dans les tableaux de l'édition V1 [1 p. 6, 7].

# Tableaux de valeurs

## Tableau 1, valeurs de Ufi

Valeurs de Ufi (coefficient surfacique du montant ou de la traverse, en W/(m².K)) du coulissant SOLEAL GY, relevées sur la page 6 du DTA 6/15-2261_V1. Une ligne par nœud ; les cas en triple vitrage sont des cases grisées « Cas non prévus par le système ». Les valeurs des nœuds montants latéraux, traverse haute et traverse basse sont calculées en faisant la moyenne des Ufi côté ouvrant de service et côté semi fixe ; les valeurs sont uniquement valables pour le calcul du Uw sur un coulissant à 2 vantaux. Le renvoi (\*) des trois premières valeurs n'est défini nulle part dans la planche : **VER-174**.

| Position | Dormant | Ouvrant | Largeur de l'élément (m) | Ufi élément, double vitrage (W/(m².K)) |
| --- | --- | --- | --- | --- |
| Montants latéraux | TGY1113+T821000+T431024 | TGY1202 | 0,103 | 3,25 -2,95(\*) |
| Traverse haute | TGY1113+T821000+T431024 | T141015 | 0,103 | 3,2 – 3,15(\*) |
| Traverse basse | TGY1114+T431025+T431024 | T141015 | 0,103 | 3,5 – 3,35(\*) |
| Montants centraux | - | TGY1207+TGY1207 | 0,033 | 4,9 |
| Montants centraux | - | TGY1207+TGY1208 | 0,033 | 5,3 |

(schéma: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf, p. 6)

## Tableaux 2, 2bis et 2ter, valeurs de Ψg

Valeurs de Ψg en W/(m.K) selon le type d'intercalaire du vitrage isolant et selon Ug en W/(m².K), relevées sur la page 6 du DTA 6/15-2261_V1. « WE » est la mention imprimée des intercalaires « WE selon EN 10077 » ; une ligne par intercalaire et par profilé.

Tableau 2, valeurs de Ψg pour les montants latéraux :

| Type d'intercalaire | Profilés | Ψg pour Ug = 1,1 | Ψg pour Ug = 1,2 | Ψg pour Ug = 1,4 | Ψg pour Ug = 1,6 | Ψg pour Ug = 1,8 | Ψg pour Ug = 2,0 | Ψg pour Ug = 2,6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ψg (aluminium) | TGY1202 | 0,095 | 0,093 | 0,089 | 0,085 | 0,081 | 0,077 | 0,065 |
| Ψg (WE selon EN 10077) | TGY1202 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 |
| Ψg (TGI Spacer) | TGY1202 | 0,049 | 0,048 | 0,045 | 0,043 | 0,040 | 0,038 | 0,030 |
| Ψg (Swisspacer V) | TGY1202 | 0,037 | 0,036 | 0,034 | 0,032 | 0,030 | 0,027 | 0,021 |

Tableau 2bis, valeurs de Ψg pour les traverses hautes et basses :

| Type d'intercalaire | Profilés | Ψg pour Ug = 1,1 | Ψg pour Ug = 1,2 | Ψg pour Ug = 1,4 | Ψg pour Ug = 1,6 | Ψg pour Ug = 1,8 | Ψg pour Ug = 2,0 | Ψg pour Ug = 2,6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ψg (aluminium) | T141015 | 0,088 | 0,086 | 0,082 | 0,078 | 0,074 | 0,070 | 0,058 |
| Ψg (WE selon EN 10077) | T141015 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 |
| Ψg (TGI Spacer) | T141015 | 0,047 | 0,046 | 0,043 | 0,040 | 0,037 | 0,035 | 0,026 |
| Ψg (Swisspacer V) | T141015 | 0,036 | 0,034 | 0,031 | 0,029 | 0,026 | 0,023 | 0,014 |

Tableau 2ter, valeurs de Ψg pour les montants centraux :

| Type d'intercalaire | Profilés | Ψg pour Ug = 1,1 | Ψg pour Ug = 1,2 | Ψg pour Ug = 1,4 | Ψg pour Ug = 1,6 | Ψg pour Ug = 1,8 | Ψg pour Ug = 2,0 | Ψg pour Ug = 2,6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ψg (aluminium) | TGY1207+TGY1207 | 0,073 | 0,071 | 0,067 | 0,063 | 0,059 | 0,055 | 0,043 |
| Ψg (aluminium) | TGY1207+TGY1208 | 0,079 | 0,077 | 0,073 | 0,069 | 0,065 | 0,061 | 0,049 |
| Ψg (WE selon EN 10077) | TGY1207+TGY1207 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 |
| Ψg (WE selon EN 10077) | TGY1207+TGY1208 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 | 0,080 |
| Ψg (TGI Spacer) | TGY1207+TGY1207 | 0,046 | 0,044 | 0,041 | 0,039 | 0,036 | 0,033 | 0,025 |
| Ψg (TGI Spacer) | TGY1207+TGY1208 | 0,049 | 0,048 | 0,045 | 0,042 | 0,039 | 0,036 | 0,027 |
| Ψg (Swisspacer V) | TGY1207+TGY1207 | 0,041 | 0,040 | 0,037 | 0,035 | 0,032 | 0,030 | 0,022 |
| Ψg (Swisspacer V) | TGY1207+TGY1208 | 0,044 | 0,043 | 0,040 | 0,038 | 0,035 | 0,033 | 0,025 |

(schéma: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf, p. 6)

## Tableau 3, exemple de coefficients Uw

Exemple de coefficients Uw pour un vitrage ayant un Ug de 1,1 W/m²K et pour le dormant réf. TGY1113 + TGY1114, relevé sur la page 6 du DTA 6/15-2261_V1. (\*) calcul effectué selon la surface équivalente à celle obtenue avec les dimensions maximales dans la norme NF EN 14351.1. (\*\*) ces valeurs ne sont valables que si le sertissage est réalisé sur des demi-coquilles brutes. Les cases « - » sont grisées « Cas non prévus par le système ».

<table>
<thead>
<tr><th rowspan="3">Type menuiserie</th><th rowspan="3">Réf. profilés ouvrants</th><th rowspan="3">Uf (W/(m².K))</th><th colspan="4">Coefficient de la fenêtre nue Uw (W/(m².K))</th></tr>
<tr><th colspan="4">Intercalaire du vitrage isolant</th></tr>
<tr><th>Alu</th><th>WE EN 10077</th><th>WE TGI Spacer</th><th>WE Swisspacer V</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Fenêtre coulissante 2 vantaux 1,48 x 1,53 m (H x L) (S&lt;2.3 m²)</td><td>TGY1207 + TGY1207</td><td>3,4 – 3,2 (\*\*)</td><td>2,0</td><td>2,0 – 1,9(\*\*)</td><td>1,9– 1,8 (\*\*)</td><td>1,9 – 1,8(\*\*)</td></tr>
<tr><td>TGY1207 + TGY1208</td><td>3,4– 3,3 (\*\*)</td><td>2,0</td><td>2,0 – 1,9(\*\*)</td><td>1,9</td><td>1,9 – 1,8(\*\*)</td></tr>
<tr><td rowspan="2">Porte-fenêtre coulissante 2 vantaux 2,18 x 2,35 m* (H x L) (S&gt;2.3 m²)</td><td>TGY1207 + TGY1207</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>TGY1207 + TGY1208</td><td>3,4– 3,3 (\*\*)</td><td>1,7</td><td>1,7</td><td>1,7 – 1,6(\*)</td><td>1,6</td></tr>
</tbody>
</table>

(schéma: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf, p. 6)

Ce système de fenêtre ne permet pas, dans tous les cas, d'être mis en œuvre dans les bâtiments relevant de la RT existant par élément car le coefficient de transmission thermique des fenêtres Uw doit être inférieur ou égal à 1,9 W/m²K (arrêté du 22 mars 2017) [1 p. 6].

## Tableaux 4a, 4b, 4c et 4d, facteurs solaires et lumineux

Facteurs solaires et lumineux des fenêtres de dimensions courantes, relevés sur la page 7 du DTA 6/15-2261_V1. Une ligne par valeur de facteur du vitrage ; la menuiserie, ses références et ses surfaces sont répétées sur chaque ligne ; (C) est la condition de consommation, (E) les conditions d'été ou de confort.

Tableau 4a, facteurs solaires Sw1 (C) et Sw1 (E) pour les fenêtres sans protection mobile ni paroi opaque et de dimensions courantes :

| Menuiserie | Réf dormant | Réf ouvrant | σ | Af (m²) | Ag (m²) | Uf menuiserie (W/(m².K)) | Sg1, facteur solaire du vitrage | Sw1 pour la condition de consommation (C) | Sw1 pour les conditions d'été ou de confort (E) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fenêtre coulissante 2 vantaux 1,48 m × 1,53 m | TGY1113+TGY1114 | TGY1207+TGY1207 | 0,726 | 0,62 | 1,64 | 3,4 | 0,40 | 0,29 | 0,29 |
| Fenêtre coulissante 2 vantaux 1,48 m × 1,53 m | TGY1113+TGY1114 | TGY1207+TGY1207 | 0,726 | 0,62 | 1,64 | 3,4 | 0,50 | 0,36 | 0,36 |
| Fenêtre coulissante 2 vantaux 1,48 m × 1,53 m | TGY1113+TGY1114 | TGY1207+TGY1207 | 0,726 | 0,62 | 1,64 | 3,4 | 0,60 | 0,44 | 0,44 |
| Porte-fenêtre coulissante 2 vantaux 2,18 m × 2,35 m | TGY1113+TGY1114 | TGY1207+TGY1208 | 0,813 | 0,96 | 4,16 | 3,4 | 0,40 | 0,33 | 0,33 |
| Porte-fenêtre coulissante 2 vantaux 2,18 m × 2,35 m | TGY1113+TGY1114 | TGY1207+TGY1208 | 0,813 | 0,96 | 4,16 | 3,4 | 0,50 | 0,41 | 0,41 |
| Porte-fenêtre coulissante 2 vantaux 2,18 m × 2,35 m | TGY1113+TGY1114 | TGY1207+TGY1208 | 0,813 | 0,96 | 4,16 | 3,4 | 0,60 | 0,49 | 0,49 |

Tableau 4b, facteurs solaires Sw2 (C) et Sw2 (E) pour les fenêtres sans protection mobile ni paroi opaque et de dimensions courantes, par valeur forfaitaire de αf (fenêtre) :

| Menuiserie | Réf dormant | Réf ouvrant | σ | Af (m²) | Ag (m²) | Uf menuiserie (W/(m².K)) | Sg2 (C), facteur solaire du vitrage | Sw2 (C), αf = 0,4 | Sw2 (C), αf = 0,6 | Sw2 (C), αf = 0,8 | Sw2 (C), αf = 1 | Sg2 (E), facteur solaire du vitrage | Sw2 (E), αf = 0,4 | Sw2 (E), αf = 0,6 | Sw2 (E), αf = 0,8 | Sw2 (E), αf = 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fenêtre coulissante 2 vantaux 1,48 m × 1,53 m | TGY1113+TGY1114 | TGY1207+TGY1207 | 0,726 | 0,62 | 1,64 | 3,4 | 0,02 | 0,03 | 0,04 | 0,04 | 0,05 | 0,02 | 0,03 | 0,04 | 0,04 | 0,05 |
| Fenêtre coulissante 2 vantaux 1,48 m × 1,53 m | TGY1113+TGY1114 | TGY1207+TGY1207 | 0,726 | 0,62 | 1,64 | 3,4 | 0,05 | 0,05 | 0,06 | 0,07 | 0,07 | 0,05 | 0,05 | 0,06 | 0,07 | 0,07 |
| Fenêtre coulissante 2 vantaux 1,48 m × 1,53 m | TGY1113+TGY1114 | TGY1207+TGY1207 | 0,726 | 0,62 | 1,64 | 3,4 | 0,08 | 0,07 | 0,08 | 0,09 | 0,10 | 0,08 | 0,07 | 0,08 | 0,09 | 0,10 |
| Porte-fenêtre coulissante 2 vantaux 2,18 m × 2,35 m | TGY1113+TGY1114 | TGY1207+TGY1208 | 0,813 | 0,96 | 4,16 | 3,4 | 0,02 | 0,03 | 0,03 | 0,04 | 0,04 | 0,02 | 0,03 | 0,03 | 0,04 | 0,04 |
| Porte-fenêtre coulissante 2 vantaux 2,18 m × 2,35 m | TGY1113+TGY1114 | TGY1207+TGY1208 | 0,813 | 0,96 | 4,16 | 3,4 | 0,05 | 0,05 | 0,06 | 0,06 | 0,07 | 0,05 | 0,05 | 0,06 | 0,06 | 0,07 |
| Porte-fenêtre coulissante 2 vantaux 2,18 m × 2,35 m | TGY1113+TGY1114 | TGY1207+TGY1208 | 0,813 | 0,96 | 4,16 | 3,4 | 0,08 | 0,08 | 0,08 | 0,08 | 0,09 | 0,08 | 0,08 | 0,08 | 0,08 | 0,09 |

Tableau 4c, facteurs solaires Sws (E) pour les fenêtres avec protection mobile extérieure opaque déployée et de dimensions courantes :

| Coloris du tablier opaque | Sws (E) |
| --- | --- |
| L\*<82 | 0,05 |
| L\*≥82 | 0,10 |

Tableau 4d, facteurs de transmission lumineuse TLw et TLws pour les fenêtres de dimensions courantes :

| Menuiserie | Réf dormant | Réf ouvrant | σ | Af (m²) | Ag (m²) | Uf menuiserie (W/(m².K)) | TLg, facteur de transmission lumineuse du vitrage | TLw | TLws |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fenêtre coulissante 2 vantaux 1,48 m × 1,53 m | TGY1113+TGY1114 | TGY1207+TGY1207 | 0,726 | 0,62 | 1,64 | 3,4 | 0,70 | 0,51 | 0 |
| Fenêtre coulissante 2 vantaux 1,48 m × 1,53 m | TGY1113+TGY1114 | TGY1207+TGY1207 | 0,726 | 0,62 | 1,64 | 3,4 | 0,80 | 0,58 | 0 |
| Porte-fenêtre coulissante 2 vantaux 2,18 m × 2,35 m | TGY1113+TGY1114 | TGY1207+TGY1208 | 0,813 | 0,96 | 4,16 | 3,4 | 0,70 | 0,57 | 0 |
| Porte-fenêtre coulissante 2 vantaux 2,18 m × 2,35 m | TGY1113+TGY1114 | TGY1207+TGY1208 | 0,813 | 0,96 | 4,16 | 3,4 | 0,80 | 0,65 | 0 |

(schéma: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf, p. 7)

# Citations

[1] [DTA CSTB n° 6/15-2261_V1 SOLEAL GY, publié le 12/07/2019](raw/SOLEAL-GY55-DTA-AG152261_V1.pdf), p. 6 et 7

# Voir aussi

- [Formules de calcul thermique du coulissant SOLEAL GY (DTA 6/15-2261, V1)](/certifications/dta-6-15-2261-soleal-gy-thermique-formules.md)
- [DTA CSTB 6/15-2261_V1 SOLEAL GY, édition de 2019 : avis, domaine d'emploi, prescriptions](/certifications/dta-6-15-2261-soleal-gy-v1-avis.md)
- [Dossier technique du DTA 6/15-2261 SOLEAL GY, édition V1](/certifications/dta-6-15-2261-soleal-gy-v1-dossier-technique.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
- [TECHNAL](/fournisseurs/technal.md)
