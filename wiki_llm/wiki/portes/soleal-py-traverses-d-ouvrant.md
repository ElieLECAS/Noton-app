---
type: Porte d'entrée
title: "Traverses d'ouvrant T215202 à T215211 de la porte SOLEAL PY 55 : débit et remplissage"
description: Débit Lv des traverses d'ouvrant T215202, T215204, T215205, T215207, T215208 et T215211 par type de porte (SA, DA, va-et-vient, tube APD), remplissage et débit des parcloses Hv1 / Hv2 de la porte à traverse d'ouvrant.
tags: [technal, soleal-py, lumine55, porte, traverse, ouvrant, debit, parclose, nomenclature]
gamme: LUMINE
systeme: SOLEAL PY
fournisseur: TECHNAL
usage: atelier
famille: traverses
status: stable
sources:
  - resource: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf
    title: SOLEAL PY 55, Conception, réf. 4944.006
    last_modified: 2018-09-01
source_pages:
  - resource: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf
    pages: 93-94
generated:
  by: process:claude-code
  at: 2026-10-10T12:20:10Z
---

La traverse d'ouvrant est la barre horizontale qui coupe un ouvrant de porte SOLEAL PY 55 de [TECHNAL](/fournisseurs/technal.md) (porte de la gamme LUMINE de PROFERM) en deux zones de remplissage, l'une au-dessus de l'autre ; six profilés la réalisent : T215202, T215204, T215205, T215207, T215208 et T215211, rangés sous le titre « Traverse d'ouvrant » de la rubrique « Options » [1 p. 93]. Lv est la largeur du vantail, Hv sa hauteur, SA la porte simple action, DA la porte double action, VV la porte va-et-vient, APD l'anti-pince-doigts (voir le [glossaire](/reference/glossaire-technal.md) et le [glossaire général](/reference/glossaire.md)).

# Débit de la traverse d'ouvrant selon le type de porte

Le débit est la longueur de coupe de la traverse en fonction de la largeur de vantail Lv : « Lv - 150 » se lit « Lv moins 150 mm ». Le tableau est le même pour chacun des six profilés T215202, T215204, T215205, T215207, T215208 et T215211 ; la coupe d'extrémité est 90°/90° pour toutes les lignes ; la quantité est exprimée pour une porte à un vantail (1 vtl) et pour une porte à deux vantaux (2 vtx) ; une case de quantité vide dans la source est notée « - » [1 p. 93].

| Type d'application | Quantité par porte 1 vtl | Quantité par porte 2 vtx | Coupe | Débit (mm) |
| --- | --- | --- | --- | --- |
| Porte SA 1 et 2 vantaux | 1 | 2 | 90°/90° | Lv - 150 |
| Porte SA 2 vtx indépendants (APD ou pas) | - | 2 | 90°/90° | Lv - 137,5 |
| Porte VV 1 et 2 vantaux (APD ou pas) | 1 | 2 | 90°/90° | Lv - 125 |
| Porte SA Tube APD Niveau 1 | 1 | - | 90°/90° | Lv - 136,5 |
| Porte SA Tube APD Niveau 2 | 1 | - | 90°/90° | Lv - 122 |
| Porte DA Tube APD Niveau 1 et 2 | 1 | - | 90°/90° | Lv - 122 |
| Porte SA et DA Tube APD Niveau 1 et 2 | - | 2 | 90°/90° | Lv - 122 |

(schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 93)

Les lignes « Porte SA Tube APD Niveau 2 » et « Porte DA Tube APD Niveau 1 et 2 » partagent dans la source une même case de quantité (1) et un même débit (Lv - 122). Les niveaux 1 et 2 de l'anti-pince-doigts sont décrits sur [Anti-pince-doigts et principes d'assemblage](/portes/soleal-py-anti-pince-doigts-et-principes-d-assemblage.md).

# Schémas de principe : coupes horizontales par type de porte

La figure donne, pour chaque type de porte, la coupe horizontale de l'ouvrant et du dormant avec la cote Lv entre les deux montants d'ouvrant ; les portes à deux vantaux indépendants, les doubles actions à deux vantaux et les portes tube à deux vantaux portent en plus la cote 23 entre les deux vantaux, et la porte tube à un vantail porte la cote 23 entre le dormant et le montant d'ouvrant [1 p. 93].

![Traverse d'ouvrant : coupes horizontales de principe des portes SA, DA et tube APD](/assets/portes/soleal-py-options/traverses-d-ouvrant-schemas-de-principe.png)

Les sept dessins sont, dans l'ordre : Porte SA 1 vantail (cote Lv), Porte SA 2 vantaux (Lv sur chaque vantail), Porte SA 2 vantaux indépendants (23 et Lv), Porte DA 1 vantail (Lv), Porte DA 2 vantaux (Lv, 23, Lv), Porte tube APD 1 vantail (23 et Lv), Porte tube APD 2 vantaux (Lv, 23, Lv).

(schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 93)

# Hauteurs Hv1 et Hv2 de part et d'autre de la traverse

La traverse sépare la hauteur du vantail Hv en deux hauteurs : Hv1 (zone basse) et Hv2 (zone haute), mesurées à partir de l'axe de la traverse, qui est cotée 31 mm de chaque côté de l'axe (50 mm entre les deux repères intérieurs). Les deux relations imprimées sont « Hv1 sol = Hv1 + jeu » et « Hv1 + Hv2 = Hv ». Le repère de la coupe est T215205 ; Hv1 sol est la hauteur de Hv1 prise depuis le sol [1 p. 94].

![Traverse d'ouvrant T215205 : hauteurs Hv1 et Hv2](/assets/portes/soleal-py-options/traverses-d-ouvrant-hv1-hv2.png)

(schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 94)

# Remplissage et débit des parcloses par traverse

Le tableau donne, pour chaque traverse d'ouvrant, la dimension de chacun des deux remplissages (vitrage ou panneau) et la longueur des parcloses qui les maintiennent, en millimètres : « Hv1-101 » se lit « Hv1 moins 101 mm ». Chaque profilé occupe deux lignes, l'une pour la zone Hv1, l'autre pour la zone Hv2 ; la porte SA et la porte DA sont chacune déclinées en ouvrant périphérique et en plinthe [1 p. 94].

<table>
<thead>
<tr><th rowspan="3">Profilé</th><th rowspan="3">Zone</th><th colspan="4">Porte SA</th><th colspan="4">Porte DA</th></tr>
<tr><th colspan="2">Ouvrant périphérique</th><th colspan="2">Plinthe</th><th colspan="2">Ouvrant périphérique</th><th colspan="2">Plinthe</th></tr>
<tr><th>Remplissage</th><th>Parclose</th><th>Remplissage</th><th>Parclose</th><th>Remplissage</th><th>Parclose</th><th>Remplissage</th><th>Parclose</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">T215202</td><td>Hv1</td><td>Hv1-101</td><td>Hv1-133</td><td>Hv1-121</td><td>Hv1-153</td><td>Hv1-88</td><td>Hv1-120</td><td>Hv1-121</td><td>Hv1-153</td></tr>
<tr><td>Hv2</td><td>Hv2-101</td><td>Hv2-133</td><td>Hv2-101</td><td>Hv2-133</td><td>Hv2-88</td><td>Hv2-120</td><td>Hv2-88</td><td>Hv2-120</td></tr>
<tr><td rowspan="2">T215204</td><td>Hv1</td><td>Hv1-107</td><td>Hv1-139</td><td>Hv1-127</td><td>Hv1-159</td><td>Hv1-95</td><td>Hv1-127</td><td>Hv1-127</td><td>Hv1-159</td></tr>
<tr><td>Hv2</td><td>Hv2-107</td><td>Hv2-139</td><td>Hv2-107</td><td>Hv2-139</td><td>Hv2-95</td><td>Hv2-127</td><td>Hv2-100</td><td>Hv2-132</td></tr>
<tr><td rowspan="2">T215205</td><td>Hv1</td><td>Hv1-112</td><td>Hv1-144</td><td>Hv1-132</td><td>Hv1-164</td><td>Hv1-100</td><td>Hv1-132</td><td>Hv1-132</td><td>Hv1-164</td></tr>
<tr><td>Hv2</td><td>Hv2-112</td><td>Hv2-144</td><td>Hv2-112</td><td>Hv2-144</td><td>Hv2-100</td><td>Hv2-132</td><td>Hv2-100</td><td>Hv2-132</td></tr>
<tr><td rowspan="2">T215207</td><td>Hv1</td><td>Hv1-122</td><td>Hv1-154</td><td>Hv1-142</td><td>Hv1-174</td><td>Hv1-110</td><td>Hv1-142</td><td>Hv1-142</td><td>Hv1-174</td></tr>
<tr><td>Hv2</td><td>Hv2-122</td><td>Hv2-154</td><td>Hv2-122</td><td>Hv2-154</td><td>Hv2-110</td><td>Hv2-142</td><td>Hv2-110</td><td>Hv2-142</td></tr>
<tr><td rowspan="2">T215208</td><td>Hv1</td><td>Hv1-127</td><td>Hv1-159</td><td>Hv1-147</td><td>Hv1-179</td><td>Hv1-115</td><td>Hv1-147</td><td>Hv1-147</td><td>Hv1-179</td></tr>
<tr><td>Hv2</td><td>Hv2-127</td><td>Hv2-159</td><td>Hv2-127</td><td>Hv2-159</td><td>Hv2-115</td><td>Hv2-147</td><td>Hv2-115</td><td>Hv2-147</td></tr>
<tr><td rowspan="2">T215211</td><td>Hv1</td><td>Hv1-142</td><td>Hv1-174</td><td>Hv1-162</td><td>Hv1-194</td><td>Hv1-130</td><td>Hv1-162</td><td>Hv1-162</td><td>Hv1-194</td></tr>
<tr><td>Hv2</td><td>Hv2-142</td><td>Hv2-174</td><td>Hv2-142</td><td>Hv2-174</td><td>Hv2-130</td><td>Hv2-162</td><td>Hv2-130</td><td>Hv2-162</td></tr>
</tbody>
</table>

(schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 94)

Les désignations des six profilés ne sont pas imprimées sur ces deux planches : leurs coupes et leurs cotes sont sur la page des profilés de traverses lorsque celle-ci existe. Les débits de parclose des portes à un vantail sans traverse sont sur les pages de nomenclature, par exemple [Porte va-et-vient 1 et 2 vantaux, ouvrant périphérique, seuil plat](/portes/soleal-py-va-et-vient-ouvrant-peripherique-seuil-plat.md).

# Citations

[1] [SOLEAL PY 55, Conception, réf. 4944.006](raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf), p. 93 et 94
