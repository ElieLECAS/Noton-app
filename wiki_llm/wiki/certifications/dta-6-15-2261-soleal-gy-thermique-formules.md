---
type: Certification
title: "Formules de calcul thermique du coulissant SOLEAL GY (DTA 6/15-2261, V1)"
description: "Formules de l'édition V1 du DTA 6/15-2261 du coulissant SOLEAL GY : Uw, Uf, Ψg, Ujn, Uwf (avec ΔR = 0,15 et 0,19), facteurs solaires Sw, Sws, Sw1, Sw2, Sf, Sp, facteur de transmission lumineuse TLw, Ks, K, avec le tableau de référence de Uwf et de Ujn."
tags: [dta, cstb, soleal-gy, coulissant, technal, thermique, coefficient, formules, aluminium, rupture-pont-thermique]
gamme: LUMINE
systeme: SOLEAL GY
fournisseur: TECHNAL
usage: chiffrage
famille: thermique-formules
status: stable
sources:
  - resource: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf
    id: technal-dta-soleal-gy-6-15-2261-v1
    title: DTA CSTB n° 6/15-2261_V1 SOLEAL GY
    last_modified: 2019-07-12
source_pages:
  - resource: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf
    pages: 3-4
generated:
  by: process:claude-code
  at: 2026-10-09T07:52:56Z
stale_after: 2024-07-31
---
# Formules de calcul thermique du coulissant SOLEAL GY, édition V1

Le coulissant SOLÉAL55 de PROFERM est fabriqué sur le système TECHNAL SOLEAL GY (voir [TECHNAL](/fournisseurs/technal.md)). Les formules de cette section sont celles de l'édition V1 du DTA 6/15-2261, publiée le 12 juillet 2019 (voir [DTA CSTB 6/15-2261_V1 SOLEAL GY, édition de 2019 : avis, domaine d'emploi, prescriptions](/certifications/dta-6-15-2261-soleal-gy-v1-avis.md)). Les coefficients sont définis au [glossaire](/reference/glossaire.md) : Uw (coefficient de transmission thermique de la fenêtre, en W/m²K), Sw (facteur solaire de la fenêtre), TLw (transmission lumineuse de la fenêtre). Les tableaux de valeurs sont sur [Tableaux Ufi, Ψg, Uw, Sw1, Sw2, Sws et TLw du coulissant SOLEAL GY (DTA 6/15-2261, V1)](/certifications/dta-6-15-2261-soleal-gy-thermique.md) [1 p. 3, 4].

# Éléments de calcul thermique lié au produit

Le coefficient de transmission thermique Uw peut être calculé selon la formule suivante [1 p. 3] :

`Uw = (Ug·Ag + Uf·Af + Ψg·Ig) / (Ag + Af)`

où :

- Uw est le coefficient de transmission surfacique de fenêtre nue en W/(m².K).
- Ug est le coefficient surfacique en partie centrale du vitrage en W/(m².K). Sa valeur est déterminée selon les règles Th-U.
- Uf est le coefficient surfacique moyen de la menuiserie en W/(m².K), calculé selon la formule `Uf = Σ(Ufi·Afi) / Af`, où Ufi est le coefficient surfacique du montant ou traverse numéro « i » et Afi son aire projetée correspondante. La largeur des montants en partie courante est supposée se prolonger sur toute la hauteur de la fenêtre.
- Ag est la plus petite des aires visibles du vitrage, vues des deux côtés de la fenêtre, en m². On ne tient pas compte des débordements des joints.
- Af est la plus grande surface projetée de la menuiserie prise sans recouvrement, incluant la surface de la pièce d'appui éventuelle, vue des deux côtés de la fenêtre, en m².
- Ig est la plus grande somme des périmètres visibles du vitrage, vus des deux côtés de la fenêtre, en m.
- Ψg est le coefficient linéique dû à l'effet thermique combiné de l'intercalaire du vitrage et du profilé, en W/(m.K).

Des valeurs pour ces différents éléments sont données dans les tableaux en fin de première partie : Ufi : voir tableau 1. Ψg : voir tableaux 2 et 2bis (le tableau 2ter, valeurs de Ψg pour les montants centraux, existe aussi : **INC-387**). Uw : voir tableaux 3. Valeurs données à titre d'exemple pour des Ug de 1,1 W/(m².K) [1 p. 3].

Le coefficient de transmission thermique moyen Ujn peut être calculé selon la formule `Ujn = (Uw + Uwf) / 2` (1), où Uw est le coefficient de transmission surfacique de fenêtre nue en W/(m².K) et Uwf le coefficient de transmission surfacique de fenêtre avec fermeture en W/(m².K), calculé selon la formule `Uwf = 1 / (1/Uw + ΔR)` (2). ΔR est la résistance thermique additionnelle, en (m².K)/W, apportée par l'ensemble fermeture-lame d'air ventilée ; les valeurs de ΔR prises en compte sont 0,15 et 0,19 (m².K)/W. Les formules (1) et (2) permettent de déterminer les valeurs de référence Ujn et Uwf en fonction de Uw ; elles sont indiquées dans le tableau suivant [1 p. 3].

Valeurs de référence de Uwf et de Ujn en W/(m².K) en fonction de Uw, pour ΔR = 0,15 et ΔR = 0,19 (m².K)/W, relevées sur la page 3 du DTA 6/15-2261_V1 (une ligne par valeur de Uw).

<table>
<thead>
<tr><th rowspan="2">Uw (W/(m².K))</th><th colspan="2">Uwf (W/(m².K))</th><th colspan="2">Ujn (W/(m².K))</th></tr>
<tr><th>ΔR = 0,15</th><th>ΔR = 0,19</th><th>ΔR = 0,15</th><th>ΔR = 0,19</th></tr>
</thead>
<tbody>
<tr><td>0,8</td><td>0,7</td><td>0,7</td><td>0,8</td><td>0,7</td></tr>
<tr><td>0,9</td><td>0,8</td><td>0,8</td><td>0,8</td><td>0,8</td></tr>
<tr><td>1,0</td><td>0,9</td><td>0,8</td><td>0,9</td><td>0,9</td></tr>
<tr><td>1,1</td><td>0,9</td><td>0,9</td><td>1,0</td><td>1,0</td></tr>
<tr><td>1,2</td><td>1,0</td><td>1,0</td><td>1,1</td><td>1,1</td></tr>
<tr><td>1,3</td><td>1,1</td><td>1,0</td><td>1,2</td><td>1,2</td></tr>
<tr><td>1,4</td><td>1,2</td><td>1,1</td><td>1,3</td><td>1,3</td></tr>
<tr><td>1,5</td><td>1,2</td><td>1,2</td><td>1,4</td><td>1,3</td></tr>
<tr><td>1,6</td><td>1,3</td><td>1,2</td><td>1,4</td><td>1,4</td></tr>
<tr><td>1,8</td><td>1,4</td><td>1,3</td><td>1,6</td><td>1,6</td></tr>
<tr><td>2,0</td><td>1,5</td><td>1,4</td><td>1,8</td><td>1,7</td></tr>
<tr><td>2,3</td><td>1,7</td><td>1,6</td><td>2,0</td><td>2,0</td></tr>
<tr><td>2,6</td><td>1,9</td><td>1,7</td><td>2,2</td><td>2,2</td></tr>
</tbody>
</table>

(schéma: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf, p. 3)

# Éléments de calcul thermique de l'ouvrage

Les valeurs Uw à prendre en compte dans le calcul du Ubât doivent tenir compte de la mise en œuvre du produit. Pour le calcul du coefficient Ubât, il y aura lieu de prendre en compte les déperditions thermiques au droit des liaisons entre le dormant et le gros-œuvre. Ces déperditions sont représentées en particulier par le coefficient Ψ. Ψ est le coefficient de transmission linéique dû à l'effet thermique combiné du gros-œuvre et de la menuiserie, en W/(m.K). La valeur du coefficient Ψ est dépendante du mode de mise en œuvre de la menuiserie. Selon les règles Th-U 5/5 de 2005 « Ponts thermiques », la valeur Ψ peut varier de 0 à 0,35 W/(m.K), pour une construction neuve ou pour une pose en rénovation avec dépose totale. Pour une pose en rénovation avec conservation du dormant existant, il y aura lieu de déterminer la valeur Ψ [1 p. 3].

# Facteurs solaires et facteurs de transmission lumineuse

## Facteur solaire de la fenêtre

Le facteur solaire Sw ou Sws de la fenêtre est déterminé selon la norme XP P50-777, selon les formules suivantes [1 p. 3] :

- `Sw = Sw1 + Sw2 + Sw3` (sans protection mobile) ;
- `Sws = Sws1 + Sws2 + Sws3` (avec protection mobile déployée).

où :

- Sw1, Sws1 est la composante de transmission solaire directe : `Sw1 = Ag / (Ap + Af + Ag) · Sg1` et `Sws1 = Ag / (Ap + Af + Ag) · Sgs1`.
- Sw2, Sws2 est la composante de réémission thermique vers l'intérieur : `Sw2 = (Ap·Sp + Af·Sf + Ag·Sg2) / (Ap + Af + Ag)` et `Sws2 = (Ap·Sps + Af·Sfs + Ag·Sgs2) / (Ap + Af + Ag)`.
- Sw3, Sws3 est le facteur de ventilation : `Sw3 = 0` et `Sws3 = Ag / (Ap + Af + Ag) · Sgs3`.

et, avec les surfaces et facteurs suivants [1 p. 3, 4] :

- Ag est la surface de vitrage la plus petite vue des deux côtés, intérieur et extérieur (m²).
- Ap est la surface de paroi opaque la plus petite vue des deux côtés, intérieur et extérieur (m²).
- Af est la surface de la menuiserie la plus grande vue des deux côtés, intérieur et extérieur (m²).
- Sg1 est le facteur de transmission directe solaire du vitrage sans protection mobile (désigné par te dans les normes NF EN 13363-2 ou NF EN 410).
- Sgs1 est le facteur de transmission directe solaire du vitrage avec protection mobile (désigné par te dans les normes NF EN 13363-2 ou NF EN 410).
- Sg2 est le facteur de réémission thermique vers l'intérieur (désigné par qi dans les normes NF EN 13363-2 ou NF EN 410).
- Sgs2 est le facteur de réémission thermique vers l'intérieur (désigné par gth+gc dans la norme NF EN 13363-2).
- Sgs3 est le facteur de ventilation (désigné par gv dans la norme NF EN 13363-2) - dans le cas d'une protection mobile extérieure, Sgs3 = 0.
- Sf est le facteur de transmission solaire cadre, avec `Sf = αf·Uf / he`, où αf est le facteur d'absorption solaire du cadre (voir le tableau ci-dessous), Uf le coefficient de transmission thermique surfacique moyen du cadre, selon NF EN ISO 10077-2 (W/m².K), et he le coefficient d'échanges superficiels, pris égal à 25 W/(m².K).
- Sfs est le facteur de transmission solaire cadre avec protection mobile extérieure (voir §11.2.5 de la norme XP P50-777).
- Sp est le facteur de transmission solaire de la paroi opaque, avec `Sp = αp·Up / he`, où (texte imprimé) αf est le facteur d'absorption solaire de la paroi opaque (voir le tableau ci-dessous), Uf le coefficient de transmission thermique de la paroi opaque, selon NF EN ISO 6946 (W/m².K), et he le coefficient d'échanges superficiels, pris égal à 25 W/(m².K) (**INC-388**).
- Sps est le facteur de transmission solaire de la paroi opaque avec protection mobile extérieure (voir §11.2.6 de la norme XP P50-777).

Le facteur d'absorption solaire αf ou αp est donné par le tableau suivant, relevé sur la page 4 du DTA 6/15-2261_V1 [1 p. 4].

| Couleur | Teintes | Valeur de αf, αp (\*) |
| --- | --- | --- |
| Claire | Blanc, jaune, orange, rouge clair | 0,4 |
| Moyenne | Rouge sombre, vert clair, bleu clair | 0,6 |
| Sombre | Brun, vert sombre, bleu vif | 0,8 |
| Noire | Noir, brun sombre, bleu sombre | 1 |

(\*) valeur forfaitaire ou valeur mesurée avec un minimum de 0,4.

(schéma: raw/SOLEAL-GY55-DTA-AG152261_V1.pdf, p. 4)

Pour une fenêtre sans protection mobile ou avec protection mobile en position relevée et sans paroi opaque, et si on considère σ le rapport de la surface de vitrage à la surface totale de la fenêtre, avec `σ = Ag / (Af + Ag)`, on obtient alors : `Sw1 = σ·Sg1` ; `Sw2 = σ·Sg2 + (1 − σ)·Sf` ; donc `Sw = σ·Sg + (1 − σ)·Sf`. Pour les fenêtres de dimensions courantes, les facteurs solaires de la fenêtre sont donnés dans les tableaux [1 p. 4] :

- 4a pour Sw1 (condition de consommation, notée C) et Sw1 (conditions d'été ou de confort, notée E) ;
- 4b pour Sw2 (condition de consommation) et, selon le texte imprimé, Sw1 (conditions d'été ou de confort), alors que le tableau 4b donne Sw2 (E) (**INC-389**) ;
- 4c pour Sws (C) et Sws (E) pour la fenêtre avec protection mobile opaque déployée, alors que le tableau 4c ne donne que Sws (E) (**INC-389**).

## Facteur de transmission lumineuse global de la fenêtre

Le facteur de transmission lumineuse global TLw ou TLws de la fenêtre est déterminé selon la norme XP P50-777, selon les formules suivantes [1 p. 4] : `TLw = Ag / (Ap + Af + Ag) · TLg` (sans protection mobile) ; `TLws = Ag / (Ap + Af + Ag) · TLgs` (avec protection mobile déployée). Ag, Ap et Af sont les surfaces définies ci-dessus (m2). TLg est le facteur de transmission lumineuse du vitrage (désigné tv par dans la norme NF EN 410). TLgs est le facteur de transmission lumineuse du vitrage associé à une protection mobile (déterminé dans la norme NF EN 13363-2) - dans le cas d'une protection mobile extérieure opaque, TLgs = 0.

Si la fenêtre n'a pas de paroi opaque, et si on considère σ le rapport de la surface de vitrage à la surface totale de la fenêtre, avec `σ = Ag / (Af + Ag)`, on obtient alors `TLw = σ·TLg`. Pour les menuiseries de dimensions courantes, les facteurs de transmission lumineuse TLw de la fenêtre et TLws de la fenêtre avec protection mobile opaque déployée sont donnés dans le tableau 4d [1 p. 4].

## Facteur solaire et facteur lumineux de la fenêtre incorporée dans la baie

Facteur solaire ramené à la baie : selon les règles Th-S 2012, le facteur solaire global ramené à la baie avec prise en compte de l'intégration à l'ouvrage de la fenêtre sans protection mobile ou avec protection mobile en position relevée en place est noté Sw_sp-C,b pour les conditions de consommation, avec `Sw_sp-C,b = Sw1_sp-C,b + Sw2_sp-C,b`, et Sw_sp-E,b pour les conditions d'été ou de confort, avec `Sw_sp-E,b = Sw1_sp-E,b + Sw2_sp-E,b`. Les facteurs solaires Sw1_sp-C,b, Sw1_sp-E,b, Sw2_sp-C,b et (texte imprimé) Sw1_sp-E,b sont exprimés en fonction de l'orientation de la baie et du coefficient Ks, avec `Ks = L·H / (dpext·(L + H))`, où L et H sont les dimensions de la baie (m) et dpext la distance entre le plan extérieur du vitrage et le nu extérieur du gros œuvre avec son revêtement (m) [1 p. 4] (**INC-389**).

Facteur de transmission lumineuse global ramené à la baie : selon les règles Th-L 2012, le facteur de transmission lumineuse ramené à la baie avec prise en compte de l'intégration à l'ouvrage de la fenêtre sans protection rapportée en place est noté Tli_sp,b. Les facteurs de transmission lumineuse Tli_sp,b sont exprimés en fonction de l'orientation de la baie, de la mise en œuvre de la fenêtre et du coefficient de forme K, avec `K = L·H / (e·(L + H))`, où L et H sont les dimensions de la baie (m) et e l'épaisseur total du gros œuvre y compris ses revêtements (m) [1 p. 4].

# Citations

[1] [DTA CSTB n° 6/15-2261_V1 SOLEAL GY, publié le 12/07/2019](raw/SOLEAL-GY55-DTA-AG152261_V1.pdf), p. 3 et 4

# Voir aussi

- [Tableaux Ufi, Ψg, Uw, Sw1, Sw2, Sws et TLw du coulissant SOLEAL GY (DTA 6/15-2261, V1)](/certifications/dta-6-15-2261-soleal-gy-thermique.md)
- [DTA CSTB 6/15-2261_V1 SOLEAL GY, édition de 2019 : avis, domaine d'emploi, prescriptions](/certifications/dta-6-15-2261-soleal-gy-v1-avis.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
- [TECHNAL](/fournisseurs/technal.md)
