---
type: Machine
title: Choix des outils d'usinage des dormants et ouvrants du SOLEAL FY 55 (matrice)
description: "Matrice opération x drainage ou version d'ouvrant donnant la référence d'outil (vérin, cassette, bloc de coupe) : drainage, goupillage, épointage, parclose TPE, tringle, tapée, pour les dormants et les ouvrants Minimal, Apparent et Eclat."
tags: [technal, soleal-fy, lumine55, outillage, usinage, drainage, goupillage, poinconnage, ouvrant-minimal, ouvrant-apparent]
gamme: LUMINE
systeme: SOLEAL FY
fournisseur: TECHNAL
usage: atelier
famille: outillage
status: draft
sources:
  - resource: raw/SOLEAL-FY-55-evolution-catalogue-conception-minimal-apparent-6057-003-072020-FR (1).pdf
    title: SOLEAL FY 55 évolution, Ouvrant Minimal et Apparent, Conception, réf. 6057.003
    last_modified: 2020-07-10
source_pages:
  - resource: raw/SOLEAL-FY-55-evolution-catalogue-conception-minimal-apparent-6057-003-072020-FR (1).pdf
    pages: 123
generated:
  by: process:claude-code
  at: 2026-10-09T20:40:00Z
---

Les profilés de la LUMINE55, fabriquée sur le système TECHNAL SOLEAL FY 55 évolution, s'usinent avec des outils de presse (cassettes, vérins, blocs de coupe) choisis par opération. Les tableaux donnent, pour chaque opération (ligne), la référence d'outil de la colonne : drainage caché ou apparent pour les dormants, version Minimal, Apparent ou Eclat pour les ouvrants. Une case grisée sur la planche est une combinaison sans outil, écrite `-` ci-dessous. Les valeurs de drainage et d'usinage de la fabrication sont sur [Fabrication et usinage SOLEAL FY](/procedures/fabrication-soleal-fy.md).

# Légende

INT = 1/2 profilé intérieur ; EXT = 1/2 profilé extérieur ; AV = angle variable ; IN = ouverture intérieure ; OUT = ouverture extérieure. [1 p. 123]

# Outils des dormants

Colonnes : drainage caché, drainage apparent. Une cellule qui porte deux outils les liste l'un sous l'autre (par exemple l'un pour l'ouverture intérieure IN, l'autre pour l'ouverture extérieure OUT). [1 p. 123]

<table>
<thead>
<tr><th rowspan="2">Opération</th><th colspan="2">Dormants</th></tr>
<tr><th>Drainage caché</th><th>Drainage apparent</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Vérin - Porte cassette</td><td colspan="2">TWU0004 - TWU0005</td></tr>
<tr><td colspan="2">T2700 - TEK7037</td></tr>
<tr><td>Drainage dormant</td><td>TFY7012 (IN)</td><td>TFY7009 (IN) ; TWU0015 (OUT)</td></tr>
<tr><td>Drainage traverse</td><td>TWU00020 + TFY7009</td><td>TFY7009 (IN) ; TWU0015 (OUT)</td></tr>
<tr><td>Usinage + drainage battue Equilibre</td><td colspan="2">TWU0045</td></tr>
<tr><td>Goupillage dormant sans rainure</td><td>TEK7008 ; TMI0001 (AV)</td><td>TWU0009 ; TMI0001 (AV)</td></tr>
<tr><td>Goupillage dormant avec rainure coupe d'onglet</td><td>TFY7010</td><td>TFY7010 ; TMI0001 + TFY7029 (AV)</td></tr>
<tr><td>Goupillage dormant avec rainure coupe droite</td><td colspan="2">TMI0007</td></tr>
<tr><td>Goupillage dormants T210085 et T215078</td><td>-</td><td>TFY7000</td></tr>
<tr><td>Goupillage inverseur T215313 et rail T215325</td><td>-</td><td>TFY7002</td></tr>
<tr><td>Tapée</td><td colspan="2">TWU0008</td></tr>
</tbody>
</table>

La référence « TWU00020 » du drainage de traverse du dormant est écrite avec cinq chiffres sur la planche, alors que le drainage de traverse de l'ouvrant porte « TWU0020 » (**INC-594**). [1 p. 123]

# Outils des ouvrants

Colonnes : version Minimal, version Apparent, version Eclat. Une cellule étendue sur plusieurs colonnes vaut pour chacune d'elles. [1 p. 123]

<table>
<thead>
<tr><th>Opération</th><th>Version Minimal</th><th>Version Apparent</th><th>Version Eclat</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Vérin - Porte cassette</td><td colspan="3">TWU0004 - TWU0005</td></tr>
<tr><td colspan="3">T2700 - TEK7037</td></tr>
<tr><td>Epointage</td><td>TWU0012</td><td colspan="2">TWU0011</td></tr>
<tr><td>Drainage parclose TPE</td><td>TFY7036</td><td>-</td><td>-</td></tr>
<tr><td>Montage parclose TPE</td><td>TFY7034</td><td>-</td><td>-</td></tr>
<tr><td>Débit 45° parclose TPE</td><td>TFY7038</td><td>-</td><td>-</td></tr>
<tr><td>Drainage ouvrant</td><td>TWU0014 + TWU0024</td><td colspan="2">TFY7007 + TEK7017 (ou TWU0015) (IN)</td></tr>
<tr><td>Drainage traverse</td><td>TWU0024</td><td>TWU0020 + TFY7009 (IN) ; TWU0015 (OUT)</td><td>TWU0020 + TFY7009</td></tr>
<tr><td>Usinage pour boîtier encastré</td><td>TFY7013</td><td>TFY7011 (IN)</td><td>case vide, non grisée</td></tr>
<tr><td>Usinage parclose TPE + chant clippable</td><td>TWU0024</td><td>-</td><td>-</td></tr>
<tr><td rowspan="4">Goupillage ouvrant</td><td>TWU0012 + TEK7005</td><td>TWU0011 (IN)</td><td>TMI0004</td></tr>
<tr><td>TEK7012 (AV-INT) ; TEK7005 (AV-EXT)</td><td>TMI0002 (AV)</td><td>TMI0002 (AV)</td></tr>
<tr><td>-</td><td>TEK7004 (OUT)</td><td>-</td></tr>
<tr><td>-</td><td>TFY7031 (EXT)</td><td>-</td></tr>
<tr><td>Tringle</td><td colspan="3">TWU0006 + TWU0025 ou TFY7008</td></tr>
</tbody>
</table>

(schéma: raw/SOLEAL-FY-55-evolution-catalogue-conception-minimal-apparent-6057-003-072020-FR (1).pdf, p. 123)

L'intitulé « Usinage parclose TPE + chant clipppable » est imprimé avec trois « p » (**INC-594**). Les outils TFY7036 (cassette de poinçonnage des drainages et équilibrages) et TFY7038 (bloc outil de la coupe d'onglet) sont employés avec les parcloses de [Parcloses arrondies, parcloses TPE à 45° et chant clippable du SOLEAL FY 55](/profiles/soleal-fy-parcloses-arrondies-et-pre-debits-tpe.md). [1 p. 96]

# Citations

[1] SOLEAL FY 55 évolution, Ouvrant Minimal et Apparent, Conception, réf. 6057.003 — `raw/SOLEAL-FY-55-evolution-catalogue-conception-minimal-apparent-6057-003-072020-FR (1).pdf`, p. 123 (et p. 96 pour les outils TFY7036 et TFY7038)

# Voir aussi

- [Fabrication et usinage SOLEAL FY](/procedures/fabrication-soleal-fy.md)
- [Drainage et équilibrage du SOLEAL 55 FY, schémas du DTA](/certifications/dta-6-12-2016-soleal-fy-drainage-et-equilibrage.md)
- [Équerres, goupilles et embouts d'assemblage du SOLEAL FY 55](/profiles/soleal-fy-equerres-goupilles-et-embouts-d-assemblage.md)
- [TECHNAL](/fournisseurs/technal.md)
