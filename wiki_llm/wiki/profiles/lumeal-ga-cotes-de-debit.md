---
type: Profilé
title: Cotes de débit Technal LUMEAL GA
description: Formules de coupe d'atelier, cotes de débit des profilés aluminium, boucliers thermiques et calcul des dimensions de vitrage du coulissant minimal LUMEAL GA de Technal.
tags: [technal, lumeal, lumeal-ga, coulissant, debit, atelier, vitrage, seuil-pmr, formules]
gamme: LUMINE
systeme: 100
status: stable
sources:
  - resource: wiki_llm/a_faire/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf
    id: technal-lumeal-ga-conception-5156-007
    title: LUMEAL GA — Catalogue de conception Le Coulissant Minimal (Réf. 5156.007)
    last_modified: 2021-01-21
  - resource: wiki_llm/a_faire/LUMEAL-GA-catalogue-fabrication-5074-007-012021-FR.pdf
    id: technal-lumeal-ga-fabrication-5074-007
    title: LUMEAL GA — Guide d'atelier et catalogue de fabrication (Réf. 5074.007)
    last_modified: 2021-01-27
source_pages:
  - resource: wiki_llm/a_faire/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf
    pages: 21-41, 51-53, 58-59
  - resource: wiki_llm/a_faire/LUMEAL-GA-catalogue-fabrication-5074-007-012021-FR.pdf
    pages: 4-15
generated:
  by: process:multimodal-direct
  at: 2026-09-19T23:55:00Z
---

# Définition

Les cotes de débit du coulissant minimal **LUMEAL GA** fixent les formules de coupe précises des profilés dormants, ouvrants cachés, traverses thermoplastiques, boucliers et volumes verriers à déduire des dimensions hors-tout de fabrication Largeur ($L$) et Hauteur ($H$) [1 p. 21].

---

# Cotes

### 1. Débits de la fenêtre et porte-fenêtre 2 vantaux 2 rails

Profilés pour châssis 2 vantaux 2 rails en coupe droite, cotes en mm relevées sur les catalogues de conception et de fabrication.

| Composant | Référence | Quantité | Formule de débit (mm) | Remarque technique |
| --- | --- | --- | --- | --- |
| **Rail bas dormant** | T141000 / T141001 | 1 | $L - 42$ | Coupe droite |
| **Rail haut dormant** | T141017 / T141019 | 1 | $L - 42$ | Coupe droite |
| **Montants dormants** | T141030 / T141031 | 2 | $H$ | Hauteur hors tout |
| **Traverses basses ouvrant** | T141015 (ou T141021) | 2 | $L/2 - 65$ | Aluminium à coupure thermique |
| **Traverses hautes ouvrant** | T821004 (ou T821005) | 2 | $L/2 - 95$ | Profilé PVC thermoplastique |
| **Montants centraux chicanes**| T141037 / T141038 | 2 | $H - 105$ | Chicane fine ou renforcée |
| **Montants latéraux ouvrant** | T141039 (ou T141040) | 2 | $H - 105$ | Reçoit le bloc serrure |
| **Chemin de roulement alu** | T341000 | 2 | $L - 98$ | Rail rapporté clippé |
| **Bouclier PVC supérieur** | T431024 | 2 | $L/2 - 57$ | Pré-percé pour drainage |
| **Bouclier PVC inférieur** | T431025 | 1 | $L - 42$ | Pré-percé pour évacuation |
| **Bouclier acoustique montant**| T823000 | 2 | $H - 115$ | Dans gorge de montant dormant |
| **Bouclier thermique montant** | T823001 | 2 | $H - 115$ | Dans gorge de montant dormant |
| **Volume de vitrage (2 vtx)** | Double vitrage 24-32 | 2 | $(H - 143) \times (L/2 - 66)$ | Hauteur x Largeur vitrage |

(schéma: wiki_llm/a_faire/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf, p. 24-27)

---

### 2. Débits de la porte-fenêtre 2 vantaux avec seuil PMR (Ressaut $\le 20\text{ mm}$)

Profilés pour porte-fenêtre 2 vantaux équipée du seuil surbaissé T141014 et de la rampe T401028.

| Composant | Référence | Quantité | Formule de débit (mm) | Remarque technique |
| --- | --- | --- | --- | --- |
| **Seuil bas PMR 2 rails** | T141014 | 1 | $L - 42$ | Seuil surbaissé 32,7 mm |
| **Rail haut dormant** | T141017 / T141019 | 1 | $L - 42$ | Coupe droite |
| **Montants dormants** | T141030 / T141031 | 2 | $H$ | Hauteur hors tout |
| **Traverses basses ouvrant** | T141015 (ou T141021) | 2 | $L/2 - 65$ | Aluminium |
| **Montants centraux** | T141037 / T141038 | 2 | $H - 87$ | Déduction réduite pour PMR |
| **Montants latéraux ouvrant** | T141039 (ou T141040) | 2 | $H - 87$ | Déduction réduite pour PMR |
| **Capot rejet d'eau PMR** | T341012 | 1 | $L/2 - 63$ | Sur vantail semi-fixe |
| **Support rejet d'eau PMR** | T341013 | 1 | $L/2 - 102$ | Clip sous rejet d'eau |
| **Bouclier PVC seuil PMR** | T821008 | 1 | $L - 42$ | Profilé PVC étanchéité basse |
| **Bouclier drainage PMR** | T821010 | 1 | $L/2 - 221$ | Profilé PVC intérieur |
| **Boucliers acoustique/therm.**| T823000 / T823001 | 2 | $H - 97$ | Recoupes montants dormants |
| **Volume de vitrage PMR** | Double vitrage 24-32 | 2 | $(H - 125) \times (L/2 - 66)$ | Hauteur x Largeur vitrage |

(schéma: wiki_llm/a_faire/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf, p. 38-41)

---

### 3. Débits de la porte-fenêtre 4 vantaux 2 rails (Percussion centrale)

| Composant | Référence | Quantité | Formule de débit (mm) | Remarque technique |
| --- | --- | --- | --- | --- |
| **Rail bas et rail haut** | T141001 / T141017 | 1 chaque | $L - 42$ | Coupe droite |
| **Montants dormants** | T141031 / T141030 | 2 | $H$ | Hauteur hors tout |
| **Traverses basses ouvrant** | T141015 (ou T141021) | 4 | $L/4 - 46$ | 4 vantaux mobiles |
| **Traverses hautes ouvrant** | T821004 (ou T821005) | 4 | $L/4 - 76$ | Profilé thermoplastique |
| **Montants centraux** | T141037 / T141038 | 2 chaque | $H - 105$ | Chicanes intermédiaires |
| **Montants latéraux ouvrant** | T141039 | 2 | $H - 105$ | Montants extérieurs |
| **Montants percussion centrale**| T141033 | 2 | $H - 105$ | Montants de battement central |
| **Profilé capot percussion** | T341007 / T341008 | 1 chaque | $H - 105$ | Finition centrale |
| **Profilé PVC percussion** | T821003 | 1 | $H - 140$ | Isolation de battement |
| **Chemin de roulement alu** | T341000 | 1 + 1 | $L - 154$ et $L - 42$ | Rails rapportés |
| **Bouclier PVC supérieur** | T431024 | 2 + 1 | $L/4 - 38$ et $L/2 - 112$ | Supérieur |
| **Volume de vitrage (4 vtx)** | Double vitrage 24-32 | 4 | $(H - 143) \times (L/4 - 47)$ | 4 volumes identiques |

(schéma: wiki_llm/a_faire/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf, p. 30-33)

---

### 4. Débits de la porte-fenêtre 3 vantaux 3 rails

| Composant | Référence | Quantité | Formule de débit (mm) | Remarque technique |
| --- | --- | --- | --- | --- |
| **Rail bas 3 rails** | TGA1101 | 1 | $L - 42$ | Module 151 mm |
| **Rail haut 3 rails** | TGA1100 | 1 | $L - 42$ | Module 151 mm |
| **Montants dormants 3 rails** | TGA1102 | 2 | $H$ | Profondeur 157 mm |
| **Traverses basses ouvrant** | T141015 (ou T141021) | 3 | $L/3 - 46$ | 3 vantaux indépendants |
| **Traverses hautes ouvrant** | T821004 (ou T821005) | 3 | $L/3 - 76$ | PVC |
| **Montants centraux** | T141037 / T141038 | 2 chaque | $H - 105$ | Chicanes |
| **Montants latéraux ouvrant** | T141039 | 2 | $H - 105$ | Montants d'extrémité |
| **Chemin de roulement alu** | T341000 | 1 + 2 | $L - 154$ et $L - 98$ | 3 rails |
| **Bouclier PVC supérieur** | T431024 | 2 + 2 | $L/3 - 38$ et $2L/3 - 76$ | Supérieur |
| **Bouclier PVC inférieur** | T431025 | 2 | $L - 42$ | Inférieur |
| **Volume de vitrage (3 vtx)** | Double vitrage 24-32 | 3 | $(H - 143) \times (L/3 - 47)$ | 3 volumes identiques |

(schéma: wiki_llm/a_faire/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf, p. 32-35)

---

### 5. Débits du monorail 1 vantail mobile + 1 partie fixe ($L_f$)

$L_f$ désigne la largeur en feuillure de la partie fixe vitrée.

| Composant | Référence | Quantité | Formule de débit (mm) | Remarque technique |
| --- | --- | --- | --- | --- |
| **Rail bas monorail** | T141026 / T141025 | 1 | $L - 42$ | Monorail |
| **Rail haut dormant** | T141017 | 1 | $L - 42$ | Traverse haute |
| **Montants dormants** | T141031 | 2 | $H$ | Montants latéraux |
| **Traverse basse ouvrant** | T141015 | 1 | $L - L_f - 65$ | Vantail coulissant |
| **Traverse haute ouvrant PVC** | T821004 | 1 | $L - L_f - 95$ | Vantail coulissant |
| **Montant central ouvrant** | T141037 | 1 | $H - 97$ | Côté fixe |
| **Montant central droit** | T141038 | 1 | $H - 105$ | Renfort |
| **Montant latéral ouvrant** | T141039 | 1 | $H - 105$ | Côté serrure |
| **Capot de rail bas** | T341025 | 1 + 1 | $L - L_f - 96$ et $L - L_f - 98$ | Finition rail |
| **Compensateur de feuillure** | T821002 | 1 + 1 | $H - 97$ et $L_f - 44$ | Pour calage vitrage fixe |
| **Bouclier bas monorail** | T821001 | 1 | $L - 42$ | Sous partie fixe |
| **Vitrage ouvrant mobile** | Double vitrage 24-32 | 1 | $(H - 143) \times (L - L_f - 66)$ | Vantail mobile |
| **Vitrage partie fixe** | Double vitrage 24-32 | 1 | $(H - 75) \times (L_f - 66)$ | Partie fixe vitrée |

(schéma: wiki_llm/a_faire/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf, p. 34-37)

---

### 6. Traverses intermédiaires et débits des vitrages

Le débit en longueur des traverses intermédiaires T141018 (vitrage 24 à 28 mm) et T141009 (vitrage 30 à 32 mm) est strictement égal au débit de la traverse basse d'ouvrant associée [1 p. 51-52].

Hauteurs des deux vitrages séparés par la traverse intermédiaire, avec $ht_1$ désignant la hauteur d'axe de traverse par rapport au bas du châssis [1 p. 53] :
- **Application standard** :
  $$h_1 = ht_1 - 111\text{ mm}$$
  $$h_2 = H - ht_1 - 69\text{ mm}$$
- **Variante seuil réduit** :
  $$h_1 = ht_1 - 94\text{ mm}$$
  $$h_2 = H - ht_1 - 69\text{ mm}$$

---

### 7. Largeurs de passage libre utiles

Formules de déduction de l'unité de passage libre maximale [1 p. 58-59] :
- **Châssis 2 vantaux 2 rails** :
  $$L_{\text{passage}} = L/2 - 241\text{ mm}$$
- **Châssis 1 vantail + fixe (Monorail)** :
  $$L_{\text{passage}} = L - L_f - 241\text{ mm}$$
- **Châssis 3 vantaux 3 rails (refoulement total des 3 vantaux)** :
  $$L_{\text{passage}} = 2L/3 - 370\text{ mm}$$
- **Châssis 3 vantaux 3 rails (refoulement partiel 1 vantail)** :
  $$L_{\text{passage}} = L/3 - 77\text{ mm}$$
  *(si présence d'une poignée de tirage extérieure, la formule devient $L_{\text{passage}} = L/3 - 177\text{ mm}$)*.

---

# Ce que la source ne donne pas

- Le catalogue ne fournit pas de formule pour des configurations à galandage dans la gamme LUMEAL GA (le galandage aluminium Technal est traité sous la gamme SOLÉAL GY).

---

# Citations

[1] LUMEAL GA — Catalogue de conception Le Coulissant Minimal (Réf. 5156.007 - 01/2021) — `wiki_llm/a_faire/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf`
[2] LUMEAL GA — Guide d'atelier et catalogue de fabrication (Réf. 5074.007 - 01/2021) — `wiki_llm/a_faire/LUMEAL-GA-catalogue-fabrication-5074-007-012021-FR.pdf`

---

# Voir aussi

- [Profilés dormants et ouvrants LUMEAL GA](/profiles/lumeal-ga-dormants-et-ouvrants.md)
- [Roulements et fermetures LUMEAL GA](/quincaillerie/lumeal-ga-roulements-et-fermetures.md)
- [Fabrication et pose LUMEAL GA](/procedures/fabrication-et-pose-lumeal-ga.md)
- [Coulissants aluminium](/gammes/coulissants-aluminium.md)
- [TECHNAL](/fournisseurs/technal.md)
