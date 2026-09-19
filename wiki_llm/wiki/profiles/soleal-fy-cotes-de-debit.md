---
type: Profilé
title: Cotes de débit SOLEAL FY
description: Formules de coupe d'atelier, déductions et cotes de débit des profilés dormants, ouvrants, battements, seuils PMR, parcloses et vitrages pour fenêtres et portes-fenêtres TECHNAL SOLEAL FY 55 et FY 65.
tags: [technal, soleal, soleal-fy, debit, cotes, atelier, coupe, vitrage, seuil-pmr]
gamme: LUMINE
systeme: [55, 65]
status: stable
sources:
  - resource: wiki_llm/a_faire/DTA 6_12-2016_V5 (1).pdf
    id: technal-dta-soleal-fy-6-12-2016-v5
    title: DTA n° 6/12-2016_V5, procédé Soleal 55 FYa - FYm
    last_modified: 2023-05-25
  - resource: wiki_llm/a_faire/SOLEAL-FY-55-evolution-catalogue-conception-minimal-apparent-6057-003-072020-FR (1).pdf
    id: technal-soleal-fy-55-conception-6057
    title: SOLEAL FY 55 évolution, Conception Ouvrant Minimal et Apparent, réf. 6057.003
    last_modified: 2020-07-10
  - resource: wiki_llm/a_faire/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf
    id: technal-soleal-fy-qc-conception-6319
    title: SOLEAL FY 55 & 65, Évolution Quincaillerie Cachée, Conception, réf. 6319.003
    last_modified: 2021-07-29
source_pages:
  - resource: wiki_llm/a_faire/SOLEAL-FY-55-evolution-catalogue-conception-minimal-apparent-6057-003-072020-FR (1).pdf
    pages: 40-79, 84-87, 92-103
  - resource: wiki_llm/a_faire/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf
    pages: 42-73, 78-86
---

# Principes généraux de débit SOLEAL FY

Toutes les cotes de débit sont exprimées en millimètres (**mm**) à partir des dimensions hors tout de fabrication du châssis : **Largeur hors tout ($L$ ou $W$)** et **Hauteur hors tout ($H$)** [1, 2 p. 40, 3 p. 42].

* **Coupe d'onglet à 45°** : utilisée systématiquement pour les cadres dormants périphériques et les cadres ouvrants à frappe.
* **Coupe droite à 90°** : réservée aux traverses intermédiaires, battements centraux, seuils plats PMR et parcloses intérieures/extérieures.
* **Tolérance de laquage** : pour les profilés laqués, prévoir les débits des traverses d'ouvrant ainsi que la largeur des vitrages **plus petits de 1 mm** pour absorber l'épaisseur du film de poudre et éviter tout coincement en feuillure [2 p. 42, 44].

---

# Formules de débit — Ouvrant Apparent (FYa / OA)

Châssis équipés de l'ouvrant tubulaire 65 mm (T215180, T215181 ou TFY1206) et parcloses droites (T591005) [2 p. 42-59, 3 p. 42-49, 58-65].

### 1. Châssis fixe et fenêtres 1 et 2 vantaux

| Élément débité | Châssis Fixe | Fenêtre 1 vantail OF / OB | Fenêtre 2 vantaux OF / OB | Type de coupe |
| --- | :---: | :---: | :---: | :---: |
| **Dormant périphérique** | $2 \times H$ (H)<br>$2 \times L$ (L) | $2 \times H$ (H)<br>$2 \times L$ (L) | $2 \times H$ (H)<br>$2 \times L$ (L) | Coupe d'onglet 45°/45° |
| **Cadre ouvrant** (montants) | — | $2 \times (H - 44)$ | $4 \times (H - 44)$ | Coupe d'onglet 45°/45° |
| **Cadre ouvrant** (traverses) | — | $2 \times (L - 44)$ | $4 \times (\frac{L}{2} - 25)$ | Coupe d'onglet 45°/45° |
| **Battement central** (T215186 / TFY1207) | — | — | $1 \times (H - 116)$ | Coupe droite 90°/90° |
| **Parcloses droites** (hauteur) | $2 \times (H - 98)$ | $2 \times (H - 175)$ | $4 \times (H - 175)$ | Coupe droite 90°/90° |
| **Parcloses droites** (largeur) | $2 \times (L - 54)$ | $2 \times (L - 131)$ | $4 \times (\frac{L}{2} - 112)$ | Coupe droite 90°/90° |
| **Remplissage vitrage** (hauteur) | $H - 66$ | $H - 143$ | $H - 143$ | Rectangle |
| **Remplissage vitrage** (largeur) | $L - 66$ | $L - 143$ | $\frac{L}{2} - 124$ | Rectangle |

### 2. Portes-fenêtres avec seuil PMR (Seuil 20 mm T215309 en 55 ou TFY1161 en 65)

| Élément débité | Porte-fenêtre 1 vantail PMR | Porte-fenêtre 2 vantaux PMR | Type de coupe |
| --- | :---: | :---: | :---: |
| **Dormant** (montants verticaux) | $2 \times H$ | $2 \times H$ | Mixte 90° bas / 45° haut |
| **Dormant** (traverse haute) | $1 \times L$ | $1 \times L$ | Coupe d'onglet 45°/45° |
| **Seuil aluminium PMR** (T215309 / TFY1161) | $1 \times (L - 104)$ | $1 \times (L - 104)$ | Coupe droite 90°/90° |
| **Cadre ouvrant** (montants verticaux) | $2 \times (H - 23,5)$ | $4 \times (H - 23,5)$ | Coupe d'onglet 45°/45° |
| **Cadre ouvrant** (traverse haute) | $1 \times (L - 44)$ | $2 \times (\frac{L}{2} - 25)$ | Coupe d'onglet 45°/45° |
| **Cadre ouvrant** (traverse basse PMR TFY1256) | $1 \times (L - 68)$ | $2 \times (\frac{L}{2} - 49)$ | Coupe d'onglet 45°/45° |
| **Battement central** (T215186 / TFY1207) | — | $1 \times (H - 91)$ | Coupe droite 90°/90° |
| **Porte-joint seuil plat** (T215310) | $1 \times (L - 77)$ | $2 \times (\frac{L}{2} - 58)$ | Coupe droite 90°/90° |
| **Porte-brosse périphérique** (T215312) | $1 \times (L - 118)$ | $2 \times (\frac{L}{2} - 99)$ | Coupe droite 90°/90° |
| **Profil rejet d'eau** (TFY2117) | $1 \times (L - 116)$ | $1 \times (\frac{L}{2} - 97) + 1 \times (\frac{L}{2} - 25)$ | Coupe droite 90°/90° |
| **Parcloses droites** (hauteur) | $2 \times (H - 154,5)$ | $4 \times (H - 154,5)$ | Coupe droite 90°/90° |
| **Parcloses droites** (largeur) | $2 \times (L - 131)$ | $4 \times (\frac{L}{2} - 112)$ | Coupe droite 90°/90° |
| **Remplissage vitrage** | $(H - 123) \times (L - 143)$ | $(H - 123) \times (\frac{L}{2} - 124)$ | Rectangle |

(schéma: wiki_llm/a_faire/SOLEAL-FY-55-evolution-catalogue-conception-minimal-apparent-6057-003-072020-FR (1).pdf, p. 52-57 et 3 p. 46-51)

---

# Formules de débit — Ouvrant Minimal (FYm / OM)

Châssis équipés des profilés d'ouvrants masqués TFY1338 / TFY1339 et parcloses isolantes extérieures en TPE (TFY4002 / TFY4003) [2 p. 62-75, 3 p. 50-57, 66-73].

### 1. Fenêtres 1 et 2 vantaux Minimal

| Élément débité | Fenêtre 1 vantail Minimal | Fenêtre 2 vantaux Minimal | Type de coupe |
| --- | :---: | :---: | :---: |
| **Dormant périphérique** | $2 \times H$ / $2 \times L$ | $2 \times H$ / $2 \times L$ | Coupe d'onglet 45°/45° |
| **Cadre ouvrant masqué** (montants) | $2 \times (H - 44)$ | $4 \times (H - 44)$ | Coupe d'onglet 45°/45° |
| **Cadre ouvrant masqué** (traverses) | $2 \times (L - 44)$ | $4 \times (\frac{L}{2} - 25)$ | Coupe d'onglet 45°/45° |
| **Battement central** (T215186 / TFY1207) | — | $1 \times (H - 116)$ | Coupe droite 90°/90° |
| **Parclose TPE extérieure** (coupe droite) | $2 \times (H - 179,5)$<br>$2 \times (L - 179,5)$ | $4 \times (H - 179,5)$<br>$4 \times (\frac{L}{2} - 160,5)$ | Coupe droite avec pièces d'angle TFY3850 |
| **Parclose TPE extérieure** (option coupe 45°) | $2 \times (H - 73)$<br>$2 \times (L - 73)$ | $4 \times (H - 73)$<br>$4 \times (\frac{L}{2} - 54)$ | Pré-débit coupe droite puis onglet outil TFY7038 |
| **Chant clippable alu TFY2405** (si option CC à 45°) | $2 \times (H - 116)$<br>$2 \times (L - 116)$ | $4 \times (H - 116)$<br>$4 \times (\frac{L}{2} - 97)$ | Coupe d'onglet 45° ($H_v, L_v \ge 900\text{ mm}$) |
| **Remplissage vitrage** (hauteur) | $H - 121$ | $H - 121$ | Rectangle |
| **Remplissage vitrage** (largeur) | $L - 121$ | $\frac{L}{2} - 102$ | Rectangle |

### 2. Portes-fenêtres Minimal avec seuil PMR

| Élément débité | Porte-fenêtre 1 vantail Minimal PMR | Porte-fenêtre 2 vantaux Minimal PMR | Type de coupe |
| --- | :---: | :---: | :---: |
| **Dormant** (montants verticaux) | $2 \times H$ | $2 \times H$ | Mixte 90°/45° |
| **Dormant** (traverse haute) | $1 \times L$ | $1 \times L$ | Coupe d'onglet 45°/45° |
| **Seuil aluminium PMR** | $1 \times (L - 104)$ | $1 \times (L - 104)$ | Coupe droite 90°/90° |
| **Cadre ouvrant masqué** (montants) | $2 \times (H - 23,5)$ | $4 \times (H - 23,5)$ | Coupe d'onglet 45°/45° |
| **Cadre ouvrant masqué** (traverse haute) | $1 \times (L - 44)$ | $2 \times (\frac{L}{2} - 25)$ | Coupe d'onglet 45°/45° |
| **Cadre ouvrant masqué** (traverse basse PMR TFY1258) | $1 \times (L - 68)$ | $2 \times (\frac{L}{2} - 49)$ | Coupe d'onglet 45°/45° |
| **Battement central** (T215186 / TFY1207) | — | $1 \times (H - 91)$ | Coupe droite 90°/90° |
| **Profil rejet d'eau bas** (TFY2116) | $1 \times (L - 117)$ | $1 \times (\frac{L}{2} - 98) + 1 \times (\frac{L}{2} - 25,5)$ | Coupe droite 90°/90° |
| **Parclose TPE extérieure** (hauteur) | $2 \times (H - 52,5)$ | $4 \times (H - 52,5)$ | Coupe droite 90°/90° |
| **Parclose TPE extérieure** (largeur) | $2 \times (L - 73)$ | $4 \times (\frac{L}{2} - 54)$ | Coupe droite 90°/90° |
| **Remplissage vitrage** | $(H - 101) \times (L - 121)$ | $(H - 101) \times (\frac{L}{2} - 102)$ | Rectangle |

(schéma: wiki_llm/a_faire/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf, p. 54-57 et 70-75)

---

# Formules de débit — Portes-fenêtres avec serrure et grande inertie

Portes-fenêtres équipées de l'ouvrant à montant serrure TFY1213 (face vue 62,5 mm) recevant une serrure 3 points à pênes basculants TFY3724 et béquille double rosette [2 p. 58-61, 74-77] :

| Élément débité | 1 vantail serrure seuil PMR | 2 vantaux serrure seuil PMR | Type de coupe |
| --- | :---: | :---: | :---: |
| **Ouvrant montant serrure** (TFY1213) | $2 \times (H - 29)$ | $4 \times (H - 29)$ | Coupe d'onglet 45°/45° |
| **Ouvrant traverse** | $2 \times (L - 44)$ | $4 \times (\frac{L}{2} - 25)$ | Coupe d'onglet 45°/45° |
| **Battement central** (T215186) | — | $1 \times (H - 88)$ | Coupe droite 90°/90° |
| **Parclose droite** (hauteur) | $2 \times (H - 226)$ | $4 \times (H - 226)$ | Coupe droite 90°/90° |
| **Parclose droite** (largeur) | $2 \times (L - 197)$ | $4 \times (\frac{L}{2} - 178)$ | Coupe droite 90°/90° |
| **Remplissage vitrage** | $(H - 194) \times (L - 209)$ | $(H - 194) \times (\frac{L}{2} - 190)$ | Rectangle |

---

# Formules de débit — Traverses intermédiaires d'ouvrant

Pour l'intégration d'un meneau ou d'une traverse d'ouvrant divisant le vantail en deux vitrages de hauteurs $H_{v1}$ et $H_{v2}$ ($H_v = H_{v1} + H_{v2}$) [2 p. 98-102] :

| Type d'ouvrant | Réf. traverse | Débit de la traverse | Débit du capot (si applicable) | Hauteur vitrage bas ($H_{v1}$) | Hauteur vitrage haut ($H_{v2}$) |
| --- | :---: | :---: | :---: | :---: | :---: |
| **Ouvrant Apparent (FYa)** | T215202 (15/77) | $L_v - 131\text{ mm}$ (1 vtl)<br>$\frac{L_v}{2} - 112\text{ mm}$ (2 vtx) | — | $H_{v1} - 69\text{ mm}$ | $H_{v2} - 69\text{ mm}$ |
| **Ouvrant Apparent (FYa)** | T215204 (28/90) | $L_v - 131\text{ mm}$ (1 vtl)<br>$\frac{L_v}{2} - 112\text{ mm}$ (2 vtx) | — | $H_{v1} - 75,5\text{ mm}$ | $H_{v2} - 75,5\text{ mm}$ |
| **Ouvrant Apparent (FYa)** | T215205 (38/100) | $L_v - 131\text{ mm}$ (1 vtl)<br>$\frac{L_v}{2} - 112\text{ mm}$ (2 vtx) | — | $H_{v1} - 80,5\text{ mm}$ | $H_{v2} - 80,5\text{ mm}$ |
| **Ouvrant Apparent (FYa)** | T215208 (68/130) | $L_v - 131\text{ mm}$ (1 vtl)<br>$\frac{L_v}{2} - 112\text{ mm}$ (2 vtx) | — | $H_{v1} - 95,5\text{ mm}$ | $H_{v2} - 95,5\text{ mm}$ |
| **Ouvrant Minimal (FYm)** | TFY1340 (80 mm) | $L_v - 67\text{ mm}$ | $L_v - 81\text{ mm}$ (TFY2306) | $H_{v1} - 66,5\text{ mm}$ | $H_{v2} - 66,5\text{ mm}$ |
| **Ouvrant Minimal (FYm)** | TFY1341 (80 mm) | $L_v - 67\text{ mm}$ | $L_v - 81\text{ mm}$ (TFY2306) | $H_{v1} - 66,5\text{ mm}$ | $H_{v2} - 66,5\text{ mm}$ |
| **Ouvrant Minimal (FYm)** | TFY1342 (100 mm) | $L_v - 67\text{ mm}$ | $L_v - 81\text{ mm}$ (TFY2307) | $H_{v1} - 76,5\text{ mm}$ | $H_{v2} - 76,5\text{ mm}$ |

---

# Formules de débit — Châssis spéciaux (soufflet, italienne, projection)

| Configuration | Profilé d'ouvrant | Débit cadre ouvrant | Débit parcloses | Remplissage vitrage |
| --- | :---: | :---: | :---: | :---: |
| **Fenêtre à soufflet Apparent** | T215180 | $2 \times (H - 44)$<br>$2 \times (L - 44)$ | $2 \times (H - 175)$<br>$2 \times (L - 131)$ | $(H - 143) \times (L - 143)$ |
| **Fenêtre à soufflet Minimal** | TFY1338 | $2 \times (H - 44)$<br>$2 \times (L - 44)$ | $2 \times (H - 179,5)$<br>$2 \times (L - 179,5)$ | $(H - 121) \times (L - 121)$ |
| **Fenêtre à l'italienne** | T215198 | $2 \times (H - 44)$<br>$2 \times (L - 44)$ | $2 \times (H - 244)$<br>$2 \times (L - 200)$ | $(H - 212) \times (L - 212)$ |
| **Fenêtre à projection** | T210082 | $2 \times (H - 44)$<br>$2 \times (L - 44)$ | $2 \times (H - 241)$<br>$2 \times (L - 197)$ | $(H - 209) \times (L - 209)$ |

---

# Citations

[1] DTA n° 6/12-2016_V5, procédé Soleal 55 FYa - FYm — `wiki_llm/a_faire/DTA 6_12-2016_V5 (1).pdf`, p. 7 à 12

[2] SOLEAL FY 55 évolution, Conception Ouvrant Minimal et Apparent, réf. 6057.003 — `wiki_llm/a_faire/SOLEAL-FY-55-evolution-catalogue-conception-minimal-apparent-6057-003-072020-FR (1).pdf`, p. 40 à 103

[3] SOLEAL FY 55 & 65, Évolution Quincaillerie Cachée, Conception, réf. 6319.003 — `wiki_llm/a_faire/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf`, p. 42 à 86

# Voir aussi

- [Dormants et ouvrants SOLEAL FY](/profiles/soleal-fy-dormants-et-ouvrants.md)
- [Parcloses et vitrages SOLEAL FY](/profiles/soleal-fy-parcloses-et-vitrage.md)
- [Fabrication et usinage SOLEAL FY](/procedures/fabrication-soleal-fy.md)
- [Quincaillerie visible et cachée SOLEAL FY](/quincaillerie/soleal-fy-quincaillerie-visible-et-cachee.md)
- [TECHNAL](/fournisseurs/technal.md)
