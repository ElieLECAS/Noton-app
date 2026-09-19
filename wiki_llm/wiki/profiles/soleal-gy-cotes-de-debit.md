---
type: Profilé
title: Cotes de débit et formules de calcul Technal SOLEAL GY 55
description: Formules de débit exhaustives, déductions de profilés (dormants, ouvrants, rails alu/inox, boucliers thermiques, chicanes), dimensionnement des vitrages et débits d'angle pour le coulissant SOLEAL GY 55 de TECHNAL.
tags: [technal, soleal, soleal-gy, debit, formules, coulissant, galandage, vitrage, seuil-pmr]
gamme: LUMINE
systeme: 55
status: stable
sources:
  - resource: wiki_llm/a_faire/SOLEAL-GY-55-Catalogue-conception-5744-005-092021-Fr (1).pdf
    id: technal-soleal-gy-55-conception-5744-005
    title: SOLEAL GY 55 — Catalogue de conception Le Coulissant Universel (Réf. 5744.005)
    last_modified: 2021-09-01
  - resource: wiki_llm/a_faire/SOLEAL-GY-55-Catalogue-fabrication-5746-003-092021-Fr.pdf
    id: technal-soleal-gy-55-fabrication-5746-003
    title: SOLEAL GY 55 — Guide de fabrication d'atelier (Réf. 5746.003)
    last_modified: 2021-09-01
generated:
  by: process:multimodal-direct
  at: 2026-09-19T22:20:00Z
---

# Formules de débit des profilés du cadre dormant

Les profilés dormants sont débités à coupe d'onglet à 45° sur les 4 côtés :

| Configuration | Profilé traverse haute & montants | Profilé traverse basse (avec recueil) | Débit montants | Débit traverses |
| --- | --- | --- | --- | --- |
| **2 rails sans rainure BTC** | TGY1104 | TGY1120 | 2x $H$ à 45° | 2x $L$ à 45° |
| **2 rails à rainure BTC** | TGY1113 | TGY1114 | 2x $H$ à 45° | 2x $L$ à 45° |
| **2 rails avec Couvre-joint** | TGY1115 | TGY1116 | 2x $H$ à 45° | 2x $L$ à 45° |
| **Dormant rénovation** | TGY1121 | TGY1121 | 2x $H$ à 45° | 2x $L$ à 45° |
| **3 rails sans rainure BTC** | TGY1111 | TGY1112 | 2x $H$ à 45° | 2x $L$ à 45° |
| **3 rails à rainure BTC** | TGY1100 | TGY1101 | 2x $H$ à 45° | 2x $L$ à 45° |
| **4 rails à rainure BTC** | TGY1102 | TGY1103 | 2x $H$ à 45° | 2x $L$ à 45° |
| **Intégration frappe 55** | TGY1119 | TGY1119 | 2x $(Hx + 8)$ à 45° | 2x $(Lx + 8)$ à 45° |
| **Intégration frappe 65** | TGY1143 | TGY1143 | 2x $(H - 19)$ à 45° | 2x $(L - 19)$ à 45° |

---

# Formules de débit des profilés ouvrants

Tous les profilés ouvrants sont débités à coupe droite à 90°.

### 1. Hauteur des montants d'ouvrants
La hauteur de tous les montants (latéraux et centraux) est strictement constante :
$$\mathbf{H_{\text{montant}} = H - 77\text{ mm}}$$

Profilés concernés :
- Montants latéraux : TGY1202, TGY1300, TGY1200, TGY1201, TGY1301, TGY1302, TGY1303, TGY1304, TGY1220.
- Montants centraux : TGY1207, TGY1208, TGY1209, TGY1210, TGY1211, TGY1213, TGY1214, TGY1215, TGY1216, TGY1217.

Profilés additionnels d'ouvrant :
- Battement de percussion centrale **TGY1204** : débit $= \mathbf{H - 170\text{ mm}}$
- Battement de percussion d'angle **TGY1206** : débit $= \mathbf{H - 170\text{ mm}}$
- Profilé d'angle prépercé **TGY2203** : débit $= \mathbf{H - 77\text{ mm}}$ (angle sortant) ou $\mathbf{H - 111\text{ mm}}$ (angle sortant avec seuil PMR ou angle rentrant)
- Couvre-joint de finition d'angle **TGY2206** : débit $= \mathbf{H - 77\text{ mm}}$ (angle sortant) ou $\mathbf{H - 111\text{ mm}}$ (avec seuil PMR)

### 2. Largeur des traverses d'ouvrants (basses et hautes)
Profilés traverses : **T141015** (vitrage 24-28 mm) ou **T141021** (vitrage 29-32 mm).

| Configuration du châssis | Nombre de vantaux / rails | Nombre de traverses | Débit unitaire traverse basse et haute |
| --- | --- | --- | --- |
| **Fenêtre / Porte-fenêtre 2 vantaux** | 2 vtx / 2 rails | 4 | $\mathbf{L/2 - 91\text{ mm}}$ |
| **Porte-fenêtre 3 vantaux indépendants** | 3 vtx / 2 rails | 6 | $\mathbf{L/3 - 61\text{ mm}}$ |
| **Porte-fenêtre 4 vantaux** | 4 vtx / 2 rails (percussion) | 8 | $\mathbf{L/4 - 73\text{ mm}}$ |
| **Porte-fenêtre 3 vantaux 3 rails** | 3 vtx / 3 rails | 6 | $\mathbf{L/3 - 61\text{ mm}}$ |
| **Porte-fenêtre 6 vantaux 3 rails** | 6 vtx / 3 rails (percussion) | 12 | $\mathbf{L/6 - 49,5\text{ mm}}$ |
| **Porte-fenêtre 4 vantaux 4 rails** | 4 vtx / 4 rails | 8 | $\mathbf{L/4 - 45,5\text{ mm}}$ |
| **Porte-fenêtre 8 vantaux 4 rails** | 8 vtx / 4 rails (percussion) | 16 | $\mathbf{L/8 - 37,5\text{ mm}}$ |
| **Galandage 1 vantail** | 1 vtl monorail | 2 | $\mathbf{L - 73\text{ mm}}$ |
| **Galandage 2 vantaux percussion** | 2 vtx monorail percussion | 4 | $\mathbf{L/2 - 37,5\text{ mm}}$ |
| **Galandage 2 vantaux 2 rails** | 2 vtx sur 2 rails | 4 | $\mathbf{L/2 - 37,5\text{ mm}}$ |
| **Intégration frappe SOLEAL FY** | 2 vtx coulissant | 4 | $\mathbf{Lx/2 - 106\text{ mm}}$ |

### 3. Traverses intermédiaires
- Pour vitrage 24 à 28 mm (**T141018**) :
  $$\text{Débit } T141018 = \text{Débit traverse basse } T141015 + 0,5\text{ mm}$$
  *(Exemple en 2 vantaux : $\mathbf{L/2 - 90,5\text{ mm}}$).*
- Pour vitrage 29 à 32 mm (**T141009**) :
  $$\text{Débit } T141009 = \text{Débit traverse basse } T141021 + 0,5\text{ mm}$$
  *(Exemple en 2 vantaux : $\mathbf{L/2 - 90,5\text{ mm}}$).*

Positionnement obligatoire de la traverse intermédiaire par rapport à la poignée $Hp$ :
- Fermeture sans clé : $ht_1 < Hp - 148\text{ mm}$ ou $ht_1 > Hp + 103\text{ mm}$.
- Fermeture avec verrouillage à clé : $ht_1 < Hp - 258\text{ mm}$ ou $ht_1 > Hp + 103\text{ mm}$.

---

# Cotes de vitrage et remplissage

Le jeu de fond de feuillure est de 14 mm sur les 4 côtés de chaque panneau verrier.

### 1. Hauteur de vitrage
$$\mathbf{H_{\text{vitrage}} = H - 179\text{ mm}}$$

En cas de traverse intermédiaire :
$$\mathbf{h_1 = H - ht_2 - 108\text{ mm}} \quad \text{et} \quad \mathbf{h_2 = H - ht_1 - 108\text{ mm}}$$

### 2. Largeur de vitrage selon typologie

| Typologie de menuiserie | Formule de débit de la largeur de vitrage |
| --- | --- |
| **2 vantaux sur 2 rails** | $\mathbf{L_v = L/2 - 91,5\text{ mm}}$ |
| **3 vantaux indépendants (2 rails)** | $\mathbf{L_v = L/3 - 61,5\text{ mm}}$ |
| **4 vantaux sur 2 rails (percussion)** | $\mathbf{L_v = L/4 - 73,5\text{ mm}}$ |
| **3 vantaux sur 3 rails** | $\mathbf{L_v = L/3 - 61,5\text{ mm}}$ |
| **6 vantaux sur 3 rails (percussion)** | $\mathbf{L_v = L/6 - 50\text{ mm}}$ |
| **4 vantaux sur 4 rails** | $\mathbf{L_v = L/4 - 46\text{ mm}}$ |
| **8 vantaux sur 4 rails (percussion)** | $\mathbf{L_v = L/8 - 38\text{ mm}}$ |
| **Galandage 1 vantail** | $\mathbf{L_v = L - 73,5\text{ mm}}$ |
| **Galandage 2 vantaux percussion** | $\mathbf{L_v = L/2 - 38\text{ mm}}$ |
| **Galandage 2 vantaux sur 2 rails** | $\mathbf{L_v = L/2 - 38\text{ mm}}$ |
| **Châssis composé avec frappe FY** | $\mathbf{H_v = Hx - 209\text{ mm}} \quad ; \quad \mathbf{L_v = Lx/2 - 106,5\text{ mm}}$ |

---

# Cotes de débit des rails de roulement (Aluminium et Inox)

### 1. Chemins de roulement aluminium rapportés (Réf. T341000)
- 2 vantaux 2 rails : 2 rails de débit $= \mathbf{L - 95\text{ mm}}$.
- 3 vantaux indépendants (2 rails) : 2 rails de débit $= \mathbf{L - 95\text{ mm}}$.
- 4 vantaux 2 rails : 1 rail de débit $= \mathbf{L - 95\text{ mm}}$ + 2 rails de débit $= \mathbf{L/2 - 63\text{ mm}}$.
- 3 vantaux 3 rails : 3 rails de débit $= \mathbf{L - 95\text{ mm}}$.
- 6 vantaux 3 rails : 2 rails de débit $= \mathbf{L - 95\text{ mm}}$ + 2 rails de débit $= \mathbf{L/2 - 63\text{ mm}}$.
- 4 vantaux 4 rails : 4 rails de débit $= \mathbf{L - 95\text{ mm}}$.
- 8 vantaux 4 rails : 3 rails de débit $= \mathbf{L - 95\text{ mm}}$ + 2 rails de débit $= \mathbf{L/2 - 63\text{ mm}}$.
- Galandage 1 vantail : 1 rail de débit $= \mathbf{2L - 116\text{ mm}}$.
- Galandage 2 vantaux percussion : 2 rails de débit $= \mathbf{L - 47,5\text{ mm}}$.
- Galandage 2 vantaux 2 rails : 2 rails de débit $= \mathbf{3L/2 - 47\text{ mm}}$.

### 2. Chemins de roulement inox rapportés (Réf. TGY4007)
- Ensemble des coulissants traditionnels (2 vtx, 3 vtx indép./dép., 4 vtx 2R, 3 vtx 3R, 6 vtx 3R, 4 vtx 4R, 8 vtx 4R) : débit $= \mathbf{L - 80\text{ mm}}$.
- Composé intégration frappe FY : débit $= \mathbf{L - 110\text{ mm}}$.
- Galandage 1 vantail : débit $= \mathbf{2L - 101\text{ mm}}$.
- Galandage 2 vantaux percussion : débit $= \mathbf{L - 32,5\text{ mm}}$.
- Galandage 2 vantaux 2 rails : débit $= \mathbf{3L/2 - 32\text{ mm}}$.

---

# Cotes de débit des boucliers thermiques PVC (T821000, T431025, TGY4006)

### 1. Bouclier thermique intermédiaire T821000
- Verticaux montants : 2 unités de débit $= \mathbf{H - 95\text{ mm}}$ (châssis 2 rails) ; 4 unités de débit $= \mathbf{H - 95\text{ mm}}$ (3 rails) ; 6 unités de débit $= \mathbf{H - 95\text{ mm}}$ (4 rails).
- Horizontal traverse haute : 1 unité de débit $= \mathbf{L - 37\text{ mm}}$ (2 rails) ; 2 unités de débit $= \mathbf{L - 37\text{ mm}}$ (3 rails) ; 3 unités de débit $= \mathbf{L - 37\text{ mm}}$ (4 rails).
- Galandage :
  - 1 vantail : 2x $(H - 95)$ et 1x $(2L - 57)$.
  - 2 vantaux percussion : 2x $(H - 95)$ et 1x $(2L - 6)$.
  - 2 vantaux 2 rails : 2x $(H - 95)$ et 1x $(3L/2 + 12)$.

### 2. Bouclier PVC supérieur TGY4006 (traverse haute et montants)
- Montants verticaux : débit $= \mathbf{H - 47\text{ mm}}$ (2 unités en 2 rails, 4 en 3 rails, 6 en 4 rails).
- Traverses horizontales :
  - 2 vantaux : 4 unités de débit $= \mathbf{L/2 - 67,5\text{ mm}}$.
  - 3 vantaux dépendants : 1x $(L/3 - 73)$ et 1x $(2L/3 - 61,5)$.
  - 3 vantaux indépendants : 4x $(L/3 - 37,5)$ et 2x $(L/3 - 132)$.
  - 4 vantaux 2 rails : 4x $(L/4 - 49,5)$ et 2x $(L/2 - 108,5)$.
  - 3 vantaux 3 rails : 4x $(L/3 - 37,5)$ et 4x $(2L/3 - 97)$.
  - 6 vantaux 3 rails : 4x $(L/3 - 73)$, 2x $(L/3 - 61)$, 4x $(L/6 - 26)$, 2x $(2L/3 - 156)$.
  - 4 vantaux 4 rails : 4x $(L/4 - 23)$, 4x $(3L/4 - 111,5)$, 4x $(L/2 - 67,5)$.
  - 8 vantaux 4 rails : 4x $(L/8 - 14)$, 2x $(3L/4 - 179,5)$, 4x $(L/4 - 49,5)$, 2x $(L/2 - 108,5)$, 4x $(3L/8 - 85)$, 2x $(L/4 - 37,5)$.

---

# Cotes de débit pour seuil PMR (TGY2100)

- Profilé seuil plat d'accès PMR **TGY2100** : débit $= \mathbf{L - 12\text{ mm}}$ (coupe droite 90°).
- Clip de maintien sur pièce d'appui **TGY3608** : quantité $= \mathbf{1 + L/500}$, vissé à l'aide de vis fournies hors tubulures.
- Joint d'étanchéité entre rails bas **TGY2531** :
  - 2 vantaux : 2 unités de débit $= \mathbf{L/2 - 67,5\text{ mm}}$.
  - 3 vantaux dépendants : 1x $(L/3 - 73)$ et 1x $(2L/3 - 61,5)$.
  - 3 vantaux indépendants : 2x $(L/3 - 37,5)$ et 1x $(L/3 - 132)$.
  - 4 vantaux 2 rails : 2x $(L/4 - 49,5)$ et 1x $(L/2 - 108,5)$.

---

# Prises de cotes et débits spécifiques aux châssis d'angle à 90°

Pour les coulissants d'angle sans poteau, les cotes $L_{1d}$ et $L_{2d}$ sont mesurées depuis l'appui de la tapée jusqu'à l'arrière du dormant. Les cotes fictives de calcul $L_1$ et $L_2$ valent :

### 1. Angle sortant
$$\mathbf{L_1 = L_{1d} - 43,7\text{ mm}} \quad \text{et} \quad \mathbf{L_2 = L_{2d} - 43,7\text{ mm}}$$
- Pour 3 rails : déduction $= - 94,9\text{ mm}$.
- Pour 4 rails : déduction $= - 146,1\text{ mm}$.
- Seuil PMR angle sortant : débit $= \mathbf{(L_1 - 12) + 208,5\text{ mm}}$ (coupes 45°-90°).

### 2. Angle rentrant
$$\mathbf{L_1 = L_{1d} + 62,5\text{ mm}} \quad \text{et} \quad \mathbf{L_2 = L_{2d} + 62,5\text{ mm}}$$
- Pour 3 rails : addition $= + 113,7\text{ mm}$.
- Pour 4 rails : addition $= + 164,9\text{ mm}$.
- Seuil PMR angle rentrant : débit $= \mathbf{(L_1 - 12) - 68,5\text{ mm}}$ (coupes 45°-90°).

---

# Voir aussi

- [Profilés dormants et rails SOLEAL GY 55](/profiles/soleal-gy-dormants-et-rails.md)
- [Roulements et fermetures SOLEAL GY 55](/quincaillerie/soleal-gy-roulements-et-fermetures.md)
- [Pose et galandage SOLEAL GY 55](/procedures/pose-soleal-gy-galandage.md)
- [Fournisseur Technal](/fournisseurs/technal.md)
