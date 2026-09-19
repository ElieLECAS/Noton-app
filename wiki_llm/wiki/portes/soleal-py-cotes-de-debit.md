---
type: Cotes de débit
title: Cotes de débit et formules de calcul des portes Technal SOLEAL PY 55
description: Recueil exhaustif des formules de débit, déductions de profilés (dormants, ouvrants en T/Z, plinthes, traverses FPI, seuils PMR PY1100), dimensionnement des vitrages et formules de coupe des tringles de crémone pour la porte SOLEAL PY 55 de TECHNAL.
tags: [technal, soleal, soleal-py, porte, debit, formules, seuil-pmr, vitrage, tringles]
gamme: LUMINE
systeme: 55
status: stable
sources:
  - resource: wiki_llm/a_faire/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf
    id: technal-soleal-py-55-conception-4944-006
    title: SOLEAL PY 55 — Catalogue de conception Porte à rupture de pont thermique (Réf. 4944.006)
    last_modified: 2018-09-01
  - resource: wiki_llm/a_faire/SOLEAL-PY-55-catalogue-fabrication-4899-007-092018-FR.pdf
    id: technal-soleal-py-55-fabrication-4899-007
    title: SOLEAL PY 55 — Guide d'atelier et catalogue de fabrication (Réf. 4899.007)
    last_modified: 2018-09-01
  - resource: wiki_llm/a_faire/SEUIL TECHNAL PY1100 DOC.pdf
    id: technal-seuil-py1100-doc
    title: F.I.T N° 15 — Évolution Porte PY et Seuil PMR PY1100
    last_modified: 2012-12-01
generated:
  by: process:multimodal-direct
  at: 2026-09-19T23:00:00Z
---

# Formules de débit des cadres dormants

### 1. Cadre dormant 3 côtés (sans traverse basse en aluminium)
Utilisé avec seuil rapporté en applique ou encastré (seuil PMR PY1100, seuil bâtiment T525060, seuil plat T525053) :
- **Montants verticaux (2 unités)** :
  $$\mathbf{H_{\text{dormant}} = H - 6\text{ mm}} \quad (\text{coupe d'onglet 45° en haut, coupe droite 90° en bas})$$
  *(Pour seuil résidentiel TPY1102 / TPY2107 : débit $= \mathbf{H - 12\text{ mm}}$).*
  *(Pour seuil TPY2104 sur dormant sans rainure TPY1102 : débit $= \mathbf{H - 7\text{ mm}}$).*
- **Traverse haute (1 unité)** :
  $$\mathbf{L_{\text{dormant}} = L} \quad (\text{coupe d'onglet 45° aux 2 extrémités})$$

### 2. Cadre dormant 4 côtés (périphérique)
- **Montants verticaux (2 unités)** : débit $= \mathbf{H}$ à 45°/45°.
- **Traverses haute et basse (2 unités)** : débit $= \mathbf{L}$ à 45°/45°.

### 3. Cadre avec Ferme-Porte Intégré (FPI)
- **Traverse haute FPI (Réf. T225205)** : débit $= \mathbf{L - 54\text{ mm}}$ (coupe droite 90°/90°).
- **Montants verticaux multi-usage (Réf. T215056)** : débit $= \mathbf{H}$ ou $\mathbf{H - 7\text{ mm}}$.
- **Profilés de battée et composition** :
  - Profilé battée dormant T525054 : 2x $(H - 75)$ et 1x $(L - 79)$.
  - Profilé composition symétrique T525055 : 2x $(H - 75)$ et 1x $(L - 104)$.
  - Profilé cache-rainure PVC T525057 : 2x $(H - 137,5)$ et 1x $(L - 79)$.

---

# Formules de débit des profilés ouvrants

### 1. Porte 1 vantail Simple Action — Ouvrant 3 côtés avec plinthe 120 mm
Configuration classique ouverture extérieure (T225114) ou intérieure (T225104) avec seuil PMR bâtiment T525060 ou PY1100 :
- **Montants d'ouvrant verticaux (2 unités)** :
  $$\mathbf{H_{\text{ouvrant}} = H - 50,5\text{ mm}} \quad (\text{coupe 90° en bas, coupe 45° en haut})$$
  *(En version seuil TPY2104 / T525053 : débit $= \mathbf{H - 57,5\text{ mm}}$).*
  *(En version FPI traverse haute : débit $= \mathbf{H - 98,5\text{ mm}}$).*
- **Traverse haute d'ouvrant (1 unité)** :
  $$\mathbf{L_{\text{ouvrant}} = L - 70\text{ mm}} \quad (\text{coupe d'onglet 45°/45°})$$
- **Plinthe basse 120 mm (Réf. T700047)** :
  $$\mathbf{L_{\text{plinthe}} = L - 220\text{ mm}} \quad (\text{coupe droite 90°/90°})$$
- **Profilé porte-brosse (Réf. T525051)** : débit $= \mathbf{L - 220\text{ mm}}$.
- **Profilé rejet d'eau clippé (Réf. T525052)** : débit $= \mathbf{L - 72\text{ mm}}$.
- **Seuil PMR (Réf. T525060 ou PY1100)** : débit $= \mathbf{L - 120\text{ mm}}$.
- **Barre de seuil d'écartement (Réf. PY4000)** : débit $= \mathbf{L - 180\text{ mm}}$.

### 2. Porte 1 vantail Simple Action — Ouvrant 4 côtés (périphérique)
Configuration avec ouvrant en Z (T225104) et seuil PMR RPT PY1100 :
- **Montants d'ouvrant verticaux (2 unités)** :
  $$\mathbf{H_{\text{ouvrant}} = H - 52,5\text{ mm}} \quad (\text{coupe 45°/45°})$$
  *(En version seuil résidentiel TPY2107 : débit $= \mathbf{H - 48\text{ mm}}$).*
- **Traverses haute et basse d'ouvrant (2 unités)** :
  $$\mathbf{L_{\text{ouvrant}} = L - 70\text{ mm}} \quad (\text{coupe 45°/45°})$$
- **Rejet d'eau ouvrant (Réf. TPY2102)** : débit $= \mathbf{L - 120\text{ mm}}$.
- **Joint d'ouvrant pour seuil PMR (Réf. TPY5000)** : débit $= \mathbf{L - 95\text{ mm}}$.

### 3. Porte 2 vantaux Simple Action (Vantail de service + Vantail semi-fixe)
- **Montants d'ouvrant verticaux (4 unités)** :
  $$\mathbf{H_{\text{ouvrant}} = H - 50,5\text{ mm}} \quad (\text{avec plinthe})$$
  $$\mathbf{H_{\text{ouvrant}} = H - 52,5\text{ mm}} \quad (\text{ouvrant périphérique 4 côtés})$$
- **Traverses d'ouvrants hautes (2 unités)** :
  $$\mathbf{L_{\text{ouvrant}} = L/2 - 26,5\text{ mm}} \quad (\text{coupe 45°/45°})$$
- **Plinthes basses T700047 et porte-brosse T525051 (2 unités)** :
  $$\mathbf{L_{\text{plinthe}} = L/2 - 176,5\text{ mm}} \quad (\text{coupe droite 90°/90°})$$
- **Rejet d'eau clippé T525052** :
  - Vantail de service : débit $= \mathbf{L/2 - 28,5\text{ mm}}$.
  - Vantail semi-fixe : débit $= \mathbf{L/2 - 53,5\text{ mm}}$.
- **Parcloses de plinthe basse T591005 (4 unités)** : débit $= \mathbf{L/2 - 176,5\text{ mm}}$.

### 4. Porte Tube Anti-Pince Doigts (APD) avec FPI
- **Montant tube cylindrique côté rotation (Réf. T225107)** : débit $= \mathbf{H - 118\text{ mm}}$.
- **Montant ouvrant va-et-vient côté fermeture (Réf. T225144)** : débit $= \mathbf{H - 118\text{ mm}}$.
- **Traverse haute d'ouvrant va-et-vient (Réf. T700050)** :
  - 1 vantail : débit $= \mathbf{L - 171,5\text{ mm}}$.
  - 2 vantaux : débit $= \mathbf{L/2 - 120,5\text{ mm}}$.
- **Plinthe basse T700047 (1 vtl)** : débit $= \mathbf{L - 234\text{ mm}}$ ; (2 vtx) : débit $= \mathbf{L/2 - 183\text{ mm}}$.
- **Profilés anti-pince doigts rapportés (Réf. TPY2505)** : débit $= \mathbf{H - 95\text{ mm}}$.

### 5. Porte Va-et-Vient (VV) sur Crapaudine et Seuil Plat
- **Montants d'ouvrant T225144** : débit $= \mathbf{H - 112\text{ mm}}$ à 45°/45°.
- **Traverses d'ouvrant T225144 (1 vtl)** : débit $= \mathbf{L - 95\text{ mm}}$ ; (2 vtx) : débit $= \mathbf{L/2 - 51,5\text{ mm}}$.

---

# Cotes de dimensionnement des vitrages

Le jeu de fond de feuillure standard est de **19 mm** en haut et sur les montants verticaux.

### 1. Tableau récapitulatif des dimensions de vitrage

| Typologie de porte | Hauteur de vitrage $H_v$ | Largeur de vitrage $L_v$ |
| --- | --- | --- |
| **Porte SA 1 vtl avec plinthe (seuil bâtiment)** | $\mathbf{H - 233\text{ mm}}$ | $\mathbf{L - 232\text{ mm}}$ |
| **Porte SA 1 vtl avec plinthe (seuil PMR PY1100)** | $\mathbf{H - 214,5\text{ mm}}$ | $\mathbf{L - 232\text{ mm}}$ |
| **Porte SA 1 vtl avec plinthe et seuil TPY2104** | $\mathbf{H - 240\text{ mm}}$ | $\mathbf{L - 232\text{ mm}}$ |
| **Porte SA 1 vtl ouvrant périphérique (seuil PMR)**| $\mathbf{H - 215\text{ mm}}$ | $\mathbf{L - 232\text{ mm}}$ |
| **Porte SA 1 vtl avec seuil résidentiel TPY2107** | $\mathbf{H - 210\text{ mm}}$ | $\mathbf{L - 232\text{ mm}}$ |
| **Porte SA 2 vtx avec plinthe (seuil bâtiment/PMR)**| $\mathbf{H - 233\text{ mm}}$ | $\mathbf{L/2 - 189\text{ mm}}$ |
| **Porte SA 2 vtx avec plinthe et seuil TPY2104** | $\mathbf{H - 240\text{ mm}}$ | $\mathbf{L/2 - 189\text{ mm}}$ |
| **Porte SA 2 vtx ouvrant périphérique (seuil PMR)**| $\mathbf{H - 215\text{ mm}}$ | $\mathbf{L/2 - 189\text{ mm}}$ |
| **Porte SA 1 vtl avec FPI et plinthe** | $\mathbf{H - 281\text{ mm}}$ | $\mathbf{L - 232\text{ mm}}$ |
| **Porte SA 2 vtx avec FPI et plinthe** | $\mathbf{H - 281\text{ mm}}$ | $\mathbf{L/2 - 189\text{ mm}}$ |
| **Porte Tube APD 1 vantail avec FPI** | $\mathbf{H - 288\text{ mm}}$ | $\mathbf{L - 246\text{ mm}}$ |
| **Porte Tube APD 2 vantaux avec FPI** | $\mathbf{H - 288\text{ mm}}$ | $\mathbf{L/2 - 195\text{ mm}}$ |
| **Porte Va-et-Vient (VV) 1 vantail avec FPI** | $\mathbf{H - 249\text{ mm}}$ | $\mathbf{L - 232\text{ mm}}$ |
| **Porte Va-et-Vient (VV) 2 vantaux avec FPI** | $\mathbf{H - 249\text{ mm}}$ | $\mathbf{L/2 - 189\text{ mm}}$ |

> [!CAUTION]
> **Règle impérative pour parcloses à clippage de face (T591221 / T591222 / T591223)** :
> Lors de l'utilisation de parcloses à pose de face avec clips **T770070**, il faut obligatoirement **diminuer la hauteur et la largeur du vitrage de 4 mm supplémentaires** par rapport aux cotes du tableau ci-dessus (pour compenser l'encombrement des nez de clips).

---

# Formules de débit des tringles de crémone (T525058)

Les tringles plates en aluminium **T525058** actionnent les pênes hauts et bas à partir du fouillot ou du barillet :

| Type de serrure | Configuration d'étanchéité basse | Tringle basse $T_1$ | Tringle haute $T_2$ |
| --- | --- | --- | --- |
| **Serrure 2 points à fouillot (T920004)** | Simple Action standard | $\mathbf{T_1 = Hp - 261\text{ mm}}$ | $\mathbf{T_2 = Hv - Hp - 210\text{ mm}}$ |
| **Serrure 2 points à cylindre (T920003)** | Avec gâche seuil PMR **TPY6000** | $\mathbf{T_1 = Hp - 276\text{ mm}}$ | $\mathbf{T_2 = Hv - Hp - 210\text{ mm}}$ |
| **Serrure 2 points** | Double Action (PSA / PDA) | $\mathbf{T_1 = Hp - 261\text{ mm}}$ | $\mathbf{T_2 = Hv - Hp - 197\text{ mm}}$ |
| **Serrure 3 points pênes verticaux (T920005/06)**| Simple Action standard | $\mathbf{T_1 = Hp - 261\text{ mm}}$ | $\mathbf{T_2 = Hv - Hp - 210\text{ mm}}$ |
| **Serrure 3 points pênes verticaux** | Avec gâche seuil PMR **TPY6000** | $\mathbf{T_1 = Hp - 276\text{ mm}}$ | $\mathbf{T_2 = Hv - Hp - 210\text{ mm}}$ |
| **Serrure 3 points modulaire (T920008)** | Pêne basculant à renvoi | $\mathbf{T_1 = Hp - 621\text{ mm}}$ | $\mathbf{T_2 = Hv - Hp - 554\text{ mm}}$ |
| **Serrure multipoints modulaire (grande hauteur)**| Renvoi haut/bas additionnel | $\mathbf{Tx_1 = \text{variable}}$ | $\mathbf{Ty_2 = 100\text{ mm mini}}$ |

*Où $Hp$ est la hauteur d'axe de la béquille/poignée par rapport au bas de l'ouvrant ($Hp_{\text{sol}} = Hp + \text{jeu} = 1\,050\text{ mm}$), et $Hv$ est la hauteur totale du vantail d'ouvrant.*

---

# Voir aussi

- [Profilés dormants et ouvrants SOLEAL PY 55](/portes/soleal-py-dormants-et-ouvrants.md)
- [Serrures et paumelles SOLEAL PY 55](/quincaillerie/soleal-py-serrures-et-paumelles.md)
- [Mise en œuvre du seuil PMR PY1100](/procedures/technal-seuil-pmr-py1100.md)
- [Fournisseur Technal](/fournisseurs/technal.md)
