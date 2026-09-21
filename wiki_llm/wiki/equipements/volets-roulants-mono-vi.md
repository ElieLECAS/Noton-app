---
type: Équipement
title: Volet roulant inversé bloc-baie MONO VI
description: Volet roulant bloc-baie SOPROFEN à enroulement extérieur et trappe d'accès affleurante intérieure — coffre compact de 140 mm à clair de jour maximisé, compatible rénovation et ITE, Uc = 0,9 + 0,11/Lc.
tags: [equipement, volet-roulant, mono-vi, volet-inverse, soprofen, compact, clair-de-jour, ite, renovation]
status: stable
sources:
  - resource: raw/moustiquaires/export_doc_132.zip
    id: soprofen-guide-technique-blocs-baies-2022
    title: Guide technique Blocs-baies SOPROFEN 2022, réf. DOC86505
    last_modified: 2022-09-01
source_pages:
  - resource: raw/moustiquaires/export_doc_132.zip
    pages: 4, 124-135
generated:
  by: process:gemini-coder
  at: 2026-09-21T14:20:00Z
---

# Présentation du système MONO VI

Le **MONO VI** (« VI » pour *Volet Inversé*) de [SOPROFEN](/fournisseurs/soprofen.md) est une fermeture bloc-baie caractérisée par un **enroulement extérieur du tablier** associé à une trappe de visite intérieure affleurante et discrète [1 p. 4, 124] :
* **Clair de jour maximal** : son coffre compact de **140 mm de hauteur seulement** libère l'ouverture vitrée, permettant d'équiper des fenêtres et portes-fenêtres jusqu'à **2 360 mm de hauteur sous coffre** [1 p. 4, 124].
* **Architecture inversée** : le tablier descend au nu extérieur de la fenêtre tandis que le mécanisme reste accessible depuis l'intérieur du logement sans dégrader l'étanchéité ni l'esthétique de façade [1 p. 124].
* **Domaines d'emploi privilégiés** : particulièrement adapté aux chantiers de **rénovation** (pose sur dormant existant de 60 ou 70 mm) ainsi qu'aux façades avec **Isolation Thermique par l'Extérieur (ITE)** [1 p. 4, 124].

---

# Performances thermo-aérauliques et acoustiques

* **Isolation thermique renforcée en standard** : formule d'évaluation du coefficient de transmission surfacique du caisson :
  $$U_c = 0,9 + \frac{0,11}{L_c}\text{ W/(m}^2\text{.K)}$$
  où $L_c$ est la longueur de coffre en mètres, assurant une parfaite conformité avec la réglementation thermique élément par élément ($U_c \le 2,5\text{ W/m}^2\text{K}$) [1 p. 4].
* **Isolation phonique** : affaiblissement acoustique certifié atteignant jusqu'à **$D_{n,e,\text{tr}} = 50\text{ dB}$** grâce à l'adjonction en usine de la bande de masse lourde isolante **VX58/9** en face interne de la trappe [1 p. 4, 135].

---

# Dimensions de fabrication et limites dimensionnelles

La cote hors-tout en hauteur répond à la formule : $H = H_{\text{châssis}} + 140\text{ mm}$ [1 p. 126].

### Limites dimensionnelles selon la typologie de lame

| Type de lame | Matériau | Pas (mm) | Largeur max simple tablier (mm) | Largeur max multi-tabliers (mm) | Hauteur max coffre compris (mm) | Poids tablier (kg/m²) |
| --- | --- | --- | --- | --- | --- | --- |
| LA 37 | Aluminium standard | 37 | 2 500 | 3 000 | 2 500 | 2,80 |
| LATH 37 | Aluminium thermique | 37 | 2 500 | 3 000 | 2 500 | 2,80 |
| L 37 | PVC extrudé | 37 | 1 500 | Non disponible | 2 500 | 3,20 |

(schéma: raw/moustiquaires/export_doc_132.zip, p. 125 et 126)

*Règle multi-tabliers* : en configuration 2 ou 3 tabliers, la largeur totale cumulée ne peut excéder **3 000 mm** (avec tabliers obligatoirement de même hauteur et de même type de lame) [1 p. 127].

---

# Adaptation sur dormants de menuiserie

Le MONO VI se raccorde mécaniquement sur la traverse haute des menuiseries au moyen d'adaptateurs clipsés et vissés en PVC [1 p. 124, 128] :
* **Sur dormant de 60 mm** : embout de trappe spécifique avec **aile de recouvrement de 10 mm** d'épaisseur [1 p. 128].
* **Sur dormant de 70 mm** : embout de trappe spécifique avec **aile de recouvrement de 18 mm** d'épaisseur [1 p. 128].
* **Appui inférieur** : pour garantir l'étanchéité et le verrouillage complet, la lame finale doit reposer sur un rejingot d'appui maçonné ou une pièce d'appui menuisée horizontale sous les coulisses [1 p. 124].

---

# Motorisations et commandes

Le MONO VI intègre l'ensemble des solutions de manœuvre [SOMFY](/fournisseurs/somfy.md) et SOPROFEN M-Soft² [1 p. 129-135] :

### Tableau des largeurs minimales ($L_{\min}$)

| Mode de manœuvre | Modèle | Code | Manœuvre de secours | Largeur mini $L_{\min}$ 1 tablier (mm) | Largeur mini multi-tabliers (mm) |
| --- | --- | --- | --- | --- | --- |
| Radio RTS | Oximo court RTS (larg max 799 mm) | R1 | Non | 550 | 550 |
| Radio RTS | Oximo RTS standard | R1 | Non | 800 | 800 |
| Radio RTS | Oximo court RTS + secours | R1S | Oui | 695 | - |
| Radio RTS | Oximo RTS + secours | R1S | Oui | 945 | - |
| Radio RTS | Oximo 40 WF solaire autonome | A35 | Non | 590 | - |
| Radio IO | Oximo court IO (larg max 659 mm) | IO1 | Non | 550 | 550 |
| Radio IO | S&SO RS100 IO standard | IO1 | Non | 660 | 660 |
| Radio IO | Oximo court IO + secours | IO1S | Oui | 695 | - |
| Radio M-SOFT² | M-Soft² MVM court (larg max 659 mm) | R11 | Non | 590 | 590 |
| Radio M-SOFT² | M-Soft² MVM standard | R11 | Non | 660 | 660 |
| Filaire WT | Ilmo court (larg max 699 mm) | M | Non | 550 | 550 |
| Filaire WT | Ilmo standard | M | Non | 700 | 700 |
| Filaire WT | Ilmo court + secours | MS | Oui | 695 | - |
| Filaire WT | Ilmo standard + secours | MS | Oui | 845 | 845 |
| Filaire WT | Ilmo 12 tours | M1 | Non | 700 | 700 |
| Filaire WT | Ilmo 12 tours + secours | M1S | Oui | 845 | - |
| Filaire M-SOFT² | M-Soft² MVE court (larg max 659 mm) | M11 | Non | 590 | 590 |
| Filaire M-SOFT² | M-Soft² MVE standard | M11 | Non | 660 | 660 |
| Hybride filaire/radio | S&SO RS100 IO Hybrid | IOH | Non | 660 | 660 |
| Hybride filaire/radio | S&SO RS100 IO Hybrid + secours | IOHS | Oui | 805 | - |
| Treuil manuel | Treuil enrouleur avec tige $\varnothing 12\text{ mm}$ | T | Non | 400 | - |

(schéma: raw/moustiquaires/export_doc_132.zip, p. 129, 133)

---

# Options et finitions

1. **Surlongueur de trappe** : en manœuvre manuelle par treuil ou secours motorisé, une surlongueur de trappe de **40 mm minimum** (débord de 61 mm avec l'embout) est obligatoire pour le passage du mécanisme [1 p. 127, 133].
2. **Mortaises VMC** : possibilité d'usinage en traverse supérieure pour grilles de ventilation Anjos VM15/22/30, E2A ou Aldes EHA [1 p. 128].
3. **Isolation acoustique renforcée** : complexe lourd VX58/9 en face interne de trappe [1 p. 135].

---

# Citations

[1] [Guide technique Blocs-baies SOPROFEN 2022, réf. DOC86505](raw/moustiquaires/export_doc_132.zip), p. 4, 124 à 135

---

# Voir aussi

- [SOPROFEN](/fournisseurs/soprofen.md)
- [SOMFY](/fournisseurs/somfy.md)
- [Volets roulants](/equipements/volets-roulants.md)
- [Volet roulant et BSO bloc-baie CHRONO PSE²](/equipements/volets-roulants-chrono-pse2.md)
- [Volet roulant coffre demi-linteau BLOC LX](/equipements/volet-roulant-bloc-lx-demi-linteau.md)
- [Guide technique Blocs-baies SOPROFEN 2022](/sources/soprofen-guide-technique-blocs-baies-2022.md)
