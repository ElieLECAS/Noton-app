---
type: Profilé
title: Cotes de débit du système 70
description: Les formules de coupe et les cotes à déduire de la dimension hors tout pour débiter dormants, meneaux, ouvrants, battements et seuils du système 70 Plateforme profine.
tags: [profine, systeme-70, debit, cote, coupe, dormant, ouvrant, meneau, atelier]
systeme: 70
famille: cotes-de-debit
status: stable
sources:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
    last_modified: 2023-09-30
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    id: profine-plans-e-volution-2008
    title: Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008
    last_modified: 2008-08-31
source_pages:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    pages: 104-119
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 109-119
generated:
  by: process:multimodal-direct
  at: 2026-09-19T20:45:00Z
---

# Cotes de débit du système 70 Plateforme

Les cotes de débit définissent la longueur exacte de coupe des barres PVC avant soudage ou assemblage mécanique [1 p. 104].
Toutes les cotes sont exprimées en millimètres (mm) à partir des dimensions hors tout (DHT en largeur, HHT en hauteur).
La surcote d'extrusion et de fusion au miroir chauffant est fixée à **+3 mm par coupe d'onglet soudée** (soit +6 mm par barre pour deux soudures) [1 p. 104].

# Cotes de débit des dormants (coupe d'onglet 45°)

Pour un cadre dormant assemblé par soudure à 45° dans les quatre angles :

| Profilé dormant | Largeur vue (mm) | Formule de coupe en largeur (mm) | Formule de coupe en hauteur (mm) | Type de coupe |
| --- | --- | --- | --- | --- |
| Dormant standard 6100 | 70 | $\text{DHT} + 6$ | $\text{HHT} + 6$ | 45° / 45° |
| Dormant rénovation 6102 (aile 40 mm) | 70 | $\text{DHT} + 6$ | $\text{HHT} + 6$ | 45° / 45° |
| Dormant rénovation 6105 (aile 60 mm) | 70 | $\text{DHT} + 6$ | $\text{HHT} + 6$ | 45° / 45° |
| Dormant élargi 6108 (96 mm) | 96 | $\text{DHT} + 6$ | $\text{HHT} + 6$ | 45° / 45° |
| Dormant élargi 6155 (115 mm) | 115 | $\text{DHT} + 6$ | $\text{HHT} + 6$ | 45° / 45° |
| Dormant monobloc 6158 (155 mm) | 155 | $\text{DHT} + 6$ | $\text{HHT} + 6$ | 45° / 45° |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 106-110)

# Cotes de débit des ouvrants (coupe d'onglet 45°)

Les cotes de débit des ouvrants dépendent du jeu en feuillure de **12 mm** et du profilé dormant associé.

### Ouvrant standard 70 mm (6112 / 6113) sur dormant standard 6100

| Type de fenêtre | Formule largeur ouvrant (mm) | Formule hauteur ouvrant (mm) | Déduction totale par rapport au hors tout |
| --- | --- | --- | --- |
| 1 vantail OF ou OB | $\text{DHT} - 86 + 6$ | $\text{HHT} - 86 + 6$ | $-86\text{ mm}$ en L et H |
| 2 vantaux OF / OB (vantail principal) | $(\text{DHT} - 92)/2 + 6$ | $\text{HHT} - 86 + 6$ | $-86\text{ mm}$ en H, axe centré |
| 2 vantaux OF / OB (semi-fixe) | $(\text{DHT} - 92)/2 + 6$ | $\text{HHT} - 86 + 6$ | $-86\text{ mm}$ en H, axe centré |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 111-113)

### Ouvrant KÖMMERLING e.VOLUTION 78 mm (6121 / 6123)

En raison de la profondeur de 78 mm et du recouvrement spécifique de cet ouvrant :
- Formule largeur 1 vantail : $\text{DHT} - 94 + 6\text{ mm}$.
- Formule hauteur 1 vantail : $\text{HHT} - 94 + 6\text{ mm}$.
- Formule 2 vantaux : largeur vantail = $(\text{DHT} - 100)/2 + 6\text{ mm}$.

# Cotes de débit des meneaux et traverses (coupe droite 90°)

Le meneau et la traverse intermédiaire (profilés 6127 et 2425) sont coupés droits à 90° et assemblés mécaniquement par set d'assemblage ou soudés à plat.

| Profilé meneau / traverse | Mode d'assemblage | Formule de débit en traverse (mm) | Formule de débit en montant (mm) |
| --- | --- | --- | --- |
| Traverse 6127 sur dormant 6100 | Assemblage mécanique avec set 9718.3 | $\text{LFF dormant} - 2\text{ mm}$ (fond de feuillure) | $\text{HFF dormant} - 2\text{ mm}$ |
| Meneau 6127 sur dormant 6100 | Coupe droite avec grugeage d'embout | $\text{DHT} - 140\text{ mm}$ | $\text{HHT} - 140\text{ mm}$ |
| Meneau lourd 2425 sur dormant 6100 | Assemblage avec connecteur acier | $\text{DHT} - 140\text{ mm}$ | $\text{HHT} - 140\text{ mm}$ |
| Traverse d'ouvrant 6127 | Soudure à plat dans ouvrant 6112 | $\text{Largeur fond de feuillure ouvrant} + 6\text{ mm}$ | — |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 112)

# Cotes de débit des battements centraux rapportés

Le battement central se coupe droit à 90° et s'installe sur le montant vertical du semi-fixe face au vantail principal.

| Profilé battement | Formule de coupe en hauteur (mm) | Jeu d'extrémité haut et bas | N° d'embout de battement requis |
| --- | --- | --- | --- |
| Battement design 6132 | $\text{Hauteur coupe ouvrant} - 46\text{ mm}$ | $23\text{ mm}$ en haut / $23\text{ mm}$ en bas | Embout M850 |
| Battement standard 6130 | $\text{Hauteur coupe ouvrant} - 46\text{ mm}$ | $23\text{ mm}$ en haut / $23\text{ mm}$ en bas | Embout M851 |
| Battement tubulaire 0140 | $\text{Hauteur coupe ouvrant} - 48\text{ mm}$ | $24\text{ mm}$ en haut / $24\text{ mm}$ en bas | Embout 9B52 |
| Battement étroit 0141 | $\text{Hauteur coupe ouvrant} - 44\text{ mm}$ | $22\text{ mm}$ en haut / $22\text{ mm}$ en bas | Embout 9B53 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 114-119)

# Cotes de débit des seuils aluminium et rejets d'eau

Pour les portes-fenêtres avec traverse basse en seuil aluminium plat PMR (seuil 9C42 ou A076) :
- **Longueur de coupe du seuil alu** : $\text{Largeur fond de feuillure dormant (LFF)} + 32\text{ mm}$ (pour encastrement sous montant dormant) ou $\text{DHT} - 140\text{ mm}$ en pose entre dormants avec pièces d'étanchéité d'embout [1 p. 118].
- **Rejet d'eau d'ouvrant alu (9F46)** : $\text{Largeur d'ouvrant} - 54\text{ mm}$ (dégagement de 27 mm de chaque côté pour passage des gâches et compas).

# Cotes de débit des renforts acier

Les renforts acier intérieurs sont coupés droits à 90° avec un retrait obligatoire par rapport à l'angle pour permettre le passage de la tête de soudure et le drainage :
- **Renfort de dormant (V600 à V618)** : $\text{Longueur de coupe PVC} - 90\text{ mm}$ (retrait de 45 mm à chaque extrémité) [1 p. 105].
- **Renfort d'ouvrant (V600 à V618)** : $\text{Longueur de coupe PVC} - 70\text{ mm}$ (retrait de 35 mm à chaque extrémité) [1 p. 105].
- **Renfort de meneau / traverse** : $\text{Longueur du profilé PVC} - 10\text{ mm}$ (retrait de 5 mm sous embout de fixation).

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 — `raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.1, p. 104 à 119
[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 — `raw/profine-plans-profiles-e-volution-2008-08.pdf`, p. 109 à 119

# Voir aussi

- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Assemblages du système 70](/profiles/systeme-70-assemblages.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Fabrication des profilés PVC](/procedures/fabrication-profiles-pvc.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
