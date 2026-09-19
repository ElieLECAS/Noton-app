---
type: Profilé
title: Statique et moments d'inertie du système 70
description: Règles de calcul statique de flèche au vent, pressions admissibles (400 à 1 600 Pa) et tables des moments d'inertie Iz requis et effectifs pour profilés et renforts du système 70 Plateforme.
tags: [profine, systeme-70, statique, inertie, iz, vent, aev, fleche, renfort]
systeme: 70
famille: statique
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
    pages: 147-180
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 138-192
generated:
  by: process:multimodal-direct
  at: 2026-09-19T20:50:00Z
---

# Statique et moments d'inertie du système 70 Plateforme

Les règles de calcul statique garantissent la résistance mécanique des menuiseries sous les efforts de pression
et de dépression du vent (norme NF EN 12210 / NF DTU 36.5) [1 p. 147].
Elles fixent le moment d'inertie minimal $I_z$ (en $\text{cm}^4$) que doit posséder le profilé renforcé (meneau, dormant ou combinaison)
pour que la flèche frontale ne compromette ni l'étanchéité ni l'intégrité du vitrage isolant.

# Critères réglementaires de déformation admissible (flèche $f$)

Selon la nature de la baie et du vitrage :
- **Flèche standard $f \le L / 200$** : règle générale des menuiseries avec vitrage isolant double (flèche limitée au 1/200 de la portée, avec maximum absolu de **15 mm** pour $L > 3\text{ m}$) [2 p. 138].
- **Flèche de confort $f \le L / 300$** : exigée pour les grands ensembles vitrés, triples vitrages lourds ou vitrages feuilletés de sécurité (maximum absolu de **8 mm** sur la hauteur de vitrage pour éviter le descellement des joints butyl/polysulfure) [2 p. 138].
- **Flèche limite admissible en rénovation $f \le L / 150$** : admise sous vent extrême instantané sans rupture [1 p. 148].

# Moments d'inertie $I_z$ requis par pression de vent et dimensions

Moments d'inertie requis pour un meneau vertical intermédiaire reprenant la demi-largeur de chaque vantail adjacent ($L/2 + L/2$), relevés sur les tables du registre 2.3.4 (p. 166-180) pour un critère de flèche $f \le L / 200$ :

### Classe de vent V\*A2 (pression de service 800 Pa)

| Hauteur de baie $H$ (mm) | Largeur de reprise $L = 1\,200\text{ mm}$ ($I_z\text{ cm}^4$) | Largeur $L = 1\,600\text{ mm}$ ($I_z\text{ cm}^4$) | Largeur $L = 2\,000\text{ mm}$ ($I_z\text{ cm}^4$) | Largeur $L = 2\,400\text{ mm}$ ($I_z\text{ cm}^4$) |
| --- | --- | --- | --- | --- |
| 1 400 | 2,15 | 2,87 | 3,58 | 4,30 |
| 1 600 | 3,36 | 4,48 | 5,60 | 6,72 |
| 1 800 | 4,96 | 6,61 | 8,26 | 9,91 |
| 2 000 | 7,02 | 9,36 | 11,70 | 14,04 |
| 2 200 | 9,61 | 12,81 | 16,02 | 19,22 |
| 2 400 | 12,81 | 17,08 | 21,35 | 25,62 |
| 2 600 | 16,70 | 22,27 | 27,84 | 33,40 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 168-172)

### Classe de vent V\*A3 (pression de service 1 200 Pa)

| Hauteur de baie $H$ (mm) | Largeur de reprise $L = 1\,200\text{ mm}$ ($I_z\text{ cm}^4$) | Largeur $L = 1\,600\text{ mm}$ ($I_z\text{ cm}^4$) | Largeur $L = 2\,000\text{ mm}$ ($I_z\text{ cm}^4$) | Largeur $L = 2\,400\text{ mm}$ ($I_z\text{ cm}^4$) |
| --- | --- | --- | --- | --- |
| 1 400 | 3,23 | 4,30 | 5,38 | 6,45 |
| 1 600 | 5,04 | 6,72 | 8,40 | 10,08 |
| 1 800 | 7,44 | 9,92 | 12,39 | 14,87 |
| 2 000 | 10,53 | 14,04 | 17,55 | 21,06 |
| 2 200 | 14,42 | 19,22 | 24,03 | 28,83 |
| 2 400 | 19,22 | 25,62 | 32,03 | 38,43 |
| 2 600 | 25,05 | 33,40 | 41,75 | 50,11 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 173-177)

# Moments d'inertie effectifs des profilés et renforts acier du Système 70

L'inertie totale d'un ensemble résulte de la somme de l'inertie du profilé PVC nu ($I_{\text{PVC}}$) pondérée par le ratio des modules d'élasticité ($E_{\text{acier}} / E_{\text{PVC}} \approx 70$) et de l'inertie propre du renfort acier ($I_{\text{acier}}$) :
$$I_{\text{total}} = I_{\text{acier}} + \frac{I_{\text{PVC}}}{70}$$

Valeurs des moments d'inertie propres des principaux renforts acier du système 70 Plateforme :

| Référence renfort | Épaisseur acier (mm) | Forme de section | $I_x$ frontal ($\text{cm}^4$) | $I_z$ latéral ($\text{cm}^4$) | Profilés associés |
| --- | --- | --- | --- | --- | --- |
| V600 | 1,5 | U standard | 1,45 | 1,95 | Dormant standard 6100 |
| V601 | 2,0 | U renforcé | 1,85 | 2,50 | Dormant standard 6100 |
| V602 | 1,5 | Tube rectangulaire | 3,80 | 4,95 | Meneau standard 6127 |
| V603 | 2,0 | Tube rectangulaire lourd | 4,90 | 6,40 | Meneau standard 6127 |
| V604 | 1,5 | U à retour | 1,20 | 1,75 | Ouvrant standard 6112 |
| V605 | 2,0 | U à retour lourd | 1,55 | 2,25 | Ouvrant standard 6112 |
| V608 | 2,0 | Tube carré lourd | 8,20 | 8,20 | Poteau d'angle 90° 70603 |
| V610 | 2,5 | Tube spécial haute inertie | 12,40 | 18,60 | Meneau lourd 2425 |
| V615 | 2,0 | Fer plat épais | 0,45 | 14,20 | Renfort statique extérieur de meneau |
| V618 | 2,5 | I à double retour | 6,10 | 28,40 | Meneau de grande baie vitrée |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 152-160)

# Règle de vérification statique à l'atelier

Avant fabrication d'un ensemble de menuiserie à meneau ou accouplement :
1. Déterminer la zone de vent du chantier selon NV 65 / Eurocode (Zone 1 à 5, rugosité de site 1 à 4).
2. Calculer la pression dynamique de référence $q_p$ (ex. 800 Pa pour V\*A2 en zone 2 normale).
3. Lire sur la table la valeur de $I_z\text{ requis}$ pour la hauteur $H$ et la largeur reprise $L$.
4. Vérifier que $I_{\text{total}}$ du meneau avec son renfort acier est supérieur ou égal à $I_z\text{ requis}$.
5. Si $I_{\text{total}} < I_z\text{ requis}$, remplacer le renfort 1,5 mm par un renfort 2,0 ou 2,5 mm, ou adjoindre un profilé d'inertie extérieur (fer plat V615 ou poteau de renfort).

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 — `raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4, p. 147 à 180
[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 — `raw/profine-plans-profiles-e-volution-2008-08.pdf`, p. 138 à 192

# Voir aussi

- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Cotes de débit du système 70](/profiles/systeme-70-cotes-de-debit.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [Classification de la résistance au vent](/reference/classification-resistance-au-vent.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
