---
type: Référence
title: Glossaire des sigles et des cotes
description: Les sigles, coefficients et repères de cote employés dans la documentation PROFERM et chez ses fournisseurs, avec leur sens et la page qui les utilise.
tags: [glossaire, sigle, abreviation, cote, coefficient, vocabulaire]
status: stable
sources:
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
    last_modified: 2023-01-31
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
    last_modified: 2023-06-30
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
    last_modified: 2025-06-19
generated:
  by: process:claude-code
  at: 2026-09-18T22:00:00Z
---

# Coefficients thermiques

Quatre coefficients circulent dans le corpus et ne se comparent pas entre eux : chacun porte sur
un objet différent. Plus la valeur est basse, plus l'objet est isolant.

| Sigle | Porte sur | Unité |
| --- | --- | --- |
| Ug | le **vitrage** seul | W/m²K |
| Uf | le **profilé** seul | W/m²K |
| Uw | la **fenêtre complète**, vitrage et menuiserie | W/m²K |
| Up | le **panneau** de porte | W/m²K |

Deux coefficients accompagnent le Uw sur les fiches produit :

| Sigle | Sens |
| --- | --- |
| Sw | facteur solaire de la fenêtre : part de l'énergie solaire transmise |
| TLw | transmission lumineuse de la fenêtre |
| Gtot(i) | facteur solaire de l'ensemble vitrage et store, selon NF EN 14501 |

Une valeur annoncée sans son sigle ne se reprend pas : un 1,0 de Uf n'est pas un 1,0 de Uw. Voir
[Performances des vitrages](/vitrages/performances-vitrages.md).

# Cotes de fabrication

Vocabulaire fixé par les [directives générales profine](/sources/profine-directives-generales.md)
et employé dans tous les manuels du groupe.

| Sigle | Sens |
| --- | --- |
| DHT | dimension hors tout, c'est-à-dire la dimension extérieure du dormant |
| DEO | dimension extérieure d'ouvrant |
| DFO | dimension de feuillure d'ouvrant |
| CCD | cote clair de dormant |
| CCO | cote clair d'ouvrant |
| CCV | cote clair de vitrage |

Les quatre premières se déduisent en cascade, chaque tableau de cotes de débit donnant la valeur
à retrancher pour **une seule coupe**. Voir
[Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md).

Repères de cote des planches profine, du dormant vers le vitrage :

| Repère | Notion | Repère | Notion |
| --- | --- | --- | --- |
| A | Dormant | J | Hauteur joint comprimé |
| B | Ouvrant | K | Recouvrement ouvrant |
| C | Parclose | L | Hauteur de calage vitrage |
| D | Rainure crémone | M | Hauteur pré-cale |
| E | Dos de dormant | N | Jeu de fonctionnement |
| F | Chambre des renforts | O | Dimension extérieure dormant |
| G | Rainure parclose | P | Clair de jour vitrage |
| H | Feuillure vitrage | Q | Clair de jour ouvrant |
| I | Feuillure quincaillerie | R | Clair de jour dormant |

# Cotes de quincaillerie

Vocabulaire des catalogues et manuels [ROTO](/fournisseurs/roto.md).

| Sigle | Sens |
| --- | --- |
| LFF | largeur de fond de feuillure d'ouvrant |
| HFF | hauteur de fond de feuillure d'ouvrant |
| PV | poids de vantail |
| FFO | hauteur d'axe de poignée au fond de la feuillure quincaillerie |
| SDB | sécurité de base, le champ d'application sans classe d'effraction |
| KSR | basculement vertical, désignation de la famille de ferrure Roto NX KSR |

**Les cotes LFF et HFF sont des cotes de feuillure de vantail**, ni des cotes de baie ni des
cotes extérieures d'ouvrant : les trois séries de limites s'appliquent en même temps et chacune
peut être la contraignante. Voir
[Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md).

# Types d'ouverture

| Sigle | Sens |
| --- | --- |
| OF | ouvrant à la française |
| OB | ouvrant oscillo-battant |

**OF ne signifie pas « ouvrant fixe ».** Tout le corpus, DTA compris, emploie OF pour l'ouvrant à
la française, y compris dans les libellés de dimensions maximales du type « 2 vantaux OF ».

# Statique et renforts

| Sigle | Sens | Unité |
| --- | --- | --- |
| IW | inertie du renfort dans la direction du vent ; borne la dimension réalisable sous une charge de vent donnée | cm⁴ |
| IG | inertie du renfort vis-à-vis du poids ; borne l'épaisseur de vitrage admissible | cm⁴ |
| Iz | notation du moment d'inertie employée par les classeurs profine antérieurs, équivalente à IW sur les planches relevées | cm⁴ |
| TBDK | directive allemande de la Gütegemeinschaft Schlösser und Beschläge, qui fixe les forces de traction à certifier selon le poids de vantail | - |
| WPK | contrôle de production en usine, au titre duquel le fabricant garantit le poids d'ouvrant | - |

Un renfort à forte IW et faible IG tient le vent mais pas le triple vitrage. Voir
[Renforts du système 76](/profiles/systeme-76-renforts.md).

# Classements de performance

Le classement **A\*E\*V** du CSTB mesure trois résistances, chacune notée séparément : A pour
l'air, E pour l'eau, V pour le vent.

| Notation | Sens |
| --- | --- |
| A\*n | perméabilité à l'air, de A\*1 à A\*4, A\*4 étant la plus étanche |
| E\*nA | étanchéité à l'eau, version A ; le nombre croît avec la performance |
| V\*Ln | résistance au vent : la lettre A, B ou C donne la classe de déformation, le chiffre la pression |

Voir [Labels et certifications](/certifications/labels-et-certifications.md) pour le classement
de chaque produit.

# Résistance à l'effraction

| Notation | Sens |
| --- | --- |
| RC 1 à RC 6 | classes de résistance à l'effraction de la série NF EN 1627 à 1630 |
| CDR 1 à CDR 6 | même classification, notation employée par les documents ROTO en français |

**RC et CDR désignent la même chose.** Le catalogue Roto NX emploie les deux, en français et en
allemand, dans le même document — entrée **INC-13** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

Le vitrage feuilleté a sa propre échelle, **P1 A à P5 A** selon la norme EN 356, mesurée par la
hauteur de chute et le nombre de billes. Voir
[Panneaux et monoblocs](/portes/panneaux-et-monoblocs.md).

# Matières et traitements

| Terme | Sens |
| --- | --- |
| PVB | PolyVinylButyral, film intercalaire d'un vitrage feuilleté |
| STADIP | désignation commerciale d'un vitrage feuilleté de sécurité |
| TGI | type d'intercalaire à bord chaud d'un vitrage isolant ; le vitrage de série PERFORM76 en emploie un de coloris noir |
| EPDM | élastomère des joints d'étanchéité |
| PCE | second élastomère de joint employé par le système 70 |
| RPT | rupture de pont thermique |
| Plaxage | application d'un film décor sur le profilé PVC |
| Laquage | application d'une laque, classée par le label QUALICOAT |
| Contretypé | teinte réalisée sur mesure pour s'approcher d'une texture donnée |
| Grain d'orge | finition de soudure d'angle, par opposition à la soudure ébavurée |
| L\* | clarté colorimétrique ; le seuil **L\* inférieur à 82** déclenche le renforcement et la décompression des profilés |

# Labels de traitement de surface

| Label | Porte sur |
| --- | --- |
| QUALICOAT | laquage de l'aluminium ; la classe 2 garantit une tenue supérieure à la classe 1 |
| QUALANOD | anodisation de l'aluminium |
| QUALIMARINE | préparation de surface de l'aluminium laqué en ambiance marine |
| CEKAL | certification des vitrages isolants |

# Documents et organismes

| Sigle | Sens |
| --- | --- |
| DTA | Document Technique d'Application, avis du CSTB sur un procédé |
| DTD | Dossier Technique Détaillé, pièce jointe au DTA qui porte les prescriptions de fabrication |
| GS | Groupe Spécialisé du CSTB ; le n° 6 traite les menuiseries |
| CSTB | Centre Scientifique et Technique du Bâtiment |
| FFCP | Fédération Française de Construction Passive |
| UFME | Union des Fabricants de Menuiseries Extérieures |
| SNEP | Syndicat National de l'Extrusion Plastique |
| PMR | personne à mobilité réduite ; qualifie un seuil surbaissé |

Les références internes des documents ROTO suivent trois préfixes : **IMO** pour une instruction
de montage, **CTL** pour un catalogue, **SUG** pour une notice d'emploi.

# Normes citées dans le corpus

| Norme | Objet |
| --- | --- |
| NF DTU 36.5 | mise en œuvre des fenêtres et portes extérieures |
| NF EN 12207 | classement de la perméabilité à l'air |
| NF EN 1627 à 1630 | résistance à l'effraction des fenêtres et portes |
| EN 356 | résistance du vitrage feuilleté au choc |
| NF EN 14501 | classement de la protection solaire des stores |
| DIN EN 13126/8 | ferrures de fenêtre oscillo-battante, dont la protection anticorrosion |
| EN 1670 | résistance à la corrosion de la quincaillerie |
| NF P 20-302 | caractéristiques des fenêtres, conformité mécanique |
| NF P20-650-1 | mise en œuvre des vitrages |
| NF P24-351 | protection contre la corrosion des menuiseries métalliques |
| Règles NV 65 | charges de neige et de vent, référentiel antérieur à l'Eurocode NF EN 1991-1-4 |

# Sigles non élucidés

| Sigle | Où | Entrée |
| --- | --- | --- |
| DV | Catalogue général, p. 17, qualifie un Uw de fenêtre | **VER-35** |
| CV | Catalogue général, p. 17, qualifie un Uw de coulissant | **VER-35** |

Aucun document du corpus ne définit ces deux sigles. Voir
[Informations à vérifier](/anomalies/informations-a-verifier.md).

# Citations

[1] Directives générales profine, version janvier 2023 —
`raw/profine-directives-generales-2023-01.pdf`, registre 1.1.2

[2] Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023 —
`raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, p. 10 à 13

[3] DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED —
`raw/dta-trocal-76-advanced-6-16-2334-v5.pdf`, p. 4 à 9

# Voir aussi

- [Directives générales profine](/sources/profine-directives-generales.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md)
- [Labels et certifications](/certifications/labels-et-certifications.md)
- [Performances des vitrages](/vitrages/performances-vitrages.md)
