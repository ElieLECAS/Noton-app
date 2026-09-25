---
type: Procédure
title: Redressement d'un montant de porte cintré par le profilé 9708
description: Préconisation profine pour redresser un montant de porte déformé au moyen du profilé acier précontraint 9708, sans dégonder l'ouvrant.
tags: [profine, 9708, porte, montant-cintre, renfort, sav, atelier, chantier]
fournisseur: KÖMMERLING
usage: sav
status: stable
sources:
  - resource: raw/profine-mise-en-oeuvre-9708-montants-cintres-2017-07.pdf
    id: profine-9708-montants-cintres
    title: Mise en œuvre 9708, profine France, juillet 2017
    last_modified: 2017-07-06
generated:
  by: process:claude-code
  at: 2026-09-18T10:00:00Z
---

# À quoi sert cette procédure

Le **profilé acier précontraint 9708** de [profine](/fournisseurs/profine.md) sert au
**rattrapage d'un montant de porte cintré** : un montant d'ouvrant qui s'est déformé et que l'on
redresse en lui opposant une contrainte, sans remplacer la porte.

C'est une intervention de **service après-vente**, réalisable en atelier comme sur chantier :
l'ouvrant **n'est pas dégondé**.

# La condition qui interdit l'intervention

> « Cette mise en œuvre **n'est pas compatible avec des traverses intermédiaires horizontales** »
> (mise en œuvre 9708, profine France, juillet 2017).

C'est la première chose à vérifier. Un ouvrant à traverse intermédiaire ne se redresse pas par ce
procédé, et la préconisation n'en donne aucun autre.

# Les six étapes

| Étape | Opération |
| --- | --- |
| 1 | Intervenir **sans dégonder l'ouvrant** |
| 2 | Déposer le vitrage ou le panneau |
| 3 | Retirer les vis de renfort en feuillure du montant cintré, **sauf une**, de préférence la première en haut de l'ouvrant, pour empêcher le renfort de glisser |
| 4 | Couper le profilé 9708 à dimension, **10 à 15 mm plus court que la longueur de parclose**, et le placer dans la feuillure intérieure du montant. **Le cintre du montant et celui du renfort plat doivent être en opposition**, jamais dans le même sens |
| 5 | Caler entre le profilé acier et l'intérieur de la rainure à parclose, **à ± 5 mm**, puis visser avec des **vis autoforantes 3,9 × 28** |
| 6 | Sur un ouvrant vitré, **coller des cales de 1 mm sur le profilé acier** pour assurer sa mise en place |

# Où caler, selon le sens du cintre

C'est le point qui décide du résultat : le calage ne se fait pas au même endroit selon le sens de
la déformation.

| Cas | Sens du cintre | Position du calage, côté rainure à parcloses |
| --- | --- | --- |
| n° 1 | porte cintrée vers l'**extérieur** | **au centre** du renfort |
| n° 2 | porte cintrée vers l'**intérieur** | **aux extrémités** du renfort |

**Inverser les deux cas aggrave la déformation** au lieu de la corriger.

L'épaisseur de calage s'augmente ou se diminue à volonté pour redresser le montant, en visant
**1 mm de contrainte supplémentaire**, contrôlé **à la règle de 2 m** (mise en œuvre 9708,
profine France, juillet 2017).

# Cotes et caractéristiques du profilé 9708

Relevées sur les schémas cotés et la coupe de détail `X 1:1` de la fiche technique :

| Grandeur | Valeur |
| --- | --- |
| Longueur nominale du profilé précontraint | **2 500 mm** |
| Flèche de précontrainte initiale au centre | **40 mm** |
| Largeur d'extrémité / appui | **25 mm** |
| Section transversale (coupe X 1:1) | profilé en U de **largeur 25 mm × hauteur 9 mm** (profondeur de rainure 6 mm) |
| Longueur de coupe d'atelier | longueur de parclose − 10 à 15 mm |
| Jeu de calage entre profilé et rainure à parclose | ± 5 mm |
| Contrainte visée après redressement | 1 mm (contrôlé à la règle de 2 m) |
| Vis de fixation | autoforante 3,9 × 28 mm |
| Cales de maintien sur ouvrant vitré | 1 mm (collées directement sur l'acier) |

**Cette vis est plus longue que celle des renforts de fabrication**, qui est une 3,9 × 16 mm dans
le système 76 Advanced — voir [Renforts du système 76](/profiles/systeme-76-renforts.md).

# Ce que le document ne précise pas

- **sur quelles gammes il s'emploie prioritairement** : le document profine est universel pour montants de porte PVC sans traverse intermédiaire (système 70 ou 76 Advanced)
- **la flèche résiduelle maximale admissible du montant avant intervention** : seule la contrainte de 1 mm sous règle de 2 m est fixée comme objectif de redressage

Entrée **VER-27** du registre [Informations à vérifier](/anomalies/informations-a-verifier.md), dont la longueur et la section du profilé sont désormais entièrement résolues.

# Provenance

Note d'une page éditée par **profine France**, Marmoutier, datée du **6 juillet 2017**. Ce n'est
pas un registre du classeur de fabrication : c'est une fiche isolée, la plus ancienne source
profine du wiki.

# Citations

[1] Mise en œuvre 9708, profine France, juillet 2017 —
`raw/profine-mise-en-oeuvre-9708-montants-cintres-2017-07.pdf`, p. 1

# Voir aussi

- [profine](/fournisseurs/profine.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Ouvrants de porte d'entrée](/portes/ouvrants-de-porte.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
