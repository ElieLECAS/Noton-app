---
type: Procédure
title: Pose de la PERFORM76
description: Les quatre principes de pose de la PERFORM76 — neuf, dépose totale et deux versions rénovation — avec le recouvrement, le drainage et l'interdiction de la mousse polyuréthane.
tags: [perform76, pose, renovation, neuf, drainage, decompression, chantier]
status: stable
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
stale_after: 2027-09-02
---

# Interdiction absolue : la mousse polyuréthane

**« L'utilisation de la mousse polyuréthane est à proscrire dans l'ensemble des cas de pose »**
(cahier technique, p. 7, mention en rouge et en tête de page).

C'est la seule interdiction générale du cahier technique, et elle ne souffre aucune exception :
ni en neuf, ni en dépose, ni en rénovation. L'étanchéité se fait au compribande et au silicone,
le calage à la cale latérale ou au bois.

# Les quatre principes de pose

Chaque principe n'accepte que certains dormants. Choisir le dormant avant le principe, ou
l'inverse, mais vérifier la correspondance (cahier technique, p. 7 et 8).

| Principe | Dormants compatibles | Constituants |
| --- | --- | --- |
| Pose en neuf | 76180 (aile 20 mm), 76171 (sans aile) | compribande, silicone, équerre de fixation, complexe isolant, tapée suivant isolant, maçonnerie crépis fini |
| Dépose totale | 76171, 76172 (sans aile) | compribande, silicone, cale latérale |
| Rénovation version 1 | 76177 (aile 40 mm), 76185 (aile 60 mm) | compensation bois, fond de joint, silicone, cale latérale |
| Rénovation version 2 | 76177, 76185 | mise à niveau bois, fond de joint, silicone, cale latérale |

Voir [Dormants PERFORM76](/profiles/perform76-dormants.md).

Les deux versions de rénovation acceptent les mêmes dormants et diffèrent par le traitement du
bois existant : **compensation** en version 1, **mise à niveau** en version 2 (cahier technique,
p. 8).

# Cotes de recouvrement sur le mur

Recouvrement en pose neuve, en mm, relevé sur le cahier technique (p. 7).

| Configuration | Recouvrement sur le mur (mm) |
| --- | --- |
| Avec tapées | 35 |
| Sans tapées | 30 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 7)

Ces deux valeurs sont portées en rouge sur le schéma de pose en neuf : ce sont des minima de
recouvrement à respecter, pas des cotes indicatives.

# Cotes de drainage et de décompression

Usinages de drainage et de décompression, en mm, relevés sur le cahier technique (p. 4).

| Usinage | Cote | Emplacement |
| --- | --- | --- |
| Drainage par fraisage | 25 × 5 | dormant bas |
| Décompression par fraisage | 25 × 5 | dormant |
| Décompression par découpe de joint | 100 de joint par ouvrant | ouvrant |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 4)

Position du fraisage sur le dessus du dormant bas, qui dépend du profil :

| Type de profil | Dormants | Fraisage dessus dormant bas (mm) |
| --- | --- | --- |
| Neuf, sans aile ou aile de 20 mm | 76171, 76172, 76180 | 49 |
| Rénovation, aile de 40 ou 60 mm | 76177, 76185 | 43 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 4)

La décompression combine deux usinages distincts — un fraisage de 25 × 5 mm **et** une découpe de
100 mm de joint par ouvrant. Les deux sont nécessaires : le cahier les présente comme deux
opérations, pas comme une alternative.

# Délignage de l'aile, sur le chantier

Le délignage de l'aile des dormants rénovation est explicitement **« à effectuer sur le
chantier »** (cahier technique, p. 4, 11 et 12), pas en atelier.

| Dormant | Aile (mm) | Délignage maxi (mm) |
| --- | --- | --- |
| 76177 | 40 | 20 |
| 76185 | 60 | 40 |

Dans les deux cas il reste 20 mm d'aile après délignage maximal.

# Pattes de pose et clameaux

La fixation en pose isolée passe par une patte de pose et un clameau, dont les références
dépendent de l'épaisseur d'isolant et du dormant. Deux configurations, sur les dormants neufs
uniquement :

| Dormant | Clameau | Cale |
| --- | --- | --- |
| 76180 | CP14GGOM0012 | sans cale |
| 76171 | CP14GGOM0012 | CTHNT0030, sauf à 80 mm d'isolant |

Le tableau complet des pattes NT1939 à NT1953 par épaisseur d'isolant est dans
[Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md) — avec la contrainte
majeure : **au-delà de 155 mm d'isolant sur un 76171, le dormant bas doit devenir un 76180**.

# Alignement à demander à la commande

L'alignement des vitrages par rapport au dessus des traverses est possible **à la demande du
client lors de la commande** (cahier technique, p. 9). Ce n'est pas un réglage de chantier : si
ce n'est pas demandé à la commande, ce n'est pas rattrapable à la pose. Voir
[Meneaux PERFORM76](/profiles/perform76-meneaux.md).

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 4, 7, 8, 9, 11 et 12

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md)
- [Meneaux PERFORM76](/profiles/perform76-meneaux.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
