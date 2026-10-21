---
type: Quincaillerie
title: 'Roto NX : bornes catalogue 2023 (CDR 3), force de traction, dimensionnement, tolérance'
description: Bornes du catalogue de juin 2023 avec la classe CDR 3, force de traction par poids de vantail (60 à 150 kg), systèmes d'axe de ferrage et cotes de rainure, tolérance de châssis fixe, fixation d'une fenêtre de sécurité.
tags: [roto, roto-nx, abaque, champ-application, poids-vantail, cdr, rc2, designo, soufflet, cintre, tbdk, chiffrage]
systeme: Roto NX
fournisseur: ROTO
usage: [chiffrage, atelier]
famille: champs-application
status: draft
sources:
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
    last_modified: 2023-06-30
source_pages:
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    pages: 9, 21-34, 36-37
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    pages: 14, 34-52, 278
generated:
  by: process:claude-code
  at: 2026-10-21T11:04:21Z
---

Bornes complémentaires, forces de traction et cotes de dimensionnement de la ferrure Roto NX, du manuel de montage KSR et du catalogue Roto NX pour profils PVC. Les champs d'application sont sur [Champs d'application Roto NX : cotes côté paumelles P et ferrure OB rectangulaire](/quincaillerie/roto-nx-champs-application.md).

# Le catalogue de juin 2023 donne d'autres bornes, et une classe de plus

Le [catalogue Roto NX pour profils PVC](/sources/roto-nx-catalogue-pvc.md) de juin 2023 reprend
les mêmes champs d'application, côté paumelles P, oscillo-battant rectangulaire, avec des bornes
qui ne sont pas celles du manuel de montage de novembre 2022.

| Version | Classe de sécurité | LFF mini (mm) | LFF maxi (mm) | HFF mini (mm) | HFF maxi (mm) | PV maxi (kg) |
| --- | --- | --- | --- | --- | --- | --- |
| 130 kg | Sécurité de base | 290 | 1600 | 280 | 2800 | 130 |
| 130 kg | CDR 1 N | 320 | 1600 | 280 | 2800 | 130 |
| 130 kg | CDR 2 et CDR 2 N | 320 | 1400 | 510 | 2800 | 130 |
| 130 kg | **CDR 3** | 490 | 1400 | 600 | 2800 | 130 |
| 150 kg | Sécurité de base | 290 | 1600 | 280 | 2800 | 150 |
| 150 kg | CDR 1 N | 320 | 1600 | 280 | 2800 | 150 |
| 150 kg | CDR 2 et CDR 2 N | 320 | 1400 | 510 | 2800 | 150 |
| 150 kg | **CDR 3** | 320 | 1400 | 510 | 2800 | 150 |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 35 et 36)

La classe CDR 3 ne figure que dans le catalogue ; le manuel de montage s'arrête à CDR 2 / CDR 2 N.

Les champs d'application configuration par configuration (chapitre 3 du même catalogue, p. 57 à
135) ne reprennent pas toujours ces bornes : HFF minimale de 600, 520 ou 490 mm en CDR 2 / CDR 2 N
selon la crémone et le type de ferrure (entrée **INC-280**), bornes CDR 3 à une seule colonne
490 – 1 400 mm (entrée **INC-281**) — voir [Aperçu des ferrures Roto NX, côté paumelles
P](/quincaillerie/roto-nx-apercu-ferrures-cote-p.md).

**Les deux documents ROTO ne donnent pas les mêmes bornes** sur quatre valeurs — hauteur de
feuillure minimale, largeur maximale en CDR 1 N, hauteur maximale en CDR 1 N et en CDR 2. Entrée
**CTR-18** du registre [Contradictions entre sources](/anomalies/contradictions-entre-sources.md) ;
en attendant l'arbitrage, la valeur retenue pour un chiffrage est la borne la plus basse des deux
documents.

Le tableau de la version 150 kg est **imprimé en allemand** dans ce catalogue français, et il y
désigne les classes par « RC » là où les pages françaises écrivent « CDR » — entrée **INC-13** du
registre [Incohérences internes](/anomalies/incoherences-internes.md).

# Cotes de force de traction par poids de vantail

Pour la fixation des pièces de ferrure porteuses affectant la sécurité, telles que palier de compas
et palier d'angle, les forces de traction doivent être orientées perpendiculairement au plan de
l'ouvrant selon le tableau ci-dessous (valeurs des forces de traction en fonction des poids de
vantail par le TBDK) [1 p. 21-22]. La *force de traction* est l'effort, en newtons (N), que la
fixation doit supporter sans s'arracher.

| Poids du vantail (kg) | Force de traction (N) |
| --- | --- |
| 60 | 1 650 |
| 70 | 1 900 |
| 80 | 2 200 |
| 90 | 2 450 |
| 100 | 2 700 |
| 110 | 3 000 |
| 120 | 3 250 |
| 130 | 3 500 |
| 140 | 3 900 |
| 150 | 4 200 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 21-22)

Les valeurs indiquées sont données par référence au palier de compas. Elles sont également valables
pour les paliers d'angle lorsque la fixation est réalisée selon le palier de compas. **Respecter la
directive TBDK pour les forces de traction en fonction du poids de vantail** ; autres informations
sur www.beschlagindustrie.de [1 p. 22].

Le catalogue Roto NX pour profils PVC de juin 2023 donne les mêmes dix valeurs (colonnes « Poids
d'ouvrant » et « Force de traction en Newton (N) »), présentées comme un extrait de la directive
TBDK de la Gütegemeinschaft Schlösser und Beschläge e. V., avec une obligation formulée
autrement : **lors de la fixation de pièces de ferrure porteuses essentielles en matière de
sécurité (côtés paumelles), le fabricant de fenêtres et de portes-fenêtres doit certifier les
forces spécifiées conformément au tableau au moyen d'essais et les garantir sur son produit**.
L'encadré « Respecter la directive TBDK pour les forces de traction en fonction du poids de
vantail ! Autres informations sur www.beschlagindustrie.de » y est répété sous chaque diagramme
d'application [2 p. 34-47]. La TBDK borne aussi les poids d'ouvrant admissibles du système profine, voir
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

# Cotes de dimensionnement des profilés

Le dessin désigne les éléments d'une fenêtre, en coupe horizontale au droit de la ferrure : le
dormant [5] en rose saumon, l'ouvrant [6] en gris clair, la parclose [7], la rainure de vantail [8]
(la rainure du profilé d'ouvrant où se loge la ferrure), et quatre cotes : la cote de l'axe [1], le
jeu en feuillure [2] (l'espace entre ouvrant et dormant), la largeur de recouvrement [3] (la largeur
dont l'ouvrant recouvre le dormant) et la hauteur de recouvrement [4] [1 p. 34].

![Désignations de l'élément de fenêtre et rainure de vantail](/assets/quincaillerie/roto-nx-ksr/champs-application/designations-element-de-fenetre.png)

Systèmes d'axe de ferrage admis par la Roto NX, en mm. La désignation se lit « jeu de joint /
largeur de recouvrement - axe de ferrage » [1 p. 34].

| Système | Axe de ferrage [1] (mm) | Jeu de joint [2] (mm) | Largeur de recouvrement [3] (mm) |
| --- | --- | --- | --- |
| 12/18-9 | 9 | 12 | 18 |
| 12/18-13 | 13 | 12 | 18 |
| 12/20-9 | 9 | 12 | 20 |
| 12/20-13 | 13 | 12 | 20 |
| 12/21-13 | 13 | 12 | 21 |
| 12/22-13 | 13 | 12 | 22 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 34)

Le jeu de joint est écrit « 12 mm - 0,5 mm / + 1,5 mm » dans une cellule commune aux six systèmes
[1 p. 34].

Le catalogue Roto NX pour profils PVC de juin 2023 (section 1.1.5 « Système – PVC ») donne les
mêmes six systèmes avec les mêmes cotes : jeu de feuillure [A] 12 mm pour les six, largeur de
recouvrement [B] 18, 18, 20, 20, 21 et 22 mm, axe de ferrure [C] 9, 13, 9, 13, 13 et 13 mm. Le jeu y
est écrit « 12 mm », sans tolérance. Sur son dessin, l'ouvrant est en gris, le dormant en rose
saumon ; [A] et [B] sont cotés sous le dessin, [C] verticalement à sa gauche [2 p. 14].

![Système PVC : jeu de feuillure A, largeur de recouvrement B, axe de ferrure C](/assets/quincaillerie/roto-nx-catalogue/informations/systeme-pvc-jeu-recouvrement-axe.png)

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 14)

Sous le dessin d'ensemble, trois détails agrandis donnent les cotes recommandées pour le
dimensionnement des profilés [1 p. 34] :

| Détail | Cotes portées (mm) |
| --- | --- |
| Rainure, premier détail | 16,2 (+0,2 / −0,1) ; 12,2 (+0,2 / −0,1) ; 9,3 ±0,2 ; 2,2 ±0,2 |
| Rainure, second détail | 16,2 (+0,2 / −0,1) ; 12,2 (+0,2 / −0,1) ; 9,3 ±0,2 ; 2,2 ±0,2 ; 2,8 |
| Feuillure du dormant, détail de la ferrure | angle ≤ 5° ; 3 (+0,5) |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 34)

Sur les deux détails de rainure, 16,2 et 12,2 sont cotées horizontalement, 9,3, 2,2 et 2,8
verticalement. Le manuel ne nomme pas les deux variantes de rainure.

# Cotes de tolérance de châssis fixe, côté paumelles P

L'*encombrement de la paumelle* est la place que prennent les paumelles (charnières) entre
l'ouvrant et le dormant. Tolérance de châssis fixe pour 20 mm de largeur de recouvrement, en mm ;
[A] est la tolérance de châssis fixe, [B] la hauteur de recouvrement, [C] et [D] les jeux en haut et
en bas du vantail [1 p. 36].

![Encombrement de la paumelle, côté paumelles P](/assets/quincaillerie/roto-nx-ksr/champs-application/encombrement-paumelle-cote-p.png)

| Poids du vantail (kg) | Angle d'ouverture | Tolérance de châssis fixe [A] (mm) | Hauteur de recouvrement mini [B] (mm) | En haut [C] (mm) | En bas [D] (mm) |
| --- | --- | --- | --- | --- | --- |
| 130 | env. 180° | 21,0 | 16 | 1,0 | 8 |
| 150 | env. 180° | 26,5 | 16 | 1,0 | 8 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 36)

**Tolérance, caches compris.** Angle d'ouverture jusqu'à 20 mm de hauteur de recouvrement
[1 p. 36]. L'angle « env. 180° » porte un renvoi « [2] » dont la note n'est imprimée nulle part sur
la page (entrée **INC-193**).

Les deux vues de face (poids de vantail 130 kg et 150 kg) montrent le vantail gris sur le dormant
rose saumon, paumelles à droite : la cote horizontale portée au droit de la paumelle haute vaut
21 mm en 130 kg et 26,5 mm en 150 kg, celle portée au droit de la paumelle basse 19 mm dans les
deux cas, et une cote verticale de 8 mm est portée sous le vantail [1 p. 36].

Le catalogue Roto NX PVC de juin 2023 donne la même tolérance avec, pour 150 kg, un angle
d'ouverture d'environ 150° au lieu de 180°, et appelle [C] l'espace extérieur de 100 mm (entrée
**CTR-118**) ; ses tableaux et vues sont sur
[Compas, paliers et pivots Roto NX](/quincaillerie/roto-nx-compas-et-paliers.md) [2 p. 278].

# Fixation d'une fenêtre de sécurité

La *fenêtre de sécurité* est une fenêtre classée retard d'effraction (classes CDR). Le manuel
suggère sa fixation dans la maçonnerie par des blocs d'écartement [1 p. 37].

![Suggestion de fixation pour fenêtre de sécurité](/assets/quincaillerie/roto-nx-ksr/champs-application/fixation-fenetre-de-securite.png)

Le dessin montre la fenêtre vue de face dans l'ouvrage de maçonnerie [1], le dormant [3] en rose
saumon et les blocs d'écartement [2] en noir, entre maçonnerie et dormant.

| Élément | Valeur |
| --- | --- |
| Dispositif | bloc d'écartement [2], entre l'ouvrage de maçonnerie [1] et le dormant [3] |
| Emplacement | installer un bloc d'écartement dans la zone des vissages de gâche de sécurité |
| Distance aux angles (montants et traverses) | ~150 mm |
| Entraxe entre blocs sur les montants | ~400 mm |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 37)

La consigne est imprimée « dans la zone des vissages de gâche de sécurités de sécurité ». Le texte
qui suit est reproduit tel quel [1 p. 37] :

> « Les fenêtre à retard d'effraction au sens de la DIN EN 1627–1630 ne doivent être désignées
> comme telles qu'uniquement lorsque le montage a été effectué en tous points selon la norme
> prescrite. »

# Ce que la source ne donne pas

- **quel côté paumelles PROFERM emploie** — P ou Designo II. Entrée **VER-34** du registre
  [Informations à vérifier](/anomalies/informations-a-verifier.md)
- **le poids des profilés** : le manuel ne donne que la conversion de l'épaisseur de verre en
  poids de vitrage

# Citations

[1] Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf.
IMO_180_NX_FR_v2, novembre 2022 —
`raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf`, p. 9, 21 à 37 (numérotation du PDF)

[2] Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023 —
`raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, p. 14, 34 à 52 et 278

# Voir aussi

Cette page est une partie d'un sujet qui en compte 6 ; les autres :

- [Champs d'application Roto NX : cotes côté paumelles P et ferrure OB rectangulaire](/quincaillerie/roto-nx-champs-application.md)
- [Champs d'application Roto NX : ferrure OB de fenêtre inclinée et de fenêtre cintrée](/quincaillerie/roto-nx-champs-application-ob-inclinee-cintree.md)
- [Champs d'application Roto NX : ferrure soufflet, compas d'arrêt et cotes d'ouverture](/quincaillerie/roto-nx-champs-application-soufflet.md)
- [Champs d'application Roto NX : paumelle à soufflet sur recouvrement et fenêtre confort](/quincaillerie/roto-nx-champs-application-paumelle-soufflet-confort.md)
- [Champs d'application Roto NX côté paumelles Designo II : 80, 100 et 150 kg](/quincaillerie/roto-nx-champs-application-designo.md)
