# Génération : trois passes sur les mêmes questions (02/10/2026)

GLM 5.3, effort high, température 0,2, une conversation par question. Notes de Claude, une passe chacune : indicatif.

- **1re passe** : avant tout changement (`test_generation_10q_2026-10-02.md`).
- **Forme** : section « La forme de la réponse » + trois réponses types + règles 2, 5, 8, 10, 12 conditionnelles.
- **Forme + garde** : plus la règle de périmètre (rule 3) et la garde serveur « page citée non lue → renvoyer lire une fois ».

| Q | 1re passe (car.) | Forme (car.) | Forme + garde (car.) | Note finale | Constat |
|---|---|---|---|---|---|
| 1 | A · 830 | 488 | 628 | A | Valeur exacte, 25 % plus court qu'à la 1re passe. |
| 2 | B · 931 | 352 | 403 | B+ | 4 000 mm en premier ; la note sur les 6 m des 4 vantaux est conservée. |
| 3 | B · 2105 | 1198 | 933 | A- | Verdict « oui », calcul de l'ouvrant écrit, parclose, zone B. |
| 4 | A- · 1159 | 605 | 854 | A | 135 / 155 mm avec tapée, patte, appui, cale ; le 76180 pour 140 mm exact. |
| 5 | B- · 1610 | 984 | 967 | B+ | Les deux éditions et CTR-88 ; plus d'arbitrage « en vigueur ». |
| 6 | B+ · 2101 | 1277 | 1205 | B+ | Tableau comparatif juste ; deux réserves (VER-02, CTR-03) conservées. |
| 7 | B+ · 1834 | 912 | 1089 | A- | Extérieur oui, intérieur conditionnel (VER-49 non tranché) ; plus de bruit. |
| 8 | C+ · 1765 | 917 | 1166 | B | « 10 ans sur le fonctionnement » puis « le wiki ne détaille pas ce que cela couvre » : la déduction a disparu. Reste une phrase « pas une garantie pièce ou matériaux » et une note sur le certificat illisible. |
| 9 | B · 2029 | 999 | 1166 | B+ | Plages exactes (+2,5 / −2,0 mm ; Designo +2,0 / −1,0), report de charge ; « quelques dixièmes » a disparu ; toutes les pages citées sont lues. Reste le goujon V, spéculatif. |
| 10 | B+ · 1280 | 542 | 787 | B+ | Absence nette, aucun délai inventé ; toutes les citations lues, au prix de 6 appels (la garde a renvoyé lire) ; ajoute la liste des formes de cintrage, hors sujet. |
| 11 | — | 1337 | 1357 | A- | Q26 : « Non », calcul, courbes 132 / 112 cm, J079, double vitrage, parclose 76503. Pas indépendante (réponse type proche dans le prompt). |

| Mesure (10 questions) | 1re passe | Forme | Forme + garde |
|---|---|---|---|
| Longueur moyenne (caractères) | 1564 | 827 | 920 |
| Pages citées sans avoir été lues | 3 | 4 | 0 |
| Coût ($) | 0.477 | 0.494 | 0.618 |
| Appels (médiane) | 3,5 | 3 | 4 |

Contrôle dur sur les nombres et les références (réponse comparée aux pages lues) : 0 référence et 0 nombre inventés sur les 33 réponses ;
les quelques alertes sont des valeurs calculées, des dates, des numéros de section ou des valeurs de registres d'anomalies.

## Q1 — Quelle parclose je prends pour le double vitrage de série d'une PERFORM76 ?

Pages lues : perform76-parcloses.md · citées sans être lues : aucune

Le double vitrage de série de la PERFORM76 fait **28 mm** (6 / 18 argon / 4, intercalaire TGI noir, Ug 1,1 W/m²K) : il se tient avec la **parclose d'ouvrant 76526** (29,5 mm d'épaisseur) et la **parclose de dormant 2634** (29,5 mm).

- 76526 sur les quatre ouvrants (76272, 76275, 76279, 76281) ; 2634 sur les cinq dormants (76171, 76172, 76177, 76180, 76185) — les deux familles ne sont pas interchangeables.
- Si le vitrage de sécurité STADIP 44²/16/4 est retenu (28,76 mm réels), il reste dans la parclose standard 76526 de 28 mm, sans dépassement de feuillure (CTR-02, levée).

(/profiles/perform76-parcloses.md, §9 et §2–3)

## Q2 — Quelle largeur maxi pour un coulissant LUMINE65 à 2 vantaux ?

Pages lues : coulissants-aluminium.md · citées sans être lues : aucune

**4 000 mm** pour un coulissant LUMINE65 à 2 vantaux sur 2 rails (largeur mini 1 000 mm, hauteur maxi 2 580 mm). (/gammes/coulissants-aluminium.md, catalogue général 2026 p. 16 et brochure LUMINE65 février 2025 p. 3)

Le catalogue précise que la largeur maximale de baie du LUMINE65 est de 6 m (configuration 4 vantaux) et demande de valider la faisabilité auprès d'un technicien PROFERM avant commande.

## Q3 — Fenêtre PERFORM76 oscillo-battante un vantail, blanche, 1 100 de large sur 1 300 de haut, double vitrage 4/16/4 : on peut la faire ?

Pages lues : dta-6-16-2334.md, perform76-parcloses.md, systeme-76-abaques-dimensionnels.md, systeme-76-cotes-de-debit.md · citées sans être lues : aucune

**Oui, c'est faisable.**

- **DTA** : 1 vantail OB, limites 1,50 m de haut × 1,40 m de large (ou 2,15 × 1,00) ; la baie 1,10 × 1,30 m passe (/certifications/dta-6-16-2334.md).
- **Ouvrant** (dormant 76171, cote à déduire 38 mm) : 1 100 − 2 × 38 = **1 024 mm** de large, 1 300 − 2 × 38 = **1 224 mm** de haut (/profiles/systeme-76-cotes-de-debit.md).
- **Abaque 76281 avec renfort V266.Z, blanc** : hauteur maxi 235 cm jusqu'à 95 cm de large, puis oblique jusqu'à (130 ; 130) ; à 102,4 cm de large, l'ouvrant de 122,4 cm est bien sous la limite. Règle des 25 % respectée : 102,4 ≤ 1,25 × 122,4 (/profiles/systeme-76-abaques-dimensionnels.md).
- **Verre** : 4 + 4 = 8 mm, sous le seuil de 12 mm — pas de courbe de verre restrictive ni de justification expérimentale NF P 20-302 (déclenchée au-delà de 12 mm de verre ou 60 kg de vantail).
- **Parclose d'ouvrant** : la **76501** pour 24 mm de vitrage (/profiles/perform76-parcloses.md).

## Q4 — Quelle tapée et quelle patte de pose pour 140 mm d'isolant sur un dormant 76171 ?

Pages lues : perform76-tapees-et-isolation.md, pose-perform76.md · citées sans être lues : aucune

**140 mm d'isolant n'existe pas sur un dormant 76171** : les seules épaisseurs documentées sont 80, 95, 115, 135, 155, 175, 195 et 215 mm. Une demande à 140 mm se traite donc en 135 ou en 155 mm :

| Iso sur 76171 | Tapée | Patte de pose | Appui | Cale |
| --- | --- | --- | --- | --- |
| 135 mm | 6140 | NT1945 | 76758 | CTHNT0030 |
| 155 mm | 6141 | NT1947 | 76758 | CTHNT0030 |

Dans les deux cas : clameau CP14GGOM0012, et la cale CTHNT0030 (le seul cas sans cale est 80 mm). Le dormant bas reste un 76171 jusqu'à 155 mm inclus ; au-delà, il devient un 76180 à aile de 20 mm.

À noter : le 140 mm est une épaisseur du dormant 76180 (tapée 6141, patte NT1947, appui 6137, sans cale) — si le chiffrage peut basculer sur ce dormant, la configuration 140 mm existe toute faite. (/profiles/perform76-tapees-et-isolation.md ; /procedures/pose-perform76.md)

## Q5 — Quelle température de miroir et quel temps de fusion pour souder du PVC système 70 ?

Pages lues : directives-generales-systeme-70-evo2008.md, fabrication-profiles-pvc.md · citées sans être lues : aucune

Deux jeux de paramètres existent pour le système 70, selon le document suivi — ils se contredisent (CTR-88) :

| Paramètre | Classeur e.VOLUTION 2008 (système 70) | Directives profine 2023 (communes 70 et 76) |
| --- | --- | --- |
| Miroir chauffant | 235 à 245 °C | 245 à 250 °C |
| Temps de fusion | 40 à 50 s | 30 à 40 s |
| Avec insert de soudure | miroir 235 °C, fusion 60 s | miroir 245 à 250 °C, fusion 40 à 45 s |

Communs aux deux documents : ajustement max. 2 s, soudure min. 25 s, refroidissement avant ébavurage min. 45 s (60 s avec insert), limitation du cordon 2 ± 0,2 mm, couteaux 45 à 50 °C, pas de refroidissement accéléré. Les deux textes disent eux-mêmes que ces valeurs sont indicatives et dépendent de la machine ; une soudeuse réglée sur un document sort des plages de l'autre (CTR-88).

Sources : (/procedures/directives-generales-systeme-70-evo2008.md, pages imprimées 11-12) et (/procedures/fabrication-profiles-pvc.md, directive de soudage).

## Q6 — Quelle différence entre une PERFORM et une PERFORM+ ?

Pages lues : perform.md, perform-plus.md · citées sans être lues : aucune

Deux gammes PVC différentes : la **PERFORM** est la fenêtre PVC classique à ouvrant visible, la **PERFORM+** est une fenêtre PVC **à ouvrant caché** (l'ouvrant disparaît derrière le dormant vu de l'extérieur), présentée en mai 2023.

| | PERFORM | PERFORM+ |
| --- | --- | --- |
| Ouvrant | visible | caché |
| Profils | KÖMMERLING GREENLINE, 70 ou 76 mm (PERFORM70 / PERFORM76) | GREENLINE 100 % renforcés, épaisseur non donnée |
| Vitrage | posé sur parcloses | **collé** dans l'ouvrant, jusqu'à 28 mm |
| Quincaillerie | pivot symétrique, crémone semi-fixe | Roto NX, poignée TOULON décalée seule |
| Uw annoncé | 1,3 (70) / 0,8 (76) W/m²K | 1,2 W/m²K avec vitrage 1,0 |
| Coloris | blanc, ivoire, plaxés, laqué sous conditions | blanc 9016 seul |
| Interdits | cintrage en 76 | traverses, poignée centrée, serrures sur portes-fenêtres, fiches, cintrage |

Garantie structure : la brochure PERFORM+ annonce 20 ans, le catalogue général 15 ans — engagement divergent de 5 ans, aucune valeur ne suit ni la date ni le produit (CTR-03).

À noter : la PERFORM+ n'apparaît pas au catalogue général de janvier 2026 ; son statut commercial est à établir (VER-02). (/gammes/perform.md ; /gammes/perform-plus.md)

## Q7 — Mon client veut une HYBRIDE en gris anthracite 7016 : c'est possible en extérieur et en intérieur ?

Pages lues : coloris-hybride.md · citées sans être lues : aucune

**Extérieur : oui. Intérieur : pas en teinté dans la masse — seulement par laquage, sous condition.**

- **Extérieur aluminium** : le gris 7016 existe en laqué satiné ou granité, à prix préférentiel (/coloris/coloris-hybride.md).
- **Intérieur PVC** : le nuancier intérieur teinté dans la masse ne compte que deux teintes — blanc 9016 et ivoire (proche 9001, 607 Kömmerling, 2 faces uniquement). Pas de 7016 teinté dans la masse.
- **Laquage 2 faces identiques** : le catalogue 2026 écrit « laquage toutes teintes possibles 2 faces identiques », mais assorti de « laquage possible si : ouverture extérieure » et « uniquement sur PVC » ; la brochure de mars 2025 écrit « uniquement sur menuiserie PVC en cas de cintrage et/ou ouverture extérieure ». La portée exacte (face PVC seule, ou PVC + capot alu dans la même teinte) n'est pas tranchée par les documents (**VER-49**) : à confirmer auprès du service technique avant de chiffrer un 7016 intérieur.

En pratique : pour un 7016 extérieur, la combinaison standard est extérieur gris 7016 laqué + intérieur blanc 9016 teinté dans la masse.

## Q8 — Quelle garantie sur la ferrure Roto NX, et pour quoi exactement ?

Pages lues : roto.md, roto-nx.md, roto-nx.md, garanties-par-composant.md, roto-nx-ksr-consignes-generales.md · citées sans être lues : aucune

**10 ans sur le fonctionnement** — c'est une garantie de fonctionnement, pas une garantie pièce ou matériaux : la grille contractuelle PROFERM écrit « Ferrure ROTO : 10 ans, sur le fonctionnement » (catalogue général 2026, p. 35 ; /garanties/garanties-par-composant.md). Une autre ferrure (non ROTO) n'est garantie que 2 ans.

Côté ROTO, la garantie est présentée comme « garantie de fonctionnement de 10 ans pour nos partenaires », certifiée pour la ferrure standard Roto NT / NX (/commercial/roto-nx.md).

Deux conditions pour qu'elle joue :
- **La garantie porte uniquement sur les composants Roto d'origine** — la ferrure complète doit être constituée exclusivement de pièces du système Roto d'origine (instructions de montage Roto NX KSR, p. 11-12 et 22 ; /procedures/roto-nx-ksr-consignes-generales.md).
- Le fabricant de ferrure décline toute responsabilité en cas de non-respect du manuel de montage, d'utilisation non conforme ou de défaut d'instruction.

Le wiki ne détaille pas au-delà ce que « fonctionnement » couvre exactement (usure, réglage, etc.) : le corps du certificat allemand de ROTO est illisible même rendu à 500 dpi (/commercial/roto-nx.md).

## Q9 — Un vantail Roto NX frotte en bas sur le dormant, comment je règle ?

Pages lues : percage-montage-roto-nx.md, roto-nx-compas-et-paliers.md · citées sans être lues : aucune

Le réglage se fait en hauteur au **palier / pivot d'angle**, dans l'angle bas du vantail côté paumelles : on y relève le vantail pour dégager la traverse basse du dormant.

- Ferrure standard (côté paumelles P) : réglage vertical à l'angle bas du vantail, plage **+2,5 / −2,0 mm** autour de la position initiale (/procedures/percage-montage-roto-nx.md, §15).
- NT Designo II : réglage en hauteur du palier / pivot d'angle, plage **+2,0 / −1,0 mm** ; le même palier se règle aussi en latéral (+2,0 / −1,5 mm) et en compression (±0,5 mm) (/procedures/percage-montage-roto-nx.md, §26).

Deux points à savoir :
- **Sur un NT Designo II avec report de charge, il faut régler le report de charge à nouveau après le réglage en hauteur** (/procedures/percage-montage-roto-nx.md, §26).
- En complément, le goujon de verrouillage V se règle aussi en hauteur par quarts de tour, jusqu'à **±0,8 mm** (360°) — utile si le frottement vient de la compression d'appui plutôt que de la hauteur (/procedures/percage-montage-roto-nx.md, §16).

Le manuel ne nomme pas les réglages : les plages ci-dessus sont celles des dessins de la page « Réglages » (IMO_180_NX_FR_v2, p. 108 et 115).

## Q10 — Quel est le délai de fabrication d'une PERFORM cintrée ?

Pages lues : perform.md, proferm.md · citées sans être lues : aucune

**Le wiki ne donne aucun délai de fabrication** — ni pour une PERFORM cintrée, ni pour une menuiserie standard. PROFERM annonce « les meilleurs délais » et « un produit et un service de qualité dans les meilleurs délais », mais ces formules ne sont rattachées à aucun délai chiffré (/commercial/proferm.md).

Ce que le wiki dit en revanche du cintré :
- le cintrage (hors triangle et trapèze) n'est possible que pour la **PERFORM70** — pas de cintrage en PERFORM76 (/gammes/perform.md) ;
- PROFERM dispose d'un atelier de cintrage et fabrique plein cintre, cintre surbaissé, œil de bœuf, anse de panier, chapeau de gendarme (à partir d'un gabarit), triangle et trapèze (/gammes/perform.md).

Pour un délai chiffré, il faut se tourner vers le service commercial — le wiki ne le porte pas.

## Q11 — Fenêtre PERFORM76 oscillo-battante un vantail, blanche, 1 300 de large sur 1 450 de haut, dormant 76171 et ouvrant 76281, avec un triple vitrage 4/12/4/12/4 : on peut la faire ?

Pages lues : systeme-76-cotes-de-debit.md, perform76-parcloses.md, perform.md, systeme-76-abaques-dimensionnels.md, dta-6-16-2334.md · citées sans être lues : aucune

**Non : l'ouvrant dépasse la courbe d'épaisseur de verre 12 mm de l'abaque du 76281.**

- Ouvrant : 1 300 − 2 × 38 = 1 224 mm de large, 1 450 − 2 × 38 = 1 374 mm de haut (cote à déduire du dormant 76171 : 38 mm, /profiles/systeme-76-cotes-de-debit.md).
- Verre : 4 + 4 + 4 = 12 mm (les abaques additionnent les couches de verre, sans les intercalaires, /profiles/systeme-76-abaques-dimensionnels.md). Courbe 12 mm du 76281 avec renfort V266.Z : 132 cm de haut à 120 cm de large, 112 cm à 130 cm ; l'ouvrant fait 137,4 cm pour 122,4 cm de large, au-dessus de la limite aux deux graduations.
- Le reste passe : DTA, OB 1 vantail jusqu'à 1,50 m de haut × 1,40 m de large (baie 1,45 × 1,30, /certifications/dta-6-16-2334.md) ; limite blanc 235 cm de haut / 130 cm de large ; règle des 25 % ; parclose d'ouvrant **76503** pour le vitrage de 36 mm (/profiles/perform76-parcloses.md).

Pour la faire : équerres de feuillure **J079 aux quatre coins** de l'ouvrant — les limitations de vitrage se décalent alors de deux courbes caractéristiques, et pour 12 mm de verre l'ouvrant retombe sous la limite blanc (235 × 130 cm), qu'il respecte largement — ou passer en double vitrage.

À vérifier : le poids du vantail — au-delà de 60 kg de masse de vantail, la conformité mécanique se démontre par voie expérimentale selon NF P 20-302 (/profiles/perform76-parcloses.md).
