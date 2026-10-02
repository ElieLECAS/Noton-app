# Test de génération — 10 questions réalistes, une passe (02/10/2026)

GLM 5.3 (`zai-glm-5-3`, effort high, température 0,2), une conversation par question, sans historique. Notes de Claude
d'après les sources du wiki et, pour la faisabilité, le vérificateur de l'application. Une seule passe, un seul correcteur : à lire
comme une base de comparaison, pas comme une statistique.

| Q | Note | Fond | Défauts | Appels | Coût | Caractères |
|---|---|---|---|---|---|---|
| 1 | A | Valeur exacte (28 mm : ouvrant 76526, dormant 2634), familles jumelles séparées, 6 lignes. | Aucun. | 3 | 0.038 $ | 830 |
| 2 | B | 4 000 mm exact, et le piège des 6 m (valable pour 4 vantaux seulement) évité. | Paragraphe sur le classement AEV (CTR-11) hors sujet. | 4 | 0.042 $ | 931 |
| 3 | B | « Oui » exact (le vérificateur donne « ok »), 24 mm / 8 mm de verre bien lus, zone B signalée. | Ouvrant non précisé dans la question : il en choisit un (76272/76279) sans le dire ; parclose 76501 non donnée ; 2 105 car. ; phrase confuse sur le dormant. | 4 | 0.0852 $ | 2105 |
| 4 | A- | 135 ou 155 mm (140 n'existe pas sur le 76171), tapée, patte, appui, cale justes. | Dernière phrase du « À noter » contradictoire. | 3 | 0.0466 $ | 1159 |
| 5 | B- | Les deux jeux de valeurs avec leur document et CTR-88. | « en vigueur pour les fabrications actuelles » : arbitrage non sourcé, qui contredit « aucun arbitrage » ; paramètres en plus. | 4 | 0.0524 $ | 1610 |
| 6 | B+ | Tableau comparatif juste et sourcé. | Deux réserves (VER-02, CTR-03) sourcées mais longues ; 2 101 car. | 3 | 0.0376 $ | 2101 |
| 7 | B+ | Extérieur oui (satiné ou granité), intérieur seulement par laquage conditionnel ; VER-49 dit honnêtement non tranché. | « À noter » final (statut draft, ancien nom) hors sujet ; 1 834 car. | 2 | 0.024 $ | 1834 |
| 8 | C+ | 10 ans sur le fonctionnement, juste. | Déductions non sourcées (« ouverture, fermeture, verrouillage », « entretien = condition implicite ») ; page citée non lue ; note sur la résolution d'un certificat ; 1 765 car. pour une valeur. | 3 | 0.0377 $ | 1765 |
| 9 | B | Réponse réelle trouvée (+2,5 / −2,0 mm au palier d'angle, report de charge à refaire sur Designo II) ; dit ce que le manuel ne nomme pas. | Page de report de charge citée sans être lue ; digression sur le goujon V ; 2 029 car. | 5 | 0.066 $ | 2029 |
| 10 | B+ | Aucun délai inventé, « meilleurs délais » non chiffré, cintrage PERFORM70 seulement. | Conditions d'atelier (diamètres) hors sujet ; narre sa recherche ; proferm.md citée sans être lue. | 4 | 0.0471 $ | 1280 |

Total : 0.477 $ ; cache 86 à 95 % ; 4,5 à 16 s par question.

## Q1 — Quelle parclose je prends pour le double vitrage de série d'une PERFORM76 ?

Étapes : chercher(parclose vitrage de série PERFORM76 | double vitrage standard PERFORM76 épaisseu) > lire(/profiles/perform76-parcloses.md (§9, §8, §2, §3))
Pages lues : perform76-parcloses.md · citées sans être lues : aucune · anomalies : aucune

Pour le double vitrage de série de la PERFORM76 — **28 mm** (composition 6 / 18 argon / 4, intercalaire TGI noir, Ug 1,1 W/m²K) — il faut deux parcloses selon l'endroit où le vitrage est posé (/profiles/perform76-parcloses.md, §9) :

| Emplacement du vitrage | Parclose | Épaisseur de la parclose |
| --- | --- | --- |
| Dans l'**ouvrant** (76272, 76275, 76279, 76281) | **76526** | 29,5 mm |
| Dans le **dormant** (76171, 76172, 76177, 76180, 76185) ou le fixe | **2634** | 29,5 mm |

Les deux familles ne sont pas interchangeables : la série 76526 ne se monte que sur les ouvrants, la 2634 que sur les dormants (§8 de la même page). À l'intérieur de sa famille, l'épaisseur de vitrage est le seul critère de choix.

Source : cahier technique PERFORM76 version 02/09/2026 CC03, p. 8, repris dans /profiles/perform76-parcloses.md.

## Q2 — Quelle largeur maxi pour un coulissant LUMINE65 à 2 vantaux ?

Étapes : chercher(LUMINE65 coulissant dimensions maximales 2 vantaux | coulissant LUMINE65 largeur) > lire(/gammes/coulissants-aluminium.md (Dimensions limites)) > lire(/gammes/coulissants-aluminium.md (§10))
Pages lues : coulissants-aluminium.md · citées sans être lues : aucune · anomalies : CTR-11

Pour un coulissant **LUMINE65 en 2 vantaux – 2 rails**, la largeur maximale de baie est **4 000 mm** (largeur mini 1 000 mm, hauteur de 500 à 2 580 mm) — (/gammes/coulissants-aluminium.md, §10, d'après le catalogue général 2026 p. 16 et la brochure LUMINE65 de février 2025 p. 3).

Nuances utiles :
- La limite de **6 m** annoncée pour la baie ne vaut que pour la configuration **4 vantaux – 2 rails** (2 000 à 6 000 mm) : ne la transposez pas au 2 vantaux.
- Le catalogue demande de **se rapprocher d'un technicien PROFERM pour valider la faisabilité avant de passer commande**.
- Vitrage 28 mm (6|18|4), 32 mm possible.

À noter si le classement AEV vous sert au chiffrage : la brochure LUMINE65 (p. 3) donne A*4 / E*6A / V*A3 sur 2 vantaux, alors que le catalogue général (p. 17) indique A*4 / E*6A / V*B2 pour tous les coulissants — la valeur à retenir est A*4 / E*6A / V*A3, seule source qui essaie ce produit précis (CTR-11).

## Q3 — Fenêtre PERFORM76 oscillo-battante un vantail, blanche, 1 100 de large sur 1 300 de haut, double vitrage 4/16/4 : on peut la faire ?

Étapes : chercher(dimensions maximales oscillo-battant un vantail PERFORM76 | dimensions maximales) > lire(/certifications/dta-6-16-2334.md (§17) ; /profiles/systeme-76-abaques-dimensionn) > lire(/profiles/systeme-76-abaques-dimensionnels.md (§3, §5, §6, §7, §8, §9, §15, §16))
Pages lues : dta-6-16-2334.md, perform76-parcloses.md, systeme-76-abaques-dimensionnels.md · citées sans être lues : aucune · anomalies : aucune

**Oui, cette fenêtre est réalisable.** Vérification sur les trois limites qui s'appliquent :

**1. Dimensions maximales de baie (DTA)** — Pour un 1 vantail oscillo-battant, le DTA 6/16-2334_V5 donne deux couples de limites pour les fabrications non certifiées : H 1,50 m × L 1,40 m, ou H 2,15 m × L 1,00 m (/certifications/dta-6-16-2334.md, §2.2.3.7). Votre baie de 1,10 × 1,30 m passe dans la première ligne (1,30 ≤ 1,50 et 1,10 ≤ 1,40).

**2. Abaque d'ouvrant (blanc)** — Sur l'abaque des ouvrants 76272 / 76279 avec renfort V326.Z ou V314.Z, catégorie blanc : hauteur maxi 250 cm jusqu'à 110 cm de large, puis oblique jusqu'au coin 150 × 150, largeur maxi 150 cm (/profiles/systeme-76-abaques-dimensionnels.md, §15–§16). Le point (110 ; 130) — pris en cotes de baie, majorantes de l'ouvrant — est largement dans la limite. La règle des 25 % est respectée (110 ≤ 1,25 × 130). En revanche, la séparation de zones à 80 cm de largeur et 130 cm de hauteur place l'ouvrant en **zone B : renforcement horizontal** (traverses) — à confirmer avec les cotes d'ouvrant réelles après débit.

**3. Vitrage 4/16/4 (24 mm)** — Épaisseur de verre de 8 mm : sous le seuil de 12 mm, donc pas de restriction dimensionnelle liée au verre ni de renforcement total obligatoire (/profiles/systeme-76-abaques-dimensionnels.md, §5). L'ouvrant PERFORM76 admet de 16 à 50 mm de vitrage, donc une parclose d'ouvrant pour 24 mm existe (/profiles/perform76-parcloses.md, §1). Deux nuances :
- le double vitrage de **série** de la PERFORM76 fait 28 mm (6/18 argon/4, Ug 1,1) ; le 4/16/4 de 24 mm est admissible mais ce n'est pas la composition de série (/profiles/perform76-parcloses.md, §9) ;
- tout le vitrage étant en ouvrant, la restriction « moins de 28 mm non tenu en dormant » ne concerne pas cette configuration.

**En résumé :** faisable en blanc avec renfort d'ouvrant (V326.Z ou V314.Z) et renforcement horizontal des traverses selon l'abaque. Si la fabrication était certifiée, des dimensions supérieures resteraient envisageables (précisées au Certificat de Qualification du menuisier), mais ce n'est pas nécessaire ici.

## Q4 — Quelle tapée et quelle patte de pose pour 140 mm d'isolant sur un dormant 76171 ?

Étapes : chercher(tapée de pose 140 mm isolant dormant 76171 | patte de pose PERFORM76 | 76171 tap) > lire(/profiles/perform76-tapees-et-isolation.md)
Pages lues : perform76-tapees-et-isolation.md · citées sans être lues : aucune · anomalies : aucune

Sur un dormant **76171**, 140 mm d'isolant n'est pas une valeur documentée : les seules épaisseurs d'isolation qui existent sur ce dormant sont **80, 95, 115, 135, 155, 175, 195 et 215 mm** — « aucune épaisseur intermédiaire n'existe » (/profiles/perform76-tapees-et-isolation.md, cahier technique PERFORM76 CC03, p. 19 et 24). Une demande à 140 mm se traite donc en 135 ou en 155 mm :

| Iso retenue (mm) | Tapée | Patte de pose | Appui | Cale |
| --- | --- | --- | --- | --- |
| 135 | 6140 | NT1945 | 76758 | CTHNT0030 |
| 155 | 6141 | NT1947 | 76758 | CTHNT0030 |

Le clameau est le **CP14GGOM0012** dans les deux cas ; la cale CTHNT0030 est de série (seule la configuration à 80 mm s'en passe).

À noter : le 140 mm exact existe sur le dormant neuf **76180** (tapée 6141, patte NT1947, appui 6137, sans cale) et sur les dormants rénovation 76177/76185 (tapée 6141, appui 6137, sans patte de pose affectée). Sur le 76171, la tapée 6141 ne donne 140 mm que sur le 76180 — un 76171 donne 15 mm d'isolant de plus que le 76180 à tapée identique.

Source : /profiles/perform76-tapees-et-isolation.md (cahier technique PERFORM76, version 02/09/2026 CC03, p. 24).

## Q5 — Quelle température de miroir et quel temps de fusion pour souder du PVC système 70 ?

Étapes : chercher(température miroir temps de fusion soudure PVC | soudage paramètres miroir fusio) > lire(/procedures/directives-generales-systeme-70-evo2008.md (Soudage) ; /procedures/f) > lire(/procedures/directives-generales-systeme-70-evo2008.md (Directives pour le souda)
Pages lues : directives-generales-systeme-70-evo2008.md, fabrication-profiles-pvc.md · citées sans être lues : aucune · anomalies : CTR-88

Le wiki porte deux jeux de paramètres pour le système 70, qui se contredisent (CTR-88) :

**Directives générales profine, janvier 2023** — en vigueur pour les fabrications actuelles (/procedures/fabrication-profiles-pvc.md) :

| Paramètre | Soudage standard | Avec inserts de soudure |
| --- | --- | --- |
| Température du miroir chauffant | 245 à 250 °C | 245 à 250 °C |
| Temps de fusion | 30 à 40 s | 40 à 45 s |
| Temps d'ajustement | max. 2 s | max. 2 s |
| Temps de soudure | min. 25 s | min. 25 s |
| Refroidissement avant ébavurage | min. 45 s (pas plus de 30 min) | min. 60 s |

**Classeur KÖMMERLING e.VOLUTION d'août 2008** — pour les menuiseries fabriquées d'après cette génération (/procedures/directives-generales-systeme-70-evo2008.md) : miroir **235 à 245 °C**, temps de fusion **40 à 50 s** ; avec insert de soudure : miroir **235 °C**, fusion **60 s**. La courbe du classeur (miroir à 240 °C) montre la résistance des angles maximale dans la bande 40–50 s de fusion.

**Contradiction à connaître (CTR-88)** : une soudeuse réglée sur les valeurs d'un document sort des plages de l'autre — miroir 235–245 °C et fusion 40–50 s au classeur de 2008, contre 245–250 °C et 30–40 s aux directives de 2023. Aucun arbitrage n'est tranché : chaque jeu reste avec son document et sa date, et les deux textes précisent que ce sont des valeurs indicatives, dépendantes de la machine.

Conditions communes aux deux documents : profilés à au moins 15 °C, écart d'angle max. 0,5°, surfaces de soudure propres et sans film, limitation du cordon 2 ± 0,2 mm, couteaux à 45–50 °C, pas de refroidissement accéléré.

## Q6 — Quelle différence entre une PERFORM et une PERFORM+ ?

Étapes : chercher(PERFORM+ ouvrant caché vitrage collé | PERFORM gamme fenêtres PVC 70 76 | différ) > lire(/gammes/perform.md ; /gammes/perform-plus.md)
Pages lues : perform.md, perform-plus.md · citées sans être lues : aucune · anomalies : VER-02, CTR-03

Deux gammes PVC sur profils KÖMMERLING GREENLINE, mais deux produits très différents :

| | **PERFORM** (/gammes/perform.md) | **PERFORM+** (/gammes/perform-plus.md) |
|---|---|---|
| **Type** | Fenêtres et portes PVC à ouvrant apparent | Fenêtre PVC **à ouvrant caché** (ouvrant dissimulé derrière le dormant, vu de l'extérieur) |
| **Déclinaisons** | PERFORM70 (profil 70 mm) et PERFORM76 (profil 76 mm, 6 chambres, 3 joints) | Une seule version, profils 100 % renforcés |
| **Vitrage** | Vitrage posé aux parcloses (arrondies ou moulurées) | **Collage périphérique** du vitrage dans l'ouvrant, jusqu'à 28 mm |
| **Quincaillerie** | Pivot symétrique, rouleau champignon anti-dégondage, crémone de semi-fixe | **Roto NX** (accès RC2 avec oscillo-battant en position ouverte via TiltSafe) |
| **Poignée** | Gamme complète de poignées et béquilles | **TOULON en poignée décalée uniquement** |
| **Uw annoncé** | 1,3 W/m²K (PERFORM70) ; 0,8 W/m²K (PERFORM76) | 1,2 W/m²K avec un vitrage 1.0 |
| **Coloris** | Blanc 9016, ivoire, six plaxés, laqué (PERFORM70 sous conditions) | **Blanc 9016 seul**, teinté dans la masse |
| **Formes** | Sur mesure : cintrage (PERFORM70 uniquement), œil de bœuf, triangle, trapèze… | **Pas de cintrage**, pas de traverse, pas de poignée centrée, pas de serrure sur porte-fenêtre, pas de fiches |

En résumé : la PERFORM est la gamme PVC classique, large en coloris et en formes ; la PERFORM+ est une fenêtre à ouvrant caché plus rigide (vitrage collé, profils renforcés), mais figée en blanc et sans personnalisation.

Deux réserves à connaître :
- **Statut commercial de la PERFORM+ à établir (VER-02)** : la brochure date de mai 2023 et la gamme n'apparaît nulle part au catalogue général de janvier 2026 — ni dans les brochures LUMINE65 (2025) et HYBRIDE (2025). Sa page wiki est d'ailleurs en statut `draft`.
- **Garantie structure (CTR-03)** : la brochure PERFORM+/HYBRIDE+ annonce 20 ans sur la structure, alors que le catalogue général (p. 35) et la brochure HYBRIDE de mars 2025 indiquent 15 ans — engagement contractuel erroné de 5 ans, sans arbitrage documenté.

## Q7 — Mon client veut une HYBRIDE en gris anthracite 7016 : c'est possible en extérieur et en intérieur ?

Étapes : lire(/coloris/coloris-hybride.md)
Pages lues : coloris-hybride.md · citées sans être lues : aucune · anomalies : VER-49

**En extérieur : oui, sans problème.** Le gris 7016 existe au nuancier extérieur aluminium de l'HYBRIDE, laqué, à prix préférentiel, en deux finitions : **satiné** ou **granité** (/coloris/coloris-hybride.md, catalogue général 2026 p. 12).

**En intérieur : pas au nuancier standard.** La face PVC intérieure n'existe teintée dans la masse qu'en deux teintes : **blanc 9016** et **ivoire** (607 Kömmerling, proche 9001, 2 faces uniquement). Le 7016 intérieur ne peut donc venir que du **laquage « 2 faces identiques »**, que le catalogue conditionne : laquage possible **si ouverture extérieure**, et **« uniquement sur PVC »**.

⚠️ **Point à vérifier avant devis (VER-49)** : la portée exacte de cet encadré n'est pas tranchée dans la documentation. Le catalogue général 2026 écrit « laquage toutes teintes possibles 2 faces identiques » puis « uniquement sur PVC » ; la brochure HYBRIDE de mars 2025 écrit « uniquement sur menuiserie PVC en cas de cintrage et/ou ouverture extérieure » ; le dépliant général de juin 2023, lui, écrit « laquage toutes teintes possibles si : ouverture extérieure, menuiserie cintrée », sans la mention PVC. Aucun document ne dit clairement si, sur une HYBRIDE standard (ouverture intérieure, non cintrée), le laquage 2 faces 7016 est réalisable — ni s'il porte sur la seule face PVC ou aussi sur le capot aluminium. À confirmer auprès du service technique avant de s'engager.

**En résumé** : extérieur 7016 oui (satiné ou granité) ; intérieur 7016 uniquement par laquage, sous condition d'ouverture extérieure selon le catalogue — condition elle-même à arbitrer (VER-49). Sinon, l'intérieur standard est blanc 9016 ou ivoire.

À noter : cette page est en statut `draft` et le 7016 s'appelait « Anthracite 7016 » dans les documents de 2023 — même teinte, utile en SAV pour les menuiseries déjà posées.

## Q8 — Quelle garantie sur la ferrure Roto NX, et pour quoi exactement ?

Étapes : chercher(garantie ferrure Roto NX 10 ans | garantie de fonctionnement Roto NX | Label ROT) > lire(/fournisseurs/roto.md (Garantie propre) ; /commercial/roto-nx.md (10 ans de gara)
Pages lues : roto.md, roto-nx.md, roto-nx.md, garanties-par-composant.md, roto-nx-ksr-consignes-generales.md · citées sans être lues : /procedures/maintenance-ferrure-roto-nx.md · anomalies : aucune

**La ferrure Roto NX est garantie 10 ans sur le fonctionnement** — c'est-à-dire sur la fonctionnalité de la ferrure (ouverture, fermeture, verrouillage), et non sur l'aspect ou d'autres composants de la menuiserie.

**Ce que disent les sources :**

| Source | Ce qu'elle dit |
| --- | --- |
| Grille contractuelle PROFERM, catalogue général 2026, p. 35 (/garanties/garanties-par-composant.md) | Ferrure ROTO : **10 ans sur le fonctionnement** ; toute autre ferrure : 2 ans |
| Catalogue Roto NX PVC, juin 2023 (/commercial/roto-nx.md) | Certificat « 10 Jahre Garantie für den Roto NT / NX Standardbeschlag » : garantie de fonctionnement de 10 ans pour la ferrure standard Roto NT / NX |
| Fiche Roto NX (/quincaillerie/roto-nx.md) | « 10 ans de garantie sur la fonctionnalité des ferrures » |

**Les limites de la garantie** (manuel de montage Roto NX KSR, /procedures/roto-nx-ksr-consignes-generales.md) :
- **La garantie porte uniquement sur les composants Roto d'origine** : la ferrure complète doit être constituée exclusivement de pièces du système Roto d'origine, faute de quoi la sécurité de fonctionnement et la garantie tombent.
- Le fabricant décline toute responsabilité en cas de non-respect du manuel et des directives applicables, d'utilisation non conforme, de non-respect des prescriptions d'installation, ou d'accroissement des salissures (l'entretien périodique est donc une condition implicite de maintien de la garantie — voir /procedures/maintenance-ferrure-roto-nx.md).

À noter : le corps du certificat allemand reproduit dans le catalogue Roto (conditions et durée, droits à garantie, mise en œuvre, composition de la ferrure standard) est imprimé en trop basse résolution pour être lu de façon sûre — le wiki n'en livre que les intertitres.

## Q9 — Un vantail Roto NX frotte en bas sur le dormant, comment je règle ?

Étapes : chercher(réglage vantail frotte dormant Roto NX | réglage pivot bas palier Roto NX | régl) > lire(/procedures/percage-montage-roto-nx.md (§15 Réglages) ; /quincaillerie/roto-nx-p) > lire(/procedures/percage-montage-roto-nx.md (Réglages (uniquement si nécessaire), Rég) > lire(/procedures/percage-montage-roto-nx.md (Réglage du palier / pivot d'angle))
Pages lues : percage-montage-roto-nx.md, roto-nx-pivots-et-paliers-angle.md · citées sans être lues : /procedures/report-de-charge-roto-nx.md · anomalies : aucune

Un frottement en bas du vantail se règle **en hauteur, au palier d'angle** (la pièce de l'angle bas du dormant, côté paumelles, sur laquelle pose le vantail).

**Côté paumelles P (ferrure standard)** — le manuel de montage Roto NX KSR donne, sur le dessin des points de réglage de l'angle bas côté paumelles, une plage verticale de **+2,5 / −2,0 mm** autour de la position initiale (/procedures/percage-montage-roto-nx.md, §15 « Réglages (uniquement si nécessaire) ») :

![Réglages de la ferrure Roto NX, palier d'angle et palier de compas](/assets/procedures/roto-nx-ksr-montage/reglages-palier-et-compas.png)

Le même dessin porte aussi un réglage latéral de ±2,0 mm le long de la traverse basse et un réglage de compression de ±0,5 mm — utiles si le frottement vient d'un appui trop fort sur le joint plutôt que de la hauteur.

**Côté paumelles NT Designo II** — le réglage en hauteur du palier / pivot d'angle est de **+2,0 / −1,0 mm** (réglage latéral +2,0 / −1,5 mm, compression ±0,5 mm) (/procedures/percage-montage-roto-nx.md, §26) :

![Réglage du palier / pivot d'angle NT Designo II : hauteur, latéral, compression](/assets/procedures/roto-nx-ksr-montage/designo-reglage-palier-pivot-d-angle.png)

**Attention, sur un ouvrant NT Designo II équipé d'un report de charge : après tout réglage en hauteur du palier / pivot d'angle, il faut régler le report de charge à nouveau** (clé de 4 mm, jusqu'à ce que les arcs rouge et argent forment un cercle plein) — /procedures/percage-montage-roto-nx.md, §26, et /procedures/report-de-charge-roto-nx.md.

Deux nuances :
- Le manuel ne nomme pas les réglages ni l'outil : il ne donne que les dessins et les plages en mm. La source n'indique pas de sens de rotation.
- Si le frottement est léger et vient du serrage du joint, le goujon de verrouillage V se règle aussi en hauteur jusqu'à ±0,8 mm par quarts de tour (/procedures/percage-montage-roto-nx.md, §16) — mais pour un vrai frottement mécanique sur la traverse basse du dormant, c'est le palier d'angle qu'il faut remonter.

## Q10 — Quel est le délai de fabrication d'une PERFORM cintrée ?

Étapes : chercher(délai fabrication fenêtre cintrée PERFORM | châssis cintré délai | cintrage PERF) > chercher(délai de fabrication menuiserie cintrée | délai cintrage fenêtre | délais de fab) > lire(/gammes/perform.md (§11) ; /procedures/chassis-cintres-trapezoidaux-systeme-70.m)
Pages lues : perform.md, chassis-cintres-trapezoidaux-systeme-70.md · citées sans être lues : /commercial/proferm.md · anomalies : aucune

Le wiki ne donne **aucun délai de fabrication** — ni pour une PERFORM cintrée, ni pour une menuiserie standard. J'ai cherché dans les pages de la gamme, les procédures de fabrication et les documents commerciaux : aucune page ne chiffre un délai. La seule mention qui s'en approche est l'argument de marque « fournir un produit et un service de qualité dans les meilleurs délais », que le wiki note précisément comme **non chiffrée** — aucun délai n'y est rattaché (/commercial/proferm.md, §9).

En revanche, si la question vise la faisabilité d'une PERFORM cintrée, deux points du wiki changent la réponse :

- **Le cintrage n'est réalisable qu'en PERFORM70** : « Pas de cintrage en PERFORM76 » — le cintrage (hors triangle et trapèze) n'est possible que pour la PERFORM70 (/gammes/perform.md, §11).
- Les conditions d'atelier sont documentées pour le système 70 : diamètre minimum de cintrage par profilé (de 0,50 m pour le dormant 6100 à 1,00 m pour les ouvrants de porte 2415/2416), profilés à cintrer sans leurs joints, joints enfilés manuellement après cintrage (/procedures/chassis-cintres-trapezoidaux-systeme-70.md).

Pour un délai chiffré, il faudra s'adresser au service commercial ou à la planification de l'usine : cette information n'existe dans aucune page du wiki.
