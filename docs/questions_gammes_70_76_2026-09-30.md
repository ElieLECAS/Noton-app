# 20 questions sur les gammes 70 et 76 — sources fournisseurs et catalogues marketing

*Écrit le 30/09/2026 pour évaluer LIA. Aucune de ces questions n'a été posée à Mistral : ce document ne coûte rien à lire et rien à garder.*

## Ce que couvre ce jeu

Les gammes PVC de PROFERM en **70 et 76 mm** (PERFORM70 / PERFORM76, HYBRIDE, TEXTURAL, INNOSLIDE) et les systèmes **profine / KÖMMERLING** sur lesquels elles reposent (système 70 Plateforme, système 76 Advanced). Les réponses viennent de deux familles de documents, et chaque question en croise au moins deux ou pose un piège :

- **documents fournisseurs** : DTA 6/16-2334, DTD 6/16-2335, manuels de mise en œuvre, posters, plans e.VOLUTION de 2008, directives générales, cahier technique PERFORM76 ;
- **catalogues marketing** : catalogue général 2026, dépliant général 2023, brochures et dépliants HYBRIDE, argumentaires.

**Hors périmètre volontairement : tout ce qui vient de Technal** (SOLEAL, LUMEAL, LUMINE), dont les documents ne sont pas encore retraités. Les deux questions d'absence portent sur des zones entièrement traitées.

## Récapitulatif

| # | Sujet | Catégorie | Difficulté | Identifiants à signaler |
| --- | --- | --- | --- | --- |
| Q01 | Uw, Uf et Ug d'une PERFORM76 : lequel donner | valeur croisée | moyenne | — |
| Q02 | Épaisseur du profil PERFORM76 : 76 mm partout ? | valeur croisée | moyenne | — |
| Q03 | RC2 sur toute PERFORM76 ? | valeur croisée | moyenne | — |
| Q04 | HYBRIDE : 72 mm et 1,2 ou 70 à 76 mm et 0,8 ? | contradiction | difficile | CTR-06 et CTR-07 |
| Q05 | Vitrage maximal en PERFORM76 : 48 ou 50 mm ? | contradiction | moyenne | CTR-17 |
| Q06 | Garantie du laquage d'une HYBRIDE blanche | contradiction | simple | CTR-05 |
| Q07 | Chambres et joints de la PERFORM70 | piège de nommage | difficile | VER-28 et VER-03 |
| Q08 | INNOSLIDE et système 76 | piège de nommage | difficile | VER-15 |
| Q09 | Facteur solaire et transmission lumineuse d'une PERFORM76 | absence | simple | — |
| Q10 | Épaisseur de paroi des profilés PVC en mm | absence | moyenne | — |
| Q11 | Parclose 76526 : 28 mm en PERFORM76, pas en système 70 | piège 70/76 | simple | — |
| Q12 | Essai NF P 20-302 : 62 kg en système 70, 60 kg en 76 | piège 70/76 | difficile | — |
| Q13 | Profilé de liaison au dos du dormant : axe de vissage 30 mm ou 25 mm | piège 70/76 | moyenne | — |
| Q14 | Fenêtre 1 vantail à la française en 70 : 1 200 × 1 600, l'abaque m'a donné un repère | faisabilité | difficile | — |
| Q15 | Vitrage de 52 mm sur PERFORM76 « avec la 76515 et le joint de 2 mm » | faisabilité | difficile | CTR-17 |
| Q16 | Largeur hors tout du dormant large 6110 | contradiction | moyenne | CTR-20 |
| Q17 | Triple vitrage à 13 mm de verre et abaques de renfort d'ouvrant du système 70 | contradiction | difficile | CTR-80 |
| Q18 | Meneau de dormant à clair intérieur de 68 mm | lecture de tableau | moyenne | INC-09 |
| Q19 | Doublage de 175 mm sur dormant 76171 : tapée, appui, patte de pose | lecture de tableau | moyenne | — |
| Q20 | Renforts admis dans le dormant 6100 et l'ouvrant 6115 (système 70) | pages découpées | moyenne | — |

Répartition : 3 valeurs croisées catalogue ↔ fournisseur · 5 contradictions enregistrées (deux valeurs et l'identifiant exigés) · 3 pièges système 70 contre 76 · 2 pièges de nommage (nom PROFERM contre système profine) · 2 faisabilités · 2 lectures de tableau avec image · 2 absences · 1 question qui demande deux des pages du système 70 découpées le 30/09.

## Comment corriger

- **Juste** : toutes les valeurs de « Réponse attendue » sont données, avec leur unité. Quand une contradiction est attendue, **les deux valeurs et leur document**, et l'identifiant (CTR-, INC-, VER-) — l'identifiant seul ne suffit pas.
- **Faux** : une des « Réponses fausses tentantes » est donnée comme réponse, ou la prémisse fausse de la question est validée.
- **Absence (Q07 partiellement, Q08, Q09, Q10)** : la réponse dit ce que le wiki ne donne pas, donne ce qu'il donne, et **ne complète pas** avec la valeur d'une gamme ou d'un système voisin.
- **Image demandée** : le chemin `/assets/…` attendu doit être celui de la référence demandée, pas celui de la ligne voisine.
- **Sources** : une page réellement lue et citée. Le chemin cité doit exister.

## Points de vigilance avant de s'en servir comme référence

1. **Q12 (62 kg contre 60 kg)** : les deux seuils sont bien ceux du wiki (DTD 2,3 : 62 kg ; DTA 2.3 : 60 kg), mais aucun registre ne confirme l'écart. Je n'ai pas pu rendre la page du PDF pour le vérifier : **à contrôler sur le DTD avant de compter cette question**. Si c'est une coquille de transcription, le wiki est à corriger et la question à retirer.
2. **Q05 et Q15** partagent CTR-17 (vitrage maximum 48 ou 50 mm) sous deux angles : la borne elle-même, puis le cas d'un vitrage de 52 mm.
3. **Q04** : la réponse sur « quoi prendre pour une HYBRIDE posée en 2024 » reprend l'hypothèse du wiki (« reste la valeur des menuiseries déjà livrées ») que le registre demande de faire confirmer au bureau d'études.
4. **Q07 et Q08** reposent sur VER-28 et VER-15 : le wiki dit que ces liens ne sont pas sourcés. Les champs `systeme` du frontmatter des pages de gamme les affirment pourtant (voir les notes plus bas). Si VER-28 est un jour tranché, ces deux questions changent de réponse.
5. **Q20 et Q11 à Q19** citent des pages telles qu'elles sont au 30/09/2026, après la découpe du système 70. Si les pages sont redécoupées, les chemins sont à reprendre.
6. Ces réponses ont été vérifiées dans le wiki par l'agent qui les a écrites, puis contrôlées par moi sur environ la moitié des valeurs chiffrées (grep des lignes citées, présence des images). Elles ne valent que **pour le wiki**, pas pour les PDF : un défaut de transcription du wiki y serait reproduit.

---

### Q01 — Uw, Uf et Ug d'une PERFORM76 : lequel donner
- **Catégorie** : valeur croisée
- **Difficulté** : moyenne
- **Question** : « Sur la PERFORM76 le catalogue annonce un Uw de 0,8, sur la fiche du système Kömmerling 76 je lis un Uf de 1,0, et le vitrage de série est à 1,1. Ça ne colle pas, si ? Je donne lequel au client ? »
- **Réponse attendue** : Ce n'est pas une contradiction : ce sont trois coefficients qui portent sur trois objets différents et « ne se comparent pas ». Uw (fenêtre complète) : « jusque 0,8 W/m²K » en PERFORM76 (catalogue général 2026, p. 7) ; c'est une valeur d'annonce en « jusque » et le wiki ne dit pas avec quel vitrage elle est atteinte. Uf (profilé seul) : « jusqu'à Uf = 1,0 W/(m²K) » pour le système 76 Advanced de profine (manuel de mise en œuvre profine, p. 15). Ug (vitrage seul) : 1,1 W/m²K pour le double vitrage de série de la PERFORM76, 28 mm, 6 / 18 argon / 4, intercalaire TGI noir (cahier technique PERFORM76, p. 4 ; catalogue p. 27 : « 1,1 (gammes PVC) »). Pour parler de la fenêtre PERFORM76, le seul chiffre de fenêtre disponible est le Uw « jusque 0,8 » du catalogue (le cahier technique PERFORM76 ne donne aucun Uw) ; le Uf caractérise le profilé du système et le Ug le vitrage, et aucun des deux ne remplace le Uw (« un 1,0 de Uf n'est pas un 1,0 de Uw »).
- **Doit signaler** : rien
- **Sources dans le wiki** : `gammes/perform.md` (§ Caractéristiques, tableau ; § Performances) ; `profiles/systeme-76-profiles-principaux.md` (§ Caractéristiques du système 76 Advanced à joint central, tableau) ; `profiles/perform76-parcloses.md` (§ Vitrage de série et limite d'épaisseur) ; `vitrages/performances-vitrages.md` (§ Vitrages thermiques et triples, tableau) ; `reference/glossaire.md` (§ des quatre coefficients Ug, Uf, Uw, Up).
- **Piège** : trois documents (catalogue, manuel profine, cahier PERFORM76) et trois grandeurs exprimées dans la même unité et voisines en valeur (0,8 / 1,0 / 1,1) : le modèle est tenté de crier à l'incohérence, de « corriger » le Uw par le Uf, ou de donner le Ug comme performance de la fenêtre.
- **Réponses fausses tentantes** : « le catalogue se trompe, le Uw réel est 1,0 » ; « Uw 1,1 W/m²K » (Ug pris pour Uw) ; « Uw 1,0 » (Ug du triple vitrage, ou Uf) ; « Uw 1,3 » (valeur PERFORM70) ; « Uf 2,0 » (Uf de la porte d'entrée du 76 Advanced) ; « Uf 1,3 à 1,9 » (directives e.VOLUTION 2008, système 70) ; « le Uw 0,8 est obtenu avec le triple vitrage » (rien ne le dit).
- **Vérifié** :
  - `gammes/perform.md` l.62 : `| PERFORM76 | 76 | 6 | 3 | 0,8 |` (colonne « Uw annoncé « jusque » ») ; l.169 : « Uw jusque 1,3 W/m²K en PERFORM70 et jusque 0,8 W/m²K en PERFORM76 ».
  - `profiles/systeme-76-profiles-principaux.md` l.74 : `| Coefficient de transfert de la chaleur Uf (profilé seul) | jusqu'à Uf = 1,0 W/(m²K) |`.
  - `profiles/perform76-parcloses.md` l.173-174 : « fait 28 mm, en 6 / 18 argon / 4, avec intercalaire TGI noir, pour un Ug de 1,1 W/m²K ».
  - `vitrages/performances-vitrages.md` l.46 : `| Double vitrage standard, en standard | 6 / 18 / 4 | gaz argon | warm edge faible émissivité | 1,1 (gammes PVC) |`.
  - `reference/glossaire.md` l.56 : « Quatre coefficients circulent dans le corpus et ne se comparent pas entre eux » ; l.74 : « un 1,0 de Uf n'est pas un 1,0 de Uw ».
  - Absence de Uw au cahier : grep `\bUw\b|\bUg\b|\bUf\b` sur `sources/cahier-technique-perform76.md`, `profiles/perform76-*.md`, `quincaillerie/perform76-*.md`, `procedures/pose-perform76.md`, `certifications/dta-6-16-2334.md` : seule occurrence = l'Ug de `perform76-parcloses.md` l.174.

### Q02 — Épaisseur du profil PERFORM76 : 76 mm partout ?
- **Catégorie** : valeur croisée
- **Difficulté** : moyenne
- **Question** : « Le catalogue dit PERFORM76 : profil de 76 mm, 6 chambres, 3 joints. Sur un ouvrant galbé ça reste 76 mm ? »
- **Réponse attendue** : Non, pas sur un ouvrant galbé. Le catalogue (p. 6-7) annonce « PERFORM76 : profil étanche grâce à ses 3 joints & 6 chambres », profil de 76 mm. Le cahier technique PERFORM76 (p. 4-5) détaille : les cinq dormants (76171, 76172, 76177, 76180, 76185) et les ouvrants droits (76281, 76272) font 76 mm ; les ouvrants galbés (76275, 76279) font 83 mm (la face arrondie déborde de 7 mm) ; dormants et ouvrants sont tous à 6 chambres ; les 3 joints sont « 1 joint à frappe sur l'ouvrant, 1 joint central, 1 joint à frappe sur le dormant », le joint de feuillure sous vitrage figurant sous un autre repère (7). Le manuel profine décrit le système 76 Advanced comme un « système à joint central de 76 mm ».
- **Doit signaler** : rien
- **Sources dans le wiki** : `gammes/perform.md` (§ Caractéristiques, tableau ; § Équipements de série, repère 10 ; § Coupes de la PERFORM76, tableau des repères 1 à 10) ; `profiles/perform76-ouvrants-et-battements.md` (§ Les quatre ouvrants ; § Cotes, tableau) ; `profiles/perform76-dormants.md` (§ Les cinq dormants ; § Cotes, tableau) ; `profiles/systeme-76-profiles-principaux.md` (§ Caractéristiques du système 76 Advanced, liste).
- **Piège** : deux documents (catalogue et cahier technique) qui parlent du même « 76 mm » à deux niveaux de détail ; le catalogue ne mentionne pas le galbé. Le modèle répond « oui, 76 mm » sans regarder la table des ouvrants, ou confond l'épaisseur du profil (76 / 83) avec la hauteur de face (39 à 110) ou avec les 83 mm de hauteur d'un autre profilé.
- **Réponses fausses tentantes** : « 76 mm sur tous les profilés » ; « 83 mm pour les dormants aussi » ; « 5 chambres » (chiffre de l'HYBRIDE et du système 70) ; « 4 joints » (en ajoutant le joint de feuillure) ; « 72 mm » (ancienne HYBRIDE).
- **Vérifié** :
  - `gammes/perform.md` l.86 : `| 10 | PERFORM76 : profil étanche grâce à ses 3 joints & 6 chambres |` ; l.225 : `| 1 | système à 3 joints d'étanchéité : 1 joint à frappe sur l'ouvrant, 1 joint central, 1 joint à frappe sur le dormant |` ; l.228 : `| 3 | dormant à 6 chambres de 76 mm d'épaisseur |` ; l.231 : `| 5b | ouvrant galbé, 83 mm d'épaisseur |` ; l.62 : `| PERFORM76 | 76 | 6 | 3 | 0,8 |`.
  - `profiles/perform76-ouvrants-et-battements.md` l.54 `| 76281 | droit | 76 |`, l.55 `| 76275 | galbé | 83 |`, l.56 `| 76272 | droit | 76 |`, l.57 `| 76279 | galbé | 83 |` ; l.49 : « dont la face arrondie déborde de 7 mm » ; § Les quatre ouvrants : « quatre ouvrants, tous à 6 chambres ».
  - `profiles/perform76-dormants.md` l.31-32 : « cinq dormants, tous à 6 chambres … et 76 mm d'épaisseur » ; tableau l.52-56 (colonnes Épaisseur = 76, Chambres = 6 pour 76171, 76172, 76177, 76180, 76185).
  - `profiles/systeme-76-profiles-principaux.md` l.83 : « Système à joint central de 76 mm dans un design moderne ».

### Q03 — RC2 sur toute PERFORM76 ?
- **Catégorie** : valeur croisée
- **Difficulté** : moyenne
- **Question** : « Un client veut du RC2 : profine annonce RC 2 sur son système 76 et le catalogue met le RC2 sur la PERFORM76, donc toute PERFORM76 est RC2, non ? »
- **Réponse attendue** : Non, le wiki ne permet pas de le dire. Le catalogue (p. 34) donne le RC2 (CERIBOIS) pour une configuration : « fenêtre PERFORM76 équipée d'un vitrage securit 44/6 collé et d'une quincaillerie spécifique (ferrage périmétrique, poignée verrouillable Sécustik) », « testée et labellisée ». Le « Anti-effraction : jusqu'à RC 2 » de profine (manuel de mise en œuvre 76 Advanced, p. 15) est une borne annoncée pour le système à joint central, pas une classe acquise par chaque fenêtre. Le Label ROTO Performance « offre l'accès à la certification RC1 / RC2 sur les fenêtres ». Pour comparaison, le double vitrage de série de la PERFORM76 est un 28 mm 6 / 18 argon / 4 (cahier technique, p. 4) ; le wiki ne dit nulle part que la PERFORM76 de série soit RC2. À retenir : à proposer seulement avec les composants cités.
- **Doit signaler** : rien
- **Sources dans le wiki** : `certifications/labels-et-certifications.md` (§ Répertoire des labels, ligne « RC2 » ; § Niveaux de résistance à l'effraction) ; `profiles/systeme-76-profiles-principaux.md` (§ Caractéristiques du système 76 Advanced à joint central, tableau) ; `vitrages/performances-vitrages.md` (§ Vitrage de la fenêtre certifiée RC2) ; `fournisseurs/roto.md` (§ Label ROTO Performance ; § Sécustik) ; `gammes/perform.md` (§ Performances).
- **Piège** : deux documents avec le même sigle « RC2 » à deux statuts différents (borne « jusqu'à » du fournisseur vs configuration essayée par PROFERM) ; prémisse fausse dans la question. Le modèle valide la prémisse ou transpose le RC2 « position OB ouverte » de la PERFORM+ / HYBRIDE+.
- **Réponses fausses tentantes** : « oui, toutes les PERFORM76 sont RC2 » ; « RC2 avec n'importe quel vitrage » ; « RC2 grâce au Label ROTO Performance seul » ; « RC2 avec la Roto NX en oscillo-battant position ouverte » (formule de la brochure PERFORM+ / HYBRIDE+ de 2023) ; « RC3 ».
- **Vérifié** :
  - `certifications/labels-et-certifications.md` l.91 : `| RC2 (résistance à l'effraction) | CERIBOIS | fenêtre PERFORM76 équipée d'un vitrage securit 44/6 collé et d'une quincaillerie spécifique (ferrage périmétrique, poignée verrouillable Sécustik) | testée et labellisée |` ; l.168 : « il offre l'accès à la certification RC1 / RC2 sur les fenêtres ».
  - `profiles/systeme-76-profiles-principaux.md` l.68 : « chaque valeur est une borne supérieure (« jusqu'à ») » ; l.79 : `| Anti-effraction | jusqu'à RC 2 |`.
  - `vitrages/performances-vitrages.md` l.101 : « vitrage securit 44/6 collé en feuillure combiné à un ferrage périmétrique et une poignée verrouillable Sécustik® ».
  - `profiles/perform76-parcloses.md` l.173-174 : double vitrage de série « 28 mm, en 6 / 18 argon / 4 ».
  - `fournisseurs/roto.md` l.259 : « l'accès à la certification **RC1 et RC2** » ; l.284 : « le label RC2 de la PERFORM76 repose sur une poignée verrouillable Sécustik ».

### Q04 — HYBRIDE : 72 mm et 1,2 ou 70 à 76 mm et 0,8 ?
- **Catégorie** : contradiction
- **Difficulté** : difficile
- **Question** : « J'ai deux documents HYBRIDE qui ne disent pas pareil : la brochure de mars 2025 parle de 72 mm de PVC et d'un Uw de 1,2, le catalogue 2026 de 70 à 76 mm et de 0,8. Pour un remplacement à l'identique d'une HYBRIDE posée en 2024, je prends quoi ? »
- **Réponse attendue** : Les deux documents divergent, entrées CTR-06 (épaisseur) et CTR-07 (Uw). Brochure HYBRIDE de mars 2025, p. 2 (et dépliant HYBRIDE de juin 2023, p. 2) : « Profilé PVC épaisseur 72mm », gamme unique, et « Uw jusqu'à 1.2 W/m²K » (5 chambres). Catalogue général de janvier 2026, p. 10-11 : profilé PVC de « 70 mm à 76 mm » (HYBRIDE70 / HYBRIDE76) et Uw « jusqu'à 0,8 W/m²K ». Pour une HYBRIDE posée en 2024 : c'est la fabrication d'avant 2026, donc profilé de 72 mm et Uw 1,2 (le wiki écrit « les fabrications antérieures à 2026 reposent sur un profilé de 72 mm » et le registre note que 1,2 « reste la valeur des menuiseries déjà livrées », hypothèse à faire confirmer par le bureau d'études) ; 70 à 76 mm et 0,8 valent pour la gamme du catalogue 2026, retenus par le registre comme « source la plus récente ». Ne pas attribuer le 0,8 à la seule HYBRIDE76 : le catalogue ne dit pas s'il vaut pour l'HYBRIDE70, l'HYBRIDE76 ou les deux (VER-04). Complément : le dépliant général de juin 2023 donnait même 1,3 pour l'HYBRIDE contre 1,2 au dépliant HYBRIDE du même mois (CTR-13, valeur du document produit : 1,2).
- **Doit signaler** : CTR-06 et CTR-07 (obligatoires) ; VER-04 ; CTR-13 (si les deux documents de juin 2023 sont comparés)
- **Sources dans le wiki** : `gammes/hybride.md` (§ Définition et déclinaisons ; § Caractéristiques, tableau par période) ; `anomalies/contradictions-entre-sources.md` (lignes CTR-06, CTR-07, CTR-13 du registre ; § « CTR-06 et CTR-07 : l'HYBRIDE a changé entre mars 2025 et janvier 2026 ») ; `anomalies/informations-a-verifier.md` (ligne VER-04) ; `commercial/hybride.md` (§ L'HYBRIDE de mars 2025).
- **Piège** : deux éditions d'un même document produit dont les blocs de texte sont repris d'une édition à l'autre ; réponse dépendante de la date de fabrication. Le modèle retient une seule valeur (0,8 « parce que c'est le plus récent ») pour un remplacement à l'identique, ou attribue le 0,8 à l'HYBRIDE76.
- **Réponses fausses tentantes** : « 0,8 W/m²K et 76 mm » ; « Uw 1,3 » (dépliant général de 2023) ; « 72 mm et 0,8 » (mélange des deux documents) ; « 0,8 pour l'HYBRIDE76 et 1,2 pour l'HYBRIDE70 » (déclinaisons non sourcées) ; « 5 chambres = système 70 ».
- **Vérifié** :
  - `gammes/hybride.md` l.69 : `| jusqu'en 2025 | gamme unique HYBRIDE | 72 | 1,2 | … (**CTR-06**, **CTR-07**, **CTR-13**) |` ; l.72 : `| à partir de janvier 2026 | gamme HYBRIDE, déclinaison non précisée | 70 à 76 | 0,8 | [1 p. 10, 11] |` ; l.58 : « Les fabrications antérieures à 2026 reposent sur un profilé de 72 mm » ; l.74 : « Le Uw de 0,8 W/m²K est donné pour la gamme … sans dire s'il vaut pour l'HYBRIDE70, l'HYBRIDE76 ou les deux ».
  - `anomalies/contradictions-entre-sources.md` l.119 (CTR-06) : brochure mars 2025 p. 2 « Profilé PVC épaisseur 72mm » contre catalogue p. 10 « 70 à 76 mm », valeur à retenir « 70 à 76 mm, source la plus récente » ; l.120 (CTR-07) : « Uw jusqu'à 1.2 W/m²K » contre « jusqu'à 0,8 W/m²K », valeur à retenir « 0,8 W/m²K, source la plus récente » ; l.126 (CTR-13) : dépliant HYBRIDE 1,2 contre dépliant général du même mois 1,3 ; l.334 : « le Uw de 1,2 W/m²K reste la valeur des menuiseries déjà livrées » ; l.337 : « À faire confirmer au bureau d'études ».
  - `anomalies/informations-a-verifier.md` l.178 (VER-04) : « une plage 70-76 mm pour la gamme entière ».

### Q05 — Vitrage maximal en PERFORM76 : 48 ou 50 mm ?
- **Catégorie** : contradiction
- **Difficulté** : moyenne
- **Question** : « Épaisseur maxi de vitrage sur une PERFORM76 ? Dans le manuel de fabrication Kömmerling que j'ai, je lis 48 mm, mais un client veut un triple vitrage de 50. »
- **Réponse attendue** : Les sources divergent, entrée CTR-17. Manuel de mise en œuvre profine du système 76 Advanced, registre 2.1.1, p. 1 (PDF p. 15, version janvier 2016) : « de 16 à 48 mm » (et « jusqu'à 48 mm » pour la porte d'entrée). DTA n° 6/16-2334_V5, p. 9 (§ 2.2.3.6) : « Isolant double ou triple jusqu'à 50 mm d'épaisseur » ; le cahier technique PERFORM76, p. 5, offre une parclose d'ouvrant jusqu'à 50 mm. Valeur à retenir en attendant l'arbitrage : 50 mm, valeur du DTA, « pièce opposable ». Donc un triple vitrage de 50 mm est admis, à condition de vérifier la parclose : côté PERFORM76, la parclose d'ouvrant la plus épaisse couvre 50 mm (76515) et celle de dormant 48 mm (76579).
- **Doit signaler** : CTR-17
- **Sources dans le wiki** : `certifications/dta-6-16-2334.md` (§ 2.2.3.6 Vitrage) ; `profiles/systeme-76-profiles-principaux.md` (§ Caractéristiques du système 76 Advanced à joint central, liste) ; `profiles/perform76-parcloses.md` (§ Comment choisir une parclose ; tableaux des parcloses d'ouvrant et de dormant ; § Vitrage de série et limite d'épaisseur) ; `anomalies/contradictions-entre-sources.md` (ligne CTR-17).
- **Piège** : deux documents supplier qui donnent deux bornes pour le même système, avec le manuel plus ancien (2016) que le DTA (2025) ; un troisième chiffre (48 mm) existe aussi côté dormant PERFORM76 pour une autre raison (dernière parclose de dormant), ce qui rend la réponse « 48 » à moitié juste. Le modèle tranche pour 48 (prémisse de la question) ou pour 50 sans citer l'entrée.
- **Réponses fausses tentantes** : « 48 mm, limite du système » ; « 50 mm partout, sans regarder la parclose » ; « 44 mm » (dormant avec rehausseur de parclose, DTA p. 38) ; « 42 mm » (limite du système 70) ; « 41 mm » (INNOSLIDE) ; « 28 mm » (vitrage de série ou limite de la PERFORM+).
- **Vérifié** :
  - `certifications/dta-6-16-2334.md` l.548-550 : « **2.2.3.6 Vitrage.** Isolant double ou triple **jusqu'à 50 mm d'épaisseur** … La borne de 48 mm donnée par le manuel de mise en œuvre profine est l'entrée **CTR-17**. »
  - `profiles/systeme-76-profiles-principaux.md` l.89-90 : « différentes épaisseurs de vitrage ou panneau de remplissage de 16 à 48 mm — borne discutée à l'entrée **CTR-17** ».
  - `anomalies/contradictions-entre-sources.md` l.130 (CTR-17) : « de 16 à 48 mm » (manuel) contre « vitrage jusqu'à **50 mm** » (DTA p. 9) et « parcloses d'ouvrant jusqu'à **50 mm** » (cahier p. 5) ; valeur à retenir « **50 mm**, valeur réglementaire du DTA, qui est la pièce opposable » ; impact « accepté sans vérifier la parclose ».
  - `profiles/perform76-parcloses.md` l.72 : `| 76515 | 50 | 9,5 |` (ouvrant) ; l.106 : `| 76579 | 48 | 10,8 |` (dormant) ; l.47 : « L'ouvrant couvre de 16 à 50 mm de vitrage, le dormant de 28 à 48 mm » ; l.177 : « plafonnée à 50 mm, double ou triple ».

### Q06 — Garantie du laquage d'une HYBRIDE blanche
- **Catégorie** : contradiction
- **Difficulté** : simple
- **Question** : « Le laquage d'une HYBRIDE en blanc 9016 brillant, il est garanti combien de temps ? »
- **Réponse attendue** : Deux valeurs selon le document, entrée CTR-05. Catalogue général janvier 2026, p. 35 : « LUMINE55 et HYBRIDE, couleurs standards » = 10 ans (Qualicoat classe 1), couleurs hors standards = 7 ans ; le blanc 9016 brillant laqué est la couleur standard de l'HYBRIDE. Dépliant HYBRIDE de juin 2023, p. 3 (et brochure PERFORM+ / HYBRIDE+ de 2023, p. 3, pour les gammes +) : « Laquage : 7 ans », sans distinction de couleur. Valeur à retenir en attendant l'arbitrage : la grille du catalogue (hors gammes +), soit 10 ans pour le blanc 9016 brillant ; l'écart de 3 ans est signalé « sous-annoncé ou sur-annoncé selon la gamme ».
- **Doit signaler** : CTR-05
- **Sources dans le wiki** : `garanties/garanties-par-composant.md` (§ Grille des garanties de laquage aluminium ; § Grille de l'HYBRIDE de juin 2023 ; § Grille des gammes PERFORM+ et HYBRIDE+) ; `coloris/coloris-hybride.md` (§ nuancier extérieur, ligne 9016) ; `anomalies/contradictions-entre-sources.md` (ligne CTR-05).
- **Piège** : deux documents de la même gamme à trois ans d'écart ; l'entrée n'a pas de page « gamme » qui la porte (elle n'est pas sur `gammes/hybride.md`) ; ne pas la confondre avec « laquage possible si ouverture extérieure » (condition de fabrication, pas de garantie) ni avec le 25 ans de la LUMINE65.
- **Réponses fausses tentantes** : « 25 ans » (LUMINE65, Qualicoat classe 2) ; « 15 ans » (structure) ; « 7 ans forfaitaires » sans mentionner le catalogue ; « 10 ans » sans mentionner la valeur de 7 ans du dépliant ; « 20 ans ».
- **Vérifié** :
  - `garanties/garanties-par-composant.md` l.123 : `| LUMINE55 et HYBRIDE, couleurs standards | 10 | Qualicoat classe 1 |` ; l.124 : `| LUMINE55 et HYBRIDE, couleurs hors standards | 7 | - |` ; l.207 (grille de l'HYBRIDE de juin 2023, source dépliant HYBRIDE p. 3) : `| Laquage | 7 | - (**CTR-05**) |` ; l.187 (brochure gammes +) : `| Laquage | 7 | - (**CTR-05**) |`.
  - `coloris/coloris-hybride.md` l.66 : « La couleur standard est le blanc 9016 brillant laqué » ; l.71 : `| 9016 | Blanc 9016 brillant | laqué, couleur standard | extérieur | brillant |`.
  - `anomalies/contradictions-entre-sources.md` l.118 (CTR-05) : catalogue p. 35 « 10 ans standards, 7 ans hors standards » contre dépliant HYBRIDE juin 2023 p. 3 « Laquage : 7 ans », valeur à retenir « La grille du catalogue hors gammes + ».

### Q07 — Chambres et joints de la PERFORM70
- **Catégorie** : piège de nommage
- **Difficulté** : difficile
- **Question** : « Combien de chambres et de joints sur le profil de la PERFORM70 ? Je pars du principe que c'est le système 70 Kömmerling, donc 5 chambres. »
- **Réponse attendue** : Le wiki ne le donne pas pour la PERFORM70. Le catalogue ne décrit que la PERFORM76 (« 3 joints & 6 chambres ») ; pour la PERFORM70 il donne seulement un profil de 70 mm et un Uw « jusque 1,3 W/m²K » : « le nombre de chambres et de joints de la PERFORM70 n'est pas donné » (entrée VER-03). Ce qui est sourcé par ailleurs : le système 70 Plateforme de profine (e.VOLUTION chez KÖMMERLING, e.MOTION chez KBE, e.XCLUSIVE chez TROCAL) est « à 5 chambres d'isolation », 70 mm. Mais aucun document PROFERM n'établit que la PERFORM70 (ni l'HYBRIDE70) repose sur ce système (VER-28) : le « 5 chambres » ne peut donc pas être affirmé pour la PERFORM70 ; il peut seulement être donné comme caractéristique du système 70 Plateforme.
- **Doit signaler** : VER-28 et VER-03
- **Sources dans le wiki** : `gammes/perform.md` (§ Caractéristiques, tableau et phrase qui suit) ; `profiles/systeme-70-profiles-et-renforts.md` (§ Le système 70 Plateforme, 1er paragraphe ; § Caractéristiques du système) ; `fournisseurs/kommerling.md` (§ Les deux systèmes KÖMMERLING du corpus) ; `anomalies/informations-a-verifier.md` (lignes VER-03 et VER-28).
- **Piège** : trap de nommage. La question emploie le nom PROFERM (PERFORM70) et une prémisse qui relie à un système profine que le wiki décrit en détail (371 pages) mais dont le lien avec la gamme n'est écrit par aucun document PROFERM. Le frontmatter `systeme: [70, 76]` de `gammes/perform.md` et la phrase de `fournisseurs/kommerling.md` l.34-35 (« un profilé GREENLINE … est donc un profilé du système 76 Advanced ») poussent à affirmer un lien.
- **Réponses fausses tentantes** : « 5 chambres » (système 70 Plateforme, ou « 5 chambres d'isolation » de l'HYBRIDE) ; « 6 chambres et 3 joints » (PERFORM76) ; « double joint de frappe PCE » (système 70) ; « c'est le système 76 Advanced » ; « oui c'est bien l'e.VOLUTION, donc 5 chambres ».
- **Vérifié** :
  - `gammes/perform.md` l.61 : `| PERFORM70 | 70 | - | - | 1,3 |` ; l.67-68 : « Le nombre de chambres et de joints de la PERFORM70 n'est pas donné. » ; l.62 : `| PERFORM76 | 76 | 6 | 3 | 0,8 |`.
  - `profiles/systeme-70-profiles-et-renforts.md` l.44-46 : « système PVC de 70 mm d'épaisseur, à 5 chambres d'isolation, commercialisé sous trois marques : e.VOLUTION chez KÖMMERLING, e.MOTION chez KBE, e.XCLUSIVE chez TROCAL » ; l.50 : « **Aucun document PROFERM n'établit que c'est le système des gammes PERFORM70 et HYBRIDE70.** » ; l.89 : « Système à 5 chambres d'isolation. »
  - `fournisseurs/kommerling.md` l.99-101 : « Le rattachement des gammes PERFORM70 et HYBRIDE70 à ce système n'est établi par aucun document PROFERM — entrée **VER-28** ».
  - `anomalies/informations-a-verifier.md` l.177 (VER-03) : « rien ne dit que la PERFORM70 repose dessus — la réponse dépend de `VER-28` » ; l.191 (VER-28) : « Aucun document PROFERM ne nomme le système 70 ni ne cite une référence en 6xxx ».

### Q08 — INNOSLIDE et système 76
- **Catégorie** : piège de nommage
- **Difficulté** : difficile
- **Question** : « Mon INNOSLIDE est bien monté sur le système 76 Advanced ? Je peux prendre les cotes de débit du 76 pour le couper ? »
- **Réponse attendue** : Le wiki ne permet pas de l'affirmer, et ne donne aucune cote de débit ni référence de profilé pour l'INNOSLIDE. Sourcé : l'INNOSLIDE est le coulissant PVC à frappe de PROFERM, présenté dans la gamme PERFORM (catalogue p. 8, dépliant INNOSLIDE janvier 2024) ; quincaillerie Roto Patio Inowa ; A*4 / E*7A / V*B3 ; Uw 1,3 W/m²K ; vitrage jusqu'à 41 mm ; largeur 1 500 à 3 200 mm (4 200 mm dormant ébavuré), hauteur 2 400 mm. Non sourcé : le système de profilé et le fabricant ; aucune page du dépliant ne nomme un système (70 ou 76), et le rattachement de l'INNOSLIDE au système 76 (`systeme: 76` de ses pages) « n'est écrit par aucune source » ; ALUPLAST est crédité des photos, sans preuve de fourniture (VER-15, entrée laissée ouverte). Le DTA du 76 Advanced vise la « fenêtre à la française, oscillo battante ou à soufflet ». Donc ne pas appliquer les cotes de débit du 76 : à faire confirmer par le service achats / bureau d'études.
- **Doit signaler** : VER-15
- **Sources dans le wiki** : `gammes/innoslide.md` (frontmatter ; § Définition et déclinaisons ; § Assemblage et vitrage ; § Performances ; § Dimensions limites) ; `anomalies/informations-a-verifier.md` (ligne VER-15 et § qui la commente) ; `certifications/dta-6-16-2334.md` (paragraphe d'ouverture, famille de produit) ; `sources/depliant-innoslide.md` (marge de la p. 4).
- **Piège** : trap de nommage avec prémisse fausse. Le champ `systeme: [76, Roto Patio Inowa]` de `gammes/innoslide.md` (et `systeme: 76` de `commercial/innoslide.md`) affirme un lien que le corps de la page et VER-15 disent non sourcé ; la page dit aussi « rattaché à la gamme PERFORM », dont la variante 76 est sur le 76 Advanced. Le modèle enchaîne « INNOSLIDE = PERFORM = KÖMMERLING = 76 Advanced » et sort les cotes de débit de `profiles/systeme-76-cotes-de-debit.md`.
- **Réponses fausses tentantes** : « oui, INNOSLIDE = 76 Advanced » ; « non, c'est du système 70 » ; « c'est du ALUPLAST » ; cotes de débit du système 76 (fenêtres) appliquées au coulissant ; « vitrage jusqu'à 50 mm » (limite du 76) ; Uw 0,8 (PERFORM76).
- **Vérifié** :
  - `gammes/innoslide.md` l.7 : `systeme: [76, Roto Patio Inowa]` ; § Définition : « coulissant PVC à frappe de PROFERM, présenté dans la gamme PERFORM » ; l.89 : « L'épaisseur de vitrage va jusqu'à 41 mm » ; l.101 : « Uw 1,3 W/m²K » ; A*4 / E*7A / V*B3 (§ Performances) ; tableau des Dimensions limites (1 500, 3 200, 4 200, 2 400).
  - `anomalies/informations-a-verifier.md` l.245 (VER-15) : « aucune page du dépliant ne nomme le fabricant du profilé ni un système (70 ou 76). Le rattachement de l'INNOSLIDE au système 76 (`systeme: 76` des pages INNOSLIDE) n'est écrit par aucune source non plus. Entrée laissée ouverte ».
  - `certifications/dta-6-16-2334.md` l.34-35 : « Famille de produit : fenêtre à la française, oscillo battante ou à soufflet en PVC ».
  - Recherche : `grep -i innoslide` hors pages INNOSLIDE / coloris / Roto Patio / registres : aucune page de profilé, de cote de débit ni de système ; `grep -i "coulissant\|inowa\|patio"` sur `certifications/dta-6-16-2334.md` et `profiles/systeme-76-cotes-de-debit.md` : aucune occurrence.

### Q09 — Facteur solaire et transmission lumineuse d'une PERFORM76
- **Catégorie** : absence
- **Difficulté** : simple
- **Question** : « Pour une note de calcul RE2020 il me faut le facteur solaire Sw et la transmission lumineuse TLw d'une PERFORM76 avec le double vitrage de série. Tu les as ? »
- **Réponse attendue** : Non, le wiki ne donne ni Sw ni TLw pour les gammes PVC (PERFORM70 / 76, HYBRIDE, TEXTURAL, INNOSLIDE). Ce qu'il donne pour la PERFORM76 : Uw « jusque 0,8 W/m²K » (catalogue p. 7) et, pour le double vitrage de série 28 mm (6 / 18 argon / 4, warm edge, intercalaire TGI noir), Ug 1,1 W/m²K ; le triple vitrage 4 / 14 / 4 / 14 / 4 est à Ug 1,0. Sw et TLw n'existent au wiki que pour des coulissants aluminium (LUMÉAL55 : Sw 0,46 et TLw 0,65 ; « coulissant standard » vitrage 6/14/4 : Sw 0,51 et TLw 0,57), qui ne se transposent pas. Le glossaire définit Sw et TLw mais sans valeur pour les gammes PVC.
- **Doit signaler** : rien
- **Sources dans le wiki** : `gammes/perform.md` (§ Performances) ; `vitrages/performances-vitrages.md` (§ Vitrages thermiques et triples) ; `profiles/perform76-parcloses.md` (§ Vitrage de série et limite d'épaisseur) ; `reference/glossaire.md` (tableau Sw / TLw) ; `gammes/coulissants-aluminium.md` (valeurs Sw / TLw des seuls coulissants aluminium, à ne pas reprendre).
- **Piège** : le glossaire définit Sw et TLw « sur les fiches produit » et une page du wiki en donne des valeurs (aluminium) : le modèle les recopie. Zone traitée (gammes PVC) : ce n'est pas une lacune de couverture, la source ne les donne pas.
- **Réponses fausses tentantes** : « Sw 0,46 et TLw 0,65 » (LUMÉAL55) ; « Sw 0,51 et TLw 0,57 » (coulissants standard 6/14/4) ; « Sw = 0,6 » ou toute valeur « typique » de double vitrage argon (hors wiki) ; « Ug 1,1 » présenté comme facteur solaire ; Gtot(i) (facteur solaire vitrage + store, stores intégrés).
- **Vérifié** :
  - `gammes/perform.md` l.169 : « Uw jusque 1,3 W/m²K en PERFORM70 et jusque 0,8 W/m²K en PERFORM76 ».
  - `vitrages/performances-vitrages.md` l.46 : double vitrage standard 6 / 18 / 4, Ug `1,1 (gammes PVC)` ; l.48 : triple vitrage `| 4 / 14 / 4 / 14 / 4 | … | 1,0 |`.
  - `gammes/coulissants-aluminium.md` l.171 : `| Sw (facteur solaire) | 0,46 |` ; l.172 : `| TLw (transmission lumineuse) | 0,65 |` ; l.286 : « coulissant standard avec vitrage 6/14/4, Sw = 0,51 et TLw = 0,57 ».
  - `reference/glossaire.md` l.70 : `| Sw | facteur solaire de la fenêtre : part de l'énergie solaire transmise |`.
  - Absence : `grep -i -E "facteur solaire|\bSw\b|TLw|transmission lumineuse|apports solaires"` sur tout le wiki hors `log.md` / `index.md` : occurrences uniquement dans `gammes/coulissants-aluminium.md`, `commercial/lumine.md` (renvoi), `reference/glossaire.md`, `equipements/stores-integres.md` et `sources/nuancier-stores.md` (Gtot(i)), `anomalies/incoherences-internes.md` (INC-02, coulissants) ; aucune ligne sur PERFORM, HYBRIDE, TEXTURAL, INNOSLIDE, PERFORM+ ou HYBRIDE+ (essais aussi sur « S_w », « g = »).

### Q10 — Épaisseur de paroi des profilés PVC en mm
- **Catégorie** : absence
- **Difficulté** : moyenne
- **Question** : « Le catalogue dit que l'épaisseur de parois est 10 à 15 % supérieure à la moyenne du marché. C'est combien de mm sur les profilés de la PERFORM76, et quelle classe de profilé ? »
- **Réponse attendue** : Le wiki ne donne ni l'épaisseur de paroi des profilés PVC (en mm), ni sa classe, ni la « moyenne du marché » de référence. La seule mention est une phrase du catalogue (p. 11) : « L'épaisseur de parois des menuiseries PROFERM est 10 à 15 % supérieure à la moyenne du marché », qui porte sur les menuiseries PROFERM en général et n'est ni chiffrée ni sourcée. Ce que le wiki donne de la PERFORM76 : profil de 76 mm de profondeur à 6 chambres, renfort acier galvanisé tubulaire de 1,5 mm dans le dormant et de 2 mm dans l'ouvrant (épaisseur d'acier, pas de PVC). Ni le DTA, ni le DTD, ni les manuels profine du wiki ne donnent l'épaisseur de paroi.
- **Doit signaler** : rien
- **Sources dans le wiki** : `commercial/hybride.md` (§ Le saviez-vous ? ; § Ce que la source ne chiffre pas) ; `commercial/proferm.md` (ligne « Qualité ») ; `gammes/perform.md` (§ Caractéristiques ; § Coupes de la PERFORM76, repères 3, 4, 5a, 6) ; `certifications/dta-6-16-2334.md` (aucune valeur).
- **Piège** : le chiffre « 10 à 15 % » est un pourcentage sans base ; trois séries de mm voisines existent (76 de profondeur, 1,5 et 2 d'acier, 2,5 d'autres renforts) et se laissent prendre pour une épaisseur de paroi. Zone entièrement traitée (PERFORM76, cahier + DTA + manuel profine) : la donnée n'y figure pas.
- **Réponses fausses tentantes** : « 1,5 mm » ou « 2 mm » (renforts acier du dormant et de l'ouvrant) ; « 2,5 mm » (renforts V308, V319, V322…) ; « 2 mm » (habillage de parois PVC du monobloc de porte THERMIXEL, `portes/panneaux-et-monoblocs.md` l.185, qui n'est pas un profilé de fenêtre) ; « 76 mm » ou « 72 mm » (profondeur du profil) ; « 2,8 mm, classe A » (connaissance générale hors wiki) ; « 10 à 15 % de plus que 2,8 mm » (calcul sur une base inventée).
- **Vérifié** :
  - `commercial/hybride.md` l.80 : « L'épaisseur de parois des menuiseries PROFERM est 10 à 15 % supérieure à la moyenne du marché » ; l.168 : « La « moyenne du marché » de l'épaisseur de parois n'est ni chiffrée ni sourcée ».
  - `commercial/proferm.md` l.66 : « Soudures, finitions, ferrures, épaisseur de parois des menuiseries : PROFERM s'engage à fournir des menuiseries haut de gamme » (aucun chiffre).
  - `gammes/perform.md` l.229 : `| 4 | renfort acier galvanisé tubulaire de 1,5 mm du dormant … |` ; l.232 : `| 6 | renfort acier galvanisé de 2 mm de l'ouvrant |` ; l.228 : dormant « à 6 chambres de 76 mm d'épaisseur ».
  - Absence : `grep -i -E "paroi|cloison|12608|classe S\b|épaisseur (de|des|du) (matière|paroi)|wall"` sur tout le wiki hors `log.md` / `index.md` : occurrences = `commercial/hybride.md` l.4, 80, 168 ; `commercial/proferm.md` l.66 ; pages de pose ou d'assemblage où « paroi » désigne une paroi de renfort ou un mur (`procedures/accouplement-elements-systeme-70.md`, `equipements/*`) ; `portes/panneaux-et-monoblocs.md` l.185 (habillage de parois PVC de 2 mm du monobloc de porte THERMIXEL) ; aucune valeur en mm ni classe de profilé PVC de fenêtre. Tableau des propriétés du PVC des directives e.VOLUTION 2008 (`procedures/directives-generales-systeme-70-evo2008.md` l.50-68) : masse volumique, résilience, Vicat… sans épaisseur de paroi.

### Q11 — Parclose 76526 : 28 mm en PERFORM76, pas en système 70
- **Catégorie** : piège 70/76
- **Difficulté** : simple
- **Question** : « Sur un châssis en système 70 Kömmerling, j'ai un double vitrage de 28 mm. Je prends la parclose 76526, comme sur mes PERFORM76 ? »
- **Réponse attendue** : Non. En système 70, la parclose 76526 ne tient pas 28 mm : elle tient 20 mm (plage admise 19,5 à 21 mm) dans la feuillure de 54 mm, et 36 mm (35,5 à 37 mm) dans la feuillure de 70 mm obtenue avec l'élargisseur 93025. Le 28 mm est la valeur de la 76526 sur le système 76 : 28 mm au cahier technique PERFORM76 (parclose d'ouvrant, cote 29,5 mm), 28 mm avec le joint A de 4 mm et 30 mm avec le joint B de 2 mm au manuel du système 76. Pour un vitrage de 28 mm en système 70 (feuillure 54 mm), la parclose est la 76503 (28 mm, plage 27,5 à 29 mm) ; les 0135, 1512 et 6148 sont écrites sous la 76503 pour la même épaisseur.
- **Doit signaler** : rien
- **Sources dans le wiki** : `/profiles/systeme-70-tableau-de-vitrage.md` (sections « Parcloses pour une feuillure de 54 mm » et « Parcloses pour une feuillure de 70 mm avec l'élargisseur 93025 ») ; `/profiles/perform76-parcloses.md` (« Cotes des parcloses d'ouvrant ») ; `/profiles/systeme-76-tableau-de-vitrage.md` (« Parcloses d'ouvrant »).
- **Piège** : la même référence existe dans les deux systèmes avec une épaisseur de vitrage très différente ; 28 mm est justement le vitrage de série de la PERFORM76, donc la prémisse (« comme sur mes PERFORM76 ») est fausse pour le système 70. Prémisse erronée à corriger, sans fabriquer une réponse sur le 76.
- **Réponses fausses tentantes** : « Oui, la 76526 tient 28 mm » ; « 36 mm » (valeur feuillure 70 prise pour la feuillure 54) ; « 30 mm » (joint B du système 76) ; « 44 mm » ; confondre avec la 76503 sans dire qu'elle est la bonne pour 28 mm en 70.
- **Vérifié** : `/profiles/systeme-70-tableau-de-vitrage.md` — ligne feuillure 54 : « | 76526 | en gras | 20 | 19,5 | 21 | » ; ligne feuillure 70 : « | 76526 | en gras | 36 | 35,5 | 37 | » ; ligne 28 mm : « | 76503 | en gras | 28 | 27,5 | 29 | ». `/profiles/perform76-parcloses.md` — « | 76526 | 28 | 29,5 | ». `/profiles/systeme-76-tableau-de-vitrage.md` — « | 76526 | 28 | 30 | +1,0 / −0,5 | ».

### Q12 — Essai NF P 20-302 : 62 kg en système 70, 60 kg en 76
- **Catégorie** : piège 70/76
- **Difficulté** : difficile
- **Question** : « Vantail de 61 kg avec 10 mm de verre cumulés : en système 70 Kömmerling, je dois faire justifier la tenue mécanique par essai (NF P 20-302) ? Et en système 76, c'est pareil ? »
- **Réponse attendue** : Système 70 : non. Le DTD 6/16-2335 (§ 2.3) n'impose la démonstration par voie expérimentale (NF P 20-302, dans la limite des charges maximum de la quincaillerie) que pour une épaisseur de verre supérieure à 12 mm ou une masse de vantail supérieure à 62 kg ; 61 kg et 10 mm restent en dessous. Système 76 : oui. Le DTA 6/16-2334 (§ 2.3) fixe le même seuil de 12 mm de verre mais une masse de vantail supérieure à 60 kg ; 61 kg le dépasse. La seule différence entre les deux systèmes est donc 62 kg contre 60 kg.
- **Doit signaler** : rien (aucune anomalie n'enregistre l'écart 62 kg / 60 kg ; chaque seuil est celui du document de son système)
- **Sources dans le wiki** : `/certifications/dtd-6-16-2335.md` (« 2.3 Disposition de conception ») ; `/certifications/dta-6-16-2334.md` (« 2.3 Disposition de conception ») ; `/profiles/perform76-parcloses.md` (« Vitrage de série et limite d'épaisseur », reprend le seuil de 60 kg).
- **Piège** : même règle, même norme, même seuil de 12 mm, un seul chiffre qui change ; les deux seuils sont dans des pages différentes et l'assistant généralise volontiers 60 kg (page PERFORM76, plus visible) au système 70. Le cas de 61 kg est choisi pour que la réponse s'inverse entre les deux systèmes. Note pour le relecteur : 62 kg n'est pas une valeur contredite dans le wiki, mais aucun registre ne la confirme non plus ; à recontrôler sur le PDF si le cas est utilisé comme référence stricte.
- **Réponses fausses tentantes** : « Oui dans les deux cas » (60 kg généralisé) ; « Non dans les deux cas » ; « seuil de 62 kg au 76 » ; « seuil de 12 mm sur le poids » ; oublier la condition « épaisseur de verre » (10 mm ne la déclenche pas).
- **Vérifié** : `/certifications/dtd-6-16-2335.md` — « Dans le cas de vitrages d'épaisseur de verre supérieure à 12 mm ou de masse de vantail supérieure à 62 kg, le fabricant devra s'assurer, par voie expérimentale… ». `/certifications/dta-6-16-2334.md` — « supérieure à **12 mm** ou de masse de vantail supérieure à **60 kg**, le fabricant devra s'assurer, par voie expérimentale… ». `/profiles/perform76-parcloses.md` — « Au-delà de **12 mm d'épaisseur de verre** ou de **60 kg de masse de vantail** ».

### Q13 — Profilé de liaison au dos du dormant : axe de vissage 30 mm ou 25 mm
- **Catégorie** : piège 70/76
- **Difficulté** : moyenne
- **Question** : « Pour coupler deux dormants avec un profilé de liaison au dos du dormant, à quelle distance de la face intérieure je place la ligne de vis ? »
- **Réponse attendue** : Cela dépend du système, et la réponse doit donner les deux. Système 70 (couplage vertical 180° par dos de dormant, profilés de liaison 1248 et 70601) : axe de vissage dans le plan du profil, pour la liaison dormant / dormant, à 30 mm de l'intérieur. Système 76 (profilés de liaison 76606 et 76604) : axe de vissage à 25 mm de l'intérieur. Les mêmes trous sont prescrits dans les deux systèmes (prépercer Ø 4 mm le dormant A et la 1re paroi du renfort du dormant B, puis agrandir à Ø 6 mm le trou du dormant A). Exception à ne pas généraliser au 76 : pour le joint de couplage G022, le texte donne 25 mm mais la coupe cote 26 mm (INC-72) ; pour le renfort V477 l'axe est à 26 mm.
- **Doit signaler** : rien d'obligatoire (INC-72 seulement si le G022 est cité)
- **Sources dans le wiki** : `/procedures/accouplement-elements-systeme-70.md` (§ 1.1 « Couplage vertical 180° avec profilé de liaison 1248 » et § 1.2 « 70601 ») ; `/procedures/accouplement-elements-systeme-76.md` (« Couplage vertical 180° avec profilé de liaison 76606 » et « 76604 » ; « G022 » pour INC-72).
- **Piège** : le nom (« profilé de liaison », « couplage vertical 180° par dos de dormant ») et la mise en œuvre sont identiques dans les deux systèmes ; seule la cote diffère de 5 mm. La question ne nomme aucun système : une réponse qui n'en donne qu'un seul est fausse. Ne pas confondre avec l'entraxe des vis le long de la barre (150 / 300 mm, directives générales) qui est une autre cote.
- **Réponses fausses tentantes** : « 25 mm » seul ou « 30 mm » seul ; « 150 mm » (première vis depuis le coin, directives générales des couplages) ; « 26 mm » généralisé au 76 ; donner l'axe du poteau d'angle à la place.
- **Vérifié** : `/procedures/accouplement-elements-systeme-70.md` — « Axe de vissage dans le plan du profil pour la liaison dormant / dormant à 30 mm de l'intérieur » (1248) ; même formule « à **30 mm de l'intérieur** » sous le 70601. `/procedures/accouplement-elements-systeme-76.md` — 76606 : « axe de vissage dans le plan du profil pour la liaison dormant / dormant à **25 mm de l'intérieur** » ; 76604 : « Axe de vissage dans le plan du profil à **25 mm de l'intérieur** » ; V477 : « **26 mm de l'intérieur** » ; G022 : « axe de vissage à 25 mm de l'intérieur » (coupe à 26 mm, INC-72).

### Q14 — Fenêtre 1 vantail à la française en 70 : 1 200 × 1 600, l'abaque m'a donné un repère
- **Catégorie** : faisabilité
- **Difficulté** : difficile
- **Question** : « Fenêtre 1 vantail à la française en système 70 Kömmerling, 1 200 mm de large sur 1 600 mm de haut. L'abaque de renfort d'ouvrant (V*A2) me donne bien un repère pour cette taille, donc je peux la fabriquer telle quelle ? »
- **Réponse attendue** : Non, pas en fabrication non certifiée. Le DTD 6/16-2335 (§ 2.2.3.8, dimensions maximales de baie H × L) limite la fenêtre française 1 vantail à 2,15 m de haut × 1,00 m de large : 1,20 m de large dépasse de 0,20 m (la hauteur de 1,60 m passe). L'abaque de renforcement (registre 2.3.3) est établi en dimensions d'ouvrant, pas de baie : il dit quel profilé renforcer, il n'ouvre pas le domaine du DTD (sur l'abaque V*A2 des ouvrants 6115 / 6123 / 6119 / 6152 en OF 1 vantail, la case ouvrant 1 100–1 200 × 1 500–1 600 porte bien le repère D4). Seule exception : pour les fabrications certifiées, des dimensions supérieures peuvent être envisagées ; elles sont alors précisées dans le Certificat de Qualification attribué au menuisier.
- **Doit signaler** : rien
- **Sources dans le wiki** : `/certifications/dtd-6-16-2335.md` (« 2.2.3.8 Dimensions maximales (Baie H × L) en m » et paragraphe qui suit) ; `/profiles/systeme-70-abaques-dimensionnels.md` (introduction : « dimensions d'ouvrant, pas … de baie ») ; `/profiles/systeme-70-abaques-va2.md` (« VA2 — OF 1 vantail, ouvrants 6115/6123/6119/6152 »).
- **Piège** : le document DTD (limite de baie) et le manuel (abaque d'ouvrant) ne mesurent pas la même chose ; la prémisse de l'utilisateur prend la présence d'un repère d'abaque pour une autorisation. Inversion H × L possible dans le tableau du DTD (H est en premier). Il faut aussi donner l'exception « fabrication certifiée / Certificat de Qualification » et ne pas répondre « sur étude » sans source.
- **Réponses fausses tentantes** : « Oui, l'abaque donne D4 » ; « Non car 1,60 m de haut dépasse » (axes inversés) ; limite 1,60 m de large (2 vantaux OB) ; limite du soufflet 0,80 × 1,30 ; limites du système 76 reprises telles quelles sans dire qu'elles sont identiques (2,15 × 1,00) mais d'un autre document (DTA 6/16-2334).
- **Vérifié** : `/certifications/dtd-6-16-2335.md` — « | Française 1 vantail | 2,15 | 1,00 | » (colonnes H (m), L (m)) ; « Pour les fabrications certifiées, des dimensions supérieures peuvent être envisagées. Elles sont alors précisées dans le Certificat de Qualification ». `/profiles/systeme-70-abaques-dimensionnels.md` — « Les dimensions portées sur les abaques sont des dimensions d'ouvrant … pas des dimensions de baie. » `/profiles/systeme-70-abaques-va2.md` — ligne « 1 500 à 1 600 | 3 | C3 | C3 | C3 | C3 | D3 | D4 | D4 | D4 | D4 » (colonne 1 100 à 1 200 = D4).

### Q15 — Vitrage de 52 mm sur PERFORM76 « avec la 76515 et le joint de 2 mm »
- **Catégorie** : faisabilité
- **Difficulté** : difficile
- **Question** : « Un triple vitrage de 52 mm d'épaisseur sur un ouvrant PERFORM76 : le manuel profine le prévoit avec la parclose 76515 et le joint de 2 mm, donc c'est bon ? »
- **Réponse attendue** : Non. L'épaisseur de vitrage du procédé est plafonnée à 50 mm (double ou triple) par le DTA 6/16-2334 (§ 2.2.3.6 : « Isolant double ou triple jusqu'à 50 mm d'épaisseur »), et la 76515 est la parclose d'ouvrant la plus étroite du cahier technique PERFORM76, pour 50 mm (cote 9,5 mm). Le tableau de vitrage du manuel du système 76 affiche bien 50 mm avec le joint A de 4 mm et 52 mm avec le joint B de 2 mm pour la 76515 (tolérance ±0,5), mais le même manuel annonce ailleurs « de 16 à 48 mm » : contradiction CTR-17, dont la valeur retenue est 50 mm, celle du DTA, pièce opposable. 52 mm sort donc du domaine d'emploi ; le wiki ne donne aucune exception à cette borne (l'exception « fabrication certifiée » ne porte que sur les dimensions de baie).
- **Doit signaler** : CTR-17
- **Sources dans le wiki** : `/certifications/dta-6-16-2334.md` (« 2.2.3.6 Vitrage ») ; `/profiles/perform76-parcloses.md` (« Cotes des parcloses d'ouvrant » ; « Vitrage de série et limite d'épaisseur ») ; `/profiles/systeme-76-tableau-de-vitrage.md` (« Parcloses d'ouvrant ») ; `/anomalies/contradictions-entre-sources.md` (CTR-17).
- **Piège** : trois documents, trois chiffres (48 mm au texte du manuel, 50 mm au DTA et au cahier, 52 mm dans une colonne du tableau du manuel). Le tableau du manuel « autorise » 52 mm mais la borne réglementaire est 50 mm. Il faut donner la borne retenue, sa source, et signaler CTR-17.
- **Réponses fausses tentantes** : « Oui, 52 mm avec la 76515 et le joint B » ; « la limite est 48 mm » (manuel seul, sans dire que le DTA dit 50) ; « 50 mm passe donc 52 aussi avec un joint plus fin » ; oublier l'identifiant CTR-17.
- **Vérifié** : `/certifications/dta-6-16-2334.md` — « Isolant double ou triple **jusqu'à 50 mm d'épaisseur** ». `/profiles/perform76-parcloses.md` — « | 76515 | 50 | 9,5 | » ; « L'épaisseur de vitrage du procédé est plafonnée à 50 mm ». `/profiles/systeme-76-tableau-de-vitrage.md` — « | 76515 | 50 | 52 | ±0,5 | ». `/anomalies/contradictions-entre-sources.md` — CTR-17 : manuel « de 16 à 48 mm » contre DTA « 50 mm », valeur à retenir « 50 mm, valeur réglementaire du DTA, qui est la pièce opposable ».

### Q16 — Largeur hors tout du dormant large 6110
- **Catégorie** : contradiction
- **Difficulté** : moyenne
- **Question** : « Quelle est la largeur hors tout du dormant large 6110 en système 70 ? Je dois commander l'élargisseur qui va avec. »
- **Réponse attendue** : 145 mm selon le manuel de mise en œuvre Système 70 Plateforme (sept. 2023, registre 2.1.2), le poster Kömmerling Gamme 70 (mars 2025) et le classeur e.VOLUTION d'août 2008 ; 135 mm selon le DTD 6/16-2335 (p. 15, planche des dormants, cote horizontale prise de l'extrémité de l'aile à la face gauche du pied droit). L'écart est de 10 mm sur les cinq dormants larges (6108 : 105 / 95, 6109 : 125 / 115, 6110 : 145 / 135, 6111 : 165 / 155, 6158 : 210 / 200). Contradiction CTR-20, non arbitrée ; valeur à retenir en attendant : celle du manuel de fabrication (145 mm), qui sert déjà aux cotes de débit ; l'écart constant suggère une convention de mesure différente non énoncée. À confirmer auprès de profine avant de commander un élargisseur ou une pièce d'appui.
- **Doit signaler** : CTR-20
- **Sources dans le wiki** : `/profiles/systeme-70-dormants.md` (« Dormants larges 6108 à 6111 et 6158 » ; « Dormants et capots du DTD » ; « Dormants des plans e.VOLUTION de 2008 ») ; `/anomalies/contradictions-entre-sources.md` (CTR-20).
- **Piège** : quatre documents pour deux valeurs ; la cote « hors tout » du manuel n'est pas la « cote horizontale en pied » du DTD, et l'assistant qui ne lit que le DTD (document réglementaire, plus autoritaire d'apparence) donne 135. Il faut les deux valeurs, chacune avec son document, et l'identifiant.
- **Réponses fausses tentantes** : « 145 mm » seul, sans le 135 du DTD ; « 135 mm » seul ; « 70 mm » (épaisseur du corps) ; « 105 mm » (6108) ; 96 mm (ancienne valeur de fiche source pour le 6108).
- **Vérifié** : `/profiles/systeme-70-dormants.md` — « | 6110 | 145 | 70 | 20 + 64 | 20 | 10 | V601, V544 | » (poster) ; « | 6110 | 135 | 64 | 20 + 64 | » (DTD) ; « | 6110 | F91-02- 6110 | 145 hors tout ; 70 pour le corps » (2008) ; « le DTD leur donne 10 mm de moins — entrée **CTR-20** ». `/anomalies/contradictions-entre-sources.md` — CTR-20 : « 6110 à 145 … DTD … 6110 à 135 ».

### Q17 — Triple vitrage à 13 mm de verre et abaques de renfort d'ouvrant du système 70
- **Catégorie** : contradiction
- **Difficulté** : difficile
- **Question** : « Sur un système 70, je monte un triple vitrage 5/12/4/12/4, soit 13 mm de verre. Je suis encore dans le domaine des abaques de renfort d'ouvrant ? »
- **Réponse attendue** : Ça dépend du document, et il n'y a pas de valeur unique retenue (CTR-80). Manuel Système 70 Plateforme, septembre 2023 (registre 2.3.3, p. 1) : les abaques sont établis pour un vitrage ne dépassant pas 35 kg/m², soit 14 mm de verre ; à 2,5 kg/m² par mm, 13 mm de verre font 32,5 kg/m² : dans le domaine. Classeur e.VOLUTION d'août 2008 (registre 4.1, p. 1) : 30 kg/m², soit 12 mm de verre maxi ; 13 mm est hors du domaine, et « au-delà de ce poids, un renforcement systématique est conseillé ». À titre complémentaire, le DTD 6/16-2335 (§ 2.3) demande une justification expérimentale NF P 20-302 au-delà de 12 mm de verre.
- **Doit signaler** : CTR-80
- **Sources dans le wiki** : `/profiles/systeme-70-abaques-dimensionnels.md` (« Remarques importantes concernant les abaques de renforcement » et « Remarques des plans e.VOLUTION de 2008 ») ; `/profiles/systeme-70-abaques-evo2008.md` (introduction) ; `/anomalies/contradictions-entre-sources.md` (CTR-80) ; `/certifications/dtd-6-16-2335.md` (« 2.3 Disposition de conception »).
- **Piège** : le vitrage de 13 mm tombe exactement entre les deux limites (12 et 14 mm) ; l'assistant qui ne lit qu'une édition répond oui ou non avec certitude. Le poids se déduit de l'épaisseur de verre (2,5 kg/m² par mm, formule donnée par la page) et l'épaisseur de verre s'entend sans les lames d'air.
- **Réponses fausses tentantes** : « Oui, jusqu'à 35 kg/m² » sans la limite de 2008 ; « Non, 12 mm maxi » (2008 seul) ; compter 4+12+4+12+4 = 36 mm comme épaisseur de verre ; confondre le 12 mm des abaques 2008 avec le 12 mm du seuil d'essai du DTD.
- **Vérifié** : `/profiles/systeme-70-abaques-dimensionnels.md` — « établis pour un vitrage ne dépassant pas 35 kg/m², soit 14 mm de verre au maximum » ; « établis pour un vitrage ne dépassant pas **30 kg/m², soit 12 mm de verre maxi** » ; « 2,5 kg/m² par mm d'épaisseur ». `/anomalies/contradictions-entre-sources.md` — CTR-80 : « Pas de valeur retenue seule : 30 kg/m² … 2008, 35 kg/m² … 2023 » ; impact « 13 ou 14 mm de verre (32,5 à 35 kg/m²) ».

### Q18 — Meneau de dormant à clair intérieur de 68 mm
- **Catégorie** : lecture de tableau
- **Difficulté** : moyenne
- **Question** : « Sur une PERFORM76, je cherche le meneau de dormant qui donne 68 mm de clair intérieur : quelle référence, quelle largeur, et tu peux me montrer la coupe ? »
- **Réponse attendue** : Le meneau 76373 : largeur 124 mm, décomposition 28 / 68 / 28 mm (ailes de 28 mm). Coupe : `/assets/profiles/perform76/meneaux/meneau-76373.png`. Ne pas le confondre avec son jumeau d'ouvrant, le 76303 (même clair intérieur de 68 mm mais largeur 110 mm, ailes de 21 mm) : un meneau d'ouvrant ne se monte jamais sur un dormant. À signaler : le sommaire des profilés du manuel profine donne 110 mm au 76373, valeur recopiée de la ligne du 76303 ; la planche de détail et le cahier technique PERFORM76 donnent 124 mm (INC-09).
- **Doit signaler** : INC-09
- **Sources dans le wiki** : `/profiles/perform76-meneaux.md` (tableau d'introduction des couples de meneaux ; « Cotes » ; « Compatibilités ») ; `/anomalies/incoherences-internes.md` (INC-09).
- **Piège** : deux meneaux au même clair intérieur (68 mm), l'un de dormant (124 mm), l'autre d'ouvrant (110 mm), sur des lignes voisines ; le 110 mm figure aussi au sommaire du manuel pour le 76373. L'image demandée est celle du 76373, pas du 76303.
- **Réponses fausses tentantes** : « 76303, 110 mm » ; « 76373, 110 mm » (sommaire du manuel) ; « 98 mm » (76372, clair 42) ; image `meneau-76303.png` ; oublier INC-09.
- **Vérifié** : `/profiles/perform76-meneaux.md` — « | 68 | 76373, 124 mm | 76303, 110 mm | » ; « | 76373 | dormant | 124 | 28 / 68 / 28 | ![Meneau 76373](/assets/profiles/perform76/meneaux/meneau-76373.png) | » ; « | 76303 | ouvrant | 110 | 21 / 68 / 21 | ». Fichier présent : `wiki_llm/wiki/assets/profiles/perform76/meneaux/meneau-76373.png`. `/anomalies/incoherences-internes.md` — INC-09 : « 76373 Meneau de 110 mm » au sommaire, « 76373 Meneau/Traverse 124 mm » sur la planche.

### Q19 — Doublage de 175 mm sur dormant 76171 : tapée, appui, patte de pose
- **Catégorie** : lecture de tableau
- **Difficulté** : moyenne
- **Question** : « Fenêtre PERFORM76 sur un dormant 76171, mur doublé avec 175 mm d'isolant : quelle tapée, quel appui et quelle patte de pose je commande ? Tu as les dessins ? »
- **Réponse attendue** : Tapée 6142 (cote propre 95 mm ; elle donne 175 mm d'isolation sur un 76171, contre 160 mm sur les 76177, 76185 et 76180), appui 6137, patte de pose NT1949 avec la cale CTHNT0030 et le clameau CP14GGOM0012. Attention : au-delà de 155 mm d'isolant sur un 76171, le dormant bas devient un 76180 à aile de 20 mm ; c'est pour cela que l'appui 6137 est possible ici alors que la table de compatibilités le donne « non » sur le 76171. Dessins : tapée `/assets/profiles/perform76/tapees/tapee-6142.png`, appui `/assets/profiles/perform76/appuis/appui-6137.png`, coupe complète `/assets/profiles/perform76/isolation/76171-6142.png`.
- **Doit signaler** : rien
- **Sources dans le wiki** : `/profiles/perform76-tapees-et-isolation.md` (« Cotes des tapées par dormant » ; « La contrainte du 76171 au-delà de 155 mm » ; « Appuis et pattes de pose sur dormant 76171 ») ; `/profiles/perform76-appuis-et-seuils.md` (« Compatibilités », note sous le tableau).
- **Piège** : les colonnes voisines du tableau des tapées donnent d'autres épaisseurs d'isolation pour la même tapée (6142 = 160 mm sur les dormants rénovation et le 76180, 175 mm sur le 76171) ; la ligne 155 mm donne la 6141 et l'appui 76758 ; la table de compatibilités interdit l'appui 6137 sur un 76171 (il faut l'exception du dormant bas 76180). La tapée, l'appui et la patte de pose sont trois familles voisines dans la même ligne.
- **Réponses fausses tentantes** : tapée 6141 (140 mm sur les autres dormants, 155 mm sur le 76171) ; appui 76758 (lignes ≤ 155 mm) ou 76768 (≥ 195 mm) ; patte NT1947 (155 mm) ou NT1951 ; oublier le dormant bas 76180 aile de 20 mm ; annoncer « l'appui 6137 est incompatible avec le 76171 ».
- **Vérifié** : `/profiles/perform76-tapees-et-isolation.md` — « | 6142 | 95 | 160 | 160 | 175 | » ; « | 175 | 6142 | NT1949 | 6137 | CTHNT0030 | » ; « | 175 | 6137 | **76180, aile de 20 mm** | » ; « au-delà de 155 mm d'isolant le dormant bas devient un 76180 à aile de 20 mm ». `/profiles/perform76-appuis-et-seuils.md` — « | Appui 6137 | non | non | oui | oui | oui | » (colonnes 76171, 76172, 76177, 76180, 76185). Fichiers présents : `tapees/tapee-6142.png`, `appuis/appui-6137.png`, `isolation/76171-6142.png`.

### Q20 — Renforts admis dans le dormant 6100 et l'ouvrant 6115 (système 70)
- **Catégorie** : pages découpées
- **Difficulté** : moyenne
- **Question** : « Sur une fenêtre système 70 avec un dormant 6100 et un ouvrant 6115, quels renforts puis-je glisser dans le dormant et dans l'ouvrant, et lequel est le plus rigide au vent dans chaque cas ? »
- **Réponse attendue** : Dormant 6100 : V600 (acier 1,5 mm, IG 0,5 cm⁴, IW 2,1 cm⁴) ou V543 (1,25 mm, IG 0,1, IW 1,5) ; le plus rigide au vent est le V600. Ouvrant 6115 : V057 (2 mm, IG 4,8, IW 3,8), V059 (2 mm, IG 4,8, IW 4,4) ou V069 (renfort pré-usiné livré en 2 m, sans inertie écrite) ; le plus rigide au vent parmi ceux dont l'inertie est donnée est le V059. Les deux parties viennent de deux pages distinctes : les dormants et les ouvrants du système 70 ne sont plus sur la même page.
- **Doit signaler** : rien d'obligatoire ; CTR-38 (IW du V543 : 1,5 au manuel, 1,62 au poster et au DTD) et CTR-69 (Iz de 2008 : V057 3,20, V059 4,50) sont à mentionner seulement si l'assistant cite ces inerties
- **Sources dans le wiki** : `/profiles/systeme-70-dormants.md` (« Renforts des dormants ») ; `/profiles/systeme-70-ouvrants.md` (« Renforts des ouvrants »). Recoupement : `/profiles/systeme-70-plans-de-combinaison-dormants-et-ouvrants.md` (tables de renforts avec Iw).
- **Piège** : la réponse exige deux pages issues du découpage du 30/09/2026 ; un assistant qui n'en ouvre qu'une (ou qui prend l'ancienne page `systeme-70-profiles-et-renforts.md`) répond pour un seul des deux profilés. Voisinage trompeur : le 6112 (ouvrant de 53 mm) porte V158 / V258, le 6116 ne porte pas de V069 au poster ; le dormant 6101 porte V601 / V544.
- **Réponses fausses tentantes** : V158 ou V258 pour le 6115 (renforts du 6112) ; V601 / V544 pour le 6100 (renforts du 6101) ; V069 comme « plus rigide » (aucune inertie écrite) ; un seul renfort par profilé.
- **Vérifié** : `/profiles/systeme-70-dormants.md` — « | 6100 | V600 | Renfort 1,5 mm | 1,5 | 0,5 | 2,1 | - | » ; « | 6100 | V543 | Renfort 1,25 mm | 1,25 | 0,1 | 1,5 | - | ». `/profiles/systeme-70-ouvrants.md` — « | 6115 | V057 | Renfort 2 mm | 2 | 4,8 | 3,8 | - | » ; « | 6115 | V059 | Renfort 2 mm | 2 | 4,8 | 4,4 | - | » ; « | 6115 | V069 | Renfort pré-usiné | - | - | - | 2 | ».

---

## Notes de l'auteur de la partie A (constats hors questions)

- Recoupements écartés : aucun sujet de `wiki_20_questions.json` ni de `wiki_controle_10.json` n'est repris (déjà pris : CTR-01 pivot bas, CTR-03 garantie structure LUMINE, CTR-62 verrou A272, INC-12 dormant 6106, garanties LUMINE, INNOSLIDE SoftOpen, parcloses 3702 / 76576, faisabilités OF / OB PERFORM76, absences Technal / ASKEY / LUMINE65).
- Tension interne du wiki utile pour Q07 : `fournisseurs/kommerling.md` l.34-35 et `fournisseurs/profine.md` l.41-42 écrivent « un profil GREENLINE de chez KÖMMERLING est donc un profilé du système 76 Advanced », alors que VER-28 dit que le système de la PERFORM70 / HYBRIDE70 n'est pas établi. Le frontmatter `systeme: [70, 76]` de `gammes/perform.md`, `hybride.md`, `textural.md` et `systeme: [76, Roto Patio Inowa]` de `gammes/innoslide.md` affirment des liens que VER-28 et VER-15 déclarent non sourcés.
- Entrée périmée probable : VER-26 (`anomalies/informations-a-verifier.md` l.249) écrit « Aucun document PROFERM n'emploie les mots « joint central » ni « joint de frappe » », alors que `gammes/perform.md` l.225 (cahier technique PERFORM76, p. 5) les emploie ; Q02 s'appuie sur la formule du cahier.
- Écart non enregistré, donc non utilisé en question : Uf du système 70 « compris entre 1,9 et 1,3 W/m²K » (directives e.VOLUTION 2008, `procedures/directives-generales-systeme-70-evo2008.md` l.68) contre « jusqu'à Uf = 1,0 » (manuel 2023, `profiles/systeme-70-profiles-et-renforts.md` l.74) : aucune entrée CTR / INC.
- Idée non retenue faute d'univocité : absence du « classement AEV de la PERFORM76 » (le catalogue donne A*4 / E*9A / V*A3 à la PERFORM70, à l'HYBRIDE, à la TEXTURAL et à un « PROFERM, produit non nommé » ; rien de propre à la PERFORM76 ; le DTA 76 ne liste que des rapports d'essai sans classe) ; la ligne « produit non nommé » rend l'absence discutable.
