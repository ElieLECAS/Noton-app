# Plan — mise à jour des outils Faisabilité / Débit et Parcloses après le retraitement du wiki

30/09/2026. Analyse faite en lecture seule sur l'état non commité de `feature/wiki` après le
retraitement des 28-29/09 (PERFORM76 / système 76, système 70 2023 + e.VOLUTION 2008, ROTO).
Les 43 tests des trois outils passent ; les défauts ci-dessous ne cassent aucun test, ils
donnent des réponses fausses ou incomplètes.

Principe inchangé : **le wiki est la source, l'outil ne fait que lire ses tableaux.** On corrige
d'abord le wiki quand sa forme empêche une lecture fiable, ensuite le code. Aucune valeur n'est
codée en dur ; une contradiction entre sources s'affiche avec son identifiant.

---

## Lot 0 — décisions à prendre avant de coder

| # | Question | Proposition |
| --- | --- | --- |
| D1 | Quand le manuel Roto KSR et le catalogue CTL_105 se contredisent sur une borne (CTR-108, CTR-110, CTR-111, CTR-112, CTR-113, CTR-114), quelle borne l'outil applique-t-il ? | La plus restrictive des deux, comme CTR-18 aujourd'hui, avec la pastille de l'anomalie et les deux valeurs dans le détail. C'est une hypothèse de l'outil, affichée comme telle. |
| D2 | Le registre CTR a une colonne « Valeur à retenir en attendant ». Pour CTR-108 elle dit « 400 mm, la plus restrictive » ; pour CTR-110 à 114 elle dit « Aucune ». Le 29/09 j'ai retiré de la page champs d'application la phrase « la borne retenue est 400 mm » en pensant qu'elle tranchait. | Si D1 est validée : remplir la colonne de la même façon pour CTR-110 à 114 (« la plus restrictive, en attendant l'arbitrage ») et remettre sur la page la phrase alignée sur le registre. Sinon : passer CTR-108 à « Aucune ». |
| D3 | Parclose système 70 : la réponse vient-elle du tableau de vitrage 2023 (manuel), du poster, ou des deux ? | Tableau 2023 en premier (plages écrites, feuillure 70 mm) ; le poster en complément pour les seules références qu'il est seul à donner (6146, 76531 à 76534). |
| D4 | 6146 et 76531 à 76534 figurent au poster et pas au tableau 2023, sans anomalie qui l'explique. | Ouvrir une VER- (« présence au tableau 2023 non vérifiée »). |
| D5 | Les tableaux de vitrage e.VOLUTION 2008 : les afficher ? | Oui, dans un contexte séparé « classeur e.VOLUTION 2008 (SAV, menuiseries anciennes) », jamais mélangés au 2023. |
| D6 | VER-113, VER-114 : le catalogue confirme le manuel, l'ancienne attribution venait d'une page du wiki depuis corrigée. | Clore les deux (clôture validée par toi). |

---

## Lot 1 — modifications du wiki (forme seulement, aucune valeur changée)

| # | Page | Modification | Pourquoi |
| --- | --- | --- | --- |
| W1 | `profiles/systeme-76-abaques-dimensionnels.md`, titres l. 416 et 440 | « Ouvrant 76272, 76279 avec renfort … » → « Ouvrant 76272 / 76279 avec renfort … » | Toutes les autres sections écrivent « / » ; le parseur découpe sur « / » et ne voit jamais ces deux abaques (le code sera aussi rendu tolérant, point F1). |
| W2 | `profiles/systeme-70-tableau-de-vitrage.md`, § *Tableaux de vitrage de 2008* (l. 647-887) | Transformer les phrases d'amorce « Feuillure 54 mm, sans élargisseur [2 p. 249] : » en sous-titres `#### Feuillure 54 mm` / `#### Feuillure 70 mm avec élargisseur 728`, un par tableau | Deux tableaux par section `###` sans titre propre : la feuillure ne se reconnaît qu'à l'ordre. Nécessaire pour P4. |
| W3 | même page, colonne « Plage admise » des tableaux 2008 | Garder le texte, mais ajouter deux colonnes « Épaisseur mini admise (mm) » / « Épaisseur maxi admise (mm) » comme le tableau 2023 | Même forme que 2023, un seul lecteur ; plus de « 37,5 – 39 » à décoder. |
| W4 | `wiki_llm/CLAUDE.md`, § *Pages read by the application* | Ajouter `systeme-70-tableau-de-vitrage.md` (et, si le lot 3 est fait, `roto-nx-apercu-ferrures-cote-p.md`, `roto-nx-apercu-ferrures-designo.md`) | La page devient lue par l'outil. |
| W5 | `anomalies/contradictions-entre-sources.md`, CTR-108 et CTR-110 à 114 | Selon D2 | Cohérence page / registre / outil. |
| W6 | `anomalies/contradictions-entre-sources.md`, CTR-40 | Le texte dit « onze » références et en liste douze | Coquille de notre registre (le code en porte 12, conforme à la liste). |
| W7 | `anomalies/informations-a-verifier.md` | Selon D4 (nouvelle VER-) et D6 (clôture VER-113, VER-114) | — |
| W8 | CTR-108 (colonne Impact) | « lu par le vérificateur de faisabilité » → à mettre à jour une fois F2 fait | Le texte décrira le comportement réel. |

---

## Lot 2 — P1 : réponses fausses aujourd'hui (chacun effort S, avec test)

| # | Outil | Code | Correction | Test à écrire |
| --- | --- | --- | --- | --- |
| F1 | Faisabilité | `faisabilite.py:268-289` | Lire les sections « Ouvrant 76272 / 76279 … » (séparateur « / » ou « , ») même sans tableau de courbes de verre : les limites de couleur et les zones suffisent ; supprimer le message faux « ne le donne qu'en deux vantaux » (`:781-786`, `faisabilite.html:375`) et rétablir le contrôle de vent deux vantaux (`:779-780`) | 76272 avec V326.Z en 1 vantail : verdict calculé, plus « sur étude » |
| F2 | Faisabilité | `faisabilite.py:398-407`, `:630-637` | Lire aussi la table « Configuration Designo (BA 13), catalogue 2023 » ; appliquer D1 ; pastilles CTR-108, CTR-114, INC-285 | Designo 100 kg, HFF 350 : refus avec CTR-108 |
| F3 | Faisabilité | `faisabilite.py:640-645`, `faisabilite.html:373` | Appliquer les bornes paumelles P hors oscillo-battant (la page écrit que les tableaux 130 / 150 kg valent pour OF et deux vantaux) | OF paumelles P hors bornes : refus, plus « non contrôlée » |
| F4 | Débit | `debit_atelier.py:80-91`, `:195-196`, `:288` | Reconnaître la première colonne « Meneau » : renfort V318.Z / V319 (76372), V323.Z / V322 (76373) et accessoires du meneau | Débit avec meneau 76372 : renfort lu |
| F5 | Parcloses | `parcloses.py:177-188`, `test_76_dormant_sans_compensateur` | Lire le dormant 76 sans compensateur dans `systeme-76-tableau-de-vitrage.md` § *Parcloses de dormant et de meneau sans compensateur* (A et B, +1 / −0,5, support M138, joints « dormant et meneau » et « capot alu et joint EPDM ») ; déplacer INC-52 sur les contextes qu'elle vise (p. 94 et 97) | 32 mm : 2638 « correspond » ; 38 mm : 2647 présente ; famille B présente |
| F6 | Parcloses | `parcloses.py:10-13`, `211-233`, `parcloses.html:166-168` | Retirer les notes périmées du système 70 (« ni ouvrant ou dormant, ni support de cale ») ; texte d'aide mis à jour | Snapshot du texte d'aide |

---

## Lot 3 — P2 : données du wiki ignorées qui changent un verdict

| # | Outil | Correction | Effort |
| --- | --- | --- | --- |
| P1 | Parcloses | Système 70 lu sur `systeme-70-tableau-de-vitrage.md` (feuillures 54 et 70 mm, plages écrites, support 9326, joints 9C32 / G342 / 9045.1), poster en complément (D3) ; à 44 mm : 76503, 0135, 1512, 6148 au lieu de rien | M |
| P2 | Parcloses | Tableaux 2008 dans un contexte séparé (D5), après W2 / W3 ; dédoubler « 2431 / 0132 » | M |
| P3 | Faisabilité | Classe de sécurité côté Designo (CDR 1 N, CDR 2 / 2 N) : bornes par classe, choix ouvert dans l'interface | M |
| P4 | Faisabilité | Diagrammes Roto en kg/m² (HFF maxi par poids de verre tous les 100 mm, droite basse) : contrôle réel au lieu du rectangle ; pastille CTR-106 ; zone « 2ᵉ compas » | M |
| P5 | Faisabilité | Poids admissible selon ferrure et acier dans le dormant (ift) : nouvelle saisie ; VER-25 | S-M |
| P6 | Faisabilité | Section « … battement 76473 sans renfort » (colonnes V326Z / V314Z) reconnue pour les 76272 / 76279 | S |

---

## Lot 4 — pastilles d'anomalies (effort S, un test par famille)

- **Faisabilité (Roto)** : VER-34 sur chaque verdict Roto ; CTR-110, INC-280, INC-281 côté P ; CTR-108, CTR-114, INC-285 côté Designo ; note « > 130 kg : compas réglé sur 80 mm » avec le report de charge.
- **Parcloses** : CTR-92 (2454, 0136, 6147 à 24 mm), INC-242 (2454 feuillure 62, 2008), CTR-91 (76509, 76515 et contexte 2008), VER-86 (pas de solution 70 à 5-7, 14, 21-23 mm), VER-87 (support 9326), CTR-19 étendue aux lignes 76 Advanced 76508 / 2454 / 2433 et à la 2638 dormant, CTR-17 (76509, 76515 du 76) ; VER-28 affichée une fois par groupe au lieu de chaque ligne.

---

## Lot 5 — P3 : nouveau périmètre (à décider plus tard)

- Débit : coupes des capots AluClip (§ *Cotes de débit des capots aluminium AluClip*) — M.
- Débit : nomenclature Roto (palier d'angle P 3/130, P 6/130, P 6/150 selon le poids ; kit report de charge 567972 / 565254 ; 2ᵉ compas ; crémone selon HFF avec CTR-67, CTR-115 à 117 affichées) — M-L.
- Faisabilité : position de poignée, paumelles Roto Solid B des 76272 / 76279 — S.
- Faisabilité / débit système 70 — L : abaques en grilles de repères, cotes de débit et inerties en HTML à deux lignes d'en-tête (réutiliser le lecteur HTML de SOLEAL FY de `parcloses.py`), pas de DTA ni de tapées 70, deux époques (2008 / 2023) ; ouvrirait un contrôle de meneau par inertie.

---

## Ordre proposé

1. Lot 0 (tes réponses D1-D6).
2. Lot 1 (wiki, forme) + tests.
3. Lot 2 (F1-F6) + Lot 4 (pastilles) — un commit par outil.
4. Lot 3, point par point.
5. Mettre à jour le root `CLAUDE.md` (description des trois outils) et `wiki_llm/CLAUDE.md` (pages lues).

Rien de ce plan n'est codé. Le test déjà en échec (`test_wiki_index.py::test_recherche_par_reference`,
76507) est indépendant et se traite à part.
