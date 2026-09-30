# Rapport — golden de 20 questions, LIA réelle, 30/09/2026

Golden : `tests/fixtures/golden/wiki_20_questions.json` (réponses attendues vérifiées ligne à
ligne dans le wiki ; faisabilités calculées par `app/services/faisabilite.py`). Passage unique,
une question à la fois, sur l'endpoint réel du chat : `mistral-small-latest`, raisonnement
`high`, température 0,2, prompt permanent de 9 265 caractères (sans les anomalies), wiki de 297
pages. Rapport brut : `logs/eval/generation_wiki_wiki-20q_20260930-100107.json`.

## Résultat

| | Note automatique (runner) | Ma relecture |
| --- | --- | --- |
| Juste | 16 / 20 (80 %) | **14 / 20 (70 %)**, dont 4 avec un défaut de forme |
| Partiel | — | 1 |
| Faux | 4 | **5** |

Le runner compte les valeurs présentes ; il ne voit ni une invention dans une abstention (20), ni
des cotes mal étiquetées (12).

## Question par question

| # | Sujet | Runner | Relecture | Constat |
| --- | --- | --- | --- | --- |
| 01 | TGY3702 → 4 points (Technal) | juste | **juste** | TGY3704, raccordée à TGY3702 ou TGY3703. Manquent la fixation (2 vis TGY3723) et « pour très grandes hauteurs ». Une recherche, la bonne page. |
| 02 | LUMINE55, vitrage 24 mm | juste | **juste** | T591005 ou TFY2412, joint TAS0017 : bonne gamme (SOLEAL FY), aucune parclose PVC. |
| 03 | NT1947 sur 76180 | juste | juste, défaut | 140 mm, exact. Signale INC-06, qui ne porte pas sur la valeur (bruit). |
| 04 | Parclose 2452, système 70 | juste | juste, défaut | 8 mm (7,5 à 9), image juste. **Aucun chemin de page cité** ; VER-86 signalée hors sujet. |
| 05 | Garantie structure LUMINE 65 | faux | **faux** | « 20 ans », CTR-03 cité mais **la contradiction n'est pas dite** (15 ans au catalogue 2026). C'est la règle 2 des consignes, non tenue. |
| 06 | SOLÉAL55, chariots doubles | juste | **juste** | 200 kg, T401012 et T441004, versions inox et leur incompatibilité avec le rail alu, règles de calage exactes. Réponse la plus complète du lot. |
| 07 | Carré W4070495 | juste | **juste** | 56 mm, ASKEY Coulissant 65 NV, vantail W1041261. |
| 08 | Volet LA 37, V*4 | juste | **juste** | 1 800 mm pour H ≤ 2,25 m. |
| 09 | Vitrage acoustique | juste | **juste** | 44.6/14/10, 40 dB, avec l'image. |
| 10 | Coupe parclose 76576 | juste | juste, défaut | Bonne image, **mais rien d'autre** : pas un mot, pas de source. |
| 11 | Dessin dormant 76177 | juste | juste, défaut | Idem : image seule. |
| 12 | Coupe et cotes du 6106 (système 70) | juste | **partiel** | Bonne image, valeurs présentes (70, 95, 40, 55, 75, 20), mais **mal étiquetées** (« largeur hors tout » pour une hauteur, une « part de la parclose » inventée), page citée qui n'est pas celle des cotes. 375 000 tokens, 59 s. |
| 13 | Notice OF → OB Roto NX | juste | **juste** | Les 5 étapes de la procédure PRO-PVC-OFOB-01, avec leurs schémas, soufflet de 140 mm, VER-31 pertinente. Excellente. |
| 14 | Report de charge Designo II | juste | juste, défaut | Ordre exact, « retirer la vis haute » mis en avant, réglage du ressort. **Aucune source citée.** |
| 15 | Réglage paumelle SOLEAL PY | juste | **juste** | TWZ0002, les 5 gestes, couple 10 à 15 N.m, source citée. |
| 16 | PERFORM76 OF 1 300 × 2 400 | faux | **faux, grave** | « **Oui** » et « aucune limite de hauteur » : faux. Le DTA limite le 1 vantail OF à 2,15 × 1,00 m, et l'abaque d'ouvrant bloque aussi. La page DTA n'a jamais été chargée. |
| 17 | PERFORM76 OB 1 200 × 1 600 | faux | **faux, grave** | « **Oui** », sur les plages de la ferrure Roto NX (290 – 1 600 mm) prises pour des limites de fenêtre. Le DTA limite le 1 vantail OB à 1,50 × 1,40 m ou 2,15 × 1,00 m. |
| 18 | SoftOpen INNOSLIDE 1 800 mm | faux | **faux, grave** | « **Oui**, la quincaillerie Roto Patio Inowa intègre bien le SoftOpen » : faux, le SoftOpen n'existe qu'à partir de 1 970 mm. Aucune source, 318 000 tokens. |
| 19 | Parclose 3702 PERFORM | abstention juste | **juste** | Dit qu'elle n'existe pas, sans rien inventer. Mais 369 000 tokens pour une absence. |
| 20 | Clé pompier TGY3731 | abstention juste | **faux** | Dit qu'il n'y a pas de référence, puis **invente** : « la clé spéciale pompier est livrée avec la fermeture elle-même ». Le wiki ne le dit nulle part (la fermeture est livrée avec ses vis T770059). Premier essai arrêté après 6 allers-retours. |

## Par capacité

| Capacité | Résultat | Lecture |
| --- | --- | --- |
| Références, pièges de fournisseur et de gamme (01, 02, 03, 04, 06, 07, 08, 09) | **8 / 8** | Quand la question porte une référence ou un nom de produit, LIA trouve la bonne gamme et le bon fournisseur : LUMINE55 → SOLEAL FY, SOLÉAL55 → SOLEAL GY, W4070495 → ASKEY, TGY3702 → Technal. |
| Images (10, 11, 12) | **3 / 3** bonnes images | Garanties par le contrôle serveur des coupes. Deux réponses réduites à l'image, une aux cotes mal lues. |
| Notices (13, 14, 15) | **3 / 3** | Étapes dans l'ordre, schémas, avertissements. Le point fort de LIA. |
| Faisabilité (16, 17, 18) | **0 / 3**, trois « oui » faux | Le point le plus grave : une faisabilité fausse, c'est une commande impossible. |
| Contradiction à signaler (05) | 0 / 1 | L'identifiant est cité, les deux valeurs non. |
| Absence (19, 20) | 1 / 2 | Une abstention propre, une abstention complétée par une invention. |
| Sources citées | 14 / 20 | 6 réponses sans aucun chemin de page (04, 10, 11, 14, 18, 19). |

**Coût.** Médiane 69 600 tokens d'entrée par question, trois tours au-dessus de 300 000 (12,
18, 19). Réponse en 8 s en médiane, 21 s au p90, 59 s au plus lent. Une question a dépassé les 6
allers-retours au premier essai (20) ; une réponse 429 de Mistral, reprise par le serveur.

## Les causes, et ce qui les corrige

| # | Cause | Questions | Correction proposée | Où |
| --- | --- | --- | --- | --- |
| 1 | **La page des limites de fabrication ne remonte pas.** Question naturelle : DTA au rang 6 ou 7, ou absent. Elle ne sort qu'avec le vocabulaire du DTA (« dimensions maximales baie 1 vantail OF »). Son titre, « DTA n° 6/16-2334_V5, système 76 Advanced », ne dit ni PERFORM76, ni dimensions maximales, ni types d'ouverture ; elle n'a pas de `gamme`. | 16, 17 | Titre, description et phrase d'ouverture selon les règles 2, 3 et 5 du protocole (« Dimensions maximales de baie de la PERFORM76 par type d'ouverture »), `gamme: PERFORM`. Page lue par `faisabilite.py` : titres de section et colonnes intacts. | wiki |
| 2 | **« Pas de limite trouvée » devient « oui ».** LIA conclut à la faisabilité faute d'avoir lu une limite, et prend les plages d'une ferrure pour des limites de fenêtre. | 16, 17, 18 | Consigne : une faisabilité ne se déclare qu'après avoir lu les dimensions maximales et l'abaque ; sinon, dire ce qui n'a pas été vérifié. | consignes |
| 3 | **Un nombre de la question pollue la recherche.** « INNOSLIDE SoftOpen 1800 » fait tomber la page INNOSLIDE du rang 3 au rang 8 : « 1800 » attire les grands tableaux. | 18 | Étapes 4 et 5 du plan : la cote demandée ne se met pas dans les mots-clés ; livrer les sections plutôt que trois pages entières. | code + consignes |
| 4 | **La contradiction est réduite à son identifiant.** | 05 | Consigne : donner les deux valeurs et leur source, pas seulement « CTR-03 ». | consignes |
| 5 | **Une absence est complétée par une hypothèse.** | 20 | Consigne : ce que le wiki ne dit pas n'est ni supposé ni complété. | consignes |
| 6 | **Réponses sans source**, et images servies sans un mot. | 04, 10, 11, 14, 18, 19 | Le serveur connaît la page d'où vient chaque coupe servie : il peut l'ajouter aux sources, sans dépendre du modèle. Consigne : une coupe s'accompagne de ce qu'elle représente. | code + consignes |
| 7 | **Volume** : 300 000 à 375 000 tokens pour une absence ou une coupe. | 12, 18, 19 | Étape 4 du plan (livraison ciblée). | code |

**Une question pour toi.** L'application sait déjà calculer une faisabilité PERFORM76 sans se
tromper (`/faisabilite`, réservé à l'admin). Le chat ne s'en sert pas : il relit les tableaux, et
se trompe. Le brancher au chat serait un quatrième outil, contraire à la règle actuelle (« trois
outils, rien d'autre »). À décider par toi ; les corrections 1 et 2 sont à faire dans tous les cas.

## Limites de ce passage

- Un seul essai par question : un tour de Mistral varie (même question, deux conduites le matin
  même). Les faux graves (16, 17, 18) sont à rejouer avant et après correction, 3 fois chacun.
- Mon verdict n'était pas celui du runner sur 12 et 20 : leurs attendus sont durcis (20 : « livrée
  avec » interdit ; 12 : « hauteur » exigée à côté de 95, et l'image). Le runner compte désormais
  une abstention qui contient une valeur interdite comme « ambigu » (règle déjà appliquée aux
  questions à valeur ; elle vaut aussi pour les pièges de `space29_sans_reponse.json`). Ce passage,
  re-noté : **70 %**, comme la relecture.
- Le runner enregistre désormais les étapes de recherche et les pages lues de chaque tour ; ce
  passage-ci ne les avait pas, le diagnostic de trouvabilité a été refait hors ligne.

## Attribution des échecs (expérience du 30/09, vraie Mistral, 26 tours)

Même question, même prompt, même boucle ; on change une seule chose. NATUREL : LIA telle quelle
(3 essais). ORACLE : le premier appel de `chercher` est imposé (les mots d'un lecteur averti) ; la
lecture et la rédaction restent celles de Mistral. Données : `logs/eval/attribution_20260930.json`.

| Question | Naturel | Bonne page imposée | Où est la cause |
| --- | --- | --- | --- |
| 16 faisabilité OF | **0 / 3**, DTA jamais lu | **3 / 3** | **recherche + wiki** : le modèle sait lire, il ne trouve pas la page |
| 17 faisabilité OB | **0 / 3**, DTA jamais lu | **3 / 3** | **recherche + wiki**, idem |
| 18 SoftOpen | 3 / 3 (mais faux dans le golden : 1 échec sur 4) | 3 / 3 | **régularité** de la recherche |
| 05 garantie LUMINE 65 | 3 / 3 (mais faux dans le golden : 1 échec sur 4) | — | **régularité** de la rédaction |
| 20 clé pompier | 2 propres, 1 invention (+ 1 invention dans le golden) | — | **génération** : page toujours lue, 2 tours sur 4 complètent |
| 12 cotes du 6106 | — | **0 / 2** (page lue, 371 000 tokens) | **lecture** : trois tableaux dont les colonnes ne disent pas la même chose (55 + 40 « à gauche » ici, 40 « aile » là), total 95 absent de la ligne, contexte énorme |

Correction de mon hypothèse d'avant mesure : je situais les erreurs graves dans la génération. Les
trois fausses faisabilités viennent de la recherche : avec le DTA sous les yeux, Mistral répond
juste 6 fois sur 6.

Détails de la recherche sur 16 et 17 : les 6 premières requêtes de Mistral portaient des filtres
(`type=Profilé`, `gamme=PERFORM76`) alors que la description de l'outil les déconseille au premier
appel ; le DTA (type Certification, sans `gamme`) n'apparaît alors dans aucun des 15 premiers
résultats. Sans filtre, il est au rang 5 pour l'une des deux formulations, absent pour l'autre : ses
mots (« Dimensions maximales de baie ») ne sont pas ceux de la question (« dimension maximale
largeur hauteur »).

Attendu de la question 20 corrigé : « n'est pas indiquée » n'était pas reconnu, une réponse correcte
était notée « abstention_ko ».
