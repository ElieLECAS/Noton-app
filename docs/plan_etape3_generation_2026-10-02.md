# Étape 3 — la génération : comprendre ce qu'on lit, répondre à la question, sans hors-sujet (analyse et plan, 02/10/2026)

La récupération est validée (étape 2). Reste ce que GLM fait de ce qu'il a lu. Cette analyse repose sur des
réponses réelles et sur des références indépendantes, pas sur une impression. Le code touché pendant l'analyse
est une réparation de ma propre régression (§ 3, D1). Le reste est une proposition.

## 1. Ce qui a été audité

- **15 réponses réelles de GLM 5.3** (conversations 1546 à 1554 du 02/10), dont trois questions posées deux ou
  trois fois (crémone 4 points, parclose 2636, coupe).
- **La faisabilité PERFORM76 (Q26 du golden)**, jouée quatre fois, comparée à deux références : le golden
  (`docs/golden_lia_20_questions_pvc_roto_2026-10-01.md` : « **Non en l'état** », avec la liste des fautes graves) et
  le vérificateur déterministe de l'application (`/faisabilite` : verdict « **hors** », vantail 122,4 × 137,4 cm).
- Pour chaque réponse : longueur, pages citées, anomalies citées, et lecture du raisonnement du dernier appel.

## 2. Ce qui est déjà bon, à ne pas casser

- **Il ne triche pas.** Sur la clé pompier (TGY3731), il refuse d'inventer une référence. Sur les élargisseurs
  de 200 mm en alu, il répond « non » et dit ce que le wiki porte à la place.
- **Il sépare les familles jumelles** (parclose 2636 : dormant, pas ouvrant), donne les deux valeurs d'une
  contradiction avec leur document (CTR-19, CTR-80, INC-129…), cite des chemins qui existent, accompagne
  une coupe d'une phrase et de ses cotes.
- **Il est stable quand la réponse est un tableau** : la crémone 4 points donne trois fois le même contenu.

## 3. Les défauts observés

| # | Symptôme | Exemple réel | Cause | État |
|---|---|---|---|---|
| **D1** | **Dimension déduite fausse, donc verdict faux** | Q26 : ouvrant estimé 115 × 130 cm (baie − 2 × 74 mm), vrai 122,4 × 137,4 cm (DEO = baie − 2 × 38 mm) ; réponse « Oui, réalisable » (deux fois), golden « Non » | GLM a lu les **tableaux** de la page des cotes de débit (§3, §5) sans leur **légende** (§1 « Ce que donnent ces tableaux », §2 « L'exemple du manuel »). Ma lecture par section ne rendait plus le sommaire : **régression de mon lot 2** | **Corrigé** (sommaire sur toute lecture partielle + règle « un tableau se lit avec sa légende »). Un rejeu : « Non », ouvrant 1 224 × 1 374 mm, courbe 12 mm à 132 / 112 cm |
| **D2** | **Erreur d'arithmétique** | « le triple 4/12/4/12/4 fait **32 mm** » (4+12+4+12+4 = 36) → mauvaise parclose (2454 au lieu de 76503) et un conflit de sources (CTR-19) sans objet | Calcul mental sans garde, 1 exécution sur 3 | À traiter (L4) |
| **D3** | **Verdict plus affirmatif que les contrôles** | « Oui… elle passe les trois contrôles » alors que le poids du vantail n'était pas vérifié (« devrait rester sous 60 kg ») | Aucune règle de calibrage du verdict | À traiter (L3) |
| **D4** | **Hors-sujet** | 6 réponses sur 15, et le rejeu de Q26 : arrêté du 25 juin 1980 sur le passage libre (clé pompier), coquille INC-03 d'une planche de l'autre document (coupe), effort d'amorçage > 100 N (faisabilité), habillage A502 et CTR-41 (élargisseurs), INC-128 et VER-96 (battement 6162), poids TBDK (rejeu) | Les règles 2 et 10 invitent à *ajouter* (anomalies, nuances), aucune règle ne dit *quand s'arrêter* | À traiter (L2) |
| **D5** | **Remède écrit dans le wiki, non donné** | Q26 : l'équerre J079 aux quatre coins décale la limitation de verre de deux courbes (abaque §4) ; absente du rejeu | Section non lue : § 4 ne figurait pas dans ses lectures | À traiter (L2, L1) |
| **D6** | **Absence affirmée sans regarder la couverture** | Clé pompier : « aucune source du wiki n'en contient une », alors que le catalogue (p. 86-103) est marqué « à faire » : il n'est pas encore repris | La fiche montrait la ligne « à faire », sans règle pour la lire | À traiter (L5) |
| **D7** | **Longueur imprévisible** | Une valeur ou une référence : 450 à 1 100 caractères. Faisabilité, élargisseurs, battement : 2 000 à 3 200 | Pas de cible de longueur | À traiter (L2) |
| **D8** | **Variabilité d'une exécution à l'autre** | Q26 : « Oui », « Oui », puis « Non » | D1 et D2 | À mesurer après correction (§ 5) |

## 4. La réponse que j'écrirais à Q26 (la cible)

> **Non en l'état.** Le DTA passe (baie 1,45 × 1,30 m, limite 1,50 × 1,40 m) mais la courbe d'épaisseur de verre de
> l'abaque d'ouvrant le refuse.
> - Ouvrant : dormant 76171, cote à déduire a = 38 mm → 1 300 − 2 × 38 = 1 224 mm de large, 1 450 − 2 × 38 = 1 374 mm de
>   haut, soit 122,4 × 137,4 cm.
> - Verre : 4 + 4 + 4 = **12 mm** (les abaques additionnent les verres, sans les lames). Courbe 12 mm du 76281 + V266.Z :
>   132 cm de haut à 120 cm de large, 112 cm à 130 cm. Votre ouvrant fait 137,4 cm : au-dessus aux deux graduations.
> - Le reste passe : limite blanc, règle des 25 %, ferrure Roto NX 130 kg, parclose 76503 pour 36 mm.
>
> **Pour le faire :** équerres de feuillure J079 aux quatre coins (la limitation de verre se décale de deux courbes),
> ou un double vitrage (4/16/4 : 8 mm de verre, aucune restriction).
> **À vérifier :** le poids du vantail (DTA : justification expérimentale au-delà de 60 kg).
> (chemins des pages lues)

Environ 150 mots. Le rejeu de GLM après correction fait ~450 mots : verdict, calcul, courbes et parclose justes, mais il
omet J079, propose trois pistes dont deux spéculatives, et termine par un paragraphe de vigilances (poids TBDK,
fabrication certifiée) qui ne change pas la décision. **Le fond est au niveau ; l'écart restant est un écart de
jugement (quoi garder) et de calibrage.** Cette réponse est écrite d'après les sections du wiki et le golden : elle
est à valider par vous avant d'en faire une référence.

## 5. Les leviers, dans l'ordre de ce que montrent les preuves

| # | Levier | Défaut visé | Preuve | Risque |
|---|---|---|---|---|
| **L1** | Sommaire sur toute lecture partielle, et règle « un tableau se lit avec sa légende ; une dimension déduite se calcule avec la formule de la page, et le calcul figure dans la réponse » | D1, D5 | **Fait.** Q26 : « Oui », « Oui » → « Non » avec le calcul écrit  ; 47 tests des fichiers touchés | +1 lecture possible. Un seul rejeu : à confirmer sur les autres questions à légende |
| **L2** | **Forme de la réponse « décision d'abord »** : (1) la réponse en première phrase ; (2) le chemin qui y mène, chaque valeur avec son unité et sa page, chaque calcul écrit ; (3) ce qui change la décision (exception, seuil, famille jumelle, contradiction **qui porte sur cette valeur**, avec les deux valeurs) ; (4) si c'est non ou incertain, la voie écrite dans le wiki (remède, gamme voisine) ; (5) une ligne sur ce qui n'a pas pu être vérifié. **Test de chaque phrase** : elle répond, ou elle justifie, ou elle change la décision, sinon elle disparaît. Cibles : 3 à 6 lignes pour une valeur ou une référence, 250 mots au plus pour une faisabilité | D4, D5, D7 | Le modèle sait déjà citer et nuancer ; il lui manque le critère d'arrêt | Une règle trop sèche supprime du contexte utile : le « contexte » autorisé est ce qui évite une erreur de commande ou de fabrication (ex. « la 2636 ne se monte que sur dormant ») |
| **L3** | **Verdict calibré** (faisabilité) : « oui » seulement si chaque limite de la chaîne a été lue et passe ; une vérification manquante qui peut inverser le verdict le rend « oui sous réserve de… » ; un contrôle hors limite le rend « non » (ou « sur étude » à moins de la précision de lecture). Le golden accepte ce repli | D3 | Q26 : « Oui » avec poids non vérifié | Peut rendre GLM trop prudent : à mesurer sur les faisabilités « oui » du golden |
| **L4** | **Composition de vitrage calculée par le serveur** et donnée dans la fiche : « 4/12/4/12/4 : épaisseur totale **36 mm**, verre cumulé **12 mm** (les abaques lisent le verre cumulé, la parclose l'épaisseur totale) ». Le code existe déjà (`faisabilite.lire_vitrage`) | D2 | 1 exécution sur 3 se trompe de 4 mm | Un mécanisme de plus : à n'adopter que si D2 se reproduit après L1-L3 sur plusieurs passes |
| **L5** | **Absence et couverture** : « un registre de couverture marqué *à faire* signifie que cette partie du document source n'est pas encore reprise : dis que le wiki ne la couvre pas encore, pas qu'elle ne contient rien » | D6 | Clé pompier | Faible |
| **L6** | **Anomalies : seulement celles qui touchent la valeur donnée.** Règle de réponse, et côté serveur resserrer `match_anomalies`, qui retient toute entrée liée à une page lue (+10), donc beaucoup sur une grosse page | D4 | Battement 6162 : cinq identifiants, dont deux hors sujet | Un mécanisme de plus côté serveur : mesurer d'abord combien d'entrées injectées sont citées à raison |
| **L7** | **Paramètres** : température 0,2 contre 1,0 (valeur recommandée par Z.ai), effort `high` contre `max` | D8 | Aucune | Coût et temps (`max`). À tester seulement si D8 reste après L1-L3 |

Hors périmètre, **décision pour vous** : faire entrer le calcul de faisabilité dans le tour (le vérificateur de l'application
donne le bon verdict, déterministe) ou l'écrire dans le wiki (une page « contrôles d'une faisabilité PERFORM76 »,
chaque pas renvoyant à sa source). La règle de la maison (trois outils, « une réponse fausse se corrige d'abord dans
le wiki ») conduit à la seconde, après la mesure de L1-L3.

## 6. Comment mesurer « le niveau de Claude » sans le deviner

1. **Un jeu noté, avec une grille fixe.** Les questions 21 à 40 du golden portent chacune un type, un « Attendu », une
   « Faute grave » et la preuve. Je les note, hors de l'application, sur sept critères :

   | Critère | Forme |
   |---|---|
   | Verdict ou valeur exacte | oui / non ; **faute grave** = le contraire ou une valeur d'une autre famille |
   | Chaîne de lecture complète | toutes les sections nécessaires lues |
   | Nuance ou remède qui change la décision | présent / absent |
   | Sources | chemins exacts, de pages lues |
   | Pertinence | nombre de phrases hors sujet |
   | Calibrage | aucune affirmation non lue ; l'incertitude est dite |
   | Concision | mots, contre la cible du type |

2. **Première passe : 12 questions, une conversation chacune**, choisies pour leur type : Q21, Q26, Q27, Q31
   (faisabilité) ; Q22, Q33 (contradiction) ; Q23, Q29, Q37 (absence) ; Q24 (fausse prémisse) ; Q28 (lecture de
   tableau) ; Q32 (cross-sujet). Environ **0,15 $ la question, 2 $ la passe**.
3. **Un levier à la fois**, sur les mêmes questions, **deux passes** pour voir la variance. L1 est déjà en place : la
   première passe est la base de comparaison. Ensuite L2 + L3 ensemble (c'est une seule réécriture de la forme),
   puis L5, puis L4 et L6 seulement si la mesure les demande.
4. **Critère de réussite proposé** : zéro faute grave sur les 12 ; au moins 10 sur 12 avec la nuance ou le remède
   attendus ; zéro phrase hors sujet sur 9 réponses sur 12 ; longueur dans la cible pour 10 sur 12.

## 7. Décisions attendues

1. **Feu vert pour la première passe** (12 questions, ~2 $, une conversation par question).
2. **La cible de forme** (§ 5, L2) vous convient-elle ? Notamment : 250 mots au plus pour une faisabilité, 3 à 6 lignes
   pour une valeur, et le « test de chaque phrase ».
3. **La réponse cible de Q26** (§ 4) : à valider avant de s'en servir comme référence de niveau.
4. **Faisabilité** : on attend la mesure de L1-L3, puis on tranche entre la page du wiki et le vérificateur dans le tour.
