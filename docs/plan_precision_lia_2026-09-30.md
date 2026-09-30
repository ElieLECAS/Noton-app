# Plan — précision de LIA : wiki, recherche, génération, modèle (30/09/2026)

Ce plan part de l'attribution des échecs du golden 20 questions
(`docs/rapport_golden_20q_2026-09-30.md`). Il remplace l'ordre des plans précédents
(`plan_wiki_trouvable`, `plan_l2_decoupage`, `plan_lia_recherche`) : il en garde les mesures et
les chantiers utiles, dans l'ordre que les échecs imposent.

## 1. La réponse courte

**Oui, il faut toucher aux outils, pas seulement au wiki. Et un modèle plus gros ne répare pas
la recherche.** C'est mesuré sur la faisabilité PERFORM76 (questions 16 et 17). J'ai repris les
sept requêtes que Mistral a réellement écrites et relevé le rang de la page du DTA dans les
résultats. Il faut qu'elle soit dans les trois premières pour être livrée entière.

| Ce qu'on change | Rang de la page du DTA (7 requêtes de Mistral) | Dans les 3 premières |
| --- | --- | --- |
| Rien | 28 à 71 | **0 / 7** |
| La recherche seule (filtres adoucis, cotes non prises pour des références) | 7 à 63 | **0 / 7** |
| Le wiki seul (les dimensions maximales dans une page à elles, titrée PERFORM76) | 2 à 25 | **1 / 7** |
| **Les deux** (page à part et filtre `type` qui ne pénalise plus) | 1 à 14 | **5 / 7** |

Même en retirant tous les filtres, la page actuelle reste entre le rang 6 et le rang 60, et
jamais dans les trois premières. Un modèle qui choisirait mieux ses filtres ne la verrait donc
pas davantage. Mistral Medium aurait le même
problème, parce qu'aucun modèle ne lit une page qu'on ne lui donne pas.

Un modèle plus gros peut en revanche aider sur trois points :

- **la régularité** : SoftOpen et garantie LUMINE 65, faux une fois sur quatre ;
- **la discipline de rédaction** : clé pompier, une invention deux fois sur quatre ;
- **la lecture des tableaux compliqués** : les cotes du 6106.

On le décidera à la fin, sur le même étalon (étape 6). Il y a aussi une question de coût : on
paie au token. Aujourd'hui, les 20 questions consomment 2,2 millions de tokens. **Trois
questions en font la moitié** (12, 18 et 19, entre 318 000 et 375 000 tokens chacune). Réduire
ce volume sert donc quel que soit le modèle, et c'est ce qui rendrait Medium abordable.

## 2. L'étalon : comment on sait qu'une étape a marché

Une étape n'est gardée que si l'étalon s'améliore sans rien casser ailleurs. On garde
toujours les cinq mêmes mesures :

1. **Trouvabilité mécanique**, sans Mistral, en quelques secondes. Pour chaque question du golden
   et du banc (77 questions), on rejoue les requêtes déjà écrites et on relève le rang de la page
   attendue. Pour le golden, ce sont celles de Mistral, que le runner enregistre désormais ;
   pour le banc, celles de Claude-LIA. Cette mesure juge le wiki et `chercher`, et elle tourne après
   chaque modification.
2. **Golden 20 questions ×3** avec Mistral Small (60 tours, concurrence 1). Trois passages,
   parce qu'un passage unique ne sépare pas la régularité du hasard. Il se relit à la main,
   comme le 30/09.
3. **Attribution** pour chaque question encore fausse : naturel ×3 contre bonne page imposée ×3.
   On sait ainsi si la cause est la recherche ou la rédaction.
4. **Dix questions de contrôle**, écrites avant les corrections et jamais regardées pendant. Ce
   sont les mêmes familles sur d'autres gammes : faisabilité 70, absence, contradiction, cote
   dans une coupe. Sans elles, on corrigerait le test au lieu de LIA.
5. **Aucune perte de qualité dans le wiki**, voir la section 4.

## 3. Les étapes, dans l'ordre

### Étape 1 — Wiki : les corrections ciblées (demi-journée)

Ce sont les pages qui font échouer des questions, et les défauts qui rendent une page
invisible.

| Correction | Pourquoi (mesuré) |
| --- | --- |
| **Les dimensions maximales du DTA dans une page à elles** : « PERFORM76 (76 Advanced) : dimensions maximales des fenêtres selon le DTA », avec `gamme: PERFORM76`, une description qui dit à quelles questions elle répond et des tags. Le DTA garde un lien vers elle, le tableau n'existe qu'une fois. | Le tableau (1 400 caractères) est noyé dans une page de 58 000 caractères titrée « système 76 Advanced ». Seule, cette correction fait passer de 0 / 7 à 1 / 7 ; avec l'étape 2, à 5 / 7. |
| **Co-modifier `faisabilite.py`**, qui lit cette section dans la page du DTA (`PAGE_DTA`, « 2.2.3.7 »). | C'est une page protégée : `tests/test_faisabilite.py` doit rester vert. |
| **Trois pages au frontmatter illisible** : `systeme-70-abaques-evo2008`, `choix-des-fenetres-exposition-au-vent-2008`, `sources/roto-nx-bras-report-de-charge`. | Leur YAML ne se lit pas : elles n'ont ni type, ni titre, ni tags, ni gamme pour l'index. |
| **Le nom PROFERM sur les pages fournisseur** : 10 pages du système 76 ne disent jamais « PERFORM76 », et le titre du DTA non plus. | Le menuisier dit PERFORM76, profine dit 76 Advanced. C'est la règle « noms » du protocole (L1). |
| **Les 30 références mal écrites** : 15 coupées par une espace, 15 abrégées avec « / ». | Une référence coupée ne se trouve pas par sa forme entière. |
| **Le tableau des cotes du 6106** : un seul tableau dont les colonnes disent ce qu'elles mesurent, et le total 95 sur sa ligne. | Page lue, 0 / 2 juste : trois tableaux aux colonnes contradictoires. |

### Étape 2 — L'outil `chercher` (demi-journée de code, mesurée à la trouvabilité)

| Changement | Pourquoi (mesuré) |
| --- | --- |
| **Le filtre `type` ne compte plus dans le score.** Il ne sert plus qu'à lister sans mot-clé. | Mistral met `type=Profilé` pour une question qui se trouve dans une Certification. Chaque filtre multiplie le score par 2, et deux filtres par 4 : dans les faits, le filtre exclut. C'est contraire à la règle de la maison « une facette remonte, elle n'exclut pas ». La consigne qui le déconseille est ignorée. Mesuré : 1 / 7 → 5 / 7. |
| **Une cote de la question n'est pas une référence.** Un nombre suivi de mm, m, kg ou × est une dimension, et il ne reçoit pas le poids ×3 des références. | « INNOSLIDE SoftOpen 1800 » fait tomber la page INNOSLIDE du rang 3 au rang 8 : « 1800 » attire les grands tableaux. |
| **Le serveur ajoute à la recherche les références de la question.** | TGY3702 : Mistral n'a gardé la référence dans sa requête que 3 fois sur 8. |
| **Mots normalisés** : œ, « 487 206 », « LUMINE 65 » / « LUMINE65 », pluriel en -s et -x. | Ce sont des défauts de l'index relevés au banc. Aujourd'hui, « manœuvre » et « manoeuvre » ne se rejoignent pas. |
| **« Absent de tout le wiki »** : quand une référence demandée n'est dans aucune page, le résultat le dit en une ligne. | Parclose 3702 : 369 000 tokens pour conclure qu'elle n'existe pas. Clé pompier : la limite de 6 allers-retours atteinte. |

### Étape 3 — Les consignes de rédaction (2 h, mesurées ×3)

Quatre règles, chacune liée à un échec observé :

1. **Faisabilité** : jamais de « oui » sans avoir lu les dimensions maximales et l'abaque. Sinon,
   dire ce qui n'a pas été vérifié. La plage d'une ferrure n'est pas une limite de fenêtre.
   Questions 16, 17 et 18 : trois « oui » faux.
2. **Contradiction** : donner les deux valeurs et leurs sources, pas seulement l'identifiant.
   Question 05 : « CTR-03 » cité, 15 ans contre 20 ans non dit.
3. **Absence** : ce que le wiki ne dit pas n'est ni supposé ni complété. Question 20 : « livrée
   avec la fermeture », inventé.
4. **Coupe servie** : dire ce qu'elle représente et citer sa page. Questions 10 et 11 : une image
   sans un mot. En complément côté code, le serveur, qui connaît la page de chaque coupe qu'il
   sert, l'ajoute aux sources : 6 réponses sur 20 n'en ont aucune.

Si Mistral Small ne tient pas ces règles au rejeu ×3, c'est une limite du modèle, et l'étape 6
le montrera.

### Étape 4 — Le volume (1 jour, décision à prendre)

On commence par identifier les pages derrière les trois tours à plus de 300 000 tokens, puis on
les découpe **une par une**, en appliquant les règles de la section 4. Ce n'est pas la découpe
massive de L2 (80 pages), qui reste en pause.

Si le volume reste trop haut ensuite, on passe à la **livraison ciblée** : la première page
entière, les sections pertinentes seulement pour les deux suivantes. Cela change la décision
« trois pages entières », donc **c'est à toi de valider** avant que je la code. Cible : médiane
divisée par deux (aujourd'hui 70 000 tokens par tour), et plus aucun tour au-dessus de 150 000.

### Étape 5 — Wiki de fond : gammes, tags, anomalies (2 à 3 jours, par famille)

On touche ici 200 pages et plus. C'est là que le risque de perte est le plus grand, d'où la
place après les corrections ciblées. On procède **famille par famille**, en commençant par
PERFORM76, puisque c'est là que sont les échecs.

- **`gamme` renseignée** là où la page est propre à une gamme : 191 pages sur 297 n'en ont
  aucune, dont 40 profilés sur 60. Dès que Mistral filtre par gamme, toutes ces pages passent
  derrière. Une page transversale, comme le glossaire ou un DTA commun, reste sans gamme.
- **Tags en liste fermée** (`wiki_llm/tags.md`, lot L4) : 736 tags distincts, dont 385 portés par
  une seule page. Comme filtre, un tag unique ne sert à rien. Comme synonyme, il aide la recherche
  (poids ×4). On ne le supprime donc jamais sans que son mot soit repassé dans la description :
  on le mesure à la trouvabilité avant et après.
- **Anomalies reliées à leurs pages** : 150 entrées sur 479 le sont. Le serveur injecte d'abord
  celles qui sont reliées aux pages lues, les autres dépendent d'un rapprochement de mots.

### Étape 6 — Le modèle (1 h de mesure)

Le même étalon est passé avec Mistral Small et avec Mistral Medium, sur le wiki et les outils
corrigés. On ne compare que ce qui reste faux, avec le coût en regard : tokens mesurés × prix.
Si Medium ne corrige que ce que les étapes 1 à 5 ont déjà corrigé, on garde Small.

## 4. Améliorer sans perdre en qualité

Chaque page modifiée ou découpée passe les mêmes contrôles avant d'être gardée :

- **Inventaire avant/après** : chaque nombre, référence, chemin d'image et lien de l'ancienne page
  doit se retrouver dans la nouvelle, ou dans la page qui en hérite. Une seule perte, et la
  modification est refaite. C'est la méthode du 19/09 : 79 pages réécrites, 0 perte.
- **Une donnée, une page** : un tableau déplacé n'est pas recopié, l'ancienne page pointe vers
  lui.
- **`pytest`** : les 8 pages lues par `faisabilite.py`, `parcloses.py` et `debit_atelier.py` ont
  leurs tests. Un en-tête changé les casse, c'est voulu.
- **Le lint du wiki** : aucune page orpheline, aucun lien fantôme, frontmatter lisible, `index.md`
  et pages de gamme à jour, entrée dans `log.md`.
- **Première page de chaque famille relue par toi** avant que je fasse les suivantes.

## 5. Ce qu'il faut de toi

| Décision | Quand |
| --- | --- |
| Go pour les étapes 1 à 3 : corrections ciblées, outil `chercher`, consignes | maintenant |
| Page à part pour les dimensions du DTA, avec `faisabilite.py` co-modifié | maintenant, fait partie de l'étape 1 |
| Livraison ciblée, qui remplace les trois pages entières | après la mesure de l'étape 4 |
| Budget pour le passage Medium (environ 6,5 millions de tokens pour le golden ×3, moins après l'étape 4) | étape 6 |
| Valider les 11 réponses attendues du banc (R03, R05, R09, R10, R18, R20, R22, R31, S01, S07, G10) | avant de s'appuyer sur les 57 questions du banc |

**Critère de réussite** : golden relu à au moins 18 / 20, sur les trois passages, et aucune
faisabilité déclarée « oui » à tort. Au plus une réponse sans source, pas de régression sur les
questions de contrôle, et le volume divisé par deux.
