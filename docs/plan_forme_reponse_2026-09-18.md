# Plan — La forme de la réponse : répondre, pas rédiger un rapport

Date : 2026-09-18 · Branche : `feature/wiki` · Porte sur `app/prompts/wiki_consignes.md` (le prompt système) et rien d'autre.

> **Verdict en trois lignes.** Les douze consignes actuelles sont toutes des consignes de **vérité** (signale l'anomalie, donne les deux familles jumelles, donne l'exception, préfère un tableau) et aucune n'est une consigne de **forme**. La seule qui y ressemble, « Reste concis », arrive en douzième position et ne dit pas quoi couper. Résultat : sur une question fermée le modèle répond juste et court, mais dès que la question est ouverte il produit un rapport avec titres, tableaux, « Points critiques », « Document source » et « À retenir ». On ajoute un bloc **Comment répondre** en tête des consignes, et on désarme les deux règles qui poussent à la mise en forme.

---

## 1. Ce qui est mesuré

Sur les 62 réponses du golden du 18/09 (questions factuelles, fermées) :

| Mesure | Valeur |
| --- | --- |
| Longueur médiane de réponse | 196 caractères |
| Moyenne | 289 caractères |
| p90 | 632 caractères |
| Maximum | 1 286 caractères |
| Réponses de moins de 250 caractères | 36 sur 62 |
| Réponses contenant un tableau | 16 sur 62 |
| Réponses contenant un titre markdown | 2 sur 62 |

**Le golden ne voit pas le problème** : ses questions sont fermées (« épaisseur de vitrage de la parclose 2452 ? »), donc la réponse est courte par construction. Les deux cas rapportés par Elie sont des questions **ouvertes** — « existe-t-il acajou 2 faces en PERFORM ? », « comment transformer un OF en OB ? » — et c'est là que la réponse enfle. La mesure manquante est donc un jeu de questions ouvertes (§ 5).

Deux défauts distincts dans les exemples rapportés :

1. **La redite.** « Oui, l'acajou existe en plaxé 1 face » suivi de « Pour résumer : Acajou 1 face : oui / Acajou 2 faces : non ». Un résumé d'une réponse de deux lignes.
2. **Le rapport.** La transformation OF→OB rend un document structuré : un titre, un tableau des 5 étapes, une section « Points critiques » en trois sous-parties numérotées, un second tableau des éléments manquants, « Référence réglementaire », « Document source », « À retenir ». Le contenu est juste ; c'est la forme qui est fausse. Une réponse utile tient en cinq étapes numérotées, deux lignes de vigilance et une ligne sur VER-31.

## 2. La cause, consigne par consigne

| Consigne actuelle | Ce qu'elle produit |
| --- | --- |
| **2** — signale l'anomalie même si la question n'en parle pas | Une section « Points critiques » ou « À vérifier », là où une ligne suffit |
| **8** — familles jumelles : « donne **les deux** » | Un développement sur la famille non demandée |
| **10** — « donne aussi la nuance qui change la décision » | Un paragraphe d'exceptions à chaque réponse |
| **11** — « préfère un tableau à une énumération » | Un tableau pour deux valeurs (16 réponses sur 62) |
| **12** — « reste concis » | Rien : elle est douzième, et les onze précédentes demandent d'ajouter |

Aucune consigne ne dit **où mettre la réponse** (en premier), ni **ce qu'il est interdit d'écrire** (un récapitulatif, une conclusion, une bibliographie).

## 3. La cible : trois gabarits

**Question fermée** (oui/non, une valeur, une référence) → **1 à 3 lignes**. La première phrase est la réponse nue.

```
Non — l'acajou n'existe pas en 2 faces sur PERFORM.
Il est disponible en plaxé 1 face extérieure ; en 2 faces, il faut passer en HYBRIDE
ou TEXTURAL (/gammes/hybride.md, /gammes/textural.md).
```

**Question de choix ou de comparaison** → la réponse, puis **au plus 5 lignes** : ce qui distingue les options, l'exception qui change la décision, l'anomalie s'il y en a une. Tableau seulement à partir de trois références comparées sur au moins deux colonnes.

**Procédure** → la liste numérotée des opérations, puis **au plus 3 lignes** de vigilance, puis la ligne d'anomalie s'il y en a une. Pas de titre, pas de tableau des étapes, pas de section « document source » : le renvoi à la page du wiki suffit.

```
1. Démonter l'équerre de compas et le compas OF d'origine.
2. Visser la têtière fournie à dimension.
3. Installer le compas OB sur la têtière.
4. Retirer l'obturateur de manœuvre — sans ça la crémone n'atteint pas la position soufflet.
5. Visser la gâche OB (droite ou gauche selon le sens d'ouverture) en traverse basse du dormant.

⚠️ VER-31 : la notice ne donne ni la référence de la têtière, ni celle du compas OB, ni les
cotes de perçage de la gâche, ni les limites dimensionnelles après transformation — à demander
au service technique avec la hauteur d'ouvrant. (/procedures/transformation-of-en-ob-roto-nx.md)
```

## 4. Les changements dans `app/prompts/wiki_consignes.md`

**A. Un bloc « Comment répondre » placé AVANT les règles de vérité** (c'est l'ordre qui compte : la première chose lue est la forme attendue) :

1. **La première phrase est la réponse.** Une question fermée se répond par « Oui », « Non » ou la valeur seule, puis le complément. Jamais de reformulation de la question, jamais de préambule (« Voici la procédure… », « D'après le wiki… »).
2. **Trois à cinq lignes suffisent.** N'ajoute une phrase que si elle change ce que le lecteur va faire : une exception, une famille jumelle qu'il risque de confondre, une anomalie. Dans le doute, ne l'ajoute pas.
3. **Jamais de récapitulatif.** Interdits : « Pour résumer », « À retenir », « En conclusion », « Points critiques », « Référence réglementaire », « Document source », et toute phrase qui redit ce que la réponse vient de dire. Une réponse courte n'a pas besoin d'être résumée.
4. **Pas de titre `#`, sauf procédure de plus de cinq étapes.** Pas de tableau pour moins de trois références. Une valeur se donne dans une phrase, pas dans un tableau d'une ligne.
5. **Une anomalie tient en une ligne**, à la fin, avec son identifiant : `⚠️ CTR-17 : le manuel profine annonce 48 mm, le DTA 50 mm — retenir 50 mm.` Jamais une section.
6. **Cite la page une fois**, entre parenthèses, à la fin de la phrase qu'elle appuie. Pas de liste de sources en fin de réponse : l'interface les affiche déjà.
7. **C'est une conversation.** L'utilisateur enchaîne les questions. Ne réexplique pas ce que tu viens de dire au tour précédent ; réponds à ce qui est demandé maintenant.

**B. Deux règles existantes réécrites**, pour qu'elles cessent de contredire le bloc A :

- **11** devient : « Pour une cote, donne l'unité. Un tableau seulement à partir de **trois** références comparées ; en dessous, une phrase. »
- **12** disparaît, absorbée par le bloc A.

**C. Trois règles conservées mot pour mot** parce qu'elles portent la valeur de la base et ne coûtent qu'une ligne quand elles s'appliquent : la vérification silencieuse des registres (2), l'interdiction d'inventer un « pourquoi » (3), l'interdiction d'interpoler dans un tableau (4).

Effet de bord attendu et voulu : modifier les consignes change le hash du prompt, donc **la clé de cache** — le premier appel qui suit paie le plein tarif, les suivants retombent à 97 %.

## 5. Mesurer

**Un second jeu, conversationnel** : `tests/fixtures/golden/wiki_conversation.json`, une douzaine de questions ouvertes du même type que celles d'Elie (existence d'un coloris, choix entre deux gammes, procédure de pose, question de suivi qui dépend du tour précédent). Chacune porte, au lieu d'une valeur attendue, trois critères vérifiables par le code :

- `premiere_ligne_repond` — la réponse tient dans la première phrase (regex sur `^(Oui|Non|<valeur>)`),
- `longueur_max` — nombre de caractères plafond, par gabarit (250 / 600 / 900),
- `interdits_de_forme` — aucune des expressions bannies (« Pour résumer », « À retenir », « Document source », un titre markdown sur une réponse de moins de 600 caractères).

**Le golden des 62** reste le garde-fou de justesse : la cible est de ne pas descendre sous les **83,9 %** actuels. Une consigne de forme qui ferait perdre en justesse serait à revoir, pas à garder.

## 6. À faire en même temps (petit défaut d'interface)

Dans `app/templates/chat.html`, une page du wiki citée dans le texte devient un lien `href="#"` piloté par JavaScript. À la copie du texte, le lien se résout en `http://localhost:8001/?conversation=817#`, ce qu'on voit dans les exemples rapportés. Le correctif : `href="/wiki#/dossier/page.md"`, le JavaScript continuant d'intercepter le clic pour ouvrir le lecteur. La copie donne alors une vraie adresse, et le clic du milieu ouvre la page dans un onglet.
