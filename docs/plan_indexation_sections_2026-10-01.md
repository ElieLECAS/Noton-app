# Plan — indexation par sections, livraison page ou section (01/10/2026)

**PROPOSITION, à valider lot par lot.** Aucune page du wiki n'est modifiée par ce plan : le découpage se
fait à l'indexation, en mémoire, avec l'instantané. Les problèmes de génération (calculs, comparaisons,
lecture de « donne accès à ») sont **hors périmètre** : ils seront traités ensuite, avec ou sans
changement de modèle.

## 1. Objectif

Que LIA reçoive **dès la première recherche** l'endroit précis du wiki qui répond, sans noyer le modèle
sous des pages entières de 100 000 caractères, et sans lui cacher le reste de la page quand il en a besoin.

Trois capacités :

1. **Chercher par section** : le classement porte sur des sections, la livraison regroupe par page.
2. **Livrer la page entière quand elle est petite**, et pour une grosse page : sa fiche, son sommaire et
   les sections trouvées.
3. **Lire à la demande** une section précise du sommaire, ou la page complète.

## 2. Ce que les mesures ont établi (01/10, hors ligne puis Mistral Small)

| Constat | Mesure |
| --- | --- |
| Le réglage « 3 pages entières » date d'un wiki de 193 pages et ~480 k tokens, sans page > 50 k car. | Aujourd'hui : 318 pages, ~1 950 k tokens, 23 pages > 50 k car. |
| Livrer des sections fait mieux, avec moins de volume | Preuve livrée : 3 pages entières 91 % (69 k car.) ; sections 60 k car. 98 % (banc, 53 q.) ; golden série 2 : 82 % → 94 % |
| Les paramètres fins comptent peu | Découpe 2 500 / 5 000 / 8 000 car., poids des titres, poids de la page : ±3 points |
| La remise des références dans la requête écarte la bonne page | Section « dimensions maximales » du DTA : rang 2 sans, rang 29 avec « 76171 76281 » |
| Un routeur à règles n'améliore pas le classement lexical à volume égal | Banc : routeur 46 % complet, BM25 6 pages 57 % |
| Test réel, 39 questions, Mistral Small, une passe | Sans sections 40 / 78, avec sections 50 / 78 ; médiane 52 k → 37 k tokens ; boucles 2 → 0 |
| Le défaut des sections seules : l'éparpillement | Q18 : sections de la bonne page livrées parmi 24 pages, le modèle conclut « le wiki ne donne pas » |

Le prototype est dans le répertoire temporaire de la session : `idxsec.py`, `variante_sections.py`,
`balayage.py`, `essai_sections.py`. Il sera repris dans `app/`, pas copié tel quel.

## 3. Architecture cible

### 3.1 Les sections (index)

Construites par `WikiIndex` avec l'instantané (`wiki_service.load_snapshot`), jetées avec lui.

- **Découpe** aux titres `#` à `####`. Une section de plus de 5 000 caractères est recoupée à une fin de
  paragraphe, à une ligne de tableau markdown ou à un `</tr>` HTML.
- **Un tableau coupé garde son en-tête** dans chaque morceau : la ligne d'en-tête et le séparateur en
  markdown ; toutes les lignes `<th>` avant la première `<td>` en HTML (y compris les en-têtes à trois
  niveaux, comme la matrice des parcloses SOLEAL FY).
- **Chaque section porte** : la page, son numéro (`§n`), son chemin de titres, ses lignes de début et de
  fin, sa taille, son ancre.
- La frontmatter n'est pas une section. Elle nourrit la **fiche** de la page (titre, description, phrase
  d'ouverture).

### 3.2 Le classement

- **BM25 par section** : texte, chemin de titres pondéré ×4, titre de la page ×2. Même normalisation que
  l'index actuel (accents, ligatures, nombres groupés, pluriels).
- **Score de la page tiré par ses sections** : meilleure section + 0,5 × deuxième, plus 0,3 × score BM25
  de la page entière. Les coefficients sont à confirmer au lot 2.
- **Facettes** (`gamme`, `systeme`, `tags`) : ×2 par facette correspondante, **sans exclusion**. Le
  `type` ne classe pas (inchangé).
- Les registres d'anomalies restent hors des résultats (injectés par le serveur, inchangé).
- Les pages `sources/` restent derrière les pages concept.

### 3.3 Les références de la question : cherchées à part

La remise des références dans la requête du modèle est **supprimée**. Si le modèle a omis une
référence de la question, le serveur lance une recherche sur la référence seule et ajoute ses deux
meilleures sections **après** les résultats principaux. On garde le correctif TGY3702 sans écraser la
requête. Le message « Absent de tout le wiki » est inchangé.

### 3.4 La livraison (`chercher`)

- **4 à 6 pages par recherche** (à fixer au lot 2), dans l'ordre du score de page.
- **Page petite** (moins de 20 000 caractères, seuil à fixer au lot 2 parmi 15, 20 et 30 k) : **page
  entière**. C'est la majorité des pages (médiane ~13 600 car.).
- **Page plus grosse** : la fiche, le **sommaire**, les sections trouvées (au plus 3 par page), et la
  section voisine quand la section trouvée coupe un tableau.
- **Budget** : environ 60 000 caractères par recherche. Au-delà, les pages suivantes passent en
  métadonnées.
- **AUTRES RÉSULTATS** : chemin, titre et section la plus proche (au lieu de l'extrait actuel).
- **Jamais deux fois la même chose** : une section ou une page déjà livrée dans le tour n'est pas
  renvoyée. Une recherche qui n'apporte rien de nouveau le dit explicitement, pour éviter la boucle de
  Q17 dans le prototype.

Forme proposée :

```
===== PAGE 1 : /certifications/dta-6-16-2334.md — page entière =====
<contenu>

===== PAGE 2 : /profiles/systeme-76-abaques-dimensionnels.md — 2 sections sur 41 (page de 96 000 car.) =====
Fiche : <titre> — <description> — <phrase d'ouverture>
Sommaire (lire_page(chemin, section) pour lire une autre section) :
  §3  Lecture des abaques (l. 95-135)
  §12 Ouvrant 76281 > Limites de couleur (l. 262-280) ← livrée
  §13 Ouvrant 76281 > Courbes d'épaisseur de verre (l. 281-300) ← livrée
  ...
§12 [Ouvrant 76281 > Limites de couleur] (l. 262-280)
| en-tête du tableau |
...

===== AUTRES RÉSULTATS =====
| Chemin | Titre | Section la plus proche |
```

Pour les très grandes pages (plus de 100 sections), le sommaire se limite aux titres de niveau 1 à 3,
40 lignes au plus, avec le nombre de sous-sections.

### 3.5 La lecture (`lire_page`)

On ajoute un paramètre **optionnel** à l'outil existant : on reste à **trois outils**.

- `lire_page(chemin, section)` : `section` accepte `§13`, `13`, un morceau de titre (« courbes ») ou une
  ancre. La réponse rend la ou les sections, avec la fiche et l'en-tête de tableau. Si rien ne
  correspond, elle rend le sommaire.
- `lire_page(chemin)` : la **page complète**, comme aujourd'hui. Les sections déjà livrées dans le tour
  sont remplacées par `[§13 déjà fourni plus haut]`, ce qui évite de payer deux fois.
- **Plafond par tour** (environ 200 000 caractères). Une page complète qui le dépasserait renvoie le
  sommaire et demande une section. Cela ne peut arriver que sur les pages géantes.

### 3.6 Ce qui ne change pas

- L'injection des anomalies, rapprochées des pages qui ont été livrées en tout ou partie.
- La vérification des coupes sur disque et du couple référence/image (elle lit le corps de la page).
- La vérification des citations (`/dossier/page.md` existe), la relance quand aucune page n'a été
  chargée, le flux SSE.
- Le tour vocal : il réutilise `WikiAnswer`, il hérite donc de tout. Moins de tokens veut aussi dire un
  premier mot plus rapide.

## 4. Consignes (`wiki_consignes.md`, nouvelle clé de cache)

- **Règle 0, réécrite** : chercher livre des pages entières (petites) ou des **sections avec le sommaire
  de leur page** (grosses). Une section est lue et peut être citée. Si la réponse dépend d'une autre
  partie de la page, lire la section au sommaire ; si toute la page est nécessaire, `lire_page(chemin)`.
- **Nouvelle phrase, contre le cas Q18** : « Avant d'écrire qu'une page ne donne pas une information,
  regarde son sommaire. Si une section pourrait la contenir, lis-la. Une page lue en partie ne prouve
  pas une absence. »
- Les descriptions des outils `chercher` et `lire_page` sont mises à jour en conséquence.

## 5. Fichiers touchés

| Fichier | Changement |
| --- | --- |
| `app/services/wiki_index.py` | Découpe, index et classement des sections, mise en forme hybride. `trier_resultats` et `formate_resultats` remplacés (code supprimé, pas désactivé). |
| `app/services/wiki_chat_service.py` | `chercher` hybride, recherche des références à part, dédoublonnage, budgets, `lire_page(section)`. `PAGES_COMPLETES`, `CHAT_PAGES_COMPLETES` et la remise des références supprimés. `MAX_TOOL_ROUNDS` : 6 → 8 à mesurer. |
| `app/prompts/wiki_consignes.md` | Règle 0 et phrase « page lue en partie ». |
| `app/templates/chat.html` | Étapes : « page entière » ou « §13 Courbes d'épaisseur de verre ». |
| `app/templates/wiki.html` (facultatif) | Ouvrir une source à sa section. Les ancres actuelles (`h-0`, `h-1`…) ne correspondent pas aux ancres des liens du wiki ; c'est à aligner. |
| `app/scripts/mesurer_recuperation.py` (nouveau) | La mesure hors ligne, versionnée (reprise du prototype). |
| `tests/fixtures/golden/` | Golden 40 questions en JSON, preuves stockées par **extrait de texte** (pas par numéro de ligne, qui bouge) ; Q20 réécrite. |
| `tests/test_wiki_sections.py` (nouveau), `test_wiki_index.py`, `test_wiki_chat.py` | Voir § 7. Les tests qui supposent « 3 pages entières » sont réécrits. |
| `CLAUDE.md` | Section « Le tour de chat » : livraison et lecture. |

## 6. Lots

Chaque lot ne passe au suivant que si sa mesure est bonne.

| Lot | Contenu | Mesure de passage | Effort |
| --- | --- | --- | --- |
| **0** | Mesure versionnée : script, golden en JSON avec extraits de preuve, banc, Q20 réécrite | Le script reproduit les chiffres du prototype (±2 points) | 0,5 j |
| **1** | Découpe et index des sections dans `WikiIndex`, sans changer le tour | Tests de découpe verts ; index construit en moins de 5 s pour 321 pages | 1 j |
| **2** | Réglage hors ligne de la livraison hybride : seuil 15 / 20 / 30 k, 4 / 6 / 8 pages, coefficients | Preuve livrée ≥ sections seules (98 % banc, 94 % golden) ; toutes preuves ≥ sections seules + 10 pts ; aucune question du banc ne perd sa preuve par rapport à aujourd'hui ; médiane ≤ 60 k car. | 0,5 j |
| **3** | `chercher` hybride dans le tour, références à part, dédoublonnage, budgets | Tests ; rejeu à vide des requêtes réelles du golden | 1 j |
| **4** | `lire_page(chemin, section)`, page complète avec marques « déjà fourni » | Tests | 0,5 j |
| **5** | Consignes, descriptions d'outils, étapes affichées | Relecture | 0,5 j |
| **6** | Test réel : golden 39 questions ×3, Mistral Small, une conversation par question | Moyenne ≥ 50 / 78 (passe du 01/10) ; zéro faux « le wiki ne donne pas » sur une page lue en partie ; Q18, Q21, Q39 pas pires qu'avant ; médiane ≤ 40 k tokens | 0,5 j + tokens |
| **7** | `CLAUDE.md`, mémoire, ouverture d'une source à sa section dans `/wiki` (facultatif) | — | 0,5 j |

Environ cinq jours au total, dont la moitié en mesure et en tests.

## 7. Tests (`docker compose exec web pytest`)

- **Découpe** : chemin de titres, lignes de début et de fin exactes, frontmatter exclue, tableau
  markdown coupé avec son en-tête répété, tableau HTML coupé avec son `<thead>` (dont la matrice SOLEAL
  FY à trois niveaux), aucune section au-delà de la limite sauf ligne unique plus longue.
- **Classement** : « PERFORM76 oscillo-battant dimensions maximales » met la section du DTA dans les
  trois premières ; la référence cherchée à part ne fait pas reculer cette section.
- **Livraison** : page sous le seuil livrée entière ; grosse page en fiche, sommaire et sections ;
  budget respecté ; rien de renvoyé deux fois ; message explicite quand rien n'est nouveau.
- **Lecture** : `section` par numéro, par titre, par ancre ; section inconnue → sommaire ; page
  complète avec marques « déjà fourni » ; plafond du tour.
- **Non-régression** : coupes, citations, anomalies, relance sans page, tour vocal.

## 8. Risques

| Risque | Parade |
| --- | --- |
| Un extrait sorti de son tableau (règle 0) | En-tête répété, fiche, section voisine, sommaire ; page entière sous le seuil |
| Le modèle conclut à une absence sur une page lue en partie (Q18) | Sommaire visible, consigne dédiée, critère explicite au lot 6 |
| Plus d'anomalies hors sujet (rapprochées de pages lues en partie) | Inchangé au départ ; à mesurer. Si besoin, rapprochement sur le texte livré seulement. |
| Temps de construction de l'index (prototype : ~20 s en Python pur) | Index inversé et normalisation mise en cache ; objectif < 5 s, mesuré au lot 1 |
| Les numéros de ligne des preuves bougent quand une page change | Preuves du golden stockées par extrait de texte |
| Le golden a été réglé sur ces 40 questions | Paramètres fixés sur le banc, vérifiés sur le golden ; nouveau lot de questions jamais vues avant de conclure |

## 9. Décisions pour Elie

1. Supprimer la remise des références dans la requête et la remplacer par la recherche à part (§ 3.3).
2. Ajouter le paramètre `section` à `lire_page` plutôt qu'un quatrième outil (§ 3.5).
3. Les valeurs de départ : seuil 20 000 car., 6 pages, 60 000 car. par recherche, 200 000 par tour,
   8 allers-retours. Le lot 2 les confirme ou les corrige.
4. L'ordre : les lots 0 à 2 ne touchent pas au comportement de LIA et peuvent commencer tout de suite.
