# Plan — cartographie du wiki et recherche « du général au précis » (11/10/2026)

Statut : **PROPOSITION**. Rien n'est codé, rien n'est réécrit. Chaque phase s'ouvre sur une mesure
et se ferme sur un go / no-go d'Elie.

## 1. Constat mesuré (wiki au 11/10, non commité)

### Ce qui est sain

| Mesure | Valeur |
|---|---|
| Pages | 942 (16,3 M car., ~4,6 M tokens bruts, ~5,5 M au tokenizer Mistral) |
| Liens cassés / pages orphelines / pages absentes de `index.md` | 0 / 0 / 0 |
| Pages de contenu sans facette `gamme`/`systeme`/`fournisseur` | 9 |
| Liens sortants vers du contenu | médiane 7, p10 4 |
| Sections (titres `##`/`###`) | 4 154 |

L'hygiène OKF est bonne. Le problème n'est pas la qualité des pages, c'est **la carte**.

### Ce qui gêne la recherche

1. **`index.md` est plat et rangé par type, pas par produit.** 357 k car. (~122 k tokens), 940
   entrées ; « Profilés » mélange 220 pages TECHNAL, ASKEY, KÖMMERLING. Pour aller du général au
   précis, le modèle doit tout lire puis filtrer de tête. Il est entier dans le prompt permanent
   (131 k tokens mesurés, seuil 60 k).
2. **Aucun niveau intermédiaire.** Pas de page `systemes/` ; les tableaux de bord par système sont
   calculés pour l'humain (`/wiki`) mais pas donnés au modèle. `fournisseurs/technal.md` a 22 liens
   pour 461 pages TECHNAL.
3. **Pages trop grosses pour une lecture précise.** 28 pages ≥ 50 k car., 96 entre 20 et 50 k
   (Roto NX 125 k, Système 70 85 k). Le protocole dit ≤ ~20 k : ce sont les anciennes pages.
4. **Registres d'anomalies géants.** 1,64 M car. dans 3 pages (10 % du wiki), un seul fichier par
   nature, toutes gammes confondues. Ils ne sont pas découpés à l'indexation.
5. **Vocabulaire trop large pour orienter.** 726 tags ; pas de synonymes (« vantaux » ≠ « vantail »,
   « OC » ≠ « ouvrant caché ») ; facette fournisseur incohérente (« profine » ×1 contre
   KÖMMERLING ×145).
6. **Descriptions écrites pour décrire, pas pour aiguiller.** Moyenne 243 car. ; elles disent ce que
   la page contient, rarement quelle question elle tranche (lot W du 02/10, jamais fait).
7. **`verified` : 0 / 942 ; `draft` : 691.**

### Ce qui gêne la mesure

- `golden_40` : 20 questions seulement avec preuves ; `wiki_20` et `controle_10` : 18/20 sur une
  page ; aucune échelle de difficulté ; les absences ne sont pas mesurées hors ligne.
- Le banc du 30/09 a des extraits périmés (pages réécrites) et son `depouiller.py` est cassé.
- 4 tests de recherche en échec ; `c10_08` du golden non recalée.
- `CLAUDE.md` décrit un état dépassé (Mistral Small, `lire_page`, 200 k) : le code tourne sur
  **GLM 5.3 (1 M tokens)**, outil `lire`, 6 appels, 150 k car.

## 2. Ce que dit l'état de l'art (recherche web, sources dans le compte rendu de session)

- **Karpathy (gist « LLM Wiki », avril 2026)** : raw / wiki / schema ; ingest, query, lint ;
  `index.md` lu en premier — « fonctionne étonnamment bien » jusqu'à ~100 sources et quelques
  centaines de pages ; au-delà, un outil de recherche (qmd). **Nous sommes au-delà.** Aucune
  implémentation publiée n'est mesurée au-delà de quelques dizaines de pages.
- **Index hiérarchique** (index maître → index par dossier), **PageIndex** (table des matières +
  recherche guidée par le LLM), **Agent Skills** (divulgation progressive : métadonnées → fiche →
  ressource) : même idée, la carte se lit par niveaux.
- **Contextual Retrieval** (Anthropic) : préfixer chaque morceau de son contexte divise par 2 les
  échecs de récupération ; applicable en BM25 seul.
- **Context rot** (Chroma) et **Lost in the middle** : la fiabilité baisse avec la longueur, et
  surtout avec des **distracteurs proches** — exactement notre corpus (gammes jumelles, lignes
  voisines). Un contexte de 1 M sert à **porter la carte**, pas à y verser le wiki (5,5 M tokens).
- **Orchestrateur + sous-agents** (Anthropic) : meilleur sur la recherche large, ~15× les tokens.
- **Évaluation** : séparer récupération et génération (BrowseComp-Plus, FRAMES : l'écart « preuve
  fournie d'office » / « preuve récupérée » dit où investir) ; questions **null** (MultiHop-RAG) ;
  une hallucination coûte plus qu'un « je ne sais pas » (CRAG).

## 3. Cible

Deux rôles, une carte à trois niveaux.

```
Question ──► AIGUILLEUR (GLM 5.3, carte complète en cache)
               │  interprète : produit(s), nature de la valeur, niveau de généralité
               │  si la gamme est ambiguë et change la réponse → UNE question de clarification
               │  sinon : plan de lecture = pages + sections, avec la raison de chacune
               │  outils : chercher (BM25, références, longue traîne), sommaire(page)
               ▼
           LECTURE par le serveur (sans modèle) : sections demandées, anomalies rapprochées
               ▼
         RÉDACTEUR (même modèle, contexte court : question + preuves + anomalies, SANS la carte)
               │  peut demander UNE lecture complémentaire (`lire`)
               ▼
           Réponse citée (citations vérifiées par le code, inchangé)
```

- **Niveau 0 — carte racine** (~3-5 k tokens) : fournisseurs → systèmes/gammes → familles, avec
  le nombre de pages et une ligne de périmètre.
- **Niveau 1 — sommaire par système** (un par système/gamme, ~5-30 k tokens) : pages groupées par
  famille (dormants, ouvrants, quincaillerie, cotes de débit, pose, certification…), chaque entrée
  avec sa description « questions tranchées ».
- **Niveau 2 — page** : fiche + sommaire des sections (existe déjà dans `lire`).

Les niveaux 0 et 1 sont **générés par script depuis la frontmatter** (comme les tableaux de bord),
jamais écrits à la main : pas de page hub qui dérive. L'aiguilleur les porte **tous** en cache
(~130 k tokens, à 0,14 $/M en cache ≈ 0,02 $/question) : c'est « l'agent qui connaît l'index par
cœur ». Le rédacteur ne les voit pas : son contexte reste court, donc moins sujet au context rot.

**Écart avec les règles de la maison, à trancher par Elie** : « rien de plus que trois outils, pas
de routeur ». Le routeur à règles du 01/10 avait perdu contre BM25 (46 % contre 57 %) ; un
aiguilleur LLM qui lit une carte structurée n'a jamais été mesuré. La phase 3 le compare au tour
actuel à volume égal ; s'il ne gagne pas, il n'est pas gardé.

## 4. Phases

### Phase 0 — Le banc d'abord (aucun changement du wiki ni du code de recherche)

1. **Banc stratifié de ~140 questions**, 7 niveaux × ~20, construit en lisant le wiki (et le PDF
   pour 1 question sur 5) :

   | Niv. | Nature | Exemple de forme |
   |---|---|---|
   | N1 | Référence exacte, une page | « À quoi sert TGY3731 ? » |
   | N2 | Cellule d'un grand tableau | « Parclose SOLEAL FY pour un vitrage de 28 mm, joint intérieur 4 ? » |
   | N3 | Générale / orientation | « Quels coulissants alu propose-t-on ? », « Que couvre le DTA de la GY ? » |
   | N4 | Plusieurs pages d'un même système | profilé → quincaillerie → cote de débit |
   | N5 | Plusieurs documents ou gammes | « Différence de drainage entre FY et ASKEY OC ? » |
   | N6 | Contradiction (anomalie à citer) | une valeur portée par une CTR ou une INC |
   | N7 | Absence, fausse prémisse, faisabilité | référence inexistante, « oui » interdit sans limite |

   Pour chaque question : pages et sections de preuve, valeur attendue, valeurs interdites,
   gamme(s) concernée(s), « ambiguë oui/non ». Un tiers est mis à part comme **contrôle**, jamais
   regardé pendant les réglages.
2. **Deux mesures séparées** : (a) hors ligne, sans modèle : la preuve arrive-t-elle (rappel page,
   rappel section, volume, N7 : la fiche dit-elle « absent ») ; (b) avec modèle : exactitude,
   citations justes, anomalie signalée, abstention juste, coût, nombre d'appels. Et le **témoin
   oracle** : le rédacteur reçoit d'office les sections de preuve → plafond atteignable.
3. Recaler le banc du 30/09 et `c10_08`, réparer `depouiller.py`, remettre `CLAUDE.md` à l'état du
   code.

Livrable : la photo du tour actuel par niveau. **Go / no-go** : sans cette photo, rien ne bouge.
Coût modèle : une passe sur ~140 questions, à faire valider avant (règle « jamais un lot sans
accord »).

### Phase 1 — La carte (wiki seulement, protocole `wiki_llm/CLAUDE.md` amendé par Elie)

1. **Générateur des niveaux 0 et 1** depuis la frontmatter (`fournisseur`, `systeme`, `gamme`,
   `famille`, `usage`) ; `index.md` devient le niveau 0 + les liens vers les niveaux 1. Une
   frontmatter incomplète apparaît dans le lint au lieu d'être réparée par la carte.
2. **Descriptions « questions tranchées »** (lot W) : réécrire les 942 descriptions pour dire ce que
   la page décide (« quelle parclose pour quel vitrage », « jusqu'à quelle hauteur ») et ce qu'elle
   ne couvre pas. Par système, en tranches, comme le retraitement.
3. **Champ `famille` partout** (liste fermée par système) et facette fournisseur normalisée.
4. **Découpe des 28 pages ≥ 50 k** selon « une donnée = une page » (Roto NX, Système 70 d'abord).
5. **Registres d'anomalies par système** (`anomalies/<systeme>-…`), identifiants inchangés.
6. **Table de synonymes** (`reference/synonymes.md`, liste fermée) lue par l'indexeur : sigles,
   pluriels irréguliers, noms commerciaux.
7. **Lint étendu** : taille, facettes, `famille`, description sans verbe de décision.

Mesure : la partie (a) du banc, tour actuel inchangé, avant / après. Le BM25 seul doit déjà gagner
en N3-N5 grâce aux descriptions et aux synonymes.

### Phase 2 — Préfixe de contexte des sections (index, sans modèle)

Indexer chaque section avec « système · famille · titre de page · chemin de titres · en-tête de
tableau » (Contextual Retrieval, version lexicale, aucun vecteur). Mesure (a) ; garder seulement si
gain en N2/N4 sans perte en N1.

### Phase 3 — Aiguilleur et rédacteur (code, après accord)

1. Prototype hors application : l'aiguilleur reçoit la carte complète + la question et rend un plan
   de lecture JSON. Mesure de **rappel du plan** sur le banc, comparée à la carte BM25 actuelle à
   volume égal. Seule décision à prendre ensuite : le garder ou non.
2. Si gagnant : le tour en deux rôles (schéma § 3), clarification d'une question quand la gamme
   est ambiguë (R10, R17, S07 du 30/09 sont de ce type), rédacteur sans la carte.
3. Mesure (b) complète sur le banc, contrôle compris ; comparaison au témoin oracle.

### Phase 4 — Boucle qualité (Karpathy : lint + réversion au wiki)

Chaque échec du banc → correction **dans le wiki** (pas dans les consignes), puis re-mesure. Les
questions réelles des utilisateurs alimentent le banc (traces déjà enregistrées).

## 5. Ce qui n'est pas proposé

- Verser le wiki dans le contexte de 1 M : 5,5 M tokens ne tiennent pas, et le context rot frappe
  précisément les distracteurs proches.
- Index vectoriel, reranker, juge : la recherche reste lexicale (deux pages qui se contredisent
  doivent remonter toutes les deux).
- Sous-agents parallèles : ~15× les tokens pour un gain attendu sur la recherche large, que la
  carte à trois niveaux doit déjà couvrir. À rouvrir si N5 reste en échec avec la preuve livrée.

## 6. Décisions demandées à Elie

1. Lancer la phase 0 (banc stratifié, puis une passe modèle chiffrée avant lancement).
2. Accepter le principe d'une carte générée à trois niveaux et l'amendement du protocole
   (`famille` obligatoire, registres par système, descriptions « questions tranchées »).
3. Ouvrir ou non la question de l'aiguilleur LLM, qui déroge à « trois outils, pas de routeur ».
4. Commit du retraitement avant d'attaquer (≈ 800 fichiers en attente).
