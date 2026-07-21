# Plan — parfaire le pipeline « question sur une référence » (cas 6111)

_2026-07-21 · branche `fix/retriever`. Basé sur le diagnostic complet du cas
« profil 6111 » : extraction clinique (chunk 47744, 411 p.15 : « 6111 :
longueur totale de 155 mm »), mais réponse fausse (111 mm) sur small ET medium,
reasoning high ET none._

---

## 1. Diagnostic final (chaque maillon vérifié)

| Maillon | État | Preuve |
|---|---|---|
| **Extraction vision (L1)** | ✅ clinique | chunk 47744 : faits par référence, propres, fidèles au dessin (155 sous le 6111) |
| **BM25** | ❌ structurellement inapte aux réfs | les 9 chunks contenant « 6111 » ont un ts_rank IDENTIQUE (0.0608) — indépartageables ; les mots de liaison (« parle-moi du profil ») classent d'autres pages devant |
| **pgvector** | ❌ | l'embedding du chunk 47744 est dominé par « 6108/dormants larges » → pour « 6111 », les amas p37/p41 passent devant |
| **KAG** | ❌ | 5 entités doublonnées « 6111 » ; liées aux pages 14/37 du doc 422, PAS au chunk propre 47744 |
| **RRF / rerank** | ❌ par héritage | aucun canal ne surface la p.15 → rien à fusionner |
| **CAG** | ❌ scelle l'échec | packe doc 411 en fenêtre 20-26 (autour du seul match p.23) → p.15 exclue ; les PNG suivent les pages packées → le dessin p.15 n'est jamais envoyé au vision |
| **Génération** | ❌ mésattribution | « 111 mm » EST dans le contexte (amas 422 p.37) mais pour d'autres coupes ; small/medium/reasoning l'ancrent sur 6111 (renforcé par la coïncidence 6111→111) |
| **Vérification (P2)** | ❌ aveugle à ce cas | test de présence littérale : « 111 mm » présent quelque part → validé à tort |

**Enseignement de fond** : une référence produit n'a pas de sémantique — elle a
une **identité littérale**. Tous nos canaux sont « flous » (embedding, ts_rank,
trigram) ; aucun ne garantit « donne-moi les pages où 6111 apparaît ». C'est le
chaînon manquant.

---

## 2. Le plan — 4 chantiers, du chirurgical au structurant

### P-A — Canal « référence exacte » (déterministe) + CAG/PNG alignés — LE CŒUR

Quand `detected_references` est non vide (fiche ET RAG normal) :

1. **Nouveau retriever exact** (zéro LLM, zéro embedding) :
   `SELECT … WHERE content ~ ('\m' || ref || '\M')` sur les chunks de l'espace
   → toutes les pages citant littéralement la référence. Sur le corpus actuel :
   9 chunks / ~7 pages pour 6111 — coût trivial, résultat EXHAUSTIF (recall
   100 % par construction). C'est le « Ctrl+F » assumé — pour une identité
   littérale, c'est l'outil correct.
2. **Injection dans la fusion** avec slots garantis (analogique aux
   `COLPALI_PROTECTED_SLOTS`) : les pages exact-match ne peuvent pas être
   évincées par les canaux flous.
3. **CAG orienté référence** : pour l'intent fiche/référence, les
   `matched_pages` du document = pages exact-match (p.12, 13, 15, 41, 45…) —
   la fenêtre de packing les inclut TOUTES (elles sont peu nombreuses par
   construction). Fini la fenêtre 20-26 qui exclut la p.15.
4. **PNG alignés** : les images envoyées au modèle vision = pages exact-match
   en priorité → le dessin de la p.15 (avec le « 155 » lisible) part au LLM
   vision, qui peut relire la cote directement sur le schéma.

### P-B — Vérification consciente de l'attribution (upgrade du P2)

Le test de présence ne suffit pas (le « 111 » traînait ailleurs). Pour les
questions à référence :
- extraire de la réponse les paires (référence, cote) ;
- valider une paire SEULEMENT s'il existe un chunk où la référence et la cote
  sont **co-localisées** dans un énoncé dont la référence est le sujet
  (heuristique : même puce/phrase, pattern « **REF** : … N mm », PAS un amas
  multi-réf « r1, r2, r3 : d1, d2, d3 ») ;
- sinon → `unsupported_claims` (aurait attrapé « 6111 = 111 mm » ET
  « 6101 = 70×58 » du capot).

### P-C — Référentiel produit canonique (le durable)

Passe OFFLINE sur tous les chunks L1 de l'espace :
- repérer les énoncés « référence-sujet » (le pattern des chunks fact-list :
  `**6111** : … 155 mm …`) — l'extraction vision les produit déjà proprement ;
- gate de fidélité : la cote doit être littérale dans le chunk (et recoupée
  avec la couche texte pymupdf de la page quand elle existe) ;
- stocker `référence → {type, matériau, dimensions, gamme, compatibilités,
  pages sources}` (table dédiée, rejouable sur les chunks existants sans
  re-ingestion) ;
- à la requête : bloc « FAITS VÉRIFIÉS (référence 6111) » injecté EN TÊTE du
  contexte de génération, au-dessus des documents. Le modèle n'a plus à
  retrouver la cote dans le jumble — elle lui est donnée, sourcée.
- Bonus : règle matériau/gamme (PVC ≠ gamme alu) → tue la contamination
  « Lumine » sur le 6101.

### P-D — Hygiène KAG références (support)

- Dédupliquer les 5 entités « 6111 » (Dormant 6111 / Profil 6111 / Référence
  6111 / Profilé 6111 / 6111) en une seule canonique ;
- lier l'entité à TOUS les chunks où la réf apparaît littéralement (simple SQL,
  même scan que P-A) — le canal KAG devient alors réellement utile pour les
  réfs au lieu de pointer sur 2 pages arbitraires.

---

## 3. Ordre d'exécution et mesure

1. **P-A** (retriever exact + CAG + PNG) — corrige le cas 6111 de bout en bout,
   mesurable immédiatement : la fiche doit sortir 155 mm.
2. **P-B** (vérification attribution) — filet pour ce que P-A ne couvre pas.
3. **P-C** (référentiel) — la robustesse durable, s'appuie sur le même scan
   exact que P-A.
4. **P-D** (KAG) — en parallèle de P-C (même infrastructure de scan).

Validation : golden espace 28 (les métriques CAG doc-recall/page-in-context
déjà branchées) + les cas nommés 6111/6101 (155 mm attendu, pas de Lumine sur
le 6101). Toujours retrieval-only d'abord, génération ensuite.

## 4. Ce qu'on ne fait PAS (écarté par les preuves)

- Changer de modèle de génération (medium échoue pareil) ;
- compter sur le reasoning pour compenser (high/none échouent pareil) ;
- re-chunker (l'extraction est clinique ; le chunk multi-réf est un BON
  format — c'est l'indexation par référence qui manque, pas le découpage) ;
- réparer ça par les seuils RRF/rerank (aucun réglage flou ne rend un canal
  flou exact).
