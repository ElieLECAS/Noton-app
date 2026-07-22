# Connaissance métier — Gammes PVC 70 & 76 (Proferm)

> **Statut : BROUILLON à valider** — généré le 2026-07-22 depuis les documents réellement en base
> (titres, contenus de chunks, attributions existantes). Les points marqués **⚠️ À VALIDER** ne sont
> écrits dans **aucun** document : ils relèvent du savoir interne Proferm et doivent être confirmés
> ou corrigés à la main. Ce fichier a deux usages :
> 1. **Injection prompt** (compréhension d'intention + génération) : le modèle connaît le paysage
>    documentaire AVANT de chercher.
> 2. **Référence de curation** : vocabulaire canonique pour l'attribution des documents
>    (`Document.proferm_gammes`, `Document.source`, `Document.materials`).
>
> Périmètre : **PVC gammes 70 et 76 uniquement**. L'ALU (Lumine / Technal) n'est mentionné que comme
> discriminant négatif (« ceci n'est PAS du 70/76 »).

---

## 1. Le fait fondateur : deux vocabulaires, aucun pont écrit

Les mots que les utilisateurs emploient et les mots que les documents contiennent **ne se recoupent
pas**. Vérifié en base : les expressions « Perform 70 », « Perform 76 », « gamme 70 », « gamme 76 »,
« Textural 70/76 », « Hybride 70/76 » n'apparaissent dans **aucun chunk** de la documentation
technique. Ce pont n'existe que dans les têtes — c'est précisément le rôle de cette fiche.

| Vocabulaire UTILISATEUR (commercial Proferm) | Vocabulaire DOCUMENTS (systèmes fournisseurs) |
|---|---|
| Perform, Hybride, Textural, Innoslide, « gamme 70 », « gamme 76 » | Kömmerling Gamme 70, Kömmerling e.VOLUTION, Trocal e.XCLUSIVE, TROCAL 76 ADVANCED / KBE 76 ADVANCED / KÖMMERLING 76 ADVANCED |

**Règle d'or pour l'assistant** : quand l'utilisateur nomme une gamme commerciale, il faut la
**traduire** en système fournisseur avant de chercher — chercher « Perform » littéralement dans les
DTD ne trouvera rien.

---

## 2. Le socle fournisseur

- **profine group** (fournisseur PVC unique du périmètre) regroupe **trois marques du même
  système** : **KÖMMERLING**, **TROCAL**, **KBE**. Le DTA du 76 les cite comme un seul procédé :
  « TROCAL 76 ADVANCED, KBE 76 ADVANCED, KÖMMERLING 76 ADVANCED » *(doc 412)*. Une question sur
  l'un vaut pour les autres.
- **PVC GREENLINE® de KÖMMERLING** = matériau de base des menuiseries **PERFORM** et de la partie
  PVC des **HYBRIDE** *(catalogue général 2024, doc 424)*.
- **ROTO** = ferrures/quincaillerie (Roto NX, Roto Safe E…). **Transverse** : une ref Roto n'indique
  pas une gamme, elle indique de la ferrure.
- **TECHNAL** = aluminium (SOLEAL FY/GY/PY, LUMEAL) → gamme commerciale **LUMINE**.
  **HORS PÉRIMÈTRE de cette fiche** — sert uniquement de discriminant négatif (§6).

---

## 3. Gamme 70 — « Kömmerling Gamme 70 »

**Source documentaire directe** : posters « Kömmerling Gamme 70 » *(docs 421 profilés principaux,
423 profilés complémentaires)* ; DTD-DBV-6-16-2335 *(doc 411)*.

- Usage : fenêtres, portes-fenêtres et coulissants PVC *(poster 421)* ; portes **Kömmerling
  e.VOLUTION** et coulissants/portes **Trocal e.XCLUSIVE** *(doc 411)*. Profondeur profils 70 mm.
- **Références de profilés principaux (poster 421)** : A107, A108, A109, A474, V000, V054, V059,
  V060, V061, V062, V063, V064, V065.
- **Références vues dans le DTD 411** : dormants 6108 / 6109 / 6110 / 6111 / 6158 (dormants larges,
  profondeur 70 mm) ; 6100, 6101, 6150, 6151 (ouvrants e.XCLUSIVE) ; seuils 9F67 / 9F68 / Z043 ;
  93051 ; rejets d'eau A465 / A466 ; parcloses A271 / A272 ; JA701-02.
- ⚠️ À VALIDER : le partage exact entre « Gamme 70 frappe » (posters) et « e.VOLUTION /
  e.XCLUSIVE » (portes/coulissants) — même système, ou sous-systèmes distincts de la même
  profondeur 70 ?

## 4. Gamme 76 — « 76 ADVANCED »

**Source documentaire directe** : DTA/DTD TROCAL 76 ADVANCED *(docs 412, 413)*.

- Système tri-marque : **TROCAL 76 ADVANCED = KBE 76 ADVANCED = KÖMMERLING 76 ADVANCED** *(cité
  tel quel dans le DTA)*.
- Usage documenté : **fenêtres à frappe PVC** — à la française, oscillo-battantes, soufflets
  *(doc 412)*. Profondeur profils 76 mm.
- ⚠️ À COMPLÉTER : inventaire des références de profilés 76 (non extrait des chunks à ce jour).
- ⚠️ À VALIDER : existe-t-il un coulissant en 76, ou le coulissant PVC est-il exclusivement porté
  par le 70 (e.XCLUSIVE) et l'INNOSLIDE ?

---

## 5. Matrice gammes commerciales ↔ systèmes fournisseurs

Slugs = ceux du système d'attribution existant (`Document.proferm_gammes`).

| Gamme commerciale (slug) | Matériau | Base fournisseur (documenté) | Système 70 | Système 76 |
|---|---|---|---|---|
| **PERFORM** (`perform`) | PVC | PVC GREENLINE® Kömmerling *(doc 424)* | Kömmerling Gamme 70 / e.VOLUTION / e.XCLUSIVE **⚠️ à valider** | 76 ADVANCED **⚠️ à valider** |
| **HYBRIDE** (`hybride`) | PVC intérieur + capot alu extérieur | Intérieur PVC GREENLINE® Kömmerling, développement Proferm *(doc 424)* | **⚠️ à valider** | **⚠️ à valider** |
| **TEXTURAL®** (`textural`) | **⚠️ à valider** (finitions décoratives : essences de bois / textures intérieur *(doc 424)*) | **⚠️ à valider** — probablement base PVC Kömmerling avec films/finitions, non écrit | **⚠️** | **⚠️** |
| **INNOSLIDE** | PVC | Coulissant PVC « à frappe », A*4/E*7A/V*B3, Uw 1,3 *(docs 424, 433)* | **⚠️ système porteur à préciser** | **⚠️** |
| ~~LUMINE~~ (`lumine`) | ALU | TECHNAL® *(doc 424)* | — hors périmètre — | — |

**Interdits de transfert** (règles dures pour la génération) :
- Une valeur (cote, Uw, ref) lue pour une gamme/profondeur **ne se transfère jamais** à une autre :
  Perform 70 ≠ Perform 76, Perform ≠ Textural ≠ Hybride.
- Une ref de la Gamme 70 (ex. dormant 6111) n'existe pas « en version 76 » par simple analogie.

---

## 6. Discriminants par préfixe de référence

Le préfixe d'une référence suffit souvent à router matière + fournisseur **avant toute recherche** :

| Motif de référence | Univers | Exemples en base |
|---|---|---|
| Numérique 4 chiffres (6xxx), A1xx/A2xx/A4xx, V0xx, 9Fxx, Z0xx, JAxxx | **PVC Kömmerling/profine (70/76)** | 6111, 6150, A107, V054, 9F67, Z043, A465 |
| **TGY**xxxx | Technal ALU — **coulissant** SOLEAL GY | TGY3702/3703/3704, TGY1202 |
| **TFY**xxxx | Technal ALU — **frappe** SOLEAL FY | ⚠️ à confirmer sur exemples |
| **TPY**xxxx / PYxxxx | Technal ALU — SOLEAL PY **⚠️ famille à préciser** | seuil PY1100 |
| Refs Roto (6 chiffres type 495096, ou noms NX/Safe) | Ferrure ROTO — transverse | 495096, 498312 |

**Règle** : une question contenant une ref `T??xxxx` est une question **ALU/Technal** — inutile (et
nuisible) de chercher dans les DTD PVC 70/76, et réciproquement.

---

## 7. Règles de routage pour l'assistant

1. **Gamme citée → traduire puis restreindre.** « Perform » → chercher dans les documents PVC
   Kömmerling/profine (70/76) ; ne pas mobiliser les documents Textural/Lumine/ALU. « Textural » →
   ses documents propres. En l'absence d'attribution fiable, appliquer au minimum un **boost
   négatif** sur les gammes non citées.
2. **Profondeur citée (70/76) → choisir le système.** « 70 » → Gamme 70 / e.VOLUTION / e.XCLUSIVE
   *(docs 421, 423, 411)* ; « 76 » → 76 ADVANCED *(docs 412, 413)*.
3. **Type d'ouverture comme indice système (PVC)** : frappe fenêtre → 76 ADVANCED **ou** Gamme 70 ;
   coulissant PVC → e.XCLUSIVE (70) ou INNOSLIDE ; porte → e.VOLUTION. **⚠️ répartition exacte à
   valider.**
4. **Ref détectée → discriminant §6 prioritaire** sur tout le reste (une ref est moins ambiguë
   qu'un nom de gamme).
5. **Ambiguïté réelle** (gamme non citée + pas de ref + terme générique type « dormant ») →
   demander la gamme plutôt que de mélanger les univers.
6. **Ne jamais répondre depuis ce tableau seul** pour une cote/valeur : cette fiche route la
   recherche, la **valeur** vient toujours du document.

---

## 8. Inventaire documentaire (les documents qui font foi, ids en base)

### PVC 70
| id | Document | Rôle |
|---|---|---|
| 421 | Poster_System_70_03_2025_Profilés_Principaux | Référentiel profilés principaux Gamme 70 |
| 423 | Poster_System_70_03_2025_Profilés_Complémentaires | Référentiel profilés complémentaires |
| 411 | DTD-DBV-6-16-2335_V5 | DTD e.VOLUTION / e.XCLUSIVE (portes, coulissants, dormants 70, seuils) |

### PVC 76
| id | Document | Rôle |
|---|---|---|
| 412 | 6-16-2334_V5_TROCAL_76_ADVANCED_PROFINE (2) | DTA 76 ADVANCED (fenêtres frappe) |
| 413 | DTD-DBV-25-06-16-2334_V5_TROCAL_76_ADVANCED (1) | DTD 76 ADVANCED |

### Marketing / transversal Proferm (définition des gammes commerciales)
| id | Document | Rôle |
|---|---|---|
| 424 | 2024-04_CATALOGUE-GENERAL_WEB | Définit PERFORM/HYBRIDE/LUMINE/TEXTURAL + fournisseurs |
| 425 | 2023-06_DEPLIANT-GENERAL | Idem, dépliant |
| 426 / 430 / 431 | Hybride, Perform Hybride, DEPLIANT-HYBRIDE | Gamme Hybride |
| 427 / 433 | Innoslide, DEPLIANT-INNOSLIDE | Coulissant Innoslide |
| 389 | Mise en oeuvre 9708 - Fer cintrés | Profine, mise en œuvre **⚠️ 70 ou 76 ?** |

### Ferrures (transverse, ne pas router par gamme)
383, 384, 391, 415, 390 (ROTO NX, Safe E, Eneo CC).

### Hors périmètre PVC 70/76 (ALU Technal → gamme LUMINE)
393-409 (SOLEAL FY/GY/PY, LUMEAL, DTA Technal, seuil PY1100), 429 (Lumine65), 428/432 (Lumeal).

---

## 9. État du système d'attribution & trous connus

- Les colonnes existent déjà (`source`, `materials`, `proferm_gammes`, `product_types`,
  `classification_status`) mais les données sont **inutilisables en l'état comme filtre** :
  la plupart des docs classés « complete » portent **toutes** les gammes à la fois (ex. docs 411,
  412, 421 → `['perform','lumine','hybride','textural']`), ce qui ne filtre rien ; **18/56 docs**
  sont vides (`Inconnu`/`incomplete`), dont tout le corpus Technal.
- **Action n°1 (rapide, 56 docs)** : réattribuer à la main avec ce vocabulaire — c'est le prérequis
  du filtrage dur « Perform ne cherche pas dans Textural ».
- Trous de connaissance à combler par un humain : cases ⚠️ des §3-§7 (Textural : base système ;
  Innoslide : système porteur ; répartition frappe/coulissant 70 vs 76 ; familles TFY/TPY).
- Références 76 ADVANCED : inventaire à extraire des docs 412/413.
