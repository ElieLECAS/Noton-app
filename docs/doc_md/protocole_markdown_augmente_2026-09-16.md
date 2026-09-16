# Protocole — markdown augmenté page par page

Version 2, 2026-09-16. Remplace la version 1 du même jour (conservée dans
`protocole_markdown_augmente_2026-09-16_v1.md`). Document de travail : c'est la règle que je suivrai
pour rédiger le markdown de chaque PDF que tu me donneras, et la spécification que l'ingestion devra
parser.

Trois sources l'étayent : la **sonde de lecture isolée** du 15/09 (62 questions, page de preuve seule,
`docs/audit_generation_texte_vs_image_2026-09-15.md`), le **premier document rédigé sous ce
protocole** (Roto Eneo CC, 10 pages, `docs/roto_eneo_cc_notice_simplifiee_2022.md`, importé comme
document 390) et une **revue de la littérature** menée le 16/09 (annexe A).

### Ce qui change en v2

Sept changements. Les six premiers sont tirés du test Eneo CC et de la relecture des deux
transcriptions de référence (Perform 76, Roto NX) ; le septième a été ajouté le même jour, pendant
la retranscription complète du Roto NX KSR, qui l'a rendu nécessaire. Aucune règle de la v1 n'est
retirée ; deux sont corrigées.

| # | Changement | Où | Pourquoi |
|---|---|---|---|
| 1 | Le titre de page porte le numéro **PDF**, le numéro imprimé entre parenthèses | § 1.3 | la v1 mettait `page_pdf="8"` au-dessus de `## Page 5` : deux numéros contradictoires sous les yeux du modèle |
| 2 | Tiret en cellule : vide par défaut, symbole autorisé **seulement s'il est déclaré** en conventions | § 1.2, § 2.3 | la v1 l'interdisait ; sa propre référence Perform 76 en compte 55 |
| 3 | Gabarit de transcription d'une **planche** : relevé `Cote / Couleur / Ce qu'elle mesure / Source du rôle` et trois marqueurs d'incertitude | § 4 | 62 % du corpus est planche CAO ; la v1 n'avait aucune règle pour un dessin |
| 4 | Ligne de **portée** sous chaque tableau de valeurs ; portée du document en tête des conventions | § 1.2, § 2.4 | c'est la règle implicite d'une autre page qui rend une valeur juste ou fausse (Perform 70 ≠ 76, feuillure 62) |
| 5 | Section normée pour le **texte du fichier non visible** à l'impression | § 2.5 | Eneo p. 3 : cotes et mentions allemandes recouvertes par l'illustration |
| 6 | Paire **nomenclature / schéma** : la page qui dessine nomme ses repères | § 2.1 | Roto p. 37 réduite à « 1, 1a, 3… » est muette pour BM25 et pour le modèle |
| 7 | Ligne déclarative **« Valeurs lues sur l'image »** pour les cotes absentes de la couche texte | § 4.4, § 7 | sans elle, un dessin vectorisé ou matriciel force à marquer `[illisible]` des cotes parfaitement lisibles (Roto NX KSR p. 89-90) |

Le fichier Eneo CC déjà importé reste conforme sans réimport : pas de pagination imprimée, symboles
déclarés, section « non visible » déjà présente sous un autre titre.

---

## 0. Ce que la mesure dit, avant toute règle de forme

| Matière donnée au générateur | Réussite (62 questions) |
|---|---|
| PNG de la page seul | 85,5 % |
| PNG + fragments indexés actuels | 12/13, aucun gain |
| **PNG + markdown de page** | **96,8 %** |
| Markdown de page seul | 98,4 % |

Le premier document rédigé sous le protocole confirme : sur Eneo CC, le croisement des nombres de la
couche texte du PDF et du markdown donne **0 omission, 0 invention** sur 10 pages ; en application,
image + markdown, **plus de 90 %** de bonnes réponses (test du 16/09).

La littérature dit la même chose sur un corpus indépendant. UniDoc-Bench (Salesforce, 70 000 pages,
8 domaines dont la construction) mesure image seule **52,7 %**, texte seul **61,9 %**, fusion des deux
**68,5 %**. L'image seule y est le pire des quatre paradigmes testés.

**Conséquence, actée** : la position « image only » du 14/09 est contredite par deux mesures
indépendantes. Le générateur reçoit **le PNG ET le markdown de la même page**, jamais l'un à la place
de l'autre.

---

## 1. Format du fichier

Un fichier par document : `media/page_markdown/<document_id>.md`. Encodage UTF-8, fins de ligne LF.

### 1.1 En-tête de fichier

```markdown
---
document_id: 438
titre: Dossier technique gamme Perform 76
source_pdf: media/documents/438.pdf
source_sha256: 3f2a…            # hachage du PDF, pour détecter la dérive
pages_pdf: 26                    # doit égaler le nombre de pages du PDF
pagination: pdf = imprimé + 3    # ou "identique", ou "aucun numéro imprimé"
version_protocole: 2
redige_le: 2026-09-16
---
```

Le hachage n'est pas décoratif : la dérive entre un PDF remplacé et son markdown **dégrade en
silence**, sans faire échouer aucune métrique RAG classique. C'est le piège le mieux documenté du
dossier.

### 1.2 Bloc de conventions du document

Immédiatement après l'en-tête, une section `## Conventions du document`. Elle énonce tout ce que le
document tient pour implicite, **dans cet ordre** :

1. **Portée du document** (obligatoire, en premier) : gamme, produits ou références couverts, version
   et date du document, et ce que le document n'est pas (« extrait, ne remplace pas IMO_438 »).
2. **Unités par défaut** et unités d'exception.
3. **Code couleur de cotation** et sens de lecture des vues.
4. **Symboles employés dans les cellules** : chaque symbole qui apparaît dans un tableau du document
   (■, –, ☐, ×, ✓…) est défini ici avec son sens. Un symbole non déclaré est une erreur de rédaction
   (§ 2.3). Une cellule vide n'est pas un symbole : elle signifie « le document n'indique rien à cet
   endroit, ce n'est pas un zéro », et cela se déclare une fois ici.
5. **Abréviations maison** et définitions relevées dans le document (HFF, galets E / V / P…).
6. **Pagination** : décalage page PDF / page imprimée, ou « aucun numéro de page imprimé ».

Exemple (Eneo CC, abrégé) :

```markdown
## Conventions du document

- Portée : serrure motorisée Roto Safe E | Eneo CC distribuée par PROFERM ; extrait de la notice
  complète IMO_438, qu'il ne remplace pas ; pages 4 à 8 = contrôle d'accès 4 en 1 associé.
- Toutes les cotes en millimètres sauf mention contraire (V, A, mA, °C, m).
- Symboles des tableaux d'entretien (page 10) : ■ = entreprise spécialisée uniquement ;
  – = pas par l'utilisateur final ; ☐ = entreprise spécialisée ou utilisateur final.
- Une cellule vide signifie que le document n'indique aucune valeur : ce n'est pas un zéro.
- Aucun numéro de page imprimé : les numéros sont ceux du PDF.
```

C'est ce bloc, et non une contextualisation générée page par page, qui porte l'autonomie. Le motif
est mesuré : les **en-têtes de chunk déterministes** (dsRAG) donnent 96,6 % sur FinanceBench contre
32 % pour un RAG vanilla. Un en-tête stable, vérifiable et reproductible bat un contexte réécrit par
un LLM, qui peut halluciner.

### 1.3 Séparateur de page

```markdown
<!-- PageBreak page_pdf="8" page_imprimee="5" -->
## Page 8 (imprimée 5) — Parcloses ouvrants et dormants
```

Sans pagination imprimée :

```markdown
<!-- PageBreak page_pdf="2" -->
## Page 2 — Montage de la serrure : fraisages et retournement du pêne
```

Le titre commence **toujours** par le numéro PDF : celui du séparateur, et celui du marqueur
`[page N]` que le générateur voit dans son contexte. Le numéro imprimé n'apparaît qu'entre
parenthèses. En v1, le titre portait le numéro imprimé et le marqueur le numéro PDF : le modèle avait
les deux sous les yeux, et c'est précisément le bug « numéro imprimé cité au lieu du numéro PDF »
(51,6 % de pages décalées le 14/09) que le séparateur devait tuer.

Trois raisons pour cette syntaxe de séparateur :

- `<!-- PageBreak -->` est la **seule syntaxe formellement spécifiée par un éditeur majeur** (Azure AI
  Document Intelligence), et la seule non ambiguë : `---` est déjà un `<hr>` markdown **et** un
  délimiteur de front-matter YAML.
- Les attributs `page_pdf` / `page_imprimee` sont un ajout. Un séparateur nu ne survit ni à une
  troncature ni à une concaténation partielle, et l'appariement au PNG a besoin d'un identifiant.
- Porter **les deux numéros** dans le séparateur, et le numéro PDF en tête du titre, élimine par
  construction le décalage entre l'index physique du PDF et son *page label* logique.

Le parseur doit **refuser** un fichier dont le nombre de séparateurs ne correspond pas au nombre de
pages du PDF. Un décalage silencieux fausse tout à partir de la première erreur.

---

## 2. Contenu d'une page

### 2.1 Règle cardinale : la page se lit seule

Aucun renvoi non résolu. Pas de « voir ci-dessus », « comme mentionné page 12 », « cf. légende ».
Une règle énoncée page 12 qui s'applique page 47 est **recopiée en toutes lettres page 47**, avec
mention de son origine :

```markdown
> **Convention rappelée (page 1)** — cotation en bleu = épaisseur du vitrage ; les cotes noires
> désignent les dimensions du profilé.
```

**Paire nomenclature / schéma.** Quand un document sépare la liste des repères (page de nomenclature)
et leur implantation (page de schéma), les deux pages se complètent à la lettre :

- la page de schéma **nomme** chaque repère qu'elle place : « Repère 7 — Verrouilleur médian
  horizontal », jamais « repères 1, 1a, 3, 7 » ;
- la page de nomenclature dit où se trouve le schéma : « implantation sur le schéma page PDF 39 ».

Une page de schéma réduite à une liste de numéros est muette pour la recherche et pour le modèle
(Roto NX page 37, v1).

C'est la tâche que la littérature appelle *decontextualization* (Choi et al., TACL 2021) : réécrire
un énoncé pour qu'il reste interprétable hors de son contexte, en préservant sa valeur de vérité.
Le guide de rédaction le plus concret du domaine (kapa.ai) en fait sa règle centrale et traite les
« comme mentionné plus haut » comme le signal d'un contexte manquant.

### 2.2 Ce qu'on ajoute, ce qu'on n'ajoute pas

**On ajoute** : les conventions applicables, les règles énoncées ailleurs qui régissent cette page,
la résolution des renvois, le rôle d'une colonne quand l'en-tête est tourné ou abrégé, l'unité quand
elle n'est écrite qu'une fois en haut du tableau, la portée de validité des valeurs (§ 2.4).

**On n'ajoute pas** : de l'expertise métier, une explication de la fonction d'une pièce, une
reformulation « pédagogique ». C'est exactement ce que faisait la couche de chunks contextuels
supprimée le 16/09, dont le prompt disait « ta mission n'est PAS de recopier ». Le markdown est une
**transcription augmentée de son contexte**, pas un commentaire.

**La frontière, sur un dessin.** Lire un trait de cote, c'est lire le dessin : ses deux extrémités
disent ce que la cote mesure (« du centre du boîtier de serrure à l'axe de la gâche haute »). C'est
une transcription, elle est permise et attendue, et elle se marque `[déduit du trait de cote]` quand
aucun libellé ne l'écrit en toutes lettres (§ 4). Dire à quoi sert la gâche, pourquoi la cote vaut
752, ou ce qu'il faut faire si elle ne convient pas, c'est de l'expertise : interdit.

### 2.3 Valeurs illisibles, rôles inconnus, cellules vides

Trois marqueurs, et aucun autre :

| Marqueur | Quand | Jamais |
|---|---|---|
| `[illisible]` | la valeur ne peut pas être lue avec certitude | une valeur supposée |
| `[rôle non écrit]` | le nombre est lisible mais rien sur la page ne dit ce qu'il mesure | un rôle deviné |
| `[déduit du trait de cote]` | le rôle vient des extrémités du trait de cote, pas d'un libellé | omettre la mention |

```markdown
| 2454 | 31 | [illisible] |
```

Le chiffre qui justifie `[illisible]` est le plus brutal du dossier : sur 443 étiquettes délibérément
floutées, un modèle frontière fabrique une valeur **96 % du temps** et ne s'abstient que dans 8 % des
cas ; un juge LLM ne détecte que 7 % de ces fabrications. Matérialiser l'incertitude dans le markdown
transforme une hallucination silencieuse en information exploitable.

**Cellules vides et symboles.** Une cellule vide reste vide : le document n'indique rien à cet
endroit, et ce n'est pas un zéro. On n'y écrit **jamais** un tiret ou un symbole de remplissage : le
« — » a été lu comme un zéro le 15/09, et la référence Perform 76 en compte 55. Quand le document
imprime lui-même un symbole dans ses cellules (■ – ☐ de la page d'entretien d'Eneo CC), on le
transcrit tel quel **et on le déclare dans les conventions** (§ 1.2, point 4). Règle mécanique : un
tiret ou un symbole en cellule qui n'est pas déclaré est une erreur de rédaction.

### 2.4 Portée de validité des valeurs

Une valeur n'est vraie que dans un domaine : une gamme, un dormant, une feuillure, une version. Quand
ce domaine n'est pas écrit dans le tableau lui-même, il l'est ailleurs sur la page, sur une autre
page, ou nulle part parce que le document le tient pour évident. Dans les trois cas, il est
**recopié sous le tableau**, en une ligne :

```markdown
Portée : dormant 76177 uniquement · feuillure de 62 mm avec joint post-extrudé · gamme Perform 76.
```

Ce que la ligne porte, quand c'est connu : gamme, référence de profilé ou de produit concernée,
conditions énoncées par le document (« valable pour… »), version. Ce qu'elle ne porte pas : une portée
supposée. Si le document ne le dit nulle part : `Portée : non précisée par le document.`

C'est le correctif des deux erreurs mesurées les plus coûteuses : le transfert d'une valeur d'une
gamme à une autre (Perform 70 ≠ Perform 76), et la note générale de la page (« valable pour une
feuillure de 62 mm ») prise pour la valeur d'un repère. Perform 76 page 5 (v1) le fait pour
l'épaisseur de remplissage et pas pour les parcloses : c'est la règle qui manquait.

### 2.5 Texte du fichier non visible à l'impression

Un PDF peut porter une couche de texte que le rendu ne montre pas : cotes recouvertes par une
illustration, libellés d'une langue source laissés sous la traduction, blocs hors page. Ces valeurs
existent pour la recherche plein texte et pour le contrôle de non-omission (§ 7), mais pas pour le
lecteur du document. Elles vont dans une section **au titre fixe**, en fin de page, jamais mêlées au
contenu visible :

```markdown
### Texte du fichier non visible à l'impression

Valeurs relevées sans rôle attribuable : 230 · 210 · 20 · 10,4 · 28 · 20,5 · 180 · 50 · 26,5 · 1 ·
4,9 ø · 8,5 · 59.
Mentions : « Einbausituation » en regard du titre « Montage » ; « [1] Rahmenteil », « [2] Flügelteil ».
```

Le générateur a pour consigne que l'image fait foi : une valeur présente ici et absente de l'image
est, pour lui, une valeur que le document ne montre pas. Cas source : Eneo CC page 3.

---

## 3. Tableaux

### 3.1 Format : markdown par défaut, HTML pour les fusions

Markdown pipe **perd physiquement** les cellules fusionnées : le texte est écrit à sa position
d'origine et toutes les autres positions du span sortent vides. Azure Document Intelligence est
passé aux tables HTML avec `rowspan`/`colspan` exactement pour cette raison, et les mesures donnent
environ **deux fois mieux en HTML sur la détection de fusions**.

Mais HTML coûte cher : +42 % de tokens à qualité quasi égale sur des tableaux simples, et à contexte
long il **sort tout simplement de la fenêtre** là où markdown tient encore.

D'où la règle conditionnelle :

| Cas | Format |
|---|---|
| Tableau régulier, en-tête sur une ligne | markdown pipe |
| Cellules fusionnées, en-têtes multi-niveaux, lignes groupées | HTML (`<table>`, `rowspan`, `colspan`, `<caption>`) |

### 3.2 Chaque ligne se suffit, chaque tableau se suffit, et on ne transpose jamais

Répéter la référence produit dans chaque ligne plutôt que de la laisser dans un en-tête de groupe
fusionné. Chaque tableau porte l'unité dans ses en-têtes de colonnes même si les conventions la
donnent, et sa ligne de portée en dessous (§ 2.4). L'orientation compte énormément : la justesse
tombe de **93,4 % à 32,5 %** entre une table à l'orientation d'origine et la même table transposée.
C'est bien plus que tout choix de balisage.

### 3.3 Découper les grands tableaux

C'est la limite qu'aucun format ne franchit : la récupération de la bonne **ligne** reste sous 12 %
pour les modèles ouverts et sous 6 % pour les propriétaires, quel que soit le balisage. Et la seule
taille du tableau coûte 10 à 18 points de justesse au-delà de ~4 000 tokens.

Donc : un tableau de plus d'une vingtaine de lignes se découpe en **sous-blocs titrés** dans la page,
par groupe naturel (ici : parcloses ouvrants / parcloses dormants), chacun avec son en-tête complet
répété et sa ligne de portée. Ne pas espérer qu'un modèle lise correctement une ligne dans un bloc de
40.

---

## 4. Planches et schémas

Une planche est un dessin coté : coupe de profilé, vue de fraisage, schéma d'implantation, plan de
câblage. Sa transcription n'est ni une description ni un commentaire : c'est un **relevé**, une ligne
par cote, dans un tableau à colonnes fixes. Motif mesuré : le 14/09, sur les planches Perform 76, le
relevé exhaustif suivi d'une décision réussit 7 fois sur 8 là où la question directe réussit 2 fois
sur 8 ; le 15/09, les quatre cotes de parclose stablement fausses passent dès que la valeur est écrite
dans une colonne nommée.

### 4.1 Gabarit, par vue

Une planche porte souvent plusieurs vues (fraisage du vantail / du dormant ; coupe verticale /
horizontale ; détail agrandi). Chaque vue a son bloc :

```markdown
### Vue : fraisage du dormant (coupe verticale, axe de fraisage à gauche, lecture de haut en bas)

Ce que montre la vue : la têtière du dormant avec ses deux gâches et le fraisage du dispositif
d'ouverture, cotés depuis le centre du boîtier de serrure.

| Cote | Couleur | Ce qu'elle mesure sur le plan | Source du rôle |
|---|---|---|---|
| 752 | noir | du centre du boîtier de serrure à l'axe de la gâche haute | [déduit du trait de cote] |
| 135 | noir | hauteur de la gâche haute | libellé « Gâche haute » |
| 19 | noir | largeur de la gâche haute | [déduit du trait de cote] |
| 24,5 | noir | | [rôle non écrit] |
| Ø 20 | noir | perçage sur l'axe de fraisage | libellé « Ø 20 » sur le plan |

Repères présents : 1 (dormant), 2 (ouvrant).
Mentions imprimées : « Le fraisage dépend des hauteurs de gâche. » « L'axe de fraisage dépend du
profil utilisé. »
Non lisible : [illisible] sur la vue de détail du dispositif d'ouverture (deux cotes).
Portée : Roto Safe E | Eneo CC, montage sur dormant ; largeur de fraisage fonction de la têtière.
```

Colonnes du relevé :

- **Cote** : le nombre tel qu'écrit, avec son préfixe (Ø, R, E92, D+20) ; jamais converti, jamais
  arrondi.
- **Couleur** : celle du chiffre sur le plan quand le document a un code couleur (bleu = vitrage,
  noir = profilé sur Perform 76) ; vide sinon. C'est la colonne qui neutralise le « plus petit des
  deux nombres » mesuré le 14/09 : quand le code couleur est dans les conventions et la couleur dans le
  relevé, le modèle n'a plus à trancher.
- **Ce qu'elle mesure sur le plan** : les deux extrémités du trait, ou l'élément coté. Vide si
  `[rôle non écrit]`.
- **Source du rôle** : d'où vient la colonne précédente. Trois valeurs possibles : un libellé cité
  entre guillemets, `[déduit du trait de cote]`, `[rôle non écrit]`. C'est la colonne qui trace la
  frontière du § 2.2 : elle dit ce qui est lu et ce qui est déduit.

Après le relevé, quatre lignes fixes : **Repères présents** (numéros et noms, § 2.1), **Mentions
imprimées** (remarques et notes de la vue, mot pour mot), **Non lisible** (ce que la transcription
n'a pas su lire), **Portée** (§ 2.4).

### 4.2 Trois interdits

- **Aucun nombre nu.** Un nombre est dans le relevé avec son rôle ou son marqueur, ou il n'est pas
  dans le markdown. « Cotes verticales : 39 ; 49 ; 70 » (Perform 76 page 11, v1) est un sac de
  nombres : le modèle y pioche.
- **Aucun regroupement par position** (« zone du boîtier : 22 · E92 · 20 · 20 · 17 ») sans que
  chaque cote ait sa ligne. Le regroupement par zone est permis comme titre de sous-bloc, pas comme
  substitut du relevé.
- **Aucune lecture sous le seuil de certitude.** Une cote dont un chiffre est douteux s'écrit
  `[illisible]`, pas « probablement 24,5 ».

### 4.3 Plan de câblage et schéma fonctionnel

Même logique, autres colonnes : `Élément | Borne ou fil | Relié à | Source`. Chaque liaison est une
ligne ; les couleurs de fils sont des valeurs comme les autres. Eneo CC page 6 en est le modèle.

### 4.4 Valeurs lues sur le dessin et non présentes dans la couche texte

Sur une planche vectorisée ou collée en image matricielle, la cote **n'existe que dans le dessin** :
la couche texte du PDF n'en porte rien. Le relevé la transcrit donc telle qu'elle est lue, et la
page le **déclare** en une ligne au format fixe, en fin de page :

```markdown
Valeurs lues sur l'image (absentes de la couche texte) : 36 · 12 · 16 · 48 · 80 · +10 · 74 · 20,0 · 18,5
```

Cette ligne n'est pas un ornement : c'est elle qui distingue une valeur **lue** d'une valeur
**fabriquée**, et le contrôle de non-invention (§ 7, assertion 2) s'en sert exactement pour cela.
Toute valeur d'une page qui ne figure ni dans la couche texte ni dans cette ligne est signalée
comme inventée.

Corollaire, appris à la transcription du Roto NX KSR : une consigne « tout nombre écrit doit figurer
dans la couche texte » est **trop stricte** et fait marquer `[illisible]` des cotes parfaitement
lisibles à l'écran. La règle juste est : tout nombre écrit vient de la couche texte, **ou** est
déclaré comme lu sur l'image, **ou** est marqué `[illisible]`. Rien d'autre.

### 4.5 Ce que la planche ne remplace pas

Le PNG de la page reste joint à la génération : le relevé dit ce qu'il y a et ce que ça mesure,
l'image reste la preuve. Une ligne de relevé qui contredit visiblement l'image doit être dite comme
telle par le générateur, pas arbitrée en silence (§ 6.3).

---

## 5. Comment le markdown entre dans l'application (livré le 16/09)

Un bouton **« Markdown »** sur la carte du document, à côté de « Chunks », visible pour qui a le droit
d'écrire. Il ouvre une modale en trois temps.

1. **État actuel** : le document a-t-il déjà un markdown, combien de pages, aligné sur le PDF ou non,
   le PDF a-t-il changé depuis la rédaction.
2. **Dépôt** : on glisse le fichier `.md` sur la zone, on le choisit, ou on colle le contenu. Le dépôt
   déclenche immédiatement une vérification.
3. **Rapport de vérification, avant toute écriture** : erreurs en rouge, avertissements en orange, et
   le tableau page par page (numéro PDF, numéro imprimé, titre, nombre de caractères). Le bouton
   **Importer** reste désactivé tant que la vérification échoue.

C'est ce contrôle à blanc qui rend l'import sûr. Le décalage de pagination est la panne la plus grave
et la plus silencieuse : on la montre avant d'écrire, jamais après.

### Ce qui bloque, ce qui avertit

| Cas | Verdict |
|---|---|
| Nombre de sections différent du nombre de pages du PDF | **refus** |
| Page du PDF absente du markdown | **refus** |
| Numéro de page en double | **refus** |
| Numéro hors des bornes du PDF | **refus** |
| Séparateur sans numéro (`<!-- PageBreak -->`) | **refus** |
| Pages dans le désordre | avertissement |
| Page sans contenu | avertissement |
| Section « Conventions » absente | avertissement |
| Hachage du PDF différent de celui déclaré | avertissement |
| PDF source introuvable | avertissement, comptage impossible |

### Endpoints

| Méthode | Chemin | Rôle |
|---|---|---|
| `GET` | `/api/library/documents/{id}/page-markdown` | état, et contenu avec `?raw=1` |
| `POST` | `…/page-markdown/validate` | rapport de vérification, **n'écrit rien** |
| `POST` | `…/page-markdown` | import, refusé en 422 si la vérification échoue |
| `DELETE` | `…/page-markdown` | détache, le document repasse sur ses fragments |

Le markdown est stocké en fichier, `media/page_markdown/<document_id>.md`, et non en base : il est
écrit et corrigé à la main, donc il doit se differ, se relire et se réimporter page par page. Le
cache du packer est purgé à chaque import.

---

## 6. Ce que l'ingestion en fait

```
markdown augmenté (1 fichier / document, sections par page)
   │
   ├─► parse déterministe par séparateur  ──► 1 enregistrement par page
   │        vérification : nb sections == nb pages du PDF, sinon REFUS
   │
   ├─► matière LUE : section de la page (+ fenêtre) jointe au PNG de la même page
   │
   └─► corpus BM25 : une ligne par page, DEPUIS le markdown (content_type page_markdown)
            → le corpus de recherche cesse d'être « col1:**ION** ■Châssis… »

ColPali continue d'indexer les PNG du PDF, inchangé (il ne dépend en rien du texte).
```

### 6.1 Fenêtre de texte et nombre d'images

Le prix mesuré tranche : une page de markdown vaut ~880 tokens, un PNG de page à 220 dpi ~2 300.

| Élément | Tokens |
|---|---|
| Une page de markdown | ≈ 880 |
| Un PNG de page | ≈ 2 300 |
| Cahier technique Perform 76 entier (26 pages) | ≈ 13 000 |
| Budget de packing actuel | 50 000 |

Donc : **markdown généreux, images rares**. Document entier sous ~15 000 tokens, sinon fenêtre de
± 2 pages autour de la page élue. Deux ou trois PNG seulement, sur les pages réellement élues.
Empiler les images est contre-productif : la performance des modèles de vision décroît de façon
marquée quand le contexte visuel s'allonge.

### 6.2 Ordre dans le prompt

Documents en tête, images avant le texte qui les commente, question en dernier. Anthropic mesure
jusqu'à **+30 % de qualité de réponse** en plaçant la requête en fin de prompt sur du multi-documents,
et recommande explicitement image-puis-texte avec un libellé court par image (`Page 47 :`).

### 6.3 En cas de conflit entre le markdown et l'image

Le conflit inter-modalités est documenté et persistant, indépendamment de la taille du modèle. Une
ligne de prompt suffit, et elle tranche dans le sens de ta règle : **l'image porte la vérité du
document ; le markdown en est la transcription. En cas de désaccord visible, le dire plutôt que
choisir.**

---

## 7. Contrôle qualité

Une batterie d'assertions binaires par page, rejouée à chaque régénération, sur le modèle
d'olmOCR-Bench :

| # | Assertion | État au 16/09 |
|---|---|---|
| 1 | **Comptage** : nombre de sections de page == nombre de pages du PDF | livré (`validate`) |
| 2 | **Non-invention** : toute valeur numérique du markdown existe dans la couche texte du PDF, ou est déclarée par la ligne « Valeurs lues sur l'image » de sa page (§ 4.4), ou est marquée `[illisible]`. Le contrôle `_ungrounded_numbers` de `sav_extraction_service.py` est réutilisable tel quel | **écrit** (Roto NX KSR, 16/09) : 0 sur 124 pages |
| 3 | **Non-omission** : tout nombre de la couche texte du PDF se retrouve dans le markdown, section « non visible » comprise (§ 2.5) | **écrit** (Roto NX KSR, 16/09) : 0 sur 124 pages |
| 4 | **Structure** : chaque tableau a autant de cellules par ligne que d'en-têtes | **écrit**, vérifié tableau par tableau |
| 5 | **Autonomie** : aucun renvoi qui SORTE de la page. Un « ci-dessous » suivi de sa localisation sur la page est résolu ; « voir page », « page précédente / suivante » ne le sont jamais | **écrit** |
| 6 | **Dérive** : le hachage du PDF source correspond à celui de l'en-tête | livré, en avertissement |
| 7 | **Titre** : le premier nombre du titre de page est égal à `page_pdf` (§ 1.3) | **écrit** |
| 8 | **Symboles** : aucun tiret ni symbole en cellule qui ne soit déclaré dans les conventions (§ 2.3) | **écrit** |

Les points 2 et 3 ne sont vérifiables que sur les pages ayant une couche texte. Sur une planche
vectorisée comme Perform 76, il n'y a pas de filet automatique : **la relecture humaine des pages à
valeurs reste obligatoire**. C'est ce qui a produit les 96,8 % de la sonde. Le relevé du § 4 rend
cette relecture mécanique : une ligne par cote, à cocher sur l'image.

---

## 8. Qui rédige quoi

Le bouton d'import est indifférent au producteur. Trois voies coexistent :

| Profil de page | Producteur | Coût |
|---|---|---|
| Prose, 1 à 3 colonnes | pymupdf4llm + nettoyage, déterministe | 0 |
| Tableau régulier | pymupdf4llm pour les lignes + en-tête réparé | 0 |
| Planche cotée, texte sur image, PDF vectorisé | rédaction ici selon le § 4, avec relecture | 1 passe + relecture |
| Scan | OCR puis une des voies ci-dessus | 1 appel |

Sur la voie « planche », deux faits à garder en tête :

- **~80 % est le plafond de l'état de l'art** pour un modèle de vision frontière lisant une planche
  cotée dense en une passe pleine page (79,96 % pour le meilleur, 40,5 % et 39,6 % pour deux autres
  modèles connus, sur extraction de cotes vérifiée manuellement). D'où la relecture humaine.
- Le seul dépassement publié passe par la **détection de régions avant lecture** : nDCG@3 de 0,21-0,29
  en pleine page contre 0,56 en restreignant aux régions détectées. À garder en réserve, hors ligne,
  au moment de la transcription, jamais dans le chemin de lecture en ligne.

---

## Annexe A — Sources

Toutes consultées le 16/09/2026.

| Sujet | Source | Chiffre retenu |
|---|---|---|
| Contextual Retrieval | anthropic.com/news/contextual-retrieval, 19/09/2024 | −49 % d'échec de retrieval avec BM25 ; 1,02 $/M tokens |
| En-têtes de chunk déterministes | github.com/D-Star-AI/dsRAG | 96,6 % vs 32 % sur FinanceBench |
| Decontextualization | Choi et al., TACL 2021 | définition formelle de la tâche |
| Chunking par page | developer.nvidia.com, 18/06/2025 | 0,648 de justesse, σ 0,107, meilleur des découpages |
| Séparateur `<!-- PageBreak -->` | Azure AI Document Intelligence, v4.0 | seule syntaxe spécifiée par un éditeur |
| Tables HTML pour les fusions | Azure DI v4.0 ; TableEval, arXiv 2506.03949 | ~2× mieux sur cellules fusionnées |
| Tables markdown en contexte long | TQA-Bench, arXiv 2411.19504 | HTML sort du contexte à 64K, markdown tient |
| Coût tokens des formats | TeleTables, arXiv 2601.04202 | markdown −42 % de tokens, 0 à 1,6 pt d'écart |
| Récupération de ligne | TabVerse, arXiv 2606.09578 | < 12 % modèles ouverts, < 6 % propriétaires |
| Transposition | NAACL 2024, arXiv 2312.16702 | 93,4 % → 32,5 % |
| Image + texte à la génération | UniDoc-Bench, arXiv 2510.03663 | 52,7 / 61,9 / 68,5 % |
| Idem | OHR-Bench v1, arXiv 2412.02592 | +24,5 % de F1 en ajoutant le texte à l'image |
| Fabrication sur valeur illisible | Pebblous, 13/08/2026 (préprint) | 96 % de fabrication, 8 % d'abstention |
| Plafond sur planches cotées | businesswaretech.com, 07-08/2025 | 79,96 % au mieux |
| Détection de régions | BLUEPRINT, arXiv 2602.13345 | 0,21-0,29 → 0,56 nDCG@3 |
| Ordre du prompt | platform.claude.com, prompt engineering | jusqu'à +30 % |
| Dérive d'index | blogs.oracle.com | dégradation silencieuse |
| Tests unitaires par page | olmOCR-Bench, arXiv 2510.19817 | méthode |
| Rédaction RAG-friendly | docs.kapa.ai | règles de dépendance contextuelle |

### Ce que la littérature ne dit pas

- « Markdown augmenté » **n'existe pas** comme pratique nommée. Rien à lire sous ce nom.
- **Aucun standard** de séparateur de page : trois syntaxes rivales, aucune dominante.
- **Aucune recommandation officielle** d'Anthropic, OpenAI ou Google sur le format de tableau en
  entrée, ni sur la façon de réécrire un document pour qu'un LLM le lise.
- **Aucune étude** sur la lecture d'un code couleur de cotation, ni **aucun benchmark** sur les
  planches de menuiserie. Angle mort complet.
- **Aucune ablation publiée** « relevé exhaustif puis décision » contre « question directe » sur
  dessin technique : la mesure interne du 14/09 (7/8 contre 2/8) est en avance sur la littérature.
