# Audit du wiki LIA — rédaction, découpage, fidélité, protocole d'ingestion

Date : 2026-09-18 · Périmètre : `wiki_llm/wiki/` (73 pages, 535 000 caractères hors `index.md` et
`log.md`), `wiki_llm/CLAUDE.md`, `wiki_llm/README.md`. **Rien n'a été modifié dans le wiki.**
Les 73 pages ont été lues en entier ; les PDF n'ont pas été relus, sauf un sondage texte sur la
page 35 du catalogue général (grille de garanties : les 15 valeurs transcrites sont exactes). Le
rendu d'image PDF n'est pas disponible dans cet environnement : aucun sondage visuel n'a été
fait sur le cahier technique PERFORM76, qui n'a pas de couche texte.

---

## 0. Verdict

1. **La charpente est solide et doit être gardée.** 73/73 pages conformes OKF, 0 lien cassé,
   0 page hors index, 0 orpheline, 0 tableau de plus de 30 lignes, une source pour chaque
   affirmation, trois registres d'anomalies à identifiants stables, contradictions notées deux
   fois (registre + page). C'est rare et c'est ce qui rend le wiki fiable.
2. **Le fond est déjà plus qu'une transcription** : le wiki recoupe les documents, hiérarchise
   (« le document produit prime »), refuse d'interpoler, enregistre ce qu'il ne sait pas. Cette
   discipline est la valeur du wiki.
3. **Le registre d'écriture, lui, est celui d'un commentaire de documents et d'un journal
   d'enquête.** La source est le sujet des phrases (« le catalogue indique », « la brochure ne
   donne pas »), le wiki raconte sa propre histoire (« jusqu'ici le wiki ne connaissait pas »,
   « c'est la première coordonnée fournisseur que le wiki possède »), et il glisse des apartés
   commerciaux et des explications physiques non sourcées. La crainte est fondée : ce n'est pas
   encore une source technique écrite par un expert, c'est une lecture critique très bien faite.
4. **La duplication est le second défaut** : la grille de garanties est recopiée sur 13 pages,
   CTR-03 est argumentée sur 9 pages, VER-20 sur 5, la table des dimensions du DTA sur 4. Deux
   copies ont déjà dérivé (CTR-03 porte trois conclusions différentes selon la page).
5. **La fidélité n'a jamais été contrôlée après écriture** : `verified` est vide sur 73 pages,
   la seule erreur de transcription trouvée (INC-01) l'a été par relecture humaine, et deux
   échecs du golden tiennent à la rédaction. La seconde passe est justifiée, et sa liste de
   travail est déjà écrite : les pages disent elles-mêmes ce qu'elles n'ont pas transcrit.

Conclusion : **le wiki est bon, il est écrit dans le mauvais registre et il se répète.** Les deux
se corrigent par des règles d'écriture et une réécriture de forme, sans toucher au fond ni à
l'application. Le plan est en § 10.

---

## 1. Mesures

| Indicateur | Valeur | Lecture |
| --- | --- | --- |
| Pages / caractères hors index et journal | 73 / 535 000 | ≈ 183 000 tokens à 2,92 car./token |
| Part de `sources/` | 128 000 car., 24 % | 21 pages, en grande partie méta (structure du PDF, provenance, « ce qu'il apporte au wiki ») |
| Part de `anomalies/` | 49 000 car., 9 % | trois registres, avec des essais « en détail » et des tables dupliquées ailleurs |
| Parenthèses de citation dans la prose `(…, p. N)` | 403 | 5,5 par page ; 18 à 20 sur chaque page de gamme |
| Tournures « le document indique / ne donne pas / n'est pas transcrit » | 166 | la source comme sujet de phrase |
| Auto-références « le wiki », « cette page », « désormais » | 103 | le wiki raconte son ingestion |
| Apartés pédagogiques ou commerciaux | 88 | « c'est l'argument de vente le plus fort », « en clientèle », « erreur classique » |
| Passages en gras | 1 426 | 20 par page : l'emphase ne signale plus rien |
| Questions ouvertes dans le corps (« à vérifier », « à confirmer »…) | 90 | en plus des 35 VER- du registre |
| Pages `status: draft` | 33 / 73 | deux sens mélangés (§ 6) |
| Pages dont `stale_after` est déjà passé | 28 / 73 | dont les trois pages de cotes du système 76 (§ 6) |
| `sources[].last_modified` renseigné | 10 / 144 | le champ qui porte la date du document |
| `verified` renseigné | 0 / 73 | le champ prévu pour la seconde passe |
| Liens cassés / pages hors index / orphelines / tableaux > 30 lignes | 0 / 0 / 0 / 0 | conformité structurelle complète |

Les compteurs viennent de recherches de motifs sur les fichiers ; ce sont des ordres de grandeur,
pas des décomptes à l'unité.

---

## 2. Ce qui est bien et qu'il ne faut pas perdre

- **Les tableaux de cotes** : une ligne par référence, unité dans l'en-tête, contexte en colonne
  et non dans la cellule (`NT1947 | 76180 | 140` / `NT1947 | 76171 | 155`), valeurs admises
  énumérées (« 80, 95, 115, 135… il n'y a pas de 140 sur ce dormant »), exclusions en clair
  (« montage uniquement compatible avec les ouvrants »). C'est ce que le modèle lit bien : le
  golden est à 83,9 % et ses échecs ne viennent pas des tableaux propres.
- **La séparation des familles jumelles** (parclose d'ouvrant / de dormant, meneau d'ouvrant / de
  dormant, tapée / appui / patte de pose) énoncée sur les pages où l'on se trompe.
- **Les registres d'anomalies** : identifiants stables, verbatim de la source, correction
  probable, impact si non corrigé, entrées retirées conservées. Le lien inline `INC-05` que
  l'application résout.
- **La règle « le document produit prime »** et son refus de trancher sur la seule date.
- **Les pages `procedures/`** : étapes numérotées, condition d'interdiction en tête, « ce que le
  document ne dit pas » en liste. Les meilleures pages du wiki pour un lecteur humain.
- **Les locators `(schéma: raw/x.pdf, p. N)`** : l'application ouvre le PDF à la page. À garder
  tels quels.

---

## 3. Le registre : pourquoi ça se lit comme une transcription commentée

### 3.1 La source est le sujet de la phrase

Sur les pages de gammes, presque chaque phrase se termine par `(catalogue général, p. 10)` et
beaucoup commencent par « Le catalogue décrit… », « La brochure annonce… », « Le cahier ne dit
pas… ». Exemples typiques :

- `gammes/hybride.md`, section Cotes : « Le catalogue ne détaille pas les épaisseurs de profilé
  HYBRIDE70 et HYBRIDE76 séparément : il donne une plage de 70 à 76 mm pour la gamme (catalogue
  général, p. 10). Les noms des déclinaisons suggèrent 70 mm et 76 mm respectivement, mais le
  document ne l'écrit pas — à vérifier auprès du bureau d'études, entrée VER-04… »
- `fournisseurs/kommerling.md`, `roto.md`, `somfy.md`, `technal.md` : le premier titre est
  « Ce que le catalogue dit de X ». Une fiche fournisseur devrait s'ouvrir sur ce que X est et ce
  que PROFERM lui achète.
- `gammes/textural.md` : « Le catalogue la décrit comme unique sur le marché et entièrement
  pensée par PROFERM. » — c'est le discours commercial rapporté, sans valeur technique.

Pour l'humain, le texte est lourd ; pour le modèle, chaque valeur est noyée dans des tokens qui
disent qui l'a dite. Ces parenthèses ne servent pas l'application : le code ne résout que les
chemins `/dossier/page.md`, les identifiants `INC-/CTR-/VER-` et les motifs `raw/x.pdf, p. N`.

### 3.2 Le wiki raconte son histoire

103 passages parlent du wiki lui-même ou de la chronologie de l'ingestion :

- « Jusqu'ici le wiki ne le connaissait que par un logo » (`fournisseurs/profine.md`)
- « C'est la première coordonnée fournisseur que le wiki possède » (idem)
- « Le wiki disposait du DTA, pas du DTD » (`sources/dtd-6-16-2334.md`)
- « C'est la première fois que le wiki documente un système profine autre que le 76 Advanced »
  (`sources/dtd-6-16-2335.md`)
- « Neuf documents plus tard, les contradictions ne se répartissent pas au hasard »
  (`anomalies/contradictions-entre-sources.md`)
- « Cette brochure est le document qui a ruiné l'explication par la chronologie »
  (`gammes/lumine.md`)
- « Le wiki a longtemps porté une hypothèse… Cette hypothèse était fausse dans les deux sens…
  C'était l'objet de l'entrée VER-12, qui peut être close » (`fournisseurs/kommerling.md`)

Ce récit est vrai le jour où il est écrit et périmé le lendemain. Il appartient à `log.md`. La
page concept doit porter l'état des connaissances, pas le chemin qui y a mené.

### 3.3 Apartés, gras, coaching

88 apartés et 1 426 gras. Trois natures :

- **Opérationnels, à garder mais à reformuler en règle** : « Lire l'épaisseur d'isolant dans la
  colonne du dormant réellement utilisé » est utile ; « C'est l'erreur qui coûte une tapée
  refaite » est du commentaire.
- **Commerciaux, hors périmètre d'une source technique** : « C'est l'argument de vente le plus
  fort du LUMINE65 » (`garanties/`), « l'argument le plus concret à opposer à une comparaison de
  prix » (`fournisseurs/roto.md`), « à signaler au client qui demande un RAL particulier »,
  « face à un prescripteur », « la confusion se paie en argument commercial faux ».
- **Rhétoriques** : « Un avertissement de confort peu banal », « Point contre-intuitif à
  retenir », « La logique est inverse de l'intuition », « c'est le grand perdant des
  accessoires ».

Le gras est posé sur des valeurs, des références, des négations, des titres de colonnes : à 20
par page il ne hiérarchise plus. Pour le modèle c'est du bruit de tokens ; pour l'humain c'est un
texte qui crie.

### 3.4 Des hypothèses dans le corps

La consigne n° 4 du modèle interdit d'inventer un « pourquoi ». Le wiki lui en fournit :

- « C'est physiquement normal — un double vitrage asymétrique est souvent meilleur en acoustique
  qu'un triple symétrique » (`vitrages/performances-vitrages.md` **et** `gammes/hybride.md`,
  même phrase) : explication non sourcée, dupliquée.
- « C'est vraisemblablement ce que sécurise le système TiltSafe » (`quincaillerie/roto-nx.md`).
- « ces deux coloris sont donc surveillés spécifiquement, vraisemblablement pour leur tenue »
  (`fournisseurs/profine.md`).
- « Attribuer ce renforcement à la dilatation thermique est une déduction plausible, que le
  document ne fait pas » (`profiles/systeme-76-abaques-dimensionnels.md`) : bien signalée, mais
  pourquoi l'écrire dans le corps ?
- Table « Correspondance probable au catalogue » des poignées LUMÉAL
  (`gammes/coulissants-aluminium.md`) : une colonne entière de déduction, présentée en tableau
  comme une donnée.

Le modèle qui lit ces phrases reçoit l'autorisation implicite de les répéter comme des faits.

### 3.5 Avant / après : le registre visé

**Règle** : le sujet de la phrase est le produit, la pièce ou la règle ; la source est une
référence en fin de ligne ; le doute est une valeur retenue suivie d'un identifiant.

| Avant (`gammes/hybride.md`) | Après |
| --- | --- |
| « Le catalogue ne détaille pas les épaisseurs de profilé HYBRIDE70 et HYBRIDE76 séparément : il donne une plage de 70 à 76 mm pour la gamme (catalogue général, p. 10). Les noms des déclinaisons suggèrent 70 mm et 76 mm respectivement, mais le document ne l'écrit pas — à vérifier auprès du bureau d'études, entrée VER-04 du registre […]. » | Tableau `Déclinaison \| Épaisseur du profilé PVC (mm) \| Uw (W/m²K)` avec `HYBRIDE70 \| 70 \| -` et `HYBRIDE76 \| 76 \| 0,8`, puis une ligne : « Épaisseurs déduites du nom des déclinaisons, le catalogue ne donnant que la plage 70 à 76 mm (VER-04) [1 p. 10]. » |

| Avant (`fournisseurs/kommerling.md`, 16 lignes) | Après (2 lignes) |
| --- | --- |
| « Le wiki a longtemps porté une hypothèse séparant les références 76xxx… Cette hypothèse était fausse dans les deux sens… Conséquence pratique : tout s'approvisionne chez profine… C'était l'objet de l'entrée VER-12, qui peut être close. » | « Toutes les références du cahier technique PERFORM76, quelle que soit leur numérotation (76xxx, 2xxx, 4xxx, 6xxx, 8xxx), sont des profilés du système 76 Advanced et s'approvisionnent chez profine [3] (VER-12). » |

| Avant (`quincaillerie/roto-nx.md`) | Après |
| --- | --- |
| « L'argument est à manier avec précaution : la brochure affirme dépasser une classe qu'elle présente elle-même comme maximale, sans nommer la norme… Vérifier la norme de référence avant de reprendre cet argument face à un prescripteur… » (puis, plus bas sur la même page, la norme est donnée : DIN EN 13126/8) | « Traitement de surface Roto Sil Level 6, au-delà de la classe anticorrosion 5 de la DIN EN 13126/8, exempt de chrome VI [2 p. 19]. » |

Le troisième exemple montre le coût du récit : la mise en garde est restée sur la page après que
le manuel a répondu à la question.

---

## 4. Duplication et cohérence

### 4.1 Les mêmes données sur plusieurs pages

| Donnée | Pages qui la portent en entier | Page propriétaire proposée |
| --- | --- | --- |
| Grille de garanties (catalogue p. 35) | 13 pages : `garanties/`, `volets-roulants`, `collection-authentique`, `securite-portes-entree`, `coulissants-aluminium`, `perform-plus`, `hybride-plus`, `innoslide`, `roto`, `technal`, `serrure-motorisee`, `somfy`, 6 pages `sources/` | `garanties/garanties-par-composant.md` ; ailleurs une ligne et un lien |
| CTR-03, garantie structure 15 / 20 ans, avec la table des 9 sources | `contradictions-entre-sources` (2 fois), `garanties-par-composant` (table complète recopiée), `lumine`, `brochure-lumine65`, `brochure-perform-plus-hybride-plus`, `depliant-general-2023`, `depliant-hybride-2023`, `perform-plus`, `hybride-plus` | le registre, une fois |
| VER-20, FERCO / ROTO | `dta-6-16-2334`, `profine`, `roto`, `dtd-6-16-2334`, `informations-a-verifier` | le registre + une ligne sur la page DTA |
| Dimensions maximales de baie du DTA | `dta-6-16-2334` (table), `perform` (table), `dtd-6-16-2334` (liste), `dtd-6-16-2335` (table comparée) | `certifications/dta-6-16-2334.md` |
| Position et réglage du pivot bas (5 lignes) | `perform76-dormants` et `perform76-poignee-et-pivot` | poignée et pivot |
| Chevauchements INC-05 (table) | `incoherences-internes` et `perform76-poignee-et-pivot` | le registre |
| « Manques comblés » Roto NX + les trois galets | `roto-nx` et `roto-nx-ksr-montage` | `roto-nx` |
| Délignage de l'aile, fraisage 49 / 43 | `perform76-dormants` et `pose-perform76` | dormants |
| Qualicoat classe 2, accroche 15 / tenue 25 | `garanties`, `lumine`, `brochure-lumine65` | garanties |
| Classement AEV des coulissants, 3 sources | `coulissants-aluminium` **deux fois sur la même page** (sections « Classement AEV du LUMÉAL » et « Classement AEV : trois valeurs »), `labels-et-certifications`, `depliant-lumeal`, `brochure-lumine65` | labels et certifications |
| KÖMMERLING est une marque de profine | `kommerling`, `profine`, `dta-trocal-76-advanced`, `dta-6-16-2334` | `profine` |

### 4.2 Copies qui ont déjà dérivé

- **CTR-03 porte trois conclusions.** La ligne du registre dit « aucune — la valeur ne suit ni la
  date ni le produit ». La section « Ce que la garantie structure suit » du même fichier et la
  page `garanties/` disent « le produit, lui, explique tout ». `sources/brochure-lumine65.md`
  dit « la valeur ne suit donc ni la date, ni un type de produit identifiable ». Le modèle peut
  citer n'importe laquelle.
- **`fournisseurs/kommerling.md` contient deux sections « # Le PVC GREENLINE »** (vers les lignes
  34 et 95), au contenu presque identique.
- **`fournisseurs/kommerling.md`** : la `description` (« Fournisseur des profilés PVC
  GREENLINE… ») et l'entrée d'index contredisent le corps (« KÖMMERLING n'est pas un fournisseur
  indépendant, c'est une marque du groupe profine »).
- **`quincaillerie/roto-nx.md` et `fournisseurs/roto.md`** affirment que la Roto NX « n'équipe
  que les gammes PERFORM+ et HYBRIDE+ », alors que la procédure PROFERM PRO-PVC-OFOB-01
  (`procedures/transformation-of-en-ob-roto-nx.md`) s'applique à toute menuiserie PVC sur
  ROTO NX. Restriction à revoir ou à passer en VER-.
- **`log.md`** a deux sections « ## 2026-09-18 » : la seconde journée a été ouverte au lieu de
  compléter la première.
- « CTR-14 peut être close », « VER-12 peut être close » sont répétés sur quatre pages alors que
  le protocole dit que seul l'humain clôt : c'est du récit, pas une donnée.

---

## 5. Découpage

Le découpage par type et par famille est juste. Trois problèmes de frontière.

### 5.1 `sources/` mélange fiche d'identité et essai

Chaque page source empile : structure du PDF, « ce qu'il apporte », « ce qu'il contredit », « ce
qu'il confirme », « ce qui n'a pas été transcrit », « provenance du fichier » (archives
`export_doc_NNN.zip` aujourd'hui supprimées), chronologie du corpus. C'est 24 % du prompt, et
l'essentiel est soit méta, soit déjà dit ailleurs.

Gabarit proposé, fixe et court (≈ 2 000 caractères) :

1. Identité : titre exact, éditeur, date ou version, pages, validité s'il y en a une.
2. Carte des pages : tableau `pages PDF → page du wiki`, avec les décalages de pagination.
3. Non transcrit : liste des planches ou tableaux laissés au PDF, avec la page.
4. Fournisseur ou nature du document (commercial, atelier, réglementaire).
5. Citations.

Ce qui sort : « ce qu'il apporte / contredit / confirme » (c'est le travail des registres et des
pages concept), la provenance des zips (c'est `log.md`), les essais chronologiques.

### 5.2 Du contenu de concept classé comme « Document source »

`sources/dtd-6-16-2334.md` (10 700 car.) et `sources/dtd-6-16-2335.md` (9 200 car.) portent des
prescriptions de fabrication : garnitures de joint référence par référence, drainage et
équilibrage de pression, assemblage mécanique du meneau, seuil L\* < 82, verrouillages
complémentaires. Un technicien qui cherche « quel joint sur parclose en caramel » ne pense pas
« DTD ». Pages concept à créer, alimentées par ces sources :

- `profiles/systeme-76-joints-et-garnitures.md`
- `procedures/drainage-et-decompression-systeme-76.md` (cahier technique + manuel profine + DTD)
- `certifications/prescriptions-de-fabrication-76-advanced.md` (ou fusion dans la page DTA)

De même, la terminologie profine (repères A à R, sigles DHT / CCD / CCO / CCV) est enfouie dans
`sources/profine-directives-generales.md`.

### 5.3 Pages qui manquent alors que le corpus les permet

- **Un glossaire** : DHT, CCD, CCO, CCV, DEO, DFO, LFF, HFF, PV, OF, OB, IW, IG, CDR, RC, A\*E\*V,
  Uw, Ug, Up, Sw, TLw, DV, CV. C'est aussi la page qui aurait évité la confusion « OF = ouvrant
  fixe » citée dans `CLAUDE.md`, et qui porterait VER-35 (DV / CV).
- **`normes/`** : le dossier est prévu et vide alors que NF DTU 36.5, EN 1627 à 1630, EN 356,
  NF EN 14501, NF EN 12207, DIN EN 13126/8, EN 1670, NF P 20-302 sont cités sur une dizaine de
  pages. Une page par norme : ce qu'elle classe, ses classes, quel produit PROFERM porte quelle
  classe. C'est ainsi qu'un expert organise une base.
- **`gammes/perform70.md`** ou une section franche : la PERFORM70 est vendue et n'a aucune donnée
  technique (VER-03, VER-28). Une page qui dit clairement « rien n'est documenté au-delà de
  l'épaisseur et du Uw » vaut mieux qu'une absence.
- **Les fiches fournisseurs** sont des stubs : réorganiser en identité, ce que PROFERM achète
  (tableau gamme → système ou produit), documents, garanties spécifiques. TECHNAL est le trou le
  plus large : toute la gamme aluminium sans une cote de profilé.

### 5.4 La convention `# Cotes` est étirée

`# Cotes des garanties` (années), `# Cotes des classes de résistance EN 356`, `# Cotes du
monobloc` (densité, kPa) : « cote » désigne une dimension. Autoriser `# Caractéristiques` pour
les tableaux non dimensionnels, et réserver `# Cotes` aux millimètres.

---

## 6. Métadonnées

- **`stale_after` est mal défini.** Il est posé comme « date du document + un an », si bien que
  28 pages sur 73 sont périmées, dont les trois pages de cotes du système 76 (`2024-12-31`) créées
  le 18/09/2026, et la page stores (`2021-12-31`). Le lint devient inutilisable. Règle proposée :
  `stale_after` seulement pour une validité réelle (DTA `2028-07-31`, certification, tarif) ;
  sinon ne pas le poser. La date du document va dans `sources[].last_modified`, renseigné sur
  10 entrées sur 144 alors que c'est le champ qui fonde « le plus récent prime ».
- **`verified` : 0 page.** C'est exactement le champ de la seconde passe (§ 8).
- **`status: draft` a deux sens** : « dépend d'une anomalie ouverte » (règle du protocole) et
  « la source n'est qu'une brochure ». Garder le premier ; le second se lit dans `sources`.
- **`sources[].id`** est une clé maison, non prévue par le protocole, jamais utilisée par
  l'application. Inoffensive ; à documenter ou à retirer.
- **`generated.by`** vaut `process:claude-code` sur 73 pages ; aucune ne porte `human:elie`,
  alors que INC-01 a été corrigée à la main.

---

## 7. Budget de contexte

Le wiki occupe 183 000 tokens sur 256 000. Le plan de refonte estime le plafond à ~25 sources ;
il y en a 18 (30 PDF). La réécriture de forme est le premier levier, avant tout mécanisme.

| Poste | Aujourd'hui (car.) | Cible estimée | Gain estimé |
| --- | --- | --- | --- |
| `sources/` (21 pages → cartes fixes) | 128 000 | 45 000 | −80 000 |
| Doublons et récit sur les pages concept | 358 000 | 300 000 | −50 000 à −70 000 |
| `anomalies/` (tables dupliquées) | 49 000 | 40 000 | −10 000 |
| **Total** | **535 000** | **≈ 390 000** | **≈ −140 000 car., soit ≈ −48 000 tokens** |

Ce sont des estimations. L'ordre de grandeur suffit : c'est la place de 5 à 7 sources techniques
de 20 000 à 30 000 caractères, sans rien perdre de ce que le wiki sait.

---

## 8. Fidélité et seconde passe

### 8.1 Pourquoi elle est justifiée

- INC-01 (« 4/14/14/4 ») était une erreur du wiki, pas du catalogue ; trouvée par relecture
  humaine, corrigée sur trois pages le 18/09.
- Le cas `NT1947 | 140 mm (76180), 155 mm (76171)` a coûté une réponse fausse avant d'être
  détecté (règle ajoutée depuis au protocole).
- Le golden attribue à la rédaction du wiki les échecs sur la table des hauteurs de poignée et sur
  la parclose 76503.
- Aucune page ne porte `verified`.
- Le wiki liste lui-même ce qu'il n'a pas transcrit : tableau de vitrage du registre 2.3.2 (9
  planches), planches p. 19 et 21 du DTA (parcloses et renforts, « cotes lisibles »), nombre de
  paumelles par hauteur (barres), positions de vissage 2.4.1, matrice p. 141 du catalogue portes
  (29 × 10), cotes des quatre posters A0, 107 des 113 pages des directives générales, annexes des
  deux DTD, cotes de section des dormants p. 4.

### 8.2 Protocole proposé

1. **L'unité est le tableau**, pas le document : un tableau du wiki ↔ une page du PDF. Vérifier
   chaque cellule, l'unité de l'en-tête, la famille (tapée / appui / patte), le contexte de
   ligne, les exclusions.
2. **Sur l'image, jamais sur la couche texte** pour le cahier technique PERFORM76, les posters,
   les brochures « contenu visuel uniquement » et les planches multi-colonnes du manuel profine :
   la couche texte entrelace les colonnes (cas du 76373).
3. **Par une session qui n'a pas écrit la page**, et qui pose `verified: {by, at}` à la fin. Un
   écart de transcription se corrige dans le wiki avec une ligne de `log.md` ; une ambiguïté de
   la source devient une INC- ou une VER-.
4. **Ordre** : (a) `profiles/perform76-*`, `profiles/systeme-76-*`,
   `quincaillerie/perform76-poignee-et-pivot`, `quincaillerie/roto-nx-champs-application` — les
   cotes d'atelier ; (b) `procedures/` ; (c) `gammes/`, `vitrages/`, `portes/`, `equipements/` ;
   (d) `sources/` après leur réduction.
5. **Trancher les « non transcrits »** : transcrire (registre 2.3.2, DTA p. 19 et 21, paumelles)
   ou inscrire une fois pour toutes « hors wiki, lire le PDF p. N » dans la carte de la source.
   Aujourd'hui l'information est répétée sur trois ou quatre pages à chaque fois.

---

## 9. `wiki_llm/CLAUDE.md` : ce qui manque au protocole

Le protocole est bon sur les tableaux, les références et les anomalies. Il ne dit rien du
registre, de la duplication ni de la vérification, et c'est là que la dérive s'est produite.
Ajouts proposés, à écrire dans le fichier :

1. **Règle de plume.** « Écris ce qui est vrai du produit, jamais ce que dit le document. La
   source est une référence en fin de phrase ou sous le tableau, jamais le sujet. Interdits :
   « le catalogue indique / précise / annonce / ne donne pas », « selon la brochure », « le wiki
   ne savait pas », « jusqu'ici », « désormais », « c'est la première fois », « peut être
   close », toute chronologie d'ingestion — cela va dans `log.md`. »
2. **Citations.** Remplacer la parenthèse `(catalogue général, p. 10)` par un renvoi court
   `[1 p. 10]` vers la liste `# Citations`, qui garde `raw/x.pdf` pour que l'application ouvre le
   PDF. Garder `(schéma: raw/x.pdf, p. N)` tel quel sous chaque tableau. Une seule citation par
   paragraphe ou par tableau.
3. **Pas de « pourquoi » non sourcé, pas de « vraisemblablement » dans le corps.** Une hypothèse
   va dans VER- avec son identifiant ; le corps porte la valeur retenue et l'identifiant.
4. **Une donnée vit sur une seule page.** Ailleurs, une ligne et un lien. Une contradiction
   s'argumente une fois, dans le registre ; la page concept porte la valeur retenue et
   l'identifiant, en une phrase.
5. **Gras** réservé aux exclusions (« ne se monte pas », « à proscrire ») et aux identifiants
   d'anomalie. Pas de conseil commercial, pas d'adjectif d'appréciation ; un avertissement
   opérationnel se formule comme une règle.
6. **Gabarit `sources/`** fixe (§ 5.1), 2 000 caractères.
7. **Métadonnées** : `sources[].last_modified` obligatoire ; `stale_after` seulement pour une
   validité réelle ; `verified` posé par la relecture, jamais par l'auteur ; `status: draft`
   réservé aux pages qui dépendent d'une anomalie ouverte.
8. **Étape 8 de l'ingestion** : relecture cellule par cellule de chaque tableau de cotes sur le
   PDF rendu, avant d'écrire dans `log.md`. Étape 9 : recherche des doublons (une donnée déjà
   portée par une page existante y reste, la nouvelle page y renvoie).
9. **Conventions** : `# Cotes` pour les millimètres, `# Caractéristiques` pour le reste ; un
   glossaire et un dossier `normes/` comme cibles de lien.
10. **Lint étendu** : densité de parenthèses de citation par page, auto-références, tables
    dupliquées (empreinte du corps de table), gras par page, `verified` absent, `stale_after`
    posé sans validité réelle. Ces contrôles peuvent vivre dans le lint de `wiki_service.py` ou
    dans un script à part ; ils ne changent pas le tour de chat.
11. **Nettoyage** : le paragraphe sur les archives `export_doc_NNN.zip` est historique (archives
    supprimées) ; garder la mise en garde sur les résumés générés et la couche texte, retirer le
    reste.

---

## 10. Continuer à ingérer

### 10.1 La file d'attente

`a_faire/` : « directives generale.pdf » (18,6 Mo) est probablement le même document que
`raw/profine-directives-generales-2023-01.pdf`, dont 107 pages restent à dépouiller — à vérifier
par empreinte avant toute ingestion ; « systeme 70.pdf » (16,1 Mo) correspond sans doute aux
« Directives de fabrication plateforme 70 » de `docs/documentations/`, la source qui répondrait à
VER-03 et VER-28 ; « roto1.pdf » (57 Mo) est à identifier.

`docs/documentations/` : 70 fichiers, dont une vingtaine TECHNAL (SOLEAL PY / GY / FY, LUMEAL GA,
catalogues de conception et de fabrication, notices de pose, quatre DTA), huit Askey, trois
Soprofen (blocs baies, volets), trois sur le vitrage (CEKAL, condensation), le « Référentiel
technique gammes » PROFERM, un dépliant portes d'entrée, le dossier technique Perform76 du
26/08/2026, les instructions de pose Kömmerling 70 déjà condensées. **La gamme aluminium est le
plus grand trou du wiki** et c'est aussi le plus gros volume à ingérer.

### 10.2 Le plafond

À 18 sources pour 183 000 tokens, le plafond de ~25 sources est atteint avec la seule
documentation TECHNAL. Sans la réduction du § 7, ingérer « parfaitement » se heurte à la
fenêtre avant la fin de l'aluminium. La réécriture de forme n'est donc pas cosmétique : c'est ce
qui permet de continuer.

---

## 11. Plan d'action proposé

| Lot | Contenu | Durée estimée | Mesure |
| --- | --- | --- | --- |
| **A — Règles** | `CLAUDE.md` v2 (§ 9), gabarit `sources/`, lint étendu | ½ journée | lint vert sur les nouvelles règles |
| **B — Forme** | `sources/` en cartes ; dédoublonnage (garanties, CTR-03, VER-20, DTA, pivot, INC-05, Roto NX, AEV) ; réécriture du registre sur `gammes/`, `fournisseurs/`, `certifications/`, `garanties/` ; description de `kommerling.md` ; `log.md` fusionné | 2 à 3 jours | caractères ≤ 400 000 ; golden 62 ≥ 83,9 % |
| **C — Fidélité** | seconde passe cellule par cellule sur PDF rendu, ordre du § 8.2, `verified` posé ; « non transcrits » tranchés | 3 à 5 jours | 100 % des pages de cotes `verified` ; golden poignée et 76503 corrigés |
| **D — Ingestion** | identifier les trois PDF d'`a_faire/` ; plateforme 70 (VER-03, VER-28) ; puis TECHNAL dans la limite du budget | continu | budget affiché dans l'admin |

B et C peuvent avancer en parallèle : C commence sur `profiles/`, dont la forme change peu ; B
commence sur `sources/` et `gammes/`. C se termine toujours sur le texte final.

Ce que ce plan ne fait pas : aucun mécanisme dans l'application, aucun changement de découpage
de dossiers, aucune suppression de contenu factuel. Tout ce que le wiki sait aujourd'hui reste
dans le wiki ; il le dit une fois, au bon endroit, sans raconter comment il l'a appris.
