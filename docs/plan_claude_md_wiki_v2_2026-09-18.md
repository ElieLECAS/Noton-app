# Plan — `wiki_llm/CLAUDE.md` v2 : écrire une documentation, pas un commentaire de documents

Date : 2026-09-18 · Suite de `docs/audit_wiki_redaction_2026-09-18.md` · **Proposition, rien n'est
modifié.**

Objet : réécrire le protocole d'écriture du wiki pour que les sessions d'ingestion produisent une
documentation technique, puis enchaîner sur la relecture inverse — de la page markdown vers le
PDF — qui vérifie et enrichit en même temps.

---

## 1. Pourquoi le protocole actuel produit du commentaire

Le fichier actuel n'a aucune règle de plume. Trois de ses règles poussent même activement vers le
commentaire, et elles ont fait leur effet :

| Règle actuelle | Effet observé |
| --- | --- |
| « Every factual claim references its source: inline `(source: fichier.pdf, p. 12)` » | 403 parenthèses dans la prose, et la source promue sujet de la phrase |
| « Create a summary page for the document itself » sans gabarit | 21 pages `sources/` devenues des essais, 24 % du prompt |
| « Record what the document says » (règle des anomalies) | mode probatoire légitime dans les registres, contaminé partout ailleurs |

À quoi s'ajoute ce que le protocole ne dit nulle part : rien sur la duplication (grille de
garanties recopiée sur 13 pages), rien sur la vérification (`verified` vide sur 73 pages), rien
sur les hypothèses dans le corps, rien sur la forme d'une page selon son `type`.

**Le correctif n'est pas une interdiction de plus, c'est une distinction qui manque** : le wiki
contient deux natures de page qui ne s'écrivent pas de la même façon, et le protocole les traite
pareil.

---

## 2. Le principe directeur : deux modes d'écriture

**Mode assertif** — toutes les pages sauf `anomalies/` et `sources/`. Le sujet de la phrase est le
produit, la pièce, la cote ou la règle. Le document est une référence en fin de ligne, jamais le
sujet. Le lecteur veut la réponse ; la source est là pour qu'il vérifie, pas pour qu'il la lise.

**Mode probatoire** — `anomalies/` et `sources/`. Là, le document *est* le sujet, parce que le
travail de la page est de dire ce qu'un document affirme, où, et dans quels mots. Verbatim,
document nommé, page donnée. C'est le seul endroit où « le catalogue écrit… » est juste.

Le test, en cas de doute : **si la phrase reste vraie après une réimpression du document dans une
autre mise en page, elle est assertive et ne doit pas nommer le document. Si elle devient fausse,
elle appartient à un registre.**

Cette distinction légitime le travail des registres, qui est excellent, et l'empêche de déborder.
Elle évite aussi la sur-correction : personne ne va aller retirer les verbatim de `INC-08`.

---

## 3. Ce que devient le fichier, section par section

Ordre cible. Les sections nouvelles sont marquées **N**, les réécrites **R**.

| # | Section | État |
| --- | --- | --- |
| 1 | Purpose, Folder structure, OKF conformance | inchangées, plus `normes/` et `glossaire` dans l'arborescence |
| 2 | **Two ways of writing** | **N** — le cœur, placé avant tout le reste |
| 3 | **One datum, one page** | **N** |
| 4 | Ingest workflow | **R** — deux étapes ajoutées |
| 5 | Page format, frontmatter | **R** — trois champs redéfinis |
| 6 | **Page skeletons** | **N** — un squelette par `type` |
| 7 | Schémas et cotes | inchangée, plus `# Caractéristiques` |
| 8 | Références | inchangée |
| 9 | Anomalies | **R** — « le registre argumente une fois » |
| 10 | Reserved files | **R** — `log.md` reçoit le récit |
| 11 | Citation rules | **R** — position grammaticale et format |
| 12 | **Reverse reading** | **N** — la seconde passe |
| 13 | Question answering | inchangée |
| 14 | Lint | **R** — huit contrôles ajoutés |
| 15 | Rules | **R** — allégée, les règles de style partent en § 2 |

Le fichier reste en anglais, avec les exemples en français, comme aujourd'hui.

---

## 4. Le texte des sections nouvelles

Rédigé pour être collé tel quel. C'est le livrable réel de ce plan.

### 4.1 Two ways of writing, and when each applies

> The wiki has two kinds of page and they are not written the same way. Mixing them is what turns
> a knowledge base into a reading report.
>
> **Assertive pages** — everything except `anomalies/` and `sources/`. The subject of every
> sentence is the product, the part, the dimension or the rule.
>
> - Write: « Le dormant 76185 porte une aile de 60 mm, délignable jusqu'à 40 mm sur le chantier
>   [1 p. 11]. »
> - Never: « Le cahier technique indique que le dormant 76185 porte une aile de 60 mm (cahier
>   technique, p. 11), et précise que le délignage est à effectuer sur le chantier. »
>
> **Evidential pages** — `anomalies/` and `sources/`. Here the document *is* the subject, because
> the page's job is to say what a document asserts, where, and in what words. Quote verbatim,
> name the document, give the page.
>
> The test, when in doubt: if the sentence would still be true after the document is reprinted in
> a different layout, it belongs on an assertive page and must not name the document. If it would
> become false, it belongs in a register.
>
> #### What never appears on an assertive page
>
> | Banned | Why | Write instead |
> | --- | --- | --- |
> | « le catalogue indique / précise / annonce / décrit / ne donne pas », « selon la brochure », « d'après le DTA » | the source becomes the subject | the fact, with `[1 p. 10]` at the end |
> | « le wiki », « cette page », « jusqu'ici », « désormais », « c'est la première fois », « peut être close » | the story of the ingestion, true for a day | `log.md` |
> | « vraisemblablement », « on peut supposer », « cela suggère », « c'est physiquement normal » | an unsourced why, which the model will repeat as a fact | a `VER-` entry; the body carries the retained value and the identifier |
> | « c'est l'argument de vente », « à signaler au client », « en clientèle », « face à un prescripteur » | commercial coaching | nothing, or an operational rule: « Le laquage hors teintes standards est garanti 7 ans au lieu de 10. » |
> | « erreur classique », « point contre-intuitif », « ce qu'on oublie », « à manier avec précaution » | commentary on the reader | the rule itself, in the indicative |
>
> #### The shape of an assertive page
>
> **Answer first.** A page opens on what it asserts — a table, a rule, a one-sentence definition —
> never on a presentation of the document it came from. `# Ce que le catalogue dit de X` is not a
> heading.
>
> **Name the subject in full, in every section.** The reader lands here from a citation: write
> « la PERFORM76 », not « cette gamme ».
>
> **Say what is missing once.** A gap is a `-` in a table with one line under it, or a `VER-`
> entry. Not both, and never a running commentary.
>
> **Bold** is for exclusions (« ne se monte pas sur un dormant », « à proscrire ») and for anomaly
> identifiers. Nothing else. A page carries a handful, not twenty.

### 4.2 One datum, one page

> A value lives on exactly one page. Everywhere else: one sentence and a link. Before writing a
> figure, search the wiki for it; if it is already there, link instead of copying. Copies drift —
> the garantie structure was argued on nine pages and ended up carrying three different
> conclusions.
>
> | Datum | Owner page |
> | --- | --- |
> | Garanties par composant, laquage, bord de mer | `/garanties/garanties-par-composant.md` |
> | Classements A\*E\*V, labels, classes d'effraction | `/certifications/labels-et-certifications.md` |
> | Dimensions maximales de baie, domaine d'emploi, prescriptions du Groupe Spécialisé | `/certifications/dta-6-16-2334.md` |
> | Cotes de débit, renforts, abaques du système 76 | the three `/profiles/systeme-76-*.md` |
> | Cotes et compatibilités d'une famille de profilés | that family's `/profiles/` page |
> | Champs d'application et charges d'une ferrure | that hardware's `/quincaillerie/` page |
> | Arbitrage d'une contradiction | the register, once |
> | Chronologie d'une source, provenance d'un fichier | `log.md` |
>
> #### When two values are both true
>
> A value that changed still describes the menuiseries already posed, which is what matters in SAV
> and in remplacement à l'identique. Carry both, as a table with a context column — never as a
> story about two documents:
>
> | Période de fabrication | Épaisseur du profilé PVC (mm) | Uw (W/m²K) |
> | --- | --- | --- |
> | jusqu'en 2025 | 72 | 1,2 |
> | à partir de janvier 2026 | 70 à 76 | 0,8 |
>
> One row carries one context — the rule for references, applied to time. The arbitration and the
> doubt stay in the register (`CTR-06`, `CTR-07`).

### 4.3 Citation rules (réécriture)

> Traceability does not change: every claim is still backed, and the audit trail is the `sources`
> frontmatter, the `# Citations` list, the `(schéma: …)` locator under each table, and the
> registers. What changes is **where the reference sits in the sentence**.
>
> - **Under a table**: `(schéma: raw/catalogue-2026.pdf, p. 42)`, unchanged. The application turns
>   it into a link that opens the PDF at that page.
> - **In prose**: a bracket at the end of the sentence or of the paragraph — `[1 p. 10]` — pointing
>   at the numbered `# Citations` list. At most one per paragraph, never inside a sentence, never
>   as its subject. A section whose sentences all come from the same page carries one bracket, at
>   the end.
> - **`# Citations`** keeps the full reference with its `raw/….pdf` path: that is what makes the
>   bracket resolvable by a human and clickable in the reader.
> - **A product document beats a general one on technical values** — unchanged, established on nine
>   PROFERM sources. This rule is now applied **when the page is written**: the page carries the
>   retained value, the register carries the arbitration.
> - If two sources disagree, the page carries the retained value and the identifier, in one
>   sentence. It does not replay the comparison.

### 4.4 Frontmatter : trois champs redéfinis

> - `sources[].last_modified` — **required**. The source document's own date. It is what makes
>   « the most recent document » decidable. Ten entries out of 144 carry it today.
> - `stale_after` — **only for a real expiry**: a DTA's validity, a certification, a tarif, an
>   agrément. Never « the document's date plus a year » — that marked 28 pages out of 73 as stale
>   and made the lint worthless. No expiry, no field.
> - `verified` — set by the reverse pass, never by the author of the page.
> - `status: draft` — only when the page depends on an open anomaly. « The source is only a
>   brochure » is not a draft; that reads from `sources`.

### 4.5 Page skeletons

> One shape per `type`, so a reader knows where to look and a session does not reinvent the
> headings. A section with nothing to say is dropped, not filled with « le document ne dit rien ».
>
> | `type` | Headings, in order |
> | --- | --- |
> | `Profilé` | one-sentence definition · `# Cotes` · `# Compatibilités` · `# Ce que la source ne donne pas` · `# Citations` · `# Voir aussi` |
> | `Quincaillerie` | definition · `# Caractéristiques` or `# Cotes` · `# Champs d'application` · `# Compatibilités` · `# Ce que la source ne donne pas` · citations |
> | `Gamme` | definition and déclinaisons · `# Caractéristiques` · `# Performances` · `# Dimensions limites` · `# Coloris` · `# Ce qui n'est pas réalisable` · citations |
> | `Procédure` | `# Ce que fait cette procédure` · `# Conditions et interdictions` · `# Étapes` · `# Cotes` · `# Ce que le document ne dit pas` · citations |
> | `Fournisseur` | who they are · `# Ce que PROFERM lui achète` (table gamme → système) · `# Documents` · `# Garanties propres` · citations |
> | `Certification`, `Norme` | what the text classifies · `# Classes` · `# Ce que PROFERM revendique` · `# Prescriptions` · citations |
> | `Document source` | the card in the next section |
> | `Anomalie` | the three registers, unchanged |
>
> `# Cotes` is for a dimension — millimetres, kilograms, cm⁴. Use `# Caractéristiques` for a table
> that is not dimensional: durées de garantie, classes, coloris.
>
> #### The `sources/` card
>
> A source page is a **map of a PDF**, not an essay about it. Target: under 2 000 characters.
>
> 1. **Identité** — exact title, editor, edition or version, pages, validity if any, and its
>    nature: commercial, atelier or réglementaire. The nature is what the « product beats general »
>    rule reads.
> 2. **Carte des pages** — a table `pages du PDF → page du wiki`, with the offset when the printed
>    numbering differs from the PDF's.
> 3. **Non transcrit** — the plates and tables deliberately left in the PDF, with their page and
>    the reason in half a line. **This list is the work queue of the reverse pass.**
> 4. `# Citations`, `# Voir aussi`.
>
> What leaves the card: « ce qu'il apporte / contredit / confirme » (the registers' and the concept
> pages' work), the provenance of the `export_doc_*.zip` archives (that is `log.md`), and every
> comparison of editions (a register entry, or a context column on the concept page).

### 4.6 Reverse reading: verifying a page against its source

> The wiki was written from the PDF to the page. It is verified the other way: from the page to
> the PDF. The pass has two outputs at once — a verdict on what is written, and what was missing.
>
> **Prerequisite.** Drawing-heavy documents have no usable text layer, and on a multi-column plate
> the extracted text interleaves neighbouring blocks — that is how the meneau 76373 came to carry
> the renfort references of the 76372. The cahier technique PERFORM76 extracts zero characters.
> **A page of cotes is verified on the rendered image, never on the text layer.**
>
> 1. **Take one table, not one document.** The unit is a `# Cotes` or `# Compatibilités` table, or
>    a rule stated in the body.
> 2. **Open the PDF at the page its locator names**, rendered as an image.
> 3. **Check every cell**: the value, the unit against the header, the family (une tapée n'est pas
>    un appui n'est pas une patte de pose), the context of the row, and the exclusions.
> 4. **Read what surrounds the table on the plate.** This is where the pass earns its keep: a
>    dimension, a note, a symbol or a footnote the page does not carry is either added to the page,
>    or listed under « Non transcrit » on the source card.
> 5. **Three outcomes, three actions**: the page is right → nothing; the page is wrong → correct
>    it, and record in `log.md` the value before and after; the document is unclear or contradicts
>    itself → a new `INC-` or `VER-`. Never silently fix a source, never close an entry.
> 6. **Set `verified`** — `by` and `at` — only once every table on the page has been checked.
>    Record in `log.md` which plates were read.
> 7. **Do not verify a page you wrote in the same session, and read the PDF before re-reading the
>    page.** Reading the page first makes the eye confirm instead of check.
>
> Order: the cotes d'atelier first (`profiles/`, `quincaillerie/perform76-poignee-et-pivot`,
> `quincaillerie/roto-nx-champs-application`), then `procedures/`, then the gammes and the rest,
> `sources/` last.

### 4.7 Ingest workflow : deux étapes ajoutées

Entre l'étape 5 (liens) et l'étape 6 (index) :

> 5 bis. **Search the wiki for every figure you are about to write.** If it is already carried by
> another page, link to it instead of copying — see *One datum, one page*.

Après l'étape 7 (log) :

> 8. **Re-read each `# Cotes` table you just wrote against the rendered plate**, cell by cell,
> before considering the ingestion done. An ingestion that has not been re-read is not finished.

### 4.8 Lint : les contrôles ajoutés

> - a `(nom du document, p. N)` inside a sentence on an assertive page
> - any banned phrase from the table in *Two ways of writing*
> - a `# Ce que <document> dit …` heading outside `sources/` and `anomalies/`
> - a figure carried by two pages with the same unit — one of them should be a link
> - a `sources/` page over 2 500 characters
> - `stale_after` set without a real expiry, `last_modified` missing, `verified` missing
> - more than ~15 bold spans on a page
> - a heading set that does not match the skeleton of the page's `type`

---

## 5. Ce qui ne change pas

À dire explicitement dans le fichier, pour qu'aucune session ne « corrige » ce qui marche :

- les tableaux de cotes et leurs sept règles (une ligne par référence, unité en en-tête, pas de
  cellule fusionnée, une ligne = un contexte, `-` pour l'illisible, phrase d'introduction,
  locator `(schéma: …)`) ;
- une page par famille de références, jamais un fichier par référence ;
- les trois registres, leurs identifiants à vie, la double inscription registre + page ;
- la conformité OKF, les fichiers réservés, le nommage, le français partout ;
- la règle « le document produit prime » ;
- l'interdiction de corriger une source et de clore une anomalie.

**La traçabilité ne diminue pas.** Elle change de place : elle sort de la grammaire des phrases et
reste entière dans `sources`, `# Citations`, les locators et les registres.

---

## 6. Dépendances et prérequis

**Un changement lié, hors du wiki.** Les consignes du modèle (`app/prompts/wiki_consignes.md`)
doivent recevoir une ligne : ne jamais reprendre un renvoi `[n p. N]` dans une réponse, ce sont des
renvois internes à une page ; la citation reste le chemin de la page. Une ligne, et elle invalide
la clé de cache une fois — coût connu et attendu.

**Le rendu d'image manque, et il bloque la seconde passe.** Aucune bibliothèque PDF n'est restée
dans `app/requirements.txt` après la suppression du retriever, et l'outil MCP de rendu est
indisponible sur ce poste. Or le cahier technique PERFORM76, qui porte la majorité des cotes,
n'a aucune couche texte. Prérequis : un utilitaire de rendu page → PNG dans l'image Docker
(PyMuPDF ou pypdfium2, dépendance Docker-only comme le reste), appelé par
`docker compose exec web`, et dont Claude Code lit le PNG. C'est un outil de développement, pas un
mécanisme du tour de chat.

---

## 7. Ordre d'application et mesure

| Lot | Contenu | Durée | Mesure |
| --- | --- | --- | --- |
| **0** | Utilitaire de rendu PDF → PNG dans Docker | ½ j | une planche du cahier technique lisible par Claude Code |
| **A** | `CLAUDE.md` v2 : § 4 collé, § 5 ajouté, sections réécrites ; une ligne aux consignes | ½ j | relecture par Elie |
| **B** | Une page pilote par mode, réécrite à la main pour servir de modèle : `profiles/perform76-dormants.md` (assertif), `sources/cahier-technique-perform76.md` (carte), `gammes/hybride.md` (deux valeurs vraies) | ½ j | golden 62 ≥ 83,9 % |
| **C** | Réécriture de forme du reste, par dossier | 2 à 3 j | ≤ 400 000 caractères ; golden ≥ 83,9 % |
| **D** | Relecture inverse selon § 4.6, dans l'ordre donné | 3 à 5 j | `verified` sur 100 % des pages de cotes ; échecs golden poignée et 76503 corrigés |

**B avant C, et B sur trois pages seulement.** Un gabarit se juge sur une page réécrite, pas sur un
protocole. Si le pilote ne te convient pas, on corrige le protocole avant d'avoir réécrit 70 pages.

C et D peuvent se croiser : D commence sur `profiles/`, dont la forme bouge peu, pendant que C
traite `sources/` et `gammes/`. Une page passe toujours par C avant de recevoir son `verified`,
sans quoi la vérification porte sur un texte qui va changer.

---

## 8. Le risque à surveiller

Le seul risque réel de cette réécriture est **de perdre une nuance en supprimant une phrase de
commentaire**. Exemple : « Le nez d'appui 4319 n'est pas compatible avec le 76768 : le cahier ne le
représente que sur les appuis 6136 et 6137 » — la justification par l'absence de représentation est
une information, pas du bavardage. Elle devient une ligne du tableau de compatibilités (`non`) plus,
si elle n'est qu'une déduction de lecture, une entrée `VER-`.

Règle de sécurité pour les lots C et D : **une phrase n'est supprimée que si son contenu est ailleurs
— dans un tableau, dans un registre, ou dans `log.md`.** Jamais parce qu'elle est mal écrite.
