# LLM Wiki

A personal knowledge base maintained by Claude Code.
Based on Andrej Karpathy's LLM Wiki pattern, stored as an OKF bundle
(Open Knowledge Format v0.2 -- https://okf.md/spec/).

## Purpose

Proferm Multitechniques is a menuiserie fabricant. This wiki is the structured, interlinked
knowledge base covering its own production and the suppliers it buys from: catalogues, fiches
techniques, tarifs, certifications, normes, procédures atelier.

**The wiki replaces the PDFs.** Its target is the whole of `raw/`, page by page: 2 382 pages
across 32 documents, of which 1 944 carry drawn content and 309 have no usable text layer at all.
A question whose answer sits in a PDF and not in the wiki is a defect of the wiki, not a reason
to reopen the PDF.

It has two consumers, and they pull in the same direction:

- **The human**, who asks a question and gets an answer with its source, instead of reopening a
  450-page manual
- **A navigation tool**, which selects pages by their metadata and hands them to a model. The
  finished wiki is far larger than any context window, so a page is reached through its `type`,
  its `systeme` and its `gamme`, its `source_pages` -- never by putting the whole bundle in a
  prompt. Much of what reads as fussy formatting in this file -- one row per reference, units in
  the header, pages that stand alone -- is there so that a page still means something when it
  arrives alone

The LIA application still concatenates every page into its system prompt. That design is
superseded and the application is on hold: **nothing in this file is sized against a context
window any more.**

Claude does the reading and the writing. The human curates what goes into `raw/`, asks the
questions, and arbitrates the judgment calls: a contradiction between two suppliers, a new
`type`, a categorization that could go either way.

### Native Multimodal Ingestion (Zero Script, Zero Data Loss)

**PDFs are ingested natively and multimodally.**
- **No Python scripts, no PyMuPDF (`fitz`), no OCR tools**: Claude directly inspects and reads each PDF page natively (visual rendering of plates, section cuts, icon tables and drawings, combined with context).
- **Zero data loss**: every drawing, dimension, tolerance, profile section, catalog reference, abaque curve and workshop procedure is faithfully extracted into structured tables and assertive markdown.
- **Autonomous Wiki**: the wiki replaces the original PDFs entirely and stands alone as professional technical documentation.

## Folder structure

```
raw/          -- source documents (immutable -- never modify these)
wiki/         -- the OKF bundle: markdown pages maintained by Claude
wiki/index.md -- table of contents for the entire wiki (reserved, no frontmatter)
wiki/log.md   -- record of all operations, newest first (reserved, no frontmatter)
```

`wiki/` is the bundle root. Bundle-relative links start there: `/fournisseurs/nom.md` means
`wiki/fournisseurs/nom.md`. `raw/` sits outside the bundle -- reference source documents with
project-root-relative paths (`raw/catalogue-2026.pdf`).

Inside `wiki/`, group pages in one subdirectory per `type` -- lowercase, unaccented, plural:
`fournisseurs/`, `gammes/`, `profiles/`, `quincaillerie/`, `vitrages/`, `normes/`,
`certifications/`, `tarifs/`, `procedures/`, `machines/`, `anomalies/`, `equipements/`,
`portes/`, `garanties/`, `entretien/`, `commercial/`, `coloris/`, and `sources/` for
`Document source` pages. `reference/` holds the glossary. A new `type` means a new subdirectory,
so raise it with the user first.

Six of them are empty today -- `normes/`, `tarifs/`, `machines/`, `entretien/`, `commercial/`,
`coloris/` -- and not because the corpus holds nothing for them. The protocol that came before
this one had nowhere to put a DTU prescription, a maintenance instruction or a tarif, so it
dropped them.

## OKF conformance

Three rules make the bundle conformant. They are non-negotiable:

1. Every `.md` file in `wiki/` except `index.md` and `log.md` has parseable YAML frontmatter
2. Every frontmatter has a non-empty `type` field
3. `index.md` and `log.md` follow the reserved-file structures below

Everything else here is convention, not validation. Broken links are valid OKF -- they mark
knowledge not yet written.

---

## Two ways of writing, and when each applies

The wiki has two kinds of page and they are not written the same way. Mixing them is what turns
a knowledge base into a reading report.

**Assertive pages** -- everything except `anomalies/` and `sources/`. The subject of every
sentence is the product, the part, the dimension or the rule. The document that proves it is a
reference, never the subject. The reader wants the answer; the source is there so they can check
it, not so they can read about it.

- Write: « Le dormant 76185 porte une aile de 60 mm, délignable jusqu'à 40 mm sur le chantier
  [1 p. 11]. »
- Never: « Le cahier technique indique que le dormant 76185 porte une aile de 60 mm (cahier
  technique, p. 11), et précise que le délignage est à effectuer sur le chantier. »

**Evidential pages** -- `anomalies/` and `sources/`. Here the document *is* the subject, because
the page's job is to say what a document asserts, where, and in what words. Quote verbatim, name
the document, give the page. This is the one place where « le catalogue écrit … » is correct.

The test, when in doubt: **if the sentence would still be true after the document is reprinted in
a different layout, it belongs on an assertive page and must not name the document. If it would
become false, it belongs in a register.**

### The register: documentation, not reading notes

The wiki is the company's technical documentation and it reads like documentation. A prescription
is stated in full, a domaine d'emploi is a paragraph, a procedure is an ordered sequence of steps,
a matrix is a table. What it is not is a column of telegraphic bullets, each one a fragment of a
sentence: that register drops the conditions, the exceptions and the units, which is most of what
a normative text is made of.

**Take the source's own words rather than rewriting them.** A prescription, a condition, a
definition, a restriction is carried in the wording of the document, stripped only of the
narrative framing that named the document (« le présent document précise que … »). The wiki is
the substitute for the PDF: once a clause has been reformulated, the reader can no longer apply
it with confidence, and nobody can tell what the rewrite dropped. Where the source's sentence is
already a well-formed rule, copy it. Taking the source's words is not the same as making the
source the subject -- the first is required, the second is what *Two ways of writing* forbids.

- **Prose where the source argues.** A rule that only holds under a condition -- « à condition
  que le support soit … », « sauf en zone de sismicité 4 » -- is written as a sentence carrying
  that condition. Splitting it into two bullets separates the rule from what limits it
- **A table where the source tabulates**, and one table per thing tabulated
- **Numbered steps where the source prescribes an order**, numbered as the source numbers them
- **Paragraphs are allowed to be paragraphs.** Two or three sentences that hold together beat
  five fragments. The reader is a professional looking for an answer they can apply, not a
  summary of a document
- **A long normative passage is transcribed whole**, not distilled. Distilling is what the
  previous protocol called « a rule in the indicative », and it is how the conditions were lost
- **No reading-report register**: « on notera que », « il est intéressant de constater », « le
  présent document aborde ». The page states; it does not comment on what it states

### What never appears on an assertive page

| Banned | Why | Write instead |
| --- | --- | --- |
| « le catalogue indique / précise / annonce / décrit / ne donne pas », « selon la brochure », « d'après le DTA » | the source becomes the subject | the fact, with `[1 p. 10]` at the end |
| « le wiki », « cette page », « jusqu'ici », « désormais », « c'est la première fois », « peut être close » | the story of the ingestion, true for a day | `log.md` |
| « vraisemblablement », « on peut supposer », « cela suggère », « c'est physiquement normal » | an unsourced why, which the model will repeat as a fact | a `VER-` entry; the body carries the retained value and the identifier |
| « c'est l'argument de vente », « à signaler au client », « en clientèle », « face à un prescripteur » | coaching the reader, on a page whose job is to state facts | the claim itself, on a `commercial/` page (« La gamme est annoncée sans entretien autre qu'un nettoyage à l'eau savonneuse [1 p. 4]. »), or an operational rule: « Le laquage hors teintes standards est garanti 7 ans au lieu de 10. » |
| « erreur classique », « point contre-intuitif », « ce qu'on oublie », « à manier avec précaution » | commentary on the reader | the rule itself, in the indicative |

### The shape of an assertive page

**Answer first.** A page opens on what it asserts -- a table, a rule, a one-sentence definition --
never on a presentation of the document it came from. `# Ce que le catalogue dit de X` is not a
heading.

**Name the subject in full, in every section.** The reader lands here from a citation: write
« la PERFORM76 », not « cette gamme ».

**Say what is missing once.** A gap is a `-` in a table with one line under it, or a `VER-`
entry. Not both, and never a running commentary.

**Bold** is for exclusions (« ne se monte pas sur un dormant », « à proscrire ») and for anomaly
identifiers. Nothing else. A page carries a handful, not twenty.

---

## One datum, one page

A value lives on exactly one page. Everywhere else: one sentence and a link. Before writing a
figure, search the wiki for it; if it is already there, link instead of copying. Copies drift --
the garantie structure was once argued on nine pages and ended up carrying three different
conclusions.

| Datum | Owner page |
| --- | --- |
| Garanties par composant, laquage, bord de mer | `/garanties/garanties-par-composant.md` |
| Classements A\*E\*V, labels, classes d'effraction | `/certifications/labels-et-certifications.md` |
| Dimensions maximales de baie, domaine d'emploi, prescriptions du Groupe Spécialisé | `/certifications/dta-6-16-2334.md` |
| Cotes de débit, renforts, abaques du système 76 | the three `/profiles/systeme-76-*.md` |
| Profilés et renforts du système 70 | `/profiles/systeme-70-profiles-et-renforts.md` |
| Cotes et compatibilités d'une famille de profilés | that family's `/profiles/` page |
| Champs d'application et charges d'une ferrure | that hardware's `/quincaillerie/` page |
| Sigles et abréviations du métier | `/reference/glossaire.md` |
| Arbitrage d'une contradiction | the register, once |
| Chronologie d'une source, provenance d'un fichier | `log.md` |

### When two values are both true

A value that changed still describes the menuiseries already posed, which is what matters in SAV
and in remplacement à l'identique. Carry both, as a table with a context column -- never as a
story about two documents:

```markdown
| Période de fabrication | Épaisseur du profilé PVC (mm) | Uw (W/m²K) |
| --- | --- | --- |
| jusqu'en 2025 | 72 | 1,2 |
| à partir de janvier 2026 | 70 à 76 | 0,8 |
```

One row carries one context -- the rule for references, applied to time. The arbitration and the
doubt stay in the register (`CTR-06`, `CTR-07`).

---

## Everything gets transcribed

The wiki is the substitute for the PDFs. **Every page of every document in `raw/` ends up
represented in the wiki**, and a page of PDF has exactly two acceptable end states:

- **transcrite** -- its content is on one or more wiki pages, both named in the coverage register
- **illisible** -- named in the register with the technical reason and the rendering already
  attempted (« planche A0 rendue à 600 dpi par quarts, les cotes restent hors résolution »)

There is no third state. **« Volume trop important », « à traiter comme un chantier séparé »,
« aucun usage identifié à ce jour », « générique au bâtiment », « se consulte sur le document »,
« destiné au client final » are not end states.** Every one of those sentences is in the wiki
today, and every one of them hides between one and 240 pages of data the wiki was supposed to
carry.

Two dividing lines were tried before this one, and both failed:

- **textual versus graphic** -- 82 % of the corpus is drawn, so that line leaves four fifths of
  the data in the PDF. It is wrong on its own terms too: the annexe of the DTD 2335 was dismissed
  as « ce sont des dessins » when pages 11 to 13 are four data tables in plain text
- **data versus discourse** -- it sent prose, method, maintenance and legal text to the bin.
  Method is what an atelier works from, maintenance is what SAV answers with, and « discourse »
  is how registre 1.3.9, the 43 planches de méthode and the check-list de contrôle final were
  lost

**The nature of a content does not decide whether it is kept. It decides where it goes and in
what form.**

| Nature du contenu | Destination | Forme |
| --- | --- | --- |
| Tableau, matrice de compatibilité, nomenclature, liste de références | page de famille (`profiles/`, `quincaillerie/`, `portes/`, …) | table, une ligne par référence, valeurs verbatim |
| Cotes portées sur une planche | page de famille | table, cotes relevées sur la planche rendue |
| Courbe, abaque, diagramme d'application | page dédiée, dans le dossier de sa famille | échantillonnage tabulé -- voir *Lire un abaque* |
| Prescription numérotée, exigence, domaine d'emploi, classement réglementaire | `normes/`, `certifications/` | la numérotation de la source conservée, la condition et l'exception portées avec la règle, dans les termes du document |
| Méthode d'atelier, geste, planche de montage ou de pose | `procedures/` | étapes numérotées, plus la table des cotes de perçage et des gabarits |
| Usage, nettoyage, entretien, condensation | `entretien/` | consignes à l'indicatif, par élément |
| Descriptif produit, argumentaire, ce que la marque annonce | `commercial/` | ce que le document annonce du produit, sans le confondre avec une valeur mesurée |
| Garantie, condition, mention légale, réserve | `garanties/` | intégral, une ligne par composant et par durée |
| Tarif, code article, conditionnement, unité de vente | `tarifs/` | table |
| Coloris, finition, nuancier, plaxage | `coloris/` | table, le code de la source en première colonne |
| Machine, outillage, réglage d'atelier | `machines/` | table ou procédure selon le contenu |
| Page de garde, sommaire, page blanche, bandeau répété, mentions d'édition | aucune page | état **sans contenu propre** au registre, page par page, après l'avoir ouverte |

The last row is the only one that produces no wiki page, and it is the one to watch. It is a
statement about a page that has been rendered and looked at, never about a section judged from
its title -- « l'annexe, ce sont des dessins » is a guess, and that guess buried four data tables.

**Normative documents are the densest and the least forgiving.** A DTU, a DTA, a DTD or an avis
technique is written to be applied clause by clause: the domaine d'emploi bounds what may be
posed, each prescription carries its condition and its exception, the abaques decide a dimension,
and the annexes carry the nomenclatures. None of that survives a summary. A prescription is
carried whole -- its number, its condition, its exception, its unit -- in the wording of the
document, and a clause that exists as an argued paragraph stays a paragraph.

**Volume is not a reason.** The Roto NX catalogue is 451 pages and carries one transcribed plate.
The fifteen tables of inertie Iz are 19 columns by 35 rows each. Both are simply long, and long
is a schedule, not a decision: the register carries them as `à faire` until they are done.

**Nothing is inferred, ever.** Exhaustivity raises the pressure to fill a cell by reasoning about
a drawing's proportions, or by carrying a value over from a neighbouring reference. An unreadable
value stays `-`, with one line in the body. A wiki that invents is worse than a wiki with holes,
because a hole is visible and an invention is not.

## Ingest workflow

The order matters. **The register is built before the transcription, from a pass over every page
of the document.** Writing the pages first and listing what was left afterwards is how the holes
were made: what nobody opened never made the list.

When the user adds a new source to `raw/` and asks you to ingest it:

1. **Parcourir le document page par page en vision multimodale.** Visualiser chaque page directement
   pour identifier son contenu exact. Aucune page n'est qualifiée depuis son titre, sa section ou sa seule couche texte.
2. **Create the `sources/` card with its coverage register**, one row per page or per contiguous
   range of pages of identical nature, every row in state `à faire`. The union of the rows covers
   page 1 to page N with no gap -- see *The `sources/` card*
3. Discuss with the user what the document holds and in what order to take it
4. **Work the register down**, family by family. Create or update the concept pages, one concept
   per file, each range going to the destination its nature calls for -- see *Everything gets
   transcribed*
5. **Search the wiki for every figure you are about to write.** If another page already carries
   it, link to that page instead of copying -- see *One datum, one page*
6. Connect pages with bundle-relative markdown links, and fill `source_pages` on every page you
   write
7. **Move each range to `transcrit` as you finish it**, naming its wiki page. A range left
   `à faire` is the normal state of unfinished work; a range quietly dropped is a defect
8. Update `wiki/index.md` with the new pages and their `description`
9. Add an entry to `wiki/log.md` under today's date, at the top of the file
10. **Re-read each `# Cotes` table you just wrote against the rendered plate**, cell by cell. An
    ingestion that has not been re-read is not finished

A 400-page manual touches dozens of wiki pages, and that is the point. **An ingestion is finished
when its register carries no `à faire`**, not when the main ideas are covered. Reporting a
document as ingested while its register still shows `à faire` rows is the one thing this workflow
exists to prevent.

**Beware of generated summaries.** Some archives ship two kinds of chunk, and only one is a
source. Chunks that reproduce the document -- its sentences, its reference lists, its table
cells -- are the document. Chunks of smooth explanatory prose that no PDF page contains are
*generated summaries*, and they have been wrong twice: they called the coulissants 2415 and 2416
« dormants renforcés » on a page whose own footnote calls them coulissants, and they glossed OF
as « Ouvrant Fixe » where the whole corpus, DTA included, uses OF for ouvrant à la française.
Never cite them, never quote them. The same caution applies to a PDF's extracted text layer: on
a multi-column plate it interleaves labels from neighbouring blocks, which is how the meneau
76373 came to carry the renfort references of the 76372 next to it.

## Reading a plate

A drawing does not have a text layer, and when it does, that layer lies about which column a
number belongs to. Both problems are solved the same way: **render the page and look at it.**

**This is the normal path, not the exception.** 1 944 of the 2 382 pages of `raw/` carry drawn
content and 309 have no usable text layer at all. A page holding a plate is never declared
transcribed on the strength of its text layer.

**La lecture est multimodale native, sans script ni outil externe.** L'agent ouvre et visualise
directement les pages de chaque PDF (rendu visuel haute définition de la planche entière, des schémas,
coupes et tableaux techniques, doublé de la couche textuelle alignée). Aucun script Python (PyMuPDF, fitz)
n'est requis : l'inspection visuelle est directe et native.

- **Inspection visuelle systématique** : chaque page technique est examinée visuellement pour repérer
  la structure, les tableaux, les cotes en coupe, les renvois et les avertissements.
- **Lecture de la légende d'abord.** Sur le catalogue ROTO NX, les en-têtes de tableaux sont des
  pictogrammes définis en p. 10-12 ; sur le cahier technique PERFORM76 l'épaisseur de vitrage est en bleu
  et la parclose en noir. La lecture visuelle native décode la légende avant de transcrire les cotes.
- **Une cote ou valeur illisible reste `-`**, avec mention dans le corps. Ne jamais déduire une
  dimension d'une proportion de dessin, et ne jamais faire confiance aveuglément à une couche texte brute
  sans contrôle visuel de la planche.
- **Une planche non exploitable va à l'état `illisible`** dans le registre de couverture avec mention
  du motif technique précis. C'est un constat mesuré, jamais une décision d'abandon.

### Lire un abaque

A curve is data, and an abaque is often what decides a dimension on a chantier. « Seules les
bornes et les règles sont exploitables » leaves that data in the PDF. Sample it:

1. **Read the axes**: quantity, unit, scale (linear or logarithmic), and the printed graduation
   step.
2. **Read the legend**: one curve per classement, per épaisseur, per ferrure. Each becomes a
   column of the table, or a context column with the reference repeated across rows.
3. **One row per graduation of the x axis**, carrying the value read where each curve crosses it.
   Sample at the graduations printed on the plate, never at a step you invented.
4. **State the sampling step and the reading precision** in the sentence above the table:
   « relevé tous les 100 mm de largeur, précision de lecture ±25 Pa ».
5. **A zone rather than a curve** -- « champ d'application non autorisé », « 2ᵉ compas
   nécessaire » -- becomes a column naming the zone, one row per graduation.
6. **A point that cannot be read stays `-`.** Never interpolate between two read points, and
   never extend a curve past its last graduation.

Documents that require plate reading: the cahier technique PERFORM76 (no text layer at all), the
A0 posters, the reference tables of the Roto NX catalogue (pictogram column headers), the cotes
de débit of the système 70, the abaques dimensionnels of the systèmes 70 and 76, the diagrammes
d'application of the Roto NX KSR, and the cotées plates of the DTA and the DTDs.

## Page format

One concept per file. Frontmatter first, then structured markdown:

```markdown
---
type: Fournisseur
title: Nom du fournisseur
description: Une phrase qui résume la page.
resource: https://site-du-fournisseur.fr
tags: [aluminium, profilé]
gamme: PERFORM
systeme: 76
fournisseur: KÖMMERLING
usage: atelier
famille: dormants
status: stable
sources:
  - resource: raw/catalogue-2026.pdf
    title: Catalogue général 2026
    last_modified: 2026-03-14
source_pages:
  - resource: raw/catalogue-2026.pdf
    pages: 42-47
generated:
  by: human:elie
  at: 2026-09-17T14:00:00Z
verified:
  by: process:claude-code
  at: 2026-09-18T10:00:00Z
---

# Heading

Headings, tables and numbered steps carry what the source tabulates or orders; paragraphs carry
what it argues, in the source's own wording. See *The register* above.

Link to related concepts with bundle-relative markdown links:
[Série XYZ](/profiles/serie-xyz.md).

# Citations

[1] [Catalogue général 2026](raw/catalogue-2026.pdf), p. 42
```

Field notes:

- `type` (required) -- short, descriptive, free-form. Reuse an existing value; inventing a new
  type is a decision worth raising with the user. Vocabulary in use: `Fournisseur`, `Gamme`,
  `Profilé`, `Quincaillerie`, `Vitrage`, `Équipement`, `Porte d'entrée`, `Garantie`,
  `Procédure`, `Certification`, `Anomalie`, `Référence`, `Document source`, and, opened by the
  exhaustive protocol: `Norme`, `Tarif`, `Machine`, `Entretien`, `Commercial`, `Coloris`
- `title` -- human-readable name; may be omitted when the filename says it
- `description` -- one sentence. This is the line that gets copied into `index.md`, and it must
  match it exactly
- `resource` -- canonical URI of the real-world asset (supplier site, product page). Omit for
  abstract concepts
- `sources` -- provenance for the claims in the body
- **`sources[].last_modified` is required**: the source document's own date, not the ingest date.
  It is what makes « the most recent document » decidable
- **`source_pages` is required on every page transcribed from a document** -- the PDF pages this
  wiki page carries, one entry per source, written as the register writes them (`42`, `42-47`,
  `42, 51, 60`). It is the reciprocal of the coverage register: the register says where a page of
  PDF went, `source_pages` says where a wiki page came from, and the two are checked against each
  other. It is also what answers « what does the wiki hold about pages 100 to 120 of this manual »
- `gamme`, `systeme`, `fournisseur`, `usage`, `famille` -- **the axes the human navigation is
  built from.** The wiki application computes its home page, one dashboard per gamme, per
  système, per fournisseur and per usage, and every breadcrumb from these fields alone: a page
  that omits them is invisible to the reader who browses by product, whatever its quality.
  Nothing of this navigation is ever written into a page body -- no hub page, no link list, no
  breadcrumb line: it would duplicate the frontmatter, drift from it, and a hub listing every
  SOLEAL FY title would outrank the real pages in the lexical search. Fill what applies, omit
  the rest; adding them afterwards means reopening every page, so they are written at creation.
  All of them accept a list (`gamme: [PERFORM+, HYBRIDE+]`).
  - `gamme` -- the PROFERM commercial offer the page serves, **spelled exactly as the `gamme`
    field of its `gammes/` page** (`PERFORM`, `PERFORM+`, `HYBRIDE`, `HYBRIDE+`, `TEXTURAL`,
    `INNOSLIDE`, `LUMINE`, `Frappe 65 Ouvrant Caché`, `Frappe 65 Ouvrant Visible`,
    `Coulissant 65 NV`). A PERFORM76 page is `gamme: PERFORM`, `systeme: 76`. A value no gamme
    page declares is reported by the lint: it would open a dashboard of its own
  - `systeme` -- the supplier's technical system, named as the supplier names it: `70` and `76`
    for the profine systems, `SOLEAL FY`, `SOLEAL GY`, `SOLEAL PY`, `LUMEAL GA` for TECHNAL,
    `Roto NX`, `Roto Patio Inowa`, `Roto Safe E` for ROTO, `Chrono One`, `Chrono PSE²`,
    `Mono VI`, `Bloc LX`, `TRADI` for SOPROFEN. **Never a profile depth** (`55`, `65`): the depth
    belongs to the gamme (LUMINE55) and to the tags. Omitted when the system is the gamme itself
    (ASKEY). A system-wide page (`systeme-76-cotes-de-debit`) has a `systeme` and no `gamme`:
    the gamme page declares its systems (`gammes/perform.md` → `systeme: [70, 76]`), and the
    application shows the system's pages in the gamme's dashboard as « commun au système »
  - `fournisseur` -- whose product the page describes, **spelled exactly as the `fournisseur`
    field of its `fournisseurs/` page** (`KÖMMERLING`, `TECHNAL`, `ASKEY`, `ROTO`, `SOPROFEN`,
    `SOMFY`). Every PVC profile page, profine directives included, is `KÖMMERLING`: that is the
    name PROFERM buys under. Omitted when the source names no maker
  - `usage` -- **who opens this page first**, from a closed list: `atelier` (débit, usinage,
    assemblage, références de commande), `pose` (chantier, fixation, étanchéité, câblage),
    `chiffrage` (choisir, prescrire, deviser : gammes, performances, coloris, garanties,
    classement AEV), `sav` (réglage, entretien, transformation, réparation). Two values at most;
    a page useful to everyone (glossaire, registres, fournisseurs, `sources/`) has none
  - `famille` -- the family of components within a system (`parcloses`, `cotes-de-debit`). Never
    a system name: `Roto NX` is a `systeme`
- **`stale_after` -- only for a real expiry**: a DTA's validity, a certification, a tarif, an
  agrément. Never « the document's date plus a year », which once marked 28 pages out of 73 as
  stale and made the lint worthless. No expiry, no field
- **`verified` -- set by the reverse reading, never by the author of the page.** `by` is
  `human:<id>` or `process:<id>`, `at` is an ISO 8601 timestamp
- `generated` -- who wrote the page, same shape
- **`status: draft` -- only when the page depends on an open anomaly.** « The source is only a
  brochure » is not a draft; that reads from `sources`
- Custom keys are allowed (`owner`, `chantier`, ...) and must be preserved when editing a page

---

## Page skeletons

One shape per `type`, so a reader knows where to look and a session does not reinvent the
headings. A section with nothing to say is dropped, not filled with « le document ne dit rien ».

**A content that fits no heading gets a heading of its own**, at the end of the page. The
skeleton is a floor, not a ceiling: dropping a table because the page's `type` has no section for
it is precisely the failure this protocol exists to prevent.

| `type` | Headings, in order |
| --- | --- |
| `Profilé` | one-sentence definition · `# Cotes` · `# Compatibilités` · `# Ce que la source ne donne pas` · `# Citations` · `# Voir aussi` |
| `Quincaillerie` | definition · `# Caractéristiques` or `# Cotes` · `# Champs d'application` · `# Compatibilités` · `# Ce que la source ne donne pas` · citations |
| `Gamme` | definition and déclinaisons · `# Caractéristiques` · `# Performances` · `# Dimensions limites` · `# Coloris` · `# Ce qui n'est pas réalisable` · citations |
| `Procédure` | `# Ce que fait cette procédure` · `# Conditions et interdictions` · `# Étapes` · `# Cotes` · `# Ce que le document ne dit pas` · citations |
| `Fournisseur` | who they are · `# Ce que PROFERM lui achète` (table gamme → système) · `# Documents` · `# Garanties propres` · citations |
| `Certification`, `Norme` | `# Domaine d'emploi` · `# Prescriptions` (one entry per numbered clause, its condition and its exception kept with it) · `# Classes` · `# Abaques` · `# Ce que PROFERM revendique` · citations |
| `Entretien` | `# Ce qui est concerné` · `# Consignes` (table élément → produit → à éviter) · `# Fréquence` · `# Ce qui annule la garantie` · citations |
| `Commercial` | `# Ce qui est annoncé` · `# Restrictions annoncées` · `# Ce que la source ne chiffre pas` · citations |
| `Tarif` | `# Tarifs` · `# Conditions et remises` · `# Validité` · citations |
| `Coloris` | one table: code de la source → nom → support → disponibilité → restriction |
| `Machine` | definition · `# Caractéristiques` · `# Réglages` · `# Consommables` · `# Maintenance` · citations |
| `Vitrage`, `Équipement`, `Porte d'entrée`, `Garantie` | definition · `# Cotes` or `# Caractéristiques` · `# Compatibilités` or `# Restrictions` · citations |
| `Référence` | the glossary: one table, sigle → sens → où il s'emploie |
| `Document source` | the card below |
| `Anomalie` | the three registers, unchanged |

`# Cotes` is for a dimension -- millimetres, kilograms, cm⁴. Use `# Caractéristiques` for a table
that is not dimensional: durées de garantie, classes, coloris.

### The `sources/` card

A source page is the **map and the work queue of a PDF**. It has no size limit: a 451-page
catalogue has a long register, and the length of that register is what makes its coverage
provable.

1. **Identité** -- exact title, editor, edition or version, page count, validity if any, and its
   nature: commercial, atelier or réglementaire. The nature is what the « product beats general »
   rule reads.
2. **Registre de couverture** -- the heart of the card, and the reason the card exists. One row
   per page, or per contiguous range of pages of identical nature:

```markdown
# Registre de couverture

Numérotation du PDF ; la page imprimée porte un décalage de -2.

| Pages PDF | Contenu | État | Page du wiki |
| --- | --- | --- | --- |
| 1-3 | page de garde, sommaire, mentions d'édition | sans contenu propre | - |
| 4-10 | domaine d'emploi et prescriptions du Groupe Spécialisé | transcrit | [DTA 6/16-2334](/certifications/dta-6-16-2334.md) |
| 11 | tableau 1, compositions vinyliques, 18 références certifiées | à faire | - |
| 21-23 | nomenclature illustrée, ~100 accessoires d'embout et de seuil | en cours | [Tapées et isolation](/profiles/perform76-tapees-et-isolation.md) |
| 24-46 | planches de méthode : assemblage, drainage, capotage, poses | à faire | - |
| 47 | planche A0 réduite, rendue à 600 dpi par quarts, cotes hors résolution | illisible | - |
```

   **The union of the ranges covers page 1 to page N exactly, with no gap and no overlap.** That
   is what turns « the document is fully transcribed » into a checkable statement instead of an
   impression. Five states, and there is no sixth: `transcrit`, `en cours`, `à faire`,
   `illisible`, `sans contenu propre`. An `illisible` row says in its `Contenu` column what
   rendering was already attempted.
3. **Décalage de pagination** -- stated once, above the register, when the printed numbering
   differs from the PDF's.
4. `# Citations`, `# Voir aussi`.

What leaves the card: « ce qu'il apporte / contredit / confirme » (the registers' and the concept
pages' work), the provenance of the export archives (that is `log.md`), and every comparison of
editions (a register entry, or a context column on the concept page).

The old « Non transcrit » rubric is gone, and so is the 2 000-character target that came with it.
A line of that rubric naming pages nobody opened is a row in state `à faire` -- work that remains,
not a decision that was taken.

## Schémas et cotes

A technical drawing cannot be stored as markdown. Extract what it asserts into a table, then
point back at the drawing. Never redraw a schéma as ASCII art, and never dissolve dimensions
into prose -- both are unretrievable.

A page is served on its own -- handed over by a navigation tool, or landed on from a citation --
so every table has to stand on its own:

- **Markdown by default, HTML when markdown cannot express the table.** A table with merged
  cells, a two-level header or a grouped column band -- the fifteen tables of inertie Iz, the
  abaques grouped by classement au vent, the vitrage matrices of the DTD -- is written in HTML
  with `colspan` and `rowspan` rather than flattened until it loses its structure. Markdown stays
  the default because it is lighter and diffs cleanly; the test for HTML is whether flattening
  would drop a header level or pack two contexts into one cell
- **One row per reference.** The first column is always the reference itself, so a row still
  identifies what it describes once separated from its header
- **One column per dimension**, named with the drawing's own label plus the trade term when it is
  known: `Cote A - largeur hors tout (mm)`
- **Units in the header, never in the cells.** Cells hold bare numbers so they stay comparable
- **No merged cells or two-level headers in a markdown table** -- markdown cannot express them.
  Either split into two tables, or write that one table in HTML. What is forbidden is faking a
  two-level header by repeating column names, and dropping the outer level altogether
- **Keep a table under ~30 rows, and split along one of the source's own axes** -- gamme, usage,
  classement au vent, marque. Never split arbitrarily, and never shorten a source table to fit
  the cap: fifteen tables of 19 columns by 35 rows become fifteen pages, one per classement, not
  two sentences about the bounds
- **Introduce every table with one sentence** naming the reference family, the source and the
  unit system. That sentence is the context the page carries with it when it is served alone
- **Locate the drawing right after the table**: `(schéma: raw/catalogue-2026.pdf, p. 42)`, so a
  human can check the extraction against the original
- **An unreadable or ambiguous dimension is left as `-`** and flagged in the body. Never infer a
  cote from the drawing's proportions

```markdown
# Cotes

Profilés dormants de la gamme Série 70, cotes en mm, relevées sur le catalogue général 2026.

| Référence | Cote A - largeur hors tout (mm) | Cote B - profondeur (mm) | Cote C - feuillure (mm) | Poids (kg/m) |
| --- | --- | --- | --- | --- |
| DOR-70-01 | 70 | 58 | 24 | 1,42 |
| DOR-70-02 | 70 | 68 | 24 | 1,61 |
| DOR-70-03 | 70 | 78 | - | 1,78 |

(schéma: raw/catalogue-2026.pdf, p. 42)

La feuillure de la référence DOR-70-03 n'est pas cotée sur le schéma -- à vérifier auprès du
fournisseur.
```

## Références

Sources like a cahier technique name dozens of references -- a dormant, un ouvrant, un
battement, une parclose, un meneau, une tapée, un appui, une patte de pose, un renfort. Each
reference is documented individually, but **never in a file of its own**: one page per family,
one row per reference. A hundred one-line files is not a wiki, it is a heap.

- **One page per family**, named after the family and the gamme it belongs to:
  `profiles/perform76-parcloses.md`, `profiles/perform76-dormants.md`
- **One row per reference** in that page's `# Cotes` table, the reference in the first column,
  written as the source writes it (`Parclose 76507`, `Seuil A076`)
- **A `# Compatibilités` table** saying what each reference mounts with, and what it excludes.
  The exclusions carry more weight than the inclusions -- « montage uniquement compatible avec
  les ouvrants » is the line that prevents a bad order
- **When the same reference behaves differently depending on context**, that is the table worth
  building: the same tapée gives a different épaisseur d'isolation on each dormant, and that
  matrix is the whole value of the page
- **One row carries one context.** Never pack two contexts into a cell -- a row reading
  `NT1947 | 140 mm (76180), 155 mm (76171)` is correct to a human who reads the parentheses and
  ambiguous to everything else: a reader looking for "140" and "76171" lands on that single line
  and pairs them. Give the context its own column and repeat the reference across rows. This
  cost one wrong answer in a chatbot evaluation before it was found
- **Never infer a missing dimension.** Absent is `-`, plus a line in the body stating that the
  source does not give it
- Split a family that exceeds ~30 rows by usage or by gamme, not arbitrarily -- parcloses
  d'ouvrant and parcloses de dormant are two tables, not one of 27 rows

## Anomalies

Sources contradict each other, contradict themselves, and contain typos. **Never silently fix a
source, and never pick a winner on your own.** Record what the document says, say what it
probably should say, and let the human arbitrate — that is what `anomalies/` is for.

Three pages, grouped by what the human has to do about them:

| Page | Contenu | Préfixe d'identifiant |
| --- | --- | --- |
| `anomalies/incoherences-internes.md` | un document se contredit lui-même, ou contient une coquille manifeste | `INC-` |
| `anomalies/contradictions-entre-sources.md` | deux documents affirment des choses différentes | `CTR-` |
| `anomalies/informations-a-verifier.md` | une affirmation périmée, invérifiable ou dont le périmètre est douteux | `VER-` |

Rules for filing one:

- **Every anomaly gets a stable identifier** (`INC-01`, `CTR-03`) and keeps it for life. The
  human works from this list, so a renumbering destroys their tracking
- **One row per anomaly**, with the source, the page, what the document says *verbatim*, the
  probable correction, and the operational impact. « Coquille probable » is not enough — say
  what breaks if nobody corrects it
- **The register argues once.** The comparison of editions, the table of the nine sources, the
  reasoning that eliminates a hypothesis: all of it lives in the register entry and nowhere else.
  The concept page carries the retained value and the identifier, in one sentence
- **The anomaly is recorded twice**: in `anomalies/`, and inline on the page where it matters.
  A reader of the parclose page must see the problem without knowing `anomalies/` exists
- **Never mark an anomaly resolved on your own.** Only the human closes one, after checking with
  the service technique or the fournisseur
- A page carrying an unresolved contradiction is `status: draft`

## Reserved files

`wiki/index.md` -- no frontmatter. Concepts grouped under section headings, one line each, the
description copied from the concept's own `description` field:

```markdown
# Fournisseurs

* [Nom du fournisseur](/fournisseurs/nom.md) - Une phrase qui résume la page.

# Normes

* [NF DTU 36.5](/normes/nf-dtu-36-5.md) - Mise en œuvre des fenêtres et portes extérieures.
```

`wiki/log.md` -- no frontmatter. ISO 8601 date headings, most recent first:

```markdown
# Update Log

## 2026-09-17

* **Create**: [Nom du fournisseur](/fournisseurs/nom.md) from `raw/catalogue-2026.pdf`.
* **Update**: [NF DTU 36.5](/normes/nf-dtu-36-5.md) -- added section on pose en applique.
```

Never rewrite past entries. Add today's section at the top, and **only one section per date**.
`log.md` is where the story of the corpus lives: what arrived when, which hypothesis fell, which
file was renamed. None of that belongs on a concept page.

## Citation rules

Traceability does not change: every claim is still backed, and the audit trail is the `sources`
frontmatter, the `# Citations` list, the `(schéma: …)` locator under each table, and the
registers. What changes is **where the reference sits in the sentence**.

- **Under a table**: `(schéma: raw/catalogue-2026.pdf, p. 42)`, unchanged. The application turns
  it into a link that opens the PDF at that page
- **In prose**: a bracket at the end of the sentence or of the paragraph -- `[1 p. 10]` --
  pointing at the numbered `# Citations` list. At most one per paragraph, never inside a
  sentence, never as its subject. A section whose sentences all come from the same page carries
  one bracket, at the end
- **`# Citations`** keeps the full reference with its `raw/….pdf` path: that is what makes the
  bracket resolvable by a human and clickable in the reader
- **A product document beats a general one on technical values.** Established on nine PROFERM
  sources: whenever a catalogue or a dépliant général disagrees with the brochure of the product
  itself, the general document is the one at fault -- it rounds, it generalises one range's
  figure to all, and it skips the test conditions. For a classement AEV, a Uw, an acoustic
  figure, a dimension limit or a component warranty, cite the product document. The general
  catalogue says what exists, not what it measures. This does **not** extend to company-wide
  statements such as the structural warranty, where the general document is the reference
- **This rule is applied when the page is written**: the page carries the retained value, the
  register carries the arbitration. If two sources disagree, the page gives the retained value
  and the identifier in one sentence, and does not replay the comparison
- Never resolve a contradiction on the edition date alone, and never build a rule on two data
  points -- both were tried on the warranty question and both were wrong

## Reverse reading: verifying a page against its source

The wiki was written from the PDF to the page. It is verified the other way: from the page to
the PDF. The pass has two outputs at once -- a verdict on what is written, and what was missing.

1. **Take one table, not one document.** The unit is a `# Cotes` or `# Compatibilités` table, or
   a rule stated in the body
2. **Open the PDF at the page its locator names**, via native multimodal direct reading
3. **Check every cell**: the value, the unit against the header, the family (une tapée n'est pas
   un appui n'est pas une patte de pose), the context of the row, and the exclusions
4. **Read what surrounds the table on the plate.** This is where the pass earns its keep: a
   dimension, a note, a symbol or a footnote the page does not carry is added to the page. If it
   belongs to another family, the range that holds it goes back to `à faire` in the coverage
   register, named for what is actually on it. Nothing found by a reverse reading is filed as
   « laissé dans le PDF »
5. **Three outcomes, three actions**: the page is right → nothing; the page is wrong → correct
   it, and record in `log.md` the value before and after; the document is unclear or contradicts
   itself → a new `INC-` or `VER-`. Never silently fix a source, never close an entry
6. **Set `verified`** -- `by` and `at` -- only once every table on the page has been checked.
   Record in `log.md` which plates were read
7. **Do not verify a page you wrote in the same session, and read the PDF before re-reading the
   page.** Reading the page first makes the eye confirm instead of check

Order: the cotes d'atelier first (`profiles/`, `quincaillerie/`), then `procedures/`, then the
gammes and the rest, `sources/` last.

## Question answering

When the user asks a question:

1. Read `wiki/index.md` first to find relevant pages
2. Read those pages and synthesize an answer
3. Cite specific wiki pages in your response
4. If the answer is not in the wiki, say so clearly
5. If the answer is valuable, offer to save it as a new wiki page

Good answers should be filed back into the wiki so they compound over time.

## Lint

When the user asks you to lint or audit the wiki:

**Conformance**

- Frontmatter parses, `type` is non-empty, reserved files match their structure
- `index.md` lists every page, with descriptions matching their `description` field
- No page is written in a language other than French
- Broken bundle links, orphan pages, concepts mentioned without a page of their own

**Metadata**

- `sources[].last_modified` missing
- **`source_pages` missing on a page transcribed from a document**, or naming pages the coverage
  register attributes to another wiki page, or pages of a document that has no `sources/` card
- `stale_after` set without a real expiry, or past its date
- `verified` missing on a page that carries a `# Cotes` table
- `status: draft` on a page that depends on no open anomaly

**Register (the rules of this file)**

- A `(nom du document, p. N)` inside a sentence on an assertive page
- Any banned phrase from the table in *Two ways of writing*
- A `# Ce que <document> dit …` heading outside `sources/` and `anomalies/`
- A figure carried by two pages with the same unit -- one of them should be a link
- More than ~15 bold spans on a page
- A heading set that does not match the skeleton of the page's `type`
- A page written in the reading-report register: fragments where the source argues, a rule cut
  away from the condition that limits it, « on notera que », « le présent document aborde »

**Tables**

- One row per reference, units in the header, an introductory sentence, a `(schéma: ...)`
  locator, no cell packing two contexts
- A markdown table faking a two-level header by repeating column names, or having dropped one --
  that table belongs in HTML
- A source table shortened to fit under ~30 rows instead of being split along one of its own axes

**Coverage -- the primary measure. It runs against `raw/`, not against the wiki alone**

- **Every PDF in `raw/` has a `sources/` card carrying a coverage register**, and that register's
  ranges cover page 1 to page N with no gap and no overlap. A gap is the measure failing, not a
  document being short
- **Count the rows by state, per document, and report the totals.** `à faire` is the size of what
  is left; it is the progress measure of the whole wiki and it belongs at the top of the report
- A row in state `sans contenu propre` covering more than three consecutive pages, or naming a
  section (« l'annexe », « les planches ») instead of what is on the pages
- A row in state `illisible` that does not say which rendering was attempted
- For each page of each PDF: the references it names that appear nowhere in the wiki. A
  normative document with dozens of them has been skimmed, not ingested. The measure over-states
  coverage, because a dimension present elsewhere in the wiki counts as seen, so treat a high
  score as a floor and the absent references as the truth
- A page whose text layer is under ~250 characters while it carries hundreds of vector paths is
  a mute plate: its data exists only as an image, and no amount of text extraction will find it.
  `raw/` holds 309 of them

Report findings as a numbered list with suggested fixes.

## Rules

- Never modify anything in the `raw/` folder, except to rename a file after its actual content at
  ingestion time
- Always update `wiki/index.md` and `wiki/log.md` after changes
- Keep file names lowercase with hyphens (e.g. `profil-alu-70mm.md`). The path minus `.md` is the
  concept's permanent ID -- renaming breaks links, so choose the name carefully
- Link with markdown links, never `[[wikilinks]]`: OKF uses the file path as concept identity
- **Every piece of markdown you generate is in French, without exception**: page bodies, page
  titles, `description` values, `index.md` entries, `log.md` entries. Keep trade terms as the
  sources spell them (dormant, ouvrant, Uw, Sw, pose en applique). What stays in English: this
  file, the OKF frontmatter *keys* (`type`, `description`, `stale_after`, ...) and the spec's
  own `status` vocabulary (`draft` | `stable` | `deprecated`) -- translating those would break
  conformance. Frontmatter *values* are French. File and folder names stay unaccented
- Every page must stand alone, because a reader lands on it from a citation. Name the subject in
  full instead of writing « ce profilé » or « cette gamme », and repeat the reference in each
  section rather than leaning on the page title
- Write the documentation of a professional, in the source's own terms: full sentences where
  the source argues, tables where it tabulates, and no reading-report commentary
- **Never report a document as ingested while its coverage register still carries an `à faire`.**
  Say how many ranges remain
- When uncertain about how to categorize something -- especially a new `type` -- ask the user
