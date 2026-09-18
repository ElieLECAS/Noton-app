# LLM Wiki

A personal knowledge base maintained by Claude Code.
Based on Andrej Karpathy's LLM Wiki pattern, stored as an OKF bundle
(Open Knowledge Format v0.2 -- https://okf.md/spec/).

## Purpose

Proferm Multitechniques is a menuiserie fabricant. This wiki is the structured, interlinked
knowledge base covering its own production and the suppliers it buys from: catalogues, fiches
techniques, tarifs, certifications, normes, procédures atelier.

It has two consumers, and they pull in the same direction:

- **The human**, who asks a question and gets an answer with its source, instead of reopening a
  200-page catalogue
- **The LIA application**, which places the WHOLE wiki in the model's context at every
  question (CAG, no retriever) and lets the user open every cited page. Much of what reads as fussy formatting in
  this file -- one row per reference, units in the header, pages that stand alone -- is there so
  that a retrieved chunk still means something once it is cut out of its page

Claude does the reading and the writing. The human curates what goes into `raw/`, asks the
questions, and arbitrates the judgment calls: a contradiction between two suppliers, a new
`type`, a categorization that could go either way.

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
`certifications/`, `tarifs/`, `procedures/`, `machines/`, `anomalies/`, and `sources/` for
`Document source` pages. A new `type` means a new subdirectory, so raise it with the user first.

## OKF conformance

Three rules make the bundle conformant. They are non-negotiable:

1. Every `.md` file in `wiki/` except `index.md` and `log.md` has parseable YAML frontmatter
2. Every frontmatter has a non-empty `type` field
3. `index.md` and `log.md` follow the reserved-file structures below

Everything else here is convention, not validation. Broken links are valid OKF -- they mark
knowledge not yet written.

## Ingest workflow

When the user adds a new source to `raw/` and asks you to ingest it:

1. Read the full source document
2. Discuss key takeaways with the user before writing anything
3. Create a summary page for the document itself (`type: Document source`), named after the source
4. Create or update a concept page for each major idea or entity -- one concept per file
5. Connect pages with bundle-relative markdown links
6. Update `wiki/index.md` with the new pages and their `description`
7. Add an entry to `wiki/log.md` under today's date, at the top of the file

A single source may touch 10-15 wiki pages. That is normal.

**The `export_doc_NNN.zip` archives ship two kinds of chunk, and only one is a source.** Chunks
that reproduce the document -- its sentences, its reference lists, its table cells -- are the
document. Chunks of smooth explanatory prose that no PDF page contains are *generated summaries*,
and they have been wrong twice: they called the coulissants 2415 and 2416 « dormants renforcés »
on a page whose own footnote calls them coulissants, and they glossed OF as « Ouvrant Fixe » where
the whole corpus, DTA included, uses OF for ouvrant à la française. Never cite them, never quote
them, and say so on the page when a reader might expect them to have been used. The same caution
applies to a PDF's extracted text layer: on a multi-column plate it interleaves labels from
neighbouring blocks, which is how the meneau 76373 came to carry the renfort references of the
76372 next to it.

## Page format

One concept per file. Frontmatter first, then structured markdown:

```markdown
---
type: Fournisseur
title: Nom du fournisseur
description: Une phrase qui résume la page.
resource: https://site-du-fournisseur.fr
tags: [aluminium, profilé]
status: stable
sources:
  - resource: raw/catalogue-2026.pdf
    title: Catalogue général 2026
    last_modified: 2026-03-14
generated:
  by: human:elie
  at: 2026-09-17T14:00:00Z
stale_after: 2027-03-14
---

# Heading

Headings, lists and tables are preferred over long prose.

Link to related concepts with bundle-relative markdown links:
[Série XYZ](/profiles/serie-xyz.md).

# Citations

[1] [Catalogue général 2026](raw/catalogue-2026.pdf), p. 42
```

Field notes:

- `type` (required) -- short, descriptive, free-form. Reuse an existing value; inventing a new
  type is a decision worth raising with the user. Vocabulary in use: `Fournisseur`, `Gamme`,
  `Profilé`, `Quincaillerie`, `Vitrage`, `Équipement`, `Porte d'entrée`, `Garantie`,
  `Procédure`, `Certification`, `Anomalie`, `Document source`. Still unused, kept for later:
  `Norme`, `Tarif`, `Machine`
- `title` -- human-readable name; may be omitted when the filename says it
- `description` -- one sentence. This is the line that gets copied into `index.md`
- `resource` -- canonical URI of the real-world asset (supplier site, product page). Omit for
  abstract concepts
- `sources` -- provenance for the claims in the body. `last_modified` is the source document's
  own date, not the ingest date
- `generated` / `verified` -- `by` is `human:<id>` or `process:<id>`, `at` is an ISO 8601
  timestamp. `generated` = who wrote the page; `verified` = who checked it against the source
- `status` -- `draft` | `stable` | `deprecated` (open vocabulary)
- `stale_after` -- `YYYY-MM-DD`. Set it on anything that expires: tarifs, certifications,
  fiches techniques, agréments. This is what makes the lint worth running
- Custom keys are allowed (`owner`, `chantier`, ...) and must be preserved when editing a page

Conventional headings: `# Cotes` for dimension tables (the French stand-in for OKF's
conventional `# Schema`), `# Examples`, `# Citations`.

## Schémas et cotes

A technical drawing cannot be stored as markdown. Extract what it asserts into a table, then
point back at the drawing. Never redraw a schéma as ASCII art, and never dissolve dimensions
into prose -- both are unretrievable.

These pages are read by a model that sees the whole wiki at once and cites them page by page,
and a reader lands on them from a citation, so every table has to stand on its own:

- **Markdown tables, not HTML.** HTML is verbose and a chunk that starts mid-table is a pile of
  orphaned `<td>` with no meaning
- **One row per reference.** The first column is always the reference itself, so a row still
  identifies what it describes once separated from its header
- **One column per dimension**, named with the drawing's own label plus the trade term when it is
  known: `Cote A - largeur hors tout (mm)`
- **Units in the header, never in the cells.** Cells hold bare numbers so they stay comparable
- **No merged cells, no two-level headers.** Markdown cannot express them and chunkers mangle
  them. Split into two tables instead
- **Keep a table under ~30 rows.** Split a large family by gamme or by usage rather than writing
  one long table
- **Introduce every table with one sentence** naming the reference family, the source and the
  unit system. That sentence is the context a retrieved chunk carries with it
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
- **The anomaly is recorded twice**: in `anomalies/`, and inline on the page where it matters.
  A reader of the parclose page must see the problem without knowing `anomalies/` exists
- **Never mark an anomaly resolved on your own.** Only the human closes one, after checking with
  the service technique or the fournisseur
- A page carrying an unresolved contradiction is `status: draft`, per the citation rules

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

Never rewrite past entries. Add today's section at the top.

## Citation rules

- Every factual claim references its source: inline `(source: fichier.pdf, p. 12)` for a single
  claim, a numbered `# Citations` section when a page draws on several documents
- The documents backing a page also go in the `sources` frontmatter field
- If two sources disagree, note the contradiction explicitly in the body and set `status: draft`
- If a claim has no source, mark it as needing verification and leave `verified` unset
- **A product document beats a general one on technical values.** Established over nine PROFERM
  sources: whenever a catalogue or a dépliant général disagrees with the brochure of the product
  itself, the general document is the one at fault -- it rounds, it generalises one range's
  figure to all, and it skips the test conditions. For a classement AEV, a Uw, an acoustic
  figure, a dimension limit or a component warranty, cite the product document. The general
  catalogue says what exists, not what it measures. This does **not** extend to company-wide
  statements such as the structural warranty, where the general document is the reference
- Never resolve a contradiction on the edition date alone, and never build a rule on two data
  points -- both were tried on the warranty question and both were wrong

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

- Check OKF conformance: frontmatter parses, `type` is non-empty, reserved files match their
  structure
- Flag pages past their `stale_after` date, and `status: draft` pages that have been sitting
- Check for contradictions between pages, and that each one is filed in `anomalies/` with an
  identifier as well as noted inline on the page where it matters
- Find orphan pages (no inbound links from other pages)
- Identify concepts mentioned in pages that lack their own page (a broken link is a valid to-do)
- Flag claims that may be outdated based on newer sources in `raw/`
- Check that `index.md` lists every page, with descriptions matching their `description` field
- Check the dimension tables: one row per reference, units in the header, no merged cells, an
  introductory sentence, a `(schéma: ...)` locator, and nothing over ~30 rows
- Check that no page is written in a language other than French
- Report findings as a numbered list with suggested fixes

## Rules

- Never modify anything in the `raw/` folder
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
- Write in clear, plain language
- When uncertain about how to categorize something -- especially a new `type` -- ask the user
