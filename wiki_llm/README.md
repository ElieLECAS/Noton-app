# La racine de connaissance de LIA

Base de connaissances structurée sur les menuiseries **PROFERM Multitechniques** et ses
fournisseurs, construite à partir des PDF commerciaux et techniques de l'entreprise.

Le principe : des PDF entrent dans `raw/`, un wiki markdown interlié en sort dans `wiki/`.

**Le wiki remplace les PDF.** Sa cible est la totalité de `raw/`, page par page : 2 382 pages
réparties sur 32 documents, dont 1 944 portent du dessin et 309 n'ont aucune couche texte
exploitable. Une information présente dans un PDF et absente du wiki est un défaut du wiki. Le
fini pèsera plus lourd qu'aucune fenêtre de contexte : on y accède par les métadonnées OKF, pas
en le mettant en entier dans un prompt.

L'application LIA (le dossier `app/` du dépôt) concatène aujourd'hui toutes les pages dans son
prompt système. **Cette conception est dépassée et l'application est en stand-by** le temps de
constituer le corpus ; l'outil de navigation par métadonnées viendra ensuite.

## Structure

```
raw/          les PDF sources, immuables — ne jamais les modifier. HORS GIT : copiés sur le
              serveur (le dossier est monté en volume dans le conteneur web)
a_faire/      les PDF en attente d'ingestion (hors git)
wiki/         le wiki markdown, bundle OKF v0.2 — VERSIONNÉ, c'est lui qu'on déploie
CLAUDE.md     les conventions du wiki, lues par Claude Code à chaque session d'ingestion
```

## Le wiki

`wiki/` est un bundle **OKF v0.2** (Open Knowledge Format) : un dossier de markdown avec
frontmatter YAML, un fichier par concept, reliés par des liens markdown relatifs à la racine du
bundle. Trois règles de conformité, non négociables :

1. tout `.md` sauf `index.md` et `log.md` porte un frontmatter YAML parsable
2. tout frontmatter porte un champ `type` non vide
3. `index.md` et `log.md` suivent leur structure réservée, sans frontmatter

**Toutes les conventions d'écriture sont dans `CLAUDE.md`** — format de page, tableaux de cotes,
règles de citation, nommage, registres d'anomalies. Ce README ne les duplique pas.

### Les registres d'anomalies

Les documents PROFERM se contredisent, contiennent des coquilles et des affirmations périmées.
**Rien n'est corrigé en silence** : chaque problème est enregistré avec un identifiant stable
(`INC-`, `CTR-`, `VER-`) et signalé aussi sur la page où il compte. LIA les consulte avant de
répondre et les cite ; l'interface les lie au registre.

## Ajouter une source

1. déposer le PDF dans `raw/`, nommé d'après son contenu et son édition
   (`catalogue-general-2026-01.pdf`, pas `doc12.pdf`) — et le copier sur le serveur
2. demander l'ingestion à Claude Code depuis ce dossier
3. Claude balaie le document page par page, **crée d'abord la carte `sources/` et son registre
   de couverture** — une ligne par page ou par plage, toutes en `à faire` — puis discute l'ordre
   de travail avec vous
4. Claude dépouille le registre : pages concept, `index.md`, `log.md`, et chaque plage passe en
   `transcrit` en nommant sa page d'arrivée
5. commiter les `.md`

Un manuel de 400 pages touche des dizaines de pages de wiki. **Une ingestion est finie quand son
registre ne porte plus aucun `à faire`**, pas quand les idées principales sont couvertes.

## Où en est la couverture

L'union des plages d'un registre couvre les pages 1 à N sans trou : c'est ce qui rend la
couverture démontrable au lieu d'être une impression. Compter les lignes par état, document par
document, donne l'avancement réel.

Mesure du 19/09/2026, avant reprise : 91 pages de wiki, 675 000 caractères, et sur les
399 références distinctes que nomment les documents profine, **154 figurent dans le wiki, soit
39 %**. Les deux gros manuels de mise en œuvre sont à 35 % et 44 %. Le catalogue Roto NX, 451
pages, porte une planche transcrite.

## Limites connues

- **Les registres de couverture restent à construire** pour les 32 sources déjà dans `raw/`.
  Tant qu'une source n'a pas le sien, son avancement n'est pas mesurable.
- **LIA n'est pas fiable à 100 %.** Toujours vérifier une cote sur la page citée, puis sur le
  PDF. Une réponse fausse se corrige dans le wiki (page plus explicite, anomalie enregistrée),
  jamais par un mécanisme dans l'application.
- **Rien n'est jamais déduit.** Une valeur illisible reste `-` avec une ligne qui le dit. Un wiki
  qui invente est pire qu'un wiki avec des trous : le trou se voit, l'invention non.
