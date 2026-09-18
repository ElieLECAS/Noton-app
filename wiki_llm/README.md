# La racine de connaissance de LIA

Base de connaissances structurée sur les menuiseries **PROFERM Multitechniques** et ses
fournisseurs, construite à partir des PDF commerciaux et techniques de l'entreprise.

Le principe : des PDF entrent dans `raw/`, un wiki markdown interlié en sort dans `wiki/`, et
l'application LIA (le dossier `app/` du dépôt) le sert : **chat CAG** (tout le wiki dans le
contexte à chaque question) et **graphe** à la Obsidian avec lecteur de pages.

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
3. Claude lit le document en entier, **discute les points clés avant d'écrire**, puis crée la
   page source, les pages concept, met à jour `index.md` et `log.md`
4. commiter les `.md`, `git pull` sur le serveur : l'application recharge le wiki au prochain
   appel (le premier appel après un changement paie le prompt plein tarif, c'est attendu)

Une source touche typiquement 10 à 15 pages. Les contradictions découvertes partent dans les
registres, jamais dans une correction silencieuse.

## Le budget de contexte

Le wiki entier part dans le prompt système à chaque question. Mesuré le 18/09/2026 : 73 pages,
542 000 caractères, **185 500 tokens** sur une fenêtre de 256 k. À ~20 000 à 30 000 caractères
par document technique, la fenêtre sature vers **25 sources** (18 aujourd'hui). La carte
« Wiki » de l'administration affiche le budget et les tokens réels du dernier appel ;
l'application avertit au-delà de 200 000 tokens estimés.

## Limites connues

- **Deux tableaux ne sont volontairement pas transcrits** : la matrice de compatibilité du
  catalogue portes (p. 141) et les cotes des posters A0 profine. Les pages concernées le disent
  et renvoient au PDF.
- **107 des 113 pages des directives générales profine** restent à dépouiller.
- **LIA n'est pas fiable à 100 %.** Toujours vérifier une cote sur la page citée, puis sur le
  PDF. Une réponse fausse se corrige dans le wiki (page plus explicite, anomalie enregistrée),
  jamais par un mécanisme dans l'application.
