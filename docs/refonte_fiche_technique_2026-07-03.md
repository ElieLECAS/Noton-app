# Refonte de la fiche technique — 2026-07-03

## Problème

L'ancienne fiche technique (fast-path déclenché par une référence nue) produisait un
**dump mécanique** :
- un gabarit déterministe de caractéristiques sourcées page par page ;
- une énorme section « Entités liées » = dump brut du graphe KAG
  (`6104 (8 mentions)`, `Aluminium (32 mentions)`…) sans valeur pour l'utilisateur ;
- un pied « Fiabilité : élevée — 10 caractéristique(s), 34 entité(s) liée(s) » ;
- des avertissements systématiques négatifs (« Aucune désignation explicite… ») ;
- des **sources par page** (« Sources textuelles (12) » : p.39, p.37…) au lieu du
  système « Documents consultés » utilisé partout ailleurs.

C'était un vestige d'avant le CAG : il tronquait les passages à 1400 chars alors que le
pipeline normal packe désormais des documents entiers.

## Refonte

La fiche devient une **génération naturelle** sur l'infrastructure CAG, cohérente avec
le chat :

- **Détection** (`detect_reference_query`) : inchangée dans l'esprit (regex, 0 LLM), mais
  élargie pour capturer les demandes de présentation de gamme/produit nommé
  (« parle-moi de la gamme Textural ») via des marqueurs documentaires — tout en excluant
  les questions ponctuées (`?`) et procédurales (« comment », « pourquoi »…).
- **Retrieval** (`_resolve_passages`) : inchangé — boost lexical sur la référence.
- **Contexte** : `build_cag_context(...)` — documents ENTIERS avec en-têtes gamme/matériau
  (budget « documentation »), sans exiger le bloc `<sources>` (nouveau param
  `emit_sources_tag=False`).
- **Génération** : un appel LLM avec `FICHE_SYSTEM_PROMPT` — prose fluide, structurée si
  utile, grounding strict (cotes mot pour mot), **ton naturel**, sans gabarit rigide, sans
  section Fiabilité, sans dump d'entités, sans disclaimer systématique.
- **Sources** : au niveau DOCUMENT (`build_document_sources`) → badge « Documents
  consultés », identique au reste de l'app.

## Code supprimé

Tout le gabarit et le KAG : `FicheCore`, `Characteristic`, `Source`, `_NoneTolerant`,
`RelatedEntity`, `extract_core`, `render_fiche_markdown`, `gather_related_entities`,
`_resolve_seed_entities`, `_group_related_by_type`, `_loads_json_lenient`,
`_format_passages_for_prompt`, `_sanitize_source`, `_allowed_page_keys`, et les imports
KAG (`knowledge_entity`, `kag_graph_service`, `kag_extraction_service`). Le service passe
de ~780 à ~300 lignes.

## Partagé

`_build_document_sources` / `_pages_span_label` ont été déplacés de `chat.py` vers
`context_packer_service.py` (sous `build_document_sources`, à côté de `build_cag_context`),
pour être réutilisés par la fiche sans import circulaire. `chat.py` les ré-exporte sous
leur ancien nom (compat tests).

## Tests

`test_fiche_technique.py` : détection (positifs/négatifs), génération de prose naturelle +
sources doc-level (pas de « Entités liées »/« Fiabilité »/gabarit, `cag_document: True`),
abstention sans passage. Les tests existants des helpers CAG et du routage restent verts.
