# -*- coding: utf-8 -*-
"""
Fusionne tous les fragments scripts/theme_fragments/*.json en une config
unique scripts/themes.json consommee par extract_artisan_theme.py.

Valide au passage : pages = entiers positifs, pas de doublon exact
(theme, source, page) qui indiquerait une erreur de copier-coller entre
fragments, et que chaque source pointe vers un fichier qui existe reellement
sous docs/documentations.

Usage : python scripts/merge_theme_fragments.py
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
FRAGMENTS_DIR = ROOT / "scripts" / "theme_fragments"
DOCS_ROOT = ROOT / "docs" / "documentations"
OUT_PATH = ROOT / "scripts" / "themes.json"


def main():
    all_entries = []
    errors = []

    fragment_files = sorted(FRAGMENTS_DIR.glob("*.json"))
    if not fragment_files:
        print("Aucun fragment trouve dans", FRAGMENTS_DIR)
        sys.exit(1)

    for fp in fragment_files:
        try:
            entries = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{fp.name}: JSON invalide ({e})")
            continue
        for e in entries:
            e["_fragment"] = fp.name
            all_entries.append(e)

    seen = defaultdict(list)
    for e in all_entries:
        required = {"theme", "source", "pages", "output_name"}
        missing = required - e.keys()
        if missing:
            errors.append(f"{e.get('_fragment')}: entree incomplete (manque {missing}): {e}")
            continue
        src_path = DOCS_ROOT / e["source"]
        if not src_path.exists():
            errors.append(f"{e['_fragment']}: source introuvable sur disque -> {e['source']}")
        if not isinstance(e["pages"], list) or not e["pages"]:
            errors.append(f"{e['_fragment']}: 'pages' vide ou invalide pour {e['source']} / {e['theme']}")
            continue
        for p in e["pages"]:
            if not isinstance(p, int) or p < 1:
                errors.append(f"{e['_fragment']}: page invalide {p} pour {e['source']} / {e['theme']}")
            key = (e["theme"], e["source"], p)
            seen[key].append(e["_fragment"])

    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    for (theme, source, page), fragments in dupes.items():
        errors.append(f"DOUBLON page {page} pour theme={theme} source={source} present dans {fragments}")

    if errors:
        print(f"=== {len(errors)} PROBLEME(S) DETECTE(S) ===")
        for err in errors:
            print(" -", err)
        print("\nCorrige les fragments avant de lancer l'extraction (le fichier themes.json n'a pas ete ecrit).")
        sys.exit(1)

    clean_entries = [
        {"theme": e["theme"], "source": e["source"], "pages": e["pages"], "output_name": e["output_name"]}
        for e in all_entries
    ]
    OUT_PATH.write_text(json.dumps(clean_entries, ensure_ascii=False, indent=2), encoding="utf-8")

    by_theme = defaultdict(lambda: [0, set()])
    for e in clean_entries:
        by_theme[e["theme"]][0] += len(e["pages"])
        by_theme[e["theme"]][1].add(e["source"])

    print(f"OK : {len(clean_entries)} entrees fusionnees depuis {len(fragment_files)} fragments -> {OUT_PATH}")
    print("\n=== Apercu par theme ===")
    for theme, (npages, sources) in sorted(by_theme.items()):
        print(f"{theme:25s} {npages:5d} pages sur {len(sources):3d} documents")


if __name__ == "__main__":
    main()
