# -*- coding: utf-8 -*-
"""
Extraction thematique "aide chantier artisan" a partir des PDF fabricant.

Produit, pour chaque document source concerne par un theme, un PDF tronque
(uniquement les pages retenues) + un manifeste CSV par theme permettant de
retracer chaque page de sortie vers sa page d'origine.

Usage (dans le conteneur, avec /docs en lecture seule, /out en ecriture,
/config/themes.json montant la config) :
    python extract_artisan_theme.py /config/themes.json

Format de la config JSON : liste d'entrees
    {
      "theme": "POSE",
      "source": "Askey/Frappe/DTA - Askey Frappe.pdf",
      "pages": [12, 13, 14, 40],
      "output_name": "dta-askey-frappe.pdf"
    }
Les pages sont 1-indexees, dans l'ordre ou elles doivent apparaitre en sortie
(doublons autorises si besoin de repeter une page, mais generalement croissant).
"""
import csv
import json
import re
import sys
from collections import defaultdict

import fitz

SRC_ROOT = "/docs"
OUT_ROOT = "/out"
DEFAULT_CONFIG = "/config/themes.json"


def clean_header(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lines = [l for l in lines if not re.fullmatch(r"[\d\s\-–|]{1,6}", l)]
    header = " / ".join(lines[:2])
    return header[:110]


def load_entries(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    by_theme = defaultdict(list)
    for e in entries:
        by_theme[e["theme"]].append((e["source"], e["pages"], e["output_name"]))
    return by_theme


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    themes = load_entries(config_path)

    summary = []
    for theme, docs in themes.items():
        manifest_rows = []
        for rel_path, pages, out_name in docs:
            src_path = f"{SRC_ROOT}/{rel_path}"
            try:
                doc = fitz.open(src_path)
            except Exception as e:
                print(f"ERREUR ouverture {rel_path}: {e}")
                continue
            new_doc = fitz.open()
            for out_idx, page_num in enumerate(pages, start=1):
                page_index = page_num - 1
                if page_index < 0 or page_index >= doc.page_count:
                    print(f"  !! page {page_num} hors bornes pour {rel_path} (skip)")
                    continue
                new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
                text = doc[page_index].get_text()
                manifest_rows.append({
                    "theme": theme,
                    "source": rel_path,
                    "page_source": page_num,
                    "page_sortie": out_idx,
                    "titre_section": clean_header(text),
                    "mots": len(text.split()),
                })
            out_path = f"{OUT_ROOT}/{theme}_{out_name}"
            new_doc.save(out_path)
            print(f"OK  {rel_path}  ->  {theme}_{out_name}  ({new_doc.page_count} pages / {doc.page_count})")
            summary.append((theme, rel_path, new_doc.page_count, doc.page_count))
            new_doc.close()
            doc.close()

        manifest_path = f"{OUT_ROOT}/{theme}_manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["theme", "source", "page_source", "page_sortie", "titre_section", "mots"],
                delimiter=";",
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Manifeste ecrit : {theme}_manifest.csv ({len(manifest_rows)} lignes)")

    print("\n=== RESUME ===")
    totals = defaultdict(lambda: [0, 0])
    for theme, rel_path, kept, total in summary:
        totals[theme][0] += kept
        totals[theme][1] += total
    for theme, (kept, total) in totals.items():
        pct = (100 * kept / total) if total else 0
        print(f"{theme:25s} {kept:5d} / {total:5d} pages ({pct:.0f}%)")


if __name__ == "__main__":
    main()
