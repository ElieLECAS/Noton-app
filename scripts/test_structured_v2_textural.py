"""Simule chunk_markdown_structured v2 sur document_TEXTURAL (sans BDD)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/noton")
os.environ.setdefault("SECRET_KEY", "test")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pymupdf4llm
from app.services.chunking_service import chunk_markdown_structured

pdf = ROOT / "docs" / "generale" / "document_TEXTURAL.pdf"
md = pymupdf4llm.to_markdown(str(pdf), page_chunks=False, write_images=False, show_progress=False)
specs = chunk_markdown_structured(md, {"document_title": "document_TEXTURAL"})

parents = [s for s in specs if not s["is_leaf"]]
leaves = [s for s in specs if s["is_leaf"]]

print(f"total={len(specs)} parents={len(parents)} leaves={len(leaves)}")
print("\n--- PARENTS ---")
for p in parents:
    meta = p["metadata_json"]
    print(f"  [{meta.get('heading_path')}] leaves={meta.get('nb_child_leaves')} chars={len(p['content'])}")

print("\n--- FEUILLES TEXTURES ---")
for s in leaves:
    meta = s["metadata_json"] or {}
    hp = meta.get("heading_path", "-")
    if "TEXTURE" in hp.upper() or "CATALOGUE" in hp.upper():
        print(f"  path={hp} parent_id={s.get('parent_node_id', '')[:8]}... len={len(s['content'])}")

ext_parent = next(
    (p for p in parents if "EXTERIEUR" in (p["metadata_json"].get("heading_path") or "").upper()),
    None,
)
if ext_parent:
    print("\n--- PARENT EXTERIEURES (extrait) ---")
    print(ext_parent["content"][:600])
