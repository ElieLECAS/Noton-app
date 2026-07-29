import fitz  # PyMuPDF
import os

FILES = [
    "POSE_classeur-technique-e-volution.pdf",
    "POSE_kommerling70_directives-fabrication-plateforme-70.pdf",
    "POSE_profine-perform70_directives-fabrication-plateforme-70.pdf",
    "POSE_kommerling70_dta-6-16-2335.pdf",
    "POSE_profine-perform70_dta-6-16-2335.pdf",
    "GARANTIES-ENTRETIEN_kommerling70_dta-6-16-2335.pdf",
    "GARANTIES-ENTRETIEN_profine-perform70_dta-6-16-2335.pdf",
    "ACCESSOIRES-PIECES_kommerling70_poster-profiles-complementaires.pdf",
    "ACCESSOIRES-PIECES_kommerling70_poster-profiles-principaux.pdf",
]

OUT_DIR = "/docs/_qa1_extracted"
# We can't write to /docs (read-only mount), so write to /scripts if writable, else print all.

for fname in FILES:
    path = os.path.join("/docs", fname)
    print("=" * 100)
    print(f"FILE: {fname}")
    print("=" * 100)
    if not os.path.exists(path):
        print(f"  !!! MISSING FILE: {path}")
        continue
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"  !!! ERROR OPENING: {e}")
        continue
    print(f"  PAGE COUNT: {doc.page_count}")
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        print("-" * 80)
        print(f"--- PAGE {i+1} / {doc.page_count} (chars={len(text)}) ---")
        print(text)
    doc.close()
    print()
