import fitz  # PyMuPDF

files = [
    "GARANTIES-ENTRETIEN_guide-technique-blocs-baies-proferm.pdf",
    "GARANTIES-ENTRETIEN_roto-eneo-cc-notice-simplifiee.pdf",
    "GARANTIES-ENTRETIEN_roto-montage-nx-ksr.pdf",
    "GARANTIES-ENTRETIEN_roto-safe-e-jonction-de-cable.pdf",
    "MANUTENTION-STOCKAGE_roto-montage-nx-ksr.pdf",
    "ACCESSOIRES-PIECES_roto-bras-report-de-charge.pdf",
    "ACCESSOIRES-PIECES_roto-eneo-cc-notice-simplifiee.pdf",
    "ACCESSOIRES-PIECES_roto-montage-nx-ksr.pdf",
    "ACCESSOIRES-PIECES_roto-safe-e-jonction-de-cable.pdf",
    "ACCESSOIRES-PIECES_roto-transformation-of-en-ob.pdf",
    "REGLAGES_roto-montage-nx-ksr.pdf",
    "POSE_guide-technique-blocs-baies-proferm.pdf",
    "VOLETS-ROULANTS_guide-technique-bloc-lx.pdf",
    "VOLETS-ROULANTS_guide-technique-blocs-baies-proferm.pdf",
    "VOLETS-ROULANTS_notice-percage-coulisse-lame-serree.pdf",
]

for fname in files:
    path = f"/docs/{fname}"
    print("=" * 100)
    print(f"FILE: {fname}")
    print("=" * 100)
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"  ERROR OPENING: {e}")
        continue
    print(f"  PAGE COUNT: {doc.page_count}")
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        print("-" * 80)
        print(f"--- PAGE {i} (chars={len(text)}) ---")
        if text.strip() == "":
            print("  [EMPTY PAGE - NO EXTRACTABLE TEXT]")
        else:
            print(text)
        # also report image count for sanity re: diagrams
        img_count = len(page.get_images(full=True))
        print(f"  [images on page: {img_count}]")
    doc.close()
    print()

print("DONE")
