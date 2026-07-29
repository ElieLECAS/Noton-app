import fitz
import json
import sys

FILES = [
    "GARANTIES-ENTRETIEN_proferm-catalogue-portes-entree.pdf",
    "GARANTIES-ENTRETIEN_proferm-referentiel-technique-gammes.pdf",
    "GARANTIES-ENTRETIEN_profine-directives-generales.pdf",
    "GARANTIES-ENTRETIEN_profine-perform70_dtd-dbv-6-16-2335.pdf",
    "GARANTIES-ENTRETIEN_profine-perform76_dta-6-16-2334.pdf",
    "GARANTIES-ENTRETIEN_profine-perform76_dtd-dbv-25-6-16-2334.pdf",
    "MANUTENTION-STOCKAGE_profine-directives-generales.pdf",
    "NON-CONFORMITE_profine-directives-generales.pdf",
    "NON-CONFORMITE_profine-perform76_dta-6-16-2334.pdf",
    "NON-CONFORMITE_vitrage-cekal-vi.pdf",
    "NON-CONFORMITE_vitrage-condensation-exterieure.pdf",
    "POSE_proferm-referentiel-technique-gammes.pdf",
    "POSE_profine-directives-generales.pdf",
    "POSE_profine-mise-en-oeuvre-fer-cintres.pdf",
    "POSE_profine-perform70_dtd-dbv-6-16-2335.pdf",
    "POSE_profine-perform76_dta-6-16-2334.pdf",
    "POSE_profine-perform76_dtd-dbv-25-6-16-2334.pdf",
    "REGLAGES_profine-directives-generales.pdf",
]

def main():
    meta = {}
    with open("/out/qa2_fulltext.txt", "w", encoding="utf-8") as ftxt:
        for fname in FILES:
            path = f"/docs/{fname}"
            try:
                doc = fitz.open(path)
            except Exception as e:
                meta[fname] = {"error": str(e)}
                continue
            page_meta = []
            for i, page in enumerate(doc):
                text = page.get_text("text")
                img_count = len(page.get_images(full=True))
                stripped = text.strip()
                page_meta.append({
                    "page": i + 1,
                    "n_chars": len(stripped),
                    "n_images": img_count,
                    "preview": stripped[:100].replace("\n", " "),
                })
                ftxt.write(f"\n===== FILE: {fname} | PAGE {i+1}/{len(doc)} | n_chars={len(stripped)} | n_images={img_count} =====\n")
                ftxt.write(text)
                ftxt.write("\n")
            meta[fname] = {"n_pages": len(doc), "pages": page_meta}
            doc.close()

    with open("/out/qa2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("DONE", file=sys.stderr)

if __name__ == "__main__":
    main()
