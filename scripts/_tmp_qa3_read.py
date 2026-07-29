import fitz  # PyMuPDF
import os

FILES = [
    # ASKEY
    "ACCESSOIRES-PIECES_askey-catalogue-frappe-oc-fabrication.pdf",
    "GARANTIES-ENTRETIEN_askey-dta-coulissant-nv.pdf",
    "GARANTIES-ENTRETIEN_askey-dta-frappe.pdf",
    "NON-CONFORMITE_askey-dta-coulissant-nv.pdf",
    "NON-CONFORMITE_askey-dta-frappe.pdf",
    "POSE_askey-catalogue-fabrication-coulissant-65-nv.pdf",
    "POSE_askey-catalogue-fabrication-frappe-ov.pdf",
    "POSE_askey-catalogue-frappe-oc-fabrication.pdf",
    "POSE_askey-dta-coulissant-nv.pdf",
    "POSE_askey-dta-frappe.pdf",
    # TECHNAL
    "ACCESSOIRES-PIECES_lumeal-ga-catalogue-conception.pdf",
    "ACCESSOIRES-PIECES_soleal-fy-55-evolution-catalogue-conception.pdf",
    "ACCESSOIRES-PIECES_soleal-gy-55-notice-installation.pdf",
    "GARANTIES-ENTRETIEN_dta-lumeal-ga-6-14-2166.pdf",
    "GARANTIES-ENTRETIEN_dta-soleal-fy-55-6-12-2016.pdf",
    "GARANTIES-ENTRETIEN_dta-soleal-gy-55-6-15-2261.pdf",
    "MANUTENTION-STOCKAGE_soleal-py-55-catalogue-fabrication.pdf",
    "NON-CONFORMITE_dta-soleal-fy-55-6-12-2016.pdf",
    "NON-CONFORMITE_dta-soleal-gy-55-ag152261.pdf",
    "NON-CONFORMITE_lumeal-ga-catalogue-conception.pdf",
    "NON-CONFORMITE_soleal-fy-55-evolution-catalogue-conception.pdf",
    "NON-CONFORMITE_soleal-gy-55-catalogue-conception.pdf",
    "NON-CONFORMITE_soleal-py-55-catalogue-conception.pdf",
    "POSE_dta-lumeal-ga-6-14-2166.pdf",
    "POSE_dta-soleal-fy-55-6-12-2016.pdf",
    "POSE_dta-soleal-gy-55-6-15-2261.pdf",
    "POSE_dta-soleal-gy-55-ag152261.pdf",
    "POSE_lumeal-ga-catalogue-conception.pdf",
    "POSE_lumeal-ga-catalogue-fabrication.pdf",
    "POSE_lumeal-ga-notice-installation.pdf",
    "POSE_soleal-fy-55-evolution-catalogue-conception.pdf",
    "POSE_soleal-gy-55-catalogue-conception.pdf",
    "POSE_soleal-gy-55-catalogue-fabrication.pdf",
    "POSE_soleal-gy-55-notice-installation.pdf",
    "POSE_soleal-py-55-catalogue-conception.pdf",
    "REGLAGES_lumeal-ga-catalogue-fabrication.pdf",
    "REGLAGES_lumeal-ga-notice-installation.pdf",
    "REGLAGES_soleal-fy-55-65-qc-catalogue-fabrication.pdf",
    "REGLAGES_soleal-fy-55-evolution-catalogue-fabrication.pdf",
    "REGLAGES_soleal-gy-55-catalogue-fabrication.pdf",
    "REGLAGES_soleal-gy-55-notice-installation.pdf",
    "REGLAGES_soleal-py-55-catalogue-fabrication.pdf",
]

print(f"TOTAL FILES: {len(FILES)}")

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

print("DONE")
