"""Tests nettoyage et chunking pymupdf4llm propre."""
from __future__ import annotations

from app.services.pdf_extraction_service import clean_pymupdf4llm_markdown, extract_first_heading
from app.services.chunking_service import chunk_pymupdf4llm_page_clean

ROTO_PAGE1_RAW = """
**Date de rédaction : 19/03/2026**

**PVC**

**Réf. :** PRO-PVC-OFOB-01

**Transformation OF en OB GAMME ROTO NX**

**==> picture [112 x 33] intentionally omitted <==**

## **1. Démonter l'equerre de compas et le compas OF d'origine.**

**==> picture [411 x 235] intentionally omitted <==**

**----- Start of picture text -----**<br>
 Compas OF<br>Equerre de Compas OF<br>**----- End of picture text -----**<br>


**2. Instaler et visser la tetiere fournis à dimension.**

**3. Instaler le compas OB adequat fournis sur la tetiere.**

**==> picture [239 x 110] intentionally omitted <==**
"""

ROTO_PAGE2_RAW = """
## **4. Retirer l'obturateur de manœuvre sur la quincaillerie d'origine afin de liberer la manœuvre OB**

**==> picture [202 x 87] intentionally omitted <==**

**5. Instaler et visser la gache OB (Droite ou GAUCHE) en traverse basse du dormant.**

**----- Start of picture text -----**<br>
Gâche OB Droite<br>Gâche OB Gauche<br>**----- End of picture text -----**<br>
"""


class TestCleanPymupdf4llmMarkdown:
    def test_removes_picture_placeholders(self):
        cleaned = clean_pymupdf4llm_markdown(ROTO_PAGE1_RAW)
        assert "intentionally omitted" not in cleaned
        assert "==> picture" not in cleaned

    def test_extracts_picture_text_as_bullets(self):
        cleaned = clean_pymupdf4llm_markdown(ROTO_PAGE1_RAW)
        assert "- Compas OF" in cleaned
        assert "- Equerre de Compas OF" in cleaned
        assert "Start of picture text" not in cleaned

    def test_removes_page_markers(self):
        raw = "<!-- page:2 -->\n\n## Section\n\nTexte"
        cleaned = clean_pymupdf4llm_markdown(raw)
        assert "<!-- page:" not in cleaned
        assert "## Section" in cleaned

    def test_unwraps_bold(self):
        cleaned = clean_pymupdf4llm_markdown("**PVC**")
        assert cleaned == "PVC"

    def test_extract_first_heading(self):
        h = extract_first_heading(ROTO_PAGE1_RAW)
        assert "1. Démonter" in h


class TestChunkPymupdf4llmPageClean:
    def test_page1_splits_into_logical_sections(self):
        specs = chunk_pymupdf4llm_page_clean(
            ROTO_PAGE1_RAW,
            page_no=1,
            metadata_base={"document_title": "ROTO NX"},
        )
        assert len(specs) >= 4  # en-tête + étape 1 + étapes 2 et 3

        contents = " ".join(s["content"] for s in specs)
        assert "PRO-PVC-OFOB-01" in contents
        assert "Compas OF" in contents
        assert "intentionally omitted" not in contents
        assert "<!-- page:" not in contents

        step_numbers = [
            s["metadata_json"].get("step_number")
            for s in specs
            if s["metadata_json"].get("step_number") is not None
        ]
        assert 1 in step_numbers
        assert 2 in step_numbers
        assert 3 in step_numbers

    def test_page2_splits_steps(self):
        specs = chunk_pymupdf4llm_page_clean(
            ROTO_PAGE2_RAW,
            page_no=2,
            metadata_base={"document_title": "ROTO NX"},
        )
        step_numbers = [
            s["metadata_json"].get("step_number") for s in specs
        ]
        assert 4 in step_numbers
        assert 5 in step_numbers
        assert any("Gâche OB Droite" in s["content"] for s in specs)

    def test_all_specs_are_leaves(self):
        specs = chunk_pymupdf4llm_page_clean(
            ROTO_PAGE1_RAW, page_no=1, metadata_base={}
        )
        assert all(s["is_leaf"] for s in specs)
        assert all(
            s["metadata_json"]["content_type"] == "semantic_leaf" for s in specs
        )
