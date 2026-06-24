"""Tests de l'extraction d'illustration ancrée sur la référence (illustration_service).

Vérifie la logique géométrique déterministe (sans appel vision) sur un PDF synthétique :
- localisation des codes (find_code_labels)
- découpe ancrée qui isole la cible et n'inclut jamais un code frère (build_anchored_crop)
- abstention quand la cible n'est pas isolable
- orchestration extract_reference_illustration (cible absente, succès, garde-fou)
"""
from __future__ import annotations

import pytest
from PIL import Image

from app.services import illustration_service as ill


# ---------------------------------------------------------------------------
# PDF synthétique : deux schémas distincts (6104 à gauche, 6105 à droite)
# ---------------------------------------------------------------------------
def _make_synth_pdf(path: str) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    def cluster(x0: float, y0: float) -> None:
        # ~10 traits formant une région ~90x90 pts (≥ MIN_OWNED_STROKES, ≥ MIN_DIAGRAM_PTS)
        for i in range(5):
            page.draw_rect(fitz.Rect(x0, y0 + i * 18, x0 + 90, y0 + 12 + i * 18), width=1)
        for j in range(5):
            page.draw_line(fitz.Point(x0 + j * 18, y0), fitz.Point(x0 + j * 18, y0 + 90), width=1)

    cluster(100, 200)   # schéma du 6104 : x ∈ [100, 190]
    cluster(380, 200)   # schéma du 6105 : x ∈ [380, 470]
    page.insert_text(fitz.Point(120, 320), "6104", fontsize=11)
    page.insert_text(fitz.Point(400, 320), "6105", fontsize=11)
    doc.save(path)
    doc.close()


@pytest.fixture()
def synth_pdf(tmp_path):
    path = str(tmp_path / "synth_profiles.pdf")
    _make_synth_pdf(path)
    return path


def _open_page(path):
    import fitz

    doc = fitz.open(path)
    return doc, doc[0]


# ---------------------------------------------------------------------------
# Géométrie déterministe
# ---------------------------------------------------------------------------
def test_find_code_labels_detects_both_codes(synth_pdf):
    doc, page = _open_page(synth_pdf)
    try:
        codes = {t for _, t in ill.find_code_labels(page)}
        assert "6104" in codes
        assert "6105" in codes
    finally:
        doc.close()


def test_anchored_crop_isolates_6104(synth_pdf):
    doc, page = _open_page(synth_pdf)
    try:
        labels = ill.find_code_labels(page)
        anchors = page.search_for("6104")
        assert anchors, "ancre 6104 introuvable"
        crop = ill.build_anchored_crop(page, "6104", anchors[0], labels)
        assert crop is not None, "le 6104 doit produire un crop"
        inside = ill.codes_inside(crop, labels)
        assert inside == {"6104"}, f"le crop ne doit contenir que 6104, obtenu {inside}"
        # le crop ne doit pas déborder sur la zone du 6105 (x ≥ 380)
        assert crop.x1 < 380
    finally:
        doc.close()


def test_anchored_crop_isolates_6105(synth_pdf):
    doc, page = _open_page(synth_pdf)
    try:
        labels = ill.find_code_labels(page)
        anchors = page.search_for("6105")
        crop = ill.build_anchored_crop(page, "6105", anchors[0], labels)
        assert crop is not None
        assert ill.codes_inside(crop, labels) == {"6105"}
        # ne déborde pas sur la zone du 6104 (x ≤ 190)
        assert crop.x0 > 190
    finally:
        doc.close()


def test_codes_inside_excludes_outside_labels(synth_pdf):
    import fitz

    doc, page = _open_page(synth_pdf)
    try:
        labels = ill.find_code_labels(page)
        # rectangle couvrant uniquement la moitié gauche
        left_half = fitz.Rect(0, 0, 260, 842)
        assert ill.codes_inside(left_half, labels) == {"6104"}
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_extract_absent_code_returns_none(synth_pdf):
    # 9999 n'est pas sur la page → abstention stricte
    result = await ill.extract_reference_illustration(
        targets=["9999"], pdf_path=synth_pdf, page_no=1, doc_title="Synthétique"
    )
    assert result is None


@pytest.mark.asyncio
async def test_extract_no_targets_returns_none(synth_pdf):
    result = await ill.extract_reference_illustration(
        targets=[], pdf_path=synth_pdf, page_no=1, doc_title="Synthétique"
    )
    assert result is None


@pytest.mark.asyncio
async def test_extract_success_gate_disabled(synth_pdf, monkeypatch):
    # On court-circuite le rendu (poppler) et le cache disque pour rester hermétique.
    monkeypatch.setattr(ill, "_make_crop_image", lambda *a, **k: Image.new("RGB", (120, 120), "white"))
    monkeypatch.setattr(ill, "_cache_crop_image", lambda *a, **k: "crop_fake.png")
    monkeypatch.setattr(ill.settings, "ILLUSTRATION_VISION_GATE_ENABLED", False)

    result = await ill.extract_reference_illustration(
        targets=["6104"], pdf_path=synth_pdf, page_no=1, doc_title="Synthétique"
    )
    assert result is not None
    assert result["reference"] == "6104"
    assert result["url"].endswith("crop_fake.png")
    assert result["page_no"] == 1


@pytest.mark.asyncio
async def test_extract_gate_rejection_returns_none(synth_pdf, monkeypatch):
    monkeypatch.setattr(ill, "_make_crop_image", lambda *a, **k: Image.new("RGB", (120, 120), "white"))
    monkeypatch.setattr(ill, "_cache_crop_image", lambda *a, **k: "crop_fake.png")
    monkeypatch.setattr(ill.settings, "ILLUSTRATION_VISION_GATE_ENABLED", True)

    async def _reject(_img, _target):
        return False

    monkeypatch.setattr(ill, "_vision_read_gate", _reject)

    result = await ill.extract_reference_illustration(
        targets=["6104"], pdf_path=synth_pdf, page_no=1, doc_title="Synthétique"
    )
    assert result is None
