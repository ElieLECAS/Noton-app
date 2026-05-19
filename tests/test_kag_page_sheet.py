"""Tests KAG fiche technique par page (sans LLM)."""
from dataclasses import dataclass, field
from typing import Optional

from app.services.chunk_metadata_utils import (
    assemble_page_text_from_chunks,
    dominant_parent_heading_for_chunks,
    group_chunks_by_page_no,
    mmr_diversity_key,
    page_sheet_node_id,
    resolve_page_range_from_metadata,
)
from app.services.kag_extraction_service import (
    _parse_page_sheet_response,
    format_page_technical_sheet_markdown,
)


@dataclass
class _FakeChunk:
    chunk_index: int
    content: str
    id: int = 0
    start_char: int = 0
    metadata_json: Optional[dict] = field(default_factory=dict)
    metadata_: Optional[dict] = None
    text: Optional[str] = None

    def __post_init__(self):
        if self.text is None:
            self.text = self.content


def test_page_sheet_node_id_stable():
    assert page_sheet_node_id(42, 7) == "page-sheet-42-7"


def test_group_chunks_by_page_no():
    c1 = _FakeChunk(0, "a", id=1, metadata_json={"page_no": 2})
    c2 = _FakeChunk(1, "b", id=2, metadata_json={"doc_items": [{"prov": [{"page_no": 2}]}]})
    c3 = _FakeChunk(2, "c", id=3, metadata_json={})
    by_page, without = group_chunks_by_page_no([c1, c2, c3])
    assert 2 in by_page
    assert len(by_page[2]) == 2
    assert without == [c3]


def test_group_chunks_multi_page_uses_start_only():
    c = _FakeChunk(0, "x", metadata_json={"page_start": 5, "page_end": 7})
    by_page, without = group_chunks_by_page_no([c])
    assert without == []
    assert 5 in by_page
    assert len(by_page[5]) == 1


def test_assemble_page_text_order():
    c0 = _FakeChunk(1, "second", id=2)
    c1 = _FakeChunk(0, "first", id=1)
    text = assemble_page_text_from_chunks([c0, c1])
    assert text.index("first") < text.index("second")
    assert "---" in text


def test_dominant_parent_heading():
    chunks = [
        _FakeChunk(0, "a", metadata_json={"parent_heading": "Section A"}),
        _FakeChunk(1, "b", metadata_json={"parent_heading": "Section A"}),
        _FakeChunk(2, "c", metadata_json={"parent_heading": "Section B"}),
    ]
    assert dominant_parent_heading_for_chunks(chunks) == "Section A"


def test_mmr_diversity_key_kag_page():
    key = mmr_diversity_key(
        {
            "kag_matched_entity": "Gamme Alpha",
            "document_id": 10,
            "page_no": 3,
        }
    )
    assert key == "kag_page:10:3"


def test_parse_page_sheet_response():
    raw = """
    {
      "page_summary": "Résumé pose monomur.",
      "key_facts": ["DTA 6/15-2261"],
      "entities": [
        {"name": "Monomur", "type": "concept_technique", "importance": 0.9}
      ]
    }
    """
    parsed = _parse_page_sheet_response(raw)
    assert "monomur" in parsed["page_summary"].lower()
    assert parsed["key_facts"]
    assert len(parsed["entities"]) == 1
    assert parsed["entities"][0]["name"] == "Monomur"


def test_format_page_technical_sheet_markdown():
    md = format_page_technical_sheet_markdown(
        24,
        {
            "page_summary": "Fin de document.",
            "key_facts": ["DTA V3"],
            "entities": [{"name": "SOLEAL", "type": "gamme_systeme", "importance": 1.0}],
        },
        document_title="DTA Test",
    )
    assert "page 24" in md
    assert "SOLEAL" in md
    assert "DTA Test" in md


def test_get_or_compute_chunk_entities_skips_llm_without_flag(monkeypatch):
    from app.services import kag_graph_service
    from app.services.kag_graph_service import _get_or_compute_chunk_entities

    called = {"n": 0}

    def _fake_extract(*_a, **_k):
        called["n"] += 1
        return [{"name": "X", "type": "concept_technique", "importance": 0.5}]

    monkeypatch.setattr(kag_graph_service, "extract_entities_sync", _fake_extract)
    monkeypatch.setattr(
        kag_graph_service.settings,
        "KAG_CHUNK_ENTITY_EXTRACTION_ENABLED",
        False,
    )

    chunk = _FakeChunk(0, "contenu technique suffisant pour test", metadata_json={})
    assert _get_or_compute_chunk_entities(None, chunk, chunk.content) == []
    assert called["n"] == 0


def test_resolve_page_from_doc_items_for_grouping():
    meta = {"doc_items": [{"prov": [{"page_no": 12}]}]}
    page_no, start, end = resolve_page_range_from_metadata(meta)
    assert page_no == 12
    assert start == 12
    assert end == 12
