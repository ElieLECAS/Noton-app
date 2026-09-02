"""Unités pures du packer et du reranker.

  * ``_apply_document_election`` : un classement de documents décidé en amont passe en
    tête (il prime sur le score d'élection interne du packer) et les pages épinglées
    deviennent les seeds les mieux valorisées de la fenêtre gloutonne ;
  * ``_pool_in_rerank_order`` + quota par document sur le chemin reranké : la coupe
    dynamique choisit COMBIEN de pages, le quota choisit LESQUELLES.
"""
from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from app.services.context_packer_service import _apply_document_election, _new_document_entry
from app.services.page_reranker_service import _pool_in_rerank_order
from app.services.page_retrieval_service import UnifiedPageHit, select_final_hits


def _entry(score_max: float, matched: dict) -> dict:
    entry = _new_document_entry()
    entry["score_max"] = score_max
    entry["matched_pages"] = dict(matched)
    entry["election_score"] = score_max
    return entry


# ---------------------------------------------------------------------------
# Élection du juge dans le packer
# ---------------------------------------------------------------------------


def test_elected_documents_move_to_head_in_judge_order():
    ranked = [(10, _entry(0.9, {66: 0.9})), (20, _entry(0.5, {111: 0.5}))]
    result = _apply_document_election(ranked, [20], None, max_documents=3)
    assert [did for did, _ in result] == [20, 10]


def test_election_caps_non_elected_documents():
    ranked = [(10, _entry(0.9, {})), (20, _entry(0.8, {})), (30, _entry(0.7, {}))]
    result = _apply_document_election(ranked, [30], None, max_documents=2)
    assert [did for did, _ in result] == [30, 10]


def test_elected_document_missing_from_ranked_gets_empty_entry():
    ranked = [(10, _entry(0.9, {}))]
    result = _apply_document_election(ranked, [99], None, max_documents=3)
    assert result[0][0] == 99
    assert result[0][1]["score_max"] == 0.0


def test_pinned_pages_outvalue_every_matched_page():
    """Les pages citées par le juge doivent gagner la course aux seeds : valeur
    strictement supérieure au meilleur score existant du document."""
    ranked = [(20, _entry(0.8, {42: 0.8, 43: 0.6}))]
    result = _apply_document_election(ranked, [20], {20: [111, 112]}, max_documents=3)
    matched = result[0][1]["matched_pages"]
    assert matched[111] > 0.8
    assert matched[112] > 0.8
    assert matched[42] == 0.8  # les pages du retriever ne sont pas écrasées


def test_pinned_pages_without_election_still_apply():
    ranked = [(20, _entry(0.8, {42: 0.8}))]
    result = _apply_document_election(ranked, None, {20: [111]}, max_documents=3)
    assert result[0][1]["matched_pages"][111] > 0.8


def test_no_election_no_pin_is_identity():
    ranked = [(10, _entry(0.9, {1: 0.9})), (20, _entry(0.5, {2: 0.5}))]
    assert _apply_document_election(ranked, None, None, max_documents=3) == ranked


# ---------------------------------------------------------------------------
# Quota par document sur le chemin reranké (B2)
# ---------------------------------------------------------------------------


def _hit(doc: int, page: int, rrf: float = 0.5) -> UnifiedPageHit:
    return UnifiedPageHit(document_id=doc, page_no=page, rrf_score=rrf)


def _scored(hits, raw_scores):
    scored = []
    for hit, raw in zip(hits, raw_scores):
        node = TextNode(id_=f"page-{hit.page_key}", text="t", metadata={"page_key": hit.page_key})
        scored.append((NodeWithScore(node=node, score=raw), raw))
    return scored


def test_pool_in_rerank_order_sorts_by_raw_score():
    hits = [_hit(1, 1), _hit(1, 2), _hit(2, 9)]
    pool_by_key = {h.page_key: h for h in hits}
    scored = _scored(hits, [0.2, 0.9, 0.5])
    ordered = _pool_in_rerank_order(scored, pool_by_key)
    assert [(h.document_id, h.page_no) for h in ordered] == [(1, 2), (2, 9), (1, 1)]


def test_quota_backfills_other_documents_from_beyond_the_cut():
    """Scénario « 17 passages sur 19 du même document » : le quota défère les pages
    excédentaires du doc 1 et repêche celles du doc 2 au-delà de la coupe."""
    hits = [_hit(1, p) for p in range(1, 7)] + [_hit(2, 101), _hit(2, 102)]
    # Ordre rerank : les 6 pages du doc 1 devant, doc 2 derrière.
    final, protected = select_final_hits(
        hits, 5, per_doc_quota_ratio=0.4, colpali_slots=0
    )
    docs = [h.document_id for h in final]
    assert len(final) == 5
    # Quota = ceil(5 × 0.4) = 2 : le doc 2 obtient ses 2 pages (elles étaient au-delà de
    # la coupe brute [:5]) ; le slot restant est rendu au doc 1 (quota souple).
    assert docs.count(2) == 2
    assert docs.count(1) == 3
    # Sans quota, la coupe brute aurait donné 5 pages du doc 1 et zéro du doc 2.
    brute, _ = select_final_hits(hits, 5, per_doc_quota_ratio=0, colpali_slots=0)
    assert [h.document_id for h in brute].count(2) == 0
    assert protected == []


def test_quota_is_soft_without_competition():
    hits = [_hit(1, p) for p in range(1, 6)]
    final, _ = select_final_hits(hits, 5, per_doc_quota_ratio=0.4, colpali_slots=0)
    assert len(final) == 5  # personne d'autre ne réclame les slots → rendus
