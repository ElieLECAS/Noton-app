"""Tests du résolveur de périmètre (pur, sans DB) — scénarios crémone & garde-fous."""
from app.services.scope_resolver_service import (
    CERTAIN,
    NON_PERTINENT,
    PROPOSE_CARD,
    RETRIEVE_DIRECT,
    TO_ASK,
    build_scope_card,
    resolve_scope,
)

# Un espace multi-familles / multi-matériaux typique (grand espace Technal).
STATS_MIXED = {
    "product_family": {"coulissants", "fenetres", "portes"},
    "material": {"aluminium", "pvc"},
    "product_range": {"perform", "lumine"},
    "supplier": {"Technal"},
}


def _status(res, key):
    return next(f.status for f in res.fields if f.key == key)


def _field(res, key):
    return next(f for f in res.fields if f.key == key)


def test_famille_explicite_certaine_materiau_demande():
    """Cas crémone : « coulissant » explicite → CERTAIN ; matériau absent+discriminant → TO_ASK."""
    signals = {"product_family": "coulissants", "material_hint": None}
    res = resolve_scope(
        signals,
        space_stats=STATS_MIXED,
        intent="specification",
        mode="confirm",
        askable_fields=["product_family", "material"],
    )
    assert _status(res, "product_family") == CERTAIN
    assert _field(res, "product_family").value == "coulissants"
    assert _status(res, "material") == TO_ASK
    assert set(_field(res, "material").options) == {"aluminium", "pvc"}
    # supplier mono-valeur (Technal) → CERTAIN sans être demandé.
    assert _status(res, "supplier") == CERTAIN
    assert res.decision == PROPOSE_CARD
    assert res.applied_scope["product_family"] == "coulissants"


def test_carte_contient_uniquement_les_valeurs_presentes():
    signals = {"product_family": "coulissants"}
    res = resolve_scope(signals, space_stats=STATS_MIXED, mode="confirm")
    card = build_scope_card(res)
    material_field = next(f for f in card["fields"] if f["key"] == "material")
    assert {o["value"] for o in material_field["options"]} == {"aluminium", "pvc"}
    assert {o["label"] for o in material_field["options"]} == {"Aluminium", "PVC"}


def test_perimetre_herite_pas_de_carte():
    """Tour suivant (« et la rallonge ? ») : périmètre hérité, sujet inchangé → retrieval direct."""
    signals = {"product_family": None, "material_hint": None}
    res = resolve_scope(
        signals,
        space_stats=STATS_MIXED,
        inherited_scope={"product_family": "coulissants", "material": "aluminium"},
        topic_shift=False,
        mode="confirm",
    )
    assert res.decision == RETRIEVE_DIRECT
    assert _status(res, "product_family") == CERTAIN
    assert _status(res, "material") == CERTAIN
    assert res.applied_scope == {
        "product_family": "coulissants",
        "material": "aluminium",
        "supplier": "Technal",
    }


def test_topic_shift_ignore_heritage():
    signals = {"product_family": "portes", "material_hint": "aluminium"}
    res = resolve_scope(
        signals,
        space_stats=STATS_MIXED,
        inherited_scope={"product_family": "coulissants"},
        topic_shift=True,
        mode="confirm",
    )
    assert _field(res, "product_family").value == "portes"


def test_espace_mono_famille_aucune_carte():
    """Espace scopé coulissant alu : tout CERTAIN par mono-valeur → aucune question."""
    stats = {
        "product_family": {"coulissants"},
        "material": {"aluminium"},
        "product_range": {"perform"},
        "supplier": {"Technal"},
    }
    res = resolve_scope({}, space_stats=stats, mode="confirm")
    assert res.decision == RETRIEVE_DIRECT
    assert all(f.status == CERTAIN for f in res.fields)


def test_intent_comparatif_ne_demande_pas_la_famille():
    """« coulissant vs frappe » : product_selection → famille NON_PERTINENT (on traverse)."""
    res = resolve_scope(
        {},
        space_stats=STATS_MIXED,
        intent="product_selection",
        mode="confirm",
        askable_fields=["product_family", "material"],
    )
    assert _status(res, "product_family") == NON_PERTINENT


def test_gamme_et_fournisseur_non_demandes_a_laveugle():
    """Gamme/fournisseur discriminants mais NON askable → jamais demandés à froid."""
    stats = {
        "product_family": {"coulissants"},   # mono → certain
        "material": {"aluminium"},           # mono → certain
        "product_range": {"perform", "lumine"},   # discriminant mais non askable
        "supplier": {"Technal", "Soprofen"},      # discriminant mais non askable
    }
    res = resolve_scope(
        {},
        space_stats=stats,
        mode="confirm",
        askable_fields=["product_family", "material"],
    )
    assert _status(res, "product_range") == NON_PERTINENT
    assert _status(res, "supplier") == NON_PERTINENT
    assert res.decision == RETRIEVE_DIRECT


def test_mode_auto_applique_certain_sans_carte():
    signals = {"product_family": "coulissants"}
    res = resolve_scope(
        signals,
        space_stats=STATS_MIXED,
        mode="auto",
        askable_fields=["product_family", "material"],
    )
    assert res.decision == RETRIEVE_DIRECT  # jamais de carte en auto
    assert res.applied_scope["product_family"] == "coulissants"


def test_mode_off_jamais_de_carte():
    signals = {"product_family": "coulissants"}
    res = resolve_scope(signals, space_stats=STATS_MIXED, mode="off")
    assert res.decision == RETRIEVE_DIRECT


def test_valeur_detectee_absente_de_lespace_non_bloquante():
    """Famille détectée mais absente de l'espace → NON_PERTINENT (ne vide pas le filtre)."""
    stats = {"product_family": {"fenetres", "portes"}, "material": {"pvc"}}
    signals = {"product_family": "coulissants"}
    res = resolve_scope(signals, space_stats=stats, mode="confirm")
    assert _status(res, "product_family") == NON_PERTINENT
    assert "product_family" not in res.applied_scope
