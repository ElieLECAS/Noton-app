"""Import d'un arbre SAV depuis le JSON pivot produit par un LLM."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.services.guided_json_import_service import (
    convert_pivot,
    is_native_payload,
    parse_json_text,
)


class _FakeSession:
    """Aucun document en bibliothèque : les sources deviennent des avertissements."""

    def __init__(self, docs=None):
        self._docs = docs or []

    def exec(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._docs

    def first(self):
        return None


PIVOT = {
    "titre": "Serrure motorisée ROTO",
    "symptome": {"nom": "Serrure motorisée", "synonymes": ["serrure qui bipe"]},
    "question_depart": "De quel équipement s'agit-il ?",
    "perimetre": {"fournisseur": ["Roto"], "familles": ["porte"], "inconnu": ["x"]},
    "cas": [
        {"id": "eneo", "nom": "Serrure Eneo CC", "description": "Motorisée.", "parents": []},
        {"id": "nx", "nom": "Ferrure NX", "description": "Ferrure.", "parents": []},
        {"id": "rien", "nom": "Rien ne se passe", "description": "Aucune réaction.",
         "parents": ["eneo"]},
        {"id": "transfo", "nom": "Le transformateur n'est pas alimenté",
         "description": "Contrôler l'alimentation 220 V.", "parents": ["rien"],
         "type": "solution", "sources": [{"document": "Eneo CC", "pages": "9"}]},
        {"id": "specialiste", "nom": "Ça ne fonctionne toujours pas",
         "description": "Entreprise spécialisée.", "parents": ["rien", "nx"], "type": "sav"},
    ],
}


def _convert(data, docs=None):
    return convert_pivot(_FakeSession(docs), data)


def test_pivot_becomes_a_graph_with_a_root():
    out = _convert(PIVOT)
    nodes = {n["node_key"]: n for n in out["payload"]["nodes"]}
    assert out["payload"]["meta"]["root_node_key"] == "root"
    assert nodes["root"]["message"] == "De quel équipement s'agit-il ?"
    # La racine propose les deux cas de premier étage.
    assert [c["next_node_key"] for c in nodes["root"]["choices"]] == ["eneo", "nx"]
    # L'arête est portée par le parent, avec le nom du cas enfant comme intitulé.
    assert nodes["rien"]["choices"][0]["label"] == "Le transformateur n'est pas alimenté"


def test_parents_multiples_donnent_un_cas_partage():
    out = _convert(PIVOT)
    nodes = {n["node_key"]: n for n in out["payload"]["nodes"]}
    assert any(c["next_node_key"] == "specialiste" for c in nodes["rien"]["choices"])
    assert any(c["next_node_key"] == "specialiste" for c in nodes["nx"]["choices"])
    assert out["report"]["shared"] == 1


def test_types_deviennent_feuilles():
    nodes = {n["node_key"]: n for n in _convert(PIVOT)["payload"]["nodes"]}
    assert (nodes["transfo"]["is_terminal"], nodes["transfo"]["termination_type"]) == (True, "resolution")
    assert (nodes["specialiste"]["is_terminal"], nodes["specialiste"]["termination_type"]) == (True, "escalation")
    assert nodes["eneo"]["is_terminal"] is False


def test_symptome_et_perimetre_normalises():
    out = _convert(PIVOT)
    assert out["symptom"]["label"] == "Serrure motorisée"
    assert out["symptom"]["aliases"] == ["serrure qui bipe"]
    assert out["payload"]["meta"]["entry_symptom"] == "serrure_motorisee"
    assert out["payload"]["meta"]["perimeter"] == {"source": ["Roto"], "product_types": ["porte"]}
    assert any("inconnu" in w for w in out["report"]["warnings"])


def test_parent_inconnu_est_une_erreur_explicite():
    data = {"titre": "T", "cas": [
        {"id": "a", "nom": "A", "parents": []},
        {"id": "b", "nom": "B", "parents": ["fantome"]},
    ]}
    with pytest.raises(HTTPException) as exc:
        _convert(data)
    errors = exc.value.detail["errors"]
    assert any("fantome" in e for e in errors)


def test_boucle_refusee():
    data = {"titre": "T", "cas": [
        {"id": "a", "nom": "A", "parents": []},
        {"id": "b", "nom": "B", "parents": ["a", "c"]},
        {"id": "c", "nom": "C", "parents": ["b"]},
    ]}
    with pytest.raises(HTTPException) as exc:
        _convert(data)
    assert any("Boucle" in e for e in exc.value.detail["errors"])


def test_sans_premier_etage_refuse():
    data = {"titre": "T", "cas": [
        {"id": "a", "nom": "A", "parents": ["b"]},
        {"id": "b", "nom": "B", "parents": ["a"]},
    ]}
    with pytest.raises(HTTPException):
        _convert(data)


def test_titre_et_noms_obligatoires():
    with pytest.raises(HTTPException) as exc:
        _convert({"cas": [{"id": "a", "parents": []}]})
    errors = exc.value.detail["errors"]
    assert any("titre" in e for e in errors)
    assert any("nom" in e for e in errors)


def test_feuille_sans_type_devient_solution_avec_avertissement():
    out = _convert({"titre": "T", "cas": [{"id": "a", "nom": "A", "parents": []}]})
    node = next(n for n in out["payload"]["nodes"] if n["node_key"] == "a")
    assert node["termination_type"] == "resolution"
    assert any("solution" in w for w in out["report"]["warnings"])


def test_type_solution_mais_avec_enfants_redevient_aiguillage():
    out = _convert({"titre": "T", "cas": [
        {"id": "a", "nom": "A", "parents": [], "type": "solution"},
        {"id": "b", "nom": "B", "parents": ["a"], "type": "solution"},
    ]})
    node = next(n for n in out["payload"]["nodes"] if n["node_key"] == "a")
    assert node["is_terminal"] is False
    assert any("aiguillage" in w for w in out["report"]["warnings"])


def test_document_absent_de_la_bibliotheque_avertit_sans_bloquer():
    out = _convert(PIVOT)
    assert out["report"]["attachments"] == 0
    assert any("Eneo CC" in w for w in out["report"]["warnings"])


def test_document_resolu_par_titre_approchant_avec_plage_de_pages():
    docs = [(390, "Proferm — Eneo CC — Notice simplifiée")]
    data = dict(PIVOT)
    data["cas"] = [
        {"id": "a", "nom": "A", "parents": [], "type": "solution",
         "sources": [{"document": "Eneo CC — Notice simplifiée", "pages": "7 à 9"}]},
    ]
    out = _convert(data, docs)
    att = out["payload"]["nodes"][1]["attachments"][0]
    assert (att["document_id"], att["page_start"], att["page_end"]) == (390, 7, 9)


def test_document_cite_par_son_nom_de_fichier():
    """Un LLM cite le PDF joint ; la bibliothèque porte un titre sans extension."""
    docs = [(500, "Notice perçage coulisse lame serrure")]
    out = _convert(
        {"titre": "T", "cas": [
            {"id": "a", "nom": "A", "parents": [], "type": "solution",
             "sources": [{"document": "noitce percage coulisse lame serruee.pdf", "pages": "1"}]},
        ]},
        docs,
    )
    assert out["payload"]["nodes"][1]["attachments"][0]["document_id"] == 500


def test_pages_listees_deviennent_une_plage():
    docs = [(501, "Guide technique blocs baies Proferm")]
    out = _convert(
        {"titre": "T", "cas": [
            {"id": "a", "nom": "A", "parents": [], "type": "solution",
             "sources": [{"document": "GUIDE TECHNIQUE BLOCS BAIES PROFERM.pdf", "pages": "24, 25"}]},
        ]},
        docs,
    )
    att = out["payload"]["nodes"][1]["attachments"][0]
    assert (att["page_start"], att["page_end"]) == (24, 25)


def test_page_fichier_prime_sur_le_numero_imprime():
    """Guide extrait : PDF de 16 pages numérotées 110→125. Le texte est indexé par rang."""
    docs = [(502, "Guide technique bloc LX")]
    out = _convert(
        {"titre": "T", "cas": [
            {"id": "a", "nom": "A", "parents": [], "type": "solution",
             "sources": [{"document": "guide technique bloc Lx.pdf", "pages": "123",
                          "page_fichier": 14, "precision": "câble de réglage"}]},
        ]},
        docs,
    )
    att = out["payload"]["nodes"][1]["attachments"][0]
    assert (att["page_start"], att["page_end"]) == (14, 14)
    assert att["caption"] == "page imprimée 123 — câble de réglage"


def test_page_fichier_identique_ne_pollue_pas_la_legende():
    docs = [(503, "Guide technique blocs baies Proferm")]
    out = _convert(
        {"titre": "T", "cas": [
            {"id": "a", "nom": "A", "parents": [], "type": "solution",
             "sources": [{"document": "GUIDE TECHNIQUE BLOCS BAIES PROFERM.pdf",
                          "pages": "24, 25", "page_fichier": "24, 25"}]},
        ]},
        docs,
    )
    att = out["payload"]["nodes"][1]["attachments"][0]
    assert (att["page_start"], att["page_end"], att["caption"]) == (24, 25, "")


def test_marqueurs_de_citation_du_llm_retires():
    """Gemini laisse des « [cite: 13] » dans son texte : ils ne doivent pas atteindre le client."""
    out = _convert({"titre": "Diagnostic [cite: 1]", "question": "Quel modèle ? [cite: 2]", "cas": [
        {"id": "a", "nom": "Coulissant LUMEAL GA [cite: 5]", "parents": [], "type": "aiguillage"},
        {"id": "b", "nom": "Roulettes à régler", "parents": ["a"], "type": "solution",
         "description": "Vérifier le réglage des roulettes[cite: 13]. Les mettre en "
                        "position haute [cite: 13, 14] .",
         "outils": "Ventouses 【4:2†source】"},
        {"id": "c", "nom": "Autre", "parents": ["a"], "type": "sav"},
    ]})
    nodes = {n["node_key"]: n for n in out["payload"]["nodes"]}
    assert out["payload"]["meta"]["title"] == "Diagnostic"
    assert nodes["root"]["message"] == "Quel modèle ?"
    assert nodes["a"]["title"] == "Coulissant LUMEAL GA"
    assert nodes["b"]["message"] == (
        "Vérifier le réglage des roulettes. Les mettre en position haute."
    )
    assert nodes["b"]["tools_hint"] == "Ventouses"
    # le libellé du choix hérite du nom nettoyé
    assert nodes["a"]["choices"][0]["label"] == "Roulettes à régler"


def test_json_entoure_de_backticks_et_de_bavardage():
    data = parse_json_text('Voici le résultat :\n```json\n{"titre": "T"}\n```\nBonne journée !')
    assert data == {"titre": "T"}
    with pytest.raises(HTTPException):
        parse_json_text("pas du json du tout")


def test_format_interne_reconnu():
    assert is_native_payload({"meta": {}, "nodes": []}) is True
    assert is_native_payload(PIVOT) is False


JETABLE = (
    '{"titre":"Arbre jetable","cas":['
    '{"id":"a","nom":"A","parents":[]},'
    '{"id":"b","nom":"B","parents":["a"],"type":"solution"},'
    '{"id":"c","nom":"C","parents":["a"],"type":"sav"}]}'
)


def test_suppression_definitive_ne_laisse_rien(db_session):
    """Un arbre supprimé emporte ses cas, notices, versions et entrées de recherche."""
    from sqlmodel import select

    from app.models.guided_entry import GuidedEntryIndex
    from app.models.guided_tree import GuidedTree, GuidedTreeNode
    from app.models.guided_tree_version import GuidedTreeVersion
    from app.services.guided_authoring_service import delete_tree
    from app.services.guided_json_import_service import import_from_json_text

    tree_id = import_from_json_text(db_session, JETABLE, space_id=None, user_id=1)["meta"]["id"]
    assert db_session.exec(select(GuidedTreeNode).where(GuidedTreeNode.tree_id == tree_id)).all()

    out = delete_tree(db_session, tree_id)
    assert out["deleted"] is True and out["sessions"] == 0
    assert db_session.get(GuidedTree, tree_id) is None
    for model in (GuidedTreeNode, GuidedTreeVersion, GuidedEntryIndex):
        assert db_session.exec(select(model).where(model.tree_id == tree_id)).all() == []


def test_suppression_arbre_inconnu_renvoie_404(db_session):
    from app.services.guided_authoring_service import delete_tree

    with pytest.raises(HTTPException) as exc:
        delete_tree(db_session, 999_999)
    assert exc.value.status_code == 404


def test_endpoint_supprime_un_brouillon(client, admin_headers, db_session):
    from app.services.guided_json_import_service import import_from_json_text

    tree_id = import_from_json_text(db_session, JETABLE, space_id=None, user_id=1)["meta"]["id"]
    r = client.delete(f"/api/admin/sav/trees/{tree_id}", headers=admin_headers)
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert client.get(f"/api/admin/sav/trees/{tree_id}", headers=admin_headers).status_code == 404


def test_editeur_sav_ne_peut_pas_supprimer_un_arbre_en_service(db_session):
    """Un arbre déjà mis en service porte un historique : réservé à l'administrateur.

    On appelle la dépendance directement : un éditeur SAV est un porteur de la permission
    `guided_trees:manage` sans le rôle admin, cas que les fixtures de rôles ne couvrent pas.
    """
    import asyncio
    from types import SimpleNamespace

    from app.models.guided_tree import GuidedTree
    from app.routers.guided_trees import delete_tree_endpoint
    from app.services.guided_json_import_service import import_from_json_text

    tree_id = import_from_json_text(db_session, JETABLE, space_id=None, user_id=1)["meta"]["id"]
    tree = db_session.get(GuidedTree, tree_id)
    tree.current_version, tree.status = 1, "published"
    db_session.add(tree)
    db_session.commit()

    editeur = SimpleNamespace(id=1, roles=["responsable"], permissions=["guided_trees:manage"])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(delete_tree_endpoint(tree_id, current_user=editeur, session=db_session))
    assert exc.value.status_code == 403
    assert "mis en service" in exc.value.detail

    admin = SimpleNamespace(id=1, roles=["admin"], permissions=[])
    out = asyncio.run(delete_tree_endpoint(tree_id, current_user=admin, session=db_session))
    assert out["deleted"] is True


def test_cles_anglaises_tolerees():
    out = _convert({"title": "T", "nodes": [
        {"id": "a", "name": "A", "description": "d", "parents": [], "type": "branch"},
        {"id": "b", "name": "B", "parents": ["A"], "type": "fix"},
    ]})
    # « parents: ["A"] » désigne le cas par son nom, pas par son id : accepté.
    nodes = {n["node_key"]: n for n in out["payload"]["nodes"]}
    assert any(c["next_node_key"] == "b" for c in nodes["a"]["choices"])
