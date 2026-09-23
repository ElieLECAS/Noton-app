"""Le dépôt du wiki : classement des chemins, plan annoncé, bascule, garde-fous.

Tous les tests travaillent sur un wiki TEMPORAIRE (``WIKI_DIR`` détourné) : le dépôt écrit sur
disque et supprime des pages, il n'a rien à faire dans ``wiki_llm/``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app.config import settings
from app.services import wiki_depot, wiki_service


def _page(titre: str, corps: str = "Contenu.") -> str:
    return f"---\ntitle: {titre}\ntype: Profilé\n---\n\n{corps}\n"


@pytest.fixture
def racine(tmp_path, monkeypatch) -> Path:
    """Un wiki de trois pages, un PDF, monté à la place du vrai."""
    base = tmp_path / "wiki_llm"
    (base / "wiki" / "profiles").mkdir(parents=True)
    (base / "wiki" / "assets").mkdir(parents=True)
    (base / "raw").mkdir()
    (base / "wiki" / "index.md").write_text(
        "# Index\n\n- [Parcloses](/profiles/parcloses.md)\n- [Joints](/profiles/joints.md)\n",
        encoding="utf-8",
    )
    (base / "wiki" / "profiles" / "parcloses.md").write_text(_page("Parcloses"), encoding="utf-8")
    (base / "wiki" / "profiles" / "joints.md").write_text(_page("Joints"), encoding="utf-8")
    (base / "raw" / "catalogue.pdf").write_bytes(b"%PDF-1.4 ancien")

    monkeypatch.setattr(settings, "WIKI_DIR", str(base))
    wiki_service.reset_snapshot()
    yield base
    wiki_service.reset_snapshot()


def _annonce(base: Path, *fichiers: tuple[str, int]) -> dict:
    return {"fichiers": [{"chemin": c, "taille": t} for c, t in fichiers]}


# ---------------------------------------------------------------------------
# Le classement des chemins : la seule barrière entre un glisser-déposer et le disque
# ---------------------------------------------------------------------------


def test_classer_trouve_la_racine_quel_que_soit_le_dossier_glisse():
    for brut in (
        "wiki_llm/wiki/profiles/x.md",
        "wiki/profiles/x.md",
        "./Documents/Code/Noton-app/wiki_llm/wiki/profiles/x.md",
        "wiki_llm\\wiki\\profiles\\x.md",  # Windows
    ):
        c = wiki_depot.classer(brut)
        assert (c.racine, c.relatif) == ("wiki", "profiles/x.md"), brut
    c = wiki_depot.classer("wiki_llm/raw/moustiquaires/notice.pdf")
    assert (c.racine, c.relatif) == ("raw", "moustiquaires/notice.pdf")


def test_classer_ecarte_ce_qui_sort_ou_ne_nous_regarde_pas():
    for brut, attendu in [
        ("wiki/../../.env", "remontée"),
        ("wiki_llm/CLAUDE.md", "hors de"),
        ("wiki_llm/a_faire/note.md", "hors de"),
        ("wiki_llm/wiki/notes.txt", ".txt"),
        ("wiki_llm/raw/exports/donnees.json", ".json"),
        ("/etc/passwd", "hors de"),
        ("wiki", "hors de"),
    ]:
        c = wiki_depot.classer(brut)
        assert not c.racine, brut
        assert attendu in c.raison, (brut, c.raison)


def test_classer_accepte_les_images_du_wiki_pas_ailleurs():
    assert wiki_depot.classer("wiki/assets/profiles/p.png").racine == "wiki"
    assert not wiki_depot.classer("raw/p.png").racine


# ---------------------------------------------------------------------------
# Le plan : ce que le serveur réclame, et ce que ça changera
# ---------------------------------------------------------------------------


def test_ouvrir_reclame_tout_le_wiki_et_les_seuls_pdf_manquants(racine, client, admin_headers):
    r = client.post(
        "/api/wiki/depot",
        headers=admin_headers,
        json=_annonce(
            racine,
            ("wiki_llm/wiki/index.md", 10),
            ("wiki_llm/wiki/profiles/parcloses.md", 10),
            ("wiki_llm/wiki/profiles/nouvelle.md", 10),
            ("wiki_llm/raw/catalogue.pdf", len(b"%PDF-1.4 ancien")),  # déjà là, même taille
            ("wiki_llm/raw/notice.pdf", 12),  # absent du serveur
            ("wiki_llm/CLAUDE.md", 5),  # écarté
        ),
    )
    assert r.status_code == 200
    data = r.json()
    cles = {a["cle"] for a in data["attendus"]}
    # Le wiki monte en entier ; des PDF, seul celui qui manque.
    assert cles == {
        "wiki/index.md",
        "wiki/profiles/parcloses.md",
        "wiki/profiles/nouvelle.md",
        "raw/notice.pdf",
    }
    plan = data["plan"]
    assert plan["miroir_wiki"] is True
    assert plan["pages_ajoutees"] == ["profiles/nouvelle.md"]
    # joints.md est sur le serveur, pas dans le dépôt : le miroir l'annonce comme supprimée.
    assert plan["pages_supprimees"] == ["profiles/joints.md"]
    assert plan["pdf_nouveaux"] == ["notice.pdf"] and plan["pdf_remplaces"] == []
    assert data["ignores_total"] == 1 and data["ignores"][0]["chemin"] == "wiki_llm/CLAUDE.md"
    # Le chemin réclamé est celui que le navigateur a annoncé : il retrouve son fichier.
    assert {a["chemin"] for a in data["attendus"]} >= {"wiki_llm/raw/notice.pdf"}


def test_pdf_de_taille_differente_est_reclame_et_marque_remplace(racine, client, admin_headers):
    data = client.post(
        "/api/wiki/depot",
        headers=admin_headers,
        json=_annonce(racine, ("raw/catalogue.pdf", 999)),
    ).json()
    assert [a["cle"] for a in data["attendus"]] == ["raw/catalogue.pdf"]
    assert data["plan"]["pdf_remplaces"] == ["catalogue.pdf"]


def test_depot_sans_page_ne_touche_pas_au_wiki(racine, client, admin_headers):
    """Glisser ``raw/`` seul ajoute des PDF — et n'efface pas 198 pages au passage."""
    data = client.post(
        "/api/wiki/depot", headers=admin_headers, json=_annonce(racine, ("raw/notice.pdf", 3))
    ).json()
    assert data["plan"]["miroir_wiki"] is False
    assert data["plan"]["pages_supprimees"] == []

    depot = data["depot"]
    assert client.put(
        f"/api/wiki/depot/{depot}/fichier?chemin=raw/notice.pdf",
        headers=admin_headers,
        content=b"PDF",
    ).status_code == 200
    r = client.post(f"/api/wiki/depot/{depot}/valider", headers=admin_headers)
    assert r.status_code == 200
    resume = r.json()["resume"]
    assert resume["pdf_nouveaux"] == ["notice.pdf"] and resume["pages_supprimees"] == []
    assert (racine / "raw" / "notice.pdf").read_bytes() == b"PDF"
    assert (racine / "wiki" / "profiles" / "joints.md").is_file()


# ---------------------------------------------------------------------------
# La bascule
# ---------------------------------------------------------------------------


def _deposer_wiki(client, headers, depot: str, fichiers: dict[str, bytes]) -> None:
    for chemin, contenu in fichiers.items():
        r = client.put(
            f"/api/wiki/depot/{depot}/fichier?chemin={chemin}", headers=headers, content=contenu
        )
        assert r.status_code == 200, (chemin, r.text)


def test_valider_remplace_le_wiki_et_recharge_linstantane(racine, client, admin_headers):
    nouvelle = _page("Nouvelle").encode("utf-8")
    index = b"# Index\n\n- [Parcloses](/profiles/parcloses.md)\n"
    parcloses = _page("Parcloses", "Corps corrige.").encode("utf-8")

    data = client.post(
        "/api/wiki/depot",
        headers=admin_headers,
        json=_annonce(
            racine,
            ("wiki/index.md", len(index)),
            ("wiki/profiles/parcloses.md", len(parcloses)),
            ("wiki/profiles/nouvelle.md", len(nouvelle)),
        ),
    ).json()
    depot = data["depot"]
    _deposer_wiki(
        client,
        admin_headers,
        depot,
        {
            "wiki/index.md": index,
            "wiki/profiles/parcloses.md": parcloses,
            "wiki/profiles/nouvelle.md": nouvelle,
        },
    )
    r = client.post(f"/api/wiki/depot/{depot}/valider", headers=admin_headers)
    assert r.status_code == 200
    resume = r.json()["resume"]
    assert resume["pages_ajoutees"] == ["profiles/nouvelle.md"]
    assert resume["pages_modifiees"] == ["index.md", "profiles/parcloses.md"]
    assert resume["pages_supprimees"] == ["profiles/joints.md"]

    # Sur disque : joints a disparu, nouvelle est là, parcloses porte le nouveau corps.
    assert not (racine / "wiki" / "profiles" / "joints.md").exists()
    assert (racine / "wiki" / "profiles" / "nouvelle.md").is_file()
    assert "Corps corrige." in (racine / "wiki" / "profiles" / "parcloses.md").read_text("utf-8")
    # Et l'assistant lit déjà le nouveau wiki, sans redémarrage.
    pages = {p.id for p in wiki_service.get_snapshot().concept_pages}
    assert "/profiles/nouvelle.md" in pages and "/profiles/joints.md" not in pages
    assert not (racine / wiki_depot.DEPOTS / depot).exists()
    assert r.json()["stats"]["pages"] == len(pages)


def test_un_envoi_interrompu_ne_touche_pas_au_wiki_servi(racine, client, admin_headers):
    contenu = _page("Parcloses").encode("utf-8")
    data = client.post(
        "/api/wiki/depot",
        headers=admin_headers,
        json=_annonce(
            racine,
            ("wiki/index.md", 5),
            ("wiki/profiles/parcloses.md", len(contenu)),
        ),
    ).json()
    depot = data["depot"]
    _deposer_wiki(client, admin_headers, depot, {"wiki/profiles/parcloses.md": contenu})

    # index.md manque : on ne bascule pas un wiki à trous.
    r = client.post(f"/api/wiki/depot/{depot}/valider", headers=admin_headers)
    assert r.status_code == 409
    assert r.json()["detail"]["manquants"] == ["wiki/index.md"]
    assert (racine / "wiki" / "profiles" / "joints.md").is_file()

    # Et l'abandon jette la préparation sans rien changer.
    assert client.delete(f"/api/wiki/depot/{depot}", headers=admin_headers).status_code == 204
    assert not (racine / wiki_depot.DEPOTS / depot).exists()
    assert (racine / "wiki" / "profiles" / "joints.md").is_file()
    assert client.post(f"/api/wiki/depot/{depot}/valider", headers=admin_headers).status_code == 404


# ---------------------------------------------------------------------------
# Garde-fous
# ---------------------------------------------------------------------------


def test_seul_un_fichier_annonce_est_accepte(racine, client, admin_headers):
    depot = client.post(
        "/api/wiki/depot", headers=admin_headers, json=_annonce(racine, ("wiki/index.md", 3))
    ).json()["depot"]
    for chemin in ("wiki/profiles/intrus.md", "../../.env", "raw/intrus.pdf", "wiki/../x.md"):
        r = client.put(
            f"/api/wiki/depot/{depot}/fichier?chemin={chemin}", headers=admin_headers, content=b"x"
        )
        assert r.status_code == 400, chemin
    assert not (racine / "wiki" / "profiles" / "intrus.md").exists()
    assert not (racine / "raw" / "intrus.pdf").exists()
    assert not (racine.parent / ".env").exists()


def test_depot_inconnu_ou_identifiant_bricole(racine, client, admin_headers):
    for depot in ("inconnu", "../../etc", "0" * 32):
        assert client.put(
            f"/api/wiki/depot/{depot}/fichier?chemin=wiki/index.md",
            headers=admin_headers,
            content=b"x",
        ).status_code == 404


def test_depot_reserve_aux_administrateurs(racine, client, lecteur_headers):
    assert client.post("/api/wiki/depot", headers=lecteur_headers, json={"fichiers": []}).status_code == 403
    assert client.post("/api/wiki/depot", json={"fichiers": []}).status_code == 401
    assert client.post(f"/api/wiki/depot/{'a' * 32}/valider", headers=lecteur_headers).status_code == 403
    assert client.delete(f"/api/wiki/depot/{'a' * 32}", headers=lecteur_headers).status_code == 403


def test_le_manifeste_survit_a_un_redemarrage(racine, client, admin_headers):
    """Aucun état en mémoire : le dépôt est relu sur disque à chaque requête."""
    depot = client.post(
        "/api/wiki/depot", headers=admin_headers, json=_annonce(racine, ("raw/notice.pdf", 3))
    ).json()["depot"]
    manifeste = json.loads(
        (racine / wiki_depot.DEPOTS / depot / "manifeste.json").read_text("utf-8")
    )
    assert manifeste["attendus"] == {"raw/notice.pdf": 3}
    assert manifeste["miroir_wiki"] is False


def test_classer_refuse_les_noms_de_fichier_tordus():
    """Les deux barrières que rien d'autre ne franchit : caractères de contrôle et noms
    Windows invalides. Un chemin n'est jamais résolu sur disque — il est validé segment
    par segment, donc ces refus sont la seule protection."""
    for brut, attendu in [
        ("wiki/profiles/page\x00.md", "caractère interdit"),
        ('wiki/profiles/pa"ge.md', "caractère interdit"),
        ("wiki/profiles/C:pipe|.md", "caractère interdit"),
        ("wiki/ profiles /x.md", "nom de fichier invalide"),
        ("wiki/profiles./x.md", "nom de fichier invalide"),
    ]:
        c = wiki_depot.classer(brut)
        assert not c.racine, brut
        assert attendu in c.raison, (brut, c.raison)


def test_annonce_trop_large_refusee(racine, client, admin_headers, monkeypatch):
    monkeypatch.setattr(wiki_depot, "ANNONCES_MAX", 3)
    r = client.post(
        "/api/wiki/depot",
        headers=admin_headers,
        json=_annonce(racine, *[(f"wiki/p{i}.md", 1) for i in range(4)]),
    )
    assert r.status_code == 400 and "trop large" in r.json()["detail"]
    assert not (racine / wiki_depot.DEPOTS).exists()


def test_fichier_trop_lourd_ecarte_du_plan(racine, client, admin_headers):
    data = client.post(
        "/api/wiki/depot",
        headers=admin_headers,
        json=_annonce(
            racine,
            ("raw/enorme.pdf", wiki_depot.TAILLE_MAX + 1),
            ("raw/normal.pdf", 10),
        ),
    ).json()
    assert [a["cle"] for a in data["attendus"]] == ["raw/normal.pdf"]
    assert data["ignores"] == [
        {"chemin": "raw/enorme.pdf", "raison": f"plus lourd que {wiki_depot.TAILLE_MAX // 2**20} Mo"}
    ]


def test_un_fichier_qui_gonfle_en_cours_denvoi_est_coupe(racine, client, admin_headers, monkeypatch):
    """La taille annoncée n'engage que celui qui l'annonce : la limite est aussi tenue
    pendant l'écriture, sinon un client bavard remplirait le disque."""
    monkeypatch.setattr(wiki_depot, "TAILLE_MAX", 64)
    depot = client.post(
        "/api/wiki/depot", headers=admin_headers, json=_annonce(racine, ("raw/notice.pdf", 10))
    ).json()["depot"]
    r = client.put(
        f"/api/wiki/depot/{depot}/fichier?chemin=raw/notice.pdf",
        headers=admin_headers,
        content=b"x" * 500,
    )
    assert r.status_code == 400 and "dépasse" in r.json()["detail"]
    assert not (racine / "raw" / "notice.pdf").exists()
    # Et le fichier partiel ne reste pas dans le dossier du dépôt.
    assert not list((racine / wiki_depot.DEPOTS / depot).glob("part-*"))


def test_un_depot_oublie_est_ramasse_a_louverture_du_suivant(racine, client, admin_headers):
    vieux = client.post(
        "/api/wiki/depot", headers=admin_headers, json=_annonce(racine, ("raw/a.pdf", 1))
    ).json()["depot"]
    dossier = racine / wiki_depot.DEPOTS / vieux
    assert dossier.is_dir()
    perime = time.time() - wiki_depot.DEPOT_PERIME_S - 60
    os.utime(dossier, (perime, perime))

    client.post("/api/wiki/depot", headers=admin_headers, json=_annonce(racine, ("raw/b.pdf", 1)))
    assert not dossier.exists()
    assert client.post(f"/api/wiki/depot/{vieux}/valider", headers=admin_headers).status_code == 404


def test_premier_depot_sur_un_serveur_sans_dossier_raw(racine, client, admin_headers):
    """Le cas du serveur neuf : ``raw/`` n'existe pas (il n'est pas versionné, rien ne l'a
    créé). Le plan doit se construire quand même, et le dossier naître au premier PDF."""
    import shutil as _shutil

    _shutil.rmtree(racine / "raw")
    assert not (racine / "raw").exists()

    data = client.post(
        "/api/wiki/depot", headers=admin_headers, json=_annonce(racine, ("raw/notice.pdf", 3))
    ).json()
    assert data["plan"]["pdf_en_ligne"] == 0
    assert data["plan"]["pdf_nouveaux"] == ["notice.pdf"]

    depot = data["depot"]
    _deposer_wiki(client, admin_headers, depot, {"raw/notice.pdf": b"PDF"})
    assert client.post(f"/api/wiki/depot/{depot}/valider", headers=admin_headers).status_code == 200
    assert (racine / "raw" / "notice.pdf").read_bytes() == b"PDF"
