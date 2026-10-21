"""Les tableaux HTML du wiki, lus comme les tableaux markdown."""
from __future__ import annotations

from app.services.wiki_tables import tables_toutes

CORPS = """# Cotes de débit

Intro.

| A | B |
| --- | --- |
| 1 | 2 |

## Battements

<table>
<thead>
<tr><th rowspan="2">Dormant</th><th colspan="2">Battement 0140 (p. 109)</th><th colspan="2">Battements 6130 (p. 110)</th></tr>
<tr><th>DEO (mm)</th><th>DFO (mm)</th><th>DEO (mm)</th><th>DFO (mm)</th></tr>
</thead>
<tbody>
<tr><td>6100</td><td>X − 30</td><td>X − 70</td><td>X − 13</td><td>X − 53</td></tr>
<tr><td rowspan="2">6102</td><td>X − 22</td><td>X − 62</td><td>X − 5</td><td>X − 45</td></tr>
<tr><td>X − 20</td><td>X − 60</td><td>X − 3<br>mm</td><td><img src="/assets/x.png" alt="coupe"></td></tr>
</tbody>
</table>

(schéma: raw/profine.pdf, p. 6)

| C | D |
| --- | --- |
| 3 | 4 |
"""


def test_les_tableaux_markdown_et_html_sortent_dans_l_ordre_du_document():
    ts = tables_toutes(CORPS)
    assert [t.entete[0] for t in ts] == ["A", "Dormant", "C"]
    assert ts[0].lignes == [["1", "2"]] and ts[2].lignes == [["3", "4"]]


def test_l_en_tete_a_plusieurs_niveaux_est_aplati_colonne_par_colonne():
    t = tables_toutes(CORPS)[1]
    assert t.entete == ["Dormant", "Battement 0140 (p. 109) / DEO (mm)", "Battement 0140 (p. 109) / DFO (mm)",
                        "Battements 6130 (p. 110) / DEO (mm)", "Battements 6130 (p. 110) / DFO (mm)"]
    assert t.chemin == ["Cotes de débit", "Battements"] and t.section == "Battements"
    assert t.localisation == "schéma: raw/profine.pdf, p. 6"


def test_une_cellule_fusionnee_en_hauteur_est_repetee_et_br_devient_un_espace():
    t = tables_toutes(CORPS)[1]
    assert t.lignes[0] == ["6100", "X − 30", "X − 70", "X − 13", "X − 53"]
    assert t.lignes[1][0] == "6102" and t.lignes[2][0] == "6102"
    assert t.lignes[2][3] == "X − 3 mm"
    # L'image est retirée du texte, gardée dans la cellule brute.
    assert t.lignes[2][4] == "" and t.brut[2][4] == "![coupe](/assets/x.png)"


def test_le_vrai_wiki_a_des_tableaux_html_lisibles():
    from app.services.wiki_service import load_snapshot, wiki_root
    snap = load_snapshot(wiki_root())
    ts = tables_toutes(snap.pages["/profiles/systeme-70-cotes-de-debit-battements.md"].body)
    battements = next(t for t in ts if t.entete[0] == "Dormant" and any("DEO" in c for c in t.entete))
    assert battements.entete[:3] == ["Dormant", "Battement 0140 (p. 109) / DEO (mm)", "Battement 0140 (p. 109) / DFO (mm)"]
    ligne = next(l for l in battements.lignes if l[0] == "6100")
    assert ligne[1] == "X − 30" and ligne[2] == "X − 70"
