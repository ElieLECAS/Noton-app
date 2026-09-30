import copy, types
from app.services.wiki_service import get_snapshot
from app.services import wiki_index as wi
s = get_snapshot(); pages = list(s.pages.values())
P = {p.id: p for p in pages}
dta, perf = P["/certifications/dta-6-16-2334.md"], P["/gammes/perform.md"]
GAMME_SYS = {"perform": ["70", "76"], "hybride": ["70", "76"], "textural": ["70", "76"], "innoslide": ["76", "roto patio inowa"],
             "lumine": ["soleal fy", "soleal gy", "lumeal ga"], "perform+": ["roto nx"], "hybride+": ["roto nx"]}
REQ = [
 ("PERFORM76 1 vantail à la française dimension maximale", dict(type="Profilé", gamme="PERFORM76")),
 ("PERFORM76 vantail à la française dimension maximale largeur hauteur", dict(type="Profilé", gamme="PERFORM76")),
 ("PERFORM76 ouvrant à la française dimension maximale largeur hauteur", dict(type="Profilé", gamme="PERFORM76")),
 ("oscillo-battant PERFORM76 dimensions maximales", dict(type="Profilé", gamme="PERFORM")),
 ("oscillo-battant PERFORM76 dimension maximale vantail 1200 1600", dict(type="Profilé", gamme="PERFORM")),
 ("oscillo-battant PERFORM76 dimension maximale vantail LFF HFF Roto NX KSR 1200 1600", dict(type="Quincaillerie", gamme="PERFORM")),
 ("ouvrant PERFORM76 dimension maximale 1600 hauteur", dict(type="Profilé", gamme="PERFORM")),
]
def wiki_decoupe():
    corps = dta.body; i = corps.index("### 2.2.3.7 Dimensions maximales"); j = corps.index("## 2.3 Disposition de conception")
    nv = copy.copy(dta); nv.id = "/certifications/dimensions-maximales-fenetres-systeme-76.md"
    nv.title = "Dimensions maximales des fenêtres PERFORM76 (système 76 Advanced) selon le DTA"
    nv.description = "Hauteur et largeur maximales de baie par type d'ouverture (1 ou 2 vantaux OF, OB, fixe latéral, soufflet) d'une fenêtre PERFORM76, système 76 Advanced, fabrication non certifiée, DTA 6/16-2334_V5."
    nv.tags = ["dta", "dimensions-maximales", "faisabilite", "oscillo-battant", "ouvrant-a-la-francaise", "systeme-76-advanced"]
    nv.body = corps[i:j]
    d2 = copy.copy(dta); d2.body = corps[:i] + "### 2.2.3.7 Dimensions maximales\n\nVoir [Dimensions maximales](/certifications/dimensions-maximales-fenetres-systeme-76.md).\n\n" + corps[j:]
    pc = perf.body; a = pc.index("# Dimensions limites"); b = pc.index("# ", a + 5)
    p2 = copy.copy(perf); p2.body = pc[:a] + "# Dimensions limites\n\nLes dimensions maximales de baie de la PERFORM76 sont sur [Dimensions maximales des fenêtres PERFORM76](/certifications/dimensions-maximales-fenetres-systeme-76.md).\n\n" + pc[b:]
    return [p for p in pages if p.id not in (dta.id, perf.id)] + [nv, d2, p2], {nv.id}
def regle(idx, type_hors, gamme_sys, cotes):
    def score(self, requete, entry):
        sc = 0.0
        for t in requete:
            tf = entry["tf"].get(t, 0)
            if not tf: continue
            norme = 1 - wi.B + wi.B * entry["longueur"] / self.longueur_moyenne
            c = self._idf(t) * tf * (wi.K1 + 1) / (tf + wi.K1 * norme)
            sc += c * (3.0 if wi.est_reference(t) and not (cotes and t.isdigit() and t in {"1200", "1600", "1800"}) else 1.0)
        return sc
    def ok(self, entry, champ, demande):
        v = [wi.deaccent(x.lower()) for x in wi.normalise(demande)]
        if champ == "gamme" and gamme_sys:
            sy = [wi.deaccent(str(x).lower()) for x in entry["systeme"]]
            for g in v:
                for cle, lst in GAMME_SYS.items():
                    if (cle in g or g in cle) and any(x in lst for x in sy): return True
        return wi.WikiIndex._facette_ok(self, entry, champ, demande)
    def search(self, mots_cles="", type=None, tags=None, gamme=None, systeme=None, statut=None, limite=10):
        req = [t for t in wi.tokenise(mots_cles) if t not in wi.VIDES]
        dem = {k: v for k, v in (("type", None if type_hors else type), ("tags", tags), ("gamme", gamme), ("systeme", systeme)) if v}
        c = [(sc * 2.0 ** self._compte_facettes(x, dem), x) for x in self.entries if (sc := self._score_bm25(req, x)) > 0]
        c.sort(key=lambda z: z[0], reverse=True); return [x for _, x in c[:limite]]
    idx._score_bm25 = types.MethodType(score, idx); idx._facette_ok = types.MethodType(ok, idx); idx.search = types.MethodType(search, idx)
    return idx
for label, (pp, porteuses) in [("wiki actuel", (pages, {dta.id, perf.id})), ("tableau dans une page à lui", wiki_decoupe())]:
    for th, gs, co in [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]:
        idx = regle(wi.WikiIndex(pp), th, gs, co)
        r = []
        for q, f in REQ:
            L = [e["chemin"] for e in idx.search(mots_cles=q, limite=300, **f)]
            r.append(min((L.index(x) + 1 for x in porteuses if x in L), default=None))
        nom = " + ".join(n for n, on in (("type hors score", th), ("gamme→systèmes", gs), ("cotes≠réf", co)) if on) or "recherche actuelle"
        print(f"{label:<28} {nom:<44} meilleur rang {r}  top3 {sum(1 for x in r if x and x <= 3)}/7")
