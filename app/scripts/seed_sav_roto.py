"""Seed du dépannage SAV « Serrure et motorisation ROTO », dans l'ordre de FLIP.

FLIP (flip-depannage.fr) enchaîne : IDENTIFIER l'équipement → DIAGNOSTIQUER → RÉPARER.
On reprend cet ordre avec notre forme actuelle (un graphe de cas) :

    niveau 1 : quel produit ?        (Eneo CC / Roto Safe E / ferrure Roto NX)
    niveau 2 : quel symptôme ?       (rien ne réagit, 3 bips, télécommande muette…)
    niveau 3 : quelle cause ?        = la solution (le « dépannage » de la notice)

Tout le contenu vient de la documentation RÉELLE du corpus, et chaque cas porte la page
d'où il sort :
  - « Proferm — Eneo CC — Notice simplifiée » p.7 (erreurs de code / reset),
    p.8 (association télécommande), p.9 (tableau des erreurs : alimentation,
    verrouillage automatique, 3 bips, télécommande) ;
  - « Roto Safe E | Jonction de câble » p.34 (§ 6.1 Dépannage + § 6.2 contrôle de
    fonctionnement : LED verte du bloc d'alimentation) ;
  - « Montage Roto NX KSR PVC » p.34 (cotes d'axe et jeux de feuillure) — cette notice
    ne contient AUCUNE table de pannes : la branche ferrure part donc au SAV au lieu
    d'inventer un diagnostic.

Cas rattachés à PLUSIEURS parents (le même contrôle sert à deux produits) :
  - les 4 symptômes de la serrure motorisée  ← Eneo CC ET Roto Safe E
  - « Pas de 24 V à la sortie du bloc d'alimentation » ← 2 symptômes différents

Usage :
    docker compose exec web python app/scripts/seed_sav_roto.py
    docker compose exec web python app/scripts/seed_sav_roto.py --draft
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlmodel import Session, select

from app.database import engine
from app.models.document import Document
from app.models.document_category import DocumentCategory
from app.models.guided_entry import GuidedSymptomAlias
from app.models.guided_tree import GuidedTree

TITLE = "Serrure et motorisation ROTO"
SYMPTOM = "serrure_motorisee"
SYMPTOM_LABEL = "Serrure ou motorisation de porte"
ALIASES = [
    "la porte ne se verrouille plus",
    "ma serrure électrique ne répond plus",
    "le boîtier bipe",
    "ma télécommande de porte ne marche plus",
    "le code ne fonctionne plus",
]
DESCRIPTION = (
    "Problème sur une serrure motorisée ou une ferrure de porte/fenêtre ROTO : la serrure "
    "ne réagit plus, le verrouillage automatique ne se fait pas, le boîtier émet des bips "
    "d'erreur, la télécommande ou le code d'accès ne fonctionne plus, ou la fenêtre manœuvre mal."
)

# Documents sources, retrouvés par titre (les identifiants diffèrent d'une base à l'autre).
DOCS = {
    "eneo": "Eneo CC",
    "cable": "Jonction de câble",
    "nx": "Montage Roto NX",
}

# (clé, nom, description, [enfants], sav, [(doc, page_debut, page_fin, légende)])
CASES = [
    # ---------- niveau 1 : identifier le produit (étape « Identifier » de FLIP) ----------
    ("p_eneo", "Serrure Eneo CC avec clavier ou lecteur d'empreinte",
     "Boîtier de contrôle d'accès 4 en 1 (code, empreinte, badge, télécommande) monté sur la "
     "porte d'entrée, associé à une serrure Roto Safe E.",
     ["s_muet", "s_verrou", "s_bips", "s_telec", "s_code"], False, []),
    ("p_safee", "Serrure Roto Safe E motorisée, sans clavier",
     "Serrure motorisée commandée par télécommande, bouton ou interphone, alimentée par une "
     "jonction de câble entre dormant et ouvrant.",
     ["s_muet", "s_verrou", "s_bips", "s_telec", "s_courant"], False, []),
    ("p_nx", "Ferrure de fenêtre Roto NX",
     "Ferrure d'ouvrant à la française ou oscillo-battant sur menuiserie PVC.",
     ["s_ferrure"], False, []),

    # ---------- niveau 2 : le symptôme (étape « Diagnostiquer » de FLIP) ----------
    ("s_muet", "Rien ne réagit, aucun bip",
     "La serrure ne donne aucun signe de vie : pas de mouvement, aucun signal sonore quand on "
     "commande l'ouverture. D'après la notice, c'est presque toujours l'alimentation.",
     ["c_220", "c_24bloc", "c_24serrure"], False,
     [("eneo", 9, 9, "tableau des erreurs : le système ne fonctionne pas")]),
    ("s_verrou", "Le verrouillage automatique ne se fait plus",
     "La porte se ferme mais la serrure ne verrouille pas d'elle-même.",
     ["c_porte", "c_modejour", "c_aimant"], False,
     [("eneo", 9, 9, "verrouillage automatique non fonctionnel")]),
    ("s_bips", "Le boîtier émet trois bips",
     "Trois bips signalent un défaut de fermeture mécanique : la serrure ne peut pas engager "
     "son pêne correctement.",
     ["c_gaches", "c_corps", "c_pene"], False,
     [("eneo", 9, 9, "signal d'erreur : 3 bips")]),
    ("s_telec", "La télécommande ne répond plus",
     "Aucun signal ne parvient au récepteur radio quand on appuie sur la télécommande.",
     ["c_assoc", "c_coderecep"], False,
     [("eneo", 8, 8, "association d'une télécommande au récepteur radio")]),
    ("s_code", "Le code n'est pas accepté ou le clavier est bloqué",
     "Le clavier refuse le code, ou ne répond plus du tout après plusieurs essais.",
     ["c_chiffres", "c_5essais", "c_reset"], False,
     [("eneo", 7, 7, "erreurs courantes du contrôle d'accès 4 en 1")]),
    ("s_courant", "Plus de courant après une intervention sur l'ouvrant",
     "Panne d'alimentation apparue après un démontage, un réglage ou une manipulation de "
     "l'ouvrant : la jonction de câble entre dormant et ouvrant est en cause.",
     ["c_connecteur", "c_24bloc"], False,
     [("cable", 34, 34, "§ 6.1 Dépannage : absence de courant électrique")]),
    ("s_ferrure", "La fenêtre manœuvre mal ou frotte",
     "Ouvrant dur à manœuvrer, qui frotte ou ferme mal. La notice de montage Roto NX donne les "
     "cotes d'axe et les jeux de feuillure, mais ne contient pas de table de pannes : le "
     "diagnostic détaillé reste à écrire, on transmet donc au SAV.",
     [], True,
     [("nx", 34, 34, "cotes d'axe de ferrage et jeux de feuillure")]),

    # ---------- niveau 3 : la cause = le dépannage (étape « Réparer » de FLIP) ----------
    ("c_220", "Pas de 220 V à l'entrée du transformateur",
     "L'installation électrique n'alimente plus le transformateur. La notice impose une "
     "intervention par un professionnel qualifié, selon les instructions IMO_438.",
     [], True, [("eneo", 9, 9, "absence d'alimentation électrique")]),
    ("c_24bloc", "Pas de 24 V à la sortie du bloc d'alimentation",
     "Le bloc d'alimentation ne délivre plus le 24 V. Vérifier les contacts du boîtier "
     "d'alimentation ; la LED du bloc doit être allumée en vert quand il y a de la tension.",
     [], False,
     [("eneo", 9, 9, "24 V non fournis à l'entrée secondaire"),
      ("cable", 34, 34, "§ 6.2 contrôle de fonctionnement : LED verte du bloc")]),
    ("c_24serrure", "Pas de 24 V jusqu'à la serrure",
     "Le bloc alimente bien, mais la tension n'arrive pas à la serrure : vérifier les câbles "
     "de liaison et leurs connexions.",
     [], False,
     [("eneo", 9, 9, "24 V non fournis à la serrure Eneo CC"),
      ("cable", 34, 34, "vérification des connexions enfichables")]),

    ("c_connecteur", "La connexion enfichable est desserrée",
     "Le connecteur 6 broches de la jonction de câble entre dormant et ouvrant s'est desserré : "
     "il suffit de refixer le connecteur. Ne jamais tirer sur le ressort pour le manipuler, "
     "utiliser une clé six-pans ou un tournevis adapté.",
     [], False,
     [("cable", 34, 34, "§ 6.1 Dépannage : la connexion enfichable est desserrée")]),

    ("c_porte", "La porte n'est pas complètement fermée",
     "Le verrouillage automatique n'est déclenché que porte entièrement fermée : la refermer "
     "franchement et réessayer.",
     [], False, [("eneo", 9, 9, "la porte n'est pas complètement fermée")]),
    ("c_modejour", "La serrure est restée en mode jour",
     "En mode jour, le verrouillage automatique est désactivé (le 24 V n'est pas fourni à "
     "l'entrée 2). Repasser en mode nuit.",
     [], False, [("eneo", 9, 9, "Eneo est en mode jour")]),
    ("c_aimant", "L'aimant en feuillure est mal aligné",
     "La serrure ne détecte pas la fermeture : vérifier la position de l'aimant en feuillure "
     "et l'ajuster.",
     [], False, [("eneo", 9, 9, "aimant en feuillure mal aligné")]),

    ("c_gaches", "La porte et les gâches sont mal alignées",
     "Ajuster la porte et les gâches selon les instructions de mise en service.",
     [], False, [("eneo", 9, 9, "porte et gâches mal alignés")]),
    ("c_corps", "Un corps étranger est dans la gâche",
     "Retirer le corps étranger présent dans la gâche, puis refaire un essai de fermeture.",
     [], False, [("eneo", 9, 9, "corps étranger dans la gâche")]),
    ("c_pene", "Le pêne n'engage pas et la porte s'ouvre légèrement",
     "Le contact reed n'est pas fermé : ouvrir la porte électriquement puis la repousser pour "
     "rétablir le contact. Si le système a été activé au cylindre, il faut le réarmer.",
     [], False, [("eneo", 9, 9, "le pêne ne s'engage pas correctement")]),

    ("c_assoc", "La télécommande n'est plus associée",
     "Reprogrammer la télécommande selon la procédure d'association : porte ouverte, serrure "
     "verrouillée à la clé, tige de 3 mm sous la zone du capteur, bip continu de 18 secondes.",
     [], False, [("eneo", 8, 8, "étapes d'association d'une télécommande")]),
    ("c_coderecep", "Le code du récepteur radio ne correspond pas",
     "Ce n'est pas le code de la télécommande qui compte mais celui du récepteur radio : les "
     "deux doivent correspondre. Vérifier le code du récepteur et les paramètres d'accès.",
     [], False, [("eneo", 8, 8, "code spécifique du récepteur radio")]),

    ("c_chiffres", "Des chiffres ont déjà été saisis sur le clavier",
     "Le code est refusé parce que des touches ont été pressées avant : appuyer sur la touche "
     "« X » du clavier pour effacer la saisie, puis entrer le code à nouveau.",
     [], False, [("eneo", 7, 7, "code non accepté")]),
    ("c_5essais", "Cinq codes faux ont bloqué le clavier",
     "Après cinq saisies incorrectes le clavier se bloque pendant 5 minutes : il suffit "
     "d'attendre la fin du blocage.",
     [], False, [("eneo", 7, 7, "clavier bloqué après plusieurs saisies incorrectes")]),
    ("c_reset", "Il faut réinitialiser l'accès",
     "Réinitialisation par le bouton Reset de la boîte noire à l'intérieur (deux bips après "
     "3 secondes), ou depuis l'application SOREX SmartLock en supprimant le premier "
     "utilisateur enregistré.",
     [], False, [("eneo", 7, 7, "réinitialisation et application SOREX SmartLock")]),
]

ROOT_CHILDREN = ["p_eneo", "p_safee", "p_nx"]


def resolve_docs(session: Session) -> dict:
    """Identifiants réels des documents, retrouvés par titre."""
    found = {}
    for key, needle in DOCS.items():
        doc = session.exec(
            select(Document).where(Document.title.ilike(f"%{needle}%"))
        ).first()
        if doc:
            found[key] = (doc.id, doc.title)
        else:
            print(f"  ! document introuvable ({needle}) — cas sans notice rattachée")
    return found


def ensure_symptom(session: Session) -> None:
    existing = session.exec(
        select(DocumentCategory).where(DocumentCategory.slug == SYMPTOM)
    ).first()
    if existing is None:
        session.add(DocumentCategory(
            slug=SYMPTOM, label=SYMPTOM_LABEL, axis="symptom",
            description="Serrure motorisée, contrôle d'accès ou ferrure de porte/fenêtre.",
        ))
        session.commit()
        print(f"  symptôme « {SYMPTOM_LABEL} » créé")
    for alias in ALIASES:
        if session.exec(
            select(GuidedSymptomAlias).where(
                GuidedSymptomAlias.symptom_slug == SYMPTOM, GuidedSymptomAlias.alias == alias
            )
        ).first() is None:
            session.add(GuidedSymptomAlias(symptom_slug=SYMPTOM, alias=alias))
    session.commit()


def build_payload(root_key: str, docs: dict) -> dict:
    by_key = {c[0]: c for c in CASES}

    def choice(child: str) -> dict:
        return {"label": by_key[child][1], "value": f"v_{child}", "hint": "",
                "next_node_key": child}

    def attachments(specs) -> list:
        out = []
        for doc_key, p1, p2, caption in specs:
            if doc_key not in docs:
                continue
            doc_id, doc_title = docs[doc_key]
            out.append({"document_id": doc_id, "document_title": doc_title,
                        "page_start": p1, "page_end": p2, "caption": caption,
                        "kind": "notice"})
        return out

    nodes = [{
        "node_key": root_key, "step_type": "question", "title": TITLE, "message": DESCRIPTION,
        "internal_note": "", "is_terminal": False, "termination_type": None,
        "ask_photo": False, "allow_free_text": True, "tools_hint": "",
        "choices": [choice(k) for k in ROOT_CHILDREN], "attachments": [],
    }]
    for key, name, desc, kids, sav, atts in CASES:
        terminal = not kids
        nodes.append({
            "node_key": key,
            "step_type": "escalation" if (terminal and sav) else ("diagnostic" if terminal else "question"),
            "title": name, "message": desc, "internal_note": "",
            "is_terminal": terminal,
            "termination_type": ("escalation" if sav else "resolution") if terminal else None,
            "ask_photo": bool(terminal and sav),
            "allow_free_text": True, "tools_hint": "",
            "choices": [choice(k) for k in kids],
            "attachments": attachments(atts),
        })
    return {"meta": {"title": TITLE, "entry_symptom": SYMPTOM, "description": DESCRIPTION,
                     "root_node_key": root_key}, "nodes": nodes}


def main() -> None:
    publish = "--draft" not in sys.argv
    from app.services.guided_authoring_service import create_tree, publish_tree, save_tree_draft

    with Session(engine) as session:
        ensure_symptom(session)
        docs = resolve_docs(session)

        for old in session.exec(select(GuidedTree).where(GuidedTree.title == TITLE)).all():
            session.delete(old)
        session.commit()

        user_id = 1
        tree = create_tree(session, title=TITLE, entry_symptom=SYMPTOM, space_id=None,
                           description=DESCRIPTION, user_id=user_id)
        save_tree_draft(session, tree.id, build_payload(tree.root_node_key, docs), user_id)

        shared = {}
        for key, _n, _d, kids, _s, _a in CASES:
            for k in kids:
                shared.setdefault(k, []).append(key)
        multi = {k: v for k, v in shared.items() if len(v) > 1}

        print(f"Arbre « {TITLE} » créé — slug={tree.slug} id={tree.id}")
        print(f"  {len(CASES) + 1} cas · 3 produits · 7 symptômes · 14 causes/solutions")
        print(f"  cas partagés entre plusieurs parents : {len(multi)}")
        for k, parents in multi.items():
            name = next(c[1] for c in CASES if c[0] == k)
            print(f"    {name[:44]:<44} ← {len(parents)} parents")

        if publish:
            res = publish_tree(session, tree.id, note="Seed ROTO (ordre FLIP)", user_id=user_id)
            print(f"  publié en v{res['version']} → proposé aux clients")
            for w in [i for i in res["lint"] if i["severity"] != "error"][:5]:
                print(f"    conseil : {w['message']}")
        else:
            print("  laissé en préparation (--draft)")


if __name__ == "__main__":
    main()
