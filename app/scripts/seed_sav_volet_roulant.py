"""Seed d'un dépannage SAV complet : « Volet roulant bloqué ».

Exemple de référence pour l'écran /admin/sav-trees : 5 étages, et surtout des cas
RATTACHÉS À PLUSIEURS cas du dessus (un même contrôle sert à plusieurs symptômes sans
être ressaisi) :
  - « Pas de courant au moteur »        ← 2 parents (aucun bruit / s'arrête n'importe où)
  - « Blocage mécanique du tablier »    ← 3 parents (moteur qui ronfle / arrêt au même
                                          endroit / frottement)

Chaque cas porte les DEUX seuls champs saisis par le SAV : un nom (le bouton du client)
et une description (elle sert à reconnaître la demande du client, à choisir le cas, et à
rédiger le message affiché).

Usage :
    docker compose exec web python app/scripts/seed_sav_volet_roulant.py
    docker compose exec web python app/scripts/seed_sav_volet_roulant.py --draft
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports (même convention que create_user.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlmodel import Session, select

from app.database import engine
from app.models.guided_tree import GuidedTree

TITLE = "Volet roulant bloqué"
SYMPTOM = "blocage_manoeuvre"
DESCRIPTION = (
    "Le tablier ne monte plus, descend mal, ou se bloque en cours de course. "
    "Couvre les pannes moteur (alimentation, condensateur, sécurité thermique), "
    "les réglages de fin de course, et les blocages mécaniques du tablier ou des coulisses."
)

# (clé, nom, description, [clés des cas en dessous], fin_sav)
CASES = [
    ("bouge_plus", "Il ne bouge plus du tout",
     "Aucun mouvement quand on appuie sur la commande, ni à la montée ni à la descente.",
     ["aucun_bruit", "ronfle"], False),
    ("sarrete", "Il démarre puis s'arrête",
     "Le tablier part normalement mais s'immobilise avant la fin de sa course.",
     ["meme_endroit", "nimporte_ou"], False),
    ("bruit", "Il bouge mais fait un bruit anormal",
     "Le volet fonctionne encore mais un bruit nouveau apparaît pendant la manœuvre.",
     ["frottement", "claquement"], False),

    ("aucun_bruit", "Le moteur ne fait aucun bruit",
     "Silence complet à l'appui sur la commande : rien ne se met en marche.",
     ["pas_courant"], False),
    ("ronfle", "Le moteur ronfle sans que le tablier bouge",
     "On entend le moteur forcer ou vibrer, mais le tablier reste immobile.",
     ["condensateur", "blocage_meca"], False),
    ("meme_endroit", "Il s'arrête toujours au même endroit",
     "L'arrêt se produit systématiquement à la même hauteur, comme s'il rencontrait une butée.",
     ["fin_course", "blocage_meca"], False),
    ("nimporte_ou", "Il s'arrête n'importe où",
     "Les arrêts sont aléatoires, parfois après quelques secondes de fonctionnement.",
     ["pas_courant", "thermique"], False),
    ("frottement", "Grincement ou frottement",
     "Bruit continu de frottement pendant la course, souvent côté coulisses.",
     ["coulisses", "blocage_meca"], False),
    ("claquement", "Claquement métallique",
     "Bruit sec et métallique, en général en haut de course ou au démarrage.",
     ["attaches"], False),

    # ——— Cas PARTAGÉS : rattachés à plusieurs cas du dessus ———
    ("pas_courant", "Pas de courant au moteur",
     "Le moteur n'est plus alimenté : problème d'alimentation générale ou de commande.",
     ["disjoncteur", "commande"], False),
    ("blocage_meca", "Blocage mécanique du tablier",
     "Quelque chose empêche physiquement le tablier de coulisser : lame sortie, tablier "
     "déformé, ou corps étranger dans les coulisses.",
     ["lame_sortie", "tablier_voile"], False),

    ("condensateur", "Condensateur du moteur hors service",
     "Le condensateur ne fournit plus le couple de démarrage : le moteur ronfle sans "
     "entraîner le tablier. Il se remplace moteur hors tension, sans démonter le coffre.",
     [], False),
    ("fin_course", "Fin de course mal réglée",
     "La butée électronique s'est décalée : le moteur coupe avant la position réelle. "
     "Le réglage se fait avec la molette de fin de course.",
     [], False),
    ("thermique", "Le moteur se met en sécurité thermique",
     "Après plusieurs manœuvres rapprochées le moteur chauffe et se coupe seul. Il repart "
     "après une quinzaine de minutes de refroidissement : ce n'est pas une panne.",
     [], False),
    ("coulisses", "Coulisses encrassées ou déformées",
     "Poussière, gravier ou choc dans la coulisse : le tablier frotte. Nettoyage et "
     "vérification de l'alignement de la coulisse.",
     [], False),
    ("attaches", "Attaches de tablier cassées",
     "Les attaches reliant le tablier à l'axe sont rompues : intervention nécessaire, le "
     "tablier peut tomber.",
     [], True),

    ("disjoncteur", "Alimentation ou disjoncteur coupé",
     "Vérifier le disjoncteur du circuit volets et la présence de tension à la commande. "
     "Souvent un simple réarmement suffit.",
     [], False),
    ("commande", "Commande ou télécommande hors service",
     "La commande n'envoie plus l'ordre : piles de la télécommande, ou interrupteur "
     "défectueux. Tester avec une autre commande si possible.",
     [], False),
    ("lame_sortie", "Lame finale sortie de sa coulisse",
     "La lame basse est sortie du guidage d'un côté et coince le tablier. Elle se remet "
     "en place tablier en position haute.",
     [], False),
    ("tablier_voile", "Tablier déformé ou voilé",
     "Une ou plusieurs lames sont pliées : le tablier ne peut plus s'enrouler correctement. "
     "Remplacement nécessaire.",
     [], True),
]

ROOT_CHILDREN = ["bouge_plus", "sarrete", "bruit"]


def build_payload(root_key: str) -> dict:
    by_key = {c[0]: c for c in CASES}
    nodes = []

    def choice(child_key: str) -> dict:
        return {
            "label": by_key[child_key][1],
            "value": f"v_{child_key}",
            "hint": "",
            "next_node_key": child_key,
        }

    nodes.append({
        "node_key": root_key,
        "step_type": "question",
        "title": TITLE,
        "message": DESCRIPTION,
        "internal_note": "",
        "is_terminal": False,
        "termination_type": None,
        "ask_photo": False,
        "allow_free_text": True,
        "tools_hint": "",
        "choices": [choice(k) for k in ROOT_CHILDREN],
        "attachments": [],
    })

    for key, name, desc, kids, sav in CASES:
        terminal = not kids
        nodes.append({
            "node_key": key,
            "step_type": "escalation" if (terminal and sav) else ("diagnostic" if terminal else "question"),
            "title": name,
            "message": desc,
            "internal_note": "",
            "is_terminal": terminal,
            "termination_type": ("escalation" if sav else "resolution") if terminal else None,
            # Une photo aide le SAV quand on transmet le dossier.
            "ask_photo": bool(terminal and sav),
            "allow_free_text": True,
            "tools_hint": "",
            "choices": [choice(k) for k in kids],
            "attachments": [],
        })

    return {"meta": {"title": TITLE, "entry_symptom": SYMPTOM, "description": DESCRIPTION,
                     "root_node_key": root_key}, "nodes": nodes}


def main() -> None:
    publish = "--draft" not in sys.argv
    from app.services.guided_authoring_service import create_tree, publish_tree, save_tree_draft

    with Session(engine) as session:
        # Rejoue proprement : on retire un éventuel seed précédent (nœuds en cascade).
        for old in session.exec(select(GuidedTree).where(GuidedTree.title == TITLE)).all():
            session.delete(old)
        session.commit()

        user_id = 1
        tree = create_tree(
            session,
            title=TITLE,
            entry_symptom=SYMPTOM,
            space_id=None,  # global : proposé dans tous les espaces
            description=DESCRIPTION,
            user_id=user_id,
        )
        save_tree_draft(session, tree.id, build_payload(tree.root_node_key), user_id)

        draft = session.get(GuidedTree, tree.id)
        print(f"Arbre « {TITLE} » créé — slug={draft.slug} id={draft.id}")
        print(f"  {len(CASES) + 1} cas, dont 2 partagés :")
        print("    « Pas de courant au moteur »     ← 2 cas du dessus")
        print("    « Blocage mécanique du tablier » ← 3 cas du dessus")

        if publish:
            result = publish_tree(session, tree.id, note="Seed de référence", user_id=user_id)
            print(f"  publié en v{result['version']} → proposé aux clients")
            warnings = [i for i in result["lint"] if i["severity"] != "error"]
            for w in warnings[:6]:
                print(f"    conseil : {w['message']}")
        else:
            print("  laissé en préparation (--draft)")


if __name__ == "__main__":
    main()
