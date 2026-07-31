"""Seed du dépannage SAV « Serrure et motorisation ROTO », dans l'ordre de FLIP.

FLIP (flip-depannage.fr) enchaîne : IDENTIFIER l'équipement → DIAGNOSTIQUER → RÉPARER.
On reprend cet ordre avec notre forme actuelle (un graphe de cas) :

    niveau 1 : quel produit ?   (contrôle d'accès 4 en 1 / Safe E / ferrure NX)
    niveau 2 : quel symptôme ?  (libellés du fabricant, tels quels)
    niveau 3 : quelle origine ? = le dépannage

⚠️ SOURCE : ce contenu est transcrit depuis la COUCHE TEXTE DES PDF, pas depuis les chunks
en base. L'ingestion de la notice Eneo CC est passée par la voie « vision » avec
ministral-3b-latest, qui a paraphrasé et parfois INVERSÉ le sens (ex. l'encart INFORMATION
de la p.8 sur le code du récepteur). Une première version de cet arbre, écrite d'après la
base, contenait donc des erreurs. Toute reprise doit repartir des PDF tant que ces
documents n'ont pas été réingérés en texte natif.

Sources exactes :
  - « Proferm — Eneo CC — Notice simplifiée » (Roto Safe E | Eneo CC)
      p.5  autotest de câblage (code 123456), p.6 plan de câblage,
      p.7  « Assistance en cas de panne » (codes) + réinitialisation usine,
      p.8  télécommande (30 max, code du récepteur, association en 6 étapes),
      p.9  « Tableau des erreurs » : 4 erreurs, 14 origines,
      p.10 consignes d'entretien et inspection.
  - « Roto Safe E | Jonction de câble » p.34 § 6.1 Dépannage (3 causes) et § 6.2
      contrôle de fonctionnement (LED verte). Les 3 dépannages y sont marqués ■ =
      « réalisation uniquement par une entreprise spécialisée » → tous en sortie SAV.
  - « Montage Roto NX KSR PVC » (pages PHYSIQUES) p.107-108 accrochage/décrochage du
      vantail, p.109 réglage des galets (goujons E/P/V), p.115 réglage du compas.
      Aucune table de pannes dans ce manuel : la branche ferrure ne propose que des
      réglages, tous réservés à une entreprise spécialisée (cf. Eneo CC p.10).

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
    "la porte ne se déverrouille pas",
    "ma serrure électrique ne répond plus",
    "le boîtier bipe",
    "ma télécommande de porte ne marche plus",
    "le code ne fonctionne plus",
    "le clavier est bloqué",
]
DESCRIPTION = (
    "Problème sur une serrure motorisée Roto Safe E, un contrôle d'accès 4 en 1 Eneo CC ou "
    "une ferrure de fenêtre Roto NX : la serrure ne réagit plus, ne se verrouille pas "
    "automatiquement, ne se verrouille pas complètement avec un signal sonore d'erreur, la "
    "porte ne se déverrouille pas, le code ou le clavier est bloqué, plus de courant après une "
    "intervention sur l'ouvrant, ou la fenêtre frotte et demande un réglage."
)

DOCS = {"eneo": "Eneo CC", "cable": "Jonction de câble", "nx": "Montage Roto NX"}

# (clé, nom, description, [enfants], sav, [(doc, p1, p2, légende)])
CASES = [
    # ================= niveau 1 : identifier le produit =================
    ("p_4en1", "Un contrôle d'accès est monté sur la porte (clavier, empreinte, badge, smartphone)",
     "Système de contrôle d'accès 4 en 1 Eneo CC : ouverture par code PIN, empreinte digitale, "
     "smartphone Bluetooth ou support RFID, associé à une serrure Roto Safe E.",
     ["s_mort", "s_verrou_auto", "s_verrou_partiel", "s_deverrou", "s_code", "s_entretien"],
     False, [("eneo", 4, 4, "système de contrôle d'accès 4 en 1")]),
    ("p_safee", "La porte est motorisée mais sans clavier ni lecteur",
     "Serrure motorisée Roto Safe E commandée par télécommande, bouton poussoir ou interphone, "
     "alimentée par une jonction de câble entre dormant et ouvrant.",
     ["s_mort", "s_verrou_auto", "s_verrou_partiel", "s_deverrou", "s_courant", "s_entretien"],
     False, []),
    ("p_nx", "Il s'agit d'une fenêtre, pas d'une porte",
     "Ferrure d'ouvrant à la française ou oscillo-battant Roto NX sur menuiserie PVC.",
     ["s_fenetre"], False, []),

    # ================= niveau 2 : les symptômes (libellés fabricant) =================
    ("s_mort", "Le système ne fonctionne pas : aucune réaction, aucun signal sonore",
     "L'Eneo CC ne réagit pas et n'émet aucun signal sonore quand on commande l'ouverture. "
     "Le tableau des erreurs donne cinq origines possibles, toutes liées à l'alimentation ou "
     "au signal, plus une procédure de dernier recours.",
     ["a_220", "a_24secondaire", "a_24serrure", "a_polarites", "a_moteur", "a_autotest", "a_recours"],
     False, [("eneo", 9, 9, "tableau des erreurs — le système ne fonctionne pas")]),
    ("s_verrou_auto", "La serrure ne se verrouille pas automatiquement",
     "La porte se ferme mais le verrouillage automatique ne se déclenche pas.",
     ["b_porte", "b_modejour", "b_aimant"], False,
     [("eneo", 9, 9, "Eneo CC ne se verrouille pas automatiquement")]),
    ("s_verrou_partiel", "La serrure ne se verrouille pas complètement et émet un signal d'erreur",
     "Le verrouillage reste incomplet et la serrure signale l'erreur par des bips : trois bips "
     "pour un défaut d'alignement ou un corps étranger, deux bips quand le contact reed n'est "
     "pas fermé.",
     ["c_gaches", "c_corps", "c_pene", "c_cylindre"], False,
     [("eneo", 9, 9, "Eneo CC ne se verrouille pas complètement (signal d'erreur)")]),
    ("s_deverrou", "La porte ne se déverrouille pas",
     "La commande n'ouvre plus la porte : aucun signal ne part de la télécommande, ou aucun "
     "signal n'arrive au récepteur de l'Eneo CC.",
     ["d_telecommande"], False,
     [("eneo", 9, 9, "la porte ne se déverrouille pas"),
      ("eneo", 8, 8, "télécommande : code du récepteur et association")]),
    ("s_code", "Le code n'est pas accepté ou le clavier ne répond plus",
     "Le clavier refuse le code saisi, ou ne réagit plus du tout après plusieurs tentatives.",
     ["e_bloque", "e_5essais", "e_reset"], False,
     [("eneo", 7, 7, "assistance en cas de panne — erreurs de code")]),
    ("s_courant", "Plus de courant après une intervention sur l'ouvrant",
     "Panne d'alimentation apparue après un démontage, un décrochage ou une manipulation de "
     "l'ouvrant : la jonction de câble entre dormant et ouvrant est en cause. La notice réserve "
     "ces trois dépannages à une entreprise spécialisée.",
     ["f_connecteur", "f_rupture", "f_connexion"], False,
     [("cable", 34, 34, "§ 6.1 Dépannage — absence de courant électrique")]),
    ("s_entretien", "Rien n'est cassé : je veux faire l'entretien",
     "Entretien préventif et inspection périodique de la serrure et des ferrures de sécurité.",
     ["g_annuel", "g_inspection"], False,
     [("eneo", 10, 10, "consignes d'entretien et inspection")]),
    ("s_fenetre", "La fenêtre frotte, ferme mal, ou le vantail est descendu",
     "Manœuvre dure, frottement, mauvais affleurement ou vantail décroché sur une ferrure "
     "Roto NX. Le manuel de montage ne contient pas de table de pannes : il donne les courses "
     "de réglage, et la notice Eneo CC précise que les travaux de réglage sur les ferrures "
     "doivent être faits par une entreprise spécialisée.",
     ["h_galets", "h_hauteur", "h_vantail"], False,
     [("nx", 109, 109, "réglage des galets — goujons E, P et V")]),

    # ================= niveau 3 : origines et dépannages =================
    # --- « Le système ne fonctionne pas » (Eneo CC p.9, 5 origines + recours) ---
    ("a_220", "Pas d'alimentation 220 V à l'entrée primaire du transformateur",
     "Le transformateur n'est plus alimenté en 220 V. La notice impose que l'installation "
     "électrique soit réalisée uniquement par un professionnel qualifié, comme précisé dans les "
     "instructions de montage IMO_438.",
     [], True, [("eneo", 9, 9, "pas d'alimentation 220 V à l'entrée primaire")]),
    ("a_24secondaire", "Pas d'alimentation 24 V à l'entrée secondaire du transformateur",
     "Le transformateur ne délivre plus le 24 V : vérifier les contacts du boîtier "
     "d'alimentation.",
     [], False,
     [("eneo", 9, 9, "pas d'alimentation 24 V à l'entrée secondaire"),
      ("eneo", 6, 6, "plan de câblage — transformateur Eneo 100-240 V AC / 24 V DC 2,5 A")]),
    ("a_24serrure", "Le 24 V n'arrive pas jusqu'à la serrure",
     "Le boîtier délivre bien le 24 V mais la serrure ne le reçoit pas : vérifier les câbles de "
     "connexion entre le boîtier d'alimentation et la serrure Eneo CC, et les remplacer si "
     "nécessaire.",
     [], False,
     [("eneo", 9, 9, "24 V non fournis à la serrure Eneo CC"),
      ("eneo", 6, 6, "plan de câblage et affectation des fils")]),
    ("a_polarites", "Le 24 V arrive à la serrure mais les polarités + et − sont inversées",
     "La tension est présente mais le + et le − ont été intervertis : inverser les polarités au "
     "niveau de l'entrée secondaire du transformateur.",
     [], False,
     [("eneo", 9, 9, "les +/- ont été intervertis"),
      ("eneo", 6, 6, "plan de câblage — brun +24 V, vert GND")]),
    ("a_moteur", "L'unité motrice est en position finale et ne reçoit pas de signal de mouvement",
     "Le moteur est arrivé en butée et n'a pas reçu l'ordre de bouger : vérifier les câbles qui "
     "transmettent le signal, ou changer la distance de l'Eneo CC (distance préconisée : 1 à 2 m).",
     [], False, [("eneo", 9, 9, "l'unité motrice est dans sa position finale")]),
    ("a_autotest", "Je veux d'abord tester le câblage",
     "La fonction autotest vérifie le câblage et les connexions avec le moteur de la serrure. "
     "Elle n'est possible qu'à l'état de livraison, et le nombre de tests n'est pas limité : "
     "entrer le code 123456 sur le clavier puis confirmer avec la touche de validation ; la "
     "porte s'ouvre si le câblage est bon.",
     [], False, [("eneo", 5, 5, "test à l'aide de la fonction autotest")]),
    ("a_recours", "Rien de tout cela n'a fonctionné",
     "Procédure de dernier recours de la notice : éteindre le transformateur, attendre "
     "10 secondes et le rallumer ; tester avec l'Unité de Contrôle Eneo ; puis contacter un "
     "spécialiste.",
     [], True, [("eneo", 9, 9, "« Ne fonctionne toujours pas ? »")]),

    # --- « Ne se verrouille pas automatiquement » (p.9, 3 origines) ---
    ("b_porte", "La porte n'est pas complètement fermée",
     "Le verrouillage automatique ne se déclenche que porte entièrement fermée : refermer la "
     "porte complètement.",
     [], False, [("eneo", 9, 9, "la porte n'est pas complètement fermée")]),
    ("b_modejour", "La serrure est en mode de fonctionnement de jour",
     "En mode jour le verrouillage automatique est inactif. Passer en mode nuit ; le 24 V ne "
     "doit pas être fourni à l'entrée 2 pour le mode nuit.",
     [], False,
     [("eneo", 9, 9, "Eneo est en mode de fonctionnement de jour"),
      ("eneo", 6, 6, "plan de câblage — contacteur jour / nuit, entrée IN2")]),
    ("b_aimant", "L'aimant en feuillure est mal aligné",
     "La serrure ne détecte pas la fermeture : vérifier la position de l'aimant en feuillure et "
     "l'ajuster.",
     [], False, [("eneo", 9, 9, "l'aimant en feuillure est mal aligné")]),

    # --- « Ne se verrouille pas complètement » (p.9, 4 origines) ---
    ("c_gaches", "La porte et les gâches ne sont pas alignées correctement — trois bips",
     "Signal d'erreur : l'Eneo bipe 3 fois. Ajuster la porte et les gâches en se référant aux "
     "instructions de mise en service.",
     [], False, [("eneo", 9, 9, "porte et gâches non alignés — l'Eneo bipe 3 x")]),
    ("c_corps", "Un corps étranger est dans la gâche — trois bips",
     "Signal d'erreur : l'Eneo bipe 3 fois. Retirer le corps étranger présent dans la gâche.",
     [], False, [("eneo", 9, 9, "corps étranger dans la gâche — l'Eneo bipe 3 x")]),
    ("c_pene", "Le pêne n'engage pas et la porte s'ouvre un peu — deux bips",
     "Signal d'erreur : contact reed non fermé, l'Eneo bipe 2 fois. Ouvrir la porte "
     "électriquement puis repousser la porte.",
     [], False, [("eneo", 9, 9, "le pêne ne s'engage pas correctement — l'Eneo bipe 2 x")]),
    ("c_cylindre", "La serrure a été activée au cylindre",
     "Si la porte a été déverrouillée manuellement au cylindre, elle doit être verrouillée à "
     "nouveau manuellement, en se référant aux instructions de mise en service.",
     [], False, [("eneo", 9, 9, "Eneo CC a été activée au cylindre")]),

    # --- « La porte ne se déverrouille pas » (p.9 + p.8) ---
    ("d_telecommande", "Aucun signal de la télécommande, ou aucun signal reçu par le récepteur",
     "Programmer la télécommande comme décrit dans la notice, puis vérifier les paramètres et "
     "le contrôle d'accès. Points utiles : jusqu'à 30 télécommandes peuvent être associées au "
     "récepteur radio ; le récepteur n'accepte les signaux que lorsque le code transmis par la "
     "télécommande correspond au sien ; un même bouton peut avoir été programmé pour une autre "
     "Eneo, et deux Eneo peuvent être commandées séparément avec une seule télécommande. "
     "Association : porte ouverte, serrure verrouillée à la clé, tige de Ø3 mm maxi dans le trou "
     "sous la zone du capteur (PVC noir), bip continu de 18 secondes, appui sur le bouton de la "
     "télécommande, confirmation par un bip de 2 secondes.",
     [], False,
     [("eneo", 8, 8, "télécommande : code du récepteur et association en 6 étapes"),
      ("eneo", 9, 9, "la porte ne se déverrouille pas")]),

    # --- Codes et clavier (p.7) ---
    ("e_bloque", "Le code est bloqué, ou des touches ont déjà été pressées",
     "Avant d'entrer le code, appuyer sur la touche « X » du clavier de code pour effacer les "
     "chiffres précédemment entrés, puis saisir le code à nouveau.",
     [], False, [("eneo", 7, 7, "le code n'a pas été accepté")]),
    ("e_5essais", "Le clavier ne répond plus après plusieurs codes incorrects",
     "Si un code incorrect a été saisi cinq fois, le clavier est bloqué pendant 5 minutes : "
     "attendre que le temps de blocage soit écoulé.",
     [], False, [("eneo", 7, 7, "le clavier cesse de répondre")]),
    ("e_reset", "Je veux remettre le contrôle d'accès aux paramètres d'usine",
     "Deux méthodes : appuyer sur le bouton Reset de la boîte noire, à l'intérieur, environ "
     "3 secondes jusqu'à ce que deux signaux soient émis en succession rapide ; ou, depuis "
     "l'application SOREX SmartLock avec le premier utilisateur enregistré, Paramètres puis "
     "« Supprimer ». Tenir compte de la portée de l'appareil.",
     [], False, [("eneo", 7, 7, "réinitialisation (paramètres d'usine)")]),

    # --- Jonction de câble Roto Safe E (p.34) : les 3 dépannages sont ■ spécialiste ---
    ("f_connecteur", "La connexion enfichable est desserrée",
     "Refixer le connecteur. La notice réserve ce dépannage à une entreprise spécialisée. Ne "
     "jamais tirer sur le ressort pour desserrer la connexion enfichable : utiliser une clé "
     "six-pans ou un tournevis adapté.",
     [], True, [("cable", 34, 34, "§ 6.1 — la connexion enfichable est desserrée")]),
    ("f_rupture", "Le câble est rompu",
     "Le câble doit être remplacé. Dépannage réservé à une entreprise spécialisée.",
     [], True, [("cable", 34, 34, "§ 6.1 — rupture du câble")]),
    ("f_connexion", "Il manque une connexion électrique",
     "Vérifier les connexions enfichables, vérifier l'alimentation électrique — la LED du bloc "
     "d'alimentation doit être allumée, elle s'allume en vert lorsqu'il y a de la tension — puis "
     "vérifier le bloc d'alimentation. Dépannage réservé à une entreprise spécialisée ; le "
     "raccordement au 230 V ne doit être effectué que par un électricien spécialisé.",
     [], True,
     [("cable", 34, 34, "§ 6.1 et § 6.2 — connexion manquante et contrôle de fonctionnement")]),

    # --- Entretien (p.10) ---
    ("g_annuel", "Entretien à faire au moins une fois par an",
     "Resserrer les vis de fixation si nécessaire, remplacer les vis endommagées, remplacer les "
     "pièces le cas échéant, et appliquer une huile spéciale sans résine ni acide sur toutes les "
     "pièces mobiles ainsi que sur les gâches en acier. Le remplacement des vis et des pièces "
     "est réservé à une entreprise spécialisée, de même que tout travail de réglage sur les "
     "ferrures.",
     [], False, [("eneo", 10, 10, "consignes d'entretien — au moins une fois par an")]),
    ("g_inspection", "Inspection périodique des ferrures de sécurité",
     "Au moins une fois par an, et tous les 6 mois dans les bâtiments scolaires et hôteliers : "
     "vérifier que les ferrures qui assurent la sécurité sont bien fixées, contrôler leur usure, "
     "vérifier le bon fonctionnement des parties mobiles et des points de fermeture. La mobilité "
     "des ferrures se contrôle à la poignée de la porte.",
     [], False, [("eneo", 10, 10, "inspection annuelle / semestrielle")]),

    # --- Ferrure Roto NX : réglages, entreprise spécialisée ---
    ("h_galets", "Le vantail serre trop ou pas assez sur le joint",
     "Réglage de la compression d'appui par les galets : selon le type de goujon (E, P ou V), la "
     "course de réglage va jusqu'à ±0,8 mm, et le goujon V permet en plus un réglage en hauteur "
     "de ±0,2 mm. Travail réservé à une entreprise spécialisée.",
     [], True, [("nx", 109, 109, "réglage des galets — goujons E, P, V et courses de réglage")]),
    ("h_hauteur", "Le vantail frotte en bas ou n'affleure plus",
     "Réglage en hauteur et latéral au palier d'angle et au compas : environ ±2 mm en hauteur, "
     "±0,5 mm en latéral et en compression selon le modèle. Après un réglage en hauteur, le "
     "report de charge doit être réglé à nouveau. Travail réservé à une entreprise spécialisée.",
     [], True,
     [("nx", 115, 115, "réglage du compas et du palier / pivot d'angle")]),
    ("h_vantail", "Le vantail a été décroché ou est mal accroché",
     "Accrochage : insérer l'ouvrant avec le compas dans le palier de compas, fermer l'ouvrant, "
     "insérer la tige d'axe par le bas et la pousser à fleur du palier. Décrochage : fenêtre "
     "fermée, pousser légèrement la tige d'axe du haut vers le bas avec l'outil d'extraction "
     "899630, puis la sortir verticalement vers le bas. Le vantail peut tomber : travail à deux "
     "et par une entreprise spécialisée.",
     [], True,
     [("nx", 107, 108, "accrochage et décrochage du vantail, outil d'extraction 899630")]),
]

ROOT_CHILDREN = ["p_4en1", "p_safee", "p_nx"]


def resolve_docs(session: Session) -> dict:
    found = {}
    for key, needle in DOCS.items():
        doc = session.exec(select(Document).where(Document.title.ilike(f"%{needle}%"))).first()
        if doc:
            found[key] = (doc.id, doc.title)
        else:
            print(f"  ! document introuvable ({needle}) — cas sans notice rattachée")
    return found


def ensure_symptom(session: Session) -> None:
    if session.exec(select(DocumentCategory).where(DocumentCategory.slug == SYMPTOM)).first() is None:
        session.add(DocumentCategory(
            slug=SYMPTOM, label=SYMPTOM_LABEL, axis="symptom",
            description="Serrure motorisée, contrôle d'accès ou ferrure de porte/fenêtre.",
        ))
        session.commit()
        print(f"  symptôme « {SYMPTOM_LABEL} » créé")
    for alias in ALIASES:
        if session.exec(select(GuidedSymptomAlias).where(
            GuidedSymptomAlias.symptom_slug == SYMPTOM, GuidedSymptomAlias.alias == alias
        )).first() is None:
            session.add(GuidedSymptomAlias(symptom_slug=SYMPTOM, alias=alias))
    session.commit()


def build_payload(root_key: str, docs: dict) -> dict:
    by_key = {c[0]: c for c in CASES}

    def choice(child: str) -> dict:
        return {"label": by_key[child][1], "value": f"v_{child}", "hint": "",
                "next_node_key": child}

    def atts(specs) -> list:
        out = []
        for doc_key, p1, p2, caption in specs:
            if doc_key not in docs:
                continue
            doc_id, doc_title = docs[doc_key]
            out.append({"document_id": doc_id, "document_title": doc_title, "page_start": p1,
                        "page_end": p2, "caption": caption, "kind": "notice"})
        return out

    nodes = [{
        "node_key": root_key, "step_type": "question", "title": TITLE, "message": DESCRIPTION,
        "internal_note": "", "is_terminal": False, "termination_type": None, "ask_photo": False,
        "allow_free_text": True, "tools_hint": "",
        "choices": [choice(k) for k in ROOT_CHILDREN], "attachments": [],
    }]
    for key, name, desc, kids, sav, specs in CASES:
        terminal = not kids
        nodes.append({
            "node_key": key,
            "step_type": "escalation" if (terminal and sav) else ("diagnostic" if terminal else "question"),
            "title": name, "message": desc, "internal_note": "",
            "is_terminal": terminal,
            "termination_type": ("escalation" if sav else "resolution") if terminal else None,
            "ask_photo": bool(terminal and sav),
            "allow_free_text": True, "tools_hint": "",
            "choices": [choice(k) for k in kids], "attachments": atts(specs),
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

        parents = {}
        for key, _n, _d, kids, _s, _a in CASES:
            for k in kids:
                parents.setdefault(k, []).append(key)
        multi = {k: v for k, v in parents.items() if len(v) > 1}
        n_att = sum(len(c[5]) for c in CASES)

        print(f"Arbre « {TITLE} » — slug={tree.slug} id={tree.id}")
        print(f"  {len(CASES) + 1} cas : 3 produits · 8 symptômes · {len(CASES) - 11} origines")
        print(f"  {n_att} notices rattachées · {sum(1 for c in CASES if c[4])} sorties SAV")
        print(f"  cas partagés entre plusieurs produits/symptômes : {len(multi)}")
        for k, ps in multi.items():
            print(f"    {next(c[1] for c in CASES if c[0] == k)[:46]:<46} ← {len(ps)}")

        if publish:
            res = publish_tree(session, tree.id, note="Seed ROTO transcrit depuis les PDF",
                               user_id=user_id)
            print(f"  publié en v{res['version']}")
            for w in [i for i in res["lint"] if i["severity"] != "error"][:4]:
                print(f"    conseil : {w['message']}")
        else:
            print("  laissé en préparation (--draft)")


if __name__ == "__main__":
    main()
