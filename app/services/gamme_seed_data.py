"""Fiches initiales des gammes commerciales Proferm.

Rédigées le 2026-08-26 depuis les documents Proferm réellement en base (catalogue
général 424, dépliant général 425, fiche Hybride 426, Innoslide 427, Lumine65 429,
dossier technique Perform 76 434). Chaque affirmation vient d'un chunk ; ce qui n'y
figure pas est signalé dans `a_valider` plutôt que deviné.

⚠️ TOUTES ces fiches naissent en statut BROUILLON. Elles décrivent le PAYSAGE (quelle
gamme, quel matériau, quels mots chercher), jamais des valeurs faisant autorité : une
cote, un Uw ou une référence se lisent toujours dans le document.

Ce module ne sert qu'à l'amorçage. Une fois les fiches relues dans l'admin, c'est la
base qui fait foi — le seed ne réécrit jamais une fiche existante.
"""
from typing import Any, Dict, List

# Note de vocabulaire : `document_ids` renvoie aux ids de la base de DÉVELOPPEMENT.
# En production ils diffèrent — le rattachement s'y refait par titre/fournisseur ou à la
# main dans l'admin. Le reste de la fiche est portable tel quel.

GAMMES_SEED: List[Dict[str, Any]] = [
    {
        "slug": "perform",
        "nom": "PERFORM",
        "accroche": "La fenêtre ultra-durable",
        "materiau": "pvc",
        "familles": ["fenetres", "portes", "coulissants"],
        "ordre": 10,
        "description": (
            "Menuiserie PVC, cœur de gamme Proferm. Tous les profils sont en PVC "
            "GREENLINE® de KÖMMERLING® : sans plomb, totalement recyclable. Couvre "
            "l'ouverture à la française, l'oscillo-battante, le soufflet, le châssis fixe "
            "et le coulissant. Performances thermiques annoncées jusqu'à Uw 1,3 W/m²K. "
            "Proferm est classé parmi les 5 meilleurs fabricants français au test A*E*V "
            "sur cette gamme. Points mis en avant : renforcement total des profils, "
            "étanchéité, tenue aux UV et aux agressions chimiques, ferrure sécurisée."
        ),
        "alias_utilisateur": [
            "Perform", "PERFORM", "gamme Perform", "Perform 70", "Perform 76",
            "gamme 70", "gamme 76", "le PVC", "menuiserie PVC",
        ],
        "termes_documentaires": [
            "PVC GREENLINE", "GREENLINE", "KÖMMERLING", "KOMMERLING",
            "TROCAL 76 ADVANCED", "KBE 76 ADVANCED", "KÖMMERLING 76 ADVANCED",
            "Kömmerling Gamme 70", "e.VOLUTION", "e.XCLUSIVE",
        ],
        "fournisseurs": ["profine", "Kömmerling", "TROCAL", "KBE"],
        "discriminants": (
            "« Perform » n'apparaît dans AUCUN document technique fournisseur : toujours "
            "traduire en système profine avant de chercher (76 ADVANCED, Gamme 70).\n"
            "Perform 70 ≠ Perform 76 : une cote, un profil ou une référence lus pour une "
            "profondeur ne valent JAMAIS pour l'autre, et une référence de la Gamme 70 "
            "(ex. dormant 6111) n'existe pas « en version 76 » par analogie.\n"
            "Ne pas mobiliser les documents Lumine (aluminium/Technal) pour une question "
            "Perform, ni l'inverse."
        ),
        "document_ids": [424, 425, 434, 412, 413, 411],
        "a_valider": (
            "Le partage exact entre « Gamme 70 frappe » (posters) et « e.VOLUTION / "
            "e.XCLUSIVE » (portes/coulissants) : même système ou sous-systèmes distincts "
            "de la même profondeur 70 ?\n"
            "Existe-t-il un coulissant en 76, ou le coulissant PVC est-il porté "
            "exclusivement par le 70 (e.XCLUSIVE) et l'INNOSLIDE ?"
        ),
    },
    {
        "slug": "hybride",
        "nom": "HYBRIDE",
        "accroche": "La fenêtre écologique",
        "materiau": "hybride",
        "familles": ["fenetres", "portes", "coulissants"],
        "ordre": 20,
        "description": (
            "Menuiserie mixte développée exclusivement par Proferm : profil intérieur en "
            "PVC GREENLINE® de KÖMMERLING®, profil extérieur en aluminium épais serti sur "
            "le PVC. Le profilé PVC fait 72 mm et intègre 5 chambres d'isolation. "
            "Présentée comme unique sur le marché. Deux niveaux de finition extérieure : "
            "DROIT ou DESIGN. Garantie 15 ans sur la structure. Certifications NF, CE, "
            "CSTB, CEKAL, Qualimarine®, Qualanod. Quincaillerie Roto (Winflex & Door "
            "Technology), pivot supportant jusqu'à 130 kg."
        ),
        "alias_utilisateur": [
            "Hybride", "HYBRIDE", "gamme Hybride", "PVC-alu", "PVC alu",
            "mixte PVC aluminium", "la mixte",
        ],
        "termes_documentaires": [
            "HYBRIDE", "PVC GREENLINE", "KÖMMERLING", "profil extérieur aluminium serti",
            "5 chambres", "72 mm",
        ],
        "fournisseurs": ["Proferm", "profine", "Kömmerling", "Roto"],
        "discriminants": (
            "Développement propre Proferm : le système n'existe pas sous ce nom chez le "
            "fournisseur. L'intérieur relève du PVC Kömmerling, l'extérieur de "
            "l'aluminium — ne pas confondre avec LUMINE (aluminium intégral) ni avec "
            "TEXTURAL® (également mixte, mais orientée décoration intérieure).\n"
            "Les couleurs intérieures et extérieures suivent deux nuanciers DISTINCTS."
        ),
        "document_ids": [426, 430, 431, 424, 425],
        "a_valider": (
            "Sur quel(s) système(s) profine la partie PVC repose-t-elle (Gamme 70 ? 76 "
            "ADVANCED ? les deux) ? Le profilé de 72 mm ne correspond ni à 70 ni à 76.\n"
            "Familles réellement disponibles : la fiche couvre-t-elle les portes et les "
            "coulissants, ou uniquement la fenêtre ?"
        ),
    },
    {
        "slug": "textural",
        "nom": "TEXTURAL®",
        "accroche": "La fenêtre qui habille votre intérieur",
        "materiau": "hybride",
        "familles": ["fenetres"],
        "ordre": 30,
        "description": (
            "Gamme décorative mixte PVC/aluminium entièrement pensée par Proferm : base "
            "PVC sertie d'aluminium. L'intérieur met en avant des essences de bois et des "
            "textures (carbone noir, gris écaille, teck foncé, noyer, wengé, cuir "
            "anthracite, argenté, terre de Sienne, chêne, hêtre, glossy RAL 3002 ou 9010) ; "
            "l'extérieur aluminium porte les couleurs et finitions. Finitions "
            "distinctives : paumelles invisibles, battement central design, parcloses "
            "arrondies, jonctions lisses des angles, jonc de finition coloré ou inox. "
            "Deux niveaux de finition extérieure : DROIT ou DESIGN. Garantie 15 ans. "
            "Structure métallique et quincaillerie anti-effraction."
        ),
        "alias_utilisateur": [
            "Textural", "TEXTURAL", "TEXTURAL®", "gamme Textural",
            "la déco", "fenêtre déco", "effet bois",
        ],
        "termes_documentaires": [
            "TEXTURAL", "base PVC sertie d'aluminium", "essences de bois",
            "paumelles invisibles", "battement central design",
        ],
        "fournisseurs": ["Proferm"],
        "discriminants": (
            "Comme HYBRIDE, c'est un mixte PVC/aluminium — la distinction est l'intention : "
            "TEXTURAL® vise la décoration intérieure (textures, essences), HYBRIDE "
            "l'écologie et le rapport qualité/prix. Ne pas transférer une finition ou une "
            "couleur de l'une à l'autre : les nuanciers diffèrent."
        ),
        "document_ids": [424, 425],
        "a_valider": (
            "Base PVC : quel système fournisseur exactement (Kömmerling 70 ? 76 ?) — non "
            "écrit dans les documents.\n"
            "Familles couvertes : la documentation ne parle que de fenêtres. Portes et "
            "coulissants existent-ils en TEXTURAL® ?"
        ),
    },
    {
        "slug": "lumine",
        "nom": "LUMINE",
        "accroche": "L'aluminium en mode sublime",
        "materiau": "aluminium",
        "familles": ["fenetres", "portes", "coulissants"],
        "ordre": 40,
        "description": (
            "Gamme aluminium de Proferm, bâtie sur les systèmes TECHNAL®. LUMINE65 "
            "(profil 65 mm avec joint central) couvre fenêtres et coulissants et permet "
            "un ouvrant véritablement caché, intégré dans le cadre, pour une surface "
            "vitrée maximale. Arguments : légèreté, durabilité, robustesse, "
            "personnalisation, confort thermique et acoustique."
        ),
        "alias_utilisateur": [
            "Lumine", "LUMINE", "Lumine 65", "Lumine65", "gamme Lumine",
            "l'alu", "aluminium", "menuiserie alu",
        ],
        "termes_documentaires": [
            "LUMINE65", "TECHNAL", "SOLEAL", "LUMEAL", "SOLEAL GY", "SOLEAL FY",
            "LUMEAL GA", "joint central", "ouvrant caché",
        ],
        "fournisseurs": ["Technal", "Proferm"],
        "discriminants": (
            "Univers ALUMINIUM/Technal, totalement disjoint du PVC profine. Une question "
            "Lumine ne doit JAMAIS mobiliser les DTD PVC (TROCAL / Kömmerling), et une "
            "question Perform ne doit jamais mobiliser les catalogues Technal.\n"
            "Les références Technal commencent par T : TGY (coulissant SOLEAL GY), "
            "TFY (frappe SOLEAL FY), TPY (SOLEAL PY). Une référence T??xxxx est une "
            "question aluminium — indice plus fiable que le nom de gamme."
        ),
        "document_ids": [429, 428, 432, 424, 425],
        "a_valider": (
            "Confirmer les préfixes : TFY = frappe, TGY = coulissant, TPY = porte.\n"
            "Askey est-il bien un second fournisseur aluminium à côté de Technal ?\n"
            "LUMINE65 est-il la seule déclinaison, ou coexiste-t-il avec d'autres "
            "systèmes Technal (LUMEAL GA, SOLEAL) sous la même gamme commerciale ?"
        ),
    },
    {
        "slug": "innoslide",
        "nom": "INNOSLIDE",
        "accroche": "La baie coulissante PVC à frappe",
        "materiau": "pvc",
        "familles": ["coulissants"],
        "ordre": 50,
        "description": (
            "Baie coulissante PVC à frappe, présentée comme une nouveauté. Arguments : "
            "confort, sécurité, hautes performances d'étanchéité, grandes dimensions. "
            "Le mécanisme « à frappe » la distingue d'un coulissant classique."
        ),
        "alias_utilisateur": [
            "Innoslide", "INNOSLIDE", "coulissant à frappe", "baie coulissante PVC",
        ],
        "termes_documentaires": ["INNOSLIDE", "coulissant à frappe", "baie coulissante"],
        "fournisseurs": ["Proferm", "profine"],
        "discriminants": (
            "Coulissant PVC : ne pas confondre avec le coulissant aluminium (Lumine / "
            "SOLEAL GY, préfixe TGY) ni avec le coulissant PVC e.XCLUSIVE de la Gamme 70."
        ),
        "document_ids": [427, 433],
        "a_valider": (
            "INNOSLIDE est ABSENT de la taxonomie (`PRODUCT_RANGE_CHOICES` ne contient que "
            "perform / lumine / hybride / textural) : gamme commerciale à part entière, ou "
            "produit rattaché à PERFORM ? En base, les documents Innoslide sont "
            "actuellement attribués à `perform`.\n"
            "Quel système fournisseur porte l'INNOSLIDE ?"
        ),
    },
]
