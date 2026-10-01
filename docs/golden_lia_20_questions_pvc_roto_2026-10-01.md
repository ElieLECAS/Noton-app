# Golden LIA — série 2 : PVC profine / KÖMMERLING, système 70 et 76, ROTO (01/10/2026)

Suite de `docs/golden_lia_20_questions_2026-10-01.md` (Q1 à Q20). Ces 20 questions (**Q21 à Q40**) portent
uniquement sur les domaines complets du wiki : **système 76 / PERFORM76, système 70 / PERFORM70 et
HYBRIDE, documentation profine et KÖMMERLING, ferrures ROTO NX**, avec les garanties, labels et gammes
PVC qui les entourent. Technal, Askey et Soprofen sont laissés de côté tant que leurs sources ne sont
pas retraitées.

Chaque réponse attendue a été **lue dans le wiki actuel (323 pages) avec fichier et ligne**, puis
recherchée dans les autres pages pour voir si une autre valeur la contredit. Les numéros de ligne sont
ceux du 01/10/2026 : ils bougeront si une page est redécoupée. Elles valent **pour le wiki**, pas pour
les PDF d'origine.

## Ce que cette série mesure

Votre critère : **un « je ne sais pas » vaut mieux qu'une erreur grave**. Chaque question porte donc
trois niveaux de réponse :

| Niveau | Points | Définition |
| --- | :---: | --- |
| **Juste** | 2 | Contient l'**Attendu** : valeurs exactes, conditions, anomalie avec ses deux valeurs, pages lues et citées |
| **Repli acceptable** | 1 | Prudente et non fautive : « sur étude », « le wiki ne le donne pas », une borne exacte sans conclure |
| **Faute grave** | −2 | Une des réponses listées en **Faute grave** : valeur voisine, « oui » à tort, prémisse validée, valeur ou référence inventée, contradiction réduite à une seule valeur |

Une réponse correcte mais incomplète (une valeur juste, une nuance manquante) vaut 1. Une absence
complétée par une valeur plausible est toujours une faute grave, même exprimée avec prudence.

La série contient : 4 faisabilités (Q21, Q26, Q27, Q31), 5 absences (Q23, Q29, Q34, Q36, Q37),
5 contradictions ou divergences (Q22, Q30, Q33, Q35, Q40), 4 fausses prémisses (Q24, Q25, Q30, Q38),
3 pièges entre systèmes ou entre pièces jumelles (Q28, Q32, Q39) et 2 questions qui croisent deux
domaines (Q25, Q39).

## Comment la passer

1. **Une conversation par question.** Le seul cas de suivi du premier golden (Q3→Q4, Q11→Q12) n'a pas
   d'équivalent ici : les 20 questions sont indépendantes. Les enchaîner dans une même conversation
   contamine les réponses (la leçon de la série 1).
2. Poser la question mot pour mot. Relever : réponse, pages chargées, nombre de recherches, tokens.
3. Classer selon le tableau ci-dessus. Pour une contradiction, vérifier que les **deux** valeurs et leur
   **document** figurent, pas seulement l'identifiant.
4. Pour une absence, vérifier qu'**aucune** valeur de remplacement n'est avancée comme réponse.

## Réserves à connaître avant de compter

| Q | Réserve |
| --- | --- |
| Q21 | La limite oblique est lue sur un diagramme (±25 mm). La question est choisie à plus de 100 mm de la limite et recoupée par la planche des positions de compas. |
| Q22 | Dans `roto-nx-apercu-ferrures-cote-p.md`, les renvois [1] et [2] sont inversés d'une page à l'autre. L'attribution retenue est celle du registre CTR-18. |
| Q25 | Le tableau 76 parle d'une « ferrure 130 kg » sans la nommer. L'équivalence avec la Roto NX 130 kg est plausible mais n'est écrite nulle part. |
| Q26 | Le verdict dépend du dormant (fixé à 76171). Avec un 76172, la réponse devient « sur étude ». La page des abaques est en `status: draft`. |
| Q27 | Le DTA parle de dimensions de baie. L'écart entre l'essai à 2,25 m et le maximum de 2,15 m n'est enregistré dans aucun registre. |
| Q28 | Même page que « doublage 175 mm » (série 1), mais autre épaisseur et piège inverse. À remplacer si trop proche. |
| Q29 | « Trois vantaux » est ambigu pour un commercial ; la question précise « ouvrants ». |
| Q30 | CTR-01 n'est pas tranchée par le wiki : la bonne réponse est « 100 kg retenu, contradiction ouverte ». |
| Q31 | Le total 10,8 cm⁴ est une somme faite par l'auteur : le wiki donne les deux lectures et la règle d'addition, pas le total. |
| Q32 | Le DTA 76 admet aussi Ø 8 : ne pas pénaliser une réponse qui le cite pour le 76. |
| Q35 | La divergence 2008 / 2023 sous 10 °C n'est enregistrée dans aucun registre. |
| Q39 | Touche au RC2, déjà traité en série 1 ; l'angle est ici la HYBRIDE et le Label Roto. |
| Q40 | « 10 ans » est retenu par le registre parce que c'est la source la plus récente ; l'explication est une hypothèse du wiki. |

## Défauts du wiki repérés en écrivant ces questions

À corriger dans les pages, pas dans les consignes :

- `roto-nx-apercu-ferrures-cote-p.md` : renvois [1] et [2] inversés (voir Q22).
- `roto-nx-ksr-soufflet.md:36` donne une LFF mini de 238 mm pour le soufflet, contre 310 mm au diagramme : divergence non enregistrée (Q21).
- `dta-6-16-2334.md` : essai de perméabilité à 2,25 m contre dimension maximale de 2,15 m, sans anomalie (Q27).
- `systeme-76-abaques-dimensionnels.md` : acier « long (5 mm) » incohérent (VER-25, déjà enregistrée) (Q25).
- Collage sous 10 °C : chiffre en minutes (2008) contre coefficient (2023), sans anomalie (Q35).

---

## Série 2A — Ferrures ROTO NX (Q21 à Q25)

Sigles : LFF = largeur fond de feuillure, HFF = hauteur fond de feuillure, « manuel » = instructions de montage Roto NX KSR (nov. 2022), « catalogue » = catalogue Roto NX PVC (juin 2023). Chemins relatifs à `wiki_llm/wiki/`.

### Q21 — Soufflet de 2 200 × 1 000 : dans la plage ou pas ?
> « Un vantail soufflet de 2 200 de large sur 1 000 de haut avec de la Roto NX, vitrage 4/16/4, ça passe ? »

**Type** : faisabilité
**Attendu** : **Non.** Les bornes chiffrées du soufflet côté paumelles P (LFF 310 – 2 400 mm, HFF 290 – 1 200 mm, 80 kg max) donneraient l'impression d'un « oui », mais le diagramme d'application exclut le coin haut droit : au-dessus de la **limite oblique**, qui part de (LFF 1 200, HFF 1 200) et descend jusqu'à environ 800 mm de HFF à LFF 2 400 (relevé à ±25 mm), le « champ d'application non autorisé » s'applique. À LFF 2 200 la HFF admissible est d'environ 870 mm (segment droit), donc 1 000 mm est hors champ ; la planche des positions de compas, lue aux graduations imprimées, le confirme : à HFF 1 000 le vantail n'est dessiné que pour LFF 1 200 à 1 800 (à HFF 900 jusqu'à 2 100, de HFF 600 à 800 jusqu'à 2 400). Un 2 200 × 800 passerait. Côté paumelles Designo II c'est aussi non (soufflet LFF 450 – 1 400 mm, HFF 370 – 1 200 mm, 80 kg), et le côté employé par PROFERM n'est pas connu (VER-34) : la réponse est « non » dans les deux cas. Le poids n'est pas en cause (4/16/4 = 8 mm de verre × 2,5 = 20 kg/m², soit environ 44 kg pour 2,2 m²). LFF/HFF sont des cotes de fond de feuillure du vantail, pas des cotes de baie.
**Repli acceptable** : « hors du champ d'application (limite oblique du diagramme soufflet) ; réduire la hauteur (≈ 800 mm ou moins à 2 200 de large) ou la largeur (≤ 1 800 mm à 1 000 de haut), sinon sur étude / voir Roto ». Dire « sur étude » sans citer la limite oblique est acceptable mais pauvre.
**Faute grave** : « Oui, 2 200 est inférieur à 2 400 et 1 000 inférieur à 1 200 » (lire les seules bornes du tableau) ; valider avec « 2 compas d'arrêt en haut » (équipement possible pour le rectangle, pas pour la zone interdite) ; prendre les bornes du soufflet Designo ou de l'OB (HFF 2 800) ; inventer une courbe de poids de vitrage (il n'y en a pas sur ce diagramme).
**À signaler** : VER-34 (côté paumelles P ou Designo II employé par PROFERM : non connu). Facultatif : INC-192 (légende [A] imprimée deux fois sur ce diagramme).
**Pages** : `/quincaillerie/roto-nx-champs-application.md` (page de référence), `/quincaillerie/roto-nx-ksr-soufflet.md` (Designo), `/quincaillerie/roto-nx-apercu-ferrures-cote-p.md`.
**Preuve** :
- `quincaillerie/roto-nx-champs-application.md:66` : « Soufflet, fenêtre rectangulaire | Sécurité de base | 310 | 2400 | 290 | 1200 | 80 » (LFF 310-2400, HFF 290-1200, 80 kg).
- `quincaillerie/roto-nx-champs-application.md:365` : « au-dessus de la limite oblique | 1200 à 2400 | de la limite oblique à 1200 | champ d'application non autorisé ».
- `quincaillerie/roto-nx-champs-application.md:371-372` : « La limite oblique est un segment droit qui part du coin haut (LFF 1 200, HFF 1 200) et descend jusqu'à environ 800 mm de HFF au bord droit (LFF 2 400), relevé à ±25 mm. » Même lecture pour le catalogue (p. 42), lignes 381-383.
- `quincaillerie/roto-nx-champs-application.md:416-418` (planche des positions, HTML) : HFF 1000 → « LFF 1200 à 1800 » ; HFF 900 → « LFF 1200 à 2100 » ; HFF 600 à 800 → « LFF 1200 à 2400 ». Ces points sont alignés sur la droite (1 200 × 1 200) – (2 400 × 800).
- `quincaillerie/roto-nx-champs-application.md:817-819` et `quincaillerie/roto-nx-ksr-soufflet.md:306-308` : Designo, soufflet « LFF 450 à 1400 mm, HFF 370 à 1200 mm, 80 kg ».
- `anomalies/informations-a-verifier.md:197` : VER-34.
- Autre lecture du wiki : aucune page ne donne un autre champ pour le soufflet P rectangulaire (le manuel et le catalogue concordent : tableau p. 26 / p. 42).
**Réserve** : la limite oblique est lue sur un diagramme (« environ 800 mm », ±25 mm) et la valeur de 870 mm à LFF 2 200 est obtenue en prolongeant le segment droit décrit par le wiki ; la règle « jamais d'interpolation » vaut pour les courbes, ici le wiki décrit un segment droit. La question est choisie à plus de 100 mm de la limite pour que la lecture ne dépende pas des ±25 mm, et la planche des positions (lecture aux graduations) corrobore. Divergence annexe non enregistrée : la page de configuration du manuel donne une LFF mini de 238 mm pour le soufflet (`roto-nx-ksr-soufflet.md:36`) contre 310 mm au diagramme ; sans effet ici.

### Q22 — Oscillo-battant de 1 500 de large en RC1 N
> « Une fenêtre oscillo-battante Roto NX de 1 500 de large sur 1 400 de haut, en classe RC1 N : on est dans le champ d'application ? »

**Type** : contradiction
**Attendu** : **Pas de réponse unique : les deux documents ROTO ne disent pas la même chose (CTR-18, non arbitré).** Pour l'oscillo-battant rectangulaire, côté paumelles P, classe CDR 1 N (le catalogue écrit « RC 1 N ») : **manuel de montage KSR (nov. 2022) : LFF 320 – 1 400 mm** (HFF 290 – 2 600 mm) → 1 500 est **hors champ** ; **catalogue Roto NX PVC (juin 2023) : LFF 320 – 1 600 mm** (HFF 280 – 2 800 mm) → 1 500 est **dans le champ**. Le wiki retient pour un chiffrage « la borne la plus basse des deux documents », soit 1 400 mm : ne pas accepter 1 500 en RC1 N sans confirmation de ROTO. Compléments : en sécurité de base les deux documents donnent LFF 290 – 1 600 mm (1 500 passe) ; dans les deux diagrammes la zone « 2ᵉ compas nécessaire » commence à LFF 1 400 mm ; la HFF de 1 400 mm est dans la plage dans les deux cas.
**Repli acceptable** : « selon le manuel non (max 1 400), selon le catalogue oui (max 1 600) ; contradiction non tranchée, à faire confirmer par ROTO ou le service technique ; prudence : traiter comme non / sur étude ». Toute réponse qui expose les deux bornes avec leur document est juste, qu'elle conclue « non » ou « sur étude ».
**Faute grave** : « Oui, jusqu'à 1 600 » ou « Non, max 1 400 » donnés comme valeur unique sans signaler l'autre document ; confondre avec les bornes CDR 2 / CDR 2 N (LFF 320 – 1 400 dans les deux documents) ou CDR 3 (catalogue seul) ; citer des bornes Designo (LFF 450/600 – 1 400) ; oublier le 2ᵉ compas au-delà de 1 400.
**À signaler** : **CTR-18** (champs d'application P, OB rectangulaire : HFF mini 290/280, LFF maxi CDR 1 N 1 400/1 600, HFF maxi CDR 1 N 2 600/2 800, HFF maxi CDR 2 2 400/2 800). Connexe : CTR-110 (HFF mini 280/290/300 selon la page).
**Pages** : `/quincaillerie/roto-nx-champs-application.md`, `/quincaillerie/roto-nx-apercu-ferrures-cote-p.md`, `/anomalies/contradictions-entre-sources.md`.
**Preuve** :
- `anomalies/contradictions-entre-sources.md:131` (CTR-18) : manuel « CDR 1 N jusqu'à 1 400 mm de LFF et 2 600 mm de HFF, CDR 2 jusqu'à 2 400 mm de HFF » ; catalogue « CDR 1 N jusqu'à 1 600 mm de LFF et 2 800 mm de HFF, CDR 2 jusqu'à 2 800 mm de HFF » ; valeur retenue : « Les bornes les plus basses des deux documents ».
- `quincaillerie/roto-nx-champs-application.md:60` : « version 130 kg | CDR 1 N | 320 | 1400 | 290 | 2600 | 130 » (manuel).
- `quincaillerie/roto-nx-champs-application.md:833` et `:837` : « 130 kg | CDR 1 N | 320 | 1600 | 280 | 2800 | 130 » et « 150 kg | CDR 1 N | 320 | 1600 | 280 | 2800 | 150 » (catalogue).
- `quincaillerie/roto-nx-champs-application.md:851-855` : « Les deux documents ROTO ne donnent pas les mêmes bornes … la valeur retenue pour un chiffrage est la borne la plus basse des deux documents ».
- `quincaillerie/roto-nx-apercu-ferrures-cote-p.md:179-187` : configuration « Ferrure OB, CDR 1 N » du catalogue : LFF 320 – 1 600, HFF 280 – 2 800, max 150 kg ; ligne 187 rappelle les deux jeux de bornes et CTR-18.
- 2ᵉ compas : `quincaillerie/roto-nx-champs-application.md:129-131` (130 kg) et `:175-177` (150 kg) : « 1400 … 2ᵉ compas nécessaire ».
- Sécurité de base 1 600 dans les deux : `quincaillerie/roto-nx-champs-application.md:59` et `:832`.
**Réserve** : dans `roto-nx-apercu-ferrures-cote-p.md` les crochets [1]/[2] sont inversés d'une page à l'autre (en :187 « [1] » désigne le catalogue, alors que `roto-nx-champs-application.md` appelle [1] le manuel) ; l'attribution à retenir est celle du registre CTR-18 (A = manuel, B = catalogue). Le terme « RC1 N » de la question est celui du catalogue/brochure ; les tableaux français disent « CDR 1 N » (INC-13).

### Q23 — Couple de serrage des vis du palier de compas
> « Quel couple de serrage je dois appliquer sur les vis du palier de compas d'une ferrure Roto NX ? »

**Type** : absence
**Attendu** : **Le wiki ne donne aucune valeur de couple de serrage des vis de la ferrure Roto NX.** Il donne seulement la consigne : « Ne pas trop serrer les vis. Respecter les couples de serrage. Sélectionner le couple de manière à ne pas déformer la ferrure et le profil. Déterminer le couple spécifique au profil par essais de butées », les vis à utiliser (acier galvanisé passivé, Ø 3,9 – 4,2 mm ; vis inox pour les composants inox ; vis acier zinc-nickel ou inox pour l'aluminium) et, côté résistance de la fixation, le tableau des forces de traction TBDK par poids de vantail (60 kg → 1 650 N … 150 kg → 4 200 N), qui est une force et non un couple. À ne pas confondre : le seul « Nm » du chapitre Roto NX est le **couple de verrouillage/déverrouillage de la manœuvre, max. 10 Nm** (contrôle fonctionnel de maintenance), qui ne concerne pas les vis.
**Repli acceptable** : « le wiki ne donne pas le couple de serrage ; la source demande de le déterminer par essais de butées, spécifique au profil ; voir ROTO / le fabricant de profils ». Citer 10 Nm à condition de préciser qu'il s'agit du couple de manœuvre est acceptable.
**Faute grave** : annoncer « 10 Nm » (ou toute autre valeur) comme couple de serrage des vis ; reprendre les valeurs d'autres produits (5 N·m vis de boîtier SOLEAL FY ; 10 à 15 N.m paumelles SOLEAL PY ; 20 Nm ASKEY 65 NV ; 10 Nm assemblage meneau système 70) ; convertir les forces de traction TBDK en couple ; inventer un couple « typique » de vis à bois.
**À signaler** : — (aucune entrée de registre sur ce point ; l'absence est celle de la source, déjà signalée dans le corps de la page).
**Pages** : `/procedures/percage-montage-roto-nx.md` (§ 8.2-8.3 vissage), `/procedures/maintenance-ferrure-roto-nx.md`, `/quincaillerie/roto-nx-champs-application.md` (forces de traction).
**Preuve** :
- `procedures/percage-montage-roto-nx.md:447-449` : « Ne pas trop serrer les vis. Respecter les couples de serrage. Sélectionner le couple de manière à ne pas déformer la ferrure et le profil. Déterminer le couple spécifique au profil par essais de butées. » ; `:399-402` (§ 8.2) : vis « Ø 3,9 – 4,2 x … » sans couple.
- `procedures/maintenance-ferrure-roto-nx.md:185` : « Couple de verrouillage et de déverrouillage : max. 10 Nm » (couple de manœuvre).
- `quincaillerie/roto-nx-champs-application.md:869-880` : forces de traction en N (60 kg → 1 650 N … 150 kg → 4 200 N).
- `procedures/roto-nx-ksr-consignes-generales.md:234` : « Nm | Couple en newton.mètres » (abréviation seule, pas de valeur).
- Recherches nulles (grep sur tout `wiki_llm/wiki`, hors `log.md`) : `[0-9] ?(N ?\.? ?m|Nm|N·m|daNm)` → 4 produits seulement, aucun Roto NX hors le 10 Nm de la maintenance (systeme-70 assemblage meneau 10 Nm, askey-coulissant-65-nv 20 Nm, soleal-fy 5 N·m, soleal-py 10 à 15 N.m / 18-29 N.m, lumeal-ga 100 daN = charge) ; `couples? de (serrage|vissage)` → askey-65-nv (« couple de serrage contrôlé », sans valeur), soleal-fy (5 N·m), percage-montage-roto-nx (consigne sans valeur), askey-frappe (« La source ne fournit pas le couple … ») ; dans les fichiers Roto (`quincaillerie/roto*`, `procedures/*roto*`, `report-de-charge`, `transformation-of-en-ob`, `certifications/roto*`, `commercial/roto*`, `sources/roto*`) : `dynamom` (1 occurrence, maintenance :186, pour le contrôle des 10 Nm), `daNm`, `N·m`, `N m`, `vis de palier`, `résistance à l'arrachement` → aucun résultat ; `serrer` → seulement des termes de catalogue (« A serrer » = type de montage des caches) et « Serrer à fond ou remplacer » (maintenance :196, sans valeur).
**Réserve** : — (la preuve d'absence porte sur le wiki, seule source de vérité de ce test ; ne prouve pas que le PDF source n'a pas de valeur).

### Q24 — Le certificat à 300 kg
> « Le certificat ift de la Roto NX parle de 300 kg de poids d'ouvrant. Je peux donc faire un oscillo-battant de 200 kg avec ? »

**Type** : fausse prémisse
**Attendu** : **Non.** Les 300 kg sont la valeur imprimée sur le certificat ift QM 328 (« max. Flügelgewicht / max. casement weight 300 kg », champ d'emploi « systèmes avec rainure de ferrure adaptée ») : c'est le poids maximal de la famille de produits certifiée, pas un champ d'application d'une configuration. Les poids admissibles réels sont ceux des champs d'application, et aucun ne dépasse **150 kg** : côté P, oscillo-battant rectangulaire 130 kg ou 150 kg selon la version (et, selon la page lue, ouvrant à la française, battement, etc., 150 kg max) ; côté Designo II, 80 ou 100 kg sans report de charge, 150 kg au maximum avec report de charge (LFF ≥ 800 mm, HFF ≥ 1 000 mm) ; la table des forces de traction TBDK s'arrête à 150 kg (4 200 N). Un vantail de 200 kg est au-delà de tous les champs d'application Roto NX OB du wiki, et la règle du manuel est « ne jamais dépasser les tailles et poids d'ouvrant maximum admissibles ».
**Repli acceptable** : « le wiki ne donne aucun champ d'application Roto NX au-delà de 150 kg ; 200 kg = hors champ / à soumettre à ROTO ». Mentionner les 300 kg du certificat en précisant qu'ils ne valent pas autorisation est correct.
**Faute grave** : « Oui, jusqu'à 300 kg » ; « 200 kg possible avec report de charge » (le report de charge plafonne à 150 kg) ; confondre 300 kg et la valeur d'un champ d'application ; donner 130 kg ou 100 kg comme seule limite sans distinguer P / Designo.
**À signaler** : VER-34 (côté paumelles P ou Designo II non établi : le plafond dépend du côté). Facultatif : CTR-110 (130 ou 150 kg selon la page pour la même configuration P).
**Pages** : `/certifications/roto-nx-certificats.md`, `/quincaillerie/roto-nx-champs-application.md`, `/quincaillerie/roto-nx.md`, `/procedures/percage-montage-roto-nx.md`.
**Preuve** :
- `certifications/roto-nx-certificats.md:49-50` : « max. Flügelgewicht / max. casement weight | 300 kg | poids d'ouvrant maximal » ; « Einsatzbereich / field of application | Systeme mit entsprechender Beschlagaufnahmenut … ».
- `certifications/roto-nx-certificats.md:88-89` : « Le poids d'ouvrant maximal de 300 kg est celui du certificat ; les poids admissibles de chaque configuration sont sur Champs d'application Roto NX. »
- `quincaillerie/roto-nx-champs-application.md:59-64` (P, OB : 130 et 150 kg) ; `:679` (Designo, report de charge : 150 kg) ; `:677-678` (Designo sans report : 80 et 100 kg) ; `:880` (« 150 | 4 200 », dernière ligne de la table TBDK).
- `quincaillerie/roto-nx.md:110` : « charges élevées jusqu'à 150 kg : P reposant côté paumelles ».
- `procedures/percage-montage-roto-nx.md:356-357` et `:362` : « L'élément ayant la capacité de charge la plus faible admissible détermine … le poids maximal admissible de l'ouvrant » ; « Ne jamais dépasser les tailles et poids d'ouvrant maximum admissibles ».
- Recherche d'une autre valeur : grep `\b(1[6-9][0-9]|[2-9][0-9]{2}) ?kg` sur `quincaillerie/roto-nx*`, `procedures/*roto*`, `report-de-charge`, `transformation-of-en-ob`, `commercial/roto-nx`, `certifications/roto*` → seul résultat : le certificat (300 kg).
**Réserve** : — (le wiki ne dit pas en toutes lettres « 200 kg interdit » : la conclusion s'appuie sur l'absence de tout champ d'application au-delà de 150 kg et sur la règle « ne jamais dépasser »).

### Q25 — Roto NX 130 kg sur un ouvrant système 76 sans acier
> « Sur un ouvrant du système 76, je veux poser un oscillo-battant Roto NX version 130 kg avec un vantail de 115 kg, sans renfort acier dans le dormant. C'est bon ? »

**Type** : cross-sujet (ferrure Roto × profilé 76)
**Attendu** : **Non en l'état : deux plafonds se superposent et le plus bas gagne** (« l'élément ayant la capacité de charge la plus faible détermine le poids maximal admissible de l'ouvrant »). Côté Roto NX, la version 130 kg admet 130 kg au plus (sous réserve des LFF/HFF du champ d'application) : 115 kg est dans la limite de la ferrure. Côté profilé, le tableau « Poids d'ouvrant admissible selon la ferrure » du système 76 Advanced (profine, essais ift selon TBDK, valeurs indicatives) donne pour une « ferrure 130 kg » : **sans acier dans le dormant (6 vis dans le PVC) : 100 kg** ; acier « court (55 mm) » (2 vis acier + 4 PVC) : **110 kg** ; acier « long » (5 vis acier + 1 PVC) : **130 kg**. Donc 115 kg dépasse 100 kg sans renfort, et dépasse encore 110 kg avec l'acier court ; seul l'acier « long » couvrirait 115 kg, mais sa longueur est imprimée « long (5 mm) », incohérente (VER-25), à faire confirmer. Le fabricant doit de toute façon contrôler et garantir les poids selon la directive TBDK. Ce que le wiki ne dit pas : que la « ferrure 130 kg » du tableau profine soit la Roto NX (simple renvoi « Voir Roto NX »), ni que la PERFORM76 soit équipée de Roto NX (la page ROTO relie la Roto NX à PERFORM+ et HYBRIDE+).
**Repli acceptable** : « le plafond du profilé (100 kg sans acier pour une ferrure 130 kg) est inférieur à 115 kg ; ajouter l'acier / réduire le poids ; valider avec le service technique (VER-25, TBDK) ». Une réponse qui annonce « 100 kg max sans acier » en précisant que c'est le tableau du système 76 et non une limite Roto est juste.
**Faute grave** : « Oui, la Roto NX 130 kg porte 115 kg » en ignorant le tableau 76 ; attribuer les 100 kg à la Roto NX ; dire que l'acier court (110 kg) suffit ; affirmer que « long (5 mm) » est établi ou inventer une longueur d'acier ; affirmer que la PERFORM76 est équipée de Roto NX ; utiliser la version 150 kg (sans rapport avec la table 76 « 100 / 130 kg »).
**À signaler** : VER-25 (longueur de l'acier « long » : « long (5 mm) »). Facultatif : VER-34 (côté paumelles non établi ; ici version 130 kg = côté P).
**Pages** : `/profiles/systeme-76-abaques-dimensionnels.md`, `/quincaillerie/roto-nx-champs-application.md`, `/procedures/percage-montage-roto-nx.md`, `/fournisseurs/roto.md`.
**Preuve** :
- `profiles/systeme-76-abaques-dimensionnels.md:915-917` : « 130 kg | sans | - | 6 | 100 », « 130 kg | court (55 mm) | 2 | 4 | 110 », « 130 kg | long (5 mm) | 5 | 1 | 130 » ; `:903-907` : « valeurs indicatives … valables uniquement pour les composants indiqués dans les rapports d'essais » ; `:927-930` : « Les poids d'ouvrants maximaux doivent être contrôlés et garantis par le fabricant de fenêtres ! » puis `:931` « Voir Roto NX ».
- `anomalies/informations-a-verifier.md:189` (VER-25) : « la seconde valeur ne peut pas être une longueur en regard de la première … c'est la ligne qui fait passer une ferrure de 80 à 100 kg admissibles ».
- `quincaillerie/roto-nx-champs-application.md:59-61` : version 130 kg, PV maxi 130.
- `procedures/percage-montage-roto-nx.md:356-357` : « L'élément ayant la capacité de charge la plus faible admissible détermine … le poids maximal admissible de l'ouvrant ».
- `fournisseurs/roto.md` (tableau « Ce que PROFERM lui achète », 1ʳᵉ ligne) : Roto NX ↔ PERFORM+ et HYBRIDE+ ; aucune ligne PERFORM76 ; `quincaillerie/perform76-poignee-et-pivot.md` (« ne traite aucune autre pièce de quincaillerie ») ne cite pas Roto NX pour les fenêtres.
- TBDK : `quincaillerie/roto-nx-champs-application.md:876-877` : 110 kg → 3 000 N ; 120 kg → 3 250 N (aucune ligne à 115 kg).
**Réserve** : le tableau 76 libelle la ferrure « 100 kg » / « 130 kg » sans la nommer ; l'équivalence avec la version 130 kg de la Roto NX est plausible mais non écrite dans le wiki. L'attendu est donc formulé sous cette condition (« ferrure de classe 130 kg »), et une réponse qui le dit explicitement est la meilleure. Table datée janvier 2016 (registre 2.3.3, p. 5). Question non testable avec certitude côté dimensions : aucune LFF/HFF n'est donnée, donc la conformité au champ d'application Roto reste à vérifier.

---

## Série 2B — PERFORM76 et système 76 Advanced (Q26 à Q30)

### Q26 — Triple vitrage sur un oscillo-battant : la courbe d'épaisseur de verre décide

> « Fenêtre PERFORM76 oscillo-battante un vantail, blanche, 1 300 de large sur 1 450 de haut, dormant 76171 et ouvrant 76281, avec un triple vitrage 4/12/4/12/4 : on peut la faire ? »

**Type** : faisabilité
**Attendu** : **Non en l'état.** Enchaînement des limites :
1. DTA 2.2.3.7, 1 vantail OB : H maxi 1,50 m × L maxi 1,40 m (ou 2,15 × 1,00). Ici 1,45 × 1,30 : dans la première ligne. OK.
2. Cotes de débit : dormant 76171, a = 38 mm. Vantail (DEO) = 1 300 − 2 × 38 = 1 224 mm de large, 1 450 − 2 × 38 = 1 374 mm de haut, soit **122,4 × 137,4 cm**.
3. Abaque d'ouvrant 76281 / 76275 (renfort V266.Z), limite blanc : 235 cm de haut jusqu'à 95 cm de large, oblique jusqu'au coin (130 ; 130), largeur maxi 130 cm. À 122,4 cm de large la hauteur admise est d'environ 153 cm : OK. Règle des 25 % : 122,4 ≤ 1,25 × 137,4 : OK.
4. Épaisseur de verre : on additionne les couches de verre sans les intercalaires, donc 4 + 4 + 4 = **12 mm** (le wiki donne exactement cet exemple). Courbe 12 mm : **132 cm de haut à 120 cm de large, 112 cm à 130 cm**. Le vantail fait 137,4 cm de haut : il dépasse la courbe aux deux graduations, donc hors limite (sans interpolation, la plus restrictive est 112 cm ; même en interpolant on trouve environ 127 cm, toujours en dessous de 137,4).
5. Ferrure Roto NX OB version 130 kg, sécurité de base : LFF 290–1 600, HFF 290–2 800 mm, 130 kg. Le vantail passe : cette limite n'est pas la plus restrictive.

La limite décisive est donc la **courbe d'épaisseur de verre 12 mm de l'abaque 76281**. Remèdes écrits dans le wiki : équerres de feuillure **J079 dans les 4 coins** (les limitations de vitrage sont décalées de 2 courbes : 12 mm se lit comme moins de 12 mm, plus de restriction), ou un double vitrage (par exemple 4/16/4 = 8 mm de verre, aucune restriction sous 12 mm).
**Repli acceptable** : « pas confirmable / sur étude : la courbe de verre de 12 mm limite l'ouvrant 76281 à 112–132 cm de haut à cette largeur ; avec J079 la restriction est levée selon l'abaque ». Un « non, sauf J079 » est aussi correct.
**Faute grave** : répondre « oui, ça passe » après avoir vérifié seulement le DTA (1,50 × 1,40), la limite de couleur blanc (235 cm) et/ou la ferrure Roto ; lire 36 mm ou 24 mm au lieu de 12 mm de verre ; ne pas regarder la courbe de verre du tout.
**À signaler** : — (aucune anomalie ne porte sur ces valeurs ; CTR-18 ne jouerait que si LIA citait les bornes Roto en CDR, ce qui n'est pas le sujet).
**Pages** : `/certifications/dta-6-16-2334.md`, `/profiles/systeme-76-cotes-de-debit.md`, `/profiles/systeme-76-abaques-dimensionnels.md` (contrôle secondaire : `/quincaillerie/roto-nx-champs-application.md`).
**Preuve** :
- `certifications/dta-6-16-2334.md:562-563` : « | 1 vantail OB | 1,50 | 1,40 | » et « | 1 vantail OB | 2,15 | 1,00 | ».
- `profiles/systeme-76-cotes-de-debit.md:101` : « | 76171 | 38 | 58 | 59 | 45 | 40 | 71 | 46 | » (DEO = 38) ; `:47` « Un dormant a deux montants : une largeur hors tout perd la valeur du montant gauche et celle du montant droit ».
- `profiles/systeme-76-abaques-dimensionnels.md:121-122` : « un vitrage 4-12-4-12-4 donne une épaisseur totale de 4+4+4 = 12 mm » ; `:289` « | 120 | 132 | 101 | 83 | 71 | 62 | » et `:290` « | 130 | 112 | 86 | 71 | 61 | 54 | » (colonne verre 12 mm) ; `:272` « | blanc | 235 | 95 | 130 | 130 | 130 | 104 | » ; `:130` règle des 25 % ; `:104-110` J079, « décalées de 2 courbes ».
- `quincaillerie/roto-nx-champs-application.md:59` : OB rectangulaire 130 kg, sécurité de base : 290 / 1600 / 290 / 2800 / 130.
**Réserve** : le résultat dépend du dormant, qui est donc fixé dans la question. Avec le 76172 (a = 56) le vantail fait 118,8 × 133,8 cm : la courbe donne 132 cm à 120 cm, on tombe à moins de 3 cm (précision de lecture) → « sur étude ». Avec un dormant rénovation (76177 / 76185, a = 15) le vantail fait 127 × 142 cm : hors. Le contrôle Roto suppose LFF/HFF = DFO profine (hypothèse du code, aucune source ne l'écrit) mais il n'est pas décisif. La page des abaques est en `status: draft`.

### Q27 — Porte-fenêtre à 2 vantaux, 1 600 × 2 250 : l'essai à 2,25 m n'est pas une limite

> « Porte-fenêtre PERFORM76 à deux vantaux à la française, 1 600 de large sur 2 250 de haut : c'est dans le DTA ? »

**Type** : faisabilité
**Attendu** : **Non.** DTA 2.2.3.7, fabrications non certifiées : « 2 vantaux OF » **H maxi 2,15 m, L maxi 1,60 m**. La largeur (1,60) est juste à la limite, mais la hauteur de 2,25 m dépasse de 10 cm. Même avec un fixe latéral (2 vantaux OF + fixe latéral, 2,15 × 2,40) la hauteur maxi reste 2,15 m. Seule voie documentée : une fabrication certifiée, dont les dimensions supérieures « sont alors précisées dans le Certificat de Qualification attribué au menuisier » (donc à confirmer avec le certificat, pas déductible du wiki). Le DTA cite bien un essai de perméabilité à l'air sur une « fenêtre 2 vantaux, 1,60 × 2,25 m » : c'est un justificatif d'essai (sous gradient thermique), pas un domaine d'emploi.
**Repli acceptable** : « hors des dimensions du DTA (2,15 m de haut maxi) ; au-delà seulement selon le Certificat de Qualification du menuisier / sur étude ».
**Faute grave** : « oui » en s'appuyant sur l'essai 1,60 × 2,25 m ou sur la hauteur d'ouvrant de 235 cm des abaques ; inverser hauteur et largeur du tableau ; dire que la limite est 2,40 m (valeur du cas avec fixe latéral, sur la largeur).
**À signaler** : —
**Pages** : `/certifications/dta-6-16-2334.md` (aussi la copie du tableau dans `/gammes/perform.md`, section Dimensions limites).
**Preuve** :
- `certifications/dta-6-16-2334.md:564` « | 2 vantaux OF | 2,15 | 1,60 | » ; `:565` « | 2 vantaux OF + fixe latéral | 2,15 | 2,40 | » ; `:554-557` (colonnes « H maxi (m) » puis « L maxi (m) », fabrications non certifiées) ; `:571-572` « Pour les fabrications certifiées, des dimensions supérieures peuvent être envisagées ; elles sont alors précisées dans le Certificat de Qualification ».
- `certifications/dta-6-16-2334.md:816` : « perméabilité à l'air sous gradient thermique | fenêtre 2 vantaux, 1,60 × 2,25 m | RE CSTB n° BV16-1292 » (le piège).
- `gammes/perform.md:195-196` : mêmes valeurs (2,15 / 1,60 et 2,15 / 2,40) ; aucune autre page ne donne une hauteur maxi de 2,25 m pour le 76.
**Réserve** : le DTA parle de dimensions « de baie » et la question de dimensions de menuiserie ; l'écart de 10 cm sur la hauteur est plus grand que le jeu de pose, donc le verdict ne change pas. L'écart entre l'essai (2,25 m) et le maxi (2,15 m) n'est pas enregistré comme anomalie dans le wiki (je ne l'ai pas créé, lecture seule).

### Q28 — Tapée 6141 et patte NT1947 : 140 mm sur un 76180, 155 mm sur un 76171

> « Je pose une PERFORM76 sur un dormant 76171 avec 140 mm de doublage : je prends la tapée 6141 avec la patte NT1947 ? »

**Type** : lecture de tableau (piège de jumelles : même tapée et même patte, isolation différente selon le dormant)
**Attendu** : **Non, pas pour 140 mm.** La tapée 6141 (cote propre 75 mm) avec la patte NT1947 donne **140 mm d'isolant sur un dormant 76180** (et la tapée 6141 donne aussi 140 mm sur 76177 / 76185), mais **155 mm sur un dormant 76171** (+15 mm). Sur le 76171 les seules épaisseurs d'isolation qui existent sont 80, 95, 115, 135, 155, 175, 195 et 215 mm ; « aucune épaisseur intermédiaire n'existe ». Une demande à 140 mm sur ce dormant « se traite en 135 ou en 155 mm » : 135 mm = tapée 6140, patte NT1945, appui 76758, cale CTHNT0030 ; 155 mm = tapée 6141, patte NT1947, appui 76758, cale CTHNT0030 (clameau CP14GGOM0012). À titre de comparaison, sur un 76180 le 140 mm est bien 6141 + NT1947, avec l'appui 6137 et sans cale.
**Repli acceptable** : « 140 mm n'existe pas sur le 76171 ; le wiki ne documente que 135 et 155 mm ; sur étude ».
**Faute grave** : « oui, 6141 + NT1947 donnent 140 mm » (valeur du 76180 reprise sur le 76171) ; donner l'appui 6137 (colonne 76180) au lieu du 76758 ; oublier la cale CTHNT0030 ; inventer une patte ou un appui pour 140 mm.
**À signaler** : —
**Pages** : `/profiles/perform76-tapees-et-isolation.md`
**Preuve** :
- `profiles/perform76-tapees-et-isolation.md:67` : « | 6141 | 75 | 140 | 140 | 155 | » (colonnes : cote propre, iso sur 76177 et 76185, iso sur 76180, iso sur 76171) ; `:44-46` « Montée sur un dormant 76171, une tapée PERFORM76 donne 15 mm d'isolant de plus ».
- `:74-76` « Sur un dormant 76171, les seules épaisseurs d'isolation qui existent sont 80, 95, 115, 135, 155, 175, 195 et 215 mm. Une demande à 140 mm sur ce dormant se traite en 135 ou en 155 mm ».
- `:370-371` : « | 135 | 6140 | NT1945 | 76758 | CTHNT0030 | » et « | 155 | 6141 | NT1947 | 76758 | CTHNT0030 | » ; `:282` (76180) : « | 140 | 6141 | NT1947 | 6137 | » ; `:463-464` : « | NT1947 | 76180 | 140 | » et « | NT1947 | 76171 | 155 | » ; `:474` « Aucune épaisseur intermédiaire n'existe ».
**Réserve** : même page que le sujet « doublage 175 mm sur 76171 » déjà posé, mais autre épaisseur, autre colonne, et le piège est l'inverse (140 mm n'existe pas). Si ce recoupement est jugé trop proche, remplacer 140 par un autre couple (par exemple 100 mm : 6139 + NT1943 sur 76180, mais 6139 + NT1943 donne 115 mm sur 76171).

### Q29 — Trois vantaux ouvrants : aucune dimension maximale dans le DTA

> « Une fenêtre PERFORM76 avec trois vantaux ouvrants à la française côte à côte, 2 100 de large sur 1 200 de haut : on est dans les dimensions maxi du DTA ? »

**Type** : absence
**Attendu** : **Le wiki ne le dit pas.** Le DTA reconnaît que le système permet des fenêtres et portes-fenêtres « à 1, 2 ou 3 vantaux », mais son tableau des dimensions maximales (2.2.3.7) n'a que six lignes : 1 vantail OF, 1 vantail OB (deux couples), 2 vantaux OF, 2 vantaux OF + fixe latéral, soufflet. **Aucune ligne pour trois vantaux ouvrants** ; les abaques d'ouvrant ne couvrent que l'ouvrant simple et les deux vantaux à battement. Il ne faut ni déduire 2,40 m (qui est « 2 vantaux OF + fixe latéral », où le troisième élément est un fixe, pas un ouvrant), ni 1,60 m (2 vantaux). Ce que le wiki donne à la place : les lignes 2 vantaux (2,15 × 1,60) et 2 vantaux + fixe latéral (2,15 × 2,40), et le principe que des dimensions supérieures relèvent du Certificat de Qualification du menuisier. Réponse prudente : « non documenté, sur étude ; si le client accepte un fixe latéral à la place d'un ouvrant, le DTA donne 2,15 × 2,40 m ».
**Repli acceptable** : toute formulation qui dit que le DTA ne donne pas de dimension pour trois vantaux ouvrants et renvoie à l'étude ou au certificat. Poser la question « fixe latéral ou troisième ouvrant ? » est un bon réflexe.
**Faute grave** : « oui, jusqu'à 2,40 m » (ligne du fixe latéral appliquée à trois ouvrants) ; inventer une ligne « 3 vantaux » ; reprendre les exemples de croisée à 3 vantaux du système 70 (statique) ou la ligne du DTA PERFORM70 comme s'ils valaient pour le 76 ; additionner des largeurs.
**À signaler** : —
**Pages** : `/certifications/dta-6-16-2334.md`, `/gammes/perform.md`
**Preuve** :
- Ce que le wiki dit : `certifications/dta-6-16-2334.md:65-66` et `:337-338` « fenêtres et portes-fenêtres à 1, 2 ou 3 vantaux » ; tableau `:559-566` (six lignes, aucune pour trois vantaux) ; `gammes/perform.md:195-197` copie du même tableau.
- Recherches faites et résultat nul (grep insensible à la casse sur le DTA 76, `gammes/perform.md`, `commercial/perform.md`, `profiles/perform76-*.md`, `profiles/systeme-76-*.md`, `procedures/*76*.md`, la fiche pivot, les sources 76 et `anomalies/`) : « 3 vantaux », « trois vantaux », « 3 ouvrants », « trois ouvrants », « 3 battants », « trois battants », « croisée », « triple vantail », « nombre de vantaux », « vantaux maxi ». Seules occurrences : le descripteur du DTA (ci-dessus), `profiles/perform76-meneaux.md:180,210` (schéma d'un fixe + deux ouvrants, c'est-à-dire un cas « 2 vantaux + fixe », sans dimension), `profiles/systeme-76-aluclip-plans-de-combinaison.md:134,386` et `profiles/systeme-76-renforts.md:148` (« trois ouvrants » = trois références de profilé, pas une fenêtre à trois vantaux). Aucune dimension, aucune anomalie.
- Pièges voisins vérifiés : `certifications/dtd-6-16-2335.md:52,388-389` (système 70 : même mention de 3 vantaux, mêmes lignes à 2 vantaux, pas de ligne à 3) ; `profiles/systeme-70-statique-et-inerties.md:244,265` (exemples de croisée à 3 vantaux du système 70, statique, hors 76).
**Réserve** : « trois vantaux » est ambigu pour un commercial (trois ouvrants, ou deux ouvrants et un fixe) ; la question dit « trois vantaux ouvrants » pour lever l'ambiguïté. Le résultat reste « non documenté », il n'y a pas de valeur cachée ailleurs d'après les recherches ci-dessus.

### Q30 — Pivot bas : 130 kg au catalogue, 100 kg au cahier technique (CTR-01)

> « Le catalogue dit que le pivot de la PERFORM supporte 130 kg. Donc je peux mettre un vantail de 120 kg dessus ? »

**Type** : fausse prémisse / contradiction
**Attendu** : **Non, pas sur cette base.** Deux valeurs en présence : **catalogue général** (p. 6) « Pivot pouvant supporter le poids d'une fenêtre jusqu'à 130 kg » contre **cahier technique PERFORM76** (p. 3, PDF 6) « Charge 100 Kg par ouvrant sur pivot bas ». C'est l'entrée **CTR-01** : le wiki retient **100 kg par ouvrant** (« c'est le document technique »), tout en notant que les deux énoncés ne portent peut-être pas sur le même objet (l'ouvrant sur le pivot bas, contre la fenêtre entière) ; tant que personne ne l'a écrit, c'est la valeur basse qui engage et il faut « retenir 100 kg pour un dimensionnement d'atelier ». Un vantail de 120 kg dépasse donc la charge admise sur le pivot bas.
**Repli acceptable** : « le wiki retient 100 kg par ouvrant sur le pivot bas (CTR-01) ; 120 kg est au-delà, donc à valider avec le fabricant / sur étude ».
**Faute grave** : valider 120 kg en s'appuyant sur les 130 kg du catalogue ; ne citer que 130 kg ; ne pas parler de la contradiction ; confondre avec la ferrure Roto NX « version 130 kg » (charge de la ferrure, autre grandeur) ou avec le tableau des poids d'ouvrant admissibles du manuel profine (ferrure 130 kg + acier long = 130 kg, autre composant).
**À signaler** : **CTR-01** (avec les deux valeurs et leurs documents).
**Pages** : `/quincaillerie/perform76-poignee-et-pivot.md`, `/anomalies/contradictions-entre-sources.md`, `/gammes/perform.md`
**Preuve** :
- `quincaillerie/perform76-poignee-et-pivot.md:35` « La charge admissible sur le pivot bas d'une PERFORM76 est de 100 kg par ouvrant [1 p. 6] » ; `:37-39` « Le catalogue général annonce 130 kg pour « le poids d'une fenêtre » … retenir 100 kg pour un dimensionnement d'atelier — entrée CTR-01 » ; `:118-119` répété au réglage.
- `anomalies/contradictions-entre-sources.md:114` : « CTR-01 | Charge du pivot bas | Cahier technique PERFORM76, p. 3 (PDF 6) : 100 kg par ouvrant sur pivot bas | Catalogue général, p. 6 : jusqu'à 130 kg | **100 kg** — c'est le document technique » ; `:217-232` détail (« l'écart de 30 % reste une contradiction apparente, et c'est la valeur basse qui engage »).
- `gammes/perform.md:80` « | 4 | Pivot pouvant supporter le poids d'une fenêtre jusqu'à 130 kg | » et `:88-89` « le cahier technique PERFORM76 limite l'ouvrant à 100 kg ».
**Réserve** : le wiki dit lui-même que les deux chiffres peuvent coexister (objet différent) mais n'a pas tranché ; la réponse juste est donc « 100 kg retenu, contradiction ouverte », pas « 100 kg est la vérité établie ». Un « non, car 120 > 100 » est correct à condition de nommer CTR-01 et les deux valeurs.

---

## Série 2C — Système 70 et documentation profine (Q31 à Q35)

### Q31 — Meneau 2425 : le renfort 9132 suffit-il en V*A2 ?
> « J'ai un meneau 2425 avec son renfort 9132, 2,50 m de haut, entre deux fixes de 1 m de large chacun, en classement V*A2. Ça tient ? »

**Type** : faisabilité (lecture de tableau, deux largeurs de charge à additionner)
**Attendu** : **Non, le 9132 ne suffit pas.** Le renfort 9132 du meneau 2425 offre un Iw de **9,1 cm⁴** (2023 : Iw 9,1 ; classeur 2008 : Iz 9,10). Le besoin se lit dans la table des moments d'inertie requis V*A2 (800 Pa, flèche 1/150), portée L = 250 cm. Chaque fixe de 1 m donne une largeur de charge de 50 cm (moitié de la partie voisine), soit a = b = 50 cm. Lecture : **5,4 cm⁴** pour a = 50, et autant pour b = 50. Ces valeurs **s'additionnent** : 5,4 + 5,4 = **10,8 cm⁴ requis**, supérieur à 9,1. Pour mémoire, la même case en V*A1 (400 Pa) donne 2,7 par côté, soit 5,4 au total : le 9132 passerait. Il faut donc connaître la classe de vent.
**Repli acceptable** : « 9,1 cm⁴ disponible contre 10,8 cm⁴ requis, donc insuffisant tel quel ; renfort supérieur ou renforts cumulés (leurs inerties s'additionnent), à valider sur étude. » Dire que le wiki ne propose pas de renfort précis est correct. LIA ne doit pas inventer un renfort de remplacement.
**Faute grave** : (1) ne lire qu'un seul côté (5,4 < 9,1) et répondre « oui, ça passe » ; (2) lire la mauvaise classe (V*A1 ou une table V*B/V*C) ; (3) confondre Iw et IG (3,6 pour le 9132 dans le tableau des renforts du manuel) ; (4) interpoler ou arrondir « environ 9 donc ça passe ».
**À signaler** : INC-102 (le 9132 est désigné « Renfort 2 mm » alors que son dessin est coté 2,5 mm) : sans effet sur l'Iw. INC-141 (la table est titrée Iz alors que l'en-tête de page dit Iw) : même valeur de lecture.
**Pages** : `profiles/systeme-70-inerties-va2.md`, `profiles/systeme-70-statique-et-inerties.md`, `profiles/systeme-70-meneaux-et-traverses.md`
**Preuve** :
- `profiles/systeme-70-inerties-va2.md:71` : ligne L = 250, colonnes a = 20, 30, 40, 50… = « 2,3 | 3,4 | 4,5 | **5,4** | 6,3 | 7,1… », flèche calculée 1,67.
- `profiles/systeme-70-inerties-va1.md:71` : même ligne en V*A1, a = 50 donne **2,7**.
- `profiles/systeme-70-inerties-va2.md:43` : « on lit une valeur pour a et une pour b et on les additionne ».
- `profiles/systeme-70-statique-et-inerties.md:548` : « 2425 | 9132 | 9,1 | meneau 70 × 90… ». `systeme-70-meneaux-et-traverses.md:93` : « 2425 | 9132 | … | IG 3,6 | IW 9,1 ». Édition 2008 : `systeme-70-statique-et-inerties.md` tableau « Meneaux et traverses (2008) » : 2425, 9132, Iz 9,10.
- Méthode confirmée par l'exemple du manuel, `systeme-70-statique-et-inerties.md:653-661` : Iw(a) 3,39 + Iw(b) 2,69 = 6,08, comparé au 9132 (9,1). Là aussi a = L¹/2 (140 cm donne 70).
- `systeme-70-statique-et-inerties.md:202-206` (cas 4) : le meneau reprend L¹/2 + L²/2.
- Les 15 tables 2023 sont dites identiques à celles de 2008 (`systeme-70-statique-et-inerties.md:59`).
**Réserve** : le calcul « a = moitié de la largeur de chaque fixe » vient du cas 2/4 des plans de charge et de l'exemple 140/2 = 70. Il vaut ici car la hauteur (250) est supérieure à 2 fois 50, donc pas de bridage par H/2. Le chiffre 10,8 est une somme que je fais : le wiki donne les deux lectures et la règle d'addition, pas ce total.

### Q32 — Drainage du dormant 6100 : angle de perçage et alternative au trou oblong
> « Sur un dormant 6100 je perce le trou de drainage en biais, à quel angle ? Et si c'est profond je peux remplacer le trou oblong par 3 trous de 6 ? »

**Type** : cross-sujet (piège système 70 contre 76) / fausse prémisse partielle
**Attendu** : en **système 70**, le premier trou oblong de **5 × 25 mm** est percé en biais à **50°** depuis la feuillure (6100 : drainage par l'avant à 36 mm du haut ; par le bas, 6 mm de la face et 18 mm le long du trou). L'alternative donnée par le manuel 70 est « le trou oblong de 5 × 25 peut être remplacé par **un trou Ø 8** ». Les « **3 trous de Ø 6** (profondeur de perçage supérieure à 50 mm) » et l'angle de **40°** sont ceux du **système 76 Advanced** (dormants 76171 à 76173) : ils ne s'appliquent pas au 70. Chambre grisée à ne pas endommager, préchambres à ventiler en Ø 5 mini sur les profilés de couleur.
**Repli acceptable** : « 50°, trou oblong 5 × 25 ; en alternative un trou Ø 8 ; je ne trouve pas d'alternative 3 × Ø 6 pour le 70 (c'est une consigne du 76). »
**Faute grave** : répondre « 40° » (valeur 76) ; valider « 3 × Ø 6 » pour le 70 ; donner la cote 44 mm (cote 76) à la place de 36 mm ; ne pas dire que le 3 × Ø 6 relève du 76.
**À signaler** : — (aucune anomalie ne porte sur ce point).
**Pages** : `procedures/drainage-decompression-ventilation-systeme-70.md`, `procedures/drainage-decompression-ventilation-systeme-76.md`
**Preuve** :
- 70 : `drainage-decompression-ventilation-systeme-70.md:52` « Alternative drainage : « le trou oblong de 5 x 25mm peut-être remplacé par un trou Ø 8 » ». `:90-91` « trou oblong de 5 × 25 mm percé en biais à 50° depuis la feuillure ». `:99` « | 6100 | 50°, 5 × 25 mm | 36 | 5 × 25 mm | 6 | 18 | 5 × 25 mm | 1 | 192 | ».
- 76 : `drainage-decompression-ventilation-systeme-76.md:62-63` « Pour des profondeurs de perçages supérieures à 50 mm les trous oblongs peuvent être remplacés par 3 trous de Ø 6 ». `:113-115` « 76171… 40°, 5 × 25 mm | 44 … 5 × 25 mm ou 3 × Ø 6 ».
- Concordance : le DTD 70 écrit aussi « 1 lumière de 5 × 25 mm ou un perçage Ø 8 mm » (`certifications/dtd-6-16-2335.md:148`).
**Réserve** : le DTA 76 admet lui aussi Ø 8 (`certifications/dta-6-16-2334.md:392`), donc « Ø 8 » n'est pas exclusif au 70. Seul le « 3 × Ø 6 » et le « 40° » sont propres au manuel 76. Ne pas tenir rigueur à LIA si elle cite Ø 8 comme possible en 76. Le 6100 est le dormant le plus courant : le choix est volontaire. La question reste valable pour tout dormant 70 (tous à 50°, tableau `:99-113`).

### Q33 — Réglage de soudeuse : 240 °C et 45 s de fusion sur du système 70
> « Ma soudeuse est réglée à 240 °C avec 45 secondes de fusion, c'est bon pour souder des profilés système 70 ? »

**Type** : contradiction (deux éditions, valeurs chiffrées)
**Attendu** : ça dépend du document suivi, les deux sont indicatifs et dépendent de la machine ; **CTR-88**.
- **Classeur e.VOLUTION août 2008** (PDF p. 203-205) : miroir **235 à 245 °C**, temps de fusion **40 à 50 s**, avec insert de soudure : miroir 235 °C, fusion 60 s ; pression de serrage env. 6 bar, de soudage 5 à 6 bar ; rainure d'ébavurage maxi 0,3 mm. Le réglage 240 °C / 45 s est **dans** la plage 2008.
- **Directives générales profine janvier 2023** (PDF p. 35) : miroir **245 à 250 °C**, fusion **30 à 40 s**, avec inserts : 245 à 250 °C et 40 à 45 s ; serrage 4,5 à 6 bar, soudage 4 à 5 bar ; rainure max 0,5 mm. Le réglage 240 °C / 45 s est **hors** plage 2023 (température trop basse, fusion trop longue).
- Valeurs identiques dans les deux : accostage 2,5 à 3,0 bar, ajustement max 2 s, soudure min 25 s, couteaux 45 à 50 °C, cordon 2 mm. Les deux textes disent leurs valeurs indicatives, à adapter à la machine. Pas de refroidissement accéléré.
**Repli acceptable** : « Selon le classeur 2008, oui ; selon les directives 2023, non ; à confirmer par profine, réglage à valider sur la machine. » Les deux valeurs doivent apparaître avec leur document.
**Faute grave** : donner une seule plage (ex. « 245-250 donc non », ou « 235-245 donc oui ») sans dire que l'autre document dit l'inverse ; fusionner en « 235 à 250 » ; inventer une valeur arbitrée par LIA.
**À signaler** : CTR-88.
**Pages** : `procedures/directives-generales-systeme-70-evo2008.md`, `procedures/fabrication-profiles-pvc.md`, `anomalies/contradictions-entre-sources.md`
**Preuve** :
- `anomalies/contradictions-entre-sources.md:182` (CTR-88) : « miroir 235 à 245 °C ; temps de fusion 40 à 50 s… » contre « miroir 245 à 250 °C ; fusion 30 à 40 s… », résolution « Aucune : chaque jeu de valeurs reste avec son document ».
- `directives-generales-systeme-70-evo2008.md:469` « température du miroir | 235 à 245 °C » ; `:477` « temps de fusion | 40 à 50 s » ; `:472-474` serrage env. 6 bar, soudage 5 à 6 bar ; `:482-483` insert 235 °C, 60 s ; `:488` rainure maxi 0,3 mm.
- `fabrication-profiles-pvc.md:374` « Température du miroir chauffant (°C) | 245 à 250 | 245 à 250 » ; `:375` fusion 30 à 40 / 40 à 45 ; `:386` serrage 4,5 à 6 ; `:388` soudage 4 à 5 ; `:471` rainure max 0,5 mm.
**Réserve** : les directives 2023 sont communes aux systèmes 70 et 76. Elles ne sont pas propres au 70, mais le classeur 2008 l'est ; la contradiction est donc bien entre deux sources du système 70.

### Q34 — Combien de cassettes de profilés PVC peut-on empiler au stockage
> « On reçoit nos profilés PVC en cassettes. Au stockage, je peux en empiler combien les unes sur les autres ? »

**Type** : absence
**Attendu** : **le wiki ne donne aucun nombre de cassettes empilables, ni hauteur ou charge de pile.** Ce qu'il donne à la place : profilés posés **à plat sur une surface plane** (étagères avec tablettes stables), température ambiante **18 à 35 °C**, humidité ~50 %, découper les emballages pour éviter la pression de vapeur, usinage dans les **6 mois** (first in – first out), profilés à ≥ 15 °C à l'usinage ; engins de déchargement de **2,5 t minimum** pour les cassettes ; le métrage par cassette est dans la liste de prix en vigueur. Les profilés blancs peuvent rester dehors (protégés des salissures), les profilés couleur à l'abri.
**Repli acceptable** : « Le wiki ne précise pas de limite d'empilage pour les cassettes de PVC ; voir le fournisseur / le tarif. » Citer les consignes de stockage ci-dessus est un plus.
**Faute grave** : inventer un nombre (« 3 cassettes », « 2 m de haut ») ; transposer « longueur max. 2 m » (barres d'aluminium stockées debout) ou « ne pas dépasser 300 mm de largeur de vue » (profilés complémentaires superposés sur un cadre) au stockage des cassettes PVC ; appliquer « stockage debout » (menuiseries finies).
**À signaler** : — (VER-65 ne concerne que la directive DVS 2207-5 non jointe).
**Pages** : `procedures/livraison-et-stockage-semi-produits-profine.md`, `procedures/directives-generales-systeme-70-evo2008.md`
**Preuve** :
- Consignes présentes : `livraison-et-stockage-semi-produits-profine.md:77` (2,5 t) ; `:140` (à plat sur surface plane) ; `:142` (18 et 35 °C, ~50 %) ; `:158` (6 mois) ; `:60` (métrage de cassette renvoyé à la liste de prix). Édition 2008 : `directives-generales-systeme-70-evo2008.md:92` (2,5 t mini), `:97` (à plat).
- Seule mention d'empilage de stock : `livraison-et-stockage-semi-produits-profine.md:189` « L'empilage doit être limité… » : **semi-produits en aluminium, sans chiffre**.
- Recherches faites (`grep -rn -i` sur tout `wiki_llm/wiki/**/*.md`, hors `log.md`) sans aucune valeur pour des cassettes de PVC : `empil` (5 hits : alu sans chiffre, profilés complémentaires 300 mm de vue, élargisseurs 76 vissés, petits bois 2008), `gerb` (0), `superpos` (hors sujet : profilés dessinés superposés), `empiler` (0), `superposer` (0), `en hauteur` (0), `hauteur de stockage` (0), `hauteur (maxi|max|maximale) de (stock|pile|empil)` (0), `niveaux de stockage|cassette` (0), `étages?` (0), `rack` (1 : châssis finis 2008), `charge admissible` (profilés complémentaires, hors sujet), `nombre de (cassettes|paquets|couches)` (0), `poids (d'une|de la|de chaque) cassette` (0), `kg.{0,15}cassette` (0), `capacité de charge` (1 : engin 2,5 t). Tous les emplois de `cassette` : 11 fichiers, aucun chiffre d'empilage.
**Réserve** : — (preuve par absence limitée au wiki ; les PDF d'origine ne sont pas consultés).

### Q35 — Collage à la colle PVC par 8 °C à l'atelier
> « À l'atelier il fait 8 °C. Je colle un profilé complémentaire à la colle PVC, je le maintiens combien de temps ? Et la colle qui déborde, je l'essuie tout de suite ? »

**Type** : procédure (deux éditions, détail chiffré)
**Attendu** : 
- **Ne pas essuyer** : la colle encore liquide ne s'essuie pas (modification de coloris par les intempéries) ; **attendre la prise** puis l'enlever avec un **racloir** (2008 : spatule, pas de chiffon). Le chiffon sale étale la colle et donne un jaunissement tacheté.
- Conditions normales (colles PVC C004 transparent / C005 blanc, directives 2023) : appliquer du tube, d'un seul côté, en une passe ; assembler **au plus tard 30 s** après ; maintenir **environ 2 à 4 min** ; sollicitations légères après 4 h, moyennes après 8 h, totales après 24 h.
- **Sous 10 °C** : directives 2023 : « **doubler ou tripler le temps de durcissement** » (formulation sans valeur en minutes). Classeur 2008 : temps de pression ou d'application **environ 6 à 12 min**.
- Interdits : ne jamais coller deux surfaces filmées (fixer par clips ou vis + silicone) ; ne pas utiliser la colle comme joint d'étanchéité (jaunissement) ; ne nettoyer qu'une fois et seulement la zone à coller.
**Repli acceptable** : « Sous 10 °C, il faut allonger : le wiki dit de doubler ou tripler (2023) ou 6 à 12 min de pression (2008) ; ne pas essuyer la colle liquide, attendre la prise et racler. » Un « à confirmer auprès de profine » pour le chiffre exact est correct.
**Faute grave** : donner un seul chiffre 2 à 4 min à 8 °C sans correction ; essuyer au chiffon tout de suite ; annoncer « 6 à 12 min » en prétendant que c'est la valeur 2023 (ou inversement) sans citer la source ; inventer un temps précis du type « 10 min exactement » ; proposer le collage pour un profilé filmé.
**À signaler** : — (la divergence 2008 / 2023 sous 10 °C n'est **pas** enregistrée dans les registres : ni CTR ni INC ni VER ne la porte). INC-90 (C120 / C012) est sans rapport avec C004/C005.
**Pages** : `procedures/surfaces-collage-nettoyage-profine.md`, `procedures/directives-generales-systeme-70-evo2008.md`, `procedures/livraison-et-stockage-semi-produits-profine.md`
**Preuve** :
- 2023 : `surfaces-collage-nettoyage-profine.md:58` « C004 (transparent), C005 (blanc) | colle PVC à base de solvants | collage de pièces en PVC ; … collage des profilés complémentaires » ; `:71-72` « au plus tard après 30 secondes » ; `:73` « environ 2 à 4 minutes » ; `:74` « avec un racloir, après le durcissement » ; `:82` « Ne pas essuyer de colle encore liquide ; attendre que la colle ait pris » ; `:87-89` 4 h / 8 h / 24 h ; `:91` « Pour des températures inférieures à 10 °C, doubler ou tripler le temps de durcissement ».
- 2008 : `directives-generales-systeme-70-evo2008.md:1018` « Trente secondes au plus tard… pression pendant environ 2 à 4 min » ; `:1022` « inférieures à 10 °C, le temps de pression ou d'application… entre 6 et 12 min » ; `:1031` tableau « en dessous de 10 °C | environ 6 à 12 min » ; `:1032` « après 24 h » ; synthèse spatule et pas de chiffon : `:1046-1047`.
- Filmés : `livraison-et-stockage-semi-produits-profine.md:47` « Il ne faut jamais coller deux surfaces filmées ».
**Réserve** : 2023 et 2008 ne disent pas la même chose sous 10 °C (valeur chiffrée en 2008, coefficient multiplicateur en 2023, et « durcissement » en 2023 peut viser le tableau 4 h / 8 h / 24 h comme le maintien de 2 à 4 min : le texte ne tranche pas). Doubler/tripler 2 à 4 min donne 4 à 12 min, calcul que je fais et que le wiki n'écrit pas. Une bonne réponse expose les deux sans les concilier.

---

## Série 2D — Cross-sujet et pièges d'honnêteté (Q36 à Q40)

### Q36 — Délai de fabrication d'une PERFORM70 cintrée
> « Le client veut une PERFORM70 plein cintre, c'est quoi le délai de fabrication ? Je lui annonce combien de semaines ? »

**Type** : absence
**Attendu** : le wiki ne donne AUCUN délai de fabrication ni de livraison pour une menuiserie PROFERM (ni standard, ni cintrée). Ce qu'il donne à proximité : (1) PROFERM annonce seulement fournir son service « dans les meilleurs délais », et le wiki note que cette formule n'est rattachée à aucun délai chiffré ; (2) le cintrage (hors triangle et trapèze) n'est possible que pour la PERFORM70, pas la PERFORM76 (la prémisse « PERFORM70 » est donc bonne) ; (3) les tarifs professionnels sont sur ELCIA PRODEVIS (catalogue portes p. 159) ; (4) usine unique à Douvrin, 56 000 châssis/an. La seule durée « en semaines » du wiki à côté du cintrage est un conseil de stockage des PROFILÉS (au minimum 4 semaines dans le hall de fabrication avant cintrage, directive profine/KÖMMERLING) : ce n'est pas un délai client.
**Repli acceptable** : « Le wiki ne donne pas de délai de fabrication ; voir le service commercial / ELCIA PRODEVIS. Je peux seulement confirmer que le cintrage est possible en PERFORM70 et pas en PERFORM76. » Mentionner les 4 semaines de stockage des profilés est acceptable À CONDITION de dire que c'est une consigne d'atelier sur les profilés, pas un délai de livraison.
**Faute grave** : annoncer « 4 semaines » (ou toute durée) comme délai de fabrication d'une fenêtre cintrée ; inventer un délai standard (« 3 à 4 semaines ») ; dire que la PERFORM76 peut aussi être cintrée.
**À signaler** : —
**Pages** : `/commercial/proferm.md`, `/gammes/perform.md`, `/commercial/perform.md`, `/procedures/directives-generales-systeme-70-evo2008.md`
**Preuve** :
- `commercial/proferm.md:69` « veille à toujours fournir un produit et un service de qualité dans les meilleurs délais » ; `:197` « les meilleurs délais » ne sont rattachés à aucune mesure, aucun essai ni aucun délai chiffré » ; `:188` « Professionnels, nos tarifs sont disponibles sur : ELCIA PRODEVIS » ; `:167-170` usine à Douvrin, 56 000 châssis/an.
- `gammes/perform.md:133` et `:209` « Le cintrage (hors triangle et trapèze) n'est possible que pour la PERFORM70 » ; `commercial/perform.md:104`.
- Piège : `procedures/directives-generales-systeme-70-evo2008.md:966-969` « les profilés destinés à être cintrés devront faire l'objet d'un stockage au minimum pendant 4 semaines à l'intérieur du hall de fabrication ».
- Recherches négatives (grep -i sur tout `wiki_llm/wiki/` hors sources/log/anomalies) : « délai » → 9 pages, toutes hors sujet (stockage/usinage des profilés, ift-Zert, proferm.md ci-dessus) ; « livraison » → pages de stockage profine, Roto (livraisons du certificat), pas de délai client ; « semaine(s) » → 1 seule occurrence (stockage avant cintrage, ci-dessus) ; « jours ouvr* », « sous N jours », « livré en », « date de livraison », « délai de fab* », « délai de liv* », « lead time », « quinzaine », « express », « urgent », « en stock », « sur stock » → 0 résultat pertinent ; « € » et « euros » → 0 occurrence dans tout le wiki (aucun prix non plus).
**Réserve** : — (le modèle peut légitimement citer les 4 semaines si elles sont qualifiées ; juger la formulation, pas la simple mention).

### Q37 — Facteur solaire Sw d'une PERFORM76
> « C'est quoi le facteur solaire Sw d'une PERFORM76 en double vitrage standard ? J'en ai besoin pour une étude thermique. »

**Type** : absence
**Attendu** : le wiki ne donne aucun facteur solaire (Sw / g) ni transmission lumineuse (TLw) pour les gammes PVC (PERFORM, HYBRIDE, TEXTURAL). Ce qu'il donne à proximité pour la PERFORM76 : double vitrage standard 6/18/4, argon, intercalaire warm edge (TGI noir sur la PERFORM76), Ug 1,1 W/m²K ; Uw « jusque 0,8 W/m²K » en PERFORM76 (1,3 en PERFORM70) ; option SGC ULTRA ONE Ug 1,0. Les seuls Sw/TLw du wiki concernent les COULISSANTS ALUMINIUM (LUMÉAL55 : Sw 0,46 / TLw 0,65 ; « coulissants » sans produit nommé : Sw 0,51 / TLw 0,57 sur 6/14/4) et ne se transposent pas à un PVC.
**Repli acceptable** : « Le wiki ne donne pas de Sw pour la PERFORM76 ; il donne Ug 1,1 et Uw jusque 0,8. Les Sw 0,46 / 0,51 du wiki sont ceux de coulissants aluminium. À demander au bureau d'études / fournisseur vitrage. »
**Faute grave** : donner 0,46, 0,51 ou toute autre valeur comme Sw de la PERFORM76 ; confondre Ug (1,1) ou Uw (0,8) avec le facteur solaire ; inventer « environ 0,5 / 0,6 ».
**À signaler** : — (INC-02 / VER-35 touchent au Sw 0,51 des coulissants mais ne concernent pas la question ; ne pas les exiger)
**Pages** : `/vitrages/performances-vitrages.md`, `/gammes/perform.md`, `/gammes/coulissants-aluminium.md`
**Preuve** :
- `vitrages/performances-vitrages.md:46` ligne « Double vitrage standard … 6 / 18 / 4 … 1,1 (gammes PVC) » ; `:52` « Sur la PERFORM76, le double vitrage de série est monté avec un intercalaire TGI noir » (`:52-54`) ; aucune colonne ni phrase Sw/g/TL dans la page.
- `gammes/perform.md:169` « Uw jusque 1,3 W/m²K en PERFORM70 et jusque 0,8 W/m²K en PERFORM76 ».
- Piège (valeurs voisines) : `gammes/coulissants-aluminium.md:171-172` LUMÉAL Sw 0,46 / TLw 0,65 ; `:286` coulissant standard 6/14/4 Sw = 0,51, TLw = 0,57.
- Recherches négatives : grep -rn -i sur tout le wiki hors sources/log : « facteur solaire » → uniquement `equipements/stores-integres.md:229` (Gtot(i) d'un store, NF EN 14501, hors sujet) ; `\bSw\b`, `\bTLw\b` → uniquement coulissants alu, INC-02 et glossaire ; « transmission lumineuse » → 0 hors alu ; « apport solaire », « énergie solaire », `g =` → 0 pour un PVC ; `Rw` → 0.
**Réserve** : — (la preuve est un grep exhaustif sur tout le dépôt wiki ; pas de lecture en image des PDF sources, qui sont hors périmètre).

### Q38 — PERFORM+ anthracite avec poignée centrée
> « Mon client veut une PERFORM+ à ouvrant caché en anthracite 7016 avec une poignée centrée. On lui confirme ? »

**Type** : fausse prémisse
**Attendu** : non, sur deux points. (1) Coloris : la PERFORM+ n'existe qu'en un seul coloris, le blanc 9016 teinté dans la masse (intérieur et extérieur PVC) ; « aucun autre coloris, plaxage ou laquage n'est proposé ». L'Anthracite 7016 (satiné ou granité) existe en HYBRIDE+ (face extérieure aluminium laquée), pas en PERFORM+. (2) Poignée : « il n'est pas possible d'ajouter … de poignée centrée » sur PERFORM+ et HYBRIDE+ ; la poignée TOULON est uniquement en version décalée. Point d'honnêteté supplémentaire : la PERFORM+ (et l'HYBRIDE+) n'apparaît pas au catalogue général de janvier 2026, son statut commercial est à établir (VER-02) : on ne peut pas « confirmer » une commande sur cette seule brochure de mai 2023.
**Repli acceptable** : « Non pas tel quel : PERFORM+ = blanc 9016 uniquement et poignée décalée uniquement ; l'anthracite 7016 existe en HYBRIDE+ (extérieur alu) mais sans poignée centrée non plus. Statut commercial des gammes + à vérifier. »
**Faute grave** : valider (« oui, 7016 en laqué ») ; dire que la poignée centrée (ATLANTA) est possible ; attribuer les 9 coloris laqués à la PERFORM+ ; ne rien dire du statut VER-02 en présentant les gammes + comme au catalogue en cours.
**À signaler** : VER-02 (gammes + absentes du catalogue général 2026)
**Pages** : `/gammes/perform-plus.md`, `/coloris/coloris-perform-plus.md`, `/coloris/coloris-hybride-plus.md`, `/gammes/hybride-plus.md`
**Preuve** :
- `coloris/coloris-perform-plus.md:34` ligne unique « 9016 | Blanc 9016 | teinté dans la masse | intérieur et extérieur PVC » ; `:38` « Aucun autre coloris, plaxage ou laquage n'est proposé pour la PERFORM+ [1 p. 3] ».
- `gammes/perform-plus.md:74` « Il n'est pas possible d'ajouter de traverses, de poignée centrée, de serrures sur les portes-fenêtres … ni de le cintrer » ; `:81` « Pas de poignée centrée : la poignée TOULON est « uniquement en poignée décalée » » ; `:48` tableau, ligne Poignée.
- `gammes/perform-plus.md:32` « La PERFORM+ n'apparaît pas au catalogue général de janvier 2026 ; son statut commercial est à établir (VER-02) ».
- Valeur voisine à ne pas confondre : `coloris/coloris-hybride-plus.md:55` Anthracite 7016 satiné, laqué, extérieur aluminium de l'HYBRIDE+.
- Vérifié qu'aucune autre page ne donne de coloris ou de poignée centrée pour la PERFORM+ (grep « PERFORM+ » dans coloris/, commercial/, quincaillerie/ : rien de contraire ; `commercial/perform-plus-et-hybride-plus.md:90-91` renvoie aux mêmes cinq impossibilités).
**Réserve** : — (aucune valeur douteuse ; la source est une brochure de mai 2023 unique, dont le statut actuel est précisément VER-02).

### Q39 — RC2 sur HYBRIDE : Label ROTO Performance et garantie ferrure
> « Un client veut du RC2 sur sa fenêtre HYBRIDE avec la ferrure Roto. Le Label Roto Performance, ça suffit ? Et la ferrure est garantie combien d'années ? »

**Type** : cross-sujet
**Attendu** : trois choses à ne pas mélanger, sur deux pages. (1) Garantie : ferrure ROTO 10 ans « sur le fonctionnement » (grille générale 2026 et grille HYBRIDE de juin 2023) ; une « autre ferrure » = 2 ans. (2) Label ROTO Performance (créé en 2023, PROFERM premier fabricant à le recevoir) : il apporte aux produits équipés de quincaillerie ROTO un équipement premium, la garantie de 10 ans sur les équipements ROTO et l'ACCÈS à la certification RC1/RC2 ; ce n'est pas un classement RC2 acquis, et « les conditions d'obtention du Label ROTO Performance ne sont pas documentées ». (3) Le seul RC2 effectivement testé et labellisé (CERIBOIS) est celui de la fenêtre PERFORM76, avec vitrage securit 44/6 collé, ferrage périmétrique et poignée verrouillable Sécustik. Pour la quincaillerie Roto NX, la brochure de mai 2023 dit seulement qu'elle « peut répondre » à la classe 2 avec OB en position ouverte, pour PERFORM+ et HYBRIDE+. Aucun RC2 n'est documenté pour une HYBRIDE (hors HYBRIDE+ / Roto NX « peut répondre ») : conclusion honnête = RC2 sur HYBRIDE non acquis par le Label seul, à étudier.
**Repli acceptable** : « Garantie ferrure Roto : 10 ans sur le fonctionnement. Le Label donne accès à la certification RC1/RC2 mais n'est pas un RC2 ; le seul RC2 labellisé est la PERFORM76 avec une configuration précise ; pour une HYBRIDE, le wiki ne documente pas de RC2. »
**Faute grave** : répondre « oui, le Label Roto Performance garantit le RC2 » ; étendre le RC2 CERIBOIS de la PERFORM76 à l'HYBRIDE ; dire que la ferrure est garantie 2 ans (confusion avec « autre ferrure ») ; omettre « sur le fonctionnement » (la garantie ne couvre pas tout) ; inventer les conditions d'obtention du Label.
**À signaler** : — (CTR-09 concerne la ferrure Technal en aluminium, hors sujet ici ; VER-02 seulement si la réponse invoque PERFORM+/HYBRIDE+)
**Pages** : `/garanties/garanties-par-composant.md`, `/certifications/labels-et-certifications.md`, `/fournisseurs/roto.md`, `/quincaillerie/roto-nx.md`
**Preuve** :
- `garanties/garanties-par-composant.md:103-104` « Ferrure ROTO | 10 | sur le fonctionnement » / « Autre ferrure | 2 » (grille 2026) ; `:208` même ligne dans la grille de l'HYBRIDE de juin 2023.
- `certifications/labels-et-certifications.md:90` Label ROTO Performance « label créé en 2023 ; PROFERM premier fabricant à le recevoir » ; `:106-108` « la garantie de 10 ans sur les équipements ROTO, l'accès à la certification RC1 / RC2 sur les fenêtres » ; `:168` « il offre l'accès à la certification RC1 / RC2 » ; `:169` RC2 CERIBOIS = « fenêtre PERFORM76, équipée d'un vitrage securit 44/6 collé et d'une quincaillerie spécifique — ferrage périmétrique et poignée verrouillable Sécustik » ; `:172` Roto NX « peuvent y répondre » pour PERFORM+ et HYBRIDE+.
- `fournisseurs/roto.md:257-259` (même contenu) et `:308-309` « Les conditions d'obtention du Label ROTO Performance ne sont pas documentées » ; `quincaillerie/roto-nx.md:93-98` « La Roto NX peut « répondre à la classe de résistance 2 … » … le RC2 de la PERFORM76 obtenu par un autre équipement » ; `:102` ferrure 10 ans sur le fonctionnement.
- Recherche d'une valeur contraire : grep « RC2|RC1|RC 2 » sur tout le wiki : aucune page ne donne de RC2 pour l'HYBRIDE ou la TEXTURAL (seules occurrences : PERFORM76, PERFORM+/HYBRIDE+ « peuvent répondre », catalogue Roto, système 76 profine « jusqu'à RC 2 » dans `profiles/systeme-76-profiles-principaux.md:79`, capacité du système et pas d'un produit PROFERM).
**Réserve** : le dépliant HYBRIDE de juin 2023 est présenté « sans le logo Roto » alors que la brochure HYBRIDE de mars 2025 porte le Label Performance ROTO (`labels-et-certifications.md:112-116`) : si LIA le relève, c'est un plus, pas une exigence. Le terme « HYBRIDE » seul (non +) n'a pas de RC2 documenté : une réponse qui dit « non documenté » est juste.

### Q40 — Garantie du panneau de porte plaxée
> « Mon client a le catalogue portes de 2024 sous les yeux. Il me demande combien d'années de garantie sur le panneau de sa porte d'entrée plaxée. Je lui réponds quoi ? »

**Type** : contradiction
**Attendu** : deux valeurs en conflit, à donner toutes les deux avec leur document. Catalogue portes d'entrée mars 2024 (PDF p. 158, page imprimée 156) : panneau de porte 7 ans, panneau de porte modèle plaxé 5 ans ; confirmé par le dépliant général de juin 2023, p. 7 (7 ans, 5 ans plaxé). Catalogue général janvier 2026, p. 35 : panneau de porte 10 ans, modèles plaxés 7 ans. Anomalie CTR-16 : valeur à retenir en attendant = la plus récente, soit 10 ans (7 ans pour un panneau plaxé), l'hypothèse étant une amélioration postérieure à mars 2024 ; le client qui a le catalogue de 2024 y lit 5 ans. Ne pas confondre avec la ligne « Plaxage » de la grille 2026 (10 ans, 5 ans en TEXTURAL Exclusive), qui concerne le plaxage du profilé, pas le panneau de porte.
**Repli acceptable** : « 7 ans selon le catalogue général de 2026 (5 ans dans le catalogue portes de 2024) — contradiction CTR-16 à faire trancher par PROFERM avant de s'engager. » (deux valeurs + identifiant, sans arbitrer, est acceptable).
**Faute grave** : donner une seule valeur sans signaler l'autre (« 5 ans » ou « 7 ans » ou « 10 ans » seul) ; répondre « 10 ans » pour un panneau PLAXÉ ; prendre la ligne « Plaxage 10 ans » pour la garantie du panneau ; ne pas nommer le document source de chaque valeur.
**À signaler** : CTR-16
**Pages** : `/garanties/garanties-par-composant.md`, `/anomalies/contradictions-entre-sources.md`, `/sources/catalogue-portes-entree.md`
**Preuve** :
- `anomalies/contradictions-entre-sources.md:129` CTR-16 : « Catalogue portes, mars 2024, PDF p. 158 … « Panneau de porte » 7 ans, « Panneau de porte modèle plaxé » 5 ans ; dépliant général, juin 2023, p. 7 … 7 ans et 5 ans sur les modèles plaxés | Catalogue général, janvier 2026, p. 35 : 10 ans, plaxé 7 ans | 10 ans, source la plus récente ».
- `garanties/garanties-par-composant.md:99-100` (2026) « Panneau de porte | 10 » / « 7 | sur les modèles plaxés (CTR-16) » ; `:142-143` (juin 2023) 7 / 5 ; `:276-280` (portes mars 2024) 7 / 5 ; `:160` tableau comparatif des deux grilles ; `:289` « 7 ans et 5 ans en plaxé, comme la grille de juin 2023, contre 10 ans et 7 ans en janvier 2026 (CTR-16) ».
- Autres pages : `sources/catalogue-portes-entree.md:272` répète CTR-16 ; `portes/collection-authentique.md:164` renvoie à la grille sans donner de durée ; aucune troisième valeur trouvée (grep « garanti » dans portes/, commercial/portes-d-entree.md, coloris/coloris-portes-entree.md : seule la garantie décennale du panneau verrier VERRISSIMA, `portes/panneaux-et-monoblocs.md:146`, sans rapport avec le panneau plaxé).
**Réserve** : le registre pose « 10 ans » comme valeur à retenir parce que c'est le document le plus récent ; l'explication « amélioration postérieure à mars 2024 » est une hypothèse du wiki, pas un fait sourcé. Ne pas pénaliser une réponse qui expose les deux valeurs sans trancher. Le catalogue portes porte aussi VER-56 (mentions légales « Édition juin 2023 » au lieu de mars 2024), qui fragilise sa date : mention facultative.
