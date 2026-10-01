# Golden LIA — 20 questions de terrain (01/10/2026)

Reconstitué depuis `docs/compte_rendu_lia_2026-09-22.md`, dont les questions étaient résumées et non
citées. Elles sont ici reformulées comme un commercial ou un menuisier les poserait. Les réponses
attendues sont celles du compte rendu, **recontrôlées le 01/10 contre le wiki actuel** (323 pages,
réécrit depuis le 22/09). Ce wiki n'est plus celui de la mesure d'origine, et certaines valeurs ont
bougé : elles sont marquées 🔶.

**Légende** : ✅ valeur retrouvée telle quelle dans le wiki le 01/10 — 🔶 le wiki dit autre chose que
le compte rendu, à arbitrer — ◻ non recontrôlé, à vérifier à la main avant de s'en servir.

## Ce qui fait un « sans faute »

Une réponse est juste seulement si **toutes** ces conditions tiennent :

1. **La valeur exacte** de la colonne et de la ligne demandées, avec son unité. Jamais interpolée,
   jamais prise dans la colonne voisine (règles 4 et 9).
2. **La condition ou l'exception** qui change la décision (règle 10).
3. **L'anomalie concernée, avec les deux valeurs en présence** et leur document, pas seulement
   l'identifiant (règle 2).
4. **Au moins une page citée par son chemin, et qui a été chargée** (règle 5).
5. **Aucune invention** : une absence est dite, jamais complétée (règle 6).
6. **Pour une faisabilité**, la limite lue et nommée avant tout « oui » (règle 14).
7. **Pour un suivi**, la gamme de la question précédente n'est pas redemandée.

À mesurer en plus de la justesse : pages lues, tokens d'entrée (max d'un appel), nombre de
recherches, temps.

---

## Série A — Quincaillerie et gammes aluminium

### Q1 — Poignées du coulissant LUMÉAL GA
> « Quelles poignées je peux mettre sur un coulissant LUMÉAL GA ? »

**Attendu** : 5 références techniques avec emplacement et compatibilité — **TGA3618, TGA3606, TGA3607,
TGA6000, T661004** — avec la note sur les supports TGA3706 / TGA3707. Puis les options du catalogue
général : ATLANTA, TOULON, DEHLI, SEOUL, BERLIN, ANTIBES, OSAKA, KOBE, TOKYO, KYOTO.
**Ne doit pas** : SYDNEY, MILAN, SHANGHAI, PEKIN, GRENADE (non proposées sur LUMÉAL).
**Piège** : croiser deux niveaux de lecture, références de quincaillerie et noms commerciaux.
**Pages** : `/quincaillerie/lumeal-ga-roulements-et-fermetures.md`, `/quincaillerie/poignees-et-croisillons.md`,
`/fournisseurs/technal.md`. **Statut** : ◻ (références TGA3618 / T661004 présentes, liste des noms commerciaux non recontrôlée).

### Q2 — Uw et vitrage SOLÉAL55 contre LUMÉAL55
> « C'est quoi le Uw et le vitrage standard du SOLÉAL55 et du LUMÉAL55 ? »

**Attendu** : LUMÉAL55 — Uw **1,2**, vitrage **28 mm (6/18/4)** ✅. SOLÉAL55 : le compte rendu donnait
« 1,4 / 6-14-4 ». 🔶 **Le wiki actuel dit que ce 1,4 n'est rattaché ni au SOLÉAL55 ni au GALANDAGE55
(INC-02, VER-35)** : le catalogue le donne pour « les coulissants » sans nom de produit. La bonne
réponse est donc : 1,4 « jusqu'à », sur 6/14/4, **non attribuable à un modèle**, avec INC-02 expliquée.
**Pages** : `/gammes/coulissants-aluminium.md`. **Statut** : ✅ pour LUMÉAL, 🔶 pour SOLÉAL55.

### Q3 — Quincaillerie invisible en LUMINE65
> « La quincaillerie invisible, on peut la faire en LUMINE65 ? »

**Attendu** : distinguer la **fenêtre battante** (non, réservée au module 55) et la **baie coulissante**
(non documentée comme telle ; poignée encastrée **MLINI** à la brochure LUMINE65, **DEHLI** ailleurs,
cuvette SEOUL — **CTR-12**, avec les deux noms et leurs documents).
**Pages** : `/gammes/lumine.md`, `/commercial/lumine.md`, `/quincaillerie/poignees-et-croisillons.md`,
`/gammes/coulissants-aluminium.md`. **Statut** : ✅ pour CTR-12 (MLINI contre DEHLI), ◻ pour le reste.

### Q4 — *Suivi de Q3*
> « Et en LUMINE55 ? »

**Attendu** : **oui** — paumelles dissimulées, poignée TOULON décalée, ouverture jusqu'à 180°. Rappelle
qu'elle n'existe pas en 65. **La gamme n'est pas redemandée.** **Statut** : ◻.

### Q5 — Ouvrants SOLEAL FY
> « Y a quoi comme ouvrants en FY et ils font quelle profondeur ? »

**Attendu** (tableau) : **FYa / OA** 65 / 75 mm ◻ ; **FYm / OM** 59,9 mm (module 55) / 69,9 mm (module 65) ✅ ;
**CC (chant clippable)** 59,9 / 69,9 mm ✅. La distinction module 55 / module 65 doit figurer.
**Pages** : `/profiles/soleal-fy-dormants-et-ouvrants.md`. **Statut** : ✅ partiel.

### Q6 — Drainage complémentaire de feuillure
> « À partir de quelle épaisseur de vitrage il faut un drainage en plus dans la feuillure ? »

**Attendu** : **27 mm** (à partir de, inclus) ; usinage **4,5 × 30 mm** dans la gorge porte-parclose au
droit de chaque lumière extérieure ; système **SOLEAL FY**.
**Pages** : `/procedures/fabrication-soleal-fy.md`. **Statut** : ✅.
**Défaut connu du wiki** : la cote y est écrite en LaTeX brut (`$4,5 \times 30\text{ mm}$`) et la
réponse la recopie telle quelle. À corriger dans la page, pas dans les consignes.

### Q7 — Volets roulants et Technal
> « Technal nous fournit nos volets roulants ? »

**Attendu** : **non**. Motorisation : **SOMFY** seul nommé. Coffres et tabliers : aucun fabricant nommé
côté Technal. **VER-16** (piste LAKAL, crédit photo seulement, à vérifier). **Et SOPROFEN**, concepteur
des blocs-baies **Chrono One**, du **Bloc LX** et de la technologie **GoodNight** — c'est cette omission
qui avait fait classer la réponse « incomplète » le 22/09.
**Pages** : `/equipements/volets-roulants.md`, `/fournisseurs/soprofen.md`. **Statut** : ✅.

### Q8 — LUMÉAL GA : module, rupture thermique, gain de clair
> « Pour le LUMÉAL GA : le module, la rupture de pont thermique et ce qu'on gagne en clair de vitrage ? »

**Attendu** : module **100 mm (2 rails) / 151 mm (3 rails)** ✅ ; barrette **33 mm dormant / 22 mm
ouvrant** ✅ ; gain de clair de **8 à 14 % selon la pose** (la fourchette complète, pas la valeur
basse) ✅ ; masse d'aluminium visible réduite de **35 %** ✅.
**Pages** : `/gammes/coulissants-aluminium.md`, `/profiles/lumeal-ga-dormants-et-ouvrants.md`.

### Q9 — LUMÉAL GA : charge maximale
> « Combien de kilos par vantail on peut mettre sur un LUMÉAL GA, et à quelle condition ? »

**Attendu** : **300 kg**, avec chariots **triples** : **TGA3608** (rail alu ou inox) ou **TGA3609**
(**rail inox uniquement**) ; **fraisage de traverse à 80 mm obligatoire** dans les deux cas ✅. Règle de
calage **TGA3817** : systématique sur T141021 ; sur T141015 au-delà de 120 kg ou 1,50 m ◻.
**Pages** : `/quincaillerie/lumeal-ga-roulements-et-fermetures.md`, `/procedures/fabrication-et-pose-lumeal-ga.md`.

### Q10 — Galandage en LUMÉAL GA
> « On peut faire du galandage avec le LUMÉAL GA ? »

**Attendu** : **non** — le catalogue de conception ne donne aucune formule de galandage. Redirection
vers **GALANDAGE55** (base SOLEAL GY 55), de 1 à 4 vantaux sur 1 à 3 rails. **Statut** : ◻.

### Q11 — Crémone SOLÉAL pour 1 300 mm à clé
> « Quelle crémone pour un coulissant SOLÉAL de 1 300 de large, que le client veut pouvoir fermer à clé ? »

**Attendu** : **TGY3703** (3 points, à clé, **H mini 1 292 mm**, Hp 721 mm, **8 vis TGY3723**) ✅. Cylindre
selon le montant : **T1040** 30 × 30 (TGY1202 / TGY1303) ou **T1044** 40 × 30 décentré (TGY1301 / TGY1302) ;
rosettes **T960013 / TPY6003** ✅. **Ne doit pas** donner TGY3702 : sans clé.
**Pages** : `/quincaillerie/soleal-gy-roulements-et-fermetures.md`. **Statut** : ✅.

### Q12 — *Suivi de Q11*
> « Et pourquoi pas la TGY3701 ? »

**Attendu** : la TGY3701 **convient aussi** (1 point à clé, **H mini 525 mm**, Hp 346 mm, 4 vis TGY3723) ✅.
La différence est le **nombre de points : 1 contre 3**. LIA doit **corriger la prémisse** de la question
au lieu de justifier son premier choix, puis donner la règle de choix. **Statut** : ✅.

### Q13 — Ce que ROTO fournit
> « Roto, ils nous fournissent quoi exactement, et qu'est-ce qu'on doit aller chercher ailleurs ? »

**Attendu** : fournit — **Roto NX** (PERFORM+ / HYBRIDE+), **Patio Inowa** (INNOSLIDE), paumelles
**Solid B** et serrures **Safe E Eneo** (portes PVC et HYBRIDE), aérateurs à entrebâillement. Échappe —
tout l'aluminium (TECHNAL, paumelles Fapim Tube), porte d'entrée 1 vantail seule, seuils, ferme-portes,
boîtes aux lettres, cales KÖMMERLING, ouvertures d'imposte, aérateurs insonorisés. **VER-20** (le DTA
nomme FERCO). **Pages** : `/fournisseurs/roto.md`. **Statut** : ◻ (VER-20 vérifiée).

## Série B — Faisabilités et limites

### Q14 — Designo II : 500 mm, 90 kg
> « Un vantail Designo II de 500 de large pour 90 kg, ça passe ? »

**Attendu** : **non**. Le wiki actuel porte **CTR-114** : le manuel de montage KSR donne **80 kg maximum
sans report de charge** (150 kg avec), le catalogue **100 kg**. 🔶 Le compte rendu du 22/09 ne citait
pas cette contradiction. À 500 mm de LFF, la version 100 kg du catalogue ne s'applique pas (**LFF
mini 600 mm**), et le report de charge (150 kg) exige **LFF ≥ 800 mm et HFF ≥ 1 000 mm** — hors de portée.
Les deux sources donnent « non » à 90 kg ; la réponse doit **dire laquelle dit quoi (CTR-114)**.
**Pages** : `/quincaillerie/roto-nx-apercu-ferrures-designo.md`, `/procedures/report-de-charge-roto-nx.md`.
**Statut** : ✅ valeurs, 🔶 anomalie.

### Q15 — PERFORM70 contre PERFORM76
> « Qu'est-ce qui change entre une PERFORM70 et une PERFORM76 au catalogue ? »

**Attendu** : tableau 70 / 76 mm ; Uw 1,3 / 0,8 ◻ ; structure et étanchéité de la 70 **« Non spécifiée »**
(case vide, pas de valeur vraisemblable) contre 6 chambres / 3 joints pour la 76 ; renvoi **VER-03** (le
système 70 Plateforme de profine compte 5 chambres, mais rien n'établit que la PERFORM70 repose dessus) ✅.
Puis cintrage (impossible en 76, autorisé en 70), laquage PVC (70 seule), certifications (RC2 et FFCP
en 76). **Pages** : `/gammes/perform.md`, `/commercial/perform.md`. **Statut** : ◻ (VER-03 vérifiée).
**Piège** : « PERFORM70 » et « PERFORM76 » sont deux produits, pas deux écritures d'une même gamme.

### Q16 — Styles TEXTURAL
> « C'est quoi les trois styles TEXTURAL, et combien de textures exclusives ? »

**Attendu** : **NATIVE, AUTHENTIQUE, EXCLUSIVES** ✅ ; textures exclusives **intérieur uniquement** ✅.
🔶 **Nombre : 26** dans le wiki actuel (`gammes/textural.md`, `coloris/coloris-textural.md`, catalogue
janvier 2026), **pas 25** comme dans le compte rendu. La page des coloris précise que 11 des 12
textures de juin 2023 figurent parmi les 26 de janvier 2026. À confirmer contre le catalogue avant de
fixer la valeur. **Statut** : 🔶.

### Q17 — INNOSLIDE
> « Comment marche l'INNOSLIDE, et pour quelles configurations ? »

**Attendu** : coulissant PVC à frappe, ferrure **Roto Patio Inowa**, verrouillage périphérique actif sans
soulèvement, SoftClose, **SoftOpen à partir de 1 970 mm** de largeur ✅. Configuration **exclusivement
2 vantaux** (1 coulissant + 1 fixe) ; au-delà, c'est l'aluminium. **Et les bornes**, qui avaient fait
classer la réponse « incomplète » : largeur **1 500 à 3 200 mm** (soudure grain d'orge) et jusqu'à
**4 200 mm** (dormant ébavuré, ouvrant grain d'orge), hauteur **2 400 mm**, vitrage jusqu'à **41 mm** ✅.
**Pages** : `/gammes/innoslide.md`, `/quincaillerie/roto-patio-inowa.md`. **Statut** : ✅.

### Q18 — Les quatre coulissants aluminium
> « Les quatre coulissants alu, ils font combien de large maxi ? »

**Attendu** : SOLÉAL55 **6 200 mm** ✅ ; GALANDAGE55 **4 800 mm** (1 800 en 1 vantail, 3 600 en 2 vantaux / 2 rails,
4 800 en 3 vantaux / 3 rails) ✅ ; LUMÉAL55 **6 000 mm** (**2,25 m par vantail**, largeur totale et par
vantail à distinguer) ✅ ; LUMINE65 **6 000 mm** ◻, avec le détail des configurations de rails.
**Pages** : `/gammes/coulissants-aluminium.md`. **Statut** : ✅ partiel.

### Q19 — Classement AEV de la PERFORM70
> « Quel classement A*E*V pour la fenêtre PERFORM70 ? »

**Attendu** : **A\*4 / E\*9A / V\*A3**, présenté comme le plus élevé du marché français ✅.
**Pages** : `/commercial/perform.md`. **Statut** : ✅.

### Q20 — *Contrôle de stabilité*
> Même question que Q9, mot pour mot, dans une **nouvelle** conversation.

**Attendu** : réponse strictement identique à Q9 (valeur, condition, pages). **Statut** : ✅ si Q9 l'est.
À rejouer 3 fois : une variance entre passages est une faute.

---

## À trancher avant de s'en servir

| Question | Écart | À faire |
| --- | --- | --- |
| Q2 | INC-02 : le 1,4 n'est pas rattaché au SOLÉAL55 | Valider la formulation de l'attendu |
| Q14 | CTR-114 apparu dans le wiki (80 kg manuel contre 100 kg catalogue) | Valider « non » + les deux sources |
| Q16 | 26 textures dans le wiki, 25 au compte rendu | Vérifier au catalogue papier |
| Q1, Q3 à Q5, Q10, Q13, Q15, Q18 | valeurs partiellement ou pas recontrôlées | Relecture contre les pages indiquées |
| Q6 | LaTeX brut dans la page `fabrication-soleal-fy.md` | Corriger le wiki |

## Ce que ce golden ne couvre pas

La suite est dans `docs/golden_lia_20_questions_pvc_roto_2026-10-01.md` (Q21 à Q40 : faisabilités
PERFORM76, ferrures Roto NX, système 70, absences et fausses prémisses), à passer **une conversation
par question**.

- **Les trois faisabilités du 30/09** (OF 1 300 × 2 400, OB 1 200 × 1 600, SoftOpen 1 800), toujours
  à ajouter : `docs/rapport_golden_20q_2026-09-30.md`.
- **Les coupes d'images** (76576, 76177, 6106).
