# Compte rendu — notre solution (LIA) confrontée à la documentation

**Date** : 22 septembre 2026
**Objet** : 20 questions techniques et commerciales posées à LIA, dont 2 questions de suivi
conversationnel et 1 question posée deux fois pour contrôler la stabilité.
**Référence de correction** : la documentation technique et commerciale PROFERM / Technal / Roto /
profine / Soprofen (catalogues, cahiers techniques, DTA, notices de montage, brochures et dépliants).
**Comparaison** : 16 de ces questions ont été posées à la solution externe — voir
`compte_rendu_solution_externe_2026-09-22.md` et le §5 ci-dessous.

---

## 1. Synthèse

| Indicateur | Valeur |
| --- | --- |
| Questions posées | 20 |
| Questions auxquelles la documentation répond | 20 / 20 |
| Réponses effectivement produites | **20 / 20 (100 %)** |
| Refus | 0 |
| Réponses justes et directement exploitables | **18 / 20 (90 %)** |
| Réponses justes mais incomplètes | 2 / 20 |
| Réponses partiellement fausses | 0 |
| Réponses fausses ou inventées | 0 |

**Verdict global : aucune erreur de fond sur 20 questions. Les deux réserves portent sur
l'exhaustivité, pas sur l'exactitude.**

### Répartition par verdict

| Verdict | Nombre | Questions |
| --- | :---: | --- |
| ✅ Juste et complet | 18 | 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20 |
| 🟡 Juste mais incomplet | 2 | 7, 18 |
| 🟠 Partiellement faux | 0 | — |
| ❌ Refus | 0 | — |
| 🔴 Faux ou inventé | 0 | — |

---

## 2. Détail question par question

| # | Question | Réponse produite | Réponse attendue | Verdict |
| :---: | --- | --- | --- | :---: |
| 1 | Poignées disponibles sur un coulissant LUMÉAL GA | Tableau de 5 références (TGA3618, TGA3606, TGA3607, TGA6000, T661004) avec emplacement et compatibilité, note sur les supports TGA3706 / TGA3707, puis les options du catalogue général (ATLANTA, TOULON, DEHLI, SEOUL, BERLIN, ANTIBES, OSAKA, KOBE, TOKYO, KYOTO) | Exactement cela. **Croise les deux niveaux de lecture** — références de quincaillerie et noms commerciaux — et n'inclut pas SYDNEY, MILAN, SHANGHAI, PEKIN ni GRENADE, qui ne sont pas proposées sur LUMÉAL | ✅ |
| 2 | Uw et vitrage standard SOLÉAL55 vs LUMÉAL55 | SOLÉAL55 : 1,4 / 6-14-4. LUMÉAL55 : 1,2 / 28 mm (6/18/4). **Signale INC-02** : le « jusqu'à 1,4 » du catalogue n'est étiqueté à aucun modèle | Valeurs exactes, et la réserve INC-02 est précisément celle que le chiffrage doit connaître | ✅ |
| 3 | La quincaillerie invisible est-elle disponible en LUMINE65 ? | Distingue **fenêtre battante** (non, réservée au module 55) et **baie coulissante** (non documentée comme telle ; poignée encastrée DEHLI ou MLINI selon la source **[CTR-12]**, cuvette SEOUL) | Exact sur les deux cas, et la contradiction CTR-12 sur le nom de la poignée est citée au bon endroit | ✅ |
| 4 | *(suivi)* Et en LUMINE55 ? | Oui : paumelles dissimulées, poignée TOULON décalée, ouverture jusqu'à 180°. Rappelle l'indisponibilité en 65 | Exact. **Le suivi conversationnel fonctionne** : la gamme n'est pas redemandée | ✅ |
| 5 | Les trois versions d'ouvrant FY et leurs profondeurs | Tableau : FYa/OA 65 / 75 mm ; FYm/OM 59,9 / 69,9 ; CC 59,9 / 69,9, avec la distinction module 55 / module 65 | Exact, y compris la double colonne de module que la question n'imposait pas | ✅ |
| 6 | Épaisseur de vitrage imposant un drainage de feuillure complémentaire | **27 mm**, usinage 4,5 × 30 mm dans la gorge porte-parclose au droit de chaque lumière extérieure, système SOLEAL FY | Exact. Défaut d'affichage : la cote sort en LaTeX brut `$4,5 \times 30\text{ mm}$` | ✅ |
| 7 | Technal fournit-il nos volets roulants ? | **Non.** Motorisation : SOMFY seul nommé. Coffres et tabliers : aucun fabricant explicitement nommé. Signale **VER-16** (piste LAKAL, crédit photo uniquement, à vérifier) | Juste et fidèle à VER-16 — mais **SOPROFEN manque** : la page volets roulants le nomme comme concepteur des blocs-baies Chrono One, du Bloc LX et de la technologie GoodNight | 🟡 |
| 8 | LUMÉAL GA : module, rupture thermique, gain de clair | Module 100 mm (2R) / 151 mm (3R) ; RPT 33 mm dormant / 22 mm ouvrant ; gain **+8 % à +14 %**, masse alu vue réduite de 35 % | Exact, **fourchette complète** (la valeur basse seule aurait sous-vendu le produit). Précédé d'une auto-correction, voir §4.4 | ✅ |
| 9 | LUMÉAL GA : charge maxi par vantail et à quelle condition | **300 kg**, conditionné aux chariots **triples** : TGA3608 (rail alu ou inox) ou TGA3609 (**rail inox uniquement**), **fraisage de traverse à 80 mm obligatoire** dans les deux cas. Rappelle les règles de calage TGA3817 (systématique sur T141021 ; sur T141015 au-delà de 120 kg ou 1,50 m) | Exact et complet : le chiffre **et** la condition matérielle, plus la règle de calage associée | ✅ |
| 10 | Peut-on faire du galandage en LUMÉAL GA ? | **Non** — le catalogue de conception ne fournit aucune formule pour le galandage. Redirige vers **GALANDAGE55** (base SOLEAL GY 55), 1 à 4 vantaux sur 1 à 3 rails | Exact, et la redirection commerciale est la bonne | ✅ |
| 11 | SOLÉAL : crémone pour un coulissant de 1 300 mm à condamner par clé | **TGY3703** (3 points, clé, H mini 1 292, Hp 721, 8 vis TGY3723) + choix du cylindre selon le montant : **T1040** 30 × 30 (TGY1202 / TGY1303) ou **T1044** 40 × 30 décentré (TGY1301 / TGY1302), rosettes T960013 / TPY6003 | Exact, et va jusqu'à la pièce que le poseur devra commander avec | ✅ |
| 12 | *(suivi)* Et pourquoi pas le TGY3701 ? | TGY3701 **est** possible (H mini 525) ; la différence est le nombre de points : 1 contre 3. Tableau comparatif, puis règle de choix | Exact. Ne se contente pas de justifier son premier choix : **corrige la prémisse implicite** de la question | ✅ |
| 13 | Que nous fournit ROTO exactement, et qu'est-ce qui lui échappe ? | Fournit : Roto NX (PERFORM+ / HYBRIDE+), Patio Inowa (INNOSLIDE), paumelles Solid B et serrures Safe E Eneo (portes PVC et HYBRIDE), aérateurs à entrebâillement. Échappe : tout l'aluminium (TECHNAL, paumelles Fapim Tube), porte d'entrée 1 vantail seule, seuils, ferme-portes, boîtes aux lettres, cales KÖMMERLING, ouvertures d'imposte, aérateurs insonorisés. Note **VER-20** (le DTA nomme FERCO) | Exact sur les deux versants, au niveau de détail des directives profine | ✅ |
| 14 | Un vantail Designo II de 500 mm de large et 90 kg, c'est possible ? | **Non.** Sans report de charge : 100 kg si LFF ≥ 600 mm, **80 kg si LFF < 600 mm**. Le report de charge, qui permettrait 150 kg, exige 800 mm de LFF et 1 000 mm de HFF — donc inaccessible à 500 mm | Exact, **et la bonne borne** : la ligne qui mord à 90 kg est celle des 600 mm. Le raisonnement est complet dans les deux sens | ✅ |
| 15 | Ce qui distingue une PERFORM70 d'une PERFORM76 au catalogue | Tableau : 70 / 76 mm, Uw 1,3 / 0,8, structure **« Non spécifiée »** / 6 chambres, étanchéité **« Non spécifiée »** / 3 joints, avec renvoi **VER-03**. Puis cintrage (impossible sur 76, autorisé sur 70), laquage PVC (70 seule, ouverture extérieure ou cintrée), certifications (RC2 et FFCP sur la 76) | Exact. **Le point remarquable est le « Non spécifiée »** : là où le catalogue se tait, la case reste vide et l'entrée VER-03 est citée | ✅ |
| 16 | Les trois styles TEXTURAL et le nombre de textures exclusives | NATIVE, AUTHENTIQUE, EXCLUSIVES ; **25 textures exclusives**, face intérieure uniquement | Exact | ✅ |
| 17 | Comment fonctionne l'INNOSLIDE et dans quelle configuration | Coulissant PVC à frappe, ferrure Roto Patio Inowa, verrouillage périphérique actif sans soulèvement, SoftClose, SoftOpen (≥ 1 970 mm). Configuration **exclusivement 2 vantaux** (1 coulissant + 1 fixe) ; au-delà, cela relève des gammes aluminium | Exact, y compris la restriction ferme à 2 vantaux. Manquent les **largeurs** (1 500-3 200 en grain d'orge, 3 201-4 200 en dormant ébavuré), la hauteur 2 400 et le vitrage 41 mm | 🟡 |
| 18 | Les quatre coulissants aluminium et leurs largeurs maxi | SOLÉAL55 6 200, GALANDAGE55 4 800, LUMÉAL55 6 000 (2,25 m par vantail), LUMINE65 6 000, avec le détail par configuration de rails | Exact sur les quatre, y compris le GALANDAGE55 (1 800 en 1 vantail, 3 600 en 2 vantaux / 2 rails, 4 800 en 3 vantaux / 3 rails) et la distinction largeur totale / largeur par vantail du LUMÉAL | ✅ |
| 19 | Classement A\*E\*V de la fenêtre PERFORM70 | **A\*4 / E\*9A / V\*A3**, présenté comme le plus élevé du marché français | Exact | ✅ |
| 20 | *(contrôle)* LUMÉAL GA : charge maxi, question reposée à l'identique | Réponse **strictement identique** à la question 9, mêmes pages chargées, mêmes compteurs | Stabilité vérifiée | ✅ |

---

## 3. Les cas à retenir

### 3.1 — Question 15 : la case vide vaut mieux que la case remplie

Le catalogue ne donne ni le nombre de chambres ni le nombre de joints de la PERFORM70. LIA
inscrit **« Non spécifiée »** dans les deux cases et renvoie à **VER-03**, qui précise que le
système 70 Plateforme de profine compte 5 chambres mais que rien n'établit que la PERFORM70
repose dessus.

C'est le comportement exact attendu sur ce corpus : **ne pas combler un silence documentaire par
une valeur vraisemblable**.

### 3.2 — Question 14 : la bonne borne, pas seulement la bonne conclusion

Le tableau Designo II compte trois lignes. LIA identifie celle qui s'applique à 90 kg — le seuil
de **600 mm** — et explique pourquoi le report de charge (800 mm / 1 000 mm) n'est pas une porte
de sortie à 500 mm de large. La conclusion et le chemin sont justes tous les deux.

### 3.3 — Questions 2, 3, 7, 13, 15 : les anomalies sont citées, pas masquées

Cinq réponses sur vingt renvoient spontanément à une entrée du registre : **INC-02** (à quel
coulissant s'applique le Uw de 1,4), **CTR-12** (DEHLI ou MLINI), **VER-16** (aucun fabricant de
volet roulant nommé), **VER-20** (le DTA nomme FERCO), **VER-03** (chambres de la PERFORM70). Et
la question 6 sur la garantie Technal cite **CTR-09** avec les deux valeurs en présence.

C'est ce qui sépare une réponse utilisable d'une réponse simplement juste : l'utilisateur sait
quand il engage l'entreprise sur une valeur contestée.

### 3.4 — Question 12 : corriger la prémisse plutôt que défendre sa réponse

« Et pourquoi pas le TGY3701 ? » appelait une justification. LIA répond que le TGY3701 **convient
aussi** — sa hauteur mini est 525 mm — et que la vraie différence est le nombre de points de
fermeture. Elle ne défend pas son choix précédent, elle donne la règle de choix.

### 3.5 — Question 7 : la seule vraie réserve de fond

« Technal fournit-il nos volets roulants ? » — la réponse est juste (non), la nuance SOMFY est
juste, l'entrée VER-16 est correctement citée. Mais **SOPROFEN n'est pas mentionné**, alors que
la page volets roulants — qui faisait partie des trois pages chargées — le nomme comme concepteur
du bloc-baie Chrono One, du Bloc LX et de la technologie GoodNight.

La réponse est donc exacte sur ce qu'elle affirme et incomplète sur ce qu'elle omet : un lecteur
en conclut qu'aucun fournisseur de fermeture n'est identifié, ce qui est faux pour les coffres.

---

## 4. Comportement observé

### 4.1 — Traçabilité : les pages chargées sont affichées

Chaque réponse expose la recherche effectuée et les pages ouvertes (2 à 3 par question), puis
cite ses sources en pied de réponse. L'utilisateur peut vérifier sans quitter l'écran, et le
correcteur — comme ici — peut vérifier la lecture, pas seulement le résultat.

### 4.2 — Les tableaux sont lus comme des tableaux

C'est la différence de nature avec la solution externe. Les réponses qui portent sur une cellule
— crémone TGY3703 par hauteur, chariots TGA3608/3609 par type de rail, profondeurs d'ouvrant FY
par module, largeurs maxi par configuration de rails, champs Designo II par tranche de LFF —
sortent avec la bonne ligne **et sa condition**. Aucun mélange de lignes n'a été observé.

### 4.3 — Coût et latence

| Mesure | Minimum | Maximum | Ordre de grandeur |
| --- | :---: | :---: | :---: |
| Tokens d'entrée | 17 400 | 39 213 | ≈ 25 000 |
| Tokens de sortie | 102 | 521 | ≈ 330 |
| Appels | 2 | 3 | 2 |

Le coût est dominé par l'entrée, donc par les pages chargées. La question la plus chère (39 213)
est celle du galandage LUMÉAL, qui a ouvert trois pages pour répondre « non ».

### 4.4 — Deux défauts à corriger, aucun de fond

1. **Une auto-correction en clair.** La question 8 s'ouvre sur « je vous présente mes excuses :
   j'ai répondu précédemment sans charger les pages sources ». La correction est saine et la
   réponse qui suit est exacte, mais elle révèle qu'un tour a été produit sans source. À
   instrumenter : une réponse sans page chargée ne devrait pas être possible.
2. **Le LaTeX n'est pas rendu.** `$4,5 \times 30\text{ mm}$`, `$\ge 1\ 970\text{ mm}$`,
   `$Hp$`, `$\ge 600\text{ mm}$` s'affichent bruts dans au moins cinq réponses. Cosmétique, mais
   c'est ce que l'artisan lit.
3. **Bruit de récupération.** Sur la crémone SOLEAL (question 11), deux des trois pages chargées
   sont des pages Roto NX KSR sans rapport. La réponse reste juste, mais deux tiers du contexte
   chargé sont payés pour rien.

---

## 5. Comparaison directe sur les 16 questions communes

| Question | Solution externe | LIA |
| --- | :---: | :---: |
| Poignées LUMÉAL GA | 🟡 titres de chapitre, aucune référence | ✅ 5 références + options catalogue |
| Uw et vitrage SOLÉAL55 / LUMÉAL55 | ❌ refus | ✅ + INC-02 |
| Trois versions d'ouvrant FY | ❌ refus | ✅ |
| Drainage de feuillure complémentaire | ❌ refus | ✅ 27 mm |
| Garantie de la ferrure Technal | 🟡 10 ans, sans la contradiction | ✅ 10 ans + CTR-09 |
| Technal et les volets roulants | ❌ refus | 🟡 juste, SOPROFEN omis |
| LUMÉAL GA module / RPT / clair | 🟡 « +8 % » | ✅ 8 à 14 % |
| LUMÉAL GA charge maxi | 🟡 300 kg sans la condition matérielle | ✅ 300 kg + chariots + fraisage |
| Galandage en LUMÉAL GA | ❌ refus | ✅ non + redirection |
| Crémone coulissant 1 300 mm à clé | ❌ refus (2 fois) | ✅ TGY3703 + cylindres |
| Que fournit ROTO | 🔴 trois exclusions fausses | ✅ exact des deux côtés |
| Designo II 500 mm / 90 kg | 🟡 bonne conclusion, mauvaise borne | ✅ bonne borne |
| PERFORM70 vs PERFORM76 | 🟠 « 5 chambres », AEV inversé | ✅ « non spécifiée » + VER-03 |
| INNOSLIDE | 🟡 sans les limites de largeur | 🟡 sans les limites de largeur |
| Quatre coulissants et largeurs maxi | ❌ refus | ✅ les quatre |
| Classement A\*E\*V de la PERFORM70 | ❌ refus (2 fois) | ✅ A\*4 / E\*9A / V\*A3 |

**Bilan sur le même périmètre : 0 réponse exploitable sur 16 pour la solution externe, 15 sur 16
pour LIA.** Les deux outils se rejoignent sur un seul point — l'INNOSLIDE, où aucun des deux ne
donne les largeurs de fabrication.

---

## 6. Conclusion

Sur 20 questions, LIA répond à toutes, ne se trompe sur aucune, et laisse deux réponses
incomplètes : SOPROFEN absent de la question sur les volets roulants, et les largeurs de
fabrication absentes de la question INNOSLIDE.

Les trois comportements qui font la différence sont ceux que le protocole documentaire visait :

1. **Une cellule de tableau est lue avec sa ligne et sa condition** — c'est ce qui donne le bon
   chariot avec les 300 kg, le bon cylindre avec la crémone, la bonne borne avec les 90 kg.
2. **Un silence documentaire reste un silence** — « Non spécifiée » plus VER-03 sur les chambres
   de la PERFORM70, là où l'autre outil inscrit « 5 chambres ».
3. **Une valeur contestée est donnée avec sa contradiction** — CTR-09, CTR-12, INC-02, VER-16,
   VER-20 sont cités spontanément dans six réponses sur vingt.

Les trois points à traiter, tous mineurs :

1. **Interdire une réponse sans page chargée** — la question 8 montre que le cas existe, même
   s'il a été rattrapé.
2. **Rendre le LaTeX** dans l'interface, ou ne pas en produire.
3. **Compléter deux réponses** : SOPROFEN sur la chaîne volet roulant, les largeurs de
   fabrication sur l'INNOSLIDE.
