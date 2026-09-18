# Update Log

## 2026-09-18

* **Ingest**: les trois PDF restés non traités dans `raw/`, renommés d'après leur contenu réel. `directives generale.pdf` ne contenait pas les directives générales : c'est le **manuel de mise en œuvre du Système 70 Plateforme** de profine, 371 pages, version septembre 2023 → `raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`. `systeme 70.pdf` → `raw/profine-plans-profiles-e-volution-2008-08.pdf`, 395 planches à l'échelle 1:1, août 2008. `roto1.pdf` → `raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, catalogue Roto NX de 451 pages, juin 2023. Aucun des trois n'est un doublon.
* **Create**: [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md) — carte des 21 registres, quatre ensembles laissés au PDF.
* **Create**: [Plans des profilés e.VOLUTION, 2008](/sources/plans-profiles-e-volution-2008.md) — superseded par le manuel de 2023 pour les cotes ; corrobore ses inerties à quinze ans d'écart.
* **Create**: [Catalogue Roto NX pour profils PVC](/sources/roto-nx-catalogue-pvc.md) — le document de commande de la ferrure, face au manuel de montage qui est le document d'atelier.
* **Create**: [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md) — 16 dormants, 11 ouvrants, 6 battements et 5 meneaux, avec 26 couples dormant/renfort et leurs inerties IG et IW. Premier relevé de cotes du système 70 dans le wiki.
* **Update**: [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md) — la **classe CDR 3**, absente du manuel de montage, et la table TBDK des forces de traction complétée de 60 à 130 kg.
* **Update**: [Incohérences internes](/anomalies/incoherences-internes.md) — **INC-12**, huit dormants du système 70 à deux largeurs dans le même manuel, écart constant de 20 mm ; **INC-13**, tableau imprimé en allemand dans le catalogue Roto français, et « RC » pour « CDR ».
* **Update**: [Contradictions entre sources](/anomalies/contradictions-entre-sources.md) — **CTR-18**, les deux documents ROTO ne donnent pas les mêmes bornes de champ d'application, sur quatre valeurs.
* **Update**: [Informations à vérifier](/anomalies/informations-a-verifier.md) — `VER-28` devient la question la plus rentable du registre : elle commande l'exploitation de 371 pages ; `VER-03` reçoit les 5 chambres du système 70, sous réserve de VER-28.
* **Update**: [Roto NX](/quincaillerie/roto-nx.md), [ROTO](/fournisseurs/roto.md), [profine](/fournisseurs/profine.md), [KÖMMERLING](/fournisseurs/kommerling.md), [DTD n° DBV-24-6/16-2335_V5](/sources/dtd-6-16-2335.md), [Posters Gamme 70](/sources/posters-kommerling-70.md) — renvois vers les nouvelles pages. Le manque « références de ferrage » de la page Roto NX est levé : elles existent au catalogue, non transcrites.
* **Note**: les cinq pages créées suivent le **registre visé par `docs/plan_claude_md_wiki_v2_2026-09-18.md`** — sujet = le produit, source en renvoi `[n p. N]`, fiches de source réduites à une carte. Elles servent de pilote avant la réécriture du reste.
* **Note**: **trois ensembles n'ont volontairement pas été transcrits**, tous pour la même raison — les en-têtes de colonnes sont des pictogrammes ou des groupes de références sur deux lignes, et le texte extrait ne permet pas d'attribuer une valeur à sa colonne : les **cotes de débit du système 70** (registre 2.3.1), les **240 pages de références du catalogue Roto** (p. 212 à 451), et les **renforts des meneaux du système 70**. Ce sont les trois premiers postes de la relecture inverse, et ils demandent le rendu des planches en image.

* **Update**: [Incohérences internes](/anomalies/incoherences-internes.md) — **INC-02 enrichie**, pas close : les trois chiffres de l'entrée sont exacts, et le 1,2 du LUMÉAL55 comme le 1,6 du LUMINE65 sont corroborés par leurs propres documents produit. Ajout de la condition d'essai omise (vitrage 6/14/4) et d'une section « INC-02 en détail » relevant tous les Uw de coulissant du corpus. Le SOLÉAL55 et le GALANDAGE55 n'en ayant aucun, le 1,4 de la page 17 pourrait être leur valeur non étiquetée plutôt qu'une contradiction.
* **Update**: [Informations à vérifier](/anomalies/informations-a-verifier.md) — ouverture de **VER-35** : les sigles « DV » et « CV » de la page 17 du catalogue général ne sont définis dans aucune source du corpus.
* **Update**: [Coulissants aluminium](/gammes/coulissants-aluminium.md) — la section devient « À quel coulissant s'applique le 1,4 W/(m².K) de la page 17 ? », avec la corroboration des valeurs produit et la piste SOLÉAL55 / GALANDAGE55.
* **Update**: [LUMINE](/gammes/lumine.md) — l'expansion de « DV » en « double vitrage » est désormais signalée comme une inférence non écrite dans la source, liée à VER-35.
* **Update**: [Incohérences internes](/anomalies/incoherences-internes.md) — **INC-01 retirée** : la coquille « 4/14/14/4 » venait du wiki, pas du catalogue. Signalée par elie, vérifiée sur `raw/catalogue-general-2026-01.pdf` — les pages 10 et 27 portent toutes deux `4/14/4/14/4`, et la notation fautive n'apparaît nulle part dans le document. L'identifiant reste réservé, section « Entrées retirées ».
* **Update**: [Performances des vitrages](/vitrages/performances-vitrages.md) — la section « Contradiction sur la composition du triple vitrage » devient un simple énoncé de composition.
* **Update**: [HYBRIDE](/gammes/hybride.md) — notation du triple vitrage corrigée en `4/14/4/14/4`, paragraphe de contradiction retiré.
* **Update**: [Catalogue menuiseries PROFERM, édition janvier 2026](/sources/catalogue-general-2026.md) — une incohérence interne au lieu de deux.
* **Ingest**: `raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, 424 pages, tome 2 du classeur profine — reçu sous le nom `76 pas encore traité.pdf`, renommé d'après sa table des matières faute de titre interne.
* **Create**: [Mise en œuvre Système 76 Advanced, profine](/sources/profine-mise-en-oeuvre-76-advanced.md) — carte des 20 registres, avec ce qui est exploitable et ce qui ne l'est pas.
* **Create**: [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md) — première source de cotes de débit du wiki, dormants, meneaux, ouvrants, battements et seuils alu.
* **Create**: [Renforts du système 76](/profiles/systeme-76-renforts.md) — les renforts acier nommés, avec leurs inerties IW et IG et le profilé qu'ils équipent.
* **Create**: [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md) — limites d'ouvrant par renfort, trois catégories de couleur, poids d'ouvrant admissibles.
* **Update**: [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md) — 76274, 76473 et 76833 identifiés, trois battements du système, huit ouvrants dont quatre au cahier PROFERM.
* **Update**: [Dormants PERFORM76](/profiles/perform76-dormants.md) — les dormants 76173 et 76178 du système profine, absents du cahier PERFORM76.
* **Update**: [Meneaux PERFORM76](/profiles/perform76-meneaux.md) — écart de cote du 76373 entre sommaire et planche de détail profine.
* **Update**: [Parcloses PERFORM76](/profiles/perform76-parcloses.md) — le manuel profine annonce 48 mm là où le DTA admet 50 mm.
* **Update**: [Directives générales profine](/sources/profine-directives-generales.md), [Posters Système 76 Advanced](/sources/posters-systeme-76-advanced.md), [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md), [profine](/fournisseurs/profine.md) — renvois vers le tome 2.
* **Update**: [Incohérences internes](/anomalies/incoherences-internes.md) — INC-09 et INC-10.
* **Update**: [Contradictions entre sources](/anomalies/contradictions-entre-sources.md) — CTR-17.
* **Update**: [Informations à vérifier](/anomalies/informations-a-verifier.md) — VER-21 à VER-26.
* **Ingest**: `raw/export_doc_122.zip` → `raw/profine-mise-en-oeuvre-9708-montants-cintres-2017-07.pdf`, note d'une page de profine France, juillet 2017.
* **Create**: [Redressement d'un montant de porte cintré par le profilé 9708](/procedures/redressement-montant-porte-cintre-9708.md) — procédure SAV complète, avec le sens de calage selon le sens du cintre.
* **Ingest**: `raw/export_doc_142.zip` et `raw/export_doc_143.zip` → les deux planches A0 `raw/poster-kommerling-70-*-2025-03.pdf`.
* **Create**: [Posters Gamme 70 KÖMMERLING, mars 2025](/sources/posters-kommerling-70.md) — premier inventaire du système 70 mm, résumés générés des archives écartés comme non sourcés.
* **Ingest**: `raw/export_doc_127.zip` → `raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf`, 60 pages, pièce jumelle du DTA déjà au wiki et non un doublon.
* **Create**: [DTD n° DBV-25-6/16-2334_V5](/sources/dtd-6-16-2334.md) — critère colorimétrique L* < 82, garnitures de joint, drainages, seuils mixtes, quincaillerie FERCO réaffirmée en mars 2025.
* **Ingest**: `raw/export_doc_124.zip` → `raw/dtd-6-16-2335-v5-e-volution.pdf`, second système profine certifié.
* **Create**: [DTD n° DBV-24-6/16-2335_V5](/sources/dtd-6-16-2335.md) — système e.XCLUSIVE / e.MOTION / e.VOLUTION, identifié comme la Gamme 70 des posters KÖMMERLING.
* **Update**: [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md) — le DTD est la pièce jumelle du DTA et conditionne le bénéfice de l'Avis Technique.
* **Update**: [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md) — le seuil L* < 82 est la raison réglementaire du renforcement des profilés sombres.
* **Update**: [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md) — 1547 et 76833 identifiés comme battements intérieurs clippés et collés.
* **Update**: [KÖMMERLING](/fournisseurs/kommerling.md), [profine](/fournisseurs/profine.md) — la Gamme 70 e.VOLUTION et les deux systèmes certifiés.
* **Update**: [Informations à vérifier](/anomalies/informations-a-verifier.md) — VER-27 à VER-30, et VER-20 renforcée par le DTD de mars 2025.
* **Delete**: les cinq archives `raw/export_doc_*.zip`, les PDF étant conservés sous leur nom.
* **Ingest**: quatre documents ROTO — `export_doc_191`, `192`, `189` et `125` → `raw/roto-nx-transformation-of-en-ob-2026-03.pdf`, `raw/roto-nx-bras-report-de-charge.pdf`, `raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf`, `raw/roto-safe-e-jonction-de-cable-2024-11.pdf`.
* **Create**: [Transformation d'un ouvrant à la française en oscillo-battant, gamme ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) — procédure PROFERM PRO-PVC-OFOB-01, document le plus récent du wiki.
* **Create**: [Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md) — montage NT Designo II et réglage du ressort au cercle plein.
* **Create**: [Contrôle d'accès 4 en 1 Roto Safe E Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) — caractéristiques, affectation des six fils, retournement du pêne, autotest et réinitialisation.
* **Create**: [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) — cinq variantes, bloc d'alimentation intégré, IP67 et les deux plages de température.
* **Update**: [ROTO](/fournisseurs/roto.md) — tableau des quatre documents, et VER-20 rappelée en clair sur la page du fournisseur.
* **Update**: [Serrure motorisée](/quincaillerie/serrure-motorisee.md), [Roto NX](/quincaillerie/roto-nx.md) — renvois vers les nouvelles pages.
* **Update**: [Informations à vérifier](/anomalies/informations-a-verifier.md) — VER-31 à VER-33.
* **Fix**: [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md) — la ligne « 80 à 155 mm » laissait croire qu'une isolation de 140 mm existait sur le dormant 76171 ; une ligne par épaisseur documentée, et rappel des trois familles tapée, appui et patte.
* **Fix**: [PERFORM](/gammes/perform.md) — dimensions maximales et nuance des fabrications certifiées portées sur la page de gamme, là où la question se pose.
* **Fix**: [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md), [ROTO](/fournisseurs/roto.md) — VER-20 signalée en clair sur les deux pages où la question de la quincaillerie se pose, et non plus seulement au registre.
* **Delete**: les quatre archives ROTO `raw/export_doc_*.zip`.
* **Ingest**: `raw/export_doc_193.zip` → `raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf`, manuel de ferrage Roto NX KSR de 124 pages, novembre 2022.
* **Create**: [Instructions de montage Roto NX KSR, PVC, novembre 2022](/sources/roto-nx-ksr-montage.md) — structure du manuel, systèmes d'axe de ferrage, tolérance de châssis fixe, fixation d'une fenêtre de sécurité.
* **Create**: [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md) — LFF, HFF et poids de vantail par type d'ouverture et par classe CDR, côtés paumelles P et Designo II, conversion épaisseur de verre en poids.
* **Create**: [Maintenance d'une ferrure Roto NX](/procedures/maintenance-ferrure-roto-nx.md) — intervalles, partage des responsabilités entre entreprise et client final, couple maximal de 10 Nm.
* **Update**: [Roto NX](/quincaillerie/roto-nx.md) — trois des quatre manques déclarés sont comblés : abaques de charge, norme anticorrosion DIN EN 13126/8, périmètre CDR du RC2.
* **Update**: [Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md) — le seuil manquant est trouvé : 100 kg sans report de charge, 150 kg avec, et compas réduit à 80 mm au-delà de 130 kg.
* **Update**: [ROTO](/fournisseurs/roto.md) — cinquième document au tableau.
* **Update**: [Incohérences internes](/anomalies/incoherences-internes.md) — INC-11, double référence de document sur chaque page du manuel Roto NX.
* **Update**: [Informations à vérifier](/anomalies/informations-a-verifier.md) — VER-34, et VER-32 recentrée sur ce qui reste inconnu.
* **Delete**: `raw/export_doc_193.zip`, le PDF étant conservé sous son nom.

## 2026-09-18

* **Create**: ingestion de quatre documents **fournisseur**, les premiers du wiki — [DTA n° 6/16-2334_V5](/sources/dta-trocal-76-advanced.md) (56 pages), [Directives générales profine](/sources/profine-directives-generales.md) (113 pages), et les deux [Posters Système 76 Advanced](/sources/posters-systeme-76-advanced.md). Pages [profine](/fournisseurs/profine.md) et [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md) créées.
* **Note**: **`VER-12` close, et l'hypothèse était fausse dans les deux sens.** Les posters profine recensent *toutes* les références du cahier PERFORM76, 76xxx comme 2xxx, 4xxx, 6xxx et 8xxx. Aucune n'appartient en propre à PROFERM : le cahier technique est une sélection commentée du catalogue profine. Tout s'approvisionne chez le fournisseur.
* **Note**: **KÖMMERLING n'est pas un fournisseur distinct** mais une marque de profine. Le DTA couvre un seul procédé sous trois marques — TROCAL 76, KBE 76 et KÖMMERLING 76 ADVANCED. [KÖMMERLING](/fournisseurs/kommerling.md) réécrite en conséquence.
* **Note**: **`CTR-02` tranchée.** Le DTA limite le procédé à **50 mm de vitrage** (p. 9) : l'absence de parclose au-delà de 50 mm n'est pas une lacune de gamme, c'est le domaine d'emploi réglementaire. Un STADIP 44²/16/4 de 64 mm sortirait de l'Avis Technique.
* **Note**: `VER-20` ouverte — le DTA nomme **FERCO** comme quincaillerie du procédé et admet d'autres quincailleries « sur justifications ». PROFERM emploie ROTO, mais la justification correspondante ne figure dans aucune source. Seul point du registre qui touche à la validité réglementaire d'une menuiserie posée.
* **Note**: le DTA apporte ce qu'aucune source PROFERM ne donnait — dimensions maximales de baie par configuration, débits de fuite par classe A\*, classement au feu **M2 nu contre M3 plaxé**, les quatre cas de renforcement obligatoire et le seuil de clarté **L\* 82** qui explique techniquement les restrictions de laquage.
* **Note**: les cotes des deux posters A0 **ne sont volontairement pas transcrites** — illisibles avec certitude à cette résolution. Ils servent de carte des références, pas de source de cotes. 107 des 113 pages des directives générales restent à dépouiller.

## 2026-09-17

* **Create**: ingestion du [Catalogue portes d'entrée](/sources/catalogue-portes-entree.md) édition mars 2024 (160 pages), extrait de `raw/export_doc_175.zip` — pages [Ouvrants de porte d'entrée](/portes/ouvrants-de-porte.md), [Panneaux et monoblocs](/portes/panneaux-et-monoblocs.md) et [Accessoires de porte d'entrée](/quincaillerie/accessoires-portes-entree.md) créées à partir de son cahier technique de 18 pages.
* **Create**: ingestion du [Nuancier stores](/sources/nuancier-stores.md) 2020 (11 pages) — page [Stores intégrés](/equipements/stores-integres.md) créée. **Famille de produits entièrement absente du wiki jusqu'ici**, et source la plus ancienne du corpus.
* **Create**: ingestion du [Nuancier des vitrages décoratifs](/sources/nuancier-vitrages-decoratifs.md), 5 pages sans date d'édition.
* **Note**: **`CTR-14` largement levée.** Le catalogue portes (p. 150) révèle que l'« Imprimé 200 » du nuancier n'est pas un vitrage décoratif au choix mais le verre de face intérieure des panneaux classiques. Les trois listes de vitrages se recouvrent une fois cette pièce replacée ; il ne reste que le Listral et le Mimosa vendables sans échantillon montrable.
* **Note**: `CTR-15` six collections de portes en mars 2024 contre deux au catalogue général, la Sélection Hexa n'apparaissant sous aucun nom en 2024. `CTR-16` garantie du panneau de porte 7 ans contre 10. `INC-08` trois désignations différentes pour la serrure de la porte aluminium dans un seul document.
* **Note**: `INC-07` classe NF EN 14501 manquante pour deux tissus de stores. `VER-18` statut commercial des stores intégrés à établir. `VER-19` nuancier vitrages sans date.
* **Update**: [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md) gagne le détail des serrures par ouvrant — 5 points à 4 galets sur le 97, 6 points dont 3 en compression sur le 118, 3 pênes pénétrants sur le SOLEAL.
* **Note**: **VOLMA** nommé pour la première fois comme fournisseur d'accessoires, et non plus comme simple crédit photo (catalogue portes, p. 146).
* **Note**: la matrice de compatibilité de la page 141 du catalogue portes (29 lignes × 10 colonnes) **n'est volontairement pas transcrite** — les symboles ne sont pas lisibles case par case avec une certitude suffisante, et une erreur sur une case ferait commander une option indisponible.

* **Create**: ingestion des trois documents de juin 2023, extraits de `raw/export_doc_179.zip`, `180` et `181` — [Dépliant général](/sources/depliant-general-2023.md) (8 pages), [Dépliant HYBRIDE](/sources/depliant-hybride-2023.md) et [Dépliant LUMÉAL](/sources/depliant-lumeal-2023.md) (4 pages chacun). Le wiki compte désormais neuf sources couvrant juin 2023 à septembre 2026.
* **Note**: `CTR-03` **quasi résolue**. La brochure PERFORM+/HYBRIDE+ annonce 20 ans en mai 2023, les trois dépliants de juin 2023 annoncent 15 ans — un mois d'écart. La lecture chronologique est définitivement écartée. Les seuls produits à 20 ans sont PERFORM+, HYBRIDE+ et la fenêtre LUMINE65, soit les trois gammes de fenêtres haut de gamme les plus récentes. Hypothèse cohérente avec les neuf documents, à confirmer au service technique.
* **Note**: `CTR-13` ajoutée — le dépliant général de juin 2023 annonce un Uw HYBRIDE de 1,3 W/m²K, le dépliant HYBRIDE du **même mois** annonce 1,2.
* **Note**: règle méthodologique dégagée et inscrite au CLAUDE.md — **sur une valeur technique, le document produit prime sur le document général**. Six contradictions du registre suivent ce schéma sans exception. Ne vaut pas pour la garantie structure.
* **Update**: [Garanties par composant](/garanties/garanties-par-composant.md) gagne le tableau d'évolution depuis 2023 — quatre postes améliorés (panneau de porte, plaxage, laquage) et un seul dégradé (volet roulant), ce qui écarte toute lecture par érosion générale.
* **Update**: [HYBRIDE](/gammes/hybride.md) — le 72 mm et le Uw 1,2 sont stables de juin 2023 à mars 2025, soit 21 mois, avant le passage à 70-76 mm et 0,8 au catalogue de janvier 2026. `CTR-06` et `CTR-07` penchent nettement vers une évolution produit fin 2025.
* **Note**: `VER-16` et `VER-17` ajoutées — LAKAL (volets roulants) et DEVGLASS (vitrage) apparaissent aux crédits photos du dépliant général. Le wiki ne connaît aucun fournisseur pour ces deux postes.

* **Create**: ingestion de la [Brochure LUMINE65](/sources/brochure-lumine65.md) édition février 2025 (6 pages), extraite de `raw/export_doc_183.zip` et renommée `brochure-lumine65-2025-02.pdf`.
* **Update**: [LUMINE](/gammes/lumine.md) passe en `status: draft` et gagne le classement A\*4/E\*9A/V\*C3 de la fenêtre LUMINE65, les 40 dB d'acoustique, les 3 joints à fil de kevlar, les applications fenêtre et porte-fenêtre, et le détail du Qualicoat classe 2 — 15 ans d'accroche et 25 ans de tenue, là où le catalogue ne retenait que 25 ans.
* **Update**: [Coulissants aluminium](/gammes/coulissants-aluminium.md) gagne les hauteurs limites du LUMINE65 (500 à 2 580 mm), absentes du catalogue, les cotes de montant central 68 mm et chicane 38 mm, et les 36 dB d'acoustique.
* **Note**: `CTR-11` classement AEV du coulissant LUMINE65 (V\*A3 contre V\*B2) et `CTR-12` poignée encastrée nommée MLINI dans la brochure et DEHLI au catalogue.
* **Note**: `CTR-03` révisée une seconde fois. La brochure LUMINE65 annonce 20 ans de garantie structure en février 2025, encadrée par deux documents qui annoncent 15 ans. **Les deux explications tentées sont fausses** : ni les gammes à ouvrant caché (le LUMÉAL en est et annonce 15 ans), ni la chronologie. Aucun critère identifiable ne sépare les documents à 20 ans de ceux à 15 ans.
* **Note**: `raw/export_doc_184.zip` non ingéré — son PDF est **strictement identique** (même empreinte MD5) au catalogue général déjà versé sous `catalogue-general-2026-01.pdf`.
* **Update**: `raw/export_doc_182.zip` versé comme `depliant-innoslide-2024-01-a4-web.pdf` — c'est la mise en page A4 web du dépliant INNOSLIDE de janvier 2024 déjà ingéré, au contenu identique. Signalé sur la page source, aucune page créée.

* **Create**: ingestion du [Dépliant LUMÉAL](/sources/depliant-lumeal.md) édition avril 2026 (2 pages), extrait de `raw/export_doc_187.zip` et renommé `depliant-lumeal-2026-04.pdf`. C'est désormais la source la plus récente du wiki.
* **Note**: trois contradictions ajoutées — `CTR-08` classement AEV du LUMÉAL (E\*7A/V\*B3 contre E\*6A/V\*B2), `CTR-09` garantie de la ferrure Technal (10 ans, là où le catalogue range Technal dans « autre ferrure » à 2 ans), `CTR-10` coloris du LUMÉAL (le Chêne doré du catalogue absent, dix coloris du dépliant absents du catalogue).
* **Note**: `CTR-04` révisée — l'hypothèse d'une réduction progressive de la garantie volet roulant de 7 à 5 ans est **falsifiée** par ce dépliant, postérieur au catalogue et qui réaffirme 7 ans. La série n'est pas monotone. Aucune valeur n'est retenue en attendant l'arbitrage.
* **Update**: [Coulissants aluminium](/gammes/coulissants-aluminium.md) passe en `status: draft` et gagne la classe 3 EN 1627-30, la perméabilité de 1,39 m³/h/m², le vitrage 28 mm, les quatre poignées et les coloris.
* **Update**: [TECHNAL](/fournisseurs/technal.md) — empreinte carbone de 2,3 kg CO₂ par kg d'aluminium, garantie ferrure de 10 ans. Question du laquage tranchée : **PROFERM laque en interne**, sur plusieurs cabines, avec couleurs sur mesure.
* **Update**: [Labels et certifications](/certifications/labels-et-certifications.md) — AFNOR identifié comme certificateur de l'Origine France Garantie, et tableau des trois niveaux de résistance à l'effraction ajouté.

* **Create**: ingestion du [Dépliant INNOSLIDE](/sources/depliant-innoslide.md) édition janvier 2024 (2 pages), extrait de `raw/export_doc_186.zip` et renommé `depliant-innoslide-2024-01.pdf`. Page [Roto Patio Inowa](/quincaillerie/roto-patio-inowa.md) créée.
* **Update**: [INNOSLIDE](/gammes/innoslide.md) gagne la hauteur maxi de 2 400 mm absente du catalogue, la précision « dormant ébavuré et ouvrant grain d'orge » pour atteindre 4 200 mm, la configuration à une partie coulissante et une partie fixe, et la quincaillerie Roto Patio Inowa.
* **Note**: `CTR-03` et `CTR-04` enrichies d'une lecture chronologique des quatre sources — les garanties structure (20 → 15 ans) et volet roulant (7 → 5 ans) décroissent avec les dates d'édition. L'hypothèse d'une réduction de garantie devient plus probable que celle de garanties propres aux gammes +.
* **Note**: `VER-15` ajoutée — ALUPLAST est crédité des photos du dépliant INNOSLIDE alors que le catalogue rattache toute la gamme PVC à KÖMMERLING. Fournisseur du profilé du coulissant à confirmer auprès du service achats.

* **Create**: ingestion de la [Brochure HYBRIDE](/sources/brochure-hybride.md) édition mars 2025 (4 pages), extraite de `raw/export_doc_185.zip` et renommée `brochure-hybride-2025-03.pdf`.
* **Note**: deux contradictions ajoutées au registre — `CTR-06` épaisseur du profilé PVC de l'HYBRIDE (72 mm en mars 2025 contre 70 à 76 mm en janvier 2026) et `CTR-07` Uw de l'HYBRIDE (1,2 contre 0,8 W/m²K). La brochure de mars 2025 ne connaît aucune déclinaison : le découpage HYBRIDE70 / HYBRIDE76 est postérieur.
* **Update**: [HYBRIDE](/gammes/hybride.md) passe en `status: draft` et gagne les deux contradictions, l'hypothèse de scission de la gamme et le détail du joint de butée périphérique.
* **Update**: `CTR-03` enrichie — la brochure HYBRIDE de mars 2025 confirme les 15 ans de garantie sur la structure, ce qui isole le 20 ans de la brochure des gammes + de mai 2023.

* **Create**: dossier `anomalies/` et nouveau `type` `Anomalie` — trois registres à identifiants stables : [Incohérences internes](/anomalies/incoherences-internes.md) (6 entrées `INC-`), [Contradictions entre sources](/anomalies/contradictions-entre-sources.md) (5 entrées `CTR-`), [Informations à vérifier](/anomalies/informations-a-verifier.md) (14 entrées `VER-`). Chaque anomalie est aussi notée en ligne sur la page où elle compte.
* **Create**: ingestion de la [Brochure Nouveautés PERFORM+ et HYBRIDE+](/sources/brochure-perform-plus-hybride-plus.md) édition mai 2023 (4 pages), extraite de `raw/export_doc_188.zip` — 4 pages créées : [PERFORM+](/gammes/perform-plus.md), [HYBRIDE+](/gammes/hybride-plus.md), [Roto NX](/quincaillerie/roto-nx.md) et la page source.
* **Note**: les gammes PERFORM+ et HYBRIDE+ n'apparaissent pas dans le catalogue général de janvier 2026, alors que la brochure date de mai 2023. Statut commercial à établir — entrée `VER-02`. Les quatre pages issues de cette source sont en `status: draft` avec `stale_after: 2024-05-31`.
* **Note**: trois contradictions de garantie entre la brochure et le catalogue — structure 20 ans contre 15, volet roulant 7 ans contre 5, laquage 7 ans forfaitaires contre la grille 25/10/7. Entrées `CTR-03` à `CTR-05`. [Garanties par composant](/garanties/garanties-par-composant.md) passe en `status: draft`.
* **Update**: `raw/` — PDF extrait du zip et renommé `brochure-perform-plus-hybride-plus-2023-05.pdf` d'après son contenu. Le zip et ses deux fichiers ColPali restent en place.

* **Create**: ingestion du [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md) version 02/09/2026 CC03 (26 pages) — 9 pages créées : 7 profilés, 1 procédure de pose, 1 quincaillerie, plus la page source.
* **Note**: contradiction entre sources sur la charge du pivot bas — 100 kg par ouvrant au cahier technique (p. 3) contre 130 kg au catalogue général (p. 6). Documentée dans [Poignée et pivot PERFORM76](/quincaillerie/perform76-poignee-et-pivot.md) et [PERFORM](/gammes/perform.md).
* **Note**: trois erreurs relevées dans le cahier technique — libellés « Dormant 76185 » sur les planches des 76171 et 76172 (p. 17 et 18), référence d'appui « 76152 » inexistante (p. 17), plages de hauteur d'ouvrant qui se chevauchent dans le tableau de position de poignée (p. 3). À signaler au service technique.
* **Note**: les vitrages de sécurité STADIP du catalogue (60 à 64 mm) dépassent la plus épaisse parclose PERFORM76 (50 mm). Compatibilité à trancher au bureau d'études — voir [Performances des vitrages](/vitrages/performances-vitrages.md).
* **Update**: [PERFORM](/gammes/perform.md), [KÖMMERLING](/fournisseurs/kommerling.md) et [Performances des vitrages](/vitrages/performances-vitrages.md) enrichies par le cahier technique — 6 chambres dormant et ouvrant, renforts acier 1,5 et 2 mm, intercalaire TGI noir, logo profine.
* **Update**: nouveau `type` `Procédure` avec son sous-dossier `procedures/`, en remplacement de `Procédure atelier` — la pose est une opération de chantier, pas d'atelier.
* **Update**: règle « Références » ajoutée au CLAUDE.md — une page par famille, une ligne par référence, jamais un fichier par référence.
* **Update**: `raw/` renommé à partir du nom interne de chaque document — `catalogue-general-2026-01.pdf` et `cahier-technique-perform76-2026-09-02-cc03.pdf`.
* **Create**: ingestion du catalogue général PROFERM édition janvier 2026 (36 pages) — 21 pages créées : [Catalogue menuiseries PROFERM](/sources/catalogue-general-2026.md), 6 gammes, 2 vitrages, 1 équipement, 2 portes d'entrée, 3 quincaillerie, 1 certifications, 1 garanties, 4 fournisseurs.
* **Create**: trois nouveaux `type` introduits par cette source — `Équipement`, `Porte d'entrée`, `Garantie` — avec leurs sous-dossiers `equipements/`, `portes/`, `garanties/`.
* **Note**: deux contradictions internes au catalogue relevées et documentées — composition du triple vitrage (p. 10 vs p. 27) et Uw des coulissants aluminium (p. 16 vs p. 17). Voir [Performances des vitrages](/vitrages/performances-vitrages.md) et [Coulissants aluminium](/gammes/coulissants-aluminium.md).
* **Note**: les 4 pages fournisseurs sont en `status: draft` — le catalogue ne donne ni coordonnées, ni références de profilés, ni abaques. À compléter quand une documentation fournisseur sera versée dans `raw/`.
* **Update**: adoption d'OKF v0.2 — format de page (frontmatter YAML), fichiers réservés et règles de citation alignés sur https://okf.md/spec/.
* **Create**: arborescence initiale — `raw/`, `wiki/`, `wiki/index.md`, `wiki/log.md`.
