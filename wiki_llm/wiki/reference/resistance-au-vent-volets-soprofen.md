---
type: Référence
title: Résistance au vent des volets roulants SOPROFEN
description: Abaques de résistance au vent des lames PVC et aluminium SOPROFEN (classes V*2 à V*6 selon NF EN 13659) et matrice de prescription par zone géographique, catégorie de terrain et hauteur selon le DTU 34-2.
tags: [reference, vent, resistance-au-vent, volet-roulant, soprofen, dtu-34-2, nf-en-13659, cstb]
fournisseur: SOPROFEN
usage: chiffrage
status: stable
sources:
  - resource: raw/moustiquaires/export_doc_132.zip
    id: soprofen-guide-technique-blocs-baies-2022
    title: Guide technique Blocs-baies SOPROFEN 2022, réf. DOC86505
    last_modified: 2022-09-01
source_pages:
  - resource: raw/moustiquaires/export_doc_132.zip
    pages: 144
generated:
  by: process:gemini-coder
  at: 2026-09-21T14:20:00Z
---

# Cadre normatif et méthodologie de dimensionnement

La résistance au vent des tabliers de volets roulants relève de la norme européenne **NF EN 13659** (*Fermetures pour baies équipées de fenêtres*) et du document technique unifié **NF DTU 34-2** [1 p. 144].
Pour déterminer la classe de résistance au vent requise sur un bâtiment, trois critères doivent être croisés [1 p. 144] :
1. **La zone géographique de vent** (Zones 1 à 4 de la carte nationale).
2. **La catégorie de terrain / rugosité** (de la zone urbaine protégée IV au front de mer 0).
3. **La hauteur de la fermeture au-dessus du sol** (tranches de 0 à 100 m).

---

# Abaques de résistance des lames SOPROFEN (NF EN 13659)

Les bancs d'essai des usines SOPROFEN établissent les largeurs maximales admissibles entre coulisses selon la classe de vent (en millimètres) pour deux hauteurs types sous coffre ($H = 2\,250\text{ mm}$ et $H = 1\,400\text{ mm}$) [1 p. 144] :

| Profilé de lame | Matériau | Pas (mm) | V\*2 ($H \le 2250$ / $H \le 1400$) (mm) | V\*3 ($H \le 2250$ / $H \le 1400$) (mm) | V\*4 ($H \le 2250$ / $H \le 1400$) (mm) | V\*5 ($H \le 2250$ / $H \le 1400$) (mm) | V\*6 ($H \le 2250$ / $H \le 1400$) (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L 37 | PVC | 37 | 1 300 / 1 300 | 1 200 / 1 200 | 1 000 / 1 000 | 850 / 850 | Non admissible |
| LA 37 | Aluminium | 37 | 2 400 / 2 400 | 2 300 / 2 300 | 1 800 / 2 100 | 1 500 / 1 800 | Non admissible |
| L 50 | PVC | 50 | 1 600 / 1 800 | 1 400 / 1 500 | 1 150 / 1 300 | 900 / 1 000 | Non admissible |
| LA 50 | Aluminium standard | 50 | 2 700 / 3 200 | 2 500 / 3 000 | 2 300 / 2 800 | 1 950 / 2 600 | 1 800 / 2 400 |
| LA 50 (CTA13) | Aluminium + coulisse profonde | 50 | 3 400 / 3 400 | 2 900 / 2 900 | 2 500 / 2 500 | 2 300 / 2 300 | 1 900 / 1 900 |

(schéma: raw/moustiquaires/export_doc_132.zip, p. 144)

*Coulisses profondes CTA13* : l'emploi de coulisses de grande profondeur CTA13 ($54,7 \times 29\text{ mm}$) permet d'augmenter significativement la retenue latérale des lames LA50 sous vents violents (classes V\*4 à V\*6) [1 p. 144].

---

# Classes de vent minimales à prescrire (DTU 34-2)

La matrice suivante donne la classe minimale de résistance au vent (V\*2 à V\*5) selon l'exposition du bâtiment [1 p. 144] :

| Zone de vent | Catégorie de terrain | $H \le 9\text{ m}$ | $9 < H \le 18\text{ m}$ | $18 < H \le 28\text{ m}$ | $28 < H \le 50\text{ m}$ | $50 < H \le 100\text{ m}$ |
| --- | --- | --- | --- | --- | --- | --- |
| Zone 1 | IV (Urbain dense) | V\*2 | V\*2 | V\*2 | V\*3 | V\*3 |
| Zone 1 | IIIb (Industriel / périurbain) | V\*2 | V\*2 | V\*3 | V\*3 | V\*4 |
| Zone 1 | IIIa (Bocage, habitat dispersé) | V\*2 | V\*3 | V\*3 | V\*3 | V\*4 |
| Zone 1 | II (Rase campagne) | V\*3 | V\*3 | V\*3 | V\*4 | V\*4 |
| Zone 1 | 0 (Bord de mer / grands lacs) | V\*3 | V\*4 | V\*4 | V\*4 | V\*4 |
| Zone 2 | IV (Urbain dense) | V\*2 | V\*2 | V\*2 | V\*3 | V\*4 |
| Zone 2 | IIIb (Industriel / périurbain) | V\*2 | V\*3 | V\*3 | V\*3 | V\*4 |
| Zone 2 | IIIa (Bocage, habitat dispersé) | V\*3 | V\*3 | V\*3 | V\*4 | V\*4 |
| Zone 2 | II (Rase campagne) | V\*3 | V\*4 | V\*4 | V\*4 | V\*4 |
| Zone 2 | 0 (Bord de mer / grands lacs) | V\*4 | V\*4 | V\*4 | V\*4 | V\*5 |
| Zone 3 | IV (Urbain dense) | V\*2 | V\*2 | V\*3 | V\*3 | V\*4 |
| Zone 3 | IIIb (Industriel / périurbain) | V\*2 | V\*3 | V\*3 | V\*4 | V\*4 |
| Zone 3 | IIIa (Bocage, habitat dispersé) | V\*3 | V\*4 | V\*4 | V\*4 | V\*4 |
| Zone 3 | II (Rase campagne) | V\*4 | V\*4 | V\*4 | V\*4 | V\*5 |
| Zone 3 | 0 (Bord de mer / grands lacs) | V\*4 | V\*4 | V\*4 | V\*5 | V\*5 |
| Zone 4 | IV (Urbain dense) | V\*3 | V\*3 | V\*3 | V\*4 | V\*4 |
| Zone 4 | IIIb (Industriel / périurbain) | V\*3 | V\*3 | V\*4 | V\*4 | V\*4 |
| Zone 4 | IIIa (Bocage, habitat dispersé) | V\*3 | V\*4 | V\*4 | V\*4 | V\*5 |
| Zone 4 | II (Rase campagne) | V\*4 | V\*4 | V\*4 | V\*5 | V\*5 |
| Zone 4 | 0 (Bord de mer / grands lacs) | V\*4 | V\*5 | V\*5 | V\*5 | V\*5 |

(schéma: raw/moustiquaires/export_doc_132.zip, p. 144)

---

# Citations

[1] [Guide technique Blocs-baies SOPROFEN 2022, réf. DOC86505](raw/moustiquaires/export_doc_132.zip), p. 144

---

# Voir aussi

- [Classification de la résistance au vent](/reference/classification-resistance-au-vent.md)
- [Classification A*E*V* à préconiser par site](/reference/classification-aev-par-site.md)
- [Volets roulants](/equipements/volets-roulants.md)
- [SOPROFEN](/fournisseurs/soprofen.md)
- [Avis Technique CSTB n° 6/16-2339_V2, coffres Chrono SOPROFEN](/certifications/at-cstb-6-16-2339-coffres-chrono.md)
