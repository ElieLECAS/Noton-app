---
type: Profilé
title: Abaques dimensionnels du système 70
description: Les limites de largeur, hauteur et poids d'ouvrant admissibles du système 70 Plateforme selon le renfort acier, la couleur du profilé (blanc vs plaxé) et le type d'ouverture.
tags: [profine, systeme-70, abaque, dimension, limite, blanc, couleur, plaxage, ouvrant]
systeme: 70
famille: abaques
status: stable
sources:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
    last_modified: 2023-09-30
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    id: profine-plans-e-volution-2008
    title: Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008
    last_modified: 2008-08-31
source_pages:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    pages: 128-146
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 120-137
generated:
  by: process:multimodal-direct
  at: 2026-09-19T20:55:00Z
---

# Abaques dimensionnels du système 70 Plateforme

Les abaques dimensionnels fixent les limites de fabrication admissibles en largeur (LFF) et hauteur (HFF)
pour les fenêtres et portes-fenêtres du système 70 Plateforme [1 p. 128].
Ces limites garantissent la tenue aux déformations d'usage et le bon fonctionnement de la quincaillerie (poids maximal de vantail $\le 100\text{ kg}$ en standard, $\le 130\text{ kg}$ ou $150\text{ kg}$ avec ferrures renforcées).

# Pourquoi la couleur restreint les dimensions

Les profilés plaxés ou teintés subissent un échauffement superficiel sous rayonnement solaire pouvant atteindre 70 à 80 °C,
contre 40 à 50 °C pour les profilés blancs [1 p. 128].
Cette dilatation thermique impose trois contraintes strictes :
1. **Réduction systématique des dimensions limites** : la surface maximale d'un vantail couleur est réduite de **15 à 20 %** par rapport au blanc.
2. **Renforcement obligatoire de tous les profilés couleur** : aucun profilé dormant ou ouvrant de couleur ne peut être débité sans armature acier, y compris pour les petites dimensions [2 p. 120].
3. **Perçages de décompression et ventilation** : orifices oblongs de $5 \times 25\text{ mm}$ ou perçages $\ge \varnothing 5\text{ mm}$ obligatoires en préchambre extérieure pour évacuer la chaleur accumulée.

# Limites dimensionnelles des fenêtres à un vantail (OF et OB)

Relevées sur les courbes du registre 2.3.3 (p. 130-136) pour ouvrant standard 6112 (70 mm) avec renfort acier V604 (1,5 mm) ou V605 (2,0 mm) :

### Profilés blancs (vitrage standard $\le 30\text{ kg/m}^2$)

| Largeur fond de feuillure LFF (mm) | Hauteur maximale HFF avec renfort 1,5 mm (mm) | Hauteur maximale HFF avec renfort 2,0 mm (mm) | Poids de vantail maxi (kg) |
| --- | --- | --- | --- |
| 600 | 2 300 | 2 450 | 100 |
| 800 | 2 300 | 2 450 | 100 |
| 1 000 | 2 200 | 2 400 | 100 |
| 1 200 | 1 950 | 2 250 | 100 |
| 1 400 | 1 650 | 1 950 | 100 (ou 130 kg renforcé) |
| 1 500 (borne max LFF) | 1 400 | 1 700 | 100 (ou 130 kg renforcé) |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 132)

### Profilés de couleur / plaxés (vitrage standard $\le 30\text{ kg/m}^2$)

| Largeur fond de feuillure LFF (mm) | Hauteur maximale HFF avec renfort 1,5 mm (mm) | Hauteur maximale HFF avec renfort 2,0 mm (mm) | Poids de vantail maxi (kg) |
| --- | --- | --- | --- |
| 600 | 2 150 | 2 250 | 100 |
| 800 | 2 100 | 2 250 | 100 |
| 1 000 | 1 950 | 2 150 | 100 |
| 1 200 | 1 700 | 1 900 | 100 |
| 1 350 (borne max LFF) | 1 400 | 1 650 | 100 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 134)

# Limites des fenêtres à deux vantaux (OF / OB)

Pour une fenêtre à deux vantaux avec battement central 6130 ou 6132 :
- **Largeur totale de baie maximale en blanc** : **2 400 mm** (deux vantaux de LFF 1 150 mm).
- **Largeur totale de baie maximale en couleur** : **2 100 mm** (deux vantaux de LFF 1 000 mm).
- **Hauteur maximale** : **2 250 mm** avec renfort 2,0 mm et verrouilleur intermédiaire de battement [1 p. 138].

# Portes-fenêtres avec traverse de soubassement

Pour les portes-fenêtres de hauteur $H > 2\,000\text{ mm}$ :
- L'utilisation d'une traverse intermédiaire 6127 soudée ou mécanisée est obligatoire dès que le vantail dépasse $2\,200\text{ mm}$ de hauteur en couleur pour stabiliser les montants d'ouvrant [2 p. 126].
- Au-delà de $100\text{ kg}$ de poids de vantail (par exemple avec vitrage phonique ou retardateur d'effraction 44.2), le montage d'un côté paumelles 130 kg ou d'un [Report de charge](/procedures/report-de-charge-roto-nx.md) est obligatoire.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 — `raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.3, p. 128 à 146
[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 — `raw/profine-plans-profiles-e-volution-2008-08.pdf`, p. 120 à 137

# Voir aussi

- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Cotes de débit du système 70](/profiles/systeme-70-cotes-de-debit.md)
- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
