---
type: Procédure
title: Poteaux d'angle 8355, 8356, 8340 et 8341 du système 76 Advanced, avec profilé 76821
description: 'Accouplement en angle : poteau d''angle 90° 8355 (largeur DED (A) + DED (B) + 96,3 mm, embouts M344 et K 8355, renfort V263), poteau d''angle 135° 8356, poteau d''angle variable 8340 / 8341 et profilé d''adaptation 76821, avec isolants, perçages et cotes de débit.'
tags: [systeme-76-advanced, profine, joint-central, accouplement, couplage, profile-de-liaison, poteau-angle, contreventement, abaque, pression-vent, atelier]
systeme: 76
fournisseur: KÖMMERLING
usage: [atelier, pose]
famille: accouplement
status: draft
sources:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
source_pages:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    pages: 260-301
generated:
  by: process:claude-code
  at: 2026-10-21T10:43:08Z
---

Cette page reprend, du manuel de mise en œuvre Système 76 Advanced de profine (registre 2.5.2), l'accouplement en angle par les poteaux d'angle et le profilé d'adaptation 76821. Les conditions, la vue d'ensemble des solutions et la lecture des abaques sont dans [Accouplement d'éléments du système 76 Advanced : abaques et couplage par dos de dormant](/procedures/accouplement-elements-systeme-76.md).

# Poteaux d'angle

## Profilé d'adaptation 76821

**Mise en œuvre 76821** : le profilé d'adaptation 76821 est collé respectivement dans le poteau
d'angle [1 PDF p. 289, registre 2.5.2, p. 30].

![Utilisation du profilé d'adaptation 76821 dans les poteaux d'angle 90°, 135° et variable](/assets/procedures/moe-76-advanced/accouplement-poteaux-adaptation-76821.png)

De haut en bas : le poteau 90°, le poteau 135°, le poteau variable ; les flèches montrent le
profilé 76821 (bandes adhésives hachurées) glissé contre le poteau, puis le dormant.

## Poteau d'angle 90°, 8355

[1 PDF p. 290, registre 2.5.2, p. 31]

- Dimension hauteur : hauteur maxi de l'élément, y compris toutes les réhausses.
- **Dimension largeur : largeur = DED (A) + DED (B) + 96,3 mm (1).**
- **Isolation : l'isolant (set I040) doit toujours être inséré dans le poteau d'angle.**
- Liaison / axe de vissage dans le plan du profil pour la liaison dormant / dormant : **25 mm de
  l'intérieur (2)**.
- Structure : rajout d'une entretoise (3) à chaque point de vissage pour répartir la pression lors
  du serrage !
- Mise en œuvre : prépercer le dormant et la 1ʳᵉ paroi du renfort du poteau d'angle avec une mèche
  Ø 4 mm ; agrandir le trou du dormant avec une mèche Ø 6 mm ; assembler dormant et poteau d'angle,
  ou, si l'assemblage se fait avec des serre-joints, prépercer avec une mèche Ø 4 et assembler
  l'ensemble ; étanchéité des remontées capillaires (5).
- Vis : en fonction du dormant utilisé, voir tableau.

![Poteau d'angle 90° 8355 : coupe, cote 96,3 mm, axe de vissage 25 mm, entretoise 10 × 38,5 mm, profilé d'adaptation 76821](/assets/procedures/moe-76-advanced/accouplement-8355-coupe.png)

Repères : 1 cote de **96,3 mm**, 2 axe de vissage 25 mm, 3 entretoise **10 × 38,5 mm**, 4 profilé
d'adaptation 76821 (écart de **20,3 mm** dans les deux sens), 5 étanchéité. Exemple : deux 76171
avec V310 de 2 mm et le « 8355V263 » de 2,5 mm, IW = 3,2 + 14,4 + IG 2,5 = **IWtotal 20,1 cm⁴** :
l'exemple additionne l'IW du premier dormant, celle du poteau et l'IG du second dormant.

| Dormant | Renfort | IW du renfort (cm⁴), entre parenthèses l'IG | IW totale, profilés de liaison + 2 dormants (cm⁴) | Vis (mm) |
| --- | --- | --- | --- | --- |
| 76171 | V310 | 3,2 (2,5) | 20,1 | 5 × 60 |
| 76172 | V314 | 5,7 (8,4) | 28,5 | 5 × 80 |
| 76173 | V310 2× | 3,2 2× (2,5 2×) | 25,8 | 5 × 100 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 290)

![Abaque dimension maxi élément A, poteau d'angle 90° 8355](/assets/procedures/moe-76-advanced/accouplement-8355-abaque.png)

L'abaque (« Maximale Größe Element A und B ») ne représente qu'un élément A, de largeur RAM (A).
Hauteur maximale de dormant (cm), relevée comme les précédentes ; « ≈ 300 » : courbe confondue avec
la droite de 300 cm.

| Largeur dormant A, B (cm) | 0,6 kN/m² | 0,8 | 1,0 | 1,2 | 1,4 | 1,6 | 1,8 | 2,0 | 2,5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | > 300 | > 300 | > 300 | ≈ 300 | 284 | 272 | 262 | 253 | 235 |
| 60 | > 300 | > 300 | > 300 | 283 | 269 | 257 | 247 | 239 | 222 |
| 70 | > 300 | > 300 | 288 | 271 | 258 | 246 | 238 | 230 | 213 |
| 80 | > 300 | 295 | 275 | 259 | 246 | 236 | 227 | 219 | 204 |
| 90 | > 300 | 288 | 267 | 252 | 240 | 230 | 221 | 214 | 200 |
| 100 | > 300 | 278 | 259 | 244 | 232 | 222 | 214 | 208 | 194 |
| 110 | 297 | 271 | 253 | 239 | 227 | 218 | 210 | 203 | 190 |
| 120 | 289 | 264 | 246 | 232 | 222 | 212 | 206 | 198 | 186 |
| 130 | 284 | 258 | 242 | 228 | 218 | 210 | 202 | 196 | 184 |
| 140 | 278 | 254 | 238 | 224 | 214 | 206 | 200 | 194 | 182 |
| 150 | 274 | ≈ 250 | 234 | 222 | 212 | 204 | 198 | 192 | 180 |
| 160 | 270 | 248 | 232 | 220 | 210 | 202 | 196 | 190 | 179 |
| 170 | 266 | 244 | 230 | 218 | 208 | 201 | 195 | 190 | 179 |
| 180 | 264 | 242 | 228 | 216 | 208 | 200 | 194 | 189 | 178 |
| 190 | 262 | 241 | 226 | 216 | 207 | 200 | 194 | 189 | 178 |
| 200 | 260 | 240 | 226 | 215 | 207 | 200 | 194 | 189 | 178 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 290)

### Variante 1 : poteau d'angle 90° avec embout et réhausse

Cotes de débit, dimension hauteur : **poteau d'angle 8355 = DED ; renfort V263 = DED + 21 mm** ;
dimension largeur inchangée. **Fraisage** : pour l'utilisation de l'embout **M344** le profilé 8355
doit être usiné (voir des.) [1 PDF p. 291, registre 2.5.2, p. 32].

![Variante 1 : renfort V263 dépassant de 21 mm, fraisage 8,5 mm, usinage au ras de la paroi intérieure, 5 mm](/assets/procedures/moe-76-advanced/accouplement-8355-variante-1-fraisage-m344.png)

En haut, vue de face du pied : le renfort V263 dépasse de **21 mm** sous le 8355, fraisé sur
**8,5 mm**. En bas, la coupe : « usinage au ras de la paroi intérieure », retrait de **5 mm**.

Montage de l'embout M344 : coller les adaptateurs ; encoller l'embout ; mise en place de l'embout ;
fixer avec 4 vis 3,9 × 45 mm [1 PDF p. 292, registre 2.5.2, p. 33].

![Montage de l'embout M344 avec adaptateurs collés, fixation par 4 vis 3,9 × 45 mm](/assets/procedures/moe-76-advanced/accouplement-8355-variante-1-montage-m344.png)

![Montage embout M344 sur poteau d'angle 8355 avec réhausse 76753, vue latérale et vue sans profilé 8355, appui de fenêtre à 9 mm](/assets/procedures/moe-76-advanced/accouplement-8355-m344-rehausse-76753-appui.png)

Vue latérale : la réhausse, l'embout M344 sous le poteau, l'appui de fenêtre. Vue sans profilé 8355,
fixation de l'appui de fenêtre : l'appui arrêté à **9 mm** du nu du dormant [1 PDF p. 293,
registre 2.5.2, p. 34].

### Variante 2 : poteau d'angle 90° avec embout et réhausse

Mêmes cotes de débit (8355 = DED, V263 = DED + 21 mm, largeur inchangée). **Fraisage** : pour
l'utilisation de l'embout **K 8355** le profilé 8355 doit être usiné (voir des.) [1 PDF p. 294,
registre 2.5.2, p. 35].

![Variante 2 : fraisage 8,7 mm, renfort V263 à 21 mm, usinage min. 3 et min. 16 mm](/assets/procedures/moe-76-advanced/accouplement-8355-variante-2-fraisage.png)

| Cote du fraisage, variante 2 | Valeur (mm) |
| --- | --- |
| profondeur du fraisage | 8,7 |
| dépassement du renfort V263 | 21 |
| retrait de l'usinage, côté extérieur | min. 3 |
| longueur usinée, côté extérieur | min. 16 |
| longueur usinée, côté intérieur | min. 16 |
| retrait de l'usinage, côté intérieur | min. 3 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 294)

Montage de l'embout M344 : encoller l'embout ; **pas besoin d'adaptateur** ; mise en place de
l'embout ; fixer avec 4 vis 3,9 × 45 mm [1 PDF p. 295, registre 2.5.2, p. 36].

![Variante 2 : montage de l'embout M344 sans adaptateur](/assets/procedures/moe-76-advanced/accouplement-8355-variante-2-montage-m344.png)

![Variante 2 : embout M344 sur poteau 8355 avec réhausse 76753, vue latérale et fixation de l'appui de fenêtre à 21 mm](/assets/procedures/moe-76-advanced/accouplement-8355-variante-2-rehausse-76753-appui.png)

En variante 2, l'appui de fenêtre est arrêté à **21 mm** du nu du dormant [1 PDF p. 296, registre
2.5.2, p. 37]. La variante 2 nomme l'embout « K 8355 » dans son texte et « M344 » sur ses deux
planches de montage (entrée **INC-72**).

## Poteau d'angle 135°, 8356

[1 PDF p. 297, registre 2.5.2, p. 38]

- Dimension hauteur : hauteur maxi de l'élément, y compris toutes les réhausses.
- **Isolation : l'isolant (set I041) doit toujours être inséré dans le poteau d'angle.**
- Liaison / axe de vissage : **25 mm de l'intérieur (2)** ; entretoise (3) à chaque point de
  vissage.
- Mise en œuvre : prépercer le dormant et la 1ʳᵉ paroi du renfort du poteau d'angle avec une mèche
  Ø 4 mm ; agrandir le trou du dormant avec une mèche Ø 6 mm. Vis : en fonction du dormant utilisé,
  voir tableau.

![Poteau d'angle 135° 8356 : coupe, cote 46 mm, axe 25 mm, entretoise 10 × 38,5, adaptateur 76821](/assets/procedures/moe-76-advanced/accouplement-8356-coupe.png)

Repères : 1 cote de débit / largeur totale, **46 mm** ; 2 axe de vissage ; 3 entretoise
« 2 × 25,4 × 100 mm » dans la légende, dessinée **10 × 38,5 mm** ; 4 adaptateur 76821 ; 5
étanchéité remontée capillaire. Exemple : deux 76171 avec V310, le 8356 avec V262 de 2,5 mm,
IW = 3,2 + 7,6 + IG 2,5 = **IWtotal 13,3 cm⁴**.

| Dormant | Renfort | IW du renfort (cm⁴), entre parenthèses l'IG | IW totale (cm⁴) | Vis (mm) |
| --- | --- | --- | --- | --- |
| 76171 | V310 | 3,2 (2,5) | 13,3 | 5 × 60 |
| 76172 | V314 | 5,7 (8,4) | 21,7 | 5 × 80 |
| 76173 | V310 2× | 3,2 2× (2,5 2×) | 19,0 | 5 × 100 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 297)

![Abaque dimension maxi élément A, poteau d'angle 135° 8356](/assets/procedures/moe-76-advanced/accouplement-8356-abaque.png)

| Largeur dormant A, B (cm) | 0,6 kN/m² | 0,8 | 1,0 | 1,2 | 1,4 | 1,6 | 1,8 | 2,0 | 2,5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | > 300 | 298 | 277 | 261 | 248 | 237 | 228 | 220 | 205 |
| 60 | > 300 | 281 | 262 | 246 | 234 | 224 | 216 | 208 | 194 |
| 70 | 298 | 271 | 252 | 238 | 226 | 216 | 208 | 201 | 187 |
| 80 | 284 | 259 | 241 | 227 | 216 | 207 | 200 | 193 | 180 |
| 90 | 274 | ≈ 250 | 233 | 220 | 209 | 200 | 194 | 187 | 174 |
| 100 | 266 | 243 | 226 | 214 | 204 | 196 | 188 | 182 | 170 |
| 110 | 260 | 237 | 220 | 208 | 199 | 192 | 184 | 178 | 167 |
| 120 | 254 | 232 | 216 | 204 | 196 | 188 | 182 | 176 | 164 |
| 130 | 248 | 227 | 212 | 201 | 192 | 185 | 179 | 173 | 163 |
| 140 | 244 | 224 | 210 | 198 | 190 | 183 | 177 | 172 | 162 |
| 150 | 240 | 221 | 208 | 196 | 188 | 182 | 176 | 171 | 161 |
| 160 | 238 | 219 | 206 | 195 | 187 | 180 | 175 | 170 | 161 |
| 170 | 236 | 217 | 204 | 194 | 186 | 180 | 175 | 170 | 161 |
| 180 | 234 | 216 | 203 | 194 | 186 | 180 | 175 | 170 | 161 |
| 190 | 232 | 215 | 202 | 193 | 186 | 180 | 175 | 170 | 161 |
| 200 | 231 | 214 | 202 | 193 | 186 | 180 | 175 | 170 | 161 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 297)

## Poteau d'angle variable, 8340 / 8341

[1 PDF p. 298-299, registre 2.5.2, p. 39-40]

- Dimension hauteur : hauteur maxi de l'élément, y compris toutes les réhausses.
- Cotes de débit X et Y selon l'angle : tableau de
  [Élargisseurs et assemblage PERFORM76](/profiles/perform76-elargisseurs-et-assemblage.md#cotes-et-renforts-des-poteaux-dangle),
  de 90° à 180° tous les 5°.
- **Isolation : les isolants (set I042 et set I043) sont toujours utilisés pour les assemblages avec
  des profilés blancs, par contre ne sont pas utilisés pour les profilés couleurs !**
- Liaison / axe de vissage : **25 mm de l'intérieur (2)**.
- Structure : rajout d'une entretoise **10 × 38,5 mm** (3) à chaque point de vissage pour répartir
  la pression lors du serrage !
- Mise en œuvre : prépercer le dormant et la 1ʳᵉ paroi du renfort du poteau d'angle avec une mèche
  Ø 4 mm ; agrandir le trou du dormant avec une mèche Ø 6 mm. Pour prépercer, utiliser le gabarit
  **T000** (voir des.).
- Vis : en fonction du dormant utilisé, voir tableau page 15.

![Poteau d'angle variable 8340 / 8341 : coupe à 140°, cotes X et Y, axe 25 mm, entretoise 10 × 38,5, adaptateur 76821](/assets/procedures/moe-76-advanced/accouplement-8340-8341-coupe.png)

Exemple : deux 76171 avec V310, le « 8340/8341 » avec « 640 (K640) » de 2,5 mm, IW = 3,2 + 8,7 +
IG 2,5 = **IWtotal 14,4 cm⁴**.

| Dormant | Renfort | IW du renfort (cm⁴), entre parenthèses l'IG | IW totale (cm⁴) | Vis (mm) |
| --- | --- | --- | --- | --- |
| 76171 | V310 | 3,2 (2,5) | 14,4 | 5 × 60 |
| 76172 | V314 | 5,7 (8,4) | 22,8 | 5 × 80 |
| 76173 | V310 2× | 3,2 2× (2,5 2×) | 20,1 | 5 × 100 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 298)

Le renfort du poteau variable est écrit « **640 (K640)** » dans l'exemple, **V265** dans la vue
d'ensemble ; le gabarit **T000** n'est dessiné nulle part ; « voir tableau page 15 » renvoie au
tableau du 76608 (entrée **INC-72**).

![Abaque dimension maxi élément A, poteau d'angle variable 8340 / 8341](/assets/procedures/moe-76-advanced/accouplement-8340-8341-abaque.png)

| Largeur dormant A, B (cm) | 0,6 kN/m² | 0,8 | 1,0 | 1,2 | 1,4 | 1,6 | 1,8 | 2,0 | 2,5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | > 300 | > 300 | 284 | 268 | 255 | 244 | 234 | 226 | 210 |
| 60 | > 300 | 291 | 270 | 255 | 242 | 232 | 224 | 216 | 200 |
| 70 | > 300 | 278 | 258 | 242 | 231 | 221 | 213 | 206 | 191 |
| 80 | 291 | 265 | 246 | 232 | 221 | 212 | 204 | 197 | 184 |
| 90 | 282 | 257 | 239 | 226 | 215 | 206 | 198 | 192 | 179 |
| 100 | 274 | ≈ 250 | 232 | 220 | 209 | 200 | 194 | 187 | 174 |
| 110 | 266 | 243 | 226 | 214 | 204 | 196 | 189 | 184 | 172 |
| 120 | 260 | 238 | 222 | 210 | 200 | 192 | 186 | 180 | 168 |
| 130 | 255 | 234 | 218 | 206 | 198 | 190 | 183 | 178 | 167 |
| 140 | ≈ 250 | 230 | 214 | 204 | 194 | 188 | 181 | 176 | 166 |
| 150 | 247 | 226 | 212 | 202 | 193 | 186 | 180 | 175 | 164 |
| 160 | 244 | 224 | 210 | 200 | 192 | 184 | 179 | 174 | 164 |
| 170 | 242 | 222 | 209 | 198 | 190 | 184 | 178 | 174 | 164 |
| 180 | 239 | 221 | 208 | 198 | 190 | 184 | 178 | 174 | 164 |
| 190 | 238 | 220 | 207 | 197 | 190 | 184 | 178 | 174 | 164 |
| 200 | 237 | 219 | 207 | 197 | 190 | 184 | 178 | 174 | 164 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 298)

# Citations

[1] [Mise en œuvre Système 76 Advanced, profine](raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf),
registre 2.5.2, PDF p. 260 à 301

# Voir aussi

Cette page est une partie d'un sujet qui en compte 6 ; les autres :

- [Accouplement d'éléments du système 76 Advanced : abaques et couplage par dos de dormant](/procedures/accouplement-elements-systeme-76.md)
- [Couplage vertical 180° avec profilé de liaison H 76605, système 76 Advanced](/procedures/accouplement-elements-systeme-76-profile-de-liaison-h-76605.md)
- [Couplage vertical 180° avec profilé de liaison H 76608, système 76 Advanced](/procedures/accouplement-elements-systeme-76-profile-de-liaison-h-76608.md)
- [Couplage vertical 180° avec contreventement A250 ou renfort V477, système 76 Advanced](/procedures/accouplement-elements-systeme-76-contreventement-a250-renfort-v477.md)
- [Habillage de contreventement des couplages du système 76 Advanced, renforts V261 et V290](/procedures/accouplement-elements-systeme-76-habillage-de-contreventement.md)
