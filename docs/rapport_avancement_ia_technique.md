# État d’avancement du projet Chatbot Technique PROFERME

**Objet : Rapport de situation structuré et professionnel**

Monsieur le Président,

Je fais suite à nos récents échanges concernant le développement de notre assistant technique. Ce projet a franchi une étape de validation majeure et je souhaite vous partager un bilan transparent des forces actuelles du système, ainsi que les axes de travail que nous suivons pour atteindre une fiabilité totale avant le déploiement.

---

## 1. Comprendre le moteur de notre assistant : LLM et Graphe de Connaissances

Avant de détailler nos avancées, il est important de rappeler le fonctionnement du cœur de notre système, basé sur un Grand Modèle de Langage (LLM). De base, une telle intelligence artificielle est une « coquille vide » : elle possède une excellente capacité de compréhension et de rédaction, mais elle est totalement incapable de lire des documents par elle-même ou de connaître nos produits. 

Pour la rendre pertinente, nous avons dû concevoir des outils informatiques spécifiques. Ces outils lui donnent la capacité d'aller naviguer, d'interroger et de lire nos documentations, que nous avons préalablement converties sous forme de texte et stockées dans une base de données. 

Pour aller plus loin, nous structurons ces informations sous forme de **Graphe de Connaissances**. Il s'agit d'une cartographie qui relie logiquement nos concepts métier entre eux (par exemple, lier une gamme à ses contraintes de pose ou à ses accessoires compatibles). L'avantage de cette approche est immense : elle permet à l'IA de ne pas lire une information de manière isolée, mais de comprendre son contexte et les relations entre nos produits, réduisant ainsi drastiquement le risque d'erreurs.

---

## 2. État du projet : Une dualité de performance entre Recherche et Raisonnement

À ce stade, nous disposons d'un socle technologique robuste capable d'indexer avec fluidité des milliers de pages de notices (Technal, Profine, catalogues internes). 

Le bilan est globalement très positif pour l’extraction d’informations textuelles ou commerciales directes. Le système est un excellent outil d'assistance rapide pour les services Marketing ou Commerciaux. En revanche, le système montre encore des fragilités dès qu'il s'agit d'appliquer une logique métier complexe (seuils, tolérances, interprétations de schémas). Pour le bureau d'études ou la production, le chatbot nécessite encore des réglages avant de pouvoir se substituer à une vérification humaine.

---

## 3. Analyse détaillée des résultats : Succès et limites identifiées

Les tests de stress poussés que nous avons menés révèlent un comportement binaire de l'intelligence artificielle, illustré par des exemples concrets :

### ✅ Les Réussites (Fiabilité confirmée)
Le système excelle de façon remarquable lorsque l'information est explicite, isolée ou de nature institutionnelle.

*   **Extraction administrative et normative :** L'outil fait preuve d'une très grande précision.
    *   *Exemples :* Identification parfaite de l'usine de Douvrin (ainsi que la nature de la menuiserie recyclée à 75%), repérage de l'altitude maximale de pose à 900 m, ou encore validité du DTA fixée à 2029.
*   **Nomenclature et variantes directes :** Très bon pour associer précisément un nom de produit à une référence.
    *   *Exemples :* Association réussie du vitrage 32 mm avec la cale marron T710002 et le joint rose T411009. Sur le Noir Carbone, l'IA a même su fouiller les documentations pour proposer une alternative de laquage sur mesure.
*   **Synthèse de concepts complexes :** L'assistant restitue parfaitement les sigles et innovations.
    *   *Exemples :* Explication impeccable du test AEV (Air, Eau, Vent) ou de la gamme TEXTURAL® (compréhension fine du système hybride PVC intérieur / Alu extérieur sans mélanger les faces).

### 🔴 Les Échecs (Points de vigilance critiques)
Les erreurs surviennent principalement lors de la lecture de tableaux denses, ou quand l'IA tente de "combler" son ignorance par des connaissances générales acquises sur le web (hallucinations).

*   **Le risque d'imprécision technique (Hallucinations) :** C'est le point le plus sensible pour la production.
    *   *Exemples :* Sur le soudage PVC (exigé à 245-250°C pour 25s), l'IA a arrondi à "240-270°C pendant quelques dizaines de secondes", ce qui est inacceptable pour un réglage machine. Sur la gamme GRAPHITE, elle a inventé une épaisseur de 42mm et un Uw de 0,8 au lieu des réels 80mm et 0,36. Sur la gamme TEXTURAL, elle a inventé des finitions "Marbre" et "Béton" au lieu de "Cuir" ou "Carbone".
*   **Confusion sur les Seuils Logiques et Conditions :** Le système peine à appliquer des mathématiques simples ou des exceptions de pose.
    *   *Exemples :* Il a affirmé qu'il fallait 2 perçages pour un profil de 1,50 m, ignorant la règle critique du "rajouter un perçage au milieu si Longueur > 1 m". Autre cas, il a estimé que les renforts de dormants étaient facultatifs "pour les petites fenêtres", omettant la condition vitale de la notice : "uniquement si fixé à la maçonnerie".
*   **Glissements de lecture et historique temporel :**
    *   *Exemples :* Dans un tableau dense, le système s'est trompé de ligne, attribuant un Uw de 1,7 (valeur du 4 vantaux) au lieu de 1,5 (valeur du 2 vantaux). De plus, il a confondu l'historique des garanties des volets roulants, lissant les données et ignorant la baisse de garantie sur le tablier entre 2023 et 2024.
*   **Angles morts documentaires :** Des schémas invisibles (ex: le fond de joint XL30202 uniquement présent en cartouche de dessin) et un échec d'indexation totale sur le catalogue de la gamme INNOSLIDE qu'elle juge "indisponible".

---

## 4. Tableau de bord des performances actuelles

Pour synthétiser la maturité de l'outil, voici l'évaluation de notre pipeline de données :

| Dimension évaluée | Score global | Constat |
| :--- | :---: | :--- |
| **Recherche Textuelle & Faits Simples** | 19 / 20 🌟 | Excellente. Très fiable sur les marques, labels et composants directs. |
| **Identification Codes & Synthèse** | 15 / 20 ✅ | Bonne. Capable d'expliquer des concepts techniques globaux. |
| **Extraction de Tableaux Complexes** | 12 / 20 ⚠️ | Moyenne. Risque de "saut de ligne" ou de mélange de données. |
| **Calculs, Seuils et Nuances Métier** | 08 / 20 ❌ | Faible. Tendance à ignorer les conditions strictes (exceptions/règles). |

---

## 5. Plan d'action et recommandations pour le déploiement technique

Pour pallier ces faiblesses techniques et sécuriser l'utilisation de l'outil par toutes nos équipes, mes priorités d'optimisation sont définies :

*   **Cadrage strict de l'IA ("Prompt System" & Citations) :** Le système se comporte parfois trop comme un consultant généraliste. Nous allons abaisser drastiquement sa "créativité" technique. S'il manque une donnée, il aura l'obligation stricte de répondre "Je ne sais pas". Nous forcerons également l'IA à citer systématiquement le fichier et le numéro de page justifiant sa réponse.
*   **Amélioration du découpage des données ("Chunking") :** Nous convertissons actuellement nos tableaux en formats lisibles par la machine (Markdown/JSON) pour lier informatiquement l'en-tête de la colonne à sa valeur. Un audit complet du fichier PDF INNOSLIDE et l'extraction de texte sur les schémas d'assemblage sont en cours.
*   **Ajout d'une étape de raisonnement (CoT - "Chain of Thought") :** Pour résoudre les échecs sur les seuils, nous allons forcer l'IA à décrire son raisonnement étape par étape avant de donner la réponse finale.
*   **Graphe de Connaissances (KAG) et Filtrage Temporel :** Nous ajoutons des métadonnées temporelles rigides pour empêcher le croisement d'anciennes garanties avec les nouvelles. Enfin, nous renforçons les liens dans la base pour que les caractéristiques d'une gamme standard ne soient plus jamais diluées avec nos produits premium.

---

Le projet dispose d'une fondation excellente. Nous entrons à présent dans la phase d'ajustements chirurgicaux, indispensables pour transformer ce lecteur hyper-rapide en un véritable expert technique sécurisé pour PROFERME.

Je reste à votre entière disposition pour échanger sur ces points.

Respectueusement,

[Votre Nom]  
Responsable du Projet IA Technique
