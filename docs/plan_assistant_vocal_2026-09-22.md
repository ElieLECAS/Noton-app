# Plan — L'assistant vocal : le même wiki, dit à voix haute

Date : 2026-09-22 · Branche : `feature/wiki` · Écran `/vocal`, route `POST /api/vocal/tour`,
service `app/services/vocal_service.py`, consignes `app/prompts/vocal_consignes.md`.

> **Verdict en trois lignes.** Voxtral n'a pas de modèle « parole vers parole » : Mistral
> recommande un enchaînement transcription → modèle → synthèse, et c'est ce qui est fait. Le
> tour de chat ne change pas d'un iota (`WikiAnswer`, trois outils, anomalies injectées,
> citations vérifiées) ; ce qui change, c'est le prompt (la forme parlée devant les règles de
> vérité), l'entrée (un enregistrement transcrit avec le vocabulaire du wiki soufflé) et la
> sortie (une voix, phrase par phrase, dès la première phrase complète).

---

## 1. Ce qui a été mesuré avant de choisir (22/09/2026, clé de production)

| Mesure | Valeur |
| --- | --- |
| Transcription par lots (`voxtral-mini-latest`), 8 s d'audio 16 kHz | 0,44–0,66 s |
| Transcription temps réel (`voxtral-mini-transcribe-realtime-2602`), premier mot | 1,3 s après le début de la parole, `done` 0,45 s après la fin |
| Temps réel en rafale (audio envoyé plus vite que le réel) | premiers mots après **22 s** — le service suit la cadence du réel |
| Biais de vocabulaire (`context_bias`, ≤ 100 termes sans espace) | accepté par lots, **refusé** en temps réel (erreur 3051) |
| Effet du biais | « Proferme » → « PROFERM », « part clause » → « parclose » |
| Synthèse (`voxtral-mini-tts-latest`, voix `fr_marie_excited`, `pcm` en flux) | premier fragment à 0,39–0,48 s, fragments de 0,4 s d'audio |
| Format `pcm` | float32 little-endian, mono, 24 kHz ; `wav` = 16 bits 24 kHz |
| Tour complet réel (question de 9 s) | transcription 0,54 s → premier outil 1,3 s → phrase d'attente 2,6 s → premier son de réponse 4,6 s → texte fini 6,8 s |

## 2. Les décisions

1. **Transcription par lots, pas temps réel.** Le temps réel donnerait des sous-titres pendant
   qu'on parle et gagnerait une demi-seconde à la fin ; il refuse le biais de vocabulaire, et sur
   ce wiki c'est le vocabulaire qui décide si `chercher` trouve la page. Le navigateur détecte la
   fin de parole (énergie, une seconde de silence) et envoie un WAV 16 kHz.
2. **Le prompt vocal = consignes parlées + consignes générales.** Les règles de vérité restent
   écrites une seule fois (`wiki_consignes.md`) ; `vocal_consignes.md` s'ajoute devant et ne
   parle que de forme (prose, pas de tableau, les citations restent, pas d'image, la question
   vient d'une transcription). Clé de cache distincte (`lia-vocal-`), même vocabulaire, même index
   des anomalies.
3. **On ne dit rien avant qu'une page soit chargée.** Tant que `pages_lues` est vide, le texte du
   modèle peut être effacé par la relance ; il attend. Le préambule avant un appel d'outil est
   jeté avec le `reset`. Ce que l'utilisateur entend est toujours la réponse finale.
4. **Phrase par phrase.** `Phraseur` émet la première phrase seule (≥ 40 caractères), puis des
   groupes d'environ 220 caractères : une requête de synthèse par groupe, une prosodie liée. Le
   navigateur enchaîne les fragments sans trou et surligne le segment en cours.
5. **Une phrase d'attente** (« Je regarde dans le wiki. ») est dite au premier appel d'outil,
   préchauffée au démarrage et servie depuis la mémoire : le silence d'une recherche de plusieurs
   secondes est ce qui rend un assistant vocal pénible.
6. **Ce qui se dit n'est pas ce qui s'affiche.** `texte_parle` retire les chemins cités, lit
   « CTR-17 » comme « contradiction entre sources 17 », déplie « mm » et « W/m².K » ; l'écran
   garde les pastilles de pages et d'anomalies. Le texte persisté est celui du modèle.
7. **Persistance et traçabilité comme le chat.** Conversation de `mode = "vocal"` (colonne
   ajoutée par la migration `lia_vocal_mode`, listée à part), message utilisateur = la
   transcription, message assistant = le texte, ses sources et sa trace enrichie des mesures
   vocales (`transcription_ms`, `attente_ms`, `premier_son_ms`, `phrases`). La réponse est
   persistée même si l'utilisateur coupe la parole.
8. **Pas de multipart, pas de WebSocket, pas de nouvelle dépendance.** Le corps de la requête est
   l'audio lui-même ; la réponse est le même SSE que le chat, l'audio base64 y est relayé tel que
   Voxtral le rend.

## 3. Ce que l'on surveille

- La citation : sur le premier tour réel, le modèle a répondu sans citer de page (la consigne 5
  a été renforcée ; l'écran montre alors les **pages consultées**, sous ce nom).
- Le gras et les images que le modèle écrit malgré la consigne : retirés à l'oral et à l'écran,
  conservés dans le texte persisté.
- Les homophones de transcription (« performe 7-6 » pour PERFORM 76) : la consigne 7 demande au
  modèle de chercher avec le terme du métier le plus proche ; il l'a fait (`chercher parclose 44
  perform76`).
- Le coût : transcription 0,003 $/min, synthèse 0,016 $/1 000 caractères ; un tour de cinq
  phrases coûte moins d'un centime hors modèle.
