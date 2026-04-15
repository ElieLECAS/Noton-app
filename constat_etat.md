# Évaluation de viabilité production (2026-04-15)

## Périmètre revu
- Pipeline d’upload bibliothèque + queue + stop/skip/reindex.
- Ajout/retrait documents ↔ espaces.
- Entités KAG et contraintes DB.
- RBAC et surfaces de sécurité applicative.
- Exécution des tests `pytest` en local.

## Verdict global
- **Application globalement cohérente**, avec des garde-fous importants contre le retraitement en boucle (run_id, statuts cancel/skip, revoke Celery).
- **Risque principal non-fonctionnel**: sécurité session/API (cookie non secure, CORS permissif en fallback, points de contrôle d’accès incomplets dans certains services).
- **Risque principal fiabilité**: quelques zones de concurrence/atomicité (uploads partiels, modifications d’espaces asynchrones sans idempotence forte).

## Points solides
1. **Anti-boucle/retraitement**: `processing_run_id` est régénéré et invalidé lors des stop/skip/reindex, ce qui permet de rendre obsolètes les jobs en retard.
2. **Stop/skip admin**: endpoints dédiés + révocation de tâches Celery + marquage DB (`cancelled_by_user`/`skipped`).
3. **Backpressure queue**: worker unique (thread) et préfetch Celery à 1, ce qui évite le chevauchement excessif.
4. **Contraintes KAG**: index d’unicité et garde-fou sur score relation.

## Détails à risque (petits angles morts)

### 1) Sécurité session / web
- Cookie d’auth créé avec `secure=False` (risque en prod si HTTP/terminaison TLS mal gérée).
- En fallback CORS sans config, toutes origines sont autorisées (`*`).
- `decode_token` imprime des infos de debug côté serveur (bruit et risque fuite d’info dans logs).

### 2) Contrôle d’accès horizontal (service layer)
- Plusieurs services de documents/liaisons espaces ignorent `user_id` en requête DB (paramètre présent mais non filtrant), donc la sécurité repose fortement sur la couche router/RBAC.
- En cas d’appel interne futur mal protégé, il existe un risque d’accès/modification inter-utilisateurs.

### 3) Queue/arrêt: comportement globalement bon mais à durcir
- `stop-all` révoque largement les tâches d’une bibliothèque (bon) mais nécessite que l’inspection Celery réponde correctement; sinon la révocation peut être partielle (le statut DB reste toutefois annulé).
- Les tâches Celery principales ont `max_retries=1`, ce qui limite les boucles infinies, mais un échec systémique peut quand même provoquer un doublon de traitement (1 retry).

### 4) Upload / bibliothèque
- Upload séquentiel fichier par fichier: robuste mais potentiellement lent à gros volume.
- Pas de validation explicite taille/type MIME côté upload (risque saturation I/O/CPU ou fichiers inattendus).
- En cas d’erreurs partielles multi-fichiers, certains docs sont créés et d’autres non (comportement attendu mais non transactionnel global).

### 5) Ajout/retrait documents ↔ espaces
- Mécanisme asynchrone correct (queue task dédiée), mais le traitement est “best effort”: si un sous-ajout échoue, la cohérence finale dépend de l’état intermédiaire.
- Pas de verrou métier explicite contre des opérations concurrentes contradictoires (add/remove simultanés sur même document/espace).

### 6) Entités KAG
- Bonne base de contraintes (unicité + index + check relevance).
- Surveiller le coût des reconstructions KAG lors des changements d’espaces multiples (peut impacter latence worker).

### 7) Tests
- Le run `pytest` **n’a pas pu démarrer** dans cet environnement (dépendance `alembic` absente), donc impossible d’attester “tout vert” ici.
- Le projet contient toutefois une couverture ciblée sur stop/skip/reindex/RBAC et espaces.

## Recommandations “avant prod” (priorité)
1. **P0 sécurité**: cookie `Secure=True` en prod, politique CORS stricte sans fallback permissif, retirer les `print` debug JWT.
2. **P0 accès données**: appliquer les filtres ownership/portée directement dans les services critiques (défense en profondeur).
3. **P1 robustesse queue**: ajouter idempotence explicite côté tâches (clé/run guard partout), et métriques d’échec/retry/revoke.
4. **P1 upload**: limites taille fichiers + whitelist extensions/MIME + quotas.
5. **P1 opérations espaces**: sérialiser/ordonner les updates concurrentes d’un même document (ou appliquer un “last-write-wins” explicite avec version).
6. **P2 observabilité**: dashboard “documents bloqués en pending/processing > X min”, “retry rate”, “revoked but still running”, “reindex backlog”.

## Conclusion
- **Stable mais pas encore “blindé prod”** sur l’axe sécurité et hardening concurrence.
- Le risque de “retraitement en boucle” est **plutôt bien contenu** par la logique run_id + statuts + retries limités, mais pas éliminé sans instrumentation/idempotence renforcée.
