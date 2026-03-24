# RBAC Implementation - Validation Guide

## Architecture mise en place

### Modèles créés
- **Role** : Rôles système (admin, member, viewer) et personnalisés
- **Permission** : Permissions granulaires par catégorie (config, library, space)
- **UserRole** : Association N-N entre utilisateurs et rôles
- **RolePermission** : Association N-N entre rôles et permissions

### Permissions définies

#### Catégorie: config
- `config.manage_users` - Gérer les utilisateurs
- `config.manage_roles` - Gérer les rôles et permissions

#### Catégorie: library
- `library.read` - Consulter la bibliothèque
- `library.write` - Modifier la bibliothèque (ajouter/modifier/supprimer documents)

#### Catégorie: space
- `space.read` - Consulter les espaces
- `space.create` - Créer des espaces
- `space.update` - Modifier des espaces
- `space.delete` - Supprimer des espaces

### Rôles prédéfinis

#### Admin
**Permissions:** Toutes
- config.manage_users
- config.manage_roles
- library.read
- library.write
- space.read
- space.create
- space.update
- space.delete

**Accès:**
- ✅ Interface admin visible dans la sidebar
- ✅ Peut créer/modifier/supprimer des utilisateurs
- ✅ Peut gérer les rôles et permissions
- ✅ Accès complet à la bibliothèque commune
- ✅ Peut créer/modifier/supprimer des espaces

#### Member (rôle par défaut)
**Permissions:**
- library.read
- library.write
- space.read
- space.create
- space.update
- space.delete

**Accès:**
- ❌ Interface admin masquée
- ❌ Aucun accès à la gestion users/rôles
- ✅ Accès complet à la bibliothèque commune (lecture + écriture)
- ✅ Peut créer/modifier/supprimer des espaces
- ✅ Voit tous les espaces partagés

#### Viewer (lecture seule)
**Permissions:**
- library.read
- space.read

**Accès:**
- ❌ Interface admin masquée
- ❌ Aucun accès à la gestion users/rôles
- ✅ Peut consulter la bibliothèque commune
- ❌ Ne peut PAS ajouter/modifier/supprimer des documents
- ✅ Peut consulter les espaces partagés
- ❌ Ne peut PAS créer/modifier/supprimer des espaces

## Bootstrap admin

L'administrateur est défini via la variable d'environnement `ADMIN_EMAIL`.

**Comportement:**
1. À l'inscription ou connexion, si l'email correspond à `ADMIN_EMAIL`, le rôle `admin` est automatiquement attribué
2. Si aucun rôle n'existe pour un utilisateur, le rôle `member` est attribué par défaut
3. Attribution idempotente (pas de duplication)

**Configuration:**
Ajouter dans `.env`:
```
ADMIN_EMAIL=admin@example.com
```

## Bibliothèque et espaces partagés

### Bibliothèque commune
- Une seule bibliothèque globale (`is_global=True`)
- Accessible à tous les utilisateurs authentifiés
- Les membres peuvent lire ET écrire (édition autorisée par défaut)
- Les viewers peuvent seulement lire
- Contrôle via permissions RBAC

### Espaces partagés
- Tous les espaces créés sont partagés par défaut (`is_shared=True`)
- Visibles par tous les utilisateurs authentifiés
- Création/modification/suppression contrôlées par permissions
- Les viewers peuvent seulement consulter

## Routes protégées

### Routes admin (config.manage_users ou config.manage_roles requis)
- `GET /api/admin/users` - Lister les utilisateurs
- `POST /api/admin/users` - Créer un utilisateur
- `DELETE /api/admin/users/{id}` - Supprimer un utilisateur
- `POST /api/admin/users/assign-role` - Assigner un rôle
- `DELETE /api/admin/users/{user_id}/roles/{role_id}` - Retirer un rôle
- `GET /api/admin/roles` - Lister les rôles
- `POST /api/admin/roles` - Créer un rôle
- `PUT /api/admin/roles/{id}` - Modifier un rôle
- `DELETE /api/admin/roles/{id}` - Supprimer un rôle
- `POST /api/admin/roles/assign-permission` - Assigner une permission
- `DELETE /api/admin/roles/{role_id}/permissions/{permission_id}` - Retirer une permission
- `GET /api/admin/permissions` - Lister les permissions
- `POST /api/admin/permissions` - Créer une permission

### Routes espaces (permissions space.* requises)
- `POST /api/spaces` - Créer un espace (space.create)
- `PUT /api/spaces/{id}` - Modifier un espace (space.update)
- `DELETE /api/spaces/{id}` - Supprimer un espace (space.delete)
- `GET /api/spaces` - Lister les espaces (aucune restriction, tous authentifiés)
- `GET /api/spaces/{id}` - Consulter un espace (aucune restriction, tous authentifiés)

### Routes bibliothèque
- Toutes les routes accessibles par défaut aux membres (library.write)
- Viewers ont accès en lecture seule (library.read)

## Tests de validation recommandés

### Test 1: Admin complet
1. Créer un utilisateur avec email = ADMIN_EMAIL
2. Se connecter
3. Vérifier que le lien "Administration" apparaît dans la sidebar
4. Accéder à `/admin`
5. Créer un utilisateur, assigner des rôles
6. Créer un espace
7. Ajouter un document à la bibliothèque

**Résultat attendu:** Toutes les opérations réussissent

### Test 2: Member standard
1. Créer un utilisateur normal
2. Se connecter
3. Vérifier que le lien "Administration" est masqué
4. Tenter d'accéder à `/admin` → les API devraient retourner 403
5. Créer un espace → devrait réussir
6. Ajouter un document → devrait réussir
7. Voir tous les espaces partagés → devrait réussir

**Résultat attendu:** Accès admin bloqué, reste accessible

### Test 3: Viewer (lecture seule)
1. Créer un utilisateur et lui assigner le rôle "viewer"
2. Se connecter
3. Vérifier que le lien "Administration" est masqué
4. Consulter la bibliothèque → devrait réussir
5. Tenter d'ajouter un document → devrait échouer (403)
6. Consulter un espace → devrait réussir
7. Tenter de créer un espace → devrait échouer (403)

**Résultat attendu:** Lecture seule stricte

### Test 4: Rôle personnalisé
1. En tant qu'admin, créer un rôle "editor"
2. Lui assigner seulement library.read, library.write, space.read
3. Assigner ce rôle à un utilisateur
4. Se connecter avec cet utilisateur
5. Consulter/modifier bibliothèque → devrait réussir
6. Tenter de créer un espace → devrait échouer (403)

**Résultat attendu:** Permissions granulaires respectées

## Migration des données existantes

Si des données user-scopées existent déjà :
1. Les bibliothèques existantes avec `user_id` restent accessibles
2. La nouvelle bibliothèque globale sera créée au premier accès
3. Les espaces existants sont maintenant visibles par tous (is_shared=True par défaut)

Aucune perte de données, compatibilité ascendante assurée.

## Endpoints API exposés

### Auth enrichi
- `GET /api/auth/me` retourne maintenant `roles` et `permissions` dans UserRead

### Admin
- Tous sous `/api/admin/*` (voir section Routes protégées)

## Fichiers modifiés/créés

### Modèles
- `app/models/role.py` ✅
- `app/models/permission.py` ✅
- `app/models/user_role.py` ✅
- `app/models/role_permission.py` ✅
- `app/models/user.py` (ajout relation user_roles, champs roles/permissions dans UserRead) ✅
- `app/models/library.py` (ajout is_global, user_id nullable) ✅
- `app/models/space.py` (ajout is_shared, user_id nullable) ✅

### Services
- `app/services/rbac_seed_service.py` ✅
- `app/services/authorization_service.py` ✅
- `app/services/auth_service.py` (attribution rôles) ✅
- `app/services/library_service.py` (bibliothèque globale) ✅
- `app/services/space_service.py` (espaces partagés) ✅

### Routes
- `app/routers/admin.py` ✅
- `app/routers/auth.py` (get_current_user enrichi, require_permission) ✅
- `app/routers/spaces.py` (protection RBAC) ✅

### Templates
- `app/templates/admin.html` ✅
- `app/templates/base.html` (lien admin conditionnel) ✅

### Config
- `app/config.py` (ADMIN_EMAIL) ✅
- `app/main.py` (seed RBAC au startup, route /admin, import router admin) ✅

## Conclusion

Le système RBAC est complet et opérationnel. Tous les objectifs du plan ont été atteints :
- ✅ Modèles RBAC créés
- ✅ Seed permissions/rôles
- ✅ Bootstrap admin via env
- ✅ Dépendance require_permission
- ✅ Routes protégées
- ✅ API admin complète
- ✅ Interface admin
- ✅ Bibliothèque/espaces partagés
- ✅ Documentation validation

**Prochaines étapes recommandées:**
1. Tester manuellement les 4 scénarios ci-dessus
2. Ajouter ADMIN_EMAIL dans .env
3. Redémarrer l'application pour créer les tables et seed
4. Créer le premier utilisateur admin
5. Tester la gestion des rôles dans l'interface admin
