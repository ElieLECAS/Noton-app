# 🚀 Quick Start - Intégration Docling

## Commandes essentielles (Docker)

### 1️⃣ Build et démarrage
```bash
docker-compose build
docker-compose up -d
```

### 2️⃣ Migration du schéma BDD
```bash
docker-compose exec app alembic upgrade head
```

### 3️⃣ Migration des données
```bash
docker-compose exec app python -m app.scripts.migrate_to_docling --migrate
```

### 4️⃣ Vérification
```bash
docker-compose exec app python -m app.scripts.migrate_to_docling --verify
```

### ✅ C'est tout ! Le système est prêt !

---

## Test rapide

1. **Ouvrir l'application** : http://localhost:8000
2. **Se connecter** avec votre compte
3. **Ouvrir un projet**
4. **Cliquer sur "Importer un document"**
5. **Glisser-déposer un PDF, Word, ou autre fichier**
6. **Attendre le parsing et l'indexation** (~5-10 secondes)
7. **Ouvrir le chatbot du projet**
8. **Poser une question sur le contenu du document**
9. **Le chatbot devrait répondre en citant les passages pertinents ! 🎯**

---

## Formats supportés

✅ PDF, DOCX, XLSX, PPTX, CSV, TXT, MD, JSON, PNG, JPEG

---

## Résolution de problèmes

### Le parsing prend trop de temps
```bash
# Vérifier les logs
docker-compose logs -f app | grep "Docling"
```

### Erreur lors de l'upload
```bash
# Vérifier les permissions du dossier uploads
docker-compose exec app ls -la app/uploads
```

### Pas d'embeddings générés
```bash
# Redémarrer pour recharger le modèle
docker-compose restart app
```

---

## Documentation complète

- 📖 **Guide de migration** : `MIGRATION_DOCLING.md`
- ✅ **Récapitulatif complet** : `INTEGRATION_DOCLING_COMPLETE.md`
- 🔧 **Script de migration** : `app/scripts/migrate_to_docling.py`

---

## Rollback (si nécessaire)

```bash
docker-compose exec app alembic downgrade -1
docker-compose restart app
```

