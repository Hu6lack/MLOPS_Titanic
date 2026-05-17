# 🚢 MLOps Titanic — Survival Prediction

> Pipeline MLOps complet : Ingestion → Prétraitement → Entraînement → Déploiement → Monitoring  
> **Filière CI 2 Informatique (IA & BI-DS) — ISMAGI 2025–2026**

---

## 📁 Structure du projet

```
hu6lack-mlops_titanic/
├── src/
│   ├── 1_ingestion.py              # Téléchargement des données Titanic
│   ├── 2_preprocessing.py          # Nettoyage + Great Expectations + split
│   ├── 3_training.py               # Entraînement 3 modèles + MLflow Registry
│   ├── 4_api.py                    # API FastAPI + métriques Prometheus
│   ├── _preprocessing_utils.py     # Fonctions partagées (tests + scripts)
│   └── great_expectations_setup.py # Validation des données (9 règles)
├── tests/
│   └── test_pipeline.py            # 7 tests Pytest
├── dags/
│   └── titanic_dag.py              # DAG Airflow (4 tâches séquentielles)
├── monitoring/
│   ├── 5_monitoring.py             # Rapport drift Evidently AI
│   └── grafana/
│       └── prometheus.yml          # Config scraping Prometheus → API
├── data/                           # Versionnées avec DVC
│   ├── titanic_raw.csv.dvc
│   ├── train.csv.dvc
│   └── test.csv.dvc
├── models/
│   └── best_model.pkl.dvc          # Modèle versionné avec DVC
├── .github/workflows/
│   └── ci_cd.yml                   # Pipeline GitHub Actions (Tests + Docker)
├── docker-compose.yml              # Stack complète (Airflow + API + Grafana)
├── Dockerfile                      # Image Docker pour l'API FastAPI
├── Dockerfile.airflow              # Image Airflow avec dépendances ML
└── requirements.txt
```

---

## ⚡ Démarrage rapide

### Option A — Exécution locale (étape par étape)

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Ingestion des données
python src/1_ingestion.py

# 3. Prétraitement + validation Great Expectations
python src/2_preprocessing.py

# 4. Entraînement + MLflow Tracking + Model Registry
python src/3_training.py

# Dans un autre terminal : interface MLflow
mlflow ui
# → http://localhost:5000  [CAPTURE D'ÉCRAN]

# 5. API FastAPI
uvicorn src.4_api:app --reload
# → http://localhost:8000/docs  [CAPTURE D'ÉCRAN]

# 6. Rapport de drift Evidently AI
python monitoring/5_monitoring.py
# → Ouvrir monitoring/reports/drift_report.html  [CAPTURE D'ÉCRAN]

# 7. Tests unitaires
pytest tests/ -v
# [CAPTURE D'ÉCRAN : 7 tests PASSED]
```

### Option B — Stack complète avec Docker Compose

```bash
# Démarrer tous les services d'un coup
docker-compose up -d

# Services disponibles :
# ┌─────────────────┬────────────────────────────────────────┐
# │ Service         │ URL                                    │
# ├─────────────────┼────────────────────────────────────────┤
# │ Airflow         │ http://localhost:8080  (airflow/airflow)│
# │ API FastAPI     │ http://localhost:8000/docs             │
# │ Prometheus      │ http://localhost:9090                  │
# │ Grafana         │ http://localhost:3000  (admin/admin)   │
# └─────────────────┴────────────────────────────────────────┘

# Arrêter les services
docker-compose down
```

---

## 🔧 Utilisation des outils

### MLflow — Tracking & Registry
```bash
# Lancer l'interface MLflow
mlflow ui

# Voir les runs de l'expérience titanic-mlops
# → http://localhost:5000/#/experiments
```

### Airflow — Orchestration du pipeline
```bash
# Démarrer via Docker
docker-compose up -d airflow-webserver airflow-scheduler

# Interface : http://localhost:8080
# 1. Activer le DAG "titanic_mlops_pipeline"
# 2. Cliquer "Trigger DAG" pour lancer manuellement
```

### API FastAPI — Test de prédiction
```bash
# Tester l'API (femme, 1ère classe, 28 ans)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": 1,
    "Age": 28,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100,
    "Embarked": 1,
    "FamilySize": 1,
    "IsAlone": 1
  }'
# Réponse attendue : {"survived":1,"probability":0.92,"label":"Survécu ✅"}
```

### DVC — Versionnage des données et du modèle
```bash
# Récupérer les données versionnées
dvc pull

# Vérifier le statut
dvc status

# Revenir à une version précédente
git checkout <commit>
dvc checkout
```

### Grafana — Dashboard monitoring
```bash
# 1. Accéder à http://localhost:3000 (admin/admin)
# 2. Configuration > Data Sources > Add Prometheus
#    URL : http://prometheus:9090
# 3. Create Dashboard > Add panels :
#    - api_requests_total        (Counter — requêtes par endpoint)
#    - api_request_latency_seconds_bucket  (Histogram — latence)
#    - predictions_total         (Counter — prédictions survived/deceased)
```

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec rapport de couverture
pytest tests/ -v --cov=src --cov-report=term-missing

# Tests disponibles :
# ✅ test_age_imputation         — Imputation des NaN dans Age
# ✅ test_drop_useless_columns   — Suppression colonnes inutiles
# ✅ test_family_size            — FamilySize = SibSp + Parch + 1
# ✅ test_is_alone               — IsAlone cohérent avec FamilySize
# ✅ test_no_data_leakage        — Survived absent de X_train [BUG ML]
# ✅ test_stratified_split       — 2 classes dans train et test [BUG ML]
# ✅ test_f1_score_range         — F1-Score dans [0, 1]
```

---

## 🚀 CI/CD GitHub Actions

Le pipeline se déclenche automatiquement à chaque push sur `main` ou `develop` :

```
Push → Job 1 : pytest tests/ -v (7 tests)
            ↓ si ✅
       Job 2 : docker build + test /health
```

Pour activer :
1. Pousser le projet sur GitHub : `git push origin main`
2. Aller dans l'onglet **Actions** du dépôt
3. Observer le pipeline s'exécuter automatiquement

---

## 📸 Captures d'écran à faire pour le rapport

| # | Outil | Commande / URL |
|---|---|---|
| 1 | Terminal ingestion | `python src/1_ingestion.py` |
| 2 | Terminal preprocessing | `python src/2_preprocessing.py` (GE validation) |
| 3 | Terminal pytest | `pytest tests/ -v` (7 PASSED) |
| 4 | Airflow Graph View | http://localhost:8080 (4 tâches vertes) |
| 5 | MLflow Runs | http://localhost:5000 (tableau comparatif) |
| 6 | MLflow Registry | http://localhost:5000/#/models (alias production) |
| 7 | FastAPI /docs | http://localhost:8000/docs |
| 8 | FastAPI /predict | Résultat JSON d'une requête |
| 9 | Evidently drift | `monitoring/reports/drift_report.html` |
| 10 | Grafana dashboard | http://localhost:3000 |
| 11 | GitHub Actions | Onglet Actions du dépôt (pipeline vert) |

---

## 📦 Stack technique

| Catégorie | Outil | Version |
|---|---|---|
| Langage | Python | 3.14 |
| ML | Scikit-learn, XGBoost | 1.8 / 3.2 |
| Tracking | MLflow | 3.12 |
| Orchestration | Apache Airflow | 2.9.1 |
| Data versioning | DVC | — |
| Data validation | Great Expectations | 0.18 |
| Serving | FastAPI + Uvicorn | 0.111 |
| Conteneurisation | Docker + Compose | — |
| Monitoring ML | Evidently AI | 0.4.30 |
| Monitoring tech | Prometheus + Grafana | latest |
| CI/CD | GitHub Actions | — |
| Tests | Pytest | 8.2 |