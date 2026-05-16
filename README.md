# MLOps Titanic — Guide complet (Windows + Docker)

## 1. Installation
```bash
pip install -r requirements.txt
```

## 2. Exécution locale (sans Docker)

```bash
# Étape 1 — Ingestion
python src/1_ingestion.py

# Étape 2 — Preprocessing + Great Expectations
python src/2_preprocessing.py

# Étape 3 — Entraînement + MLflow + Model Registry
python src/3_training.py

# → Dans un autre terminal : interface MLflow
mlflow ui
# Ouvrir http://localhost:5000
# CAPTURE D'ÉCRAN : tableau comparatif des 3 modèles + Model Registry

# Étape 4 — API FastAPI
uvicorn src.4_api:app --reload
# Ouvrir http://localhost:8000/docs
# CAPTURE D'ÉCRAN : Swagger UI + résultat /predict

# Étape 5 — Rapport de drift
python monitoring/5_monitoring.py
# Ouvrir monitoring/reports/drift_report.html
# CAPTURE D'ÉCRAN : rapport Evidently

# Tests
pytest tests/ -v
# CAPTURE D'ÉCRAN : 7 tests PASSED
```

## 3. Stack complète avec Docker Compose

```bash
# Démarrage de tous les services
docker-compose up -d

# Services disponibles :
# - Airflow    : http://localhost:8080  (user: airflow / pass: airflow)
# - API FastAPI: http://localhost:8000/docs
# - Prometheus : http://localhost:9090
# - Grafana    : http://localhost:3000  (user: admin / pass: admin)
```

### Airflow
1. Aller sur http://localhost:8080
2. Activer le DAG `titanic_mlops_pipeline`
3. Cliquer "Trigger DAG" pour lancer manuellement
4. **CAPTURE D'ÉCRAN** : DAG Graph View avec 4 tâches vertes (SUCCESS)

### Grafana
1. Aller sur http://localhost:3000
2. Ajouter Prometheus comme datasource : http://prometheus:9090
3. Créer un dashboard avec les métriques :
   - `api_requests_total` — nombre de requêtes
   - `api_request_latency_seconds` — latence
   - `predictions_total` — prédictions par résultat
4. **CAPTURE D'ÉCRAN** : dashboard avec les panels

## 4. CI/CD GitHub Actions
1. Pousser le projet sur GitHub
2. Aller dans l'onglet "Actions"
3. **CAPTURE D'ÉCRAN** : pipeline vert (Tests ✅ + Docker ✅)

## 5. DVC (versionnage des données)
```bash
pip install dvc
dvc init
dvc add data/titanic_raw.csv
dvc add data/train.csv
git add data/.gitignore data/titanic_raw.csv.dvc data/train.csv.dvc
git commit -m "Versionnage des données avec DVC"
```

## Captures d'écran à faire pour le rapport
- [ ] Terminal : python src/1_ingestion.py
- [ ] Terminal : python src/2_preprocessing.py (avec validation GE)
- [ ] MLflow UI : tableau comparatif 3 modèles
- [ ] MLflow UI : Model Registry (Staging → Production)
- [ ] FastAPI : http://localhost:8000/docs
- [ ] FastAPI : résultat d'un appel /predict
- [ ] Evidently : drift_report.html
- [ ] Airflow : DAG Graph View (4 tâches vertes)
- [ ] Grafana : dashboard monitoring
- [ ] GitHub Actions : pipeline CI/CD vert
- [ ] Terminal : pytest tests/ -v (7 tests PASSED)
