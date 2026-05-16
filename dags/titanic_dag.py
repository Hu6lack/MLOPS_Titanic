"""DAG Apache Airflow — Pipeline MLOps Titanic complet."""

# Démarrage Airflow via Docker :
#   docker-compose up airflow-init
#   docker-compose up
# Interface : http://localhost:8080  (user: airflow / pass: airflow)

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner":           "etudiant-mlops",
    "retries":         1,
    "retry_delay":     timedelta(minutes=2),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="titanic_mlops_pipeline",
    default_args=default_args,
    description="Pipeline MLOps Titanic : Ingestion → Preprocessing → Training → Monitoring",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "titanic", "classification"],
)


# ── Fonctions Python (pas de subprocess) ─────────────────────────────────────

def task_ingestion(**context) -> None:
    """Tâche 1 : Télécharge et sauvegarde les données brutes."""
    import pandas as pd
    from pathlib import Path

    URL    = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    OUTPUT = Path("/opt/airflow/data/titanic_raw.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(URL)
    df.to_csv(OUTPUT, index=False)

    logger.info("✅ Ingestion : %d lignes | Survie : %.1f%%", len(df), df["Survived"].mean()*100)
    context["ti"].xcom_push(key="n_rows", value=len(df))


def task_preprocessing(**context) -> None:
    """Tâche 2 : Nettoie, encode, split et sauvegarde les données."""
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    RAW   = Path("/opt/airflow/data/titanic_raw.csv")
    TRAIN = Path("/opt/airflow/data/train.csv")
    TEST  = Path("/opt/airflow/data/test.csv")

    df = pd.read_csv(RAW)
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df["Age"]        = df["Age"].fillna(df["Age"].median())
    df["Embarked"]   = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    df["Sex"]        = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"]   = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train = X_train.copy(); train["Survived"] = y_train.values
    test  = X_test.copy();  test["Survived"]  = y_test.values
    train.to_csv(TRAIN, index=False)
    test.to_csv(TEST, index=False)

    logger.info("✅ Preprocessing : train=%d | test=%d", len(train), len(test))


def task_training(**context) -> None:
    """Tâche 3 : Entraîne les modèles et track avec MLflow."""
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from pathlib import Path
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier
    import joblib

    TRAIN  = Path("/opt/airflow/data/train.csv")
    TEST   = Path("/opt/airflow/data/test.csv")
    MODELS = Path("/opt/airflow/models")
    MODELS.mkdir(exist_ok=True)

    train_df = pd.read_csv(TRAIN)
    test_df  = pd.read_csv(TEST)
    X_train, y_train = train_df.drop(columns=["Survived"]), train_df["Survived"]
    X_test,  y_test  = test_df.drop(columns=["Survived"]),  test_df["Survived"]

    mlflow.set_experiment("titanic-mlops")
    best_model, best_name, best_f1 = None, "", -1.0

    for name, model in {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest":       RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "XGBoost":            XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss", random_state=42),
    }.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1    = f1_score(y_test, preds, zero_division=0)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, artifact_path="model")
            if f1 > best_f1:
                best_f1, best_model, best_name = f1, model, name

    joblib.dump(best_model, MODELS / "best_model.pkl")
    logger.info("🏆 Meilleur : %s (F1=%.4f)", best_name, best_f1)
    context["ti"].xcom_push(key="best_model", value=best_name)
    context["ti"].xcom_push(key="best_f1",    value=best_f1)


def task_monitoring(**context) -> None:
    """Tâche 4 : Monitoring avec Evidently"""
    import subprocess
    from pathlib import Path

    script_path = Path("/opt/airflow/src/monitoring/5_monitoring.py")  # ou monitoring/5_monitoring.py
    if not script_path.exists():
        script_path = Path("/opt/airflow/monitoring/5_monitoring.py")

    try:
        result = subprocess.run(["python", str(script_path)], 
                              capture_output=True, text=True, cwd="/opt/airflow")
        print(result.stdout)
        if result.returncode != 0:
            print("Error:", result.stderr)
            raise Exception("Monitoring script failed")
        logger.info("✅ Monitoring task completed successfully")
    except Exception as e:
        logger.error("Monitoring failed: %s", e)
        raise
    """Tâche 4 : Génère le rapport de drift Evidently AI."""
    import pandas as pd
    from pathlib import Path
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    TRAIN  = Path("/opt/airflow/data/train.csv")
    TEST   = Path("/opt/airflow/data/test.csv")
    REPORT = Path("/opt/airflow/monitoring/drift_report.html")
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=pd.read_csv(TRAIN), current_data=pd.read_csv(TEST))
    report.save_html(str(REPORT))
    logger.info("✅ Rapport drift sauvegardé : %s", REPORT)


# ── Tâches Airflow ────────────────────────────────────────────────────────────

t1 = PythonOperator(task_id="ingestion",     python_callable=task_ingestion,     dag=dag)
t2 = PythonOperator(task_id="preprocessing", python_callable=task_preprocessing, dag=dag)
t3 = PythonOperator(task_id="training",      python_callable=task_training,      dag=dag)
t4 = PythonOperator(task_id="monitoring",    python_callable=task_monitoring,    dag=dag)

t1 >> t2 >> t3 >> t4
