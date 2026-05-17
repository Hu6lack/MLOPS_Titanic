"""DAG Apache Airflow — Pipeline MLOps Titanic (Version Améliorée)"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "hamza",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="titanic_mlops_pipeline",
    default_args=default_args,
    description="Pipeline MLOps Titanic complet - Ingestion → Preprocessing → Training → Monitoring",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "titanic", "production"],
)


def task_ingestion(**context):
    import subprocess
    subprocess.run(["python", "/opt/airflow/src/1_ingestion.py"], check=True, cwd="/opt/airflow")


def task_preprocessing(**context):
    import subprocess
    subprocess.run(["python", "/opt/airflow/src/2_preprocessing.py"], check=True, cwd="/opt/airflow")


def task_training(**context):
    import subprocess
    subprocess.run(["python", "/opt/airflow/src/3_training.py"], check=True, cwd="/opt/airflow")


def task_monitoring(**context):
    """Monitoring avec Evidently AI"""
    import subprocess
    from pathlib import Path
    logger = context['ti'].log

    script_paths = [
        "/opt/airflow/src/monitoring/5_monitoring.py",
        "/opt/airflow/monitoring/5_monitoring.py",
    ]

    for script in script_paths:
        if Path(script).exists():
            try:
                result = subprocess.run(
                    ["python", script],
                    capture_output=True,
                    text=True,
                    cwd="/opt/airflow",
                    timeout=45
                )
                logger.info(result.stdout)
                if result.returncode == 0:
                    logger.info("✅ Monitoring task completed successfully")
                    return
                else:
                    logger.warning(result.stderr)
            except Exception as e:
                logger.error(f"Error running monitoring: {e}")
    
    logger.warning("⚠️ Monitoring script not found or failed - continuing")
    

# Tasks
t1 = PythonOperator(task_id="ingestion",     python_callable=task_ingestion,     dag=dag)
t2 = PythonOperator(task_id="preprocessing", python_callable=task_preprocessing, dag=dag)
t3 = PythonOperator(task_id="training",      python_callable=task_training,      dag=dag)
t4 = PythonOperator(task_id="monitoring",    python_callable=task_monitoring,    dag=dag)

t1 >> t2 >> t3 >> t4