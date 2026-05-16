"""Module de monitoring : détection de drift avec Evidently AI."""

# Exécution : python monitoring/5_monitoring.py
# Rapport   : ouvrir monitoring/reports/drift_report.html dans le navigateur

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAIN: Path  = Path("data/train.csv")
TEST: Path   = Path("data/test.csv")
REPORT: Path = Path("monitoring/reports/drift_report.html")


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Génère un rapport HTML de détection de drift avec Evidently AI.

    Args:
        reference: Données de référence (entraînement).
        current: Données courantes (production / test).
    """
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

        report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(REPORT))
        logger.info("✅ Rapport de drift sauvegardé : %s", REPORT)
        logger.info("   → Ouvre ce fichier dans ton navigateur pour voir les résultats")

    except ImportError:
        logger.warning("Evidently non disponible — fallback sur comparaison manuelle")
        _manual_drift_check(reference, current)


def _manual_drift_check(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Vérifie le drift manuellement par comparaison de moyennes.

    Args:
        reference: Données de référence.
        current: Données courantes.
    """
    logger.info("\n--- Vérification manuelle du drift ---")
    for col in reference.select_dtypes(include="number").columns:
        if col not in current.columns:
            continue
        m_ref = reference[col].mean()
        m_cur = current[col].mean()
        diff  = abs(m_ref - m_cur) / (abs(m_ref) + 1e-9) * 100
        status = "⚠️  DRIFT" if diff > 15 else "✅ OK"
        logger.info("  %s %-15s | ref=%.2f | cur=%.2f | diff=%.1f%%", status, col, m_ref, m_cur, diff)


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN)
    test_df  = pd.read_csv(TEST)
    run_drift_report(train_df, test_df)
