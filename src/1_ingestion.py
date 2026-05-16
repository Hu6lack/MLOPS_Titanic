"""Module d'ingestion des données Titanic depuis une URL distante."""

# Exécution : python src/1_ingestion.py

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

URL: str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
OUTPUT: Path = Path("data/titanic_raw.csv")


def download_data(url: str = URL, output: Path = OUTPUT) -> pd.DataFrame:
    """Télécharge et sauvegarde les données Titanic.

    Args:
        url: URL source du fichier CSV Titanic.
        output: Chemin de sauvegarde local.

    Returns:
        DataFrame contenant les données brutes.

    Raises:
        ConnectionError: Si le téléchargement échoue.
    """
    logger.info("Téléchargement depuis %s", url)
    try:
        df: pd.DataFrame = pd.read_csv(url)
    except Exception as exc:
        raise ConnectionError(f"Téléchargement échoué : {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    logger.info("✅ %d lignes sauvegardées dans %s", len(df), output)
    logger.info("   Taux de survie : %.1f%%", df["Survived"].mean() * 100)
    logger.info("   Valeurs manquantes :\n%s", df.isnull().sum()[df.isnull().sum() > 0])
    return df


if __name__ == "__main__":
    download_data()
