"""Module de prétraitement et validation des données Titanic."""

# Exécution : python src/2_preprocessing.py

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW: Path   = Path("data/titanic_raw.csv")
TRAIN: Path = Path("data/train.csv")
TEST: Path  = Path("data/test.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données brutes : suppression, imputation, encodage.

    Args:
        df: DataFrame brut chargé depuis la source.

    Returns:
        DataFrame nettoyé et enrichi avec les nouvelles features.

    Raises:
        KeyError: Si une colonne requise est absente.
    """
    df = df.copy()

    # Suppression des colonnes inutiles
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    # Imputation des valeurs manquantes
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())

    # Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # Encodage des variables catégorielles
    df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    return df


def validate_with_great_expectations(df: pd.DataFrame) -> bool:
    """Valide les données avec Great Expectations.

    Args:
        df: DataFrame à valider après nettoyage.

    Returns:
        True si toutes les validations passent, False sinon.
    """
    try:
        import great_expectations as gx

        context = gx.get_context()
        ds = context.sources.add_or_update_pandas(name="titanic")
        da = ds.add_dataframe_asset(name="train_asset")
        batch = da.build_batch_request(dataframe=df)

        suite = context.add_or_update_expectation_suite("titanic_suite")
        validator = context.get_validator(
            batch_request=batch, expectation_suite=suite
        )

        validator.expect_column_values_to_not_be_null("Age")
        validator.expect_column_values_to_not_be_null("Sex")
        validator.expect_column_values_to_be_in_set("Survived", [0, 1])
        validator.expect_column_values_to_be_between("Age", min_value=0, max_value=120)
        validator.expect_table_row_count_to_be_between(min_value=500, max_value=2000)

        results = validator.validate()
        passed = results["success"]
        logger.info("Great Expectations : %s", "✅ PASS" if passed else "❌ FAIL")
        return bool(passed)

    except Exception as exc:
        logger.warning("Great Expectations non disponible (%s), validation simplifiée.", exc)
        return _simple_validation(df)


def _simple_validation(df: pd.DataFrame) -> bool:
    """Validation simplifiée si Great Expectations n'est pas disponible.

    Args:
        df: DataFrame à valider.

    Returns:
        True si toutes les règles passent.
    """
    rules = {
        "Pas de NaN dans Age"     : df["Age"].isnull().sum() == 0,
        "Pas de NaN dans Sex"     : df["Sex"].isnull().sum() == 0,
        "Survived contient 0/1"   : df["Survived"].isin([0, 1]).all(),
        "Age entre 0 et 120"      : df["Age"].between(0, 120).all(),
        "Au moins 500 lignes"     : len(df) >= 500,
    }
    all_ok = True
    for name, result in rules.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info("  %s : %s", status, name)
        if not result:
            all_ok = False
    return all_ok


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divise les données en train/test de façon stratifiée.

    Args:
        df: DataFrame prétraité contenant la colonne 'Survived'.
        test_size: Proportion des données de test.
        seed: Graine aléatoire pour la reproductibilité.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: Si la colonne 'Survived' est absente.
    """
    if "Survived" not in df.columns:
        raise ValueError("Colonne 'Survived' absente — data leakage possible !")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Pipeline complet : nettoyage → validation → split → sauvegarde.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    df = pd.read_csv(RAW)
    df = clean(df)

    validate_with_great_expectations(df)

    X_train, X_test, y_train, y_test = split(df)

    train = X_train.copy(); train["Survived"] = y_train.values
    test  = X_test.copy();  test["Survived"]  = y_test.values
    train.to_csv(TRAIN, index=False)
    test.to_csv(TEST,   index=False)

    logger.info("✅ Train : %d lignes | Test : %d lignes", len(train), len(test))
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess()
