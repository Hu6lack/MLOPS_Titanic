"""Fonctions de preprocessing partagées entre les scripts et les tests."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et enrichit le DataFrame Titanic brut.

    Args:
        df: DataFrame brut.

    Returns:
        DataFrame nettoyé avec nouvelles features.
    """
    df = df.copy()
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df["Age"]        = df["Age"].fillna(df["Age"].median())
    df["Embarked"]   = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"]       = df["Fare"].fillna(df["Fare"].median())
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    df["Sex"]        = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"]   = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split stratifié train/test.

    Args:
        df: DataFrame nettoyé avec colonne 'Survived'.
        test_size: Proportion du test set.
        seed: Graine aléatoire.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
