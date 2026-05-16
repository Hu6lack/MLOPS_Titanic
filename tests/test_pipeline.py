"""
Tests unitaires du pipeline MLOps Titanic.

Couvre :
- Nettoyage des données (imputation, suppression colonnes)
- Feature Engineering (FamilySize, IsAlone)
- Split stratifié et absence de Data Leakage
- Métriques de classification

Exécution : pytest tests/ -v --cov=src
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src._preprocessing_utils import clean, split


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Dataset Titanic minimal (20 lignes) pour les tests unitaires."""
    return pd.DataFrame({
        "PassengerId": range(1, 21),
        "Survived":   [0,1,1,0,0,1,0,1,0,1, 0,1,0,0,1,1,0,1,0,1],
        "Pclass":     [3,1,3,1,3,3,1,3,3,2, 3,1,3,1,3,3,1,3,3,2],
        "Name":       [f"Person_{i}" for i in range(20)],
        "Sex":        ["male","female"] * 10,
        "Age":        [22,38,26,35,np.nan,54,2,27,14,4] * 2,
        "SibSp":      [1,1,0,1,0,0,0,3,0,1] * 2,
        "Parch":      [0,0,0,0,0,0,4,1,0,0] * 2,
        "Ticket":     [f"T{i}" for i in range(20)],
        "Fare":       [7.25,71.28,7.92,53.10,8.05,8.46,51.86,21.08,11.13,30.07] * 2,
        "Cabin":      [None] * 20,
        "Embarked":   ["S","C","S","S","S","Q","S","S","S","C"] * 2,
    })


# ── Tests nettoyage ───────────────────────────────────────────────────────────

def test_age_imputation(sample_df: pd.DataFrame) -> None:
    """Vérifie que les NaN dans Age sont bien imputés après nettoyage."""
    df = clean(sample_df)
    assert df["Age"].isnull().sum() == 0, "Age contient encore des NaN !"


def test_drop_useless_columns(sample_df: pd.DataFrame) -> None:
    """Vérifie que les colonnes inutiles sont supprimées."""
    df = clean(sample_df)
    for col in ["Name", "Ticket", "Cabin", "PassengerId"]:
        assert col not in df.columns, f"Colonne '{col}' aurait dû être supprimée"


# ── Tests Feature Engineering ─────────────────────────────────────────────────

def test_family_size(sample_df: pd.DataFrame) -> None:
    """Vérifie que FamilySize = SibSp + Parch + 1."""
    df = clean(sample_df)
    expected = df["SibSp"] + df["Parch"] + 1
    pd.testing.assert_series_equal(df["FamilySize"], expected, check_names=False)


def test_is_alone(sample_df: pd.DataFrame) -> None:
    """Vérifie que IsAlone vaut 1 uniquement quand FamilySize == 1."""
    df = clean(sample_df)
    assert ((df["IsAlone"] == 1) == (df["FamilySize"] == 1)).all()


# ── Tests Split ───────────────────────────────────────────────────────────────

def test_no_data_leakage(sample_df: pd.DataFrame) -> None:
    """Bug ML classique — vérifie l'absence de Data Leakage dans X_train et X_test."""
    df = clean(sample_df)
    X_train, X_test, _, _ = split(df)
    assert "Survived" not in X_train.columns, "DATA LEAKAGE : Survived présent dans X_train !"
    assert "Survived" not in X_test.columns,  "DATA LEAKAGE : Survived présent dans X_test !"


def test_stratified_split(sample_df: pd.DataFrame) -> None:
    """Bug ML classique — vérifie que le split est bien stratifié (les 2 classes présentes)."""
    df = clean(sample_df)
    _, _, y_train, y_test = split(df)
    assert set(y_train.unique()) == {0, 1}, "Split non stratifié : classe manquante dans y_train"
    assert set(y_test.unique())  == {0, 1}, "Split non stratifié : classe manquante dans y_test"


# ── Tests métriques ───────────────────────────────────────────────────────────

def test_f1_score_range() -> None:
    """Vérifie que le F1-Score est bien dans [0, 1]."""
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    score  = f1_score(y_true, y_pred, zero_division=0)
    assert 0.0 <= score <= 1.0, f"F1-Score hors plage : {score}"
