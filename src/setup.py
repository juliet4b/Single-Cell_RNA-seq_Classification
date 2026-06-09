"""setup.py — caricamento dati single-cell RNA-seq (PBMC) e percorsi del progetto.

I CSV sono in `data_train/` e `data_test/`: lo split è predefinito.
NON va rimescolato e il test NON va usato per il tuning.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_TRAIN_DIR = PROJECT_ROOT / "data_train"
DATA_TEST_DIR = PROJECT_ROOT / "data_test"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"


def load_pbmc():
    """Carica X_train, X_test, y_train, y_test dai CSV."""
    X_train = pd.read_csv(DATA_TRAIN_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_TEST_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_TRAIN_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_TEST_DIR / "y_test.csv").squeeze("columns")
    y_train.name = "cell_type"
    y_test.name = "cell_type"
    return X_train, X_test, y_train, y_test


def sanity_checks(X_train, X_test, y_train, y_test):
    """Controlli rapidi: shape, geni coincidenti, NaN, range valori."""
    return {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "same_genes": list(X_train.columns) == list(X_test.columns),
        "nan_X_train": int(X_train.isna().sum().sum()),
        "nan_X_test": int(X_test.isna().sum().sum()),
        "min_value": float(X_train.values.min()),
        "max_value": float(X_train.values.max()),
        "mean_value": float(X_train.values.mean()),
        "n_classes_train": int(y_train.nunique()),
        "n_classes_test": int(y_test.nunique()),
    }


def class_counts(y_train, y_test):
    """Numero di celle per tipo cellulare in train e test."""
    return pd.DataFrame({
        "train": y_train.value_counts(),
        "test": y_test.value_counts(),
    }).fillna(0).astype(int)
