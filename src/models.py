"""models.py — fabbriche per i classificatori del progetto.

Tutti i modelli usano `class_weight='balanced'` per gestire le classi rare.
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


RANDOM_STATE = 42


def make_logistic_regression(C=1.0, max_iter=1000):
    """Regressione logistica multinomiale con StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def make_svm_linear(C=1.0, pca_components=50):
    """SVM lineare; PCA a 50 componenti per mantenere il training rapido.

    Con 1976 feature e 19 classi LinearSVC senza PCA può richiedere minuti.
    Le 50 PC catturano la maggior parte della varianza dei dati scRNA-seq.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)),
        ("clf", LinearSVC(
            C=C,
            class_weight="balanced",
            dual="auto",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])


def make_random_forest(n_estimators=300, max_depth=None):
    """Random Forest: poche ipotesi, robusto, ottimo come baseline."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_lightgbm(n_estimators=300, learning_rate=0.05, num_leaves=63, n_jobs=-1):
    """LightGBM: gradient boosting veloce e accurato per dati tabulari."""
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm non installato. Installa con: pip install lightgbm")
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbosity=-1,
    )
