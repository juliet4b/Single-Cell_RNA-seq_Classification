"""metrics.py — metriche e matrici di confusione.

Usato nei notebook dopo `model.fit(...)` e `model.predict(...)`.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .setup import METRICS_DIR


def compute_metrics(y_true, y_pred):
    """Accuracy, F1 (weighted/macro), precision/recall macro, balanced accuracy."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def per_class_report(y_true, y_pred):
    """Report per classe (precision, recall, F1, support) come DataFrame."""
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return pd.DataFrame(report).transpose()


def make_confusion_matrix(y_true, y_pred, labels=None):
    """Matrice di confusione come DataFrame (righe = vero, colonne = predetto)."""
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def save_metrics(name, metrics, metrics_dir=METRICS_DIR):
    """Salva un dict di metriche come JSON in outputs/metrics/<name>.json."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{name}.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    return out
