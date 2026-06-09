"""
main.py — esecuzione end-to-end del progetto single-cell RNA-seq da terminale.

Uso rapido / Uso rápido:

    # esegue tutti gli step con parametri di default
    python main.py --step all

    # solo un singolo step
    python main.py --step baseline
    python main.py --step lightgbm
    python main.py --step shap
    python main.py --step imbalance

    # versione "veloce" (meno alberi, meno iterazioni)
    python main.py --step all --fast

    # esegue solo i modelli senza interpretabilità (più veloce)
    python main.py --step all --skip-shap

Il main.py NON sostituisce i notebook: è pensato per riprodurre tutti i
risultati end-to-end da terminale, utile per reproducibility e per
integrarlo in una pipeline automatizzata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Setup path per importare src/ anche lanciando main.py da altre dir
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.setup import load_pbmc, METRICS_DIR, FIGURES_DIR
from src.metrics import (
    compute_metrics,
    make_confusion_matrix,
    per_class_report,
    save_metrics,
)
from src.models import (
    make_logistic_regression,
    make_svm_linear,
    make_random_forest,
    make_lightgbm,
)
from src.plots import (
    plot_confusion_matrix,
    plot_top_features,
    plot_model_comparison,
)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def log(msg: str, level: int = 0) -> None:
    """Stampa messaggi con indentazione per leggibilità."""
    prefix = "  " * level
    print(f"{prefix}{msg}", flush=True)


def time_it(label: str):
    """Context manager per misurare il tempo di uno step."""
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            log(f"▶ {label}...")
            return self

        def __exit__(self, *a):
            dt = time.time() - self.t0
            log(f"✓ {label} — {dt:.1f}s", 1)

    return _T()


def print_metrics(name: str, metrics: dict) -> None:
    """Stampa metriche formattate."""
    log(f"{name}:", 1)
    for k, v in metrics.items():
        log(f"{k:>18s}: {v:.4f}", 2)


# ------------------------------------------------------------------
# STEP 1 — Baseline: LogReg + SVM linear + RF
# ------------------------------------------------------------------

def step_baseline(X_train, X_test, y_train, y_test, fast: bool = False):
    log("\n=== STEP: BASELINE MODELS ===")
    results = {}

    # Logistic Regression
    with time_it("Logistic Regression"):
        model = make_logistic_regression(max_iter=500 if fast else 1000)
        model.fit(X_train.values, y_train.values)
        metrics = compute_metrics(y_test, model.predict(X_test.values))
        save_metrics("02_logistic_regression", metrics)
        results["LogReg"] = metrics
        print_metrics("LogReg", metrics)

    # SVM linear (with PCA for speed)
    with time_it("SVM linear (+ PCA 50)"):
        model = make_svm_linear()
        model.fit(X_train.values, y_train.values)
        metrics = compute_metrics(y_test, model.predict(X_test.values))
        save_metrics("02_svm_linear", metrics)
        results["SVM_linear"] = metrics
        print_metrics("SVM linear", metrics)

    # Random Forest
    with time_it("Random Forest"):
        n_trees = 100 if fast else 300
        model = make_random_forest(n_estimators=n_trees)
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)
        metrics = compute_metrics(y_test, y_pred)
        save_metrics("02_random_forest", metrics)
        results["RandomForest"] = metrics
        print_metrics("Random Forest", metrics)

        # CM + top features per RF
        cm = make_confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm, name="02_confusion_matrix_rf",
            title="Matrice di confusione — Random Forest",
        )
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        plot_top_features(
            importances, name="02_top_genes_rf",
            title="Top 20 geni — Random Forest",
        )

    # Comparazione
    with time_it("Salvataggio comparazione baseline"):
        comparison = pd.DataFrame(results)
        plot_model_comparison(
            comparison.T[["accuracy", "weighted_f1", "balanced_accuracy"]],
            name="02_model_comparison_all_baselines",
        )

    return results


# ------------------------------------------------------------------
# STEP 2 — LightGBM + GridSearchCV minimale
# ------------------------------------------------------------------

def step_lightgbm(X_train, X_test, y_train, y_test, fast: bool = False):
    log("\n=== STEP: LIGHTGBM + TUNING ===")

    from sklearn.model_selection import GridSearchCV

    # LightGBM è lento sulla nostra combinazione 19 classi × 1976 feature.
    # In --fast usiamo 30 alberi (≈2 min totali); in modo normale 100.
    n_est = 30 if fast else 100

    with time_it(f"LightGBM baseline (n_estimators={n_est})"):
        model = make_lightgbm(n_estimators=n_est, learning_rate=0.1, n_jobs=1)
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)
        metrics = compute_metrics(y_test, y_pred)
        save_metrics("03_lightgbm", metrics)
        print_metrics("LightGBM baseline", metrics)

        cm = make_confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm, name="03_confusion_matrix_lgbm",
            title="Matrice di confusione — LightGBM",
        )
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        plot_top_features(
            importances, name="03_top_genes_lgbm",
            title="Top 20 geni — LightGBM",
        )

    # Grid search minimale: 2 combos × cv=2 = 4 fit ≈ 4×2 min
    # In --fast: salta il tuning per risparmiare tempo.
    if fast:
        log("⚠ Skip GridSearchCV in --fast mode", 1)
        return {"LightGBM": metrics}

    with time_it("GridSearchCV (2 combos × cv=2)"):
        base = make_lightgbm(n_estimators=n_est, n_jobs=1)
        grid = GridSearchCV(
            estimator=base,
            param_grid={"learning_rate": [0.05, 0.1]},
            scoring="accuracy",
            cv=2,
            n_jobs=1,
            verbose=0,
        )
        grid.fit(X_train.values, y_train.values)
        best = grid.best_estimator_
        metrics_tuned = compute_metrics(y_test, best.predict(X_test.values))
        save_metrics("03_lightgbm_tuned", metrics_tuned)
        log(f"Best params: {grid.best_params_}", 1)
        print_metrics("LightGBM tuned", metrics_tuned)

    return {"LightGBM": metrics, "LightGBM_tuned": metrics_tuned}


# ------------------------------------------------------------------
# STEP 3 — SHAP (interpretabilità + IL7R, PF4, CD3E)
# ------------------------------------------------------------------

def step_shap(X_train, X_test, y_train, y_test, fast: bool = False):
    log("\n=== STEP: SHAP INTERPRETABILITY ===")

    try:
        import shap
    except ImportError:
        log("⚠ shap non installato. Salta questo step.", 1)
        log("  Installare con: pip install shap", 1)
        return None

    n_est = 50 if fast else 100

    with time_it("Train LightGBM per SHAP"):
        model = make_lightgbm(n_estimators=n_est, learning_rate=0.1, n_jobs=1)
        model.fit(X_train.values, y_train.values)

    with time_it("Calcolo SHAP values (300 campioni)"):
        rng = np.random.default_rng(42)
        n_samples = min(200 if fast else 300, len(X_test))
        sample_idx = rng.choice(len(X_test), size=n_samples, replace=False)
        X_sample = X_test.iloc[sample_idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample.values)

        # Normalizzazione 3D → lista per classe
        if isinstance(shap_values, list):
            sv_per_class = shap_values
        else:
            sv_per_class = [shap_values[:, :, c] for c in range(shap_values.shape[-1])]

        log(f"{len(sv_per_class)} classi, shape per classe: {sv_per_class[0].shape}", 1)

    with time_it("SHAP summary plot globale"):
        import matplotlib.pyplot as plt
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(
            sv_per_class, X_sample, feature_names=list(X_sample.columns),
            plot_type="bar", show=False, max_display=20,
        )
        fig = plt.gcf()
        fig.tight_layout()
        out = FIGURES_DIR / "04_shap_summary_bar.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"Figura: {out}", 1)

    # Analisi specifica dei geni richiesti dalla tutora
    with time_it("Analisi IL7R / PF4 / CD3E"):
        genes_of_interest = ["IL7R", "PF4", "CD3E"]
        classes = list(model.classes_)
        rows = []
        for gene in genes_of_interest:
            if gene not in X_sample.columns:
                log(f"⚠ Gene {gene} non trovato.", 2)
                continue
            gene_idx = X_sample.columns.get_loc(gene)
            row = {}
            for c_idx, class_name in enumerate(classes):
                mean_abs = float(np.abs(sv_per_class[c_idx][:, gene_idx]).mean())
                row[class_name] = mean_abs
            rows.append(pd.Series(row, name=gene))

        if rows:
            df = pd.DataFrame(rows).T
            out_csv = METRICS_DIR / "04_genes_of_interest_shap.csv"
            df.to_csv(out_csv)
            log(f"Tabella SHAP per geni di interesse: {out_csv}", 1)
            log("Top classe per ogni gene:", 1)
            for gene in df.columns:
                top_class = df[gene].idxmax()
                log(f"{gene:>6s} → classe {top_class!r}  (|SHAP| = {df[gene].max():.4f})", 2)


# ------------------------------------------------------------------
# STEP 4 — Imbalance: class_weight vs SMOTE
# ------------------------------------------------------------------

def step_imbalance(X_train, X_test, y_train, y_test, fast: bool = False):
    log("\n=== STEP: IMBALANCE (class_weight vs SMOTE) ===")

    n_est = 50 if fast else 100

    # Baseline: class_weight='balanced' (dentro make_lightgbm)
    with time_it("LightGBM + class_weight='balanced'"):
        model = make_lightgbm(n_estimators=n_est, learning_rate=0.1, n_jobs=1)
        model.fit(X_train.values, y_train.values)
        metrics_cw = compute_metrics(y_test, model.predict(X_test.values))
        save_metrics("06_baseline_classweight", metrics_cw)
        print_metrics("class_weight='balanced'", metrics_cw)

    # SMOTE
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log("⚠ imblearn non installato. Salta SMOTE.", 1)
        log("  Installare con: pip install imbalanced-learn", 1)
        return {"class_weight": metrics_cw}

    with time_it("SMOTE oversampling"):
        smote = SMOTE(k_neighbors=1, random_state=42)
        X_res, y_res = smote.fit_resample(X_train.values, y_train.values)
        log(f"Dopo SMOTE: {X_res.shape}", 1)

    with time_it("LightGBM + SMOTE"):
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=n_est, learning_rate=0.1, num_leaves=31,
            class_weight=None, random_state=42, n_jobs=1, verbosity=-1,
        )
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test.values)
        metrics_smote = compute_metrics(y_test, y_pred)
        save_metrics("06_smote", metrics_smote)
        print_metrics("SMOTE", metrics_smote)

        cm = make_confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm, name="06_confusion_matrix_smote",
            title="Matrice di confusione — LightGBM + SMOTE",
        )

    comparison = pd.DataFrame({
        "Baseline_classweight": metrics_cw,
        "SMOTE": metrics_smote,
    })
    plot_model_comparison(comparison.T, name="06_comparison_smote")

    return {"class_weight": metrics_cw, "smote": metrics_smote}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Project 2 — single-cell RNA-seq classification end-to-end runner"
    )
    parser.add_argument(
        "--step",
        choices=["baseline", "lightgbm", "shap", "imbalance", "all"],
        default="all",
        help="Quale step eseguire (default: all)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Modalità veloce: meno alberi/iterazioni per test rapidi",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Salta lo step SHAP (è il più lento)",
    )
    args = parser.parse_args()

    log("=" * 60)
    log("Project 2 — Machine Learning on Single-Cell RNA-seq Data")
    log(f"Step: {args.step} | Fast mode: {args.fast}")
    log("=" * 60)

    t_start = time.time()

    # Caricamento dati (condiviso)
    with time_it("Caricamento dati"):
        X_train, X_test, y_train, y_test = load_pbmc()
        log(f"Train: {X_train.shape}  |  Test: {X_test.shape}", 1)
        log(f"Classi: {y_train.nunique()}", 1)

    # Esecuzione step scelti
    steps_to_run = (
        [args.step] if args.step != "all"
        else ["baseline", "lightgbm", "shap", "imbalance"]
    )
    if args.skip_shap and "shap" in steps_to_run:
        steps_to_run.remove("shap")

    step_fn = {
        "baseline": step_baseline,
        "lightgbm": step_lightgbm,
        "shap": step_shap,
        "imbalance": step_imbalance,
    }

    for step in steps_to_run:
        step_fn[step](X_train, X_test, y_train, y_test, fast=args.fast)

    total = time.time() - t_start
    log("\n" + "=" * 60)
    log(f"✓ Completato in {total:.1f}s ({total/60:.1f} min)")
    log(f"  Metriche salvate in: {METRICS_DIR}")
    log(f"  Figure salvate in:   {FIGURES_DIR}")
    log("=" * 60)


if __name__ == "__main__":
    main()
