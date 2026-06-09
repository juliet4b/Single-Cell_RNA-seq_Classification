"""data_loader.py — caricamento dati single-cell RNA-seq (PBMC) e anteprima.

Uso da terminale:
    python -m src.data_loader

Uso da notebook:
    from src.data_loader import load_and_preview
"""

from __future__ import annotations

import pandas as pd

from .setup import load_pbmc


def load_datasets():
    """Carica X_train, X_test, y_train, y_test (split predefinito)."""
    return load_pbmc()


def dataset_shapes(X_train, X_test, y_train, y_test) -> dict:
    """Dimensioni train/test come dict."""
    return {
        "X_train": X_train.shape,
        "X_test": X_test.shape,
        "y_train": y_train.shape,
        "y_test": y_test.shape,
        "n_features": X_train.shape[1],
        "n_genes_train": X_train.shape[1],
        "n_cells_train": X_train.shape[0],
        "n_cells_test": X_test.shape[0],
    }


def print_shapes(X_train, X_test, y_train, y_test) -> None:
    """Stampa shape train/test (righe = celle, colonne = geni)."""
    shapes = dataset_shapes(X_train, X_test, y_train, y_test)
    print("Shape del dataset (righe = celle, colonne = geni):")
    print(f"  X_train: {shapes['X_train']}")
    print(f"  X_test : {shapes['X_test']}")
    print(f"  y_train: {shapes['y_train']}")
    print(f"  y_test : {shapes['y_test']}")


def preview_head(X_train, y_train, n: int = 5):
    """Prime n righe di X e y."""
    return X_train.head(n), y_train.head(n)


def load_and_preview(n: int = 5):
    """Carica, stampa shape e restituisce dati + anteprima."""
    X_train, X_test, y_train, y_test = load_datasets()
    print_shapes(X_train, X_test, y_train, y_test)
    xh, yh = preview_head(X_train, y_train, n=n)
    return X_train, X_test, y_train, y_test, xh, yh


def main() -> None:
    print("=== DATA LOADER ===")
    load_and_preview()
    print("OK - dati caricati.")


if __name__ == "__main__":
    main()
