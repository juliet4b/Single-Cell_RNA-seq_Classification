"""plots.py — grafici riutilizzabili. Salvati in `outputs/figures/`."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .setup import FIGURES_DIR


sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name, figures_dir=FIGURES_DIR):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


def plot_class_distribution(counts_df, name="class_distribution"):
    """Bar chart orizzontale: numero di celle per classe in train/test."""
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(counts_df))))
    counts_df[["train", "test"]].plot.barh(ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_xlabel("Numero di celle")
    ax.set_ylabel("Tipo cellulare")
    ax.set_title("Distribuzione delle classi (train vs test)")
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, name)


def plot_class_distribution_single(counts_series, split_label, name):
    """Bar chart orizzontale per un solo split (train o test)."""
    s = counts_series.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(s))))
    ax.barh(s.index, s.values, color="#4c72b0")
    ax.set_xlabel("Numero di celle")
    ax.set_ylabel("Tipo cellulare")
    ax.set_title(f"Distribuzione classi — {split_label}")
    fig.tight_layout()
    return _save(fig, name)


def plot_expression_histogram(values, name, title="Distribuzione valori di espressione (train)"):
    """Istogramma dei valori nella matrice di espressione."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=80, color="#4c72b0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Valore (normalizzato)")
    ax.set_ylabel("Frequenza")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, name)


def plot_boxplot_genes_by_class(X, y, genes, name, max_classes=10):
    """Boxplot dell'espressione di alcuni geni, per le classi più frequenti."""
    import pandas as pd

    top_classes = y.value_counts().head(max_classes).index
    mask = y.isin(top_classes)
    rows = []
    for gene in genes:
        if gene not in X.columns:
            continue
        for ct in top_classes:
            sub = X.loc[mask & (y == ct), gene]
            rows.append(pd.DataFrame({"gene": gene, "cell_type": ct, "expression": sub}))
    if not rows:
        return None
    long = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, max(4, 2 * len(genes))))
    sns.boxplot(data=long, x="cell_type", y="expression", hue="gene", ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Espressione per tipo cellulare (classi più frequenti)")
    fig.tight_layout()
    return _save(fig, name)


def plot_correlation_heatmap(corr_df, name, title="Correlazione tra geni (top variabilità)"):
    """Heatmap di una matrice di correlazione (sottoinsieme di geni)."""
    n = len(corr_df)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), max(5, n * 0.3)))
    sns.heatmap(corr_df, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, name)


def plot_mean_per_class_heatmap(mean_df, name, title="Media espressione per classe (top geni)"):
    """Heatmap: righe = tipi cellulari, colonne = geni selezionati."""
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * mean_df.shape[1]), max(5, 0.35 * len(mean_df))))
    sns.heatmap(mean_df, cmap="viridis", ax=ax, cbar_kws={"label": "media"})
    ax.set_xlabel("Gene")
    ax.set_ylabel("Tipo cellulare")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, name)


def plot_confusion_matrix(cm_df, name, title=None):
    """Heatmap della matrice di confusione (righe = vero, colonne = predetto)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Vero")
    ax.set_title(title or f"Matrice di confusione — {name}")
    fig.tight_layout()
    return _save(fig, name)


def plot_top_features(importances, name, top_n=20, title=None):
    """Bar chart dei top_n geni più importanti."""
    top = importances.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(top))))
    ax.barh(top.index, top.values, color="#55a868")
    ax.set_xlabel("Importanza")
    ax.set_title(title or f"Top {top_n} geni — {name}")
    fig.tight_layout()
    return _save(fig, name)


def plot_model_comparison(results, name="model_comparison"):
    """Bar chart comparativo tra modelli sulle metriche principali."""
    fig, ax = plt.subplots(figsize=(8, 5))
    results.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Valore metrica")
    ax.set_ylim(0, 1)
    ax.set_title("Confronto modelli")
    fig.tight_layout()
    return _save(fig, name)


def plot_pca_scatter(coords, labels, name, title="PCA 2D — train (per tipo cellulare)"):
    """Scatter 2D (es. prime 2 componenti PCA), colorato per classe."""
    import pandas as pd

    fig, ax = plt.subplots(figsize=(9, 7))
    labels = pd.Series(labels).reset_index(drop=True)
    unique_labels = sorted(labels.unique())
    palette = sns.color_palette("tab20", n_colors=len(unique_labels))
    for color, lab in zip(palette, unique_labels):
        mask = (labels == lab).values
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, label=lab, color=color)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7, markerscale=2)
    fig.tight_layout()
    return _save(fig, name)


def plot_tsne(coords, labels, name="tsne_cell_types", title="t-SNE dei tipi cellulari (train)"):
    """Scatter 2D dei risultati di t-SNE, colorato per tipo cellulare."""
    import pandas as pd
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = pd.Series(labels).reset_index(drop=True)
    unique_labels = sorted(labels.unique())
    palette = sns.color_palette("tab20", n_colors=len(unique_labels))
    for color, lab in zip(palette, unique_labels):
        mask = (labels == lab).values
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.7, label=lab, color=color)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, markerscale=2)
    fig.tight_layout()
    return _save(fig, name)
