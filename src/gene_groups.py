"""gene_groups.py — categorizzazione dei geni più importanti in sottogruppi funzionali."""

from __future__ import annotations

import pandas as pd

# Marker noti per tipi cellulari PBMC (subset presente negli HVG del corso)
GENE_SUBGROUPS: dict[str, list[str]] = {
    "T_cell": ["CD3E", "CD3D", "CD3G", "IL7R", "TRAC", "CD8A", "CD8B", "CD4"],
    "B_cell": ["MS4A1", "CD79A", "CD79B", "CD19", "CD74", "JCHAIN", "IGHA1", "IGHG1"],
    "Myeloid": ["LYZ", "S100A8", "S100A9", "FCGR3A", "CD14", "CST3", "TYROBP"],
    "NK": ["NKG7", "GNLY", "PRF1", "GZMB", "KLRD1"],
    "Platelet": ["PPBP", "PF4"],
    "DC_pDC": ["FCER1A", "CST3", "IL3RA", "GZMB"],
    "Interferon_ISG": ["ISG15", "MX1", "IFIT1", "IFIT3"],
    "MHC": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1"],
}

# Firmas trascrittomiche da Wilk et al. (2020), Nature Medicine
WILK_SIGNATURES: dict[str, list[str]] = {
    "interferon_response": ["IFI6", "IFI44L", "ISG15", "MX1", "STAT1", "ISG20"],
    "monocitos_inflamatorios": ["S100A8", "S100A9", "IL1B", "TNF", "VCAN"],
    "linfocitos_T_citotoxicos": ["GZMB", "PRF1", "GNLY", "NKG7"],
}


def signature_genes_present(
    signatures: dict[str, list[str]], columns: list[str] | pd.Index
) -> dict[str, list[str]]:
    """Restituisce, per ogni firma, solo i geni presenti nelle colonne del dataset."""
    col_set = set(columns)
    return {name: [g for g in genes if g in col_set] for name, genes in signatures.items()}


def categorize_genes(gene_names: list[str]) -> pd.DataFrame:
    """Assegna ogni gene a un sottogruppo (o 'Other')."""
    rows = []
    for gene in gene_names:
        subgroup = "Other"
        for name, markers in GENE_SUBGROUPS.items():
            if gene in markers:
                subgroup = name
                break
        rows.append({"gene": gene, "subgroup": subgroup})
    return pd.DataFrame(rows)


def summarize_top_genes_by_subgroup(
    importances: pd.Series, top_n: int = 50
) -> pd.DataFrame:
    """Conta quanti dei top_n geni cadono in ogni sottogruppo."""
    top = importances.sort_values(ascending=False).head(top_n).index.tolist()
    cat = categorize_genes(top)
    summary = (
        cat.groupby("subgroup")
        .size()
        .reset_index(name="n_genes_in_top")
        .sort_values("n_genes_in_top", ascending=False)
    )
    return summary
